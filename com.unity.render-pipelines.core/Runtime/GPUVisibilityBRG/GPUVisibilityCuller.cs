using System;
using System.Collections.Generic;
using UnityEngine.Assertions;
using Unity.Collections;
using Unity.Jobs;

namespace UnityEngine.Rendering
{
    internal struct GPUVisibilityCullerDesc
    {
        public int visibleClusterIndexBufferByteSize;

        public static GPUVisibilityCullerDesc NewDefault()
        {
            return new GPUVisibilityCullerDesc()
            {
                visibleClusterIndexBufferByteSize = 16 * 1024 * 1024 //16mb
            };
        }
    }

    internal struct GPUVisibilityCuller : IDisposable
    {
        private static class GPUCullerIDs
        {
            public static readonly int _DeferredMaterialInstanceDataAddress = Shader.PropertyToID("_DeferredMaterialInstanceDataAddress");
            public static readonly int _TotalInstanceCount = Shader.PropertyToID("_TotalInstanceCount");
            public static readonly int _InputInstances = Shader.PropertyToID("_InputInstances");
            public static readonly int _BigInstanceBuffer = Shader.PropertyToID("_BigInstanceBuffer");
            public static readonly int _InstanceOutCounter = Shader.PropertyToID("_InstanceOutCounter");
            public static readonly int _VisibleInstanceBuffer = Shader.PropertyToID("_VisibleInstanceBuffer");

            public static readonly int _MaxCounterValue = Shader.PropertyToID("_MaxCounterValue");
            public static readonly int _InputCounterBuffer = Shader.PropertyToID("_InputCounterBuffer");
            public static readonly int _OutIndirectArgs = Shader.PropertyToID("_OutIndirectArgs");

            public static readonly int _MaxOutputClusterOffset = Shader.PropertyToID("_MaxOutputClusterOffset");
            public static readonly int _InputInstanceCounter = Shader.PropertyToID("_InputInstanceCounter");
            public static readonly int _InputVisibleInstanceBuffer = Shader.PropertyToID("_InputVisibleInstanceBuffer");
            public static readonly int _ClustersOutCounter = Shader.PropertyToID("_ClustersOutCounter");
            public static readonly int _OutVisibleClustersBuffer = Shader.PropertyToID("_OutVisibleClustersBuffer");
        }

        private int m_LocalToWorldAddress;
        private int m_WorldToLocalAddress;
        private int m_DeferredMaterialInstanceAddress;
        private int m_ActiveInstanceCapacity;

        private int m_ActiveInstanceCount;
        private int m_VisibleClusterOutputCount;
        private int m_VisibleIndexOutputCount;

        private ComputeShader m_CullerCS;
        private int m_KernelMainWriteVisibleInstances;
        private int m_KernelMainWriteVisibleClusters;
        private int m_KernelMainWriteIndirectArgs;

        private GraphicsBuffer m_BigBuffer;
        private GraphicsBuffer m_ActiveInstanceBuffer;

        private NativeArray<int> m_ZeroArray;

        //TODO: these resources must be kept per view.
        //NOTE: do not do any view management on this class. Move any view management to
        //      the GPUVisibilityBRG instead. Ensure that this class remains referential transparent in terms of view rendering.
        ////////////////////////////////
        //Per view resources
        ////////////////////////////////
        private GraphicsBuffer m_VisibleInstanceBuffer;
        private GraphicsBuffer m_VisibleClusterBuffer;
        private GraphicsBuffer m_VisibleIndexBuffer;

        private GraphicsBuffer m_InstanceOutCounter;
        private GraphicsBuffer m_ClustersOutCounter;
        private GraphicsBuffer m_VisibleClustersArgBuffer;
        private GraphicsBuffer m_VisibleIndicesArgBuffer;
        ////////////////////////////////


        public void Initialize(
            GPUVisibilityCullerDesc desc,
            GraphicsBuffer bigBuffer,
            int localToWorldAddress, int worldToLocalAddress, int deferredMaterialInstanceDataAddress)
        {
            LoadShaders();
            m_LocalToWorldAddress = localToWorldAddress;
            m_WorldToLocalAddress = worldToLocalAddress;
            m_DeferredMaterialInstanceAddress = deferredMaterialInstanceDataAddress;
            m_BigBuffer = bigBuffer;
            m_ActiveInstanceBuffer = null;
            m_ActiveInstanceCapacity = 0;
            m_ActiveInstanceCount = 0;
            m_VisibleInstanceBuffer = null;
            ResizeActiveInstanceBuffer(2048);

            ////////////////////////////////
            //Per view resources
            ////////////////////////////////
            m_ZeroArray = new NativeArray<int>(1, Allocator.Persistent);
            m_ZeroArray[0] = 0;
            m_InstanceOutCounter = new GraphicsBuffer(GraphicsBuffer.Target.Raw, 1, 4);
            m_ClustersOutCounter = new GraphicsBuffer(GraphicsBuffer.Target.Raw, 1, 4);

            // hold the expanded indices of a cluster and then the actual index of the cluster
            int sizePerCluster = GeometryPoolConstants.GeoPoolClusterPrimitiveCount * 4 + 4;
            m_VisibleClusterOutputCount = (desc.visibleClusterIndexBufferByteSize / sizePerCluster);
            m_VisibleIndexOutputCount = m_VisibleClusterOutputCount * GeometryPoolConstants.GeoPoolClusterPrimitiveCount;
            m_VisibleClusterBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Raw, m_VisibleClusterOutputCount, 4);
            m_VisibleIndexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Raw, m_VisibleIndexOutputCount, 4);

            m_VisibleClustersArgBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.IndirectArguments, 3, 4);
            m_VisibleIndicesArgBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.IndirectArguments, 3, 4);
            ////////////////////////////////

        }

        private void LoadShaders()
        {
            m_CullerCS = (ComputeShader)Resources.Load("VisibilityClusterCuller");
            m_KernelMainWriteVisibleInstances = m_CullerCS.FindKernel("MainWriteVisibleInstances");
            m_KernelMainWriteVisibleClusters = m_CullerCS.FindKernel("MainWriteVisibleClusters");
            m_KernelMainWriteIndirectArgs = m_CullerCS.FindKernel("MainWriteIndirectArgs");
        }

        private void ResizeActiveInstanceBuffer(int length)
        {
            if (m_ActiveInstanceBuffer != null && length <= m_ActiveInstanceCapacity)
                return;

            if (m_ActiveInstanceBuffer != null)
                m_ActiveInstanceBuffer.Release();

            m_ActiveInstanceBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Raw, length, 4);
            m_VisibleInstanceBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Raw, length, 4);
            m_ActiveInstanceCapacity = length;
        }

        public void UpdateActiveInstanceBuffer(NativeArray<int> instanceIds)
        {
            ResizeActiveInstanceBuffer(instanceIds.Length);
            m_ActiveInstanceCount = instanceIds.Length;
            m_ActiveInstanceBuffer.SetData(instanceIds, 0, 0, instanceIds.Length);
        }

        public bool GPUCull(GeometryPool geometryPool, CommandBuffer cmdBuffer)
        {
            if (m_ActiveInstanceCount == 0)
                return false;

            ///////////////////////////////////////////
            //Visible Instance pass & arguments buffer.
            ///////////////////////////////////////////
            {
                cmdBuffer.SetBufferData(m_InstanceOutCounter, m_ZeroArray, 0, 0, 1);
                geometryPool.BindResourcesCS(cmdBuffer, m_CullerCS, m_KernelMainWriteVisibleInstances);
                cmdBuffer.SetComputeIntParam(m_CullerCS, GPUCullerIDs._DeferredMaterialInstanceDataAddress, m_DeferredMaterialInstanceAddress);
                cmdBuffer.SetComputeIntParam(m_CullerCS, GPUCullerIDs._TotalInstanceCount, m_ActiveInstanceCount);
                cmdBuffer.SetGlobalBuffer(GPUCullerIDs._InputInstances, m_ActiveInstanceBuffer);
                cmdBuffer.SetGlobalBuffer(GPUCullerIDs._BigInstanceBuffer, m_BigBuffer);
                cmdBuffer.SetGlobalBuffer(GPUCullerIDs._InstanceOutCounter, m_InstanceOutCounter);
                cmdBuffer.SetGlobalBuffer(GPUCullerIDs._VisibleInstanceBuffer, m_VisibleInstanceBuffer);
                cmdBuffer.DispatchCompute(m_CullerCS, m_KernelMainWriteVisibleInstances, (m_ActiveInstanceCount + 63) / 64, 1, 1);

                cmdBuffer.SetComputeIntParam(m_CullerCS, GPUCullerIDs._MaxCounterValue, m_ActiveInstanceCount);
                cmdBuffer.SetGlobalBuffer(GPUCullerIDs._InputCounterBuffer, m_InstanceOutCounter);
                cmdBuffer.SetGlobalBuffer(GPUCullerIDs._OutIndirectArgs, m_VisibleClustersArgBuffer);
                cmdBuffer.DispatchCompute(m_CullerCS, m_KernelMainWriteIndirectArgs, 1, 1, 1);
            }
            //////////////////////////////////////////

            ///////////////////////////////////////////
            //Visible Cluster & arguments buffer.
            ///////////////////////////////////////////
            {
                cmdBuffer.SetBufferData(m_ClustersOutCounter, m_ZeroArray, 0, 0, 1);
                cmdBuffer.SetComputeIntParam(m_CullerCS, GPUCullerIDs._MaxOutputClusterOffset, m_VisibleClusterOutputCount);
                cmdBuffer.SetGlobalBuffer(GPUCullerIDs._InputInstanceCounter, m_InstanceOutCounter);
                cmdBuffer.SetGlobalBuffer(GPUCullerIDs._InputVisibleInstanceBuffer, m_VisibleInstanceBuffer);
                cmdBuffer.SetGlobalBuffer(GPUCullerIDs._ClustersOutCounter, m_ClustersOutCounter);
                cmdBuffer.SetGlobalBuffer(GPUCullerIDs._OutVisibleClustersBuffer, m_VisibleClusterBuffer);
                cmdBuffer.DispatchCompute(m_CullerCS, m_KernelMainWriteVisibleClusters, m_VisibleClustersArgBuffer, 0);

                cmdBuffer.SetComputeIntParam(m_CullerCS, GPUCullerIDs._MaxCounterValue, m_VisibleClusterOutputCount);
                cmdBuffer.SetGlobalBuffer(GPUCullerIDs._InputCounterBuffer, m_ClustersOutCounter);
                cmdBuffer.SetGlobalBuffer(GPUCullerIDs._OutIndirectArgs, m_VisibleIndicesArgBuffer);
                cmdBuffer.DispatchCompute(m_CullerCS, m_KernelMainWriteIndirectArgs, 1, 1, 1);
            }
            ///////////////////////////////////////////

            return true;
        }

        public void Dispose()
        {
            if (m_ActiveInstanceBuffer != null)
                m_ActiveInstanceBuffer.Release();

            if (m_InstanceOutCounter != null)
                m_InstanceOutCounter.Release();

            if (m_ClustersOutCounter != null)
                m_ClustersOutCounter.Release();

            if (m_ZeroArray.IsCreated)
                m_ZeroArray.Dispose();

            if (m_VisibleClusterBuffer != null)
                m_VisibleClusterBuffer.Release();

            if (m_VisibleIndexBuffer != null)
                m_VisibleIndexBuffer.Release();

            if (m_VisibleClustersArgBuffer != null)
                m_VisibleClustersArgBuffer.Release();

            if (m_VisibleIndicesArgBuffer != null)
                m_VisibleIndicesArgBuffer.Release();

        }
    }

}
