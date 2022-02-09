using System;
using System.Collections.Generic;
using UnityEngine.Assertions;
using Unity.Collections;
using Unity.Jobs;

namespace UnityEngine.Rendering
{

    internal struct GPUInstanceBatchHandle : IEquatable<GPUInstanceBatchHandle>
    {
        public int index;
        public static GPUInstanceBatchHandle Invalid = new GPUInstanceBatchHandle() { index = -1 };
        public bool valid => index != -1;
        public bool Equals(GPUInstanceBatchHandle other) => index == other.index;
    }

    internal class GPUVisibilityBRG : IDisposable
    {
        public JobHandle OnPerformCulling(BatchRendererGroup rendererGroup, BatchCullingContext cullingContext,
                    BatchCullingOutput cullingOutput, IntPtr userContext)
        {
            return new JobHandle();
        }

        private GeometryPool m_GeoPool = null;
        private GPUVisibilityInstancePool m_InstancePool = new GPUVisibilityInstancePool();
        private GPUInstanceDataBufferUploader.GPUResources m_UploadResources;

        internal struct GPUInstanceBatchData
        {
            public bool isValid;
            public bool validLightMaps;
            public NativeList<GPUVisibilityInstance> instances;
            public LightMaps lightmaps;

            public void Dispose()
            {
                if (instances.IsCreated)
                    instances.Dispose();

                if (validLightMaps)
                    lightmaps.Destroy();
            }
        }

        private List<GPUInstanceBatchData> m_Batches = new List<GPUInstanceBatchData>();
        private NativeList<GPUInstanceBatchHandle> m_FreeBatchSlots = new NativeList<GPUInstanceBatchHandle>(64, Allocator.Persistent);

        public void Initialize(GeometryPool geometryPool, int maxEntities)
        {
            m_GeoPool = geometryPool;
            m_InstancePool.Initialize(maxEntities);
            m_UploadResources = new GPUInstanceDataBufferUploader.GPUResources();
        }

        public void Update()
        {
        }

        public GPUInstanceBatchHandle CreateInstanceBatch()
        {
            GPUInstanceBatchHandle handle = GPUInstanceBatchHandle.Invalid;
            if (m_FreeBatchSlots.IsEmpty)
            {
                handle = new GPUInstanceBatchHandle() { index = m_Batches.Count };
                m_Batches.Add(new GPUInstanceBatchData());
            }
            else
            {
                handle = m_FreeBatchSlots[m_FreeBatchSlots.Length - 1];
                m_FreeBatchSlots.RemoveAt(m_FreeBatchSlots.Length - 1);
            }

            GPUInstanceBatchData batchState = m_Batches[handle.index];
            Assert.IsTrue(!batchState.isValid);
            Assert.IsTrue(!batchState.instances.IsCreated || batchState.instances.IsEmpty);
            batchState.isValid = true;
            if (!batchState.instances.IsCreated)
                batchState.instances = new NativeList<GPUVisibilityInstance>(64, Allocator.Persistent);
            m_Batches[handle.index] = batchState;
            return handle;
        }

        public void RegisterGameObjectInstances(GPUInstanceBatchHandle batchHandle, List<MeshRenderer> meshRenderers)
        {
            if (!batchHandle.valid || batchHandle.index >= m_Batches.Count)
                throw new Exception("Batch handle is invalid");

            if (meshRenderers.Count == 0)
                return;

            GPUInstanceBatchData batchState = m_Batches[batchHandle.index];
            batchState.validLightMaps = true;
            var bigBufferUploader = new GPUInstanceDataBufferUploader(m_InstancePool.bigBuffer);

            var lightmappingData = LightMaps.GenerateLightMappingData(meshRenderers);
            batchState.lightmaps = lightmappingData.lightmaps;
            LightProbesQuery lpq = new LightProbesQuery(Allocator.Temp);


            //////////////////////////////////////////////////////////////////////////////
            // indices of the properties that the uploader will write to the big buffer //
            //////////////////////////////////////////////////////////////////////////////
            int paramIdxLightmapIndex = m_InstancePool.bigBuffer.GetPropertyIndex(GPUInstanceDataBuffer.DefaultSchema.unity_LightmapIndex);
            int paramIdxLightmapScale = m_InstancePool.bigBuffer.GetPropertyIndex(GPUInstanceDataBuffer.DefaultSchema.unity_LightmapST);
            int paramIdxLocalToWorld = m_InstancePool.bigBuffer.GetPropertyIndex(GPUInstanceDataBuffer.DefaultSchema.unity_ObjectToWorld);
            int paramIdxWorldToLocal = m_InstancePool.bigBuffer.GetPropertyIndex(GPUInstanceDataBuffer.DefaultSchema.unity_WorldToObject);
            int paramIdxProbeOcclusion = m_InstancePool.bigBuffer.GetPropertyIndex(GPUInstanceDataBuffer.DefaultSchema.unity_ProbesOcclusion);
            int paramIdxSHAr = m_InstancePool.bigBuffer.GetPropertyIndex(GPUInstanceDataBuffer.DefaultSchema.unity_SHAr);
            int paramIdxSHAg = m_InstancePool.bigBuffer.GetPropertyIndex(GPUInstanceDataBuffer.DefaultSchema.unity_SHAg);
            int paramIdxSHAb = m_InstancePool.bigBuffer.GetPropertyIndex(GPUInstanceDataBuffer.DefaultSchema.unity_SHAb);
            int paramIdxSHBr = m_InstancePool.bigBuffer.GetPropertyIndex(GPUInstanceDataBuffer.DefaultSchema.unity_SHBr);
            int paramIdxSHBg = m_InstancePool.bigBuffer.GetPropertyIndex(GPUInstanceDataBuffer.DefaultSchema.unity_SHBg);
            int paramIdxSHBb = m_InstancePool.bigBuffer.GetPropertyIndex(GPUInstanceDataBuffer.DefaultSchema.unity_SHBb);
            int paramIdxSHC = m_InstancePool.bigBuffer.GetPropertyIndex(GPUInstanceDataBuffer.DefaultSchema.unity_SHC);
            //////////////////////////////////////////////////////////////////////////////

            for (int i = 0; i < meshRenderers.Count; ++i)
            {
                var meshRenderer = meshRenderers[i];
                var meshFilter = meshRenderer.gameObject.GetComponent<MeshFilter>();
                meshRenderer.forceRenderingOff = true;
                Assert.IsTrue(meshFilter != null && meshFilter.sharedMesh != null, "Ensure mesh object has been filtered properly before sending to the visibility BRG.");

                //Register entity in instance pool
                var visibilityInstance = m_InstancePool.AllocateVisibilityEntity(meshRenderer.transform, meshRenderer.lightProbeUsage == LightProbeUsage.BlendProbes);
                batchState.instances.Add(visibilityInstance);

                ///////////////////////////////////////////////
                //Writing all the properties to the uploader.//
                ///////////////////////////////////////////////
                {
                    var uploadHandle = bigBufferUploader.AllocateInstance(visibilityInstance.index);

                    if (!lightmappingData.lightmapIndexRemap.TryGetValue(meshRenderer.lightmapIndex, out var newLmIndex))
                        newLmIndex = 0;

                    int tetrahedronIdx = -1;
                    lpq.CalculateInterpolatedLightAndOcclusionProbe(meshRenderer.transform.position, ref tetrahedronIdx, out var lp,
                        out var probeOcclusion);

                    bigBufferUploader.WriteParameter<Vector4>(uploadHandle, paramIdxLightmapIndex, new Vector4(newLmIndex, 0, 0, 0));
                    bigBufferUploader.WriteParameter<Vector4>(uploadHandle, paramIdxLightmapScale, meshRenderer.lightmapScaleOffset);
                    bigBufferUploader.WriteParameter<BRGMatrix>(uploadHandle, paramIdxLocalToWorld, BRGMatrix.FromMatrix4x4(meshRenderer.transform.localToWorldMatrix));
                    bigBufferUploader.WriteParameter<BRGMatrix>(uploadHandle, paramIdxWorldToLocal, BRGMatrix.FromMatrix4x4(meshRenderer.transform.worldToLocalMatrix));
                    bigBufferUploader.WriteParameter<Vector4>(uploadHandle, paramIdxProbeOcclusion, probeOcclusion);

                    var sh = new SHProperties(lp);
                    bigBufferUploader.WriteParameter<Vector4>(uploadHandle, paramIdxSHAr, sh.SHAr);
                    bigBufferUploader.WriteParameter<Vector4>(uploadHandle, paramIdxSHAg, sh.SHAg);
                    bigBufferUploader.WriteParameter<Vector4>(uploadHandle, paramIdxSHAb, sh.SHAb);
                    bigBufferUploader.WriteParameter<Vector4>(uploadHandle, paramIdxSHBr, sh.SHBr);
                    bigBufferUploader.WriteParameter<Vector4>(uploadHandle, paramIdxSHBg, sh.SHBg);
                    bigBufferUploader.WriteParameter<Vector4>(uploadHandle, paramIdxSHBb, sh.SHBb);
                    bigBufferUploader.WriteParameter<Vector4>(uploadHandle, paramIdxSHC, sh.SHC);
                }
                ///////////////////////////////////////////////
            }

            ///////////////////////////////////////////////
            //Flush all properties to the big buffer     //
            ///////////////////////////////////////////////
            bigBufferUploader.SubmitToGpu(ref m_UploadResources);
            bigBufferUploader.Dispose();
            m_Batches[batchHandle.index] = batchState;
            ///////////////////////////////////////////////
        }

        public GPUInstanceBatchHandle CreateBatchFromGameObjectInstances(List<MeshRenderer> meshRenderers)
        {
            GPUInstanceBatchHandle newHandle = CreateInstanceBatch();
            RegisterGameObjectInstances(newHandle, meshRenderers);
            return newHandle;
        }

        public void DestroyInstanceBatch(GPUInstanceBatchHandle batchHandle)
        {
            if (!batchHandle.valid || batchHandle.index >= m_Batches.Count)
                throw new Exception("Batch handle is invalid");

            GPUInstanceBatchData batchState = m_Batches[batchHandle.index];
            batchState.isValid = false;
            //free the instances
            for (int i = 0; i < batchState.instances.Length; ++i)
            {
                GPUVisibilityInstance instance = batchState.instances[i];
                m_InstancePool.FreeVisibilityEntity(instance);
            }

            batchState.instances.Clear();
            m_Batches[batchHandle.index] = batchState;
        }

        public void Dispose()
        {
            m_InstancePool.Dispose();
            m_FreeBatchSlots.Dispose();
            m_UploadResources.Dispose();
            foreach (var b in m_Batches)
            {
                b.Dispose();
            }
            m_Batches.Clear();
        }
    }
}
