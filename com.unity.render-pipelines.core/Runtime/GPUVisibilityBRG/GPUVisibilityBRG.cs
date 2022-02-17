using System;
using System.Collections.Generic;
using UnityEngine.Assertions;
using Unity.Collections;
using Unity.Jobs;

namespace UnityEngine.Rendering
{
    internal struct GPUVisibilityBRGDesc
    {
        public int maxInstances;
        public GeometryPoolDesc geometryPoolDesc;
        public GPUVisibilityCullerDesc cullerDesc;

        public static GPUVisibilityBRGDesc NewDefault()
        {
            return new GPUVisibilityBRGDesc()
            {
                maxInstances = 4096,
                geometryPoolDesc = GeometryPoolDesc.NewDefault(),
                cullerDesc = GPUVisibilityCullerDesc.NewDefault()
            };
        }
    }

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
        private bool m_InstancePoolDirty = false;

        private GPUVisibilityCuller m_GPUCuller = new GPUVisibilityCuller();

        public GeometryPool geometryPool => m_GeoPool;

        internal struct GPUInstanceBatchData
        {
            public bool isValid;
            public bool validLightMaps;
            public NativeList<GPUVisibilityInstance> instances;
            public NativeList<GeometryPoolHandle> geoPoolHandles;
            public LightMaps lightmaps;

            public void Dispose()
            {
                if (instances.IsCreated)
                    instances.Dispose();

                if (geoPoolHandles.IsCreated)
                    geoPoolHandles.Dispose();

                if (validLightMaps)
                    lightmaps.Destroy();
            }
        }

        private List<GPUInstanceBatchData> m_Batches = new List<GPUInstanceBatchData>();
        private NativeList<GPUInstanceBatchHandle> m_FreeBatchSlots = new NativeList<GPUInstanceBatchHandle>(64, Allocator.Persistent);

        public void Initialize(GPUVisibilityBRGDesc desc)
        {
            m_GeoPool = new GeometryPool(desc.geometryPoolDesc);
            m_InstancePool.Initialize(desc.maxInstances);
            m_UploadResources = new GPUInstanceDataBufferUploader.GPUResources();
            m_GPUCuller.Initialize(
                desc.cullerDesc,
                m_InstancePool.bigBuffer.gpuBuffer,
                m_InstancePool.bigBuffer.GetGpuAddress(GPUInstanceDataBuffer.DefaultSchema.unity_ObjectToWorld),
                m_InstancePool.bigBuffer.GetGpuAddress(GPUInstanceDataBuffer.DefaultSchema.unity_WorldToObject),
                m_InstancePool.bigBuffer.GetGpuAddress(GPUInstanceDataBuffer.DefaultSchema._DeferredMaterialInstanceData));
        }

        public void Update()
        {
            if (m_InstancePoolDirty)
            {
                m_GPUCuller.UpdateActiveInstanceBuffer(m_InstancePool.aliveInstanceIndices);
                m_InstancePoolDirty = false;
            }
        }

        public void StartInstanceTransformUpdateJobs()
        {
            m_InstancePool.StartUpdateJobs();
        }

        public bool EndInstanceTransformUpdateJobs(CommandBuffer cmdBuffer)
        {
            return m_InstancePool.EndUpdateJobs(cmdBuffer);
        }

        public int GetInstanceCount(GPUInstanceBatchHandle batchHandle)
        {
            if (!batchHandle.valid || batchHandle.index >= m_Batches.Count)
                return 0;

            return m_Batches[batchHandle.index].instances.Length;
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

            if (!batchState.geoPoolHandles.IsCreated)
                batchState.geoPoolHandles = new NativeList<GeometryPoolHandle>(64, Allocator.Persistent);

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

            var rendererMaterialInfos = lightmappingData.rendererToMaterialMap;

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
            int paramIdxDeferredMaterialIdx = m_InstancePool.bigBuffer.GetPropertyIndex(GPUInstanceDataBuffer.DefaultSchema._DeferredMaterialInstanceData);
            //////////////////////////////////////////////////////////////////////////////

            /// TMP lists for materials / submeshes ///
            var sharedMaterials = new List<Material>();
            //////////////////////////////////////////

            for (int i = 0; i < meshRenderers.Count; ++i)
            {
                var meshRenderer = meshRenderers[i];
                var meshFilter = meshRenderer.gameObject.GetComponent<MeshFilter>();
                meshRenderer.forceRenderingOff = true;
                Assert.IsTrue(meshFilter != null && meshFilter.sharedMesh != null, "Ensure mesh object has been filtered properly before sending to the visibility BRG.");

                /////////////////////////////////////////////////////
                //Construct geometry pool and material information.//
                /////////////////////////////////////////////////////
                GeometryPoolHandle geometryHandle = GeometryPoolHandle.Invalid;

                {
                    sharedMaterials.Clear();
                    meshRenderer.GetSharedMaterials(sharedMaterials);
                    var startSubMesh = meshRenderer.subMeshStartIndex;
                    var geoPoolEntryDesc = new GeometryPoolEntryDesc()
                    {
                        mesh = meshFilter.sharedMesh,
                        submeshData = new GeometryPoolSubmeshData[sharedMaterials.Count]
                    };

                    for (int matIndex = 0; matIndex < sharedMaterials.Count; ++matIndex)
                    {
                        Material matToUse;
                        if (!rendererMaterialInfos.TryGetValue(new Tuple<Renderer, int>(meshRenderer, matIndex), out matToUse))
                            matToUse = sharedMaterials[matIndex];

                        int targetSubmeshIndex = (int)(startSubMesh + matIndex);

                        geoPoolEntryDesc.submeshData[matIndex] = new GeometryPoolSubmeshData()
                        {
                            submeshIndex = targetSubmeshIndex,
                            material = matToUse
                        };
                    }

                    if (!m_GeoPool.Register(geoPoolEntryDesc, out geometryHandle))
                    {
                        Debug.LogError("Could not register instance in geometry pool. Check the geo pool capacity.");
                        continue;
                    }
                }

                Assert.IsTrue(geometryHandle.valid);

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

                    //TODO register into the geometry pool.

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

                    bigBufferUploader.WriteParameter<Vector4>(uploadHandle, paramIdxSHC, new Vector4(geometryHandle.index, 0.0f, 0.0f, 0.0f));
                }
                ///////////////////////////////////////////////
            }

            ///////////////////////////////////////////////
            //Flush all properties to the big buffer     //
            ///////////////////////////////////////////////
            bigBufferUploader.SubmitToGpu(ref m_UploadResources);
            bigBufferUploader.Dispose();
            m_Batches[batchHandle.index] = batchState;
            m_InstancePoolDirty = true;
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
                m_InstancePool.FreeVisibilityEntity(batchState.instances[i]);
            }

            //free the geo pool handles
            for (int i = 0; i < batchState.geoPoolHandles.Length; ++i)
            {
                m_GeoPool.Unregister(batchState.geoPoolHandles[i]);
            }

            batchState.instances.Clear();
            batchState.geoPoolHandles.Clear();
            m_Batches[batchHandle.index] = batchState;
        }

        public void Dispose()
        {
            m_InstancePool.Dispose();
            m_FreeBatchSlots.Dispose();
            m_UploadResources.Dispose();
            m_GeoPool.Dispose();
            foreach (var b in m_Batches)
            {
                b.Dispose();
            }
            m_Batches.Clear();
            m_GPUCuller.Dispose();
        }
    }
}
