using NUnit.Framework;
using Unity.Mathematics;
using System;
using System.Collections.Generic;
using Unity.Collections;
using UnityEngine.Rendering;
using Unity.Collections.LowLevel.Unsafe;

namespace UnityEngine.Rendering.Tests
{
    class GPUPipelineTests
    {
        [SetUp]
        public void OnSetup()
        {
        }

        [TearDown]
        public void OnTearDown()
        {
        }

        internal static class TestSchema
        {
            public static readonly int _InternalId0 = Shader.PropertyToID("_InternalId0");
            public static readonly int _BaseColor = Shader.PropertyToID("_BaseColor");
            public static readonly int _TestMatrix = Shader.PropertyToID("_TestMatrix");
            public static readonly int _InternalId1 = Shader.PropertyToID("_InternalId1");
            public static readonly int _TestMatrixInv = Shader.PropertyToID("_TestMatrixInv");
            public static readonly int _InternalId2 = Shader.PropertyToID("_InternalId2");
            public static readonly int _TestMatrix2 = Shader.PropertyToID("_TestMatrix2");
            public static readonly int _VecOffset = Shader.PropertyToID("_VecOffset");
        }

        GPUInstanceDataBuffer ConstructTestInstanceBuffer(int instanceCount)
        {
            using (var builder = new GPUInstanceDataBufferBuilder())
            {
                builder.AddComponent<Vector4>(TestSchema._InternalId0, isOverriden: false, isPerInstance: false);
                builder.AddComponent<Vector4>(TestSchema._BaseColor, isOverriden: true, isPerInstance: true);
                builder.AddComponent<BRGMatrix>(TestSchema._TestMatrix, isOverriden: true, isPerInstance: true);
                builder.AddComponent<BRGMatrix>(TestSchema._TestMatrixInv, isOverriden: true, isPerInstance: true);
                builder.AddComponent<Vector4>(TestSchema._InternalId1, isOverriden: true, isPerInstance: false);
                builder.AddComponent<Vector4>(TestSchema._InternalId2, isOverriden: true, isPerInstance: false);
                builder.AddComponent<BRGMatrix>(TestSchema._TestMatrix2, isOverriden: true, isPerInstance: true);
                builder.AddComponent<Vector4>(TestSchema._VecOffset, isOverriden: true, isPerInstance: true);
                return builder.Build(instanceCount);
            }
        }

        internal struct BigBufferCPUReadbackData : IDisposable
        {
            public NativeArray<Vector4> data;
            GPUInstanceDataBuffer m_BigBuffer;

            public BigBufferCPUReadbackData(GPUInstanceDataBuffer bigBuffer)
            {
                m_BigBuffer = bigBuffer;
                var cmdBuffer = new CommandBuffer();
                int vec4Size = UnsafeUtility.SizeOf<Vector4>();
                var localData = new NativeArray<Vector4>(bigBuffer.byteSize / vec4Size, Allocator.Persistent);
                cmdBuffer.RequestAsyncReadback(bigBuffer.gpuBuffer, (AsyncGPUReadbackRequest req) =>
                {
                    if (req.done)
                        localData.CopyFrom(req.GetData<Vector4>());
                });
                cmdBuffer.WaitAllAsyncReadbackRequests();
                Graphics.ExecuteCommandBuffer(cmdBuffer);
                cmdBuffer.Release();
                data = localData;
            }

            public T LoadData<T>(int instanceId, int propertyID) where T : unmanaged
            {
                int vec4Size = UnsafeUtility.SizeOf<Vector4>();
                int propertyIndex = m_BigBuffer.GetPropertyIndex(propertyID);
                Assert.IsTrue(m_BigBuffer.descriptions[propertyIndex].isPerInstance);
                int gpuBaseAddress = m_BigBuffer.gpuBufferComponentAddress[propertyIndex];
                int indexInArray = (gpuBaseAddress + m_BigBuffer.descriptions[propertyIndex].byteSize * instanceId) / vec4Size;

                unsafe
                {
                    Vector4* dataPtr = (Vector4*)data.GetUnsafePtr<Vector4>() + indexInArray;
                    T result = *(T*)(dataPtr);
                    return result;
                }
            }

            public void Dispose()
            {
                data.Dispose();
            }
        }

        struct TestObjectProperties
        {
            public int baseColorIndex;
            public int matrixIndex;
            public int matrixInvIndex;

            public void Initialize(in GPUInstanceDataBuffer buffer)
            {
                baseColorIndex = buffer.GetPropertyIndex(TestSchema._BaseColor);
                matrixIndex = buffer.GetPropertyIndex(TestSchema._TestMatrix);
                matrixInvIndex = buffer.GetPropertyIndex(TestSchema._TestMatrixInv);
            }
        }

        struct TestObject
        {
            public Vector4 color;
            public BRGMatrix matrix;
            public BRGMatrix matrixInv;

            public void Upload(in TestObjectProperties props, ref GPUInstanceDataBufferUploader uploader, int instanceId)
            {
                var instanceHandle = uploader.AllocateInstance(instanceId);
                uploader.WriteParameter<Vector4>(instanceHandle, props.baseColorIndex, color);
                uploader.WriteParameter<BRGMatrix>(instanceHandle, props.matrixIndex, matrix);
                uploader.WriteParameter<BRGMatrix>(instanceHandle, props.matrixInvIndex, matrixInv);
            }

            public void Download(in BigBufferCPUReadbackData readbackData, int insanceId)
            {
                color = readbackData.LoadData<Vector4>(insanceId, TestSchema._BaseColor);
                matrix = readbackData.LoadData<BRGMatrix>(insanceId, TestSchema._TestMatrix);
                matrixInv = readbackData.LoadData<BRGMatrix>(insanceId, TestSchema._TestMatrixInv);
            }

            public bool Equals(in TestObject other)
            {
                return color.Equals(other.color) && matrix.Equals(other.matrix) && matrixInv.Equals(other.matrixInv);
            }

        };

        [Test]
        public void TestBigInstanceBuffer()
        {
            var gpuResources = new GPUInstanceDataBufferUploader.GPUResources();
            using (var instanceBuffer = ConstructTestInstanceBuffer(12))
            {
                var matrixA = new BRGMatrix()
                {
                    localToWorld0 = new float4(1.0f, 2.0f, 3.0f, 4.0f),
                    localToWorld1 = new float4(5.0f, 6.0f, 7.0f, 8.0f),
                    localToWorld2 = new float4(7.0f, 6.0f, 5.0f, 4.0f),
                };
                var matrixB = new BRGMatrix()
                {
                    localToWorld0 = new float4(4.0f, 5.0f, 6.0f, 7.0f),
                    localToWorld1 = new float4(1.0f, 2.0f, 2.0f, 1.0f),
                    localToWorld2 = new float4(0.0f, 1.0f, 1.0f, 0.0f),
                };
                var colorA = new float4(1.0f, 2.0f, 3.0f, 4.0f);
                var colorB = new float4(4.0f, 5.0f, 6.0f, 7.0f);

                var objectA = new TestObject()
                {
                    color = colorA,
                    matrix = matrixA,
                    matrixInv = matrixB
                };

                var objectB = new TestObject()
                {
                    color = colorB,
                    matrix = matrixB,
                    matrixInv = matrixA
                };

                var properties = new TestObjectProperties();
                properties.Initialize(instanceBuffer);

                var instanceUploader0 = new GPUInstanceDataBufferUploader(instanceBuffer);
                {
                    objectA.Upload(properties, ref instanceUploader0, 0);
                    objectA.Upload(properties, ref instanceUploader0, 1);
                    objectA.Upload(properties, ref instanceUploader0, 3);
                    objectA.Upload(properties, ref instanceUploader0, 4);

                    objectB.Upload(properties, ref instanceUploader0, 2);
                    objectB.Upload(properties, ref instanceUploader0, 5);
                    objectB.Upload(properties, ref instanceUploader0, 8);
                    objectB.Upload(properties, ref instanceUploader0, 11);

                    instanceUploader0.SubmitToGpu(ref gpuResources);
                }
                instanceUploader0.Dispose();

                using (var readbackData = new BigBufferCPUReadbackData(instanceBuffer))
                {
                    var obj = new TestObject();

                    obj.Download(readbackData, 0);
                    Assert.IsTrue(obj.Equals(objectA));

                    obj.Download(readbackData, 1);
                    Assert.IsTrue(obj.Equals(objectA));

                    obj.Download(readbackData, 3);
                    Assert.IsTrue(obj.Equals(objectA));

                    obj.Download(readbackData, 4);
                    Assert.IsTrue(obj.Equals(objectA));

                    obj.Download(readbackData, 2);
                    Assert.IsTrue(obj.Equals(objectB));

                    obj.Download(readbackData, 5);
                    Assert.IsTrue(obj.Equals(objectB));

                    obj.Download(readbackData, 8);
                    Assert.IsTrue(obj.Equals(objectB));

                    obj.Download(readbackData, 11);
                    Assert.IsTrue(obj.Equals(objectB));

                }
            }
            gpuResources.Dispose();
        }

        [Test]
        public void TestInstancePool()
        {
            var instancePool = new GPUVisibilityInstancePool();
            instancePool.Initialize(5);

            var o = new GameObject();
            var t = o.transform;
            var a = instancePool.AllocateVisibilityEntity(t, true);
            var b = instancePool.AllocateVisibilityEntity(t, true);
            var c = instancePool.AllocateVisibilityEntity(t, true);

            Assert.IsTrue(instancePool.InternalSanityCheckStates());

            instancePool.FreeVisibilityEntity(b);

            Assert.IsTrue(instancePool.InternalSanityCheckStates());

            b = instancePool.AllocateVisibilityEntity(t, true);
            var d = instancePool.AllocateVisibilityEntity(t, true);
            var e = instancePool.AllocateVisibilityEntity(t, true);

            Assert.IsTrue(instancePool.InternalSanityCheckStates());

            instancePool.FreeVisibilityEntity(b);
            instancePool.FreeVisibilityEntity(e);
            instancePool.FreeVisibilityEntity(a);

            Assert.IsTrue(instancePool.InternalSanityCheckStates());

            var g = instancePool.AllocateVisibilityEntity(t, true);

            Assert.IsTrue(instancePool.InternalSanityCheckStates());


            instancePool.Dispose();
        }
    }
}
