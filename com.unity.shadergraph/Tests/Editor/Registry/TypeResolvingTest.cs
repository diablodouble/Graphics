using NUnit.Framework;
using com.unity.shadergraph.defs;
using System.Collections.Generic;
using static UnityEditor.ShaderGraph.Registry.Types.GraphType;
using UnityEditor.ShaderGraph.GraphDelta;
using UnityEngine.TestTools;

namespace UnityEditor.ShaderGraph.Registry.UnitTests
{
    [TestFixture]
    class TypeResolving
    {
        private Registry m_registry;
        private IGraphHandler m_graph;

        private void SetupGraph()
        {
            // create the registry
            m_registry = new Registry();
            m_registry.Register<Types.GraphType>();

            // create the graph
            m_graph = GraphUtil.CreateGraph();
        }

        [Test]
        public void ResolveFromConcretizationTest()
        {
            SetupGraph();

            // make a FunctionDescriptor and register it
            var parameters = new LinkedList<ParameterDescriptor>();
            parameters.AddFirst(new ParameterDescriptor("In", TYPE.Vector, Usage.In));
            parameters.AddFirst(new ParameterDescriptor("Out", TYPE.Vector, Usage.Out));
            FunctionDescriptor fd = new(1, "Test", parameters, "Out = In;");
            RegistryKey registryKey = m_registry.Register(fd);

            // add a single node to the graph
            string nodeName = $"{fd.Name}-01";
            INodeWriter nodeWriter = m_graph.AddNode(registryKey, nodeName, m_registry);

            // check that the the node was added
            var nodeReader = m_graph.GetNodeReader(nodeName);
            bool didRead = nodeReader.GetField("In.Length", out Length len);
            Assert.IsTrue(didRead);

            // EXPECT that both In and Out are concretized into length = 4 (default)
            Assert.AreEqual(Length.Four, len);
            didRead = nodeReader.GetField("Out.Length", out len);
            Assert.IsTrue(didRead);
            Assert.AreEqual(Length.Four, len);

            // make In a Vec3
            nodeWriter.SetPortField("In", kLength, Length.Three);
            nodeReader = m_graph.GetNodeReader(nodeName);

            // EXPECT that In now reads as a Vec3
            didRead = nodeReader.GetField("In.Length", out len);
            Assert.IsTrue(didRead);
            Assert.AreEqual(Length.Three, len);

            // EXPECT that Out has not changed
            didRead = nodeReader.GetField("Out.Length", out len);
            Assert.IsTrue(didRead);
            Assert.AreEqual(Length.Four, len);

            // reconcretize the node
            bool didReconcretize = m_graph.ReconcretizeNode(nodeName, m_registry);
            Assert.IsTrue(didReconcretize);

            // EXPECT that In is still a Vec3
            nodeReader = m_graph.GetNodeReader(nodeName);
            didRead = nodeReader.GetField("In.Length", out len);
            Assert.IsTrue(didRead);
            Assert.AreEqual(Length.Three, len);

            // EXPECT that Out has resolved into a Vec3
            didRead = nodeReader.GetField("Out.Length", out len);
            Assert.IsTrue(didRead);
            Assert.AreEqual(Length.Three, len);
        }
    }
}
