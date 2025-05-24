import unittest
import numpy as np
from graph_utils import Node, Graph, calculate_shd_and_differences, find_discoverable_parents, bfs_shortest_path, create_erdos_renyi_graph

class TestNode(unittest.TestCase):
    def test_node_creation(self):
        node = Node(1)
        self.assertIsNotNone(node)
        self.assertTrue(node.node_type in ["AND", "OR"])

    def test_add_child(self):
        parent = Node(1)
        child = Node(2)
        parent.add_child(child)
        self.assertIn(child, parent.children)
        self.assertIn(parent, child.parents)

class TestGraph(unittest.TestCase):
    def setUp(self):
        self.nodes = [Node(i) for i in range(4)]
        self.adj_matrix = np.array([[0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 0, 0]])

    def test_graph_creation(self):
        graph = Graph(nodes=self.nodes)
        self.assertEqual(len(graph.nodes), 4)

    def test_remove_edges(self):
        graph = Graph(nodes=self.nodes)
        graph.remove_edges()
        for node in graph.nodes:
            self.assertEqual(len(node.children), 0)
            self.assertEqual(len(node.parents), 0)

    def test_is_ancestor_of_goal(self):
        graph = Graph(nodes=self.nodes)
        graph.set_graph_adjacency(graph_adjacency=self.adj_matrix)
        self.assertTrue(graph.is_ancestor_of_goal(0))

    def test_create_adjacency_matrix(self):
        graph = Graph()
        graph.set_graph_adjacency(graph_adjacency=self.adj_matrix)
        matrix = graph.create_adjacency_matrix()
        np.testing.assert_array_equal(matrix, self.adj_matrix)

class TestStandaloneFunctions(unittest.TestCase):
    def test_calculate_shd_and_differences(self):
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[1, 0], [0, 1]])
        shd, extra_edges, missing_edges, _, _ = calculate_shd_and_differences(A, B)
        self.assertEqual(shd, 3)
        self.assertEqual(extra_edges, 2)
        self.assertEqual(missing_edges, 1)

    def test_find_discoverable_parents(self):
        adj_matrix = np.array([[0, 0], [1, 0]])
        gate_types = ['AND', 'OR']
        result = find_discoverable_parents(adj_matrix, gate_types)
        expected = [[0, 0], [1, 0]]
        np.testing.assert_array_equal(result, expected)

    def test_bfs_shortest_path(self):
        start = Node(0)
        goal = Node(1)
        start.add_child(goal)
        self.assertTrue(bfs_shortest_path(start, goal))

    def test_create_erdos_renyi_graph(self):
        graph, start_node, goal_node = create_erdos_renyi_graph(5, 0.5)
        self.assertEqual(len(graph.nodes), 5)
        self.assertTrue(bfs_shortest_path(start_node, goal_node))

if __name__ == '__main__':
    unittest.main()