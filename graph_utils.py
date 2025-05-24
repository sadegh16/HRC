import numpy as np
import random
import networkx as nx
from collections import deque
from collections import defaultdict

class Node:
    def __init__(self, value, node_type=None):
        self.value = value
        if node_type is None:
            if random.random() < 0.5:
                self.node_type = "AND"
            else:
                self.node_type = "OR"
        else:
            self.node_type = node_type
        self.children = []
        self.parents = []

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parents.append(self)

    def __repr__(self):
        return f"{self.value} ({self.node_type})"


class Graph:
    def __init__(self, nodes=None,):
        self.nodes = nodes
        if self.nodes is not None:
            size = len(self.nodes)
            self.weighted_graph_adjacency= [[0 for _ in range(size)] for _ in range(size)]


    def set_node_types(self, graph_node_types):
        if self.nodes is None:
            self.nodes = [Node(idx, node_type) for idx, node_type in enumerate(graph_node_types)]
        else:
            for idx, node_type in enumerate(graph_node_types):
                self.nodes[idx].node_type = node_type

    def set_graph_adjacency(self, graph_adjacency):

        if self.nodes is None:
            self.nodes = [Node(idx, node_type="OR") for idx in range(len(graph_adjacency))]
        self.remove_edges()
        for i in range(len(graph_adjacency)):
            for j in range(len(graph_adjacency)):
                if graph_adjacency[i, j] == 1:
                    self.nodes[i].add_child(self.nodes[j])

    def remove_edges(self):
        for node in self.nodes:
            node.children = []
            node.parents = []

    def add_node(self, node):
        self.nodes.append(node)

    def is_ancestor_of_goal(self, identifier):
        # Check if the given node is an ancestor of the goal_node
        visited = set()

        def dfs(current_node):
            if current_node == self.nodes[-1]:
                return True
            visited.add(current_node)
            for neighbor in current_node.children:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
            return False

        if isinstance(identifier, int):
            node = self.nodes[identifier]
        elif isinstance(identifier, Node):
            node = identifier
        else:
            raise ValueError("Identifier must be a Node or an integer index")

        return dfs(node)

    def create_adjacency_matrix(self, ):
        size = len(self.nodes)
        matrix = [[0 for _ in range(size)] for _ in range(size)]

        for node in self.nodes:
            for child in node.children:
                matrix[node.value][child.value] = 1

        return np.array(matrix)

    def find_shortest_path_andor(self, root_nodes):
        # Initialize distances and parents tracker
        size = len(self.nodes)
        distances = {node: float('inf') for node in self.nodes}
        parents_tracker = {node: [] for node in self.nodes}  # To track the path taken

        queue = deque()

        # Set root nodes distance to 0 and add to the queue
        for root_node in root_nodes:
            distances[root_node] = 0
            queue.append(root_node)

        # Perform BFS
        while queue:
            current_node = queue.popleft()

            # Goal node check
            if current_node == self.nodes[-1]:
                break  # We've reached the goal, can exit early

            for child in current_node.children:
                # Check the type of the child node
                if child.node_type == "AND":
                    # Only consider the child if all its parents are already visited
                    if all(distances[parent] < float('inf') for parent in child.parents):
                        # Update the distance if a shorter path is found
                        if distances[child] > distances[current_node] + 1:
                            distances[child] = distances[current_node] + 1
                            parents_tracker[child] = [current_node]
                            queue.append(child)
                        elif distances[child] == distances[current_node] + 1:
                            # If same distance, add this path as a potential route
                            parents_tracker[child].append(current_node)

                elif child.node_type == "OR":
                    # For OR node, only one parent needs to be visited
                    if distances[child] > distances[current_node] + 1:
                        distances[child] = distances[current_node] + 1
                        parents_tracker[child] = [current_node]
                        queue.append(child)
                    elif distances[child] == distances[current_node] + 1:
                        # If same distance, add this path as a potential route
                        parents_tracker[child].append(current_node)

        # Backtrack to find the path from the goal to the root(s)
        def backtrack_path(goal_node):
            path = []
            current = goal_node
            while parents_tracker[current]:
                path.append(current)
                current = parents_tracker[current][0]  # Always take the first parent in the list
            path.append(current)  # Add the root node
            return path[::-1]  # Reverse the path to go from root to goal

        if distances[self.nodes[-1]] == float('inf'):
            return None  # No path found

        return backtrack_path(self.nodes[-1])


    def find_shortest_path(self, root_nodes):
        # This will store the shortest path to the goal
        goal_node = self.nodes[-1]

        # Queue for BFS: it stores tuples (current_node, path_to_node)
        queue = deque()

        # Set to keep track of visited nodes to avoid cycles
        visited = set()

        # Initialize the queue with root nodes and the paths to them
        for root in root_nodes:
            if isinstance(root, int):
                root_node = self.nodes[root]
            elif isinstance(root, Node):
                root_node = root
            else:
                raise ValueError("Root nodes must be either Node instances or integer indices")

            queue.append((root_node, [root_node]))
            visited.add(root_node)

        # BFS to find the shortest path to the goal node
        while queue:
            current_node, path = queue.popleft()

            # If we reached the goal node, return the path
            if current_node == goal_node:
                return path

            # Otherwise, explore the children of the current node
            for child in current_node.children:
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))

        # If no path is found, return an empty list or None
        return None

    def count_nodes_in_all_paths(self, node_list):
        goal_node = self.nodes[-1]

        def dfs(current_node, path):
            if current_node == goal_node:
                return [set(path)]

            paths = []
            for child in current_node.children:
                paths.extend(dfs(child, path + [child]))
            return paths

        def common_nodes_across_paths(paths):
            if not paths:
                return set()

            # Find the intersection of all sets (nodes present in all paths)
            common_nodes = paths[0]
            for path in paths[1:]:
                common_nodes &= path
            return common_nodes

        result = {}

        for node in node_list:
            if isinstance(node, int):
                current_node = self.nodes[node]
            elif isinstance(node, Node):
                current_node = node
            else:
                raise ValueError("Nodes in the list must be either Node instances or integer indices")

            # Get all paths from current_node to the goal_node
            all_paths = dfs(current_node, [current_node])

            # Convert list of sets into a single set of common nodes across all paths
            common_nodes = common_nodes_across_paths(all_paths)

            # Store the count of common nodes for the current node
            if len(common_nodes)==0:
                result[current_node.value]=1e10
            else:
                result[current_node.value] = len(common_nodes)

        return result

    def dfs_paths(self, current_node, goal_node):
        # Helper function to return all paths from current_node to goal_node as sets of nodes
        def dfs(node, path):
            if node == goal_node:
                return [set(path)]

            paths = []
            for child in node.children:
                paths.extend(dfs(child, path + [child]))
            return paths

        return dfs(current_node, [current_node])

    def common_nodes_between_pairs(self, node_list):
        goal_node = self.nodes[-1]

        # Dictionary to store the paths for each node in node_list
        node_paths = {}

        # Get all paths from each node in the node_list to the goal node
        for node in node_list:
            if isinstance(node, int):
                current_node = self.nodes[node]
            elif isinstance(node, Node):
                current_node = node
            else:
                raise ValueError("Nodes in the list must be either Node instances or integer indices")

            node_paths[current_node.value] = self.dfs_paths(current_node, goal_node)

        # Helper function to compute common nodes between two sets of paths
        def common_nodes_across_two_sets(paths1, paths2):
            common_nodes = set.intersection(*paths1) if paths1 else set()
            for path in paths2:
                common_nodes &= path
            return common_nodes

        # Dictionary to store the common nodes between every pair of nodes
        common_nodes_count = {}

        # Iterate over all pairs of nodes in the node_list and find common nodes
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                node1 = node_list[i]
                node2 = node_list[j]

                # Get the paths sets for both nodes
                paths1 = node_paths[node1.value]
                paths2 = node_paths[node2.value]

                # Find the common nodes across all paths between the two nodes
                common_nodes = common_nodes_across_two_sets(paths1, paths2)

                if len(common_nodes) == 0:
                    common_nodes_count[(node1.value, node2.value)]  = 1e10
                else:
                    # Store the count of common nodes between the two nodes
                    common_nodes_count[(node1.value, node2.value)] = len(common_nodes)

        return common_nodes_count


def calculate_shd_and_differences(A, B):
    if len(A) != len(B):
        raise ValueError("The graphs must have the same number of nodes")

    extra_edges = 0
    missing_edges = 0

    # Count extra and missing edges
    for i in range(len(A)):
        for j in range(len(A[i])):
            if A[i][j] == 0 and B[i][j] == 1:
                extra_edges += 1
            elif A[i][j] == 1 and B[i][j] == 0:
                missing_edges += 1

    # SHD is the sum of differences in edges
    shd = extra_edges + missing_edges

    return shd, extra_edges, missing_edges, np.sum(A), np.sum(B)


def find_discoverable_parents(adj_matrix, gate_types):
    n = len(adj_matrix)  # Number of gates
    discoverable_adj_matrix = [[0] * n for _ in range(n)]

    for i in range(n):  # Iterate over each gate
        gate_type = gate_types[i]
        parents = [j for j in range(n) if adj_matrix[j][i] == 1]  # Find parents of gate i

        for j in parents:
            other_parents = [p for p in parents if p != j]

            if gate_type == 'OR':
                # For OR gates, parent j is discoverable if it can be the sole activator
                # Check if no other parent can activate the gate i
                if all(adj_matrix[p][i] == 0 for p in other_parents):
                    discoverable_adj_matrix[j][i] = 1

            elif gate_type == 'AND':
                # For AND gates, parent j is discoverable if gate i remains active even if j is off
                # Check if all other parents can keep the gate active
                if all(adj_matrix[p][i] == 1 for p in other_parents):
                    discoverable_adj_matrix[j][i] = 1

    return discoverable_adj_matrix


def bfs_shortest_path(start, goal):
    queue = [start]
    visited = set()

    while queue:
        current_node = queue.pop(0)
        if current_node == goal:
            return True
        for child in current_node.children:
            if child not in visited:
                visited.add(child)
                queue.append(child)

    return False



def create_erdos_renyi_graph(n, p):
    # nxG = nx.random_tree(n)
    graph = Graph(nodes=[Node(i) for i in range(n)])
    start_node, goal_node = graph.nodes[0], graph.nodes[-1]

    #create a connected graph where every nore is reachable by start node.
    for i in range(1, n):
        parent_idx = np.random.randint(0, i)
        graph.nodes[parent_idx].add_child(graph.nodes[i])

    #Now add edges to it for erdos reyni
    for i in range(n - 1):
        for j in range(i + 1, n):
            if random.random() < p:
                graph.nodes[i].add_child(graph.nodes[j])

    return graph, start_node, goal_node



def create_semi_erdos_renyi_graph(n, p):
    # nxG = nx.random_tree(n)
    graph = Graph(nodes=[Node(i) for i in range(n)])
    start_node, goal_node = graph.nodes[0], graph.nodes[-1]
    #Now add edges to it for erdos reyni
    for i in range(n - 1):
        for j in range(i + 1, n):
            if random.random() < p:
                graph.nodes[i].add_child(graph.nodes[j])

    return graph, start_node, goal_node

def create_tree(depth, branching_factor):
    graph = Graph(nodes=[])

    root = Node(0)  # Root node starts with value 0
    graph.add_node(root)
    current_level_nodes = [root]

    node_counter = 1  # Start counting nodes from 1

    for i in range(1, depth):
        next_level_nodes = []

        for parent_node in current_level_nodes:
            for _ in range(branching_factor):
                child_node = Node(node_counter)  # Use an integer for node value
                graph.add_node(child_node)
                parent_node.add_child(child_node)
                next_level_nodes.append(child_node)
                node_counter += 1  # Increment the counter for the next node

        current_level_nodes = next_level_nodes

    # If you need to set a specific node's value to 'G', convert it to an integer
    # or handle it according to your requirements
    goal_node = graph.nodes[-1]

    return graph, root, goal_node