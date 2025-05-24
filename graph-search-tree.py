import multiprocessing
import pdb
from multiprocessing import Manager
import numpy as np
from numpy.random import randn, seed
from SCM import ScmLR
from graph_utils import Graph, Node, calculate_shd_and_differences
import pickle
from copy import copy, deepcopy

from utils import create_pool, set_shared_data_global


def stochastic_search(true_graph, verbose=True, uniform=True, ):
    start, goal = true_graph.nodes[0], true_graph.nodes[-1]
    I, C = [], [start],
    beta_R = {}
    itr = 0
    # while goal not in C:
    # Set up a multiprocessing pool
    while goal not in C and itr < 3**10:
        print("Iteration:", itr,)
        if len(C) > 0:
            if uniform is False:
                preference = [True if true_graph.is_ancestor_of_goal(node_c) else False for node_c in C ]
                prob = []
                good_to_go = sum(preference)
                not_good_to_go = len(preference) - good_to_go
                for node in preference:
                    if node == True and not_good_to_go > 0:
                        prob.append(len(I) / (good_to_go * ( len(I) + 1)))
                        # prob.append(1 / good_to_go)
                    elif node == True and not_good_to_go == 0:
                        prob.append(1 / good_to_go)
                    elif node == False and good_to_go>0:
                        prob.append(1 / (not_good_to_go * (len(I) + 1)))
                    elif node == False and good_to_go==0:
                        prob.append(1 / not_good_to_go)
                        # prob.append(0)

                g = np.random.choice(C, p=prob)
                # print(C,prob)
            else:
                g = np.random.choice(C, )

            if verbose:
                print("g", g)
            C.remove(g)
            I.append(g)

        # check if the node is trained, or can be controllable or not
        for intervened_node in I:
            CH_g = intervened_node.children
            for g_prime in CH_g:
                if g_prime not in I and g_prime not in C:
                    original_g_prime = true_graph.nodes[g_prime.value]
                    if original_g_prime.node_type == "OR" and any([
                        true_graph.nodes[parent.value] in I for parent in original_g_prime.parents]):
                        C.append(g_prime)
                        beta_R[g_prime] = len(I) + 1
                    elif original_g_prime.node_type == "AND" and all([
                        true_graph.nodes[parent.value] in I for parent in original_g_prime.parents]):
                        C.append(g_prime)
                        beta_R[g_prime] = len(I) + 1

        if verbose:
            print("C", C)
        itr += 1

    return beta_R, sum([beta_R[item] for item in beta_R]), len(I)



def run_task(trials=10):
    data_to_save={}
    with open('collected-dataset/exploration-data-tree-d3-9.pickle', 'rb') as file:
        loaded_data = pickle.load(file)
    node_sizes_data=loaded_data["node_sizes_data"]
    # Initialize the pool once
    shared_dict=Manager().dict()
    set_shared_data_global(shared_dict)
    pool, _ = create_pool(shared_data=shared_dict)
    for node_size, node_size_data  in node_sizes_data.items():
        true_graph_adjacency_graphs = node_size_data[0]
        true_graph_node_type_graphs = node_size_data[1]
        uniform_cost_graphs = []
        our_cost_graphs = []

        for true_graph_adjacency, graph_node_types in zip(true_graph_adjacency_graphs,
                                                                             true_graph_node_type_graphs):
            graph = Graph()
            graph.set_graph_adjacency(true_graph_adjacency)
            graph.set_node_types(graph_node_types)
            print("Connected edges to goal:", np.where(true_graph_adjacency[:, - 1])[0])
            for _ in range(trials):
                _, cost, _ = stochastic_search(
                    graph,
                    verbose=True,
                    uniform=True,)
                uniform_cost_graphs.append(cost)

                _, cost, _ = stochastic_search(
                     graph,
                    verbose=True,
                    uniform=False, )

                our_cost_graphs.append(cost)

        data_to_save[node_size] = {
            'true_graph_adjacency_graphs': true_graph_adjacency_graphs,
            'uniform_cost_graphs': uniform_cost_graphs,
            'our_cost_graphs': our_cost_graphs,
        }
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()
    print("Saving...")
    # At the end of the function, where you need to save data


    with open('collected-dataset/search_causal_tree_d3-9.pickle', 'wb') as file:
        pickle.dump(data_to_save, file)

    print("Data saved successfully.")


# Main function that initiates the entire process
def main():
    run_task()  # Example function call to run the task with specified parameters


if __name__ == '__main__':
    # Explicitly set the start method for multiprocessing
    multiprocessing.set_start_method('spawn')
    main()
