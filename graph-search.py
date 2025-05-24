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


def stochastic_search_ancestral(true_graph, verbose=True, uniform=True, ):
    start, goal = true_graph.nodes[0], true_graph.nodes[-1]
    I, C = [], [node for node in true_graph.nodes if len(node.parents)==0]
    beta_R = {}
    itr = 0
    # while goal not in C:
    # Set up a multiprocessing pool
    while goal not in C and itr < 1000:
        print("Iteration:", itr,)
        if len(C) > 0:
            if uniform is False:
                preference = [True if true_graph.is_ancestor_of_goal(node_c) else False for node_c in C ]
                prob = []
                good_to_go = sum(preference)
                not_good_to_go = len(preference) - good_to_go
                for node in preference:
                    if node == True and not_good_to_go > 0:
                        prob.append( len(I) / (good_to_go * ( len(I) + 1)))
                        # prob.append(1 / good_to_go)
                    elif node == True and not_good_to_go == 0:
                        prob.append(1 / good_to_go)
                    elif node == False and good_to_go>0:
                        prob.append(1 / (not_good_to_go * ( len(I) + 1)))
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


def stochastic_search_shortest_path(true_graph, verbose=True, uniform=True, ):
    _, goal = true_graph.nodes[0], true_graph.nodes[-1]
    I, C = [], [node for node in true_graph.nodes if len(node.parents)==0]
    beta_R = {}
    itr = 0
    # while goal not in C:
    # Set up a multiprocessing pool
    while goal not in C and itr < 1000:
        print("Iteration:", itr,)
        if len(C) > 0:
            if uniform is False:
                shortest_path = true_graph.find_shortest_path(C)
                preference = [node_c in shortest_path for node_c in C]
                prob = []
                good_to_go = sum(preference)
                not_good_to_go = len(preference) - good_to_go
                for node in preference:
                    if node == True and not_good_to_go > 0:
                        prob.append( len(I) / (good_to_go * ( len(I) + 1)))
                        # prob.append(1 / good_to_go)
                    elif node == True and not_good_to_go == 0:
                        prob.append(1 / good_to_go)
                    elif node == False and good_to_go>0:
                        prob.append(1 / (not_good_to_go * ( len(I) + 1)))
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



def stochastic_search_hybrid(true_graph, verbose=True, uniform=True, ):
    _, goal = true_graph.nodes[0], true_graph.nodes[-1]
    I, C = [], [node for node in true_graph.nodes if len(node.parents)==0]
    g_func={}
    for node in C:
        g_func[node]=0
    beta_R = {}
    itr = 0
    # while goal not in C:
    # Set up a multiprocessing pool
    while goal not in C and itr < 1000:
        print("Iteration:", itr,)
        if len(C) > 0:
            if uniform is False:
                # node_counts=true_graph.count_nodes_in_all_paths(C)
                # node_counts={}
                # for node in C:
                #     sp=true_graph.find_shortest_path([node])
                #     if sp:
                #         node_counts[node]=len(sp)
                #     else:
                #         node_counts[node] = 10**10
                # # Sort root_nodes by their count in node_counts in decreasing order
                # sorted_nodes = sorted(C, key=lambda x: node_counts.get(x, 10**10)+g_func[x], reverse=True)
                # # breakpoint()
                # to_be_prefered=[]
                # for n in sorted_nodes[-2:]:
                #     if true_graph.is_ancestor_of_goal(n): to_be_prefered.append(n)

                sp=true_graph.find_shortest_path_andor(C)
                # breakpoint()
                preference = [node_c in sp  for node_c in C]
                prob = []
                good_to_go = sum(preference)
                not_good_to_go = len(preference) - good_to_go
                for node in preference:
                    if node == True and not_good_to_go > 0:
                        prob.append( len(I) / (good_to_go * ( len(I) + 1)))
                        # prob.append(1 / good_to_go)
                    elif node == True and not_good_to_go == 0:
                        prob.append(1 / good_to_go)
                    elif node == False and good_to_go>0:
                        prob.append(1 / (not_good_to_go * ( len(I) + 1)))
                    elif node == False and good_to_go==0:
                        prob.append(1 / not_good_to_go)

                g = np.random.choice(C, p=prob)

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
                if g_prime not in I:
                    # update g_func
                    if g_prime in g_func:
                        g_func[g_prime] = 0
                        for parent in g_prime.parents:
                            if parent in I:
                                if g_func[parent] + len(parent.children) + 1 < g_func[g_prime]:
                                    g_func[g_prime] = g_func[parent] + len(parent.children) + 1
                    else:
                        g_func[g_prime]=g_func[intervened_node]+len(CH_g)+1
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


def run_task():
    data_to_save={}
    with open('collected-dataset/exploration-data-no-causal-semi-erandor0.5-n10-500-c=0.5.pickle', 'rb') as file:
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
        our_ancestor_cost_graphs = []
        our_shortest_cost_graphs = []

        for true_graph_adjacency, graph_node_types in zip(true_graph_adjacency_graphs,
                                                                             true_graph_node_type_graphs):
            graph = Graph()
            graph.set_graph_adjacency(true_graph_adjacency)
            graph.set_node_types(graph_node_types)
            print("Connected edges to goal:", np.where(true_graph_adjacency[:, - 1])[0])
            _, cost, _ = stochastic_search_ancestral(
                graph,
                verbose=True,
                uniform=True,)
            uniform_cost_graphs.append(cost)

            _, cost, _ = stochastic_search_ancestral(
                 graph,
                verbose=True,
                uniform=False, )
            our_ancestor_cost_graphs.append(cost)

            _, cost, _ = stochastic_search_shortest_path(
                 graph,
                verbose=True,
                uniform=False, )

            our_shortest_cost_graphs.append(cost)

        data_to_save[node_size] = {
            'true_graph_adjacency_graphs': true_graph_adjacency_graphs,
            'uniform_cost_graphs': uniform_cost_graphs,
            'our_ancestor_cost_graphs': our_ancestor_cost_graphs,
            'our_shortest_cost_graphs': our_shortest_cost_graphs,
        }
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()
    print("Saving...")
    # At the end of the function, where you need to save data


    with open('collected-dataset/ancestor_shortest_semi-erandor0.5_n10-500-no-casual-c=0.5.pickle', 'wb') as file:
        pickle.dump(data_to_save, file)

    print("Data saved successfully.")


# Main function that initiates the entire process
def main():
    run_task()  # Example function call to run the task with specified parameters


if __name__ == '__main__':
    # Explicitly set the start method for multiprocessing
    multiprocessing.set_start_method('spawn')
    main()
