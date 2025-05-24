import multiprocessing
import pdb
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample
# from sklearn.exceptions import ConvergenceWarning
# from warnings import simplefilter
# from sklearn.utils.class_weight import compute_class_weight
from numpy.random import randn, seed
# simplefilter("ignore", category=ConvergenceWarning)
import time
import os
import pickle
from graph_utils import create_erdos_renyi_graph, create_semi_erdos_renyi_graph
from utils import create_pool




def run_task(node_sizes=[], n_graphs=100000):
    node_sizes_data={}
    for number_of_nodes in node_sizes:
        print("number_of_nodes: ",number_of_nodes)
        true_graph_adjacency_graphs = []
        true_graph_node_type_graphs = []
        for _ in range(n_graphs):
            p = 1 * np.log(number_of_nodes) / number_of_nodes
            print("p:", p)
            graph, _, _, = create_semi_erdos_renyi_graph(number_of_nodes, p)
            true_graph_adjacency = graph.create_adjacency_matrix()
            true_graph_adjacency_graphs.append(true_graph_adjacency)
            true_graph_node_type_graphs.append([node.node_type for node in graph.nodes])
            print("Connected edges to goal:", np.where(true_graph_adjacency[:, number_of_nodes - 1])[0])


        node_sizes_data[number_of_nodes]=[true_graph_adjacency_graphs,
                                          true_graph_node_type_graphs]
    print("Saving...")
    # At the end of the function, where you need to save data
    data_to_save = {
        'node_sizes_data':node_sizes_data
    }

    with open('collected-dataset/exploration-data-no-causal-semi-erandor0.5-n10-c=1.0.pickle', 'wb') as file:
        pickle.dump(data_to_save, file)

    print("Data saved successfully.")


# Main function that initiates the entire process
def main():
    run_task([i for i in range(10, 11)], )  # Example function call to run the task with specified parameters


if __name__ == '__main__':
    # Explicitly set the start method for multiprocessing
    multiprocessing.set_start_method('spawn')
    main()
