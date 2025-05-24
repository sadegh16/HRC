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
from graph_utils import create_tree
from utils import create_pool
from collections import ChainMap


def run_task(min_depth,max_depth):
    node_sizes_data={}
    for depth in range(min_depth, max_depth):
        number_of_nodes = int((3 ** depth - 1)/2)
        print("number_of_nodes: ", number_of_nodes)
        true_graph_adjacency_graphs = []
        true_graph_node_type_graphs = []
        graph, root, goal, = create_tree(depth, 3)
        true_graph_adjacency = graph.create_adjacency_matrix()
        true_graph_adjacency_graphs.append(true_graph_adjacency)
        true_graph_node_type_graphs.append([node.node_type for node in graph.nodes])

        node_sizes_data[number_of_nodes]=[true_graph_adjacency_graphs,
                                          true_graph_node_type_graphs]
    print("Saving...")
    # At the end of the function, where you need to save data
    data_to_save = {
        'node_sizes_data':node_sizes_data
    }

    with open('collected-dataset/exploration-data-tree-d3-9.pickle', 'wb') as file:
        pickle.dump(data_to_save, file)

    print("Data saved successfully.")


# Main function that initiates the entire process
def main():
    run_task(3,10, )  # Example function call to run the task with specified parameters


if __name__ == '__main__':
    # Explicitly set the start method for multiprocessing
    multiprocessing.set_start_method('spawn')
    main()
