import argparse
import numpy as np
import copy
import time
import os
import multiprocessing
import random
import torch

seed_number = 20
np.random.seed(seed_number)
random.seed(seed_number)
shared_data_global=None


class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0

    def add(self, transition):
        assert len(transition) == 7, "transition must have length = 7"

        # transiton is tuple of (state, action, reward, next_state, goal, gamma, done)
        self.buffer.append(copy.deepcopy(transition))
        self.size += 1

    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size / 5)]
            self.size = len(self.buffer)

        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        states, actions, rewards, next_states, goals, gamma, dones = [], [], [], [], [], [], []

        for i in indexes:
            states.append(np.asarray(self.buffer[i][0]))
            actions.append(np.asarray(self.buffer[i][1]))
            rewards.append(np.asarray(self.buffer[i][2]))
            next_states.append(np.asarray(self.buffer[i][3]))
            goals.append(np.asarray(self.buffer[i][4]))
            gamma.append(np.asarray(self.buffer[i][5]))
            dones.append(np.asarray(self.buffer[i][6]))

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(goals), np.array(
            gamma), np.array(dones)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_worker(shared_data):
    """
    Initialize each worker process with a unique seed.
    This function is called when each worker process starts.
    """
    global shared_data_global
    shared_data_global=shared_data
    seed = (os.getpid() * int(time.time())) % 123456789
    random.seed(seed)
    np.random.seed(seed)

def get_shared_data_global():
    global shared_data_global
    return shared_data_global

def set_shared_data_global(data):
    global shared_data_global
    shared_data_global=data


def create_pool(shared_data=None):
    number_of_processes = multiprocessing.cpu_count()
    # seeds = [(random.randint(0, 1000000),) for _ in range(number_of_processes)]
    pool = multiprocessing.Pool(number_of_processes, initializer=init_worker,initargs=(shared_data,) )
    return pool, number_of_processes


def one_hot_encode(indices, num_classes):
    """
    Manually one-hot encode the given indices.

    :param indices: A tensor of indices to be one-hot encoded.
    :param num_classes: The total number of classes.
    :return: A one-hot encoded tensor.
    """
    # Create a tensor of zeros with the desired shape
    one_hot = torch.zeros(indices.shape[0], num_classes, dtype=torch.float32)
    # Use scatter_ to assign 1s to the appropriate indices
    one_hot.scatter_(1, indices.unsqueeze(1), 1)
    return one_hot