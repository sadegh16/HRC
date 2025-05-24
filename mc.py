from minecraft.mazeenv import MazeEnv
import gym
import torch
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
import numpy as np
import gymnasium

class McEnv(gym.Env):
    def __init__(self, seed = 42, sub_task_id_list=[12]):
        self.variable_names = [
                               'workspace',#0
                               'furnace',#1
                               'jewelershop',#2
                               'wood',#3
                               'stone',#4
                               'coal',#5
                               'ironore',#6
                               'silverore',#7
                               'goldore',#8
                               'diamond',#9
                               'stick',#10
                               'stonepickaxe',#11
                               'iron',#12
                               'silver',#13
                               'ironpickaxe',#14
                               'gold',#15
                               'earings',#16
                               'ring',#17
                               'goldware',#18
                               'bracelet',#19
                               'necklace',#20
                               'Action'#21
                               ]
        self.variable_ranges = [2 for i in range(len(self.variable_names))]   
        self.variable_ranges[-1] = 6
        self.variable_num = len(self.variable_ranges)
        self.variable_operators = [0]
        #self.aux_info_ranges = [11 for i in range(4)]
        #self.aux_info_num = len(self.aux_info_ranges)
        #self.aux_info_ranges = [11, 2, 2, 2, 2]
        #self.aux_info_num = 5
        self.aux_info_ranges = [11]
        self.aux_info_num = 1
        #self.aux_info_ranges = []
        #self.aux_info_num = 0
        self.render_mode = None
        #sub_task_id_list = [i for i in range(18)] 
        # sub_task_id_list = [12]
        #sub_task_id_list = [7] 
        self.env = MazeEnv(game_len=100, sub_task_id_list=sub_task_id_list)
        self.action_space = gym.spaces.Discrete(6)
        self.reset(seed = seed)
        self.observation_space = gym.spaces.Box(shape=self.state.shape, low=-10000, high=10000)
        obj_num = [0 for i in range(self.env.config.nb_map_obj_type)]
        for x in range(10):
            for y in range(10):
                if self.env.map.item_map[x, y] >= 3:
                    obj_idx = self.env.map.item_map[x, y] - 3 
                    obj_num[obj_idx] = obj_num[obj_idx]+1
        print('map items', [(self.variable_names[i], obj_num[i]) for i in range(self.env.config.nb_map_obj_type)])
         
    def reset(self, seed=None, options=None):
        self.state, self.variables, self.info = self.env.reset(seed)
        self.variables = self.variables[-self.variable_num:]
        return self.state, self.variables
    
    def step(self, action, print_en = False):
        self.state, self.variables, self.reward, done, self.info = self.env.step(action, print_en)
        self.variables = self.variables[-self.variable_num:]
        return self.state, self.variables, self.reward, done, self.info


class LatentMcEnv(gym.Env):
    def __init__(self, seed=42, sub_task_id_list=[12]):
        self.variable_names = [
            'workspace',  # 0
            'furnace',  # 1
            'jewelershop',  # 2
            'wood',  # 3
            'stone',  # 4
            'coal',  # 5
            'ironore',  # 6
            'silverore',  # 7
            'goldore',  # 8
            'diamond',  # 9
            'stick',  # 10
            'stonepickaxe',  # 11
            'iron',  # 12
            'silver',  # 13
            'ironpickaxe',  # 14
            'gold',  # 15
            'earings',  # 16
            'ring',  # 17
            'goldware',  # 18
            'bracelet',  # 19
            'necklace',  # 20
            'Action'  # 21
        ]
        self.variable_ranges = [2 for i in range(len(self.variable_names))]
        self.variable_ranges[-1] = 6
        self.variable_num = len(self.variable_ranges)
        self.variable_operators = [0]
        # self.aux_info_ranges = [11 for i in range(4)]
        # self.aux_info_num = len(self.aux_info_ranges)
        # self.aux_info_ranges = [11, 2, 2, 2, 2]
        # self.aux_info_num = 5
        self.aux_info_ranges = [11]
        self.aux_info_num = 1
        # self.aux_info_ranges = []
        # self.aux_info_num = 0
        self.render_mode = None
        # sub_task_id_list = [i for i in range(18)]
        # sub_task_id_list = [12]
        # sub_task_id_list = [7]
        self.env = MazeEnv(game_len=100, sub_task_id_list=sub_task_id_list)
        self.action_space = gym.spaces.Discrete(6)
        self.reset(seed=seed)
        self.observation_space = gym.spaces.Box(shape=self.state.shape, low=-10000, high=10000)
        obj_num = [0 for i in range(self.env.config.nb_map_obj_type)]
        for x in range(10):
            for y in range(10):
                if self.env.map.item_map[x, y] >= 3:
                    obj_idx = self.env.map.item_map[x, y] - 3
                    obj_num[obj_idx] = obj_num[obj_idx] + 1
        print('map items', [(self.variable_names[i], obj_num[i]) for i in range(self.env.config.nb_map_obj_type)])

    def reset(self, seed=None, options=None):
        self.state, self.variables, self.info = self.env.reset(seed)
        self.variables = self.variables[-self.variable_num:]
        new_variables=self.variables.clone()
        new_variables[0]=new_variables[1]=new_variables[2]=new_variables[3]=0

        return self.state, new_variables

    def step(self, action, print_en=False):
        self.state, self.variables, self.reward, done, self.info = self.env.step(action, print_en)
        self.variables = self.variables[-self.variable_num:]
        new_variables=self.variables.clone()
        new_variables[0]=new_variables[1]=new_variables[2]=new_variables[3]=0
        return self.state, new_variables, self.reward, done, self.info


def complete_graph(var_num=22, ):
    dag = torch.zeros((var_num, var_num)).bool()
    dag[3, var_num - 1] = True
    dag[4, var_num - 1] = True
    dag[11, 4] = True
    dag[11, 10] = True
    dag[10, 3] = True
    dag[6, 11] = True
    dag[5, 11] = True
    dag[7, 11] = True
    dag[12, 6] = True
    dag[12, 5] = True
    dag[13, 5] = True
    dag[13, 7] = True
    dag[14, 10] = True
    dag[14, 12] = True
    dag[8, 14] = True
    dag[15, 5] = True
    dag[15, 8] = True
    dag[9, 14] = True
    dag[16, 13] = True
    dag[16, 9] = True
    dag[17, 12] = True
    dag[17, 9] = True
    dag[18, 15] = True
    dag[19, 12] = True
    dag[19, 13] = True
    dag[19, 15] = True
    dag[20, 15] = True
    dag[20, 9] = True

    return dag


def true_graph(var_num=22, ):

    dag = torch.zeros((var_num, var_num)).bool()
    dag[3,var_num-1]=True
    dag[4,var_num-1]=True
    dag[11,4]=True
    dag[11,10]=True
    dag[10,3]=True
    dag[6,11]=True
    dag[5,11]=True
    dag[12,6]=True
    dag[12,5]=True

    return dag


class McEnvForKir(gym.Env):
    def __init__(self, seed=42, sub_task_id_list=[12]):
        self.variable_names = [
            'workspace',  # 0
            'furnace',  # 1
            'jewelershop',  # 2
            'wood',  # 3
            'stone',  # 4
            'coal',  # 5
            'ironore',  # 6
            'silverore',  # 7
            'goldore',  # 8
            'diamond',  # 9
            'stick',  # 10
            'stonepickaxe',  # 11
            'iron',  # 12
            'silver',  # 13
            'ironpickaxe',  # 14
            'gold',  # 15
            'earings',  # 16
            'ring',  # 17
            'goldware',  # 18
            'bracelet',  # 19
            'necklace',  # 20
            'Action'  # 21
        ]
        self.variable_ranges = [2 for i in range(len(self.variable_names))]
        self.variable_ranges[-1] = 6
        self.variable_num = len(self.variable_ranges)
        self.variable_operators = [0]
        # self.aux_info_ranges = [11 for i in range(4)]
        # self.aux_info_num = len(self.aux_info_ranges)
        # self.aux_info_ranges = [11, 2, 2, 2, 2]
        # self.aux_info_num = 5
        self.aux_info_ranges = [11]
        self.aux_info_num = 1
        # self.aux_info_ranges = []
        # self.aux_info_num = 0
        self.render_mode = None
        # sub_task_id_list = [i for i in range(18)]
        # sub_task_id_list = [12]
        # sub_task_id_list = [7]
        self.env = MazeEnv(game_len=100, sub_task_id_list=sub_task_id_list)
        self.action_space = gymnasium.spaces.discrete.Discrete(6)
        self.reset(seed=seed)
        self.observation_space = gymnasium.spaces.Box(shape=(22,), low=-10000, high=10000)
        obj_num = [0 for i in range(self.env.config.nb_map_obj_type)]
        for x in range(10):
            for y in range(10):
                if self.env.map.item_map[x, y] >= 3:
                    obj_idx = self.env.map.item_map[x, y] - 3
                    obj_num[obj_idx] = obj_num[obj_idx] + 1
        print('map items', [(self.variable_names[i], obj_num[i]) for i in range(self.env.config.nb_map_obj_type)])

    def reset(self, seed=None, options=None):
        self.state, self.variables, self.info = self.env.reset(seed)
        self.variables = self.variables[-self.variable_num:]
        return self.variables, {}

    def step(self, action, print_en=False):
        self.state, self.variables, self.reward, done, self.info = self.env.step(action, print_en)
        self.variables = self.variables[-self.variable_num:]
        return self.variables, self.reward, self.env.game_over ,done, self.info

