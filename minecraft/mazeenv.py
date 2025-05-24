import os
import sys
import torch
from minecraft.mazemap import Mazemap
import numpy as np
from minecraft.utils import WHITE, BLACK, DARK, LIGHT, GREEN, DARK_RED
from minecraft.mining import Mining


class MazeEnv(object):  # single batch
    def __init__(self, game_len, sub_task_id_list, render_config={}):
        game_config = Mining()

        self.config = game_config
        self.max_task = self.config.nb_subtask_type
        self.subtask_list = self.config.subtask_list
        if len(render_config)>0:
            self.rendering = render_config['vis']
        else:
            self.rendering = False

        self.sub_task_id_list = sub_task_id_list

        # map
        self.map = Mazemap(game_config, render_config)

        # init
        self.game_length = game_len
        self.step_reward = 0.0

    def step(self, action, print_en = False):
        if self.game_over or self.time_over:
            raise ValueError(
                'Environment has already been terminated. need to be reset!')
        self.map.act(action)
        self.step_count += 1
        self.time_over = self.step_count >= self.game_length
        self.game_over = self.game_over_num>0 
        self.game_over_num = 1 if self.map.get_task_achieved() else 0
        self.reward = 1 if self.game_over else 0
        done = (self.game_over or self.time_over)
        
        last_vars = self.variables.clone()
        last_info = self.aux_info.clone()
        distance = 0
        items = [3, 4, 5, 6, 9, 10, 11, 12, 14] 
        distance = 0
        for item_idx in items:
            if last_vars[10+item_idx] == 0:
                distance += 1

        self.last_game_over = self.game_over
        if self.rendering:
            self.render()
        if done:
            self.state, self.variables, _ = self.reset()
        else:

            bp = torch.zeros(self.config.nb_obj_type)
            for obj in self.map.backpack:
                bp[obj] = 1 
            bp = torch.nn.functional.one_hot(bp.long(), num_classes=2).view(-1)
            self.state = torch.cat([self.rel_state(), bp], dim=0)
            variables, aux_info = self.map.get_variables()
            self.variables = torch.from_numpy(variables)
            self.aux_info = torch.from_numpy(aux_info)
        return self.state, self.variables, self.reward, done, {'step_time':1, 'aux_info':self.aux_info, 'distance':distance}

    def reset(self, seed=None):  # after every episode
        if seed is not None:
            np.random.seed(seed)
        self.step_count = 0
        self.time_over = False 
        self.game_over_num = 0
        self.game_over = False 
        # reset map
        self.map.reset(self.sub_task_id_list)
        if self.rendering:
            self.render()
        obs = torch.from_numpy(self.map.obs)
        bp = torch.zeros(self.config.nb_obj_type)
        for obj in self.map.backpack:
            bp[obj] = 1 
        bp = torch.nn.functional.one_hot(bp.long(), num_classes=2).view(-1)
        self.state = torch.cat([self.rel_state(), bp], dim=0)
        variables, aux_info = self.map.get_variables()
        self.variables = torch.from_numpy(variables)
        self.aux_info = torch.from_numpy(aux_info) 

        return self.state, self.variables, {'step_time':0, 'aux_info':self.aux_info, 'distance':0} 
    def rel_state(self):
        a_pos = [self.map.agent_x, self.map.agent_y]
        rel_pos = torch.zeros(5*self.config.nb_map_obj_type+4)
        for x in range(10):
            for y in range(10):
                obj_type = self.map.item_map[x, y] - 3
                if obj_type >= 0:
                    if x > a_pos[0]:
                        rel_pos[obj_type*5] = 1
                    elif x < a_pos[0]:
                        rel_pos[obj_type*5+1] = 1
                    elif y > a_pos[1]:
                        rel_pos[obj_type*5+2] = 1
                    elif y < a_pos[1]:
                        rel_pos[obj_type*5+3] = 1
                    else:
                        rel_pos[obj_type*5+4] = 1
        if a_pos[0] == 1:
            rel_pos[-4] = 1
        if a_pos[0] == 8:
            rel_pos[-3] = 1
        if a_pos[1] == 1:
            rel_pos[-2] = 1
        if a_pos[1] == 8:
            rel_pos[-1] = 1
        return rel_pos
                    
