try:
    import pygame
    import pygame.freetype
except ImportError:
    pygame = None

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from minecraft.utils import MOVE_ACTS, AGENT, BLOCK, WATER, KEY, OBJ_BIAS,\
    TYPE_PICKUP, TYPE_TRANSFORM, \
    WHITE, BLACK, DARK, LIGHT, GREEN, DARK_RED

CHR_WIDTH = 9
TABLE_ICON_SIZE = 40
MARGIN = 10
LEGEND_WIDTH = 250

__PATH__ = os.path.abspath(os.path.dirname(__file__))


class Mazemap(object):
    def __init__(self, game_config, render_config):
        # visualization
        if len(render_config)>0:
            self._rendering = render_config['vis']
            self.save_flag = render_config['save'] and self._rendering
            self.cheatsheet = render_config['key_cheatsheet'] and self._rendering
        else:
            self._rendering = False

        self.render_dir = './render'

        # load game config (outcome)
        self.config = game_config
        self.render_scale = 48
        self.table_scale = 32

        self.operation_list = self.config.operation_list
        self.legal_actions = self.config.legal_actions

        self.step_penalty = 0.0
        self.w = game_config.width
        self.h = game_config.height

        # map tensor
        self.obs = np.zeros(
            (self.config.nb_map_obj_type+OBJ_BIAS, self.w, self.h), dtype=np.uint8)
        self.wall_mask = np.zeros((self.w, self.h), dtype=np.bool_)
        self.item_map = np.zeros((self.w, self.h), dtype=np.int16)
        self.backpack = [] 
        self.backpack_time = [] 
        self.obj_time = 200

        self.init_screen_flag = False
        if self._rendering:
            if pygame is None:
                raise ImportError(
                    "Rendering requires pygame installed on your environment: e.g. pip install pygame")
            self._init_pygame()
            self._load_game_asset()

    def reset(self, subtask_id_list):
        self.subtask_id_list = subtask_id_list
        self.nb_subtask = len(subtask_id_list)
        self.obs.fill(0)
        self.wall_mask.fill(0)
        self.item_map.fill(-1)
        self.backpack=[]
        self.backpack_time = [] 
        self.empty_list = []

        self._add_blocks()
        self._add_targets()

    def act(self, action):
        oid = -1
        act_suc = True
        for legal_action in self.legal_actions:
            if action in legal_action.value:
                action = legal_action
                break
        assert action in self.legal_actions, 'Illegal action: '
        if action in {KEY.UP, KEY.DOWN, KEY.LEFT, KEY.RIGHT}:  # move
            new_x = self.agent_x
            new_y = self.agent_y
            if action == KEY.RIGHT:
                new_x += 1
            elif action == KEY.LEFT:
                new_x -= 1
            elif action == KEY.DOWN:
                new_y += 1
            elif action == KEY.UP:
                new_y -= 1
            # wall_collision
            if not (new_x, new_y) in self.walls and not (new_x, new_y) in self.waters:
                self.obs[AGENT, self.agent_x, self.agent_y] = 0
                self.agent_x = new_x
                self.agent_y = new_y
                self.obs[AGENT, new_x, new_y] = 1
            else:
                act_suc = False
        else:  # perform
            iid = self._get_cur_item()
            if iid > -1:
                oid = iid-OBJ_BIAS
                self._perform(action, oid)  # perform action in the map
        #self._process_obj()  # moving objects
        return act_suc

    def get_task_achieved(self):
        achieved = True
        for tind in range(self.nb_subtask):
            subid = self.subtask_id_list[tind]
            _, _, _, output = self.config.subtask_param_list[subid] 
            if not(set(output) <= set(self.backpack)):
                achieved = False
                break
        return achieved
            
        
    def get_variables(self):
        # map variables + backpack variables
        variables = np.zeros((self.config.nb_map_obj_type+self.config.nb_obj_type+1))
        
        map_obj = self._get_cur_item()
        if map_obj > -1:
            map_obj = map_obj-OBJ_BIAS
            variables[map_obj] = 1
        for bp_obj in self.backpack:
            variables[self.config.nb_map_obj_type+bp_obj] = 1

        aux_info = np.zeros((1))
        if self.item_map[self.agent_x, self.agent_y] >= OBJ_BIAS:
            aux_info[0] = self.item_map[self.agent_x, self.agent_y] - OBJ_BIAS + 1
        return variables, aux_info

        

    def _add_item(self, oid, pos):
        obj = dict(oid=oid, pos=pos)
        self.obs[oid+OBJ_BIAS, pos[0], pos[1]] = 1
        self.item_map[pos[0], pos[1]] = oid+OBJ_BIAS
        self.object_list.append(obj)

    def _perform(self, action, oid):
        assert(action not in MOVE_ACTS)
        act_type = self.operation_list[action]['oper_type']
        obj = None
        for oind in range(len(self.object_list)):
            o = self.object_list[oind]
            if o['pos'] == (self.agent_x, self.agent_y):
                obj = o
                break
        assert obj is not None

        # pickup
        if act_type == TYPE_PICKUP:
            if self.config.object_param_list[oid]['pickable']:
                _, required, _, output = self.config.subtask_param_list[self.config.subtask_output_to_id[oid]] 
                assert(oid in output)
                if set(required) <= set(self.backpack):
                    self.backpack = list(set(self.backpack).union(set(output)))
            else:
                assert(not self.config.object_param_list[oid]['transformable'])
        # transform
        elif act_type == TYPE_TRANSFORM:
            trans_params_list = self.config.subtask_obj_to_id[oid] 
            for subid in trans_params_list:
                _, required, _, output = self.config.subtask_param_list[subid]
                out_id = output[0]
                if not self.config.object_param_list[out_id]['transformable']:
                    continue
                if set(required) <= set(self.backpack):
                    self.backpack = list(set(self.backpack).union(set(output)))

    def _add_blocks(self):
        # boundary
        self.walls = [(0, y) for y in range(self.h)]  # left wall
        self.walls = self.walls + [(self.w-1, y)
                                   for y in range(self.h)]  # right wall
        self.walls = self.walls + [(x, 0)
                                   for x in range(self.w)]  # bottom wall
        self.walls = self.walls + [(x, self.h-1)
                                   for x in range(self.w)]  # top wall

        for x in range(self.w):
            for y in range(self.h):
                if (x, y) not in self.walls:
                    self.empty_list.append((x, y))
                else:
                    self.item_map[x, y] = 1  # block

        # random block
        if self.config.nb_block[0] < self.config.nb_block[1]:
            nb_block = np.random.randint(
                self.config.nb_block[0], self.config.nb_block[1])
            nb_block = 0
            pool = np.random.permutation(self.empty_list)
            count = 0
            for (x, y) in pool:
                if count == nb_block:
                    break
                # based on self.item_map & empty_list
                # try add block
                self.empty_list.remove((x, y))
                self.item_map[x, y] = 1  # block
                if self._check_block(self.empty_list):
                    self.walls.append((x, y))
                    count += 1
                else:
                    self.empty_list.append((x, y))
                    self.item_map[x, y] = -1  # block
            if count != nb_block:
                print(
                    'cannot generate a map without inaccessible regions! Decrease the #blocks')
                assert(False)
        for (x, y) in self.walls:
            self.obs[BLOCK, x, y] = 1

        # random water
        self.waters = []
        if self.config.nb_water[0] < self.config.nb_water[1]:
            nb_water = np.random.randint(
                self.config.nb_water[0], self.config.nb_water[1])
            nb_water = 0
            pool = np.random.permutation(self.empty_list)
            count = 0
            for (x, y) in pool:
                if count == nb_water:
                    break
                self.empty_list.remove((x, y))
                self.item_map[x, y] = 2
                if self._check_block(self.empty_list):  # success
                    self.waters.append((x, y))
                    # water
                    self.obs[WATER, x, y] = 1
                    count += 1
                else:
                    self.empty_list.append((x, y))
                    self.item_map[x, y] = -1
            if count != nb_water:
                map_str = ''
                for i in range(10):
                    for j in range(10):
                        if self.item_map[i, j] == 1:
                            map_str = map_str +'X'
                        elif self.item_map[i, j] == 2:
                            map_str = map_str +'W'
                        else:
                            map_str = map_str +'_'
                    map_str = map_str +'\n'
                print(map_str)
                raise RuntimeError('Cannot generate a map without\
                    inaccessible regions! Decrease the #waters or #blocks')

    def _add_targets(self):
        # reset
        self.object_list = []

        # create objects
        # 1. create required objects
        required_objs = []
        pool = np.random.permutation(self.empty_list)
        for tind in range(self.nb_subtask):
            # make sure each subtask is executable
            subid = self.subtask_id_list[tind]
            _, _, _, [oid] = self.config.subtask_param_list[subid] 
            self._add_object(oid, required_objs)
        for obj_idx in range(len(required_objs)):
            self._add_item(required_objs[obj_idx], (pool[obj_idx][0], pool[obj_idx][1]))
            self.empty_list.remove((pool[obj_idx][0], pool[obj_idx][1]))
        # 2. create additional objects
        index = len(required_objs)
        for obj_param in self.config.object_param_list:
            oid = obj_param['oid']
            if 'max' in obj_param and oid in required_objs:
                #nb_obj = np.random.randint(0, obj_param['max']+1)
                nb_obj = obj_param['max']
                for i in range(nb_obj):
                    self._add_item(oid, (pool[index][0], pool[index][1]))
                    self.empty_list.remove((pool[index][0], pool[index][1]))
                    index += 1

        # create agent
        (self.agent_init_pos_x, self.agent_init_pos_y) = pool[index]
        self.agent_x = self.agent_init_pos_x
        self.agent_y = self.agent_init_pos_y

        self.obs[AGENT, self.agent_x, self.agent_y] = 1

    def _add_object(self, oid, required_objs):
        if oid in required_objs:
            return
        obj_pickable = self.config.object_param_list[oid]['pickable']
        obj_transformable = self.config.object_param_list[oid]['transformable']
        if not obj_pickable and not obj_transformable:
            if oid not in required_objs:
                required_objs.append(oid)
        else:
            if obj_pickable:
                if oid not in required_objs:
                    required_objs.append(oid)
            subid = self.config.subtask_output_to_id[oid]
            _, materials, tools, output = self.config.subtask_param_list[subid]
            assert(oid in output)
            for required_obj in (materials+tools):
                self._add_object(required_obj, required_objs)
            
    def _check_block(self, empty_list):
        nb_empty = len(empty_list)
        mask = np.copy(self.item_map)
        #
        queue = deque([empty_list[0]])
        x, y = empty_list[0]
        mask[x, y] = 1
        count = 0
        while len(queue) > 0:
            [x, y] = queue.popleft()
            count += 1
            candidate = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            for item in candidate:
                if mask[item[0], item[1]] == -1:  # if empty
                    mask[item[0], item[1]] = 1
                    queue.append(item)
        return count == nb_empty

    def _get_cur_item(self):
        return self.item_map[self.agent_x, self.agent_y]
    # map

    # rendering
    def _init_pygame(self):
        self.title_height = 30
        pygame.init()
        pygame.freetype.init()

    def _init_screen(self, graph_img, text_widths, num_lines):
        obs_w = self.w*self.render_scale
        obs_h = self.h*self.render_scale + 30
        graph_w, graph_h = graph_img.get_size()

        if self.cheatsheet:
            list_w = sum(text_widths)*CHR_WIDTH + TABLE_ICON_SIZE\
                    + MARGIN*(len(text_widths)-1)
            list_h = num_lines*TABLE_ICON_SIZE+10
        else:
            list_w, list_h = 0, 0
        size = [obs_w+graph_w+list_w+45+LEGEND_WIDTH,
                max(obs_h, list_h, graph_h)+self.title_height+10]

        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("  ")

    def _load_game_asset(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        object_param_list = self.config.object_param_list
        self.object_image_list, self.obj_img_plt_list = [], []
        img_folder = os.path.join(ROOT_DIR, 'asset', self.gamename, 'Icon')
        for obj in object_param_list:
            image = pygame.image.load(os.path.join(img_folder, obj['imgname']))
            self.object_image_list.append(image)
            image = plt.imread(os.path.join(img_folder, obj['imgname']))
            self.obj_img_plt_list.append(image)
        self.agent_img = pygame.image.load(
            os.path.join(img_folder, 'agent.png'))
        if self.gamename == 'mining':
            self.block_img = pygame.image.load(
                os.path.join(img_folder, 'mountain.png'))
            self.water_img = pygame.image.load(
                os.path.join(img_folder, 'water.png'))
        else:
            self.block_img = pygame.image.load(
                os.path.join(img_folder, 'block.png'))

    def render(self, step_count, text_lines, text_widths, status, bg_colors):
        if not self._rendering:
            return
        pygame.event.pump()
        GAME_FONT = pygame.freetype.SysFont('Arial', 20)
        STAT_FONT = pygame.freetype.SysFont('Arial', 24)
        TITLE_FONT = pygame.freetype.SysFont('Arial', 30)
        graph_img = pygame.image.load(os.path.join(
            __PATH__, '../render/temp/subtask_graph.png'))

        if not self.init_screen_flag:
            self._init_screen(graph_img, text_widths, len(text_lines))
            self.init_screen_flag = True
            self.arrow_img = pygame.image.load(os.path.join(
                __PATH__, 'asset/arrow.png'))

        self.screen.fill(WHITE)
        size = [self.render_scale, self.render_scale]
        w_bias, h_bias = 0, 0
        tbias = self.title_height

        # 1. Observation
        # items
        for i in range(len(self.object_list)):
            obj = self.object_list[i]
            oid = obj['oid']
            obj_img = self.object_image_list[oid]
            if obj_img.get_width() != self.render_scale:
                obj_img = pygame.transform.scale(obj_img, size)
            self.screen.blit(
                obj_img, (obj['pos'][0]*self.render_scale, tbias+obj['pos'][1]*self.render_scale))
        # walls
        if self.block_img.get_width() != self.render_scale:
            self.block_img = pygame.transform.scale(self.block_img, size)
        for wall_pos in self.walls:
            pos = [self.render_scale * wall_pos[0],
                   tbias+self.render_scale * wall_pos[1]]
            self.screen.blit(self.block_img, pos)
        # waters
        if self.gamename == 'mining':
            if self.water_img.get_width() != self.render_scale:
                self.water_img = pygame.transform.scale(self.water_img, size)
            for water_pos in self.waters:
                pos = [self.render_scale * water_pos[0],
                       tbias+self.render_scale * water_pos[1]]
                self.screen.blit(self.water_img, pos)
        # agent
        if self.agent_img.get_width() != self.render_scale:
            self.agent_img = pygame.transform.scale(self.agent_img, size)
        pos = (self.render_scale * self.agent_x,
               tbias+self.render_scale * self.agent_y)
        self.screen.blit(self.agent_img, pos)
        # grid
        for x in range(self.w+1):
            pygame.draw.line(self.screen, DARK, [x*self.render_scale, tbias],
                             [x*self.render_scale, tbias+self.h*self.render_scale], 3)
        for y in range(self.h+1):
            pygame.draw.line(self.screen, DARK, [0, tbias+y*self.render_scale],
                             [self.w*self.render_scale, tbias+y*self.render_scale], 3)

        title_x = round(self.w*self.render_scale/2)-80
        TITLE_FONT.render_to(self.screen, (title_x, 0),
                             'Observation', (0, 0, 0))

        # 2. Graph
        w_bias = self.w*self.render_scale
        self.screen.blit(graph_img, [w_bias+25, tbias])
        title_x = round(w_bias + graph_img.get_width()/2)-80
        TITLE_FONT.render_to(self.screen, (title_x, 0),
                             'Subtask graph', (0, 0, 0))

        # 2-1. Legend 1
        w_bias = self.w*self.render_scale + graph_img.get_width()+50
        h_bias = y+tbias
        title_x = w_bias + 45
        TITLE_FONT.render_to(self.screen, (title_x, 0), 'Legend')
        #
        W, H, h_gap = 120, 26, 35
        x = w_bias
        GAME_FONT.render_to(self.screen, (x, h_bias+5), "Subtask:")
        GAME_FONT.render_to(self.screen, (x, h_bias+H+5), "  (OR)")
        GAME_FONT.render_to(self.screen, (x, h_bias+h_gap+H+5), "  AND:")
        GAME_FONT.render_to(self.screen, (x, h_bias+h_gap*2+H+5), "  NOT:")
        x += 80
        # OR
        x += MARGIN
        y = h_bias
        rect_h_min = h_bias-5
        pygame.draw.rect(self.screen, WHITE, (x, y, W, 2*H), 0)
        pygame.draw.rect(self.screen, BLACK, (x, y, W, 2*H), 2)
        GAME_FONT.render_to(self.screen, (x+5, y+5), 'Obj: Name')
        GAME_FONT.render_to(self.screen, (x+17, y+H+5), 'Reward')

        # AND
        h_bias += h_gap + H
        pygame.draw.ellipse(self.screen, LIGHT, (x+30, h_bias, 30, 20), 0)
        pygame.draw.ellipse(self.screen, BLACK, (x+30, h_bias, 30, 20), 2)
        # NOT
        h_bias += h_gap
        self.screen.blit(self.arrow_img, [x, h_bias])
        h_bias += h_gap
        pygame.draw.rect(self.screen, BLACK, (w_bias-MARGIN,
                                              rect_h_min, 225, h_gap*3+H), 2)

        # 2-2. Legend 2
        h_bias += MARGIN
        x = w_bias
        GAME_FONT.render_to(self.screen, (x, h_bias+h_gap+5), "subtask")
        GAME_FONT.render_to(self.screen, (x, h_bias+h_gap*2-5), "status")
        x += 80
        self._add_box_with_label(
            x+MARGIN, h_bias, W, H, 'Eligible', GAME_FONT, WHITE)
        self._add_box_with_label(
            x+MARGIN, h_bias+h_gap, W, H, 'Ineligible', GAME_FONT, LIGHT)
        self._add_box_with_label(x+MARGIN, h_bias+h_gap*2,
                                 W, H, 'Success', GAME_FONT, GREEN)
        self._add_box_with_label(x+MARGIN, h_bias+h_gap*3,
                                 W, H, 'Fail', GAME_FONT, DARK_RED)

        width = 225
        pygame.draw.rect(self.screen, BLACK,
                         (w_bias-MARGIN, h_bias-5, width, h_gap*4), 2)
        w_bias += width + MARGIN
        if self.cheatsheet:
            # 3. Subtask list
            h_bias = MARGIN
            title_x = round(w_bias + sum(text_widths)*CHR_WIDTH/2)-70
            TITLE_FONT.render_to(self.screen, (title_x, 0),
                                 'Subtask list', (0, 0, 0))
            y = h_bias
            for n, line in enumerate(text_lines):
                x = w_bias
                for nn, item in enumerate(line):
                    if nn == 0:
                        GAME_FONT.render_to(
                            self.screen, (x, tbias+y), item, fgcolor=(0, 0, 0), bgcolor=bg_colors[n])
                        # GAME_FONT.render_to(
                        #    self.screen, (x, tbias+y), item, fgcolor=(0, 0, 0))
                        x += text_widths[nn]*CHR_WIDTH
                    elif nn == 1:
                        if n == 0:
                            GAME_FONT.render_to(
                                self.screen, (x, tbias+y), item, fgcolor=(0, 0, 0))
                        else:
                            oid = item
                            obj_img = self.object_image_list[oid]
                            if obj_img.get_width() != self.table_scale:
                                obj_img = pygame.transform.scale(
                                    obj_img, [self.table_scale]*2)
                            self.screen.blit(obj_img, (x, tbias+y - MARGIN))
                        x += TABLE_ICON_SIZE
                    else:
                        GAME_FONT.render_to(
                            self.screen, (x, tbias+y), item, fgcolor=(0, 0, 0))
                        x += text_widths[nn]*CHR_WIDTH
                    x += MARGIN
                y += TABLE_ICON_SIZE
                maxx = x
            pygame.draw.rect(self.screen, BLACK, (w_bias-MARGIN,
                                                  tbias+5, maxx-w_bias, y), 2)

        # 4. Status
        loc = (0, tbias + self.w * self.render_scale + MARGIN)
        STAT_FONT.render_to(self.screen, loc, status, fgcolor=(0, 0, 0))

        # draw
        pygame.display.flip()
        if self.save_flag:
            self._save_image(step_count)

    def _add_box_with_label(self, x, y, W, H, label, font, color):
        pygame.draw.rect(self.screen, color, (x, y, W, H), 0)
        pygame.draw.rect(self.screen, BLACK, (x, y, W, H), 2)
        font.render_to(self.screen, (x+5, y+5), label)

    def _save_image(self, step_count):
        if self._rendering and self.render_dir is not None:
            pygame.image.save(self.screen, self.render_dir +
                              '/render' + '{:02d}'.format(step_count) + '.jpg')
        else:
            raise ValueError(
                '_rendering is False and/or environment has not been reset')
