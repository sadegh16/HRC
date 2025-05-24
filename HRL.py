import torch
import numpy as np
from DQN import DQN
from utils import ReplayBuffer
import copy
import pickle


class HRL:
    def __init__(self, k_level, H_list, state_dim, action_dim, goal_dim, lr,
                 goal_ranges, device, eps, lamda, gamma_list, task_gamma, env, operators=[0, 1], norm_scale=10, hidden_size=128, final_goal=12):

        self.norm_scale = norm_scale
        self.env=env
        self.hidden_size=hidden_size
        self.final_goal=final_goal
        # adding lowest level
        self.HRL = [DQN(state_dim, action_dim, goal_dim, lr, H=H_list[0], EPISILO=eps, device=device,
                        norm_scale=self.norm_scale, hidden_size=self.hidden_size)]
        self.replay_buffer = [ReplayBuffer()]
        # adding remaining levels
        for i in range(k_level - 1):
            self.HRL.append(DQN(state_dim, goal_dim, goal_dim, lr, H=H_list[i + 1], EPISILO=eps, device=device,
                                norm_scale=self.norm_scale, hidden_size=self.hidden_size))
            self.replay_buffer.append(ReplayBuffer())

        # set some parameters
        self.k_level = k_level
        assert (len(H_list) == k_level)
        self.H_list = H_list
        self.lr = lr
        self.eps = eps
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.device = device
        self.k_update_idx = [0 for i in range(k_level)]

        # set when initial
        self.operators = operators
        self.goal_ranges = goal_ranges

        # set when train goals or train tasks
        self.goal_depth = [0 for i in range(self.goal_dim)]
        self.goal_valid = [False for i in range(self.goal_dim)]

        # set based dag and goal_valid
        self.goal_masks = torch.ones((self.goal_dim, self.goal_dim)).bool().to(device)

        self.goal_exc_times = [0 for i in range(self.goal_dim)]

        self.lamda = lamda
        self.gamma_list = gamma_list
        self.task_gamma = task_gamma

    def set_goal_valid_depth(self, var_depths, H_list, gamma_list):
        max_depth = 1
        for goal_idx in range(self.goal_dim):
            depth = int(var_depths[goal_idx // (len(self.operators))].item())
            if depth >= 1:
                max_depth = max(max_depth, depth)
                self.goal_valid[goal_idx] = True
                self.goal_depth[goal_idx] = depth
        self.set_hierarchy(max_depth, H_list, gamma_list)

    def set_goal_masks(self, dag):
        goal_masks = torch.ones((self.goal_dim, self.goal_dim)).bool()
        for goal_idx in range(self.goal_dim):
            if self.goal_valid[goal_idx]:
                for parent_idx in range(self.goal_dim):
                    if dag[goal_idx // len(self.operators), parent_idx // len(self.operators)]:
                        goal_masks[goal_idx][parent_idx] = not self.goal_valid[parent_idx]
                goal_masks[goal_idx][goal_idx] = False
        self.goal_masks = goal_masks.to(self.device)

    def set_hierarchy(self, k, H_list, gamma_list):
        if k > self.k_level:
            self.H_list = H_list
            self.gamma_list = gamma_list
            for i in range(k - self.k_level):
                self.HRL.append(DQN(self.state_dim, self.goal_dim, self.goal_dim,
                                    lr=self.lr, H=H_list[self.k_level + i], EPISILO=self.eps, device=self.device,
                                    norm_scale=self.norm_scale, hidden_size=self.hidden_size))
                self.replay_buffer.append(ReplayBuffer())
                self.k_update_idx.append(0)
            self.k_level = k

    def check_cant_goal(self, state, goal):
        if goal == self.goal_dim:
            return False
        state_idx = goal // (len(self.operators))
        goal_type = goal % (len(self.operators))
        if goal_type == 0 and state[state_idx] == self.goal_ranges[state_idx] - 1:
            return True
        elif goal_type == 1 and state[state_idx] == 0:
            return True
        else:
            return False

    def check_goal(self, state, next_state, goal, fake=True):
        if goal == self.goal_dim:
            return False
        # assert (self.goal_valid[goal])
        state_idx = goal // (len(self.operators))
        goal_type = goal % (len(self.operators))
        if next_state[state_idx] > state[state_idx] \
                and goal_type == 0:
            return True
        elif next_state[state_idx] < state[state_idx] \
                and goal_type == 1:
            return True
        else:
            return False

    def check_new_action(self, state, next_state, goal_masks, fake):
        new_actions = []
        for goal in range(self.goal_dim):
            if self.goal_valid[goal] and (goal_masks is None or goal_masks[goal] == False):
                if self.check_goal(state, next_state, goal, fake):
                    new_actions.append(goal)
        return new_actions

    def get_action(self, i_level, state, goal, complex_state):
        if goal < self.goal_dim and i_level > 0:
            if self.goal_depth[goal] != i_level + 1:
                return self.get_action(i_level - 1, state, goal, complex_state)

        goal_mask = None
        if goal < self.goal_dim and i_level > 0:
            goal_mask = self.goal_masks[goal]
        elif goal == self.goal_dim:
            goal_mask = torch.logical_not(torch.tensor(self.goal_valid).bool()).to(self.device)
        action = None
        for _ in range(self.H_list[i_level]):
            action_mask = goal_mask
            action, _ = self.HRL[i_level].select_action(complex_state, torch.tensor([goal]),
                                                        mask=action_mask, goal_test=True)
            if i_level > 0:
                action = self.get_action(i_level - 1, state, action, complex_state)

        return action


    def train_goals(self, env, dag, var_depths, cand_variables, args, do_variables, all_episode_time, task_log_writer,):
        max_k = 1
        all_episode_time = all_episode_time
        for goal_idx in range(self.goal_dim):
            depth = int(var_depths[goal_idx].item())
            if int(goal_idx) in cand_variables:
                max_k = max(depth, max_k)
                self.goal_valid[goal_idx] = True
                self.goal_depth[goal_idx] = depth

        self.set_hierarchy(max_k, self.H_list + [5], self.gamma_list + [0.9])
        self.set_goal_masks(dag)
        # train goals in each layer
        suc_ratios = [0 for i in range(self.goal_dim)]
        suc_times = [[] for i in range(self.goal_dim)]
        exc_times = [0 for i in range(self.goal_dim)]
        complex_state, state = env.reset()
        for step in range(1, args.goal_train_steps + 1):
            cand_goals = []
            for cand_goal in cand_variables:
                state_idx = cand_goal
                if state[state_idx] < self.goal_ranges[state_idx] - 1:
                    cand_goals.append(cand_goal)


            # 
            do_goals=[]
            for do_goal in do_variables[1:]:
                state_idx = do_goal
                if state[state_idx] < self.goal_ranges[state_idx] - 1:
                    do_goals.append(do_goal)

            #

            if len(cand_goals) == 0:
                complex_state, state = env.reset()
                continue

            import random
            if self.final_goal not in do_variables:
                if random.random() < 0.1 and len(do_goals)>0:
                    rand_goal = do_goals[torch.randint(0, len(do_goals), (1,)).item()]
                else:
                    rand_goal = cand_goals[torch.randint(0, len(cand_goals), (1,)).item()]
            if self.final_goal in do_variables:
                rand_goal=self.final_goal

            state, done, goal_achieved, env_infos, complex_state = self.run_HRL(env, self.k_level - 1, state, rand_goal, False,
                                                                        complex_state)
            suc_times[rand_goal].append(1 if goal_achieved else 0)
            exc_times[rand_goal] += 1
            self.goal_exc_times[rand_goal] += 1

            ### 
            sum_times, distance = env_infos
            all_episode_time += sum_times
            ##

            rand_goal_name = env.variable_names[rand_goal] + '_incre'
            args.log_writer.add_scalar('goals/' + rand_goal_name, 1 if goal_achieved else 0,
                                       self.goal_exc_times[rand_goal])
            task_log_writer.add_scalar('train_success_ratio_vs_sysprobs/'+ rand_goal_name, 1 if goal_achieved else 0,
                                       all_episode_time)
            if (exc_times[rand_goal] + 1) % 20 == 0:
                suc_ratios[rand_goal] = np.array(suc_times[rand_goal][-100:]).sum() / (
                    min(100, len(suc_times[rand_goal])))
                exc_times[rand_goal] = 0
                print('step', step, rand_goal_name, suc_ratios[rand_goal])
            if step % args.update_steps == 0 and self.replay_buffer[self.k_level - 1].size >= args.batch_size:
                self.update(self.k_level - 1, args.n_iter, args.batch_size, args.log_writer)
            if self.final_goal not in do_variables:
                escape = True
                for cand_goal in cand_variables:
                    if suc_ratios[cand_goal] < 0.5:
                        escape = False
                if escape: break
        trained_variables = []
        for cand_variable in cand_variables:
            cand_variable_idx = cand_variables.index(cand_variable)
            print('cand', env.variable_names[cand_variable], int(var_depths[cand_variable].item()))
            suc_ratio = suc_ratios[cand_variable]
            print(env.variable_names[cand_variable], 'suc_ratio', suc_ratio)
            if suc_ratio > 0.4:
                print('cand', env.variable_names[cand_variable], 'trained')
                trained_variables.append(cand_variable)
            else:
                self.goal_valid[cand_variable] = False
        self.set_goal_masks(dag)
        return trained_variables, all_episode_time

    def evaluate(self, env, i_level, state, goal, complex_state):
        if goal < self.goal_dim and i_level > 0:
            if self.goal_depth[goal] != i_level + 1:
                return self.evaluate(env, i_level - 1, state, goal, complex_state)
        next_state = None
        next_complex_state = None
        done = None
        sum_times = 0
        goal_mask = None
        if goal < self.goal_dim and i_level > 0:
            goal_mask = self.goal_masks[goal]
        elif goal == self.goal_dim:
            goal_mask = torch.logical_not(torch.tensor(self.goal_valid).bool()).to(self.device)
        for _ in range(self.H_list[i_level]):
            action_mask = goal_mask
            action, _ = self.HRL[i_level].select_action(complex_state, torch.tensor([goal]),
                                                        mask=action_mask, goal_test=True)
            if i_level > 0:
                next_state, done, _, env_infos, next_complex_state = self.evaluate(env, i_level - 1, state, action,
                                                                                   complex_state)
                time, distance = env_infos
                sum_times += time
            else:
                next_complex_state, next_state, rew, done, info = env.step(action)
                sum_times += info['step_time']
                distance = info['distance']
            goal_achieved = self.check_goal(state, next_state, goal) and not done

            state = next_state
            complex_state = next_complex_state

            if done or goal_achieved or self.check_cant_goal(state, goal):
                break
        return next_state, done, goal_achieved, (sum_times, distance), next_complex_state

    def run_HRL(self, env, i_level, state, goal, is_subgoal_test, complex_state, log_writer=None, train_top=False):
        if goal < self.goal_dim and i_level > 0:
            if self.goal_depth[goal] != i_level + 1:
                return self.run_HRL(env, i_level - 1, state, goal, is_subgoal_test, complex_state, log_writer)
        if goal == self.goal_dim:
            assert (i_level == self.k_level - 1)
        next_state = None
        next_complex_state = None
        done = None
        goal_transitions = []
        sum_times = 0

        goal_mask = None
        if goal < self.goal_dim and i_level > 0:
            goal_mask = self.goal_masks[goal]
        elif goal == self.goal_dim:
            goal_mask = torch.logical_not(torch.tensor(self.goal_valid).bool()).to(self.device)
        for _ in range(self.H_list[i_level]):
            is_next_subgoal_test = is_subgoal_test
            action_mask = goal_mask
            action, _ = self.HRL[i_level].select_action(complex_state, torch.tensor([goal]), mask=action_mask,
                                                        goal_test=is_subgoal_test)
            if i_level > 0:
                if np.random.random_sample() < self.lamda and goal < self.goal_dim:
                    is_next_subgoal_test = True
                next_state, done, _, env_infos, next_complex_state = self.run_HRL(env, i_level - 1, state, action,
                                                                                  is_next_subgoal_test, complex_state,
                                                                                  log_writer=log_writer)
                time, distance = env_infos
                sum_times += time

                if is_next_subgoal_test and not self.check_goal(state, next_state, action):
                    self.replay_buffer[i_level].add(
                        (complex_state, action, -self.H_list[i_level], next_complex_state, goal, 0.0, float(done)))

            else:
                next_complex_state, next_state, rew, done, info = env.step(action)
                sum_times += info['step_time']
                distance = info['distance']

            goal_achieved = self.check_goal(state, next_state, goal) and not done

            # hindsight action transition
            if goal >= self.goal_dim:
                new_actions = self.check_new_action(state, next_state, goal_mask, fake=False) if not done else []
                if done:
                    if env.env.last_game_over == True:
                        self.replay_buffer[i_level].add(
                            (complex_state, action, 0, next_complex_state, goal, self.task_gamma, float(done)))
                    else:
                        self.replay_buffer[i_level].add((complex_state, action, -self.H_list[i_level],
                                                         next_complex_state, goal, self.task_gamma, float(done)))
                else:
                    for new_action in new_actions:
                        self.replay_buffer[i_level].add(
                            (complex_state, new_action, 0, next_complex_state, goal, self.task_gamma, float(done)))
            else:
                if goal_achieved:
                    if i_level > 0:
                        new_actions = self.check_new_action(state, next_state, goal_mask,
                                                            fake=False) if not done else []
                        for new_action in new_actions:
                            self.replay_buffer[i_level].add(
                                (complex_state, new_action, 0.0, next_complex_state, goal, 0.0, float(done)))
                    else:
                        self.replay_buffer[i_level].add(
                            (complex_state, action, 0.0, next_complex_state, goal, 0.0, float(done)))
                elif i_level > 0:
                    new_actions = self.check_new_action(state, next_state, goal_mask, fake=False) if not done else []
                    for new_action in new_actions:
                        self.replay_buffer[i_level].add((complex_state, new_action, -1.0, next_complex_state, goal,
                                                         self.gamma_list[i_level], float(done)))
                else:
                    self.replay_buffer[i_level].add(
                        (complex_state, action, -1.0, next_complex_state, goal, self.gamma_list[i_level], float(done)))

            # copy for goal transition
            last_transition_valid = 0
            if goal < self.goal_dim:
                if i_level == 0:
                    goal_transitions.append(
                        [complex_state, action, -1.0, next_complex_state, None, self.gamma_list[i_level], float(done)])
                    last_transition_valid = 1
                else:
                    new_actions = self.check_new_action(state, next_state, None, fake=False) if not done else []
                    for new_action in new_actions:
                        goal_transitions.append(
                            [complex_state, new_action, -1.0, next_complex_state, None, self.gamma_list[i_level],
                             float(done)])
                        last_transition_valid += 1
            last_transition_valid = last_transition_valid if not done else 0

            pre_state = state.clone()
            state = next_state
            complex_state = next_complex_state

            if done or goal_achieved or self.check_cant_goal(state, goal):
                break

        if log_writer is not None and goal < self.goal_dim:
            if self.goal_depth[goal] == i_level + 1:
                if goal_achieved or not self.check_cant_goal(state, goal):
                    gc = 1.0 if goal_achieved else 0.0
                    log_writer.add_scalar(
                        'goal_achieved/' + env.variable_names[goal // len(env.variable_operators)] + str(
                            env.variable_operators[goal % len(env.variable_operators)]), gc, self.goal_exc_times[goal])
                    self.goal_exc_times[goal] += 1

        if last_transition_valid > 0:
            for last_idx in range(last_transition_valid):
                goal_transitions[-(1 + last_idx)][2] = 0.0
                goal_transitions[-(1 + last_idx)][5] = 0.0
            achieved_goals = []
            for temp_goal in range(self.goal_dim):
                if (self.goal_depth[temp_goal] == i_level + 1 or i_level == 0) and self.goal_valid[
                    temp_goal] and self.check_goal(pre_state, next_state, temp_goal, fake=False):
                    achieved_goals.append(temp_goal)
            for achieved_goal in achieved_goals:
                for transition in goal_transitions:
                    transition[4] = achieved_goal
                    if i_level == 0 or not self.goal_masks[int(achieved_goal)][int(transition[1])]:
                        self.replay_buffer[i_level].add(tuple(transition))

        return next_state, done, goal_achieved, (sum_times, distance), next_complex_state

    def update(self, k_level, n_iter, batch_size, writer, train_top=False):
        losses = []
        start = 0 if not train_top else k_level
        for i in range(start, k_level + 1):
            if self.replay_buffer[i].size >= batch_size:
                masks = None
                if i > 0 and i < self.k_level - 1:
                    masks = self.goal_masks
                avg_loss = self.HRL[i].update(self.replay_buffer[i], n_iter, batch_size, masks)
                writer.add_scalar('hrl_loss/' + str(i), avg_loss, self.k_update_idx[i])
                self.k_update_idx[i] += 1
                losses.append(round(avg_loss, 4))
        return losses

    def save(self, directory, name):
        hrl_info = [self.k_level, self.H_list, self.gamma_list, self.goal_masks.cpu().numpy(), self.goal_depth,
                    self.goal_valid]
        pickle.dump(hrl_info, open('%s/hrl_%s.info' % (directory, name), 'wb'))
        for i in range(self.k_level):
            self.HRL[i].save(directory, name + '_level_{}'.format(i))

    def load(self, directory, name, H_list=None, gamma_list=None):
        hrl_info = pickle.load(open('%s/hrl_%s.info' % (directory, name), "rb"))
        if len(hrl_info) == 6:
            self.goal_masks = torch.from_numpy(hrl_info[3]).to(self.device)
            self.goal_depth = hrl_info[4]
            self.goal_valid = hrl_info[5]
            self.set_hierarchy(int(hrl_info[0]), hrl_info[1], hrl_info[2])
        else:
            self.goal_masks = torch.from_numpy(hrl_info[1]).to(self.device)
            self.goal_depth = hrl_info[2]
            self.goal_valid = hrl_info[3]
            self.set_hierarchy(int(hrl_info[0]), H_list, gamma_list)
        for i in range(self.k_level):
            self.HRL[i].load(directory, name + '_level_{}'.format(i))

    def pack_params(self):
        hrl_info = [self.k_level, self.H_list, self.gamma_list, self.goal_masks.cpu().numpy(), self.goal_depth,
                    self.goal_valid]
        weights = []
        for i in range(self.k_level):
            weights.append(self.HRL[i].pack_weights())
        return hrl_info + [weights]

    def unpack_params(self, hrl_info):
        k_level = hrl_info[0]
        self.set_hierarchy(k_level, hrl_info[1], hrl_info[2])
        self.goal_masks = torch.from_numpy(hrl_info[3]).to(self.device)
        self.goal_depth = hrl_info[4]
        self.goal_valid = hrl_info[5]
        weights = hrl_info[6]
        for i in range(self.k_level):
            self.HRL[i].unpack_weights(weights[i])





