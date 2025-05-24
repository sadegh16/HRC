import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Bernoulli
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
import pickle
from graph_utils import Graph, Node
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from utils import get_shared_data_global, set_shared_data_global, init_worker
import pdb
from multiprocessing import Array
import time
import os
import multiprocessing
import random


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class StructureParams(nn.Module):
    def __init__(self, v_num):
        super(StructureParams, self).__init__()
        self.v_num = v_num
        edge_params = torch.nn.Parameter(torch.zeros((v_num, v_num)))
        self.register_parameter('edge_params', edge_params)

    def L1_regularization(self, update_gs_masks, partial_graph):
        masks = torch.logical_or((1 - update_gs_masks).bool(), partial_graph)
        masked_edge_params = self.edge_params.masked_fill(partial_graph, 0)
        return torch.sum(torch.abs(masked_edge_params))

    def DAG_regularization(self):
        e = torch.eye(self.v_num, device=self.edge_params.device).bool()
        probs = torch.sigmoid(self.edge_params).view(self.v_num, self.v_num).masked_fill(e, 0)
        ldag = torch.sum(torch.cosh(probs * probs.transpose(0, 1)))
        return ldag


class FunctionalNet(nn.Module):
    def __init__(self, in_size_list, aux_size_list=[]):
        super(FunctionalNet, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.v_num = len(in_size_list)
        self.in_size_list = in_size_list
        self.a_num = len(aux_size_list)
        self.aux_size_list = aux_size_list
        self.fs = []
        for i in range(self.v_num):
            i_dim = int(np.array(in_size_list + aux_size_list).sum())
            h_dim = 4 * i_dim
            o_dim = in_size_list[i]
            self.fs.append(nn.Sequential(
                init_(nn.Linear(i_dim, h_dim)),
                nn.LeakyReLU(negative_slope=0.1),
                init_(nn.Linear(h_dim, h_dim)),
                nn.LeakyReLU(negative_slope=0.1),
                init_(nn.Linear(h_dim, o_dim))))
        self.fs = nn.ModuleList(self.fs)

    def forward(self, inputs, masks, aux_inputs=None):
        N = inputs.shape[0]
        out_probs_logits = []
        expand_params = torch.tensor(self.in_size_list, device=inputs.device)
        for i in range(self.v_num):
            expand_masks = torch.repeat_interleave(masks[i], expand_params).bool()
            x = inputs.masked_fill(expand_masks.view(1, -1), 0).float()
            x = torch.cat([x, aux_inputs], dim=1).view(N, int(np.array(
                self.in_size_list + self.aux_size_list).sum())).float()
            y = self.fs[i](x.float())
            out_probs_logits.append(y)
        return out_probs_logits


class ScmBaseline:
    def __init__(self, variable_range_list, s_lr, f_lr, device, aux_range_list, variable_names):
        self.v_num = len(variable_range_list)
        self.variable_range_list = variable_range_list
        self.aux_range_list = aux_range_list
        self.s_params = StructureParams(self.v_num).to(device)
        self.f_nets = FunctionalNet(variable_range_list, aux_range_list).to(device)
        self.device = device

        self.s_optimizer = optim.Adam(self.s_params.parameters(), lr=s_lr)
        self.f_optimizer = optim.Adam(self.f_nets.parameters(), lr=f_lr)
        self.variable_names = variable_names

    def save(self, directory, name, dag):
        np_dag = dag.numpy()
        pickle.dump(dag, open('%s/s_dag_%s.pth' % (directory, name), 'wb'))
        torch.save(self.s_params.state_dict(), '%s/s_params_%s.pth' % (directory, name))
        torch.save(self.f_nets.state_dict(), '%s/f_nets_%s.pth' % (directory, name))

    def load(self, directory, name):
        dag = pickle.load(open('%s/s_dag_%s.pth' % (directory, name), 'rb'))
        self.s_params.load_state_dict(torch.load('%s/s_params_%s.pth' % (directory, name), map_location='cpu'))
        self.f_nets.load_state_dict(torch.load('%s/f_nets_%s.pth' % (directory, name), map_location='cpu'))
        # return torch.from_numpy(np_dag)
        return dag

    def sample_configuration(self):
        configuration = torch.bernoulli(torch.sigmoid(self.s_params.edge_params))
        e = torch.eye(configuration.shape[0], device=configuration.device).bool()
        configuration = configuration.masked_fill(e, 1)
        return configuration.to(self.device)

    def get_DAG(self, causal_threshold, do_variables, names, dag, trained_variables=[]):
        for src_idx in range(self.v_num):
            for dst_idx in range(self.v_num):
                if src_idx not in do_variables+trained_variables or dst_idx not in do_variables+trained_variables:
                    dag[dst_idx, src_idx] = False
        dag_probs = torch.sigmoid(self.s_params.edge_params)
        for src_idx in do_variables:
            for dst_idx in range(self.v_num):
                if dst_idx in do_variables or dst_idx in trained_variables: continue
                if src_idx != dst_idx and dag_probs[dst_idx, src_idx] > causal_threshold:
                    dag[dst_idx, src_idx] = True
        dag[-1, :] = False
        print('DAG:')
        for src_idx in range(self.v_num):
            print(names[src_idx])
            for dst_idx in range(self.v_num):
                if dag[dst_idx, src_idx]:
                    print('------->', names[dst_idx])
        var_depth = 0
        visited_vars = []
        var_depths = torch.zeros(self.v_num).long()
        while (len(visited_vars) < self.v_num):
            for dst_idx in range(self.v_num):
                if dst_idx in visited_vars: continue
                var_depth = 0
                for src_idx in range(self.v_num):
                    if not dag[dst_idx, src_idx]:
                        continue
                    if src_idx in visited_vars:
                        var_depth = max(var_depth, var_depths[src_idx] + 1)
                    else:
                        var_depth = -1
                        break
                if var_depth >= 0:
                    visited_vars.append(dst_idx)
                    var_depths[dst_idx] = var_depth
        return dag, var_depths



    def get_DAG_HAC(self, do_variables, names, dag):
        dag[-1, :] = False
        print('DAG:')
        for src_idx in range(self.v_num):
            print(names[src_idx])
            for dst_idx in range(self.v_num):
                if dag[dst_idx, src_idx]:
                    print('------->', names[dst_idx])
        var_depth = 0
        visited_vars = []
        var_depths = torch.zeros(self.v_num).long()
        while (len(visited_vars) < self.v_num):
            for dst_idx in range(self.v_num):
                if dst_idx in visited_vars: continue
                var_depth = 0
                for src_idx in range(self.v_num):
                    if not dag[dst_idx, src_idx]:
                        continue
                    if src_idx in visited_vars:
                        var_depth = max(var_depth, var_depths[src_idx] + 1)
                    else:
                        var_depth = -1
                        break
                if var_depth >= 0:
                    visited_vars.append(dst_idx)
                    var_depths[dst_idx] = var_depth
        return dag, var_depths





    def compute_logregrets(self, x, y, configuration, do_variable, z=None):
        B = x.shape[0]
        logregrets = []
        input_masks = (1 - configuration).bool()
        if self.f_nets.a_num > 0:
            out_prob_logits = self.f_nets(x.view(B, -1), input_masks, z.view(B, -1))
        else:
            out_prob_logits = self.f_nets(x.view(B, -1), input_masks)
        for i in range(self.v_num):
            v = out_prob_logits[i].view(B, self.variable_range_list[i]).log_softmax(dim=1)
            index = y[:, i].view(B, 1).long()
            v = v.gather(dim=1, index=index).view(B, 1)
            logregrets.append(v)
        logregrets = torch.cat(logregrets, dim=1).view(B, self.v_num)  # M*1*B, l(i): L(Vi)
        vn = logregrets.mean(0).detach()
        vn[do_variable] = 0
        return vn

    def train_f(self, datas, do_variables, args):
        # train function params
        B = args.B
        Fs = args.Fs
        Ns = args.Ns

        x = datas[0]
        y = datas[1]
        z = datas[2]
        N = x.shape[0]
        batch = x.shape[1]
        assert (B * Ns == batch)

        x = torch.cat([F.one_hot(x.view(N * batch, self.v_num)[:, i].long(),
                                 num_classes=self.variable_range_list[i]).view(N * batch, self.variable_range_list[i])
                       for i in range(self.v_num)], dim=1)
        x = x.view(N, batch, -1)
        z = torch.cat([F.one_hot(z.view(N * batch, self.f_nets.a_num)[:, i].long(),
                                 num_classes=self.aux_range_list[i]).view(N * batch, self.aux_range_list[i]) for i in
                       range(self.f_nets.a_num)], dim=1)
        z = z.view(N, batch, -1)

        for f_idx in range(Fs):
            batch_index = list(SubsetRandomSampler(range(B * Ns)))
            configuration = self.sample_configuration().detach()
            # configuration[10, -1] = True
            # configuration[10, 0] = True
            data_x = x[:, batch_index, :].view(N * batch, -1)
            data_y = y[:, batch_index, :].view(N * batch, -1)
            input_masks = (1 - configuration).bool()
            self.f_optimizer.zero_grad()
            out_prob_logits = self.f_nets(data_x, input_masks, z[:, batch_index, :].view(N * batch, -1))
            loss = 0
            loss_vector = [[] for i in range(len(do_variables))]
            for i in range(self.v_num):
                v_loss = F.cross_entropy(out_prob_logits[i].view(N * batch, self.variable_range_list[i]),
                                         data_y[:, i].view(N * batch).long(), reduction='none').view(N * batch)
                loss += v_loss.mean()
            if (f_idx + 1) % 100 == 0:
                print('F', f_idx, 'loss', round(loss.item(), 4))
                # print('wood->stick', configuration[10, 3], loss_vector[do_variables.index(3)][10])
                # print('stone->stonepickaxe', configuration[11, 4], loss_vector[do_variables.index(4)][11])
                # print('F', f_idx, 'loss', round(loss.item(), 4))
            loss.backward()
            self.f_optimizer.step()
            # assert(0)

    def train_s(self, datas, do_variables, args):
        B = args.B
        Ns = args.Ns
        Cs = args.Cs
        Qs = args.Qs

        x = datas[0]
        y = datas[1]
        z = datas[2]
        N = x.shape[0]
        batch = x.shape[1]
        assert (B * Ns == batch)

        x = torch.cat([F.one_hot(x.view(N * batch, self.v_num)[:, i].long(),
                                 num_classes=self.variable_range_list[i]).view(N * batch, self.variable_range_list[i])
                       for i in range(self.v_num)], dim=1)
        x = x.view(N, batch, -1)
        z = torch.cat([F.one_hot(z.view(N * batch, self.f_nets.a_num)[:, i].long(),
                                 num_classes=self.aux_range_list[i]).view(N * batch, self.aux_range_list[i]) for i in
                       range(self.f_nets.a_num)], dim=1)
        z = z.view(N, batch, -1)
        # train structure params
        for q_idx in range(Qs):
            gammagrads = [[] for i in range(len(do_variables))]
            logregrets = [[] for i in range(len(do_variables))]
            batch_indexs = list(SubsetRandomSampler(range(B * Ns)))
            for n_idx in range(Ns):
                gammagrad = [0 for i in range(len(do_variables))]
                logregret = [0 for i in range(len(do_variables))]
                for c_idx in range(Cs):
                    for do_idx in range(len(do_variables)):
                        configuration = self.sample_configuration().detach()
                        batch_index = batch_indexs[B * n_idx:B * n_idx + B]
                        data_x = x[do_idx, batch_index, :].view(B, -1)
                        data_y = y[do_idx, batch_index, :].view(B, self.v_num)
                        logpn = self.compute_logregrets(data_x, data_y,
                                                        configuration, do_variables[do_idx],
                                                        z=z[do_idx, batch_index, :].view(B, -1))
                        with torch.no_grad():
                            logregret[do_idx] += logpn
                            gammagrad[do_idx] += (torch.sigmoid(self.s_params.edge_params) - configuration).view(
                                self.v_num, self.v_num)

                        if c_idx == Cs - 1:
                            gammagrads[do_idx].append(gammagrad[do_idx])
                            logregrets[do_idx].append(logregret[do_idx])
            gammagrads = [torch.stack(gammagrads[do_idx]) for do_idx in range(len(do_variables))]
            normregret = [torch.stack(logregrets[do_idx]).softmax(0) for do_idx in range(len(do_variables))]
            dRdgamma = [torch.einsum("kij,ki->ij", gammagrads[do_idx], normregret[do_idx]) for do_idx in
                        range(len(do_variables))]

            siggamma = self.s_params.edge_params.sigmoid()
            Lmaxent = ((siggamma) * (1 - siggamma)).sum().mul(-args.lmax_coef)
            Lsparse = siggamma.sum().mul(args.l1_coef)

            dRdgamma = torch.stack(dRdgamma).view(len(do_variables), self.v_num, self.v_num).sum(0)
            self.s_params.edge_params.grad = torch.zeros_like(self.s_params.edge_params)
            self.s_params.edge_params.grad.copy_(dRdgamma)
            (Lmaxent + Lsparse).backward()
            self.s_optimizer.step()

            print('Q', q_idx, 'complete')


class ScmLR:

    def __init__(self, n_vars,C=0.01):
        self.n_vars = n_vars
        self.DAG = Graph(nodes=[Node(i) for i in range(self.n_vars)])
        self.lr_models = [LogisticRegression(penalty='l1', solver='saga', C=C, class_weight='balanced', n_jobs=-1,
                                             warm_start=True, )
                          for _ in range(self.n_vars)]

    def prepare_data(self, trajectories):
        data = []
        for trajectory in trajectories:
            data.append([0 for _ in range(2 * self.n_vars)])
            x_1 = [0 for _ in range(2 * self.n_vars)]  # Initialize the row for this trajectory
            data.append(x_1.copy())
            for node_idx in range(len(trajectory)):
                current_entry = trajectory[node_idx]
                current_node = current_entry[0]
                x_1 = data[-1].copy()
                if current_entry[1] == 1:
                    # set y to zero or one based on the x vector
                    data[-1][self.n_vars + current_node.value] = 1
                    if current_entry[2] == "valid":
                        x_1[current_node.value] = 1  # Set the feature for this node to 1
                else:
                    data[-1][self.n_vars + current_node.value] = 0
                data.append(x_1)

        data = np.array(data)
        return data


    def update_DAG(self, regress_node_idx, coef):
        coef = np.array(coef[:self.n_vars - 1])
        remaining_indices = np.array([i for i in range(self.n_vars) if i != regress_node_idx])
        parents = remaining_indices[coef > 2]
        for parent in parents:
            self.DAG.nodes[parent].add_child(self.DAG.nodes[regress_node_idx])
        for idx in range(len(coef)):
            self.DAG.weighted_graph_adjacency[remaining_indices[idx]][regress_node_idx] = coef[idx]

    def update_shared_data(self, data,):
        """
        Update the global shared data used by the pool workers.
        """
        get_shared_data_global()["data"] = data.flatten().tolist()  # Flatten and convert to list for sharing


    def schedule_regression(self, trajectories, pool ):
        print("prepare_data starts...")
        data = self.prepare_data(trajectories)
        print("prepare_data ends...")
        # Update the shared data for this session
        self.update_shared_data(data)
        # Map the train_regression_model function to the pool
        result_async = pool.starmap_async(self.train_regression_model,
                               [(regress_node_idx,data.shape) for regress_node_idx in range(1, self.n_vars)])


        self.DAG.remove_edges()
        # Wait for all tasks to complete and get results
        results = result_async.get()
        for result in results:
            if result:
                self.update_DAG(result[0], result[1])
                self.lr_models[result[0]] = result[2]

    def train_regression_model(self, regress_node_idx,shared_data_shape):
        # print(get_shared_data_global()["data"])
        shared_data = np.array(get_shared_data_global()["data"]).reshape(shared_data_shape)
        y = shared_data[1:, regress_node_idx]
        X = shared_data[:-1, [i for i in range(2 * self.n_vars) if i != regress_node_idx]]
        if np.sum(y == 1) == 0:
            print("**** Not seen:", regress_node_idx)
            return

        mask = (y[:-1] == 1)
        mask = np.insert(mask, 0, False)
        X_filtered = X[~mask]
        y_filtered = y[~mask]
        print("fit starts...")
        model = self.lr_models[regress_node_idx]
        model.fit(X_filtered, y_filtered)
        print("fit ends...")
        return regress_node_idx, model.coef_[0], model


