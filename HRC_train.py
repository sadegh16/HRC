import argparse
from deterministic import set_seed
import torch
import torch.nn as nn
import torch.optim as optim
from mpi4py.MPI import COMM_WORLD as comm
from mpi4py import MPI
from torch.utils.data import Dataset, DataLoader
from pdb import set_trace as bp
from HRL import HRL
from SCM import ScmBaseline
from mc import McEnv
from models import EVModel, LinearRegressionModel, MultiTaskLassoL1
from utils import str2bool
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter


# ----------------------------- Utility Functions -----------------------------

def do_sample_parallel(env, hrl, do_variables, data_size, rank, size, sample_step_size):
    # print(rank, 'sample data, do', [env.variable_names[v_idx] for v_idx in do_variables])
    x_datas, y_datas, z_datas, episodes_time = do_sample(env, hrl, do_variables, data_size // size, sample_step_size)
    datas = (x_datas, y_datas, z_datas)
    datas = comm.gather(datas, root=0)
    # Perform the reduction to sum all_episode_time across all ranks
    total_episode_time = comm.reduce(episodes_time, op=MPI.SUM, root=0)
    # intervening_datas = comm.gather(intervening_data, root=0)
    if rank == 0:
        x_datas = torch.cat([datas[i][0] for i in range(size)], dim=1).view(len(do_variables), data_size,
                                                                            env.variable_num)
        y_datas = torch.cat([datas[i][1] for i in range(size)], dim=1).view(len(do_variables), data_size,
                                                                            env.variable_num)
        z_datas = torch.cat([datas[i][2] for i in range(size)], dim=1).view(len(do_variables), data_size,
                                                                            env.aux_info_num)
    # bp()
    return x_datas, y_datas, z_datas, total_episode_time


def sort_do_variables(do_variables, lasso_model, var_num, final_subgoal=12, IS=[], look_ahead=5):
    """
    For each do_variable, check how the environment goes if you continue the env model
    """
    if len(do_variables) == 1:
        return do_variables
    alpha = 1
    final_goal_diff = []
    for do_variable in do_variables:
        X = np.zeros(var_num)
        X[do_variable] = alpha
        for var in IS: X[var] = 1
        do_variable_effect = []
        for trial in range(20):
            for step_idx in range(look_ahead):
                action = torch.randint(0, env.action_space.n, (1,)).item()
                X[-1] = action
                # set do_variable to 1 if it goes to zero
                X[do_variable] = alpha
                next_X = lasso_model.predict(X.reshape(1, -1))[0]
                X = next_X
            do_variable_effect.append(X[final_subgoal])

        final_goal_diff.append(np.mean(do_variable_effect).item())
    print("ring_diff", final_goal_diff)
    # return the sorted list
    return [do_variables[i] for i in sorted(range(len(do_variables)), key=lambda k: final_goal_diff[k], reverse=True)]


def create_dataset(datas, intervening_datas):
    """
    Create pytorch dataset from the current collected data to train LSTM

    """

    class CustomDataset(Dataset):
        def __init__(self, collected_data):
            # combine all do variable datas
            self.x = collected_data[0].reshape(-1, 22).double()

            self.y = collected_data[1].reshape(-1, 22).double()

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    return CustomDataset(datas)


def learn_environment_model(datas, env_model, intervening_datas):
    """
    Learn environment variables
    """
    env_model = env_model.double().to(args.device)
    # create torch dataset to train the model on
    dataset = create_dataset(datas, intervening_datas)
    optimizer = optim.Adam(env_model.parameters())
    loss_fn = nn.MSELoss()
    loader = DataLoader(dataset, shuffle=True, batch_size=128)
    n_epochs = 100
    for epoch in range(n_epochs):
        print("training lstm epoch:", epoch)
        env_model.train()
        for X_batch, y_batch in loader:
            # bp()
            action = X_batch[:, :, -1:].contiguous()
            state = X_batch[:, :, :-1].squeeze(dim=1).unsqueeze(dim=0).contiguous()
            y_pred = env_model(action, state)
            next_state = y_batch[:, :, :-1]
            loss = loss_fn(y_pred, next_state)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def custom_mse_loss_with_l1(y_pred, y_true, model, lambda_l1):
    """
        Custom loss function combining MSE and L1 regularization.
    """
    mse_loss = nn.MSELoss()(y_pred, y_true)
    l1_loss = torch.norm(model.linear.weight, p=1)
    total_loss = mse_loss + lambda_l1 * l1_loss
    # print(mse_loss,lambda_l1 * l1_loss)

    return total_loss


def learn_regression_model(datas, env_model):
    """
    Learn environment variables using linear regression with L0 penalty
    """
    env_model = env_model.double().to(args.device)
    # Create torch dataset to train the model on
    dataset = create_dataset(datas, None)
    optimizer = optim.Adam(env_model.parameters(), lr=0.0001)
    loader = DataLoader(dataset, shuffle=True, batch_size=512)

    n_epochs = 100
    lambda_l1 = 11115  # Adjust this parameter as needed

    for epoch in range(n_epochs):
        print("Training linear regression epoch:", epoch)
        env_model.train()

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            X_batch = X_batch
            y_batch = y_batch

            y_pred = env_model(X_batch)
            loss = custom_mse_loss_with_l1(y_pred, y_batch, env_model, lambda_l1)

            loss.backward()
            optimizer.step()


def learn_regression_sklearn(datas, lasso_multi):
    """
        Train a regression model using sklearn's MultiTaskLasso.
    """
    global coefficients, intercept
    # Use MultiTaskLasso for multi-output regression
    x = datas[0].reshape(-1, 22).double()
    y = datas[1].reshape(-1, 22).double()

    lasso_multi.fit(x, y)
    return lasso_multi.coef_


def create_dag(weights, var_num, do_variables, names, dag, trained_variables, old_var_depths):
    # bp()
    for src_idx in range(var_num):
        for dst_idx in range(var_num):
            if src_idx in do_variables and dst_idx in do_variables:
                pass
            else:
                dag[dst_idx, src_idx] = False

    for src_idx in do_variables:
        for dst_idx in range(var_num):
            if dst_idx in do_variables: continue
            if src_idx != dst_idx and weights[dst_idx, src_idx] > 0.0:
                dag[dst_idx, src_idx] = True
    dag[-1, :] = False

    # modify dag for our setting and set depth
    var_depths = old_var_depths.copy()
    for dst_idx in range(var_num):
        if (dst_idx not in do_variables) and (dst_idx not in trained_variables) and torch.any(dag[dst_idx, :]):
            depth = 0
            for src_idx in do_variables:
                depth = max(depth, var_depths[src_idx] + 1)
                var_depths[dst_idx] = depth
                dag[dst_idx, src_idx] = True
    print('DAG:')
    for src_idx in range(var_num):
        print(names[src_idx])
        for dst_idx in range(var_num):
            if dag[dst_idx, src_idx]:
                print('------->', names[dst_idx])

    return torch.tensor(dag), torch.tensor(var_depths)


def do_sample(env, hrl, do_variables, expect_data_size, sample_step_size):
    episodes_time = 0
    current_goal_index = 0
    action_variable = env.variable_num - 1
    data = [[[], [], []] for i in range(len(do_variables))]
    data_size = [0 for i in range(len(do_variables))]
    complex_state, state = env.reset()
    aux_info = env.info['aux_info']
    while (min(data_size) < expect_data_size):
        operators = []
        for do_idx in range(len(do_variables)):
            do_variable = do_variables[do_idx]
            if do_variable == action_variable or state[do_variable] == 0:
                if data_size[do_idx] < expect_data_size:
                    operators.append(do_idx)
        if set(operators) <= set(
                [do_variables.index(action_variable)]) and len(
            do_variables) > 1:
            complex_state, state = env.reset()
            aux_info = env.info['aux_info']
            continue

        do_variable_idx = current_goal_index
        do_variable = do_variables[do_variable_idx]
        if do_variable != action_variable:
            goal = do_variable
            state, done, goal_achieved, eval_info, complex_state = hrl.evaluate(env, hrl.k_level - 1, state, goal,
                                                                                complex_state)
            # episodes_time+=eval_info[0]
            aux_info = env.env.aux_info
            if not goal_achieved:
                continue
        if len(do_variables) > current_goal_index + 1:
            current_goal_index += 1
        else:
            current_goal_index = 0

        for step_idx in range(sample_step_size):
            action = torch.randint(0, env.action_space.n, (1,)).item()
            next_complex_state, next_state, _, done, info = env.step(action)
            episodes_time += 1
            next_aux_info = info['aux_info']
            state[-1] = action
            next_state[-1] = action
            state_valid = not done and action > 3 and aux_info[0] > 0
            if state_valid and data_size[do_variable_idx] < expect_data_size:
                if (state == next_state).all() and do_variable != env.variable_num - 1: continue
                data[do_variable_idx][0].append(state)
                data[do_variable_idx][1].append(next_state)
                data[do_variable_idx][2].append(aux_info)
                data_size[do_variable_idx] += 1
            if done or next_state[do_variable] != state[do_variable] or len(
                    data[do_variable_idx][0]) >= expect_data_size:
                state = next_state
                complex_state = next_complex_state
                aux_info = next_aux_info
                break
            else:
                state = next_state
                complex_state = next_complex_state
                aux_info = next_aux_info
    for i in range(len(do_variables)):
        assert (len(data[i][0]) == expect_data_size)
    x_datas = [torch.stack(data[i][0]).view(expect_data_size, env.variable_num) for i in range(len(do_variables))]
    y_datas = [torch.stack(data[i][1]).view(expect_data_size, env.variable_num) for i in range(len(do_variables))]
    x_datas = torch.stack(x_datas).view(len(do_variables), expect_data_size, env.variable_num)
    y_datas = torch.stack(y_datas).view(len(do_variables), expect_data_size, env.variable_num)
    z_datas = [torch.stack(data[i][2]).view(expect_data_size, env.aux_info_num) for i in range(len(do_variables))]
    z_datas = torch.stack(z_datas).view(len(do_variables), expect_data_size, env.aux_info_num)
    return x_datas, y_datas, z_datas, episodes_time


def evaluate_and_log(env, all_episode_time, task_log_writer, final_goal):
    #
    goal_achieve_ratio = 0
    complex_state, state = env.reset()
    for test_idx in range(5):
        done = False
        episode_time = 0
        while (not done):
            state, done, _, env_infos, complex_state = hrl.evaluate(env, hrl.k_level - 1, state, final_goal,
                                                                    complex_state)
            sum_times, distance = env_infos
            episode_time += sum_times
        goal_achieve_ratio = goal_achieve_ratio + (1 if env.env.last_game_over else 0)
    task_log_writer.add_scalar('eval_success_ratio_vs_sysprobs', goal_achieve_ratio / 5,
                               all_episode_time)


# ----------------------------- Main Training Logic -----------------------------
def train(env, scm, hrl, args, env_model, scm_model):
    all_episode_time = 0
    var_num = len(env.variable_ranges)
    trained_variables = [var_num - 1, ]
    do_variables = []
    dag = torch.zeros((var_num, var_num)).bool()
    early_stop = 0
    task_log_directory = args.model_path + '/log/rank_' + str(args.rank) + '/'
    task_log_writer = SummaryWriter(task_log_directory)
    final_goal = 12
    selected_do_variables = None
    lasso_multi = MultiTaskLassoL1(alpha=0.0001, warm_start=True, )

    for iter_idx in range(args.scm_model_id, args.I):

        if rank == 0:
            print('iter', iter_idx)
            if args.causal:
                sorted_trained_variables = sort_do_variables(trained_variables, lasso_multi, var_num, IS=do_variables)
                sorted_trained_variables = sorted_trained_variables[:1]
            else:
                if len(trained_variables) > 0:
                    sorted_trained_variables = [np.random.choice(trained_variables, 1, replace=False)[0].item()]
                else:
                    sorted_trained_variables = []

            with open(args.model_path + "/max-score.txt", "a") as myfile:
                myfile.write("{}-{}\n".format(sorted_trained_variables, do_variables))
            print("********sorted_trained_variables:", sorted_trained_variables, )
            if final_goal in do_variables:
                print("be happy boy!")
            elif final_goal in trained_variables:
                do_variables = do_variables + [final_goal]
            else:
                do_variables = do_variables + sorted_trained_variables

            trained_variables = [var for var in trained_variables if var not in do_variables]

            print("do_variables", do_variables)
            if len(do_variables) > 2:
                # selected_do_variables = do_variables[:1] + np.random.choice(do_variables[1:], 2, replace=False).tolist()
                selected_do_variables = do_variables[:1] + do_variables[-1:]
            else:
                selected_do_variables = do_variables

        do_variables = comm.bcast(do_variables, root=0)
        if final_goal not in do_variables:
            selected_do_variables = comm.bcast(selected_do_variables, root=0)
            # Collect data
            print("Collecting data...")
            x_datas, y_datas, z_datas, total_intervene_time = do_sample_parallel(env, hrl, selected_do_variables,
                                                                                 args.Ns * args.B, args.rank,
                                                                                 args.size, args.sample_step_size)
        if rank == 0:
            print("data collected, training started...")
            if final_goal not in do_variables:
                all_episode_time += total_intervene_time

                if args.cda == "reg":
                    weights = learn_regression_sklearn(
                        [x_datas.to(args.device), y_datas.to(args.device), z_datas.to(args.device)],
                        lasso_multi)
                    np.savetxt(f'{args.model_path}/reg_weights_iteration_{iter_idx}.txt', weights,
                               delimiter=',')  # Save as text file
                    dag, var_depths = create_dag(weights, var_num, do_variables, env.variable_names, dag,
                                                 trained_variables, hrl.goal_depth)
                    print("regression finished", sum(dag))


                elif args.cda == "bengio":
                    # train scm on primary process
                    scm.train_f([x_datas.to(args.device), y_datas.to(args.device), z_datas.to(args.device)],
                                selected_do_variables,
                                args)
                    scm.train_s([x_datas.to(args.device), y_datas.to(args.device), z_datas.to(args.device)],
                                selected_do_variables,
                                args)
                    dag_probs = torch.sigmoid(scm.s_params.edge_params).detach().cpu().numpy()
                    np.savetxt(f'{args.model_path}/bengio_weights_iteration_{iter_idx}.txt', dag_probs,
                               delimiter=',')  # Save as text file
                    # update dag
                    dag, var_depths = scm.get_DAG(args.causal_threshold, do_variables, env.variable_names, dag,
                                                  trained_variables=trained_variables)

                    print("Bengio finished")

                cand_variables = []
                for v_idx in range(var_num):
                    if dag[v_idx, :].any() and v_idx not in do_variables:  # and var_depths[v_idx] == train_depth+1:
                        cand_en = True
                        for parent_idx in range(var_num):
                            if dag[v_idx, parent_idx] and parent_idx not in do_variables:
                                cand_en = False
                                break
                        if cand_en:
                            cand_variables.append(v_idx)
                print('var_depths', var_depths)
                print('cand vars', [(env.variable_names[int(var)], var_depths[int(var)]) for var in cand_variables])

                if len(cand_variables) == 0:
                    early_stop += 1
                    if early_stop >= 100:
                        break
                else:
                    early_stop = 0
                    print("start scm saving")
                    scm.save(args.model_path, str(iter_idx), dag)
                    assert (do_variables[0] == var_num - 1)
                    print("start subgoal training")
                    # train discovered variables
                    new_trained_variables, all_episode_time = hrl.train_goals(env, dag, var_depths, cand_variables,
                                                                              args, do_variables, all_episode_time,
                                                                              task_log_writer)
                    print("new trained variables:", new_trained_variables)
                    print("trained variables:", trained_variables)
                    trained_variables += new_trained_variables

                    hrl.save(args.model_path, "HRL_{}".format(str(iter_idx)))
                    print('valid:', hrl.goal_valid)
                    with open(args.model_path + "/hrl-layers.txt", "a") as myfile:
                        myfile.write("{}\n".format(trained_variables))
            else:
                new_trained_variables, all_episode_time = hrl.train_goals(env, dag, var_depths, [final_goal], args,
                                                                          do_variables, all_episode_time,
                                                                          task_log_writer)
                print("new trained variables:", new_trained_variables)
                print("trained variables:", trained_variables)
                trained_variables += new_trained_variables
                ###
                hrl.save(args.model_path, "HRL_{}".format(str(iter_idx)))
                print('valid:', hrl.goal_valid)
                with open(args.model_path + "/hrl-layers.txt", "a") as myfile:
                    myfile.write("{}\n".format(trained_variables))

            hrl_info = hrl.pack_params()
        else:
            hrl_info = None
        hrl_info = comm.bcast(hrl_info, root=0)
        do_variables = comm.bcast(do_variables, root=0)
        if args.rank != 0:
            hrl.unpack_params(hrl_info)


if __name__ == '__main__':

    rank = comm.Get_rank()
    size = comm.Get_size()
    parser = argparse.ArgumentParser(description='cdhrl')
    # parameters for ScmBaseline
    # I, B, Fs, Qs, Ns, Cs, CausalThreshold
    parser.add_argument('--I', type=int, default=100)
    parser.add_argument('--B', type=int, default=32)
    parser.add_argument('--Fs', type=int, default=1000)
    parser.add_argument('--Qs', type=int, default=100)
    parser.add_argument('--Ns', type=int, default=4)
    parser.add_argument('--Cs', type=int, default=25)
    parser.add_argument('--l1_coef', type=float, default=0.05)
    parser.add_argument('--lmax_coef', type=float, default=0.05)
    parser.add_argument('--causal_threshold', type=float, default=0.5)
    parser.add_argument('--f_lr', type=float, default=5e-3)
    parser.add_argument('--s_lr', type=float, default=5e-2)
    parser.add_argument('--scm_model_id', type=int, default=0)
    parser.add_argument('--sample_step_size', type=int, default=50)

    # parameters for HRL
    parser.add_argument('--H', type=int, default=15)
    parser.add_argument('--eps', type=float, default=0.95)
    parser.add_argument('--lamda', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--task_gamma', type=float, default=0.95)
    parser.add_argument('--update_steps', type=int, default=3)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--goal_train_steps', type=int, default=1000)
    parser.add_argument('--task_train_steps', type=int, default=100000)
    parser.add_argument('--hrl_lr', type=float, default=0.0001)
    parser.add_argument('--train_threshold', type=float, default=0.5)
    parser.add_argument('--hrl_model_id', type=int, default=0)

    # parameters for causal
    parser.add_argument('--causal', type=str2bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--cda', type=str, default='reg')

    # parameters for logs
    parser.add_argument('--trained_model_path', type=str, default='pretrained_models/')
    parser.add_argument('--trained_model_name', type=str, default='minecraft')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--gpu_num', type=int, default=2)
    parser.add_argument('--load_training_model', type=str2bool, default=False)
    parser.add_argument('--train_task', type=str2bool, default=False)
    parser.add_argument('--task_eval_interval', type=int, default=10)
    parser.add_argument('--norm_scale', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    set_seed(seed=args.seed)

    args.rank = rank
    args.size = size
    if rank == 0:
        log_writer = SummaryWriter(args.model_path + '/log/')
        args.log_writer = log_writer
        print(args)

    env = McEnv(seed=args.seed + args.rank ** 2)
    print(rank, 'init minecraft!')
    action_num = env.action_space.n
    variable_num = len(env.variable_ranges)
    variable_ranges = env.variable_ranges

    hrl = HRL(k_level=1,
              H_list=[args.H],
              state_dim=env.observation_space.shape[0],
              action_dim=action_num,
              goal_dim=variable_num * len(env.variable_operators),
              lr=args.hrl_lr,
              eps=args.eps,
              goal_ranges=variable_ranges,
              device=args.device,
              lamda=args.lamda,
              gamma_list=[args.gamma],
              task_gamma=args.task_gamma,
              env=env,
              operators=env.variable_operators,
              norm_scale=args.norm_scale,
              hidden_size=128)
    if rank == 0:
        scm = ScmBaseline(variable_ranges, args.s_lr, args.f_lr, args.device, aux_range_list=env.aux_info_ranges,
                          variable_names=env.variable_names)
        env_model = EVModel(num_vars=variable_num)
        scm_model = LinearRegressionModel(variable_num, variable_num)
        print(rank, 'init scm model!')
        train(env, scm, hrl, args, env_model, scm_model)
    else:
        train(env, None, hrl, args, None, None)
