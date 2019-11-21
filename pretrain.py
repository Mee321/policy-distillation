from itertools import count
from time import time
import gym
import scipy.optimize
from tensorboardX import SummaryWriter

from core.models import *

from torch.autograd import Variable
from torch import Tensor
import torch.tensor as tensor
# from core.agent import AgentCollection
from core.agent_ray import AgentCollection
from utils.utils import *
from core.running_state import ZFilter
# from core.common import estimate_advantages_parallel
from core.common_ray import estimate_advantages_parallel
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
import numpy as np
from torch.distributions.kl import kl_divergence
# from core.natural_gradient import conjugate_gradient_gloabl
from core.natural_gradient_ray import conjugate_gradient_global
from core.policy_gradient import compute_policy_gradient_parallel
from core.log_determinant import compute_log_determinant
# from envs.mujoco.half_cheetah import HalfCheetahVelEnv_FL
import ray
import os
import envs
import pickle

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')


def main(args):
    ray.init(num_cpus=args.num_workers, num_gpus=1)
    dtype = torch.double
    torch.set_default_dtype(dtype)
    for i in range(50):
        env = gym.make(args.env_name)
        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]
        env.seed(args.seed)
        torch.manual_seed(args.seed)
        policy_net = Policy(num_inputs, num_actions, hidden_sizes = (args.hidden_size,) * args.num_layers)
        print("Network structure:")
        for name, param in policy_net.named_parameters():
            print("name: {}, size: {}".format(name, param.size()[0]))
        flat_param = parameters_to_vector(policy_net.parameters())
        matrix_dim = flat_param.size()[0]
        print("number of total parameters: {}".format(matrix_dim))
        value_net = Value(num_inputs)
        batch_size = args.batch_size
        running_state = ZFilter((env.observation_space.shape[0],), clip=5)

        algo = "trpo_pre_train"
        logdir = "./algo_{}/env_{}/batchsize_{}_nworkers_{}_seed_{}".format(algo, str(args.env_name), batch_size, args.agent_count, args.seed)
        writer = SummaryWriter(logdir)

        agents = AgentCollection(env, policy_net, 'cpu', running_state=running_state, render=args.render,
                                 num_agents=args.agent_count, num_parallel_workers=args.num_workers)

        def trpo_loss(advantages, states, actions, params, params_trpo_ls):
            # This is the negative trpo objective
            with torch.no_grad():
                set_flat_params_to(policy_net, params)
                log_prob_prev = policy_net.get_log_prob(states, actions)
                set_flat_params_to(policy_net, params_trpo_ls)
                log_prob_current = policy_net.get_log_prob(states, actions)
                negative_trpo_objs = -advantages * torch.exp(log_prob_current - log_prob_prev)
                negative_trpo_obj = negative_trpo_objs.mean()
                set_flat_params_to(policy_net, params)
            return negative_trpo_obj

        def compute_kl(states, prev_params, xnew):
            with torch.autograd.no_grad():
                set_flat_params_to(policy_net, prev_params)
                pi = policy_net(Variable(states))
                set_flat_params_to(policy_net, xnew)
                pi_new = policy_net(Variable(states))
                set_flat_params_to(policy_net, prev_params)
                kl = torch.mean(kl_divergence(pi, pi_new))
            return kl

        for i_episode in count(1):
            losses = []
            # Sample Trajectories
            print('Episode {}. Sampling trajectories...'.format(i_episode))
            time_begin = time()
            memories, logs = agents.collect_samples(batch_size)
            time_sample = time() - time_begin
            print('Episode {}. Sampling trajectories is done, using time {}.'.format(i_episode, time_sample))

            # Process Trajectories
            print('Episode {}. Processing trajectories...'.format(i_episode))
            time_begin = time()
            advantages_list, returns_list, states_list, actions_list = \
                estimate_advantages_parallel(memories, value_net, args.gamma, args.tau)
            time_process = time() - time_begin
            print('Episode {}. Processing trajectories is done, using time {}'.format(i_episode, time_process))

            # Computing Policy Gradient
            print('Episode {}. Computing policy gradients...'.format(i_episode))
            time_begin = time()
            policy_gradients, value_net_update_params = compute_policy_gradient_parallel(policy_net, value_net, states_list, actions_list, returns_list, advantages_list)
            pg = np.array(policy_gradients).mean(axis=0)
            pg = torch.from_numpy(pg)
            value_net_average_params = np.array(value_net_update_params).mean(axis=0)
            value_net_average_params = torch.from_numpy(value_net_average_params)
            vector_to_parameters(value_net_average_params, value_net.parameters())
            time_pg = time() - time_begin
            print('Episode {}. Computing policy gradients is done, using time {}.'.format(i_episode, time_pg))

            # Computing Conjugate Gradient
            print('Episode {}. Computing the harmonic mean of natural gradient directions...'.format(i_episode))
            fullstep = conjugate_gradient_global(policy_net, states_list, pg,
                                                   args.max_kl, args.cg_damping, args.cg_iter)


            # Linear Search
            print('Episode {}. Linear search...'.format(i_episode))
            time_begin = time()
            prev_params = get_flat_params_from(policy_net)
            for advantages, states, actions in zip(advantages_list, states_list, actions_list):
                losses.append(trpo_loss(advantages, states, actions, prev_params, prev_params).detach().numpy())
            fval = np.array(losses).mean()

            ls_flag = False
            for (n_backtracks, stepfrac) in enumerate(0.5 ** np.arange(10)):
                new_losses = []
                kls = []
                xnew = prev_params + stepfrac * fullstep
                for advantages, states, actions in zip(advantages_list, states_list, actions_list):
                    new_losses.append(trpo_loss(advantages, states, actions, prev_params, xnew).data)
                    kls.append(compute_kl(states, prev_params, xnew).detach().numpy())
                new_loss = np.array(new_losses).mean()
                kl = np.array(kls).mean()
                # print(new_loss - fval, kl)
                if new_loss - fval < 0 and kl < args.max_kl:
                    set_flat_params_to(policy_net, xnew)
                    writer.add_scalar("n_backtracks", n_backtracks, i_episode)
                    ls_flag = True
                    break
            time_ls = time() - time_begin
            if ls_flag:
                print('Episode {}. Linear search is done in {} steps, using time {}'
                      .format(i_episode, n_backtracks, time_ls))
            else:
                print('Episode {}. Linear search is done but failed, using time {}'
                      .format(i_episode, time_ls))

            rewards = [log['avg_reward'] for log in logs]
            average_reward = np.array(rewards).mean()

            if i_episode % args.log_interval == 0:
                print('Episode {}. Average reward {:.2f}'.format(
                    i_episode, average_reward))
                writer.add_scalar("Avg_return", average_reward, i_episode*args.agent_count*batch_size)
            if i_episode > 1000:
                break
        file_path = './pretrained_models/{}_{}'.format(args.env_name, i)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        torch.save({'state_dict': policy_net.state_dict()}, file_path+'_pretrain.pth.tar')
        with open('./pretrained_models/{}_{}.pkl'.format(args.env_name, i), 'wb') as f:
            pickle.dump(env, f, -1)

if __name__ == '__main__':
    import argparse
    # import multiprocessing as mp
    # mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='TRPO with iid Environment')

    # MDP
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--agent-count', type=int, default=10, metavar='N',
                        help='number of agents (default: 100)')
    parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                        help='discount factor (default: 0.995)')
    parser.add_argument('--env-name', default="2DNavigation-v1", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                        help='gae (default: 0.97)')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of hidden layers')

    # Optimization
    parser.add_argument('--max-kl', type=float, default=1e-1, metavar='G',
                        help='max kl value (default: 1e-1)')
    parser.add_argument('--cg-damping', type=float, default=1e-2, metavar='G',
                        help='damping for conjugate gradient (default: 1e-2)')
    parser.add_argument('--cg-iter', type=int, default=10, metavar='G',
                        help='maximum iteration of conjugate gradient (default: 1e-1)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization parameter for critics (default: 1e-3)')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='per-iteration batch size for each agent (default: 1000)')

    # Miscellaneous
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda)')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='number of workers for parallel computing')

    args = parser.parse_args()

    args.device = torch.device(args.device
                        if torch.cuda.is_available() else 'cpu')
    main(args)
