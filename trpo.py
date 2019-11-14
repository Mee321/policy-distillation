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

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')


def trpo(env, args):
    dtype = torch.double
    torch.set_default_dtype(dtype)
    # env = gym.make(args.env_name)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    policy_net = Policy(num_inputs, num_actions, hidden_sizes = (args.hidden_size,) * args.num_layers)
    # print("Network structure:")
    # for name, param in policy_net.named_parameters():
    #     print("name: {}, size: {}".format(name, param.size()[0]))
    # flat_param = parameters_to_vector(policy_net.parameters())
    # matrix_dim = flat_param.size()[0]
    # print("number of total parameters: {}".format(matrix_dim))
    value_net = Value(num_inputs)
    batch_size = args.teacher_batch_size
    running_state = ZFilter((env.observation_space.shape[0],), clip=5)
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
        # print('Episode {}. Sampling trajectories...'.format(i_episode))
        # time_begin = time()
        memories, logs = agents.collect_samples(batch_size)
        # time_sample = time() - time_begin
        # print('Episode {}. Sampling trajectories is done, using time {}.'.format(i_episode, time_sample))

        # Process Trajectories
        # print('Episode {}. Processing trajectories...'.format(i_episode))
        # time_begin = time()
        advantages_list, returns_list, states_list, actions_list = \
            estimate_advantages_parallel(memories, value_net, args.gamma, args.tau)
        # time_process = time() - time_begin
        # print('Episode {}. Processing trajectories is done, using time {}'.format(i_episode, time_process))

        # Computing Policy Gradient
        # print('Episode {}. Computing policy gradients...'.format(i_episode))
        # time_begin = time()
        policy_gradients, value_net_update_params = compute_policy_gradient_parallel(policy_net, value_net, states_list, actions_list, returns_list, advantages_list)
        pg = np.array(policy_gradients).mean(axis=0)
        pg = torch.from_numpy(pg)
        value_net_average_params = np.array(value_net_update_params).mean(axis=0)
        value_net_average_params = torch.from_numpy(value_net_average_params)
        vector_to_parameters(value_net_average_params, value_net.parameters())
        # time_pg = time() - time_begin
        # print('Episode {}. Computing policy gradients is done, using time {}.'.format(i_episode, time_pg))

        # Computing Conjugate Gradient
        # print('Episode {}. Computing the harmonic mean of natural gradient directions...'.format(i_episode))
        fullstep = conjugate_gradient_global(policy_net, states_list, pg,
                                               args.max_kl, args.cg_damping, args.cg_iter)


        # Linear Search
        # print('Episode {}. Linear search...'.format(i_episode))
        # time_begin = time()
        prev_params = get_flat_params_from(policy_net)
        for advantages, states, actions in zip(advantages_list, states_list, actions_list):
            losses.append(trpo_loss(advantages, states, actions, prev_params, prev_params).detach().numpy())
        fval = np.array(losses).mean()

        # ls_flag = False
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
                # writer.add_scalar("n_backtracks", n_backtracks, i_episode)
                ls_flag = True
                break
        # time_ls = time() - time_begin
        # if ls_flag:
        #     print('Episode {}. Linear search is done in {} steps, using time {}'
        #           .format(i_episode, n_backtracks, time_ls))
        # else:
        #     print('Episode {}. Linear search is done but failed, using time {}'
        #           .format(i_episode, time_ls))
        rewards = [log['avg_reward'] for log in logs]
        average_reward = np.array(rewards).mean()

        if i_episode % args.log_interval == 0:
            print('Episode {}. Average reward {:.2f}'.format(
                i_episode, average_reward))
        if i_episode > args.num_teacher_episodes:
            break

    return policy_net
