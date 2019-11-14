from itertools import count
from time import time
import gym
import scipy.optimize
from tensorboardX import SummaryWriter
from core.models import *
from core.agent_ray_pd import AgentCollection
from utils.utils import *
import numpy as np
import random
from torch.distributions.kl import kl_divergence
import ray
import envs
from trpo import trpo
from utils2.math import get_kl, get_wasserstein
import os

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')
dtype = torch.double
torch.set_default_dtype(dtype)

'''
1. train single or multiple teacher policies
2. collect samples from teacher policy
3. use KL or W2 distance as metric to train student policy
4. test student policy
'''
def test(test_env, test_policy, teacher_reward):
    student_agent = AgentCollection([test_env], [test_policy], 'cpu', running_state=None, render=args.render,
                                    num_agents=1, num_parallel_workers=1)
    memories, logs = student_agent.collect_samples(args.testing_batch_size)
    rewards = [log['avg_reward'] for log in logs]
    average_reward = np.array(rewards).mean()
    print("Students_average_reward: {:.3f} (teacher_reaward:{:3f})".format(teacher_reward, average_reward))

def get_expert_sample(agents, policies):
    time_begin = time()
    print('Sampling expert data, batch size is {}...'.format(args.sample_batch_size))
    memories, logs = agents.collect_samples(args.sample_batch_size)
    teacher_rewards = [log['avg_reward'] for log in logs if log is not None]
    teacher_average_reward = np.array(teacher_rewards).mean()
    time_sample = time() - time_begin
    print('Sampling expert data finished, using time {}'.format(time_sample))
    # TODO better implementation of dataset and sampling
    # construct training dataset containing pairs {X:state, Y:output of teacher policy}
    dataset = []
    for memory, policy in zip(memories, policies):
        batch = memory.sample()
        states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
        means = policy.mean_action(states).detach()
        stds = policy.get_std(states).detach()
        dataset += [(state, mean, std) for state, mean, std in zip(states, means, stds)]
    return dataset, teacher_average_reward

def main(args):
    ray.init(num_cpus=args.num_workers, num_gpus=1)
    # policy and envs for sampling
    teacher_policies = []
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # load saved models if args.load_models
    if args.load_models:
        # TODO save and load envs, incorrect for non-iid env
        envs = [gym.make(args.env_name) for _ in range(args.num_teachers)]
        dummy_env = gym.make(args.env_name)
        num_inputs = dummy_env.observation_space.shape[0]
        num_actions = dummy_env.action_space.shape[0]
        for i in range(args.num_teachers):
            model = Policy(num_inputs, num_actions, hidden_sizes=(args.hidden_size,) * args.num_layers)
            file_path = '/pretrained_models/{}_pretrain_{}.pth.tar'.format(args.env_name, i)
            if os.path.isfile(file_path):
                pretrained_model = torch.load(file_path)
            else:
                pretrained_model = torch.load('./pretrained_models/{}_pretrain.pth.tar'.format(args.env_name))
            model.load_state_dict(pretrained_model['state_dict'])
            teacher_policies.append(model)
    else:
        envs = []
        time_begin = time()
        print('Training {} teacher policies...'.format(args.num_teachers))
        for i in range(args.num_teachers):
            print('Training no.{} teacher policy...'.format(i+1))
            env = gym.make(args.env_name)
            envs.append(env)
            teacher_policies.append(trpo(env, args))
        time_pretrain = time() - time_begin
        print('Training teacher is done, using time {}'.format(time_pretrain))

    # agents for sampling
    agents = AgentCollection(envs, teacher_policies, 'cpu', running_state=None, render=args.render,
                             num_agents=args.agent_count, num_parallel_workers=args.num_workers)
    # Create the student policy
    env = gym.make(args.env_name)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    student_policy = Policy(num_inputs, num_actions, hidden_sizes=(args.hidden_size,) * args.num_layers)

    optimizer = torch.optim.Adam(student_policy.parameters())
    print('Training student policy...')
    time_beigin = time()
    # train student policy
    for iter in count(0):
        if iter % args.sample_interval == 0:
            expert_data, expert_reward = get_expert_sample(agents, teacher_policies)
        batch = random.sample(expert_data, args.student_batch_size)
        states = torch.stack([x[0] for x in batch])
        means_teacher = torch.stack([x[1] for x in batch])
        stds_teacher = torch.stack([x[2] for x in batch])
        means_student = student_policy.mean_action(states)
        stds_student = student_policy.get_std(states)
        if args.loss_metric is 'kl':
            loss = get_kl([means_teacher, stds_teacher], [means_student, stds_student])
        elif args.loss_metric is 'w2':
            loss = get_wasserstein([means_teacher, stds_teacher], [means_student, stds_student])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Episode{} {} loss: {:.2f}'.format(iter, args.loss_metric, loss.data))
        if iter % args.test_interval == 0:
            test(env, student_policy, expert_reward)
        if iter > args.num_student_episodes:
            break
    time_train = time() - time_beigin
    print('Training student policy finished, using time {}'.format(time_train))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Policy distillation')
    # Network, env, MDP, seed
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--env-name', default="HalfCheetah-v2", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                        help='discount factor (default: 0.995)')
    parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                        help='gae (default: 0.97)')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--load-models', action='store_true',
                        help='load_pretrained_models')

    # Teacher policy training
    parser.add_argument('--agent-count', type=int, default=10, metavar='N',
                        help='number of agents (default: 100)')
    parser.add_argument('--num-teachers', type=int, default=1, metavar='N',
                        help='number of teacher policies (default: 1)')
    parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                        help='max kl value (default: 1e-1)')
    parser.add_argument('--cg-damping', type=float, default=1e-2, metavar='G',
                        help='damping for conjugate gradient (default: 1e-2)')
    parser.add_argument('--cg-iter', type=int, default=10, metavar='G',
                        help='maximum iteration of conjugate gradient (default: 1e-1)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization parameter for critics (default: 1e-3)')
    parser.add_argument('--teacher-batch-size', type=int, default=1000, metavar='N',
                        help='per-iteration batch size for each agent (default: 1000)')
    parser.add_argument('--sample-batch-size', type=int, default=10000, metavar='N',
                        help='expert batch size for each teacher (default: 10000)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda)')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='number of workers for parallel computing')
    parser.add_argument('--num-teacher-episodes', type=int, default=10, metavar='N',
                        help='num of teacher training episodes (default: 100)')

    # Student policy training
    parser.add_argument('--lr', type=float, default=1e-3, metavar='G',
                        help='adam learnig rate (default: 1e-3)')
    parser.add_argument('--test-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--student-batch-size', type=int, default=1000, metavar='N',
                        help='per-iteration batch size for student (default: 1000)')
    parser.add_argument('--sample-interval', type=int, default=10, metavar='N',
                        help='frequency to update expert data (default: 10)')
    parser.add_argument('--testing-batch-size', type=int, default=10000, metavar='N',
                        help='batch size for testing student policy (default: 10000)')
    parser.add_argument('--num-student-episodes', type=int, default=1000, metavar='N',
                        help='num of teacher training episodes (default: 1000)')
    parser.add_argument('--loss_metric', type=str, default='kl',
                        help='metric to build student objective')
    args = parser.parse_args()

    main(args)
