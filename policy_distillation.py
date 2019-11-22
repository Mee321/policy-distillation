from itertools import count
from time import time, strftime, localtime
import gym
import scipy.optimize
from tensorboardX import SummaryWriter
from core.models import *
from core.agent_ray_pd import AgentCollection
from utils.utils import *
import numpy as np
import ray
import envs
from trpo import trpo
from student import Student
from teacher import Teacher
import os
import pickle

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

def train_teachers():
    envs = []
    teacher_policies = []
    time_begin = time()
    print('Training {} teacher policies...'.format(args.num_teachers))
    for i in range(args.num_teachers):
        print('Training no.{} teacher policy...'.format(i + 1))
        env = gym.make(args.env_name)
        envs.append(env)
        teacher_policies.append(trpo(env, args))
    time_pretrain = time() - time_begin
    print('Training teacher is done, using time {}'.format(time_pretrain))
    return envs, teacher_policies

def main(args):
    ray.init(num_cpus=args.num_workers, num_gpus=1)
    # policy and envs for sampling
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    exp_date = strftime('%Y.%m.%d', localtime(time()))
    writer = SummaryWriter(log_dir='./exp_data/{}/{}_{}'.format(exp_date, args.env_name, time()))
    # load saved models if args.load_models
    if args.load_models:
        envs = []
        teacher_policies = []
        dummy_env = gym.make(args.env_name)
        num_inputs = dummy_env.observation_space.shape[0]
        num_actions = dummy_env.action_space.shape[0]
        for i in range(args.num_teachers):
            # load envs
            env_path = './pretrained_models/{}_{}.pkl'.format(args.env_name, i)
            if not os.path.isfile(env_path):
                env_path = './pretrained_models/{}.pkl'.format(args.env_name)
            with open(env_path, 'rb') as input:
                env = pickle.load(input)
            # load policies
            model = Policy(num_inputs, num_actions, hidden_sizes=(args.hidden_size,) * args.num_layers)
            file_path = './pretrained_models/{}_{}_pretrain.pth.tar'.format(args.env_name, i)
            if os.path.isfile(file_path):
                pretrained_model = torch.load(file_path)
            else:
                pretrained_model = torch.load('./pretrained_models/{}_pretrain.pth.tar'.format(args.env_name))
            model.load_state_dict(pretrained_model['state_dict'])
            envs.append(env)
            teacher_policies.append(model)
    else:
        envs, teacher_policies = train_teachers()

    teachers = Teacher(envs, teacher_policies, args)
    student = Student(args)
    print('Training student policy...')
    time_beigin = time()
    # train student policy
    for iter in count(1):
        if iter % args.sample_interval == 1:
            expert_data, expert_reward = teachers.get_expert_sample()
        if args.algo == 'npg':
            loss = student.npg_train(expert_data)
        elif args.algo == 'storm':
            if iter == 1:
                loss, prev_params, prev_grad, direction = student.storm_train(None, None, None, expert_data, iter)
            else:
                loss, prev_params, prev_grad, direction = student.storm_train(prev_params, prev_grad, direction, expert_data, iter)
        else:
            loss = student.train(expert_data)
        writer.add_scalar('{} loss'.format(args.loss_metric), loss.data, iter)
        print('Itr {} {} loss: {:.2f}'.format(iter, args.loss_metric, loss.data))
        if iter % args.test_interval == 0:
            average_reward = student.test()
            writer.add_scalar('Students_average_reward', average_reward, iter)
            writer.add_scalar('teacher_reward', expert_reward, iter)
            print("Students_average_reward: {:.3f} (teacher_reaward:{:3f})".format(average_reward, expert_reward))
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
                        help='max kl value (default: 1e-2)')
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
    parser.add_argument('--loss-metric', type=str, default='kl',
                        help='metric to build student objective')
    parser.add_argument('--algo', type=str, default='sgd',
                        help='update method')
    parser.add_argument('--storm-interval', type=int, default=10, metavar='N',
                        help='frequency of storm (default: 10)')
    parser.add_argument('--init-alpha', type=float, default=1.0, metavar='G',
                        help='storm init alpha (default: 1.0)')
    args = parser.parse_args()

    main(args)
