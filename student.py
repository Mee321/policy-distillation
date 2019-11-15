import gym
from core.models import *
from torch.optim import Adam
import random
from utils2.math import get_wasserstein, get_kl
from core.agent_ray_pd import AgentCollection
import numpy as np

class Student(object):
    def __init__(self, args, optimizer=None):
        self.env = gym.make(args.env_name)
        num_inputs = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        self.training_batch_size = args.student_batch_size
        self.testing_batch_size = args.testing_batch_size
        self.loss_metric = args.loss_metric
        self.policy = Policy(num_inputs, num_actions, hidden_sizes=(args.hidden_size,) * args.num_layers)
        self.agents = AgentCollection([self.env], [self.policy], 'cpu', running_state=None, render=args.render,
                                        num_agents=1, num_parallel_workers=1)
        if not optimizer:
            self.optimizer = Adam(self.policy.parameters(), lr=args.lr)

    def train(self, expert_data):
        batch = random.sample(expert_data, self.training_batch_size)
        states = torch.stack([x[0] for x in batch])
        means_teacher = torch.stack([x[1] for x in batch])
        stds_teacher = torch.stack([x[2] for x in batch])
        means_student = self.policy.mean_action(states)
        stds_student = self.policy.get_std(states)
        if self.loss_metric == 'kl':
            loss = get_kl([means_teacher, stds_teacher], [means_student, stds_student])
        elif self.loss_metric == 'wasserstein':
            loss = get_wasserstein([means_teacher, stds_teacher], [means_student, stds_student])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def test(self):
        memories, logs = self.agents.collect_samples(self.testing_batch_size)
        rewards = [log['avg_reward'] for log in logs]
        average_reward = np.array(rewards).mean()
        return average_reward