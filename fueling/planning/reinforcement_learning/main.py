import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hypterparameters
BATCH_SIZE = 256
LR = 0.01                   # learning rate
BUFFER_CAPACITY = 1000000     # capacity of replay buffer, integer!


class ADSEnv(object):
    def __init__(self):
        pass

    def step(self, action):
        """return next observation, reward, done, info"""
        # TODO key input
        # TODO send planning msg (action)
        while not self.is_env_ready():
            time.sleep(0.1)  # second

        # TODO generate env
        # TODO generate reward
        return next observation, reward, done, info

    def received(self, msg):
        """env msg (perception, prediction, chassis, ...)"""
        pass

    def reset(self):
        # TODO define the interface with simulation team
        pass

    def close(self):
        # TODO define the interface with simulation team
        pass


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.replay_buffer = []
        self.position = 0

    def store(self, *args):
        """Saves a transition experience."""
        if len(self.replay_buffer) < self.capacity:
            self.replay_buffer.append(None)
        self.replay_buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.replay_buffer, batch_size)

    def __len__(self):
        return len(self.replay_buffer)


class ActorNetwork(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        pass


class CriticNetwork(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        pass


class DDPG(object):
    def __init__(self):
        self.eval_net, self.target_net = CriticNetwork().to(device), CriticNetwork().to(device)
        self.policy_net = ActorNetwork().to(device)
        self.learn_step_counter = 0     # counter to update target network
        self.replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        """select an action based on the current observation"""
        pass

    def learn(self):
        """update actor and critic network"""
        pass

    def save(self):
        """save model"""
        pass

    def load(self):
        """load model"""
        pass


if __name__ == '__main__':
    # training loop
    env = ADSEnv()
    rl = DDPG()  # initiate the RL framework

    for i_episode in range(1000):
        s = env.reset()
        time_count = 1
        while True:

            a = rl.choose_action(s)

            # select action and get feedback
            s_, r, done, info = env.step(a)

            # store transition experience into replay buffer
            rl.replay_buffer.store(s, a, r, s_, done)

            # learning
            rl.learn()

            # episode terminates, enter next episode
            if done:
                break

            s = s_
            time_count += 1
        if i_episode % 50 == 0:
            print("Episode finished after {} timesteps".format(time_count + 1))
    env.close()
