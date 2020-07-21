import random
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fueling.learning.network_utils import generate_lstm
from fueling.learning.network_utils import generate_lstm_states
from fueling.planning.reinforcement_learning.environment import *
from fueling.planning.reinforcement_learning.rl import *

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hypterparameters
BATCH_SIZE = 256
LR = 0.01                   # learning rate
BUFFER_CAPACITY = 1000000     # capacity of replay buffer, integer!
TARGET_REPLACE_ITER = 100   # regulate frequency to update the target network


class OUNoise(object):
    """the Ornstein-Uhlenbeck process"""

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3,
                 min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def choose_action(self, action, t=0):
        """Adding time-correlated noise to the actions taken by the deterministic policy"""
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * \
            min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.replay_buffer = []
        self.position = 0

    def store(self, state, hidden, action, reward, next_state, next_hidden, done):
        """Saves a transition experience"""
        if len(self.replay_buffer) < self.capacity:
            self.replay_buffer.append(None)
        self.replay_buffer[self.position] = (state, hidden, action, reward, next_state,
                                             next_hidden, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.replay_buffer, batch_size)
        state, hidden, action, reward, next_state, next_hidden, done = \
            map(np.stack, zip(*batch))
        return state, hidden, action, reward, next_state, next_hidden, done

    def __len__(self):
        return len(self.replay_buffer)


class RLNetwork(nn.Module):
    """modified from TrajectoryImitationCNNFCLSTM"""

    def __init__(self, history_len, pred_horizon, embed_size=64,
                 hidden_size=128, cnn_net=models.mobilenet_v2,
                 pretrained=True, num_actions=4):
        super(RLNetwork, self).__init__()
        self.compression_cnn_layer = nn.Conv2d(12, 3, 3, padding=1)
        self.cnn = cnn_net(pretrained=pretrained)
        self.cnn_out_size = 1000
        for param in self.cnn.parameters():
            param.requires_grad = True

        self.history_len = history_len
        self.pred_horizon = pred_horizon

        self.embedding_fc_layer = torch.nn.Sequential(
            nn.Linear(4, embed_size),
            nn.ReLU(),
        )

        self.h0, self.c0, self.lstm = generate_lstm(embed_size, hidden_size)

        self.output_fc_layer = torch.nn.Sequential(
            nn.Linear(hidden_size + self.cnn_out_size, 4),
        )

        # the following layers belong to the value network branch
        self.valuenet_fc1_layer = nn.Linear(self.cnn_out_size + num_actions, hidden_size)
        self.valuenet_fc2_layer = nn.Linear(hidden_size, hidden_size)
        self.valuenet_fc3_layer = nn.Linear(hidden_size, 1)

    def forward(self, X, rl=False, hidden=0, action=[[0, 0, 0, 0]]):
        img_feature, hist_points, hist_points_step = X
        batch_size = img_feature.size(0)
        if rl:
            h0, c0 = hidden[0], hidden[1]
        else:
            # manually add the unsqueeze before repeat to avoid onnx to tensorRT parsing error
            h0 = self.h0.unsqueeze(0)   # size: 1, hidden_size
            c0 = self.c0.unsqueeze(0)
        ht, ct = h0.repeat(1, batch_size, 1),\
            c0.repeat(1, batch_size, 1)

        img_embedding = self.cnn(
            self.compression_cnn_layer(img_feature)).view(batch_size, -1)
        pred_traj = torch.zeros(
            (batch_size, 1, 4), device=img_feature.device)

        for t in range(1, self.history_len + self.pred_horizon):
            if t < self.history_len:
                cur_pose_step = hist_points_step[:, t, :].float()
                cur_pose = hist_points[:, t, :].float()
            else:
                pred_input = torch.cat(
                    (ht.view(batch_size, -1), img_embedding), 1)
                cur_pose_step = self.output_fc_layer(
                    pred_input).float().clone()
                cur_pose = cur_pose + cur_pose_step
                # dim of the pred_traj: batch, time horizon, states dimension
                # state: (dx, dy, dheading, speed)
                pred_traj = torch.cat(
                    (pred_traj, cur_pose.clone().unsqueeze(1)), dim=1)

            disp_embedding = self.embedding_fc_layer(
                cur_pose_step.clone()).view(batch_size, 1, -1)

            _, (ht, ct) = self.lstm(disp_embedding, (ht, ct))

        # the following calculates the output for value network branch
        # here the action only includes the first point of pred_traj
        x = torch.cat([img_embedding, action], 1)
        x = F.relu(self.valuenet_fc1_layer(x))
        x = F.relu(self.valuenet_fc2_layer(x))
        x = self.valuenet_fc3_layer(x)

        return pred_traj[:, 1:, :], (ht, ct), x


def main():
    # training loop
    env = ADSEnv()
    rl = DDPG(history_len, pred_horizon)  # initiate the RL framework

    for i_episode in range(1000):
        state, hidden = env.reset()
        time_count = 1
        while True:

            action, next_hidden = rl.choose_action(state, hidden)

            # select action and get feedback
            next_state, reward, done, info = env.step(action)

            # store transition experience into replay buffer
            rl.replay_buffer.store(state, hidden, action, reward, next_state, next_hidden, done)

            # learning
            rl.learn()

            # episode terminates, enter next episode
            if done:
                break

            state = next_state
            hidden = next_hidden
            time_count += 1
        if i_episode % 50 == 0:
            print("Episode finished after {} timesteps".format(time_count + 1))
    env.close()


if __name__ == '__main__':
    main()
