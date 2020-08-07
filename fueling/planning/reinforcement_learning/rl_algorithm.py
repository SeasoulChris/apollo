import copy
import random

from torchvision import models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fueling.learning.network_utils import generate_lstm


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hypterparameters
BATCH_SIZE = 256
LR = 0.01                   # learning rate
BUFFER_CAPACITY = 1000000     # capacity of replay buffer, integer!
TARGET_REPLACE_ITER = 100   # regulate frequency to update the target network


class DDPG(object):
    def __init__(self, history_len, pred_horizon):
        self.learn_step_counter = 0     # counter to update target network

        self.rl_net = RLNetwork(history_len, pred_horizon).to(device)
        self.target_net = RLNetwork(history_len, pred_horizon).to(device)

        self.optimizer = torch.optim.Adam(self.rl_net.parameters(), lr=LR)
        self.value_loss_func = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    def choose_action(self, state, hidden):
        """
        select an action based on the current state and hidden
        return action and next_hidden
        """
        state = tuple([torch.FloatTensor(state_element).unsqueeze(
            0).to(device) for state_element in state])
        hidden = tuple([torch.FloatTensor(hidden_element).unsqueeze(
            0).to(device) for hidden_element in hidden])
        action, next_hidden, _ = self.rl_net(
            state, policy=True, hidden=hidden)
        action = action.detach().cpu().numpy()[0]
        next_hidden[0] = next_hidden[0].detach().cpu().numpy()[0]
        next_hidden[1] = next_hidden[1].detach().cpu().numpy()[0]
        return action, next_hidden

    def learn(self, gamma=0.99, min_value=-np.inf, max_value=np.inf):
        """update actor and critic network"""
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        state, hidden, action, reward, next_state, next_hidden, done = \
            self.replay_buffer.sample(BATCH_SIZE)

        state = tuple([torch.FloatTensor(state_element).unsqueeze(
            0).to(device) for state_element in state])
        next_state = tuple([torch.FloatTensor(state_element).unsqueeze(
            0).to(device) for state_element in next_state])
        hidden = tuple([torch.FloatTensor(hidden_element).unsqueeze(
            0).to(device) for hidden_element in hidden])
        next_hidden = tuple([torch.FloatTensor(hidden_element).unsqueeze(
            0).to(device) for hidden_element in next_hidden])
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        pred_traj, _, _ = self.rl_net(state, policy=True, hidden=hidden)
        _, _, policy_loss = self.rl_net(
            state, value=True, action=pred_traj[:, 0, :])
        policy_loss = -policy_loss.mean()

        next_pred_traj, _, _ = self.target_net(
            next_state, policy=True, hidden=next_hidden)
        next_action = next_pred_traj[:, 0, :]
        _, _, target_value = self.target_net(
            next_state, value=True, action=next_action.detach())
        expected_value = reward + (1.0 - done) * gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)

        _, _, value = self.rl_net(state, value=True, action=action[:, 0, :])
        value_loss = self.value_loss_func(value, expected_value.detach())

        self.optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        self.optimizer.step()

        # update target net
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.rl_net.state_dict())
        self.learn_step_counter += 1

    def save(self, filename):
        """save model"""
        torch.save(self.rl_net.state_dict(), filename + "_net.pt")
        torch.save(self.optimizer.state_dict(), filename + "_optimizer.pt")

    def load(self, filename):
        """load model"""
        self.rl_net.load_state_dict(torch.load(filename + "_net.pt"))
        self.optimizer.load_state_dict(torch.load(filename + "_optimizer.pt"))
        self.target_net = copy.deepcopy(self.rl_net)


class TD3(object):
    def __init__(self, history_len, pred_horizon, lr=3e-4):
        self.learn_step_counter = 0     # counter to update target network

        self.rl_net1 = RLNetwork(history_len, pred_horizon).to(device)
        self.rl_net2 = RLNetwork(history_len, pred_horizon).to(device)
        self.target_net1 = RLNetwork(history_len, pred_horizon).to(device)
        self.target_net2 = RLNetwork(history_len, pred_horizon).to(device)

        self.target_net1.load_state_dict(self.rl_net1.state_dict())
        self.target_net2.load_state_dict(self.rl_net2.state_dict())

        self.optimizer1 = torch.optim.Adam(self.rl_net1.parameters(), lr=lr)
        self.optimizer2 = torch.optim.Adam(self.rl_net2.parameters(), lr=lr)
        self.value_loss_func = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    def choose_action(self, state, hidden):
        """
        select an action based on the current state and hidden
        return action and next_hidden
        """
        state = tuple([torch.FloatTensor(state_element).unsqueeze(
            0).to(device) for state_element in state])
        hidden = tuple([torch.FloatTensor(hidden_element).unsqueeze(
            0).to(device) for hidden_element in hidden])
        action, next_hidden, _ = self.rl_net1(
            state, policy=True, hidden=hidden)
        action = action.detach().cpu().numpy()[0]
        next_hidden[0] = next_hidden[0].detach().cpu().numpy()[0]
        next_hidden[1] = next_hidden[1].detach().cpu().numpy()[0]
        return action, next_hidden

    def learn(self, gamma=0.99, soft_tau=0.005, noise_std=0.2, noise_clip=0.5, policy_update=2):
        """update actor and critic network"""
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        state, hidden, action, reward, next_state, next_hidden, done = \
            self.replay_buffer.sample(BATCH_SIZE)

        state = tuple([torch.FloatTensor(state_element).unsqueeze(
            0).to(device) for state_element in state])
        next_state = tuple([torch.FloatTensor(state_element).unsqueeze(
            0).to(device) for state_element in next_state])
        hidden = tuple([torch.FloatTensor(hidden_element).unsqueeze(
            0).to(device) for hidden_element in hidden])
        next_hidden = tuple([torch.FloatTensor(hidden_element).unsqueeze(
            0).to(device) for hidden_element in next_hidden])
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        # target policy smoothing
        next_pred_traj, _, _ = self.target_net1(
            next_state, policy=True, hidden=next_hidden)
        next_action = next_pred_traj[:, 0, :]
        noise = torch.normal(torch.zeros(
            next_action.size()), noise_std).to(device)
        noise = torch.clamp(noise, -noise_clip, noise_clip)
        next_action += noise

        # clipped double-Q learning
        _, _, target_value1 = self.target_net1(
            next_state, value=True, action=next_action.detach())
        _, _, target_value2 = self.target_net2(
            next_state, value=True, action=next_action.detach())
        target_value = torch.min(target_value1, target_value2)
        expected_value = reward + (1.0 - done) * gamma * target_value

        _, _, value1 = self.rl_net1(state, value=True, action=action[:, 0, :])
        value_loss1 = self.value_loss_func(value1, expected_value.detach())
        _, _, value2 = self.rl_net1(state, value=True, action=action[:, 0, :])
        value_loss2 = self.value_loss_func(value2, expected_value.detach())

        self.optimizer1.zero_grad()
        value_loss1.backward()
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        value_loss2.backward()
        self.optimizer2.step()

        # delayed update
        if self.learn_step_counter % policy_update == 0:
            pred_traj, _, _ = self.rl_net1(state, policy=True, hidden=hidden)
            _, _, policy_loss = self.rl_net1(
                state, value=True, action=pred_traj[:, 0, :])
            policy_loss = -policy_loss.mean()

            self.optimizer1.zero_grad()
            policy_loss.backward()
            self.optimizer1.step()

            self.soft_update(self.rl_net1, self.target_net1, soft_tau=soft_tau)
            self.soft_update(self.rl_net2, self.target_net2, soft_tau=soft_tau)

        self.learn_step_counter += 1

    def soft_update(self, net, target_net, soft_tau=1e-2):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

    def save(self, filename):
        """save model"""
        torch.save(self.rl_net1.state_dict(), filename + "_net1.pt")
        torch.save(self.optimizer1.state_dict(), filename + "_optimizer1.pt")

        torch.save(self.rl_net2.state_dict(), filename + "_net2.pt")
        torch.save(self.optimizer2.state_dict(), filename + "_optimizer2.pt")

    def load(self, filename):
        """load model"""
        self.rl_net1.load_state_dict(torch.load(filename + "_net1.pt"))
        self.optimizer1.load_state_dict(
            torch.load(filename + "_optimizer1.pt"))
        self.target_net1 = copy.deepcopy(self.rl_net1)

        self.rl_net2.load_state_dict(torch.load(filename + "_net2.pt"))
        self.optimizer2.load_state_dict(
            torch.load(filename + "_optimizer2.pt"))
        self.target_net2 = copy.deepcopy(self.rl_net2)


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.replay_buffer = []
        self.position = 0

    def store(self, state, hidden, action, reward, next_state, next_hidden, done):
        """Saves a transition experience"""
        if len(self.replay_buffer) < self.capacity:
            self.replay_buffer.append(None)
        self.replay_buffer[self.position] = state + hidden + (action,) + (reward,) + \
            next_state + next_hidden + (done,)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.replay_buffer, batch_size)
        s1, s2, s3, h1, h2, action, reward, s1_, s2_, s3_, h1_, h2_, done = \
            map(np.stack, zip(*batch))
        # state is a tuple of (s1, s2, s3)
        state = (s1, s2, s3)
        hidden = (h1, h2)
        next_state = (s1_, s2_, s3_)
        next_hidden = (h1_, h2_)
        return state, hidden, action, reward, next_state, next_hidden, done

    def __len__(self):
        return len(self.replay_buffer)


class RLNetwork(nn.Module):
    """modified from TrajectoryImitationCNNFCLSTM"""

    def __init__(self, history_len, pred_horizon, embed_size=64,
                 hidden_size=128, cnn_net=models.mobilenet_v2,
                 pretrained=True, num_actions=40):
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
        self.valuenet_fc1_layer = nn.Linear(
            self.cnn_out_size + num_actions, hidden_size)
        self.valuenet_fc2_layer = nn.Linear(hidden_size, hidden_size)
        self.valuenet_fc3_layer = nn.Linear(hidden_size, 1)

    def forward(self, X, policy=False, value=False, hidden=0, action=0):
        """
        This is a two-headed network for both policy and value network.

        When policy = True, it works in the policy network branch.
        When value = True, it works in the value network branch.

        To reduce the computation of imitation learning, the value branch is cut off
        by setting x = None when value = False.
        """
        img_feature, hist_points, hist_points_step = X
        batch_size = img_feature.size(0)
        if policy is True:
            ht, ct = hidden[0], hidden[1]
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
        if value is True:
            action = action.view(batch_size, -1)
            x = torch.cat([img_embedding, action], 1)
            x = F.relu(self.valuenet_fc1_layer(x))
            x = F.relu(self.valuenet_fc2_layer(x))
            x = self.valuenet_fc3_layer(x)
        else:
            x = None

        return pred_traj[:, 1:, :], (ht, ct), x
