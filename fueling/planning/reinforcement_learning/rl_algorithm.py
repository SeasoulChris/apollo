import copy

import torch


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
        hidden = torch.FloatTensor(hidden).unsqueeze(0).to(device)
        action, next_hidden, _ = self.rl_net.forward(
            state, rl=True, hidden=hidden)
        return action.detach().cpu().numpy()[0], next_hidden.detach().cpu().numpy()[0]

    def learn(self, gamma=0.99, min_value=-np.inf, max_value=np.inf):
        """update actor and critic network"""
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        state, hidden, action, reward, next_state, next_hidden, done = \
            self.replay_buffer.sample(BATCH_SIZE)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        hidden = torch.FloatTensor(hidden).to(device)
        next_hidden = torch.FloatTensor(next_hidden).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        pred_traj, _, _ = self.rl_net(state, rl=True, hidden=hidden)
        _, _, policy_loss = self.rl_net(
            state, rl=True, action=pred_traj[:, 0, :])
        policy_loss = -policy_loss.mean()

        next_pred_traj, _, _ = self.target_net(
            next_state, rl=True, hidden=next_hidden)
        next_action = next_pred_traj[:, 0, :]
        _, _, target_value = self.target_net(
            next_state, rl=True, action=next_action.detach())
        expected_value = reward + (1.0 - done) * gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)

        _, _, value = self.rl_net(state, rl=True, action=action[:, 0, :])
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

    def load(self):
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
        hidden = torch.FloatTensor(hidden).unsqueeze(0).to(device)
        action, next_hidden, _ = self.rl_net1.forward(
            state, rl=True, hidden=hidden)
        return action.detach().cpu().numpy()[0], next_hidden.detach().cpu().numpy()[0]

    def learn(self, gamma=0.99, soft_tau=0.005, noise_std=0.2, noise_clip=0.5, policy_update=2):
        """update actor and critic network"""
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        state, hidden, action, reward, next_state, next_hidden, done = \
            self.replay_buffer.sample(BATCH_SIZE)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        hidden = torch.FloatTensor(hidden).to(device)
        next_hidden = torch.FloatTensor(next_hidden).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        # target policy smoothing
        next_pred_traj, _, _ = self.target_net1(
            next_state, rl=True, hidden=next_hidden)
        next_action = next_pred_traj[:, 0, :]
        noise = torch.normal(torch.zeros(
            next_action.size()), noise_std).to(device)
        noise = torch.clamp(noise, -noise_clip, noise_clip)
        next_action += noise

        # clipped double-Q learning
        _, _, target_value1 = self.target_net1(
            next_state, rl=True, action=next_action.detach())
        _, _, target_value2 = self.target_net2(
            next_state, rl=True, action=next_action.detach())
        target_value = torch.min(target_value1, target_value2)
        expected_value = reward + (1.0 - done) * gamma * target_value

        _, _, value1 = self.rl_net1(state, rl=True, action=action[:, 0, :])
        value_loss1 = self.value_loss_func(value1, expected_value.detach())
        _, _, value2 = self.rl_net1(state, rl=True, action=action[:, 0, :])
        value_loss2 = self.value_loss_func(value2, expected_value.detach())

        self.optimizer1.zero_grad()
        value_loss1.backward()
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        value_loss2.backward()
        self.optimizer2.step()

        # delayed update
        if self.learn_step_counter % policy_update == 0:
            pred_traj, _, _ = self.rl_net1(state, rl=True, hidden=hidden)
            _, _, policy_loss = self.rl_net1(
                state, rl=True, action=pred_traj[:, 0, :])
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

    def load(self):
        """load model"""
        self.rl_net1.load_state_dict(torch.load(filename + "_net1.pt"))
        self.optimizer1.load_state_dict(
            torch.load(filename + "_optimizer1.pt"))
        self.target_net1 = copy.deepcopy(self.rl_net1)

        self.rl_net2.load_state_dict(torch.load(filename + "_net2.pt"))
        self.optimizer2.load_state_dict(
            torch.load(filename + "_optimizer2.pt"))
        self.target_net2 = copy.deepcopy(self.rl_net2)
