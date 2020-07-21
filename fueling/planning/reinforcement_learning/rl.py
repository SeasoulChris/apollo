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

    def learn(self):
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
        _, _, policy_loss = self.rl_net(state, rl=True, action=pred_traj[:, 0, :])
        policy_loss = -policy_loss.mean()

        next_pred_traj, _, _ = self.target_net(
            next_state, rl=True, hidden=next_hidden)
        next_action = next_pred_traj[:, 0, :]
        _, _, target_value = self.target_net(next_state, rl=True, action=next_action.detach())
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
        torch.save(self.rl_net.state_dict(), filename + "_net")
        torch.save(self.optimizer.state_dict(), filename + "_optimizer")

    def load(self):
        """load model"""
        self.rl_net.load_state_dict(torch.load(filename + "_net"))
        self.optimizer.load_state_dict(torch.load(filename + "_optimizer"))
        self.target_net = copy.deepcopy(self.rl_net)
