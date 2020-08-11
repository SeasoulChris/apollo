import numpy as np
import torch

from fueling.planning.reinforcement_learning.environment import ADSEnv
from fueling.planning.reinforcement_learning.rl_algorithm import DDPG

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hypterparameters
BATCH_SIZE = 256
LR = 0.01                   # learning rate
BUFFER_CAPACITY = 1000000     # capacity of replay buffer, integer!
TARGET_REPLACE_ITER = 100   # regulate frequency to update the target network


class OUNoise(object):
    """the Ornstein-Uhlenbeck process"""

    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.3,
                 min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.low = 0
        self.high = 1
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.randn(self.action_dim[0], self.action_dim[1])
        self.state = x + dx
        return self.state

    def choose_action(self, action, t=0):
        """Adding time-correlated noise to the actions taken by the deterministic policy"""
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * \
            min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


def main():
    # training loop
    history_len = 10
    pred_horizon = 10
    hidden_size = 128

    env = ADSEnv(history_len=history_len, hidden_size=hidden_size)
    ou_noise = OUNoise((10, 4))   # the dim of the action is 10*4
    # initiate the RL framework
    rl = DDPG(history_len, pred_horizon, hidden_size=hidden_size)

    for i_episode in range(1000):
        state, hidden = env.reset()
        time_count = 1
        ou_noise.reset()
        while True:

            action, next_hidden = rl.choose_action(state, hidden)
            action = ou_noise.choose_action(action, time_count)

            # select action and get feedback
            next_state, reward, done, info = env.step(action)

            # store transition experience into replay buffer
            rl.replay_buffer.store(state, hidden, action,
                                   reward, next_state, next_hidden, done)

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
    cyber.shutdown()
    env.close()


if __name__ == '__main__':
    main()
