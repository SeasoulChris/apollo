# Reinforcement learning

This folder contains all the files for Reinforcement Learning (RL).

## main
The `main.py` includes:
* the Ornstein-Uhlenbeck process to add time-correlated noise to the actions taken by the deterministic policy
* the replay buffer for RL
* the RLNetwork is a two-headed network including both the policy network and the target network
* the main training loop 

## rl_algorithm
The `rl_algorithm.py` has RL algorithms, including:
* DDPG

## environment
The `environment.py` defines an interactive simulator with the openAI gym interface. 
Please put all the code related with our simulator here.

## Demo
The `DDPG_demo.ipynb` is a demo for our first step coding.

The environment in this demo can be found in [Pendulum]<https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py>.