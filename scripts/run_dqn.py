
from src.agents.dqn_agent import DQNAgent
from src.critics.dqn_critic import DQNCritic
from src.policies.greedy_policy import EpsilonGreedyPolicy

import math


import torch
import random
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import seaborn as sns

env_config = {
    'id' : 'FrozenLake-v1',
    'is_slippery' : False,
    'map_name' : '4x4'
}
config = {
    'algorithm' : 'DQN',
    'train_iters' : 2000,
    'transitions_per_train_iter' : 1000,
    'target_update_freq' : 100,
    'batch_size':500,
    'critic_hidden_layer_dim' : 16,
    'critic_num_hidden_layers': 2,
    'critic_updates_per_batch' : 20,
    'exploration_rate' : 0.2,
    'critic_lr' : 1e-3,
    'discount_factor' : 0.99,
    'eval_episodes' : 200,
    'seed' : 2711,
    **env_config
}
random.seed(config['seed'])
torch.manual_seed(config['seed'])
from datetime import datetime, timezone
run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f'runs/{config['algorithm']}/seed_{config['seed']}/{run_id}')

import gymnasium as gym
from src.utils.environment import collect_transitions, collect_episodes, discrete_state_encoder
train_env = gym.make(**env_config)
obs_dim = train_env.observation_space.n
act_dim = train_env.action_space.n
obs, info = train_env.reset(seed=config['seed'])
total_env_steps = 0

if isinstance(train_env.observation_space, gym.spaces.Discrete):
    state_encoder = lambda obs: discrete_state_encoder(obs, obs_dim)
else:
    state_encoder = lambda x : x

obs = state_encoder(obs)
critic = DQNCritic(
    obs_dim=obs_dim,
    hidden_layer_dim=config['critic_hidden_layer_dim'],
    act_dim=act_dim,
    num_hidden_layers=config['critic_num_hidden_layers'],
    lr=config['critic_lr'],
    discount_factor=config['discount_factor']
)

behavior_policy = EpsilonGreedyPolicy(critic=critic, epsilon=config['exploration_rate'])
target_policy = EpsilonGreedyPolicy(critic=critic, epsilon=0)

agent = DQNAgent(
    critic=critic,
    batch_size=config['batch_size'],
    critic_updates_per_batch=config['critic_updates_per_batch'],
    target_update_freq=config['target_update_freq'],
    behavior_policy = behavior_policy,
    target_policy=target_policy,
    discount_factor=config['discount_factor']
)

for i in range(config['train_iters']):
    print(f'Currently at training iteration {i + 1} / {config['train_iters']}')
    obs, transitions = collect_transitions(train_env, obs, agent.behavior_policy.get_action, num_transitions=config['transitions_per_train_iter'], state_encoder=state_encoder, seed=config['seed'])
    total_env_steps += len(transitions)
    metrics = agent.train(transitions=transitions)
    for (key, value) in metrics.items():
        writer.add_scalar(key, value, total_env_steps)


eval_env = gym.make(**env_config)

eval_trajectories = collect_episodes(eval_env, agent.target_policy.get_action, config['eval_episodes'], state_encoder=state_encoder, seed=config['seed'])
success_rate = sum([sum(episode.rewards) for episode in eval_trajectories]) / config['eval_episodes']

grid_length = math.isqrt(obs_dim)
states = state_encoder(torch.arange(0, obs_dim))
import numpy as np
values = np.max(critic.get_state_action_value(states), axis=-1).reshape((grid_length, grid_length))

fig, ax = plt.subplots()
sns.heatmap(values, annot=True, cmap='viridis', ax=ax)
ax.set_title('Critic State Value Esimates')
writer.add_figure('critic/state_value_estimates', fig, global_step=total_env_steps)

writer.add_hparams(
    hparam_dict=config,
    metric_dict={
        'eval/success_rate': success_rate,
    },

)
writer.close()
train_env.close()
eval_env.close()
    