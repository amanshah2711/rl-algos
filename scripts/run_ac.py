
import gymnasium as gym

from src.agents.ac_agent import ACAgent
from src.policies.pg_policy import PGPolicy
from src.critics.value_critic import DeepValueCritic
from src.utils.environment import collect_episodes
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
    'algorithm' : 'Actor-Critic',
    'train_iters' : 1000,
    'episodes_per_train_iter' : 100,
    'actor_hidden_layer_dim' : 4,
    'actor_num_hidden_layers' : 2,
    'actor_lr' : 1e-3,
    'critic_hidden_layer_dim' : 16,
    'critic_num_hidden_layers' : 2,
    'critic_lr': 1e-3,
    'discount_factor' : 0.99,
    'critic_updates_per_actor_update' : 5,
    'eval_episodes' : 200,
    'seed' : 2711,
    **env_config
}

random.seed(config['seed'])
torch.manual_seed(config['seed'])

# logging setup
writer = SummaryWriter()
writer = SummaryWriter(f'runs/{config['algorithm']}/seed_{config['seed']}')

train_env = gym.make(**env_config)
obs_dim = train_env.observation_space.n
act_dim = train_env.action_space.n
total_env_steps = 0

actor = PGPolicy(obs_dim=obs_dim,
                 hidden_layer_dim=config['actor_hidden_layer_dim'],
                 act_dim=act_dim, 
                 num_hidden_layers=config['actor_num_hidden_layers'],    
                 lr=config['actor_lr'])

critic = DeepValueCritic(obs_dim=obs_dim,
                         hidden_layer_dim=config['critic_hidden_layer_dim'],
                         num_hidden_layers=config['critic_num_hidden_layers'],      
                         discount_factor=config['discount_factor'],
                         lr=config['critic_lr'])

agent = ACAgent(target_policy=actor,
                critic=critic,
                critic_updates_per_actor_update=config['critic_updates_per_actor_update'],
                discount_factor=config['discount_factor'])

for i in range(config['train_iters']):
    print(f'Currently at training iteration {i + 1} / {config['train_iters']}')

    trajectories = collect_episodes(train_env, agent.target_policy.sample_action, config['episodes_per_train_iter'], seed=config['seed'])
    total_env_steps += sum([len(episode.obs) for episode in trajectories])

    metrics = agent.train(trajectories)
    for (key, value) in metrics.items():
        writer.add_scalar(key, value, total_env_steps)




eval_env = gym.make('FrozenLake-v1', is_slippery=False, map_name='4x4')

eval_trajectories = collect_episodes(eval_env, agent.target_policy.get_action, config['eval_episodes'])
success_rate = sum([sum(episode.rewards) for episode in eval_trajectories]) / config['eval_episodes']

grid_length = math.isqrt(obs_dim)
states = torch.arange(0, obs_dim)
with torch.no_grad():
    values = critic(states).reshape((grid_length, grid_length)).numpy()

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
    


