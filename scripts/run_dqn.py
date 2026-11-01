
from src.agents.dqn_agent import DQNAgent
from src.critics.dqn_critic import DQNCritic

env_config = {
    'id' : 'FrozenLake-v1',
    'is_slippery' : False,
    'map_name' : '4x4'
}
config = {
    'algorithm' : 'Actor-Critic',
    'train_iters' : 1000,
    'episodes_per_train_iter' : 100,
    'discount_factor' : 0.99,
    'eval_episodes' : 200,
    'seed' : 2711,
    **env_config
}

import gymnasium as gym
from src.utils.environment import collect_transitions
train_env = gym.make(**env_config)
obs_dim = train_env.observation_space.n
act_dim = train_env.action_space.n
obs, info = train_env.reset(seed=config['seed'])

critic = DQNCritic(
    obs_dim=obs_dim,
    hidden_layer_dim=1,
    act_dim=act_dim,
    num_hidden_layers=1,
    lr=1,
    gradient_steps_per_batch=1,
    discount_factor=1
)

agent = DQNAgent(
    critic=critic,
    batch_size=1,
    critic_updates_per_batch=1,
    target_update_freq=1,
    discount_factor=1
)

for _ in range(config['train_iters']):
    obs, transitions = collect_transitions(train_env, obs, agent.behavior_policy, num_transitions=, seed=seed)
    agent.train(transitions=transitions)

