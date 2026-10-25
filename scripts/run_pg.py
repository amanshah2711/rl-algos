
import gymnasium as gym

from src.agents.pg_agent import PGAgent
from src.policies.pg_policy import PGPolicy
from src.utils.environment import collect_episodes

n_iters = 1000
episodes_per_iter = 100

train_env = gym.make('FrozenLake-v1', is_slippery=False, map_name='4x4')
agent = PGAgent(target_policy=PGPolicy(train_env.observation_space.n, train_env.action_space.n, learning_rate=1e-3))

for i in range(n_iters):
    print('current iterations is iteration, ', i)
    trajectories = collect_episodes(train_env, agent.target_policy, episodes_per_iter)
    agent.train(trajectories)



eval_env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human', map_name='4x4')
observation, info = eval_env.reset()

episode_over = False
total_reward = 0

while not episode_over:
    action = agent.target_policy.get_action(observation)
    observation, reward, terminated, truncated, info = eval_env.step(action)

    total_reward += reward
    episode_over = terminated or truncated



