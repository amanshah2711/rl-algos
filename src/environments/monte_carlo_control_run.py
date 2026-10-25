

import gymnasium as gym
from ..algorithms.monte_carlo_control import MCControl


env = gym.make('FrozenLake-v1', success_rate = 1, render_mode='human', map_name='4x4')



agent = MCControl(env, discount_factor=0.5)
agent.learn()

observation, info = env.reset()

episode_over = False
total_reward = 0

while not episode_over:
    action = agent.policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated



print(env.unwrapped.P)
print(f"Episode finished! Total reward: {total_reward}")

env.close()