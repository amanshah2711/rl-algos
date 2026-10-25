
import gymnasium as gym
from ..algorithms.exact_value_iteration import ExactValueIteration


env = gym.make('FrozenLake-v1', success_rate = 1.0, map_name='8x8', render_mode='human')

observation, info = env.reset()

episode_over = False
total_reward = 0

agent = ExactValueIteration(env, discount_factor=0.5)
agent.learn()

while not episode_over:
    action = agent.policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated


print(env.unwrapped.P)
print(f"Episode finished! Total reward: {total_reward}")
print(agent.values)
print(agent.q)

env.close()