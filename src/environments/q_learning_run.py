



import gymnasium as gym
from ..algorithms.q_learning import QLearning 


train_env = gym.make('FrozenLake-v1', is_slippery=False, map_name='4x4')



agent = QLearning(train_env, discount_factor=0.95)
agent.learn()

print('potato')

eval_env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human', map_name='4x4')
observation, info = eval_env.reset()

episode_over = False
total_reward = 0

while not episode_over:
    action = agent.policy(observation)
    observation, reward, terminated, truncated, info = eval_env.step(action)

    total_reward += reward
    episode_over = terminated or truncated


print(agent.q)
print(f"Episode finished! Total reward: {total_reward}")

train_env.close()
eval_env.close()