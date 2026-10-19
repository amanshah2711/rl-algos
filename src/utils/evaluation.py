import numpy as np
def mc_evaluation(agent, batch_size, discount_factor):
    episodes = collect_episodes(agent, agent.env, batch_size)
    rewards = [episode['rewards'] for episode in episodes]
    mc_estimate = np.mean([discounted_cumsum(reward, discount_factor) for reward in rewards])
    return mc_estimate


def discounted_cumsum(rewards, discount_factor):
    total = 0
    for reward in rewards[::-1]:
        total += reward + discount_factor * total
    return total

def collect_episodes(agent, env, batch_size):
    episodes = []
    for _ in range(batch_size):
        observation, info = env.reset()
        episode_over = False
        episode = {'observations':[], 'rewards':[], 'next_observations':[], 'terminated':[], 'truncated':[]}
        while not episode_over:
            action = agent.policy(observation)
            episode['observations'].append(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode['rewards'].append(reward)
            episode['next_observations'].append(observation)
            episode['terminated'].append(terminated)
            episode['truncated'].append(truncated)
            episode_over = episode_over or terminated
        episodes.append(episode)
    return episodes

