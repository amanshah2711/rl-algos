import numpy as np

def mc_evaluation_starting_state(agent, batch_size, discount_factor): # NOTE: Only for the first state do we get a MC estimate of the value function
    episodes = collect_episodes(agent, agent.env, batch_size)
    rewards = [episode['rewards'] for episode in episodes]
    mc_estimate = np.mean([discounted_cumsum(reward, discount_factor) for reward in rewards])
    return mc_estimate

def mc_policy_evaluation(agent, batch_size, discount_factor):
    episodes = collect_episodes(agent, agent.env, batch_size)
    rewards = [episode['rewards'] for episode in episodes]
    reward_process_samples = [discounted_cumsum(reward, discount_factor) for reward in rewards]

    values = np.zeros(agent.env.observation_space.n)
    counts = np.zeros(agent.env.observation_space.n)

    for state_process_sample, reward_process_sample in zip([episode['observations'] for episode in episodes], reward_process_samples):
        for state, reward_to_go in zip(state_process_sample, reward_process_sample):
            values[state] += reward_to_go
            counts[state] += 1

    for state in range(agent.env.observation_space.n):
        values[state] = values[state] / counts[state] if counts[state] > 0 else 0

    return values

def monte_carlo_q_evaluation(agent, batch_size, discount_factor):
    episodes = collect_episodes(agent, agent.env, batch_size)
    rewards = [episode['rewards'] for episode in episodes]
    reward_process_samples = [discounted_cumsum(reward, discount_factor) for reward in rewards]

    q_values = np.zeros((agent.env.observation_space.n, agent.env.action_space.n))
    counts = np.zeros((agent.env.observation_space.n, agent.env.action_space.n))

    for state_process_sample, action_process, reward_process_sample in zip([episode['observations'] for episode in episodes], [episode['actions'] for episode in episodes], reward_process_samples):
        for state, action, reward_to_go in zip(state_process_sample, action_process, reward_process_sample):
            q_values[state][action] += reward_to_go
            counts[state][action] += 1

    for state in range(agent.env.observation_space.n):
        for action in range(agent.env.action_space.n):
            q_values[state][action] = q_values[state][action] / counts[state][action] if counts[state][action] > 0 else 0

    return q_values

def discounted_cumsum(rewards, discount_factor):
    rewards_to_go = [0.0] * len(rewards)
    total = 0
    for i, reward in enumerate(rewards[::-1]):
        total = reward + discount_factor * total
        rewards_to_go[len(rewards)-i-1] = total
    return rewards_to_go 

def collect_episodes(agent, env, batch_size):
    episodes = []
    for _ in range(batch_size):
        observation, info = env.reset()
        episode_over = False
        episode = {'observations':[], 'rewards':[], 'actions':[], 'next_observations':[], 'terminated':[], 'truncated':[]}
        while not episode_over:
            action = agent.policy(observation)
            episode['observations'].append(observation)
            episode['actions'].append(action)
            observation, reward, terminated, truncated, info = env.step(action)
            episode['rewards'].append(reward)
            episode['next_observations'].append(observation)
            episode['terminated'].append(terminated)
            episode['truncated'].append(truncated)
            episode_over = episode_over or terminated
        episodes.append(episode)
    return episodes

