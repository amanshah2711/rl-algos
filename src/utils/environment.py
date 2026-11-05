from dataclasses import dataclass
import numpy as np
from collections import deque

@dataclass
class Trajectory:
    obses : list[int]
    acts : list[int]
    rewards : list[int]
    next_obses : list[int]
    terminals : list[bool]

class ReplayBuffer:

    def __init__(self, buffer_size, state_encoder):
        self.buffer = deque([])
        self.buffer_size = buffer_size 
        self.obses = np.empty(shape=(buffer_size, ), dtype=np.int64)
        self.acts = np.empty(shape=(buffer_size, ))
        self.rewards = np.empty(shape=(buffer_size, ))
        self.next_obses = np.empty(shape=(buffer_size, ), dtype=np.int64)
        self.terminals = np.empty(shape=(buffer_size, ))
        self.state_encoder = state_encoder
        self.end = 0

    def add_to_buffer(self, transitions):
        for i, (obs, act, reward, next_obs, terminal)  in enumerate(transitions):
            self.obses[(self.end + i) % self.buffer_size] = obs
            self.acts[(self.end + i) % self.buffer_size] = act 
            self.rewards[(self.end + i) % self.buffer_size] = reward 
            self.next_obses[(self.end + i) % self.buffer_size] = next_obs
            self.terminals[(self.end + i) % self.buffer_size] = terminal 

        self.end += len(transitions) 

    def sample_transitions(self, n):
        indices = np.random.randint(0, min(self.end, self.buffer_size), size=n)
        return self.state_encoder(self.obses[indices]), self.acts[indices], self.rewards[indices], self.state_encoder(self.next_obses[indices]), self.terminals[indices]


def discrete_state_encoder(obs, obs_dim): 
    return np.eye(obs_dim)[obs]

def collect_episodes(env, behavior_policy, num_episodes, state_encoder, seed, save_episodes=[]) -> list[Trajectory]:
    trajectories = []
    recordings = []
    for i in range(num_episodes):
        frames = []
        episode_over = False
        obs, info = env.reset()
        if i in save_episodes:
            frames.append(env.render())
        obses, acts, next_obses, rewards, terminals = [], [], [], [] ,[]
        while not episode_over:
            obses.append(obs)

            act = behavior_policy(state_encoder(obs))
            obs, reward, terminated, truncated, info = env.step(act)
            if i in save_episodes:
                frames.append(env.render())

            episode_over = terminated or truncated

            acts.append(act)
            rewards.append(reward)
            next_obses.append(obs)
            terminals.append(terminated)

        trajectories.append(Trajectory(obses, acts, rewards, next_obses, terminals))
        if i in save_episodes:
            recordings.append(np.array(frames))
    if save_episodes:
        return trajectories, recordings
    else:
        return trajectories

def collect_transitions(env, obs, behavior_policy, num_transitions, state_encoder, seed):
    transitions = []
    for _ in range(num_transitions):
        act = behavior_policy(state_encoder(obs))
        next_obs, reward, terminated, truncated, info = env.step(act)
        transitions.append((obs, act, reward, next_obs, terminated))
        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs = next_obs
    return (obs, transitions)
