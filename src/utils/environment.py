from dataclasses import dataclass
from typing import Tuple
from numpy.typing import NDArray
import numpy as np
from collections import deque
import random

@dataclass
class Trajectory:
    obses : list[int]
    acts : list[int]
    next_obses : list[int]
    rewards : list[int]
    terminals : list[bool]

@dataclass
class Transition:
    obs : int
    act : int
    next_obs : int
    reward : int
    terminal : int

class ReplayBuffer:

    def __init__(self, buffer_size):
        self.buffer = deque([])
        self.buffer_size = buffer_size 

    def add_to_buffer(self, transitions):
        self.add_to_buffer.extend(transitions)
        while (len(self.buffer) > self.buffer_size):
            self.buffer.popleft()

    def sample_transitions(self, n):
        return random.choices(self.buffer, k=n)


def collect_episodes(env, behavior_policy, num_episodes, seed=2711) -> list[Trajectory]:
    trajectories = []
    for _ in range(num_episodes):
        episode_over = False
        ob, info = env.reset(seed=seed)
        obses, acts, next_obses, rewards, terminals = [], [], [], [] ,[]
        while not episode_over:
            obses.append(ob)

            act = behavior_policy(ob)
            ob, reward, terminated, truncated, info = env.step(act)

            episode_over = terminated or truncated

            acts.append(act)
            rewards.append(reward)
            next_obses.append(ob)
            terminals.append(terminated)

        trajectories.append(Trajectory(obses, acts, next_obses, rewards, terminals))
    return trajectories

def collect_transitions(env, obs, behavior_policy, num_transitions, seed=2711):
    transitions = []
    for _ in range(num_transitions):
        act = behavior_policy(obs)
        next_obs, reward, terminated, truncated, info = env.step(act)
        transitions.append((obs, act, reward, next_obs, terminated))
        if terminated or truncated:
            obs, info = env.reset(seed=seed)
        else:
            obs = next_obs
    return (obs, transitions)

def compute_returns(trajectories : list[Trajectory], gamma: float):
    reward_process_samples = [trajectory.rewards for trajectory in trajectories]
    reward_to_go_samples = []
    for reward_process_sample in reward_process_samples:
        reward_to_go_sample = np.zeros(len(reward_process_sample))
        reward_to_go = 0.0
        for index in reversed(range(len(reward_process_sample))):
            reward = reward_process_sample[index]
            reward_to_go = reward + gamma * reward_to_go
            reward_to_go_sample[index]= reward_to_go
        reward_to_go_samples.append(reward_to_go_sample)
    return reward_to_go_samples