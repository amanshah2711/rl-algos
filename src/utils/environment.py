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
    rewards : list[int]
    next_obses : list[int]
    terminals : list[bool]

@dataclass
class Transition:
    obs : int
    act : int
    next_obs : int
    reward : int
    terminal : int

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
            self.obses[(self.end + i) % self.buffer_size] = np.argmax(obs)
            self.acts[(self.end + i) % self.buffer_size] = act 
            self.rewards[(self.end + i) % self.buffer_size] = reward 
            self.next_obses[(self.end + i) % self.buffer_size] = np.argmax(next_obs)
            self.terminals[(self.end + i) % self.buffer_size] = terminal 

        self.end += len(transitions) 
        #self.buffer.extend(transitions)
        #while (len(self.buffer) > self.buffer_size):
            #self.buffer.popleft()

    def sample_transitions(self, n):
        indices = np.random.randint(0, min(self.end, self.buffer_size), size=n)
        return self.state_encoder(self.obses[indices]), self.acts[indices], self.rewards[indices], self.state_encoder(self.next_obses[indices]), self.terminals[indices]
        #transitions = random.choices(self.buffer, k=n)
        #return (
            #np.array([transition[0] for transition in transitions]),
            #np.array([transition[1] for transition in transitions]),
            #np.array([transition[2] for transition in transitions]),
            #np.array([transition[3] for transition in transitions]),
            #np.array([transition[4] for transition in transitions])
        #)


def discrete_state_encoder(obs, obs_dim): 
    return np.eye(obs_dim)[obs]

def collect_episodes(env, behavior_policy, num_episodes, state_encoder, seed, save_episodes=[]) -> list[Trajectory]:
    trajectories = []
    recordings = []
    for i in range(num_episodes):
        frames = []
        episode_over = False
        obs, info = env.reset(seed=seed)
        obs = state_encoder(obs)
        if i in save_episodes:
            frames.append(env.render())
        obses, acts, next_obses, rewards, terminals = [], [], [], [] ,[]
        while not episode_over:
            obses.append(obs)

            act = behavior_policy(obs)
            obs, reward, terminated, truncated, info = env.step(act)
            obs = state_encoder(obs)
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
        act = behavior_policy(obs)
        next_obs, reward, terminated, truncated, info = env.step(act)
        transitions.append((obs, act, reward, state_encoder(next_obs), terminated))
        if terminated or truncated:
            obs, info = env.reset(seed=seed)
        else:
            obs = next_obs
        obs = state_encoder(obs)
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