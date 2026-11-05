
from .environment import Trajectory
import numpy as np

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

def compute_mean_return(episodes : list[Trajectory]):
    return np.mean([sum(episode.rewards) for episode in episodes])

def compute_mean_discounted_return(episodes : list[Trajectory], discount_factor):
    return # TODO

def compute_mean_average_return(episodes : list[Trajectory]):
    return # TODO

def compute_mean_episode_length(episodes : list[Trajectory]):
    return np.mean([len(episode.obses) for episode in episodes])