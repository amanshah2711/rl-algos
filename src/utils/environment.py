from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np

@dataclass
class Trajectory:
    obs : list[int]
    acs : list[int]
    next_obs : list[int]
    rewards : list[int]
    terminals : list[bool]


def collect_episodes(env, collect_policy, num_episodes, seed=2711) -> list[Trajectory]:
    trajectories = []
    for _ in range(num_episodes):
        episode_over = False
        ob, info = env.reset(seed=seed)
        obs, acs, next_obs, rewards, terminals = [], [], [], [] ,[]
        while not episode_over:
            obs.append(ob)

            ac = collect_policy(ob)
            ob, reward, terminated, truncated, info = env.step(ac)

            episode_over = terminated or truncated

            acs.append(ac)
            rewards.append(reward)
            next_obs.append(ob)
            terminals.append(terminated)

        trajectories.append(Trajectory(obs, acs, next_obs, rewards, terminals))
    return trajectories

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