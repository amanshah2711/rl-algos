
from .base_agent import Agent
from ..utils.environment import Trajectory, compute_returns
import numpy as np

class PGAgent(Agent):

    def __init__(self, target_policy, discount_factor):
        self.target_policy = target_policy
        self.discount_factor = discount_factor

    def train(self, trajectories : list[Trajectory]):
        flattened_q_values= np.concat([sample for sample in compute_returns(trajectories, self.discount_factor)])
        flattened_obs = np.concat([trajectory.obses for trajectory in trajectories])
        flattened_acs = np.concat([trajectory.acts for trajectory in trajectories])

        policy_metrics = self.target_policy.update(flattened_obs, flattened_acs, flattened_q_values)
        return {
            **policy_metrics,
            'train/mean_reward_to_go' : np.mean(flattened_q_values),
            'train/mean_episode_return' : np.mean([sum(episode.rewards) for episode in trajectories])
        }

    

