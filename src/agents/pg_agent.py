
from .base_agent import Agent
from ..utils.environment import Trajectory, compute_returns
import torch

class PGAgent(Agent):

    def __init__(self, target_policy, discount_factor):
        self.target_policy = target_policy
        self.discount_factor = discount_factor

    def train(self, trajectories : list[Trajectory]):
        flattened_q_values= torch.concat([torch.tensor(sample) for sample in compute_returns(trajectories, self.discount_factor)])
        flattened_obs = torch.concat([torch.tensor(trajectory.obs, dtype=torch.long) for trajectory in trajectories])
        flattened_acs = torch.concat([torch.tensor(trajectory.acs, dtype=torch.long) for trajectory in trajectories])

        policy_metrics = self.target_policy.update(flattened_obs, flattened_acs, flattened_q_values)
        return {
            **policy_metrics,
            'train/mean_reward_to_go' : flattened_q_values.mean().item(),
            'train/mean_episode_return' : torch.tensor([sum(episode.rewards) for episode in trajectories], dtype=torch.float).mean().item()
        }

    

