
from .pg_agent import PGAgent
from ..critics.value_critic import DeepValueCritic
from ..policies.pg_policy import PGPolicy
from ..utils.environment import Trajectory
import torch
import numpy as np

class ACAgent(PGAgent):

    def __init__(self, target_policy, critic, critic_updates_per_actor_update, discount_factor):
        super().__init__(target_policy, discount_factor=discount_factor)
        self.critic = critic
        self.critic_updates_per_actor_update = critic_updates_per_actor_update

    def train(self, trajectories : list[Trajectory]):

        flattened_obs = torch.concat([torch.tensor(trajectory.obs, dtype=torch.long) for trajectory in trajectories])
        flattened_acs = torch.concat([torch.tensor(trajectory.acs, dtype=torch.long) for trajectory in trajectories])
        flattened_rewards = torch.concat([torch.tensor(trajectory.rewards, dtype=torch.float) for trajectory in trajectories])
        flattened_next_obs = torch.concat([torch.tensor(trajectory.next_obs, dtype=torch.long) for trajectory in trajectories])
        flattened_terminals = torch.concat([torch.tensor(trajectory.terminals, dtype=torch.long) for trajectory in trajectories])

        for _ in range(self.critic_updates_per_actor_update):
            critic_metrics = self.critic.update(flattened_obs, flattened_rewards, flattened_next_obs, flattened_terminals)

        with torch.no_grad():
            advantages = (flattened_rewards + self.discount_factor * (1.0 - flattened_terminals) * self.critic(flattened_next_obs) - self.critic(flattened_obs))
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_metrics = self.target_policy.update(flattened_obs, flattened_acs, advantages)
        return {
            **actor_metrics,
            'train/mean_episode_return' : torch.tensor([sum(episode.rewards) for episode in trajectories], dtype=torch.float).mean().item(),
            **critic_metrics
        }


