
from .pg_agent import PGAgent
from ..utils.environment import Trajectory
import numpy as np

class ACAgent(PGAgent):

    def __init__(self, target_policy, critic, critic_updates_per_actor_update, discount_factor):
        super().__init__(target_policy, discount_factor=discount_factor)
        self.critic = critic
        self.critic_updates_per_actor_update = critic_updates_per_actor_update

    def train(self, trajectories : list[Trajectory]):
        
        flattened_obses = np.concat([trajectory.obses for trajectory in trajectories])
        flattened_acts = np.concat([trajectory.acts for trajectory in trajectories])
        flattened_rewards = np.concat([trajectory.rewards for trajectory in trajectories])
        flattened_next_obses = np.concat([trajectory.next_obses for trajectory in trajectories])
        flattened_terminals = np.concat([trajectory.terminals for trajectory in trajectories])

        for _ in range(self.critic_updates_per_actor_update):
            critic_metrics = self.critic.update(flattened_obses, flattened_rewards, flattened_next_obses, flattened_terminals)

        advantages = (flattened_rewards + self.discount_factor * (1.0 - flattened_terminals) * self.critic.get_value(flattened_next_obses) - self.critic.get_value(flattened_obses))
        actor_metrics = self.target_policy.update(flattened_obses, flattened_acts, advantages)

        return {
            **actor_metrics,
            'train/mean_episode_return' : np.mean([sum(episode.rewards) for episode in trajectories]),
            **critic_metrics
        }


