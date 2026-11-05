
from .base_agent import Agent

class DQNAgent(Agent):

    def __init__(self, critic, batch_size, critic_updates_per_batch, target_update_freq, behavior_policy, target_policy, discount_factor, replay_buffer):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.critic = critic
        self.behavior_policy = behavior_policy
        self.target_policy = target_policy
        self.batch_size = batch_size
        self.critic_updates_per_batch = critic_updates_per_batch
        self.target_update_freq = target_update_freq
        self.discount_factor = discount_factor
        self.t = 0

    def train(self, transitions) -> None:
        self.replay_buffer.add_to_buffer(transitions)
        
        for _ in range(self.critic_updates_per_batch): # Mathematically an approximation of the bellman backup operator
            obses, acts, rewards, next_obses, terminals = self.replay_buffer.sample_transitions(self.batch_size)
            metrics = self.critic.update(obses, acts, rewards, next_obses, terminals)

        self.t += 1
        if (self.t % self.target_update_freq == 0): 
            self.critic.update_target_network()

        return metrics

