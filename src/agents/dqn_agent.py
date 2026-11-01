
from .base_agent import Agent
from ..policies.greedy_policy import EpsilonGreedyPolicy
from ..utils.environment import Trajectory, ReplayBuffer
from ..utils.neural_nets import create_mlp

class DQNAgent(Agent):

    def __init__(self, critic, batch_size, critic_updates_per_batch, target_update_freq, epsilon, discount_factor):
        super().__init__()
        self.replay_buffer = ReplayBuffer(1_000_000)
        self.critic = critic
        self.behavior_policy = EpsilonGreedyPolicy(critic=critic, epsilon=epsilon)
        self.target_policy = EpsilonGreedyPolicy(critic=critic, epsilon=0)
        self.batch_size = batch_size
        self.critic_updates_per_batch = critic_updates_per_batch
        self.target_update_freq = target_update_freq
        self.discount_factor = discount_factor
        self.t = 0

    def train(self, transitions) -> None:
        self.replay_buffer.add_to_buffer(transitions)
        for _ in range(self.critic_updates_per_batch): # Mathematically an approximation of the bellman backup operator
            transitions = self.replay_buffer.sample_transitions(self.batch_size)
            self.critic.update(transitions)

        self.t += 1
        if (self.t % self.target_update_freq): # TODO: Find proper name for this
            self.critic.update_target_network()

