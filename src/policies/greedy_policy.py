
from .base_policy import Policy 
import numpy as np

class EpsilonGreedyPolicy(Policy):
    def __init__(self, critic, epsilon, action_sampler, rng):
        super().__init__()
        self.critic = critic
        self.epsilon = epsilon
        self.action_sampler = action_sampler
        self.rng = rng

    def get_action(self, obses):
        if self.rng.uniform() < self.epsilon:
            return self.action_sampler() 
        else:
            return np.argmax(self.critic.get_state_action_value(obses), axis=-1)
