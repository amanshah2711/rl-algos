
from .base_policy import Policy 
import numpy as np
class EpsilonGreedyPolicy(Policy):
    def __init__(self, critic, epsilon):
        super().__init__()
        self.critic=critic
        self.epsilon = epsilon

    def get_action(self, obses):
        return np.argmax(self.critic.get_state_action_value(obses), axis=-1)
