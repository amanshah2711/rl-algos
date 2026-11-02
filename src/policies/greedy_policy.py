
from .base_policy import Policy 
import torch
import numpy as np
class EpsilonGreedyPolicy(Policy):
    def __init__(self, critic, epsilon):
        super().__init__()
        self.critic=critic
        self.epsilon = epsilon

    def get_action(self, obs : int) -> int:
        return torch.argmax(self.critic(torch.tensor(obs)), dim=-1).item()
