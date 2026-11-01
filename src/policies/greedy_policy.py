
from .base_policy import Policy 
import torch
class EpsilonGreedyPolicy(Policy):
    def __init__(self, critic, epsilon):
        super().__init__()
        self.critic=critic
        self.epsilon = epsilon

    def get_action(self, obs : int) -> int:
        return torch.argmax(self.critic(obs), dim=-1).item()