
from .base_policy import Policy

class RandomPolicy(Policy):

    def __init__(self, acs_dim):
        super().__init__()