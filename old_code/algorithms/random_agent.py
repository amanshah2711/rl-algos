
from ..utils.replay_buffer import ReplayBuffer

class RandomAgent:

    def __init__(self, env):
        self.env=env
        self.replay_buffer = ReplayBuffer()

    def policy(self, observation):
        return self.env.action_space.sample()