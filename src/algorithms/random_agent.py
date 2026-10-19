

class RandomAgent:

    def __init__(self, env):
        self.env=env

    def policy(self, observation):
        return self.env.action_space.sample()