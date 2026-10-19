
import numpy as np

class DPAgent:

    def __init__(self, env, discount_factor):
        self.env = env
        self.values = np.zeros(self.env.observation_space.n)
        self.q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.discount_factor = discount_factor

    def policy(self, observation):
        return np.argmax(self.q[observation])

    def learn(self):
        P = self.env.unwrapped.P

        keep_iterating = True
        stages = 0
        while keep_iterating:
            updated_values = np.zeros(self.env.observation_space.n)
            for state in range(self.env.observation_space.n):
                for action in range(self.env.action_space.n):
                    possible_update = 0
                    for prob, next_state, reward, terminated in P[state][action]:
                        possible_update += prob * reward + self.discount_factor * prob * self.values[next_state]
                    updated_values[state] = max(updated_values[state], possible_update)

            stages += 1
            self.values = updated_values
            keep_iterating = stages <= 100 # NOTE: hardcoded figure out ideal solution

        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                q_value = 0
                for prob, next_state, reward, terminated in P[state][action]:
                    q_value += prob * reward + self.discount_factor * prob * self.values[next_state]
                self.q[state][action]=q_value
    
        
        
