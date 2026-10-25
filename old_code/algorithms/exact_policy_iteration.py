import numpy as np
class ExactPolicyIteration:

    def __init__(self, env, discount_factor):
        self.env = env
        self.discount_factor = discount_factor
        self.policy = lambda observation : self.env.action_space.sample()

    def learn(self):
        P = self.env.unwrapped.P

        for _ in range(100):
            transition_matrix = np.zeros((self.env.observation_space.n, self.env.observation_space.n))
            r = np.zeros(self.env.observation_space.n)
            for state in range(self.env.observation_space.n):
                action = self.policy(state)
                for prob, next_state, reward, _ in P[state][action]:
                    transition_matrix[state][next_state] = prob
                    r[state] += prob * reward

            value = np.linalg.solve(np.eye(self.env.observation_space.n) - self.discount_factor * transition_matrix, r)

            q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
            for state in range(self.env.observation_space.n):
                for action in range(self.env.action_space.n):
                    q_value = 0
                    for prob, next_state, reward, _ in P[state][action]:
                        q_value += reward * prob + prob * self.discount_factor * value[next_state]

                    q[state][action] = q_value
            self.policy = lambda observation : np.argmax(q[observation]) 

                



