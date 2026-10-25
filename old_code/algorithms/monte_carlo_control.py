
from ..utils.evaluation import monte_carlo_q_evaluation
import numpy as np

class MCControl:

    def __init__(self, env, discount_factor):
        self.env=env
        self.policy = lambda observation : env.action_space.sample()
        self.discount_factor = discount_factor

    def learn(self):
        for _ in range(100):
            q_function_estimate  = monte_carlo_q_evaluation(self, 10, self.discount_factor)
            print(q_function_estimate)
            epsilon = 0.5
            self.policy = lambda observation : self.env.action_space.sample() if np.random.random() < epsilon else np.argmax(q_function_estimate[observation])

        
        self.policy = lambda observation : np.argmax(q_function_estimate[observation])
 