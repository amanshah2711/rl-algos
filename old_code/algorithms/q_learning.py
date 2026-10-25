import numpy as np

class QLearning:

    def __init__(self, env, discount_factor):
        self.env = env
        self.discount_factor = discount_factor
        self.exploration_rate = 0.5
        self.q = np.zeros((env.observation_space.n, env.action_space.n))
        rng = np.random.default_rng(123)
        self.policy = lambda observation : self.env.action_space.sample() if np.random.uniform(0, 1) < self.exploration_rate else rng.choice(np.where(self.q[observation, :] == max(self.q[observation, :]))[0])
        
    def learn(self) -> dict:
        observation, info = self.env.reset() 
        n_iters = 1000000
        alpha = 0.8
        total_reward = 0
        for val in range(n_iters):
            if val > 300:
                self.exploration_rate = 0.1
            action = self.policy(observation)
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            sample = reward + self.discount_factor * np.max(self.q[next_observation])
            self.q[observation][action] = (1 - alpha) * self.q[observation][action] + alpha * sample
            if terminated or truncated:
                if total_reward > 0:
                    print(f"Goal reached by iteration {val}!")

                total_reward =0
                observation, info = self.env.reset()
            else:
                observation = next_observation
        self.policy = lambda observation: np.argmax(self.q[observation])


