
from .base_agent import Agent
from ..utils.environment import Trajectory
import numpy as np

class PGAgent(Agent):

    def __init__(self, target_policy):
        self.target_policy = target_policy
        self.critic = None

    def train(self, trajectories : list[Trajectory]):
        gamma = 0.9
        reward_process_samples = [trajectory.rewards for trajectory in trajectories]
        reward_to_go_samples = []
        print('training iteration started', len(reward_process_samples))
        count = 0
        for reward_process_sample in reward_process_samples:
            reward_to_go_sample = np.zeros(len(reward_process_sample))
            reward_to_go = 0.0
            #print("reward process sampling iteration", count)
            for index in reversed(range(len(reward_process_sample))):
                reward = reward_process_sample[index]
                reward_to_go = reward + gamma * reward_to_go
                reward_to_go_sample[index]= reward_to_go
            reward_to_go_samples.append(reward_to_go_sample)
            count += 1


        flattened_obs = np.concatenate([trajectory.obs for trajectory in trajectories])
        flattened_acs = np.concatenate([trajectory.acs for trajectory in trajectories])
        flattened_q_values = np.concatenate([reward_to_go_sample for reward_to_go_sample in reward_to_go_samples])
        #flatted_terminals = np.concatenate([[trajectory.terminals for trajectory in trajectories]])

        print('entering policy improvement')
        self.target_policy.policy_improvement_step(flattened_obs, flattened_acs, flattened_q_values)

        print('training iteration ended')
    

