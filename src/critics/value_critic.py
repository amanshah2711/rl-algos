
from .base_critic import Critic
import torch.nn as nn
from ..utils.environment import Trajectory
from ..utils.neural_nets import create_mlp
from numpy.typing import NDArray
import torch.nn.functional as F
import torch
from torch import optim

class DeepValueCritic(Critic, nn.Module):

    def __init__(self, obs_dim, hidden_layer_dim, num_hidden_layers, discount_factor, lr, gradient_steps_per_update):
        super().__init__()
        self.obs_dim = obs_dim
        self.critic_net = create_mlp(obs_dim, hidden_layer_dim, 1, num_hidden_layers) 
        self.discount_factor = discount_factor
        self.optimizer = optim.Adam(self.critic_net.parameters(), lr=lr)
        self.gradient_steps_per_update = gradient_steps_per_update
    
    def get_value(self, obs): 
        with torch.no_grad():
            return self(torch.as_tensor(obs, dtype=torch.float)).numpy()

    def update(self, obses : NDArray, rewards : NDArray, next_obses : NDArray, terminals : NDArray):

        obses = torch.from_numpy(obses).to(dtype=torch.float)
        rewards = torch.from_numpy(rewards).to(dtype=torch.float)
        next_obses = torch.from_numpy(next_obses).to(dtype=torch.float)
        terminals = torch.from_numpy(terminals).to(dtype=torch.float)

        with torch.no_grad():
            targets = rewards + self.discount_factor * (1 - terminals) * self(next_obses)

        for _ in range(self.gradient_steps_per_update):
            self.optimizer.zero_grad()
            loss = F.mse_loss(self(obses), targets, reduction='mean')
            loss.backward()
            self.optimizer.step()

        return {
            'train/critic_loss' : loss.item()
        }

    def forward(self, obs):
        return self.critic_net(obs).squeeze(-1) # (num_observations, )
