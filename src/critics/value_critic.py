
from .base_critic import Critic
import torch.nn as nn
from ..utils.environment import Trajectory
from ..utils.neural_nets import create_mlp
from numpy.typing import NDArray
import torch.nn.functional as F
import torch
from torch import optim

class DeepValueCritic(Critic, nn.Module):

    def __init__(self, obs_dim, hidden_layer_dim, num_hidden_layers, discount_factor, lr):
        super().__init__()
        self.obs_dim = obs_dim
        self.critic_net = create_mlp(obs_dim, hidden_layer_dim, 1, num_hidden_layers) # TODO : Go back and changes the hidden layer dim to be customizable and customize num hidden layer
        self.discount_factor = discount_factor
        self.optimizer = optim.Adam(self.critic_net.parameters(), lr=lr)
    
    def get_value(self, obs : int) -> int: 
        with torch.no_grad():
            return self(torch.tensor(obs, dtype=torch.long)).item()

    def update(self, obs : torch.LongTensor, rewards : torch.FloatTensor, next_obs : torch.LongTensor, terminals : torch.LongTensor):

        self.optimizer.zero_grad()

        with torch.no_grad():
            targets = rewards + self.discount_factor * (1 - terminals) * self(next_obs)

        loss = F.mse_loss(self(obs), targets, reduction='mean')
        loss.backward()

        self.optimizer.step()

        return {
            'train/critic_loss' : loss.item()
        }

    def forward(self, obs : torch.LongTensor) -> torch.FloatTensor:
        obs = F.one_hot(obs, num_classes=self.obs_dim).to(dtype=torch.float) # (num_observations, ob_dim)
        return self.critic_net(obs).squeeze(-1) # (num_observations, )
