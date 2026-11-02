
from .base_critic import Critic
from ..utils.neural_nets import create_mlp
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

class DQNCritic(Critic, nn.Module):

    def __init__(self, obs_dim, hidden_layer_dim, act_dim, num_hidden_layers, lr, discount_factor):
        super().__init__()
        self.obs_dim = obs_dim
        self.critic_net = create_mlp(obs_dim, hidden_layer_dim, act_dim, num_hidden_layers) # output shape (num_transitions, act_dim)
        self.target_critic_net = create_mlp(obs_dim, hidden_layer_dim, act_dim, num_hidden_layers) # output shape (num_transitions, act_dim)
        self.optimizer = torch.optim.Adam(params=self.critic_net.parameters(), lr=lr)
        self.discount_factor = discount_factor
        self.update_target_network()

    def update_target_network(self):
        for (target_param, param) in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(param)

    def update(self, transitions):
        obses = torch.tensor([transition[0] for transition in transitions], dtype=torch.long) # Shape (num_transitions, )
        acts = torch.tensor([transition[1] for transition in transitions], dtype=torch.long) # Shape (num_transitions, )
        rewards = torch.tensor([transition[2] for transition in transitions], dtype=torch.float) # Shape(num_transitions, )
        next_obses = torch.tensor([transition[3] for transition in transitions], dtype=torch.long) # Shape (num_transitions, )
        terminals = torch.tensor([transition[4] for transition in transitions], dtype=torch.float) # Shape (num_transitions, )

        with torch.no_grad():
            one_hot_obses = F.one_hot(next_obses, num_classes=self.obs_dim).to(dtype=torch.float) # Shape (num_transitions, obs_dim)
            state_action_values = self.target_critic_net(one_hot_obses) # Shape (num_transitions, act_dim)
            targets = rewards + self.discount_factor * (1-terminals) * torch.max(state_action_values, dim=1).values # Shape (num_transitions, )

        self.optimizer.zero_grad()
        loss = F.huber_loss(self(obses)[torch.arange(0, len(transitions)), acts], targets, reduction='mean')
        loss.backward()
        self.optimizer.step()

        return {
            'train/critic_loss' : loss.item()
        }

    def forward(self, obs : torch.LongTensor) -> torch.LongTensor:
        obs = F.one_hot(obs, num_classes=self.obs_dim).to(dtype=torch.float) # (num_observations, obs_dim)
        return self.critic_net(obs) # (num_observations, act_dim)