
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

    def get_state_action_value(self, obses):
        with torch.no_grad():
            return self(torch.as_tensor(obses, dtype=torch.float)).numpy() 

    def update_target_network(self):
        for (target_param, param) in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(param)

    def update(self, obses, acts, rewards, next_obses, terminals):

        obses = torch.as_tensor(obses, dtype=torch.float) # Shape (num_transitions, obs_dim)
        acts = torch.as_tensor(acts, dtype=torch.long) # Shape (num_transitions, )
        rewards = torch.as_tensor(rewards, dtype=torch.float) # Shape(num_transitions, )
        next_obses = torch.as_tensor(next_obses, dtype=torch.float) # Shape (num_transitions, obs_dim)
        terminals = torch.as_tensor(terminals, dtype=torch.long) # Shape (num_transitions, )
        num_transitions = obses.size(0)

        with torch.no_grad():
            state_action_values = self.target_critic_net(next_obses) # Shape (num_transitions, act_dim)
            targets = rewards + self.discount_factor * (1-terminals) * torch.max(state_action_values, dim=1).values # Shape (num_transitions, )

        self.optimizer.zero_grad()
        loss = F.huber_loss(self(obses)[torch.arange(0, num_transitions), acts], targets, reduction='mean')
        loss.backward()
        self.optimizer.step()

        return {
            'train/critic_loss' : loss.item()
        }

    def forward(self, obs):
        return self.critic_net(obs) # (num_observations, act_dim)