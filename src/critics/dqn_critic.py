
from .base_critic import Critic
from ..utils.neural_nets import create_mlp
import torch.optim
import torch.nn.functional as F

class DQNCritic(Critic):

    def __init__(self, obs_dim, hidden_layer_dim, act_dim, num_hidden_layers, lr, gradient_steps_per_batch, discount_factor):
        super().__init__()
        self.critic_net = create_mlp(obs_dim, hidden_layer_dim, act_dim, num_hidden_layers) # output shape (num_transitions, act_dim)
        self.target_critic_net = create_mlp(obs_dim, hidden_layer_dim, act_dim, num_hidden_layers) # output shape (num_transitions, act_dim)
        self.optimizer = torch.optim.Adam(params=self.critic_net.parameters(), lr=lr)
        self.gradient_steps_per_batch = gradient_steps_per_batch
        self.discount_factor = discount_factor
        self.update_target_network()

    def update_target_network(self):
        for (target_param, param) in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(param)

    def update(self, transitions):
        obses = torch.tensor([transition[0] for transition in transitions]) # Shape (num_transitions, obs_dim)
        acts = torch.tensor([transition[0] for transition in transitions]) # Shape (num_transitions, act_dim)
        rewards = torch.tensor([transition[0] for transition in transitions]) # Shape(num_transitions, )
        next_obses = torch.tensor([transition[0] for transition in transitions]) # Shape (num_transitions, obs_dim)
        terminals = torch.tensor([transition[0] for transition in transitions]) # Shape (num_transitions, )

        with torch.no_grad():
            targets = rewards + self.discount_factor * (1-terminals) * torch.argmax(self.target_critic_net(next_obses), dim=1) # Shape (num_transitions, )

        self.optimizer.zero_grad()
        loss = F.huber_loss(self.critic_net(obses)[torch.arange(0, len(transitions)), acts], targets, reduction='mean')
        loss.backward()
        self.optimizer.step()

        return {
            'train/critic_loss' : loss.item()
        }