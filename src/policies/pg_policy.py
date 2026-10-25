
from .mlp_policy import MLPPolicy
import torch
import torch.nn as nn
from torch import optim
from ..utils.neural_nets import create_mlp
import torch.nn.functional as F

from torch.distributions import Categorical

class PGPolicy(MLPPolicy):
    def __init__(self, obs_dim, ac_dim, learning_rate):
        # assume a discrete action space and discrete observation space
        # for frozen lake we will consider a simple two layer MLP
        super().__init__()
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        self.logits_net = create_mlp(input_dim=obs_dim, 
                                 hidden_layer_dim=ac_dim, 
                                 output_dim=ac_dim, 
                                 num_hidden_layers=2)
        self.optimizer = optim.Adam(self.logits_net.parameters(), learning_rate)

    def policy_improvement_step(self, obs, actions, q_values):
        obs = torch.from_numpy(obs).float()
        actions = torch.from_numpy(actions).float()
        q_values = torch.from_numpy(q_values).float()

        self.optimizer.zero_grad()

        logits = self(obs)      # Shape : (num_transitions, ac_dim)
        ac_dist = Categorical(logits=logits)  # Shape (num_transitions, ac_dim) policy dist for each obs
        log_likelihood = ac_dist.log_prob(actions) # Shape : (num_transitions, )
        pseudo_loss = (-log_likelihood * q_values).mean() # Not a loss in the truest sense, simply a function whose gradient is the PG gradient

        pseudo_loss.backward()
        self.optimizer.step()

    def sample(self, obs):
        with torch.no_grad():
            logits = self(obs)
        return Categorical(logits=logits).sample().item()
 
    def get_action(self, obs):
        with torch.no_grad():
            logits = self(obs)
            action = torch.argmax(logits, dim=-1)
        return action.item()

    def forward(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.long)
        obs = F.one_hot(obs, num_classes=self.obs_dim).float()
        return self.logits_net(obs)
