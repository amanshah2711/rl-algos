
from .mlp_policy import MLPPolicy
import torch
import torch.nn as nn
from torch import optim
from ..utils.neural_nets import create_mlp
import torch.nn.functional as F

from numpy.typing import NDArray

from torch.distributions import Categorical

class PGPolicy(MLPPolicy):
    def __init__(self, obs_dim, hidden_layer_dim, act_dim, num_hidden_layers, lr):
        # assume a discrete action space and discrete observation space
        # for frozen lake we will consider a simple two layer MLP
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.logits_net = create_mlp(input_dim=obs_dim, 
                                 hidden_layer_dim=hidden_layer_dim, 
                                 output_dim=act_dim, 
                                 num_hidden_layers=num_hidden_layers)
        self.optimizer = optim.Adam(self.logits_net.parameters(), lr)

    def sample_action(self, obs : int) -> int:
        with torch.no_grad():
            logits = self(torch.tensor(obs))  # (ac_dim, )
            return Categorical(logits=logits).sample().item()
 
    def get_action(self, obs : int) -> int:
        with torch.no_grad():
            logits = self(torch.tensor(obs)) # (ac_dim, )
            action = torch.argmax(logits)
            return action.item()

    def update(self, obs : torch.LongTensor, actions : torch.LongTensor, weights : torch.FloatTensor) -> None:

        self.optimizer.zero_grad()

        logits = self(obs)      # Shape : (num_transitions, ac_dim)
        ac_dist = Categorical(logits=logits)  # batch shape for this distribution is (num_transitions, )
        log_likelihood = ac_dist.log_prob(actions) # Shape : (num_transitions, )
        pseudo_loss = (-log_likelihood * weights).mean() # Not a loss in the truest sense, simply a function whose gradient is the PG gradient

        pseudo_loss.backward()
        self.optimizer.step()

        return {'train/actor_loss' : pseudo_loss.item()}

    def forward(self, obs : torch.LongTensor):
        obs = F.one_hot(obs, num_classes=self.obs_dim).to(dtype=torch.float32)
        return self.logits_net(obs)
