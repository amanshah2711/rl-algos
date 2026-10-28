
import gymnasium as gym

from src.agents.pg_agent import PGAgent
from src.policies.pg_policy import PGPolicy
from src.utils.environment import collect_episodes

from torch.utils.tensorboard import SummaryWriter
import random
import torch

# hyperparameter initialization for training
env_config = {
    'id' : 'FrozenLake-v1',
    'is_slippery' : False,
    'map_name' : '4x4'
}
config = {
    'algorithm' : 'REINFORCE',
    'train_iters' : 1000,
    'episodes_per_train_iter' : 100,
    'actor_hidden_layer_dim': 4,
    'actor_num_hidden_layers': 2,
    'actor_lr' : 1e-3,
    'discount_factor' : 0.99,
    'eval_episodes' : 200,
    'seed' : 2711,
    **env_config
}

random.seed(config['seed'])
torch.manual_seed(config['seed'])

# logging setup
writer = SummaryWriter(f'runs/{config['algorithm']}/seed_{config['seed']}')

# setting environment
train_env = gym.make(**env_config)
obs_dim = train_env.observation_space.n
act_dim = train_env.action_space.n
total_env_steps = 0

# setting up agent

target_policy = PGPolicy(obs_dim=obs_dim,
                         hidden_layer_dim= config['actor_hidden_layer_dim'], 
                         act_dim=act_dim,          
                         num_hidden_layers=config['actor_num_hidden_layers'],
                         lr=config['actor_lr'])

agent = PGAgent(target_policy=target_policy,
                 discount_factor=config['discount_factor'])

# training loop
for i in range(config['train_iters']):
    print('current iterations is iteration, ', i)
    trajectories = collect_episodes(train_env, agent.target_policy.sample_action, config['episodes_per_train_iter'], seed=config['seed'])
    total_env_steps += sum([len(episode.obs) for episode in trajectories])
    
    metrics = agent.train(trajectories)
    for name, value in metrics.items():
        writer.add_scalar(name, value, total_env_steps)


# hyperparameters for evaluation
eval_episodes = config['eval_episodes']

eval_env = gym.make('FrozenLake-v1', is_slippery=False, map_name='4x4')

eval_trajectories = collect_episodes(eval_env, agent.target_policy.get_action, eval_episodes)
success_rate = sum([sum(episode.rewards) for episode in eval_trajectories]) / eval_episodes

writer.add_hparams(
    hparam_dict=config,
    metric_dict={
        "eval/success_rate": success_rate,
    },

)
writer.close()
train_env.close()
eval_env.close
    


