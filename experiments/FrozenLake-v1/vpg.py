
# Python Libraries
import argparse
import random
from datetime import datetime, timezone

# RL Implementations
from src.agents.pg_agent import PGAgent
from src.policies.pg_policy import PGPolicy
from src.utils.environment import collect_episodes, discrete_state_encoder
from src.utils.metrics import compute_mean_return, compute_mean_episode_length

# Environment 
import gymnasium as gym
import torch
import numpy as np

# Logging Data
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True, help='Random seed used for reproducibility and verifiability of experiments')
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
rng = np.random.default_rng(seed=args.seed)

# hyperparameter initialization for training
env_config = {
    'id' : 'FrozenLake-v1',
    'is_slippery' : False,
    'map_name' : '4x4',
    'render_mode' : 'rgb_array'
}

config = {
    'algorithm' : 'REINFORCE',
    'train_iters' : 2000,
    'episodes_per_train_iter' : 100,
    'actor_hidden_layer_dim': 4,
    'actor_num_hidden_layers': 2,
    'actor_lr' : 1e-3,
    'discount_factor' : 0.99,
    'eval_episodes' : 200,
    'eval_frequency' : 50,
    'seed' : args.seed,
    **env_config
}

random.seed(config['seed'])
torch.manual_seed(config['seed'])

# Logging Setup
run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f'runs/{env_config['id']}/{config['algorithm']}/seed_{config['seed']}/{run_id}')

# Environment Setup 
train_env = gym.make(**env_config)
obs_dim = train_env.observation_space.n
act_dim = train_env.action_space.n
total_env_steps = 0

eval_env = gym.make(**env_config)

if isinstance(train_env.observation_space, gym.spaces.Discrete):
    state_encoder = lambda obs: discrete_state_encoder(obs, obs_dim)
else:
    state_encoder = lambda x : x

# RL Setup
target_policy = PGPolicy(obs_dim=obs_dim,
                         hidden_layer_dim= config['actor_hidden_layer_dim'], 
                         act_dim=act_dim,          
                         num_hidden_layers=config['actor_num_hidden_layers'],
                         lr=config['actor_lr'])

agent = PGAgent(target_policy=target_policy,
                 discount_factor=config['discount_factor'])

# training loop
for i in range(config['train_iters']):
    print(f'Current iteration is iteration {i} / {config['train_iters']}')
    trajectories = collect_episodes(train_env, agent.target_policy.sample_action, config['episodes_per_train_iter'], state_encoder=state_encoder, seed=config['seed'])
    total_env_steps += sum([len(episode.obses) for episode in trajectories])
    
    metrics = agent.train(trajectories)
    for name, value in metrics.items():
        writer.add_scalar(name, value, total_env_steps)

    if i % config['eval_frequency'] == 0:
        eval_trajectories = collect_episodes(eval_env,
                                            agent.target_policy.get_action, 
                                            config['eval_episodes'], 
                                            state_encoder=state_encoder, 
                                            seed=config['seed'])
        mean_return = compute_mean_return(eval_trajectories)
        mean_episode_length = compute_mean_episode_length(eval_trajectories)
        writer.add_scalar('eval/mean_return', mean_return, total_env_steps)
        writer.add_scalar('eval/mean_episode_length', mean_episode_length, total_env_steps)


save_episodes = rng.choice([i for i in range(config['eval_episodes'])], size=3, replace=False)
eval_trajectories, recordings = collect_episodes(eval_env, agent.target_policy.get_action, config['eval_episodes'], state_encoder=state_encoder, seed=config['seed'], save_episodes=save_episodes)

# Compute success rate 
success_rate = compute_mean_return(eval_trajectories)

# Record Videos
videos = [torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0) for video in recordings]
for i, video in enumerate(videos): 
    writer.add_video(
        f"evaluation/video_{i}",
        video,
        global_step=total_env_steps,
        fps=4
    )

writer.add_hparams(
    hparam_dict=config,
    metric_dict={
        'eval/success_rate': success_rate,
    },

)

writer.close()
train_env.close()
eval_env.close
    


