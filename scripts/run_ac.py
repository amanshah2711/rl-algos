
# Python Libraries
import argparse
import random
from datetime import datetime, timezone

# RL Implementations
from src.agents.ac_agent import ACAgent
from src.policies.pg_policy import PGPolicy
from src.critics.value_critic import DeepValueCritic
from src.utils.environment import collect_episodes, discrete_state_encoder
from src.utils.visualization import plot_value

# Environment 
import gymnasium as gym
import torch

# Logging Data
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True, help='Random seed used for reproducibility and verifiability of experiments')
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

env_config = {
    'id' : 'FrozenLake-v1',
    'is_slippery' : True,
    'map_name' : '4x4',
    'render_mode' : 'rgb_array'
}

config = {
    'algorithm' : 'Actor-Critic',
    'train_iters' : 1000,
    'episodes_per_train_iter' : 100,
    'actor_hidden_layer_dim' : 4,
    'actor_num_hidden_layers' : 2,
    'actor_lr' : 1e-3,
    'critic_hidden_layer_dim' : 16,
    'critic_num_hidden_layers' : 2,
    'critic_lr': 1e-3,
    'critic_gradient_steps_per_update' : 100,
    'discount_factor' : 0.99,
    'critic_updates_per_actor_update' : 5,
    'eval_episodes' : 200,
    'seed' : args.seed,
    **env_config
}


# Logging Setup
run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f'runs/{config['algorithm']}/seed_{config['seed']}/{run_id}')

# Environment Configuration
train_env = gym.make(**env_config)
obs_dim = train_env.observation_space.n
act_dim = train_env.action_space.n
total_env_steps = 0

if isinstance(train_env.observation_space, gym.spaces.Discrete):
    state_encoder = lambda obs: discrete_state_encoder(obs, obs_dim)
else:
    state_encoder = lambda x : x

# RL Setup
actor = PGPolicy(obs_dim=obs_dim,
                 hidden_layer_dim=config['actor_hidden_layer_dim'],
                 act_dim=act_dim, 
                 num_hidden_layers=config['actor_num_hidden_layers'],    
                 lr=config['actor_lr'])

critic = DeepValueCritic(obs_dim=obs_dim,
                         hidden_layer_dim=config['critic_hidden_layer_dim'],
                         num_hidden_layers=config['critic_num_hidden_layers'],      
                         discount_factor=config['discount_factor'],
                         lr=config['critic_lr'],
                         gradient_steps_per_update=config['critic_gradient_steps_per_update'])

agent = ACAgent(target_policy=actor,
                critic=critic,
                critic_updates_per_actor_update=config['critic_updates_per_actor_update'],
                discount_factor=config['discount_factor'])


# Training Loop
for i in range(config['train_iters']):
    print(f'Currently at training iteration {i + 1} / {config['train_iters']}')

    trajectories = collect_episodes(train_env, agent.target_policy.sample_action, config['episodes_per_train_iter'], state_encoder=state_encoder, seed=config['seed'])
    total_env_steps += sum([len(episode.obses) for episode in trajectories])

    metrics = agent.train(trajectories)
    for (key, value) in metrics.items():
        writer.add_scalar(key, value, total_env_steps)


# Final Evaluation
eval_env = gym.make(**env_config)

save_episodes = random.choices([i for i in range(config['eval_episodes'])], k = 3)
eval_trajectories, recordings = collect_episodes(eval_env, agent.target_policy.get_action, config['eval_episodes'], state_encoder=state_encoder, seed=config['seed'], save_episodes=save_episodes)

# Compute success rate 
success_rate = sum([sum(episode.rewards) for episode in eval_trajectories]) / config['eval_episodes']

# Record Videos
videos = [torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0) for video in recordings]
for i, video in enumerate(videos): 
    writer.add_video(
        f"evaluation/video_{i}",
        video,
        global_step=total_env_steps,
        fps=4
    )

# Visualizations (Value function heatmap)
writer.add_figure('critic/state_value_estimates', plot_value(value_func=critic.get_value, state_encoder=state_encoder), global_step=total_env_steps)

writer.add_hparams(
    hparam_dict=config,
    metric_dict={
        'eval/success_rate': success_rate,
    },

)
writer.close()
train_env.close()
eval_env.close()
    


