from dataclasses import dataclass
from numpy.typing import NDArray



@dataclass
class Trajectory:
    obs : NDArray
    acs : NDArray
    next_obs : NDArray
    rewards : NDArray
    terminals : NDArray


def collect_episodes(env, collect_policy, num_episodes) -> list[Trajectory]:
    trajectories = []
    for _ in range(num_episodes):
        episode_over = False
        ob, info = env.reset()
        obs, acs, next_obs, rewards, terminals = [], [], [], [] ,[]
        while not episode_over:
            obs.append(ob)

            ac = collect_policy.sample(ob)
            ob, reward, terminated, truncated, info = env.step(ac)

            acs.append(ac)
            rewards.append(reward)
            next_obs.append(ob)
            terminals.append(terminated)

            episode_over = episode_over or terminated
        trajectories.append(Trajectory(obs, acs, next_obs, rewards, terminals))
    return trajectories
