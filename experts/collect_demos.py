import argparse
from pathlib import Path
from typing import Any, Dict
import random

import d4rl
import gym
import numpy as np
import torch
import tqdm
from stable_baselines3 import SAC

from garage.utils.common import PROJECT_ROOT, ENV_ABBRV_TO_FULL
from garage.utils.gym_wrappers import TremblingHandWrapper


def get_trajectory_data_dict() -> Dict[str, Any]:
    data = dict(
        observations=[],
        next_observations=[],
        actions=[],
        rewards=[],
        terminals=[],
        timeouts=[],
        logprobs=[],
        qpos=[],
        qvel=[],
        seed=[],
    )
    return data


def rollout(
    env_name: str, expert_root_dir: str, max_path: int = 1000, num_data: int = 100_000, tremble: float = 0.0
) -> Dict[str, Any]:
    expert_ckpt_path = f"{expert_root_dir}/best_model"
    model = SAC.load(expert_ckpt_path)

    env = gym.make(env_name)
    env = TremblingHandWrapper(env, tremble)
    data = get_trajectory_data_dict()
    traj_data = get_trajectory_data_dict()

    total_rew = []
    _returns = 0
    t = 0
    done = False
    seed = 0
    env.seed(seed)
    s = env.reset()
    while len(data["rewards"]) < num_data:
        a = model.predict(s, deterministic=True)
        if isinstance(a, tuple):
            a = a[0]
        qpos, qvel = env.sim.data.qpos.ravel().copy(), env.sim.data.qvel.ravel().copy()

        ns, rew, done, infos = env.step(a)
        _returns += rew

        t += 1
        timeout = False
        terminal = False
        if t == max_path:
            timeout = True
        elif done:
            terminal = True

        traj_data["observations"].append(s)
        traj_data["actions"].append(a)
        traj_data["next_observations"].append(ns)
        traj_data["rewards"].append(rew)
        traj_data["terminals"].append(terminal)
        traj_data["timeouts"].append(timeout)
        traj_data["qpos"].append(qpos)
        traj_data["qvel"].append(qvel)
        traj_data["seed"].append(seed)

        s = ns
        if terminal or timeout:
            print(
                "Finished trajectory. Len=%d, Returns=%f. Progress:%d/%d"
                % (t, _returns, len(data["rewards"]), num_data)
            )
            if timeout:
                total_rew.append(_returns)
                for k in data:
                    data[k].extend(traj_data[k])
            t = 0
            _returns = 0
            seed += 1
            env.seed(seed)
            s = env.reset()
            # s = env.reset(seed=seed)
            traj_data = get_trajectory_data_dict()

    new_data = dict(
        observations=np.array(data["observations"]).astype(np.float64),
        actions=np.array(data["actions"]).astype(np.float32),
        next_observations=np.array(data["next_observations"]).astype(np.float64),
        rewards=np.array(data["rewards"]).astype(np.float64),
        terminals=np.array(data["terminals"]).astype(bool),
        timeouts=np.array(data["timeouts"]).astype(bool),
        qpos=np.array(data["qpos"]).astype(np.float64),
        qvel=np.array(data["qvel"]).astype(np.float64),
        seed=np.array(data["seed"]).astype(np.uint8),
    )
    print(f"{np.mean(total_rew)=}, {np.std(total_rew)=}")

    for k in new_data:
        new_data[k] = new_data[k][:num_data]
    return new_data

def parse_antmaze_demos(env_name, drop_prop=0.0, num_trajs=0):
    env = gym.make(env_name)
    dataset = env.get_dataset()
    q_dataset = d4rl.qlearning_dataset(env)
    dataset, q_dataset = filter_successful_trajs(dataset, q_dataset, drop_prop, num_trajs)

    term = np.argwhere(np.logical_or(dataset["timeouts"] > 0, dataset["terminals"] > 0))
    start = 0
    expert_ranges = []
    for i in range(len(term)):
        expert_ranges.append([start, term[i][0] + 1])
        start = term[i][0] + 1

    curr_obs_pt = 0
    curr_obs_indices = []
    next_obs_pt = 0
    next_obs_indices = []
    for i in tqdm.tqdm(range(len(q_dataset["observations"]))):
        while (
            np.linalg.norm(
                q_dataset["observations"][i] - dataset["observations"][curr_obs_pt]
            )
            > 1e-10
        ):
            curr_obs_pt += 1
        curr_obs_indices.append(curr_obs_pt)
        while (
            np.linalg.norm(
                q_dataset["next_observations"][i] - dataset["observations"][next_obs_pt]
            )
            > 1e-10
        ):
            next_obs_pt += 1
        next_obs_indices.append(next_obs_pt)
    curr_obs_indices = np.array(curr_obs_indices)
    goals_flat = dataset["infos/goal"][curr_obs_indices]
    qpos_flat = dataset["infos/qpos"][curr_obs_indices]
    qvel_flat = dataset["infos/qvel"][curr_obs_indices]
    next_obs_indices = np.array(next_obs_indices)
    next_goals_flat = dataset["infos/goal"][next_obs_indices]

    observations = np.concatenate([q_dataset["observations"], goals_flat], axis=1)
    next_observations = np.concatenate(
        [q_dataset["next_observations"], next_goals_flat], axis=1
    )
    rewards = q_dataset["rewards"]
    terminals = q_dataset["terminals"]
    actions = q_dataset["actions"]

    new_dataset = {
        "observations": observations,
        "actions": actions,
        "next_observations": next_observations,
        "rewards": rewards,
        "terminals": terminals,
    }

    term = np.argwhere(terminals.flatten() > 0)
    start = 0
    qpos, qvel, goals = [], [], []
    for i in range(len(term)):
        qpos.append(qpos_flat[start : term[i][0] + 1])
        qvel.append(qvel_flat[start : term[i][0] + 1])
        goals.append(goals_flat[start : term[i][0] + 1])
        start = term[i][0] + 1
    qpos = np.array(qpos, dtype=object)
    qvel = np.array(qvel, dtype=object)
    goals = np.array(goals, dtype=object)

    obs_goal_cat = np.concatenate(
        [dataset["observations"], dataset["infos/goal"]], axis=1
    )
    expert_reset_states = np.array(
        [
            obs_goal_cat[expert_ranges[i][0] : expert_ranges[i][1]]
            for i in range(len(expert_ranges))
        ],
        dtype=object,
    )

    # antmaze-large trajectories are not all the same length, so we
    # extend the shorter ones to match the longest by duplicating
    # each state consecutively until the length matches the longest
    # i.e. [1, 2, 3] -> [1, 1, 2, 2, 3, 3]
    expert_reset_states_square = []
    max_length = max(len(row) for row in expert_reset_states)
    for i in range(len(expert_reset_states)):
        if len(expert_reset_states[i]) == max_length:
            expert_reset_states_square.append(expert_reset_states[i])
            continue
        repeat_times = max_length // len(expert_reset_states[i])
        remainder = max_length % len(expert_reset_states[i])
        if remainder > 0:
            expert_reset_states[i] = np.concatenate(
                (
                    np.repeat(
                        expert_reset_states[i][:remainder],
                        repeat_times + 1,
                        axis=0,
                    ),
                    np.repeat(expert_reset_states[i][remainder:], repeat_times, axis=0),
                )
            )
        else:
            expert_reset_states[i] = np.repeat(
                expert_reset_states[i], repeat_times, axis=0
            )
        assert len(expert_reset_states[i]) == max_length
        expert_reset_states_square.append(expert_reset_states[i])

    expert_reset_states = np.array(expert_reset_states_square)
    expert_sa_pairs = torch.cat(
        (torch.from_numpy(observations), torch.from_numpy(actions)), dim=1
    )
    return {
        **new_dataset,
        "expert_sa_pairs": expert_sa_pairs,
        "expert_reset_states": expert_reset_states,
        "qpos": qpos,
        "qvel": qvel,
        "goals": goals,
    }


def filter_successful_trajs(dataset, q_dataset, drop_prop, num_trajs=0):
    succesful_traj_ranges = get_successful_trajs(dataset)
    if drop_prop > 0.0:
        num_dropped_trajs = int(drop_prop * succesful_traj_ranges.shape[0])
    elif num_trajs > 0:
        num_dropped_trajs = succesful_traj_ranges.shape[0] - num_trajs
    else:
        num_dropped_trajs = 0 
    
    if num_dropped_trajs > 0:
        drop_idxs = random.sample(range(succesful_traj_ranges.shape[0]), num_dropped_trajs) 
        drop_traj_ranges = succesful_traj_ranges[drop_idxs]
        dataset = remove_trajs(dataset, drop_traj_ranges)
        
        in_dataset = np.isin(q_dataset["observations"], dataset["observations"])
        indices = np.where(np.all(in_dataset, axis=1))[0]
        for key in q_dataset.keys():
            q_dataset[key] = q_dataset[key][indices]
            
        in_dataset = np.isin(q_dataset["next_observations"], dataset["observations"])
        indices = np.where(np.all(in_dataset, axis=1))[0]
        for key in q_dataset.keys():
            q_dataset[key] = q_dataset[key][indices]
    
    return dataset, q_dataset

def get_successful_trajs(dataset):
    timeout_idxs = np.where(dataset["timeouts"] == True)[0]

    success_idxs = np.where(dataset["rewards"] == 1.0)[0]

    success_ranges = []

    for idx in success_idxs:
        smaller_timeouts = timeout_idxs[timeout_idxs < idx]
        if len(smaller_timeouts) > 0:
            success_ranges.append([smaller_timeouts[-1], idx])

    np_success_ranges = np.array(success_ranges)

    unique_first_elements = np.unique(np_success_ranges[:, 0])

    filtered_pairs = []

    for first in unique_first_elements:
        subset = np_success_ranges[np_success_ranges[:, 0] == first]
        max_pair = subset[np.argmax(subset[:, 1])]
        filtered_pairs.append(max_pair)

    return np.array(filtered_pairs)


def remove_trajs(dataset, traj_ranges):
    indices_to_remove = set()
    for pair in traj_ranges:
        indices_to_remove.update(range(pair[0], pair[1] + 1))
    for key in dataset.keys():
        dataset[key] = np.delete(dataset[key], list(indices_to_remove), axis=0)
    
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect expert demonstrations.")
    parser.add_argument(
        "--env",
        choices=ENV_ABBRV_TO_FULL.keys(),
        required=True,
    )
    parser.add_argument(
        "--tremble",
        type=float,
        required=False,
        default=0.0,
    )
    parser.add_argument(
        "--maze-drop-prop",
        type=float,
        required=False,
        default=0.0
    )
    parser.add_argument(
        "--maze-keep-k-trajs",
        type=int,
        required=False,
        default=0
    )
    args = parser.parse_args()

    env_name_full = ENV_ABBRV_TO_FULL[args.env]
    expert_root_dir = Path(PROJECT_ROOT, "experts", env_name_full)
    expert_root_dir.mkdir(parents=True, exist_ok=True)
    if "maze" in env_name_full:
        data = parse_antmaze_demos(env_name_full, args.maze_drop_prop, args.maze_keep_k_trajs)
        if args.maze_keep_k_trajs > 0:
            data_save_path = Path(expert_root_dir, f"{env_name_full}_{args.maze_keep_k_trajs}_demos")
        else:
            data_save_path = Path(expert_root_dir, f"{env_name_full}_{args.maze_drop_prop}_demos")
    else:
        data = rollout(env_name_full, expert_root_dir=expert_root_dir, tremble=args.tremble)
        data_save_path = Path(expert_root_dir, f"{env_name_full}_{args.tremble}_demos")
    np.savez(data_save_path, **data)
    print(f"Saved demonstrations to {data_save_path}.npz")