import argparse
import pickle

from cs285.agents import agents as agent_types
from cs285.envs import Pointmass

import os

import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import gym
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.replay_buffer import ReplayBuffer

from scripting_utils import make_logger, make_config
from run_hw5_explore import visualize

MAX_NVIDEO = 2


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    exploration_schedule = config.get("exploration_schedule", None)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    assert discrete, "DQN only supports discrete action spaces"

    agent_cls = agent_types[config["agent"]]
    agent = agent_cls(
        env.observation_space.shape,
        env.action_space.n,
        **config["agent_kwargs"],
    )

    ep_len = env.spec.max_episode_steps or env.max_episode_steps
    env_pointmass: Pointmass = env.unwrapped

    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=config["total_steps"])

    offline_rb = None
    dataset_path = os.path.join(args.dataset_dir, f"{config['dataset_name']}.pkl")
    if os.path.exists(dataset_path):
        with open(dataset_path, "rb") as f:
            offline_rb = pickle.load(f)
        print(f"Loaded offline dataset from {dataset_path}")
    else:
        print(f"[WARN] Offline dataset not found at {dataset_path}. Running online-only after warmup.")

    observation = env.reset()

    recent_observations = []

    num_offline_steps = config["offline_steps"]
    total_steps = config["total_steps"]

    for step in tqdm.trange(total_steps, dynamic_ncols=True):
        if step < num_offline_steps and offline_rb is not None:
            # Main training loop
            batch_np = offline_rb.sample(config["batch_size"])

            # Convert to PyTorch tensors
            batch = ptu.from_numpy(batch_np)

            update_info = agent.update(
                batch["observations"],
                batch["actions"],
                batch["rewards"] * (1 if config.get("use_reward", False) else 0),
                batch["next_observations"],
                batch["dones"],
                step,
            )
            epsilon = None

        else:
            if exploration_schedule is not None:
                epsilon = exploration_schedule.value(step - num_offline_steps)
            else:
                epsilon = 0.0

            action = agent.get_action(observation, epsilon)

            next_observation, reward, done, info = env.step(action)
            next_observation = np.asarray(next_observation)
            truncated = info.get("TimeLimit.truncated", False)

            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                done=done and not truncated,
                next_observation=next_observation,
            )
            recent_observations.append(observation)

            if done:
                logger.log_scalar(info["episode"]["r"], "train_return", step)
                logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
                observation = env.reset()
            else:
                observation = next_observation

            batch_np = replay_buffer.sample(config["batch_size"])
            batch = ptu.from_numpy(batch_np)
            update_info = agent.update(
                batch["observations"],
                batch["actions"],
                batch["rewards"] * (1 if config.get("use_reward", False) else 0),
                batch["next_observations"],
                batch["dones"],
                step,
            )

        if epsilon is not None:
            update_info["epsilon"] = epsilon

        if step % args.log_interval == 0:
            for k, v in update_info.items():
                logger.log_scalar(v, k, step)
            logger.flush()

        if step % args.eval_interval == 0:
            # Evaluate
            trajectories = utils.sample_n_trajectories(
                env,
                agent,
                args.num_eval_trajectories,
                ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

        if step % args.visualize_interval == 0 and len(recent_observations) > 0:
            observations_np = np.stack(recent_observations)
            recent_observations = []
            logger.log_figure(
                visualize(env_pointmass, agent, observations_np),
                "exploration_trajectories",
                step,
                "eval",
            )

    # Save the final dataset
    dataset_file = os.path.join(args.dataset_dir, f"{config['dataset_name']}.pkl")
    with open(dataset_file, "wb") as f:
        pickle.dump(replay_buffer, f)
        print("Saved dataset to", dataset_file)

    # Render final heatmap
    # fig = visualize(
    #     env_pointmass, agent, replay_buffer.observations[: config["total_steps"]]
    # )
    # fig.suptitle("State coverage")
    # filename = os.path.join("exploration", f"{config['log_name']}.png")
    # fig.savefig(filename)
    # print("Saved final heatmap to", filename)


banner = """
======================================================================
Exploration

Generating the dataset for the {env} environment using algorithm {alg}.
The results will be stored in {dataset_dir}.
======================================================================
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
    parser.add_argument("--visualize_interval", "-vi", type=int, default=1000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--log_interval", type=int, default=1)

    parser.add_argument("--use_reward", action="store_true")
    parser.add_argument("--dataset_dir", type=str, required=True)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw5_finetune_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    os.makedirs(args.dataset_dir, exist_ok=True)
    print(
        banner.format(
            env=config["env_name"], alg=config["agent"], dataset_dir=args.dataset_dir
        )
    )

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
