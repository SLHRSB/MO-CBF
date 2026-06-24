"""
Evaluate a trained PPO agent for the MO-CBF ACC environment.

Example:
    python scripts/evaluate_ppo.py \
        --model-path models/PPO_TL20250201_234020/10000.zip \
        --agent-name B2_TLSafe_CBF \
        --num-episodes 100
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import envs  # Registers PreScanEnv-v1

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter


def evaluate_agent(env, model, writer, num_episodes: int) -> dict:
    episode_rewards = []
    episode_mean_speeds = []
    episode_collision_rates = []
    episode_cbf_interventions = []

    global_step = 0

    for episode_idx in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False

        total_reward = 0.0
        episode_speeds = []

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            total_reward += float(reward)

            current_speed = getattr(env, "host_velocity", np.nan)
            current_acceleration = getattr(env, "host_acceleration", np.nan)
            current_jerk = getattr(env, "host_jerk", np.nan)
            current_engine_torque = getattr(env, "host_engine_torque", np.nan)
            current_engine_power = getattr(env, "host_engine_power", np.nan)

            writer.add_scalar("Test/Speed", current_speed, global_step)
            writer.add_scalar("Test/Acceleration", current_acceleration, global_step)
            writer.add_scalar("Test/Jerk", current_jerk, global_step)
            writer.add_scalar("Test/Engine_Torque", current_engine_torque, global_step)
            writer.add_scalar("Test/Engine_Power", current_engine_power, global_step)

            if not np.isnan(current_speed):
                episode_speeds.append(current_speed)

            global_step += 1

        terminal_type = getattr(env, "terminalType", None)

        mean_speed = float(np.mean(episode_speeds)) if episode_speeds else np.nan

        num_collision = getattr(env, "num_collision", 0)
        num_episode_env = getattr(env, "episode", episode_idx + 1)
        collision_rate = num_collision / num_episode_env if num_episode_env != 0 else num_collision

        num_cbf_intervention = getattr(env, "num_cbf_intervention", 0)

        writer.add_scalar("Test/Episode_Reward", total_reward, episode_idx)
        writer.add_scalar("Test/Episode_Mean_Speed", mean_speed, episode_idx)
        writer.add_scalar("Test/CBF_Interventions", num_cbf_intervention, episode_idx)
        writer.add_scalar("Test/Collision_Rate", collision_rate, episode_idx)

        episode_rewards.append(total_reward)
        episode_mean_speeds.append(mean_speed)
        episode_collision_rates.append(collision_rate)
        episode_cbf_interventions.append(num_cbf_intervention)

        print(
            f"Episode {episode_idx + 1}/{num_episodes} | "
            f"Reward={total_reward:.2f} | "
            f"Mean speed={mean_speed:.2f} | "
            f"Collision rate={collision_rate:.3f} | "
            f"CBF interventions={num_cbf_intervention} | "
            f"Terminal={terminal_type}"
        )

    results = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_speed": float(np.nanmean(episode_mean_speeds)),
        "mean_collision_rate": float(np.mean(episode_collision_rates)),
        "mean_cbf_interventions": float(np.mean(episode_cbf_interventions)),
        "num_episodes": num_episodes,
    }

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent.")

    parser.add_argument(
        "--env-id",
        type=str,
        default="PreScanEnv-v1",
        help="Gymnasium environment ID.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained PPO model zip file.",
    )

    parser.add_argument(
        "--agent-name",
        type=str,
        default="PPO_MOCBF",
        help="Name used for TensorBoard log folder.",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/test",
        help="TensorBoard test log directory.",
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.agent_name}_{timestamp}"

    log_path = Path(args.log_dir) / run_name
    log_path.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating agent: {args.agent_name}")
    print(f"Model path: {args.model_path}")
    print(f"TensorBoard log path: {log_path}")

    test_env = gym.make(args.env_id)
    test_env = Monitor(test_env)

    writer = SummaryWriter(str(log_path))

    try:
        model = PPO.load(args.model_path, env=test_env)
        results = evaluate_agent(
            env=test_env,
            model=model,
            writer=writer,
            num_episodes=args.num_episodes,
        )

        print("\n----- Evaluation Summary -----")
        print(f"Mean reward: {results['mean_reward']:.3f}")
        print(f"Std reward: {results['std_reward']:.3f}")
        print(f"Mean speed: {results['mean_speed']:.3f}")
        print(f"Mean collision rate: {results['mean_collision_rate']:.3f}")
        print(f"Mean CBF interventions: {results['mean_cbf_interventions']:.3f}")
        print(f"Number of episodes: {results['num_episodes']}")

    finally:
        writer.close()
        test_env.close()


if __name__ == "__main__":
    main()
