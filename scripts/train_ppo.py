"""
Train PPO agent for the MO-CBF ACC environment.

This script supports:
- training from scratch
- transfer learning from a pretrained model
- TensorBoard logging
- periodic model saving
- custom training callbacks
"""

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import envs  # Registers PreScanEnv-v1

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

from callbacks import (
    SpeedLoggerCallback,
    CollisionLoggerCallback,
    EmgBrLoggerCallback,
    RLThrottleActionCallback,
    SafeThrottleActionCallback,
    RLBrakeActionCallback,
    SafeBrakeActionCallback,
    CBFInterventionLoggerCallback,
)


def entropy_coefficient_schedule(
    epoch: int,
    total_epochs: int,
    start_ent_coef: float = 0.1,
    end_ent_coef: float = 0.01,
) -> float:
    """
    Linearly decay entropy coefficient during training.
    """

    progress = epoch / total_epochs
    return start_ent_coef - (start_ent_coef - end_ent_coef) * progress


def create_callbacks() -> CallbackList:
    """
    Create TensorBoard/custom callbacks used during training.
    """

    callbacks = [
        SpeedLoggerCallback(),
        CollisionLoggerCallback(),
        EmgBrLoggerCallback(),
        RLThrottleActionCallback(),
        SafeThrottleActionCallback(),
        RLBrakeActionCallback(),
        SafeBrakeActionCallback(),
        CBFInterventionLoggerCallback(),
    ]

    return CallbackList(callbacks)


def train(args: argparse.Namespace) -> None:
    """
    Train PPO model.
    """

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.experiment_name}_{timestamp}"

    models_dir = Path(args.models_dir) / run_name
    log_dir = Path(args.log_dir)

    models_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment name: {run_name}")
    print(f"Models directory: {models_dir}")
    print(f"TensorBoard log directory: {log_dir}")

    env = gym.make(args.env_id)

    callbacks = create_callbacks()

    if args.train_from_scratch:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            clip_range=args.clip_range,
            clip_range_vf=args.clip_range_vf,
            ent_coef=args.start_ent_coef,
            verbose=args.verbose,
            tensorboard_log=str(log_dir),
        )

        print("----- PPO MODEL CREATED FROM SCRATCH -----")

    else:
        if args.pretrained_model is None:
            raise ValueError(
                "You selected transfer learning, but no pretrained model was provided. "
                "Use --pretrained-model path/to/model.zip"
            )

        model = PPO.load(
            args.pretrained_model,
            env=env,
            tensorboard_log=str(log_dir),
            verbose=args.verbose,
        )

        print(f"----- PPO MODEL LOADED FROM {args.pretrained_model} -----")

    for epoch in range(1, args.total_epochs + 1):
        reset_num_timesteps = epoch == 1

        model.learn(
            total_timesteps=args.timesteps_per_epoch,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=run_name,
            callback=callbacks,
        )

        new_ent_coef = entropy_coefficient_schedule(
            epoch=epoch,
            total_epochs=args.total_epochs,
            start_ent_coef=args.start_ent_coef,
            end_ent_coef=args.end_ent_coef,
        )

        model.ent_coef = new_ent_coef

        save_path = models_dir / f"{args.timesteps_per_epoch * epoch}.zip"
        model.save(str(save_path))

        print(
            f"Epoch {epoch}/{args.total_epochs} completed | "
            f"Model saved to {save_path} | "
            f"ent_coef={new_ent_coef:.4f}"
        )

    env.close()
    print("----- TRAINING COMPLETED -----")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Train PPO agent with MO-CBF environment.")

    parser.add_argument(
        "--env-id",
        type=str,
        default="PreScanEnv-v1",
        help="Gymnasium environment ID.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="PPO_MOCBF",
        help="Name of the training experiment.",
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory for saving trained models.",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/train",
        help="Directory for TensorBoard logs.",
    )

    parser.add_argument(
        "--train-from-scratch",
        action="store_true",
        help="Train PPO from scratch.",
    )

    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="Path to pretrained PPO model for transfer learning.",
    )

    parser.add_argument(
        "--timesteps-per-epoch",
        type=int,
        default=100,
        help="Number of timesteps per training epoch.",
    )

    parser.add_argument(
        "--total-epochs",
        type=int,
        default=100,
        help="Total number of training epochs.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate.",
    )

    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO policy clipping range.",
    )

    parser.add_argument(
        "--clip-range-vf",
        type=float,
        default=0.2,
        help="PPO value-function clipping range.",
    )

    parser.add_argument(
        "--start-ent-coef",
        type=float,
        default=0.1,
        help="Initial entropy coefficient.",
    )

    parser.add_argument(
        "--end-ent-coef",
        type=float,
        default=0.01,
        help="Final entropy coefficient.",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Stable-Baselines3 verbosity level.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
