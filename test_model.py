import os, sys
# Add the project root directory to Python path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)  # Go up one level from training/ to CubeNav-RL/
sys.path.insert(0, project_root)
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import TQC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
import argparse
from envs.CubeNavSimEnv import CubeSimulationEnv_class

INTERRUPTED = False
print_one = 0


import warnings
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
def evaluate_with_info(model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()

    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )

        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, info
    return mean_reward, std_reward, info


if __name__ == "__main__":
    '''-----------------RUN COMMANDS FOR venv10 and venv12. Use venv10!---------------'''
    # ~/Desktop/Thesis/CubeNav-RL/venv12/bin/mjpython training/SB3_train.py --train --run_id Test_Kris --trainer TQC
    # ~/Desktop/Thesis/python_envs/venv10/bin/mjpython training/SB3_train.py --train --run_id Test_Kris --trainer TQC
    # after activating the tensorboard venv: tensorboard --logdir TENSORBOARD
    train = False
    parser = argparse.ArgumentParser(description="Train a PPO agent to navigate a cube to a target sphere.")
    parser.add_argument("--num_rays", type=int, default=180, help="Number of LiDAR rays around the sphere.")
    parser.add_argument("--model_path", type=str, default="xml_models/simulation_model.xml",
                        help="Path to the MuJoCo model XML file.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
    parser.add_argument("--run_id", type=str, default="TEST_lidarscnas", help="Run ID for TensorBoard logging.")
    parser.add_argument("--trainer", type=str, default="TQC", help="Trainer to use (PPO, SAC, TD3, TQC).")
    args = parser.parse_args()
    name = args.run_id

    base_dir = os.path.dirname(__file__)
    name = os.path.join(base_dir, "models/", name)


    # Load the last trained model with the correct algorithm class
    if args.trainer == "PPO":
        model = PPO.load(f"{name}")
    elif args.trainer == "SAC":
        model = SAC.load(f"{name}")
    elif args.trainer == "TD3":
        model = TD3.load(f"{name}")
    elif args.trainer == "TQC":
        model = TQC.load(f"{name}")

    eval_env = DummyVecEnv([lambda: Monitor(CubeSimulationEnv_class(
        num_rays=args.num_rays,
        render_mode="human",
        model_path=args.model_path,
        training_mode=False
    ))])

    # Debug: Print observation space shapes
    # print(f"Current observation space shape: {eval_env.observation_space.shape}")
    # print("PATH:", os.path.join(base_dir, "TENSORBOARD/", f"{args.run_id}.pkl"))
    # Try to load normalization, but handle shape mismatch
    try:
        if os.path.exists(os.path.join(base_dir, "TENSORBOARD/", f"{args.run_id}.pkl")):
            vec_normalize = VecNormalize.load(os.path.join(base_dir, "TENSORBOARD/", f"{args.run_id}.pkl"), eval_env)
            print(f"Loaded normalization with shape: {vec_normalize.observation_space.shape}")
        else:
            vec_normalize = VecNormalize.load(os.path.join(base_dir, "TENSORBOARD/", "vecnormalize.pkl"), eval_env)
            print(f"Loaded default normalization with shape: {vec_normalize.observation_space.shape}")
        eval_env.training = False
        eval_env.norm_reward = False

    except AssertionError as e:
        print(f"Shape mismatch error: {e}")
        print("Using environment without normalization")
    print('----------ENVIRONMENT eval--------------\n', eval_env)


    # Evaluate the trained agent
    mean_reward, std_reward, info = evaluate_with_info(model, vec_normalize, n_eval_episodes=1,
                                              deterministic=True)  # , print_actions_means=True)
    eval_env.close()

    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

    t = np.arange(len(info["x_positions"]))

    fig, axs = plt.subplots(5, 1, figsize=(15, 8), sharex=True)

    # X position
    axs[0].plot(t, info["x_positions"], linewidth=2)
    axs[0].set_ylabel("x [m]")
    axs[0].set_title("X Position")
    axs[0].grid(True)

    # Y position
    axs[1].plot(t, info["y_positions"], linewidth=2)
    axs[1].set_ylabel("y [m]")
    axs[1].set_title("Y Position")
    axs[1].grid(True)

    # Theta
    axs[2].plot(t, info["theta_positions"], linewidth=2)
    axs[2].set_ylabel("theta [rad]")
    axs[2].set_xlabel("Step")
    axs[2].set_title("Orientation")
    axs[2].grid(True)

    # Linear velocity
    axs[3].plot(t, info["lin_vel"], linewidth=2)
    axs[3].set_ylabel("Lin Velocity [m/s]")
    axs[3].set_xlabel("Step")
    axs[3].set_title("Linear velocity")
    axs[3].grid(True)

    # Linear velocity
    axs[4].plot(t, info["ang_vel"], linewidth=2)
    axs[4].set_ylabel("Ang Velocity [m/s]")
    axs[4].set_xlabel("Step")
    axs[4].set_title("Angular velocity")
    axs[4].grid(True)

    plt.tight_layout()
    # plt.savefig("trajectory_plot_new_reward_velocities_new_network.png", dpi=300, bbox_inches="tight")
    plt.close()

