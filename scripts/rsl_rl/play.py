"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--slow", type=float, default=0.0, help="Delay between steps in seconds for slow playback (e.g., 0.1).")
parser.add_argument("--log_joints", action="store_true", default=False, help="Log and plot PD targets vs actual joint angles.")
parser.add_argument("--log_steps", type=int, default=500, help="Number of steps to log for joint plotting.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

        if args_cli.motion_file is not None:
            env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=None,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    
    # Setup joint logging if requested
    joint_log = None
    if args_cli.log_joints:
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        joint_names = robot.joint_names
        num_joints = len(joint_names)
        print(f"[INFO] Logging {num_joints} joints for {args_cli.log_steps} steps")
        joint_log = {
            "joint_names": joint_names,
            "targets": [],  # PD targets (action * scale + default)
            "actual": [],   # Actual joint positions
            "actions": [],  # Raw actions from policy
        }
    
    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    import time
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            
            # Log joint data before stepping
            if joint_log is not None and timestep < args_cli.log_steps:
                base_env = env.unwrapped
                robot = base_env.scene["robot"]
                
                # Get actual joint positions
                actual_pos = robot.data.joint_pos[0].cpu().numpy()  # First env
                joint_log["actual"].append(actual_pos.copy())
                
                # Get the action and compute PD target
                # Target = action * scale + default_pos
                action_manager = base_env.action_manager
                # Get the processed action (after scaling)
                processed_action = action_manager.action
                if processed_action is not None:
                    # The JointPositionAction applies: target = action * scale + default
                    # We can get the target from the robot's joint_pos_target
                    target_pos = robot.data.joint_pos_target[0].cpu().numpy()
                    joint_log["targets"].append(target_pos.copy())
                else:
                    joint_log["targets"].append(actual_pos.copy())  # First step fallback
                
                joint_log["actions"].append(actions[0].cpu().numpy().copy())
            
            # env stepping
            obs, _, _, info = env.step(actions)
            
        if args_cli.slow > 0:
            time.sleep(args_cli.slow)
        
        timestep += 1
        
        # Check if we should stop logging and plot
        if joint_log is not None and timestep == args_cli.log_steps:
            print(f"[INFO] Finished logging {args_cli.log_steps} steps, generating plots...")
            plot_joint_tracking(joint_log, os.path.dirname(resume_path))
            print(f"[INFO] Plots saved to {os.path.dirname(resume_path)}")
        
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


def plot_joint_tracking(joint_log: dict, save_dir: str):
    """Plot PD targets vs actual joint positions."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    joint_names = joint_log["joint_names"]
    targets = np.array(joint_log["targets"])  # [T, num_joints]
    actual = np.array(joint_log["actual"])    # [T, num_joints]
    
    num_joints = len(joint_names)
    num_steps = targets.shape[0]
    time_axis = np.arange(num_steps)
    
    # Create subplots - 6 joints per figure
    joints_per_fig = 6
    num_figs = (num_joints + joints_per_fig - 1) // joints_per_fig
    
    for fig_idx in range(num_figs):
        start_j = fig_idx * joints_per_fig
        end_j = min(start_j + joints_per_fig, num_joints)
        num_subplots = end_j - start_j
        
        fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 2.5 * num_subplots), sharex=True)
        if num_subplots == 1:
            axes = [axes]
        
        for i, j in enumerate(range(start_j, end_j)):
            ax = axes[i]
            ax.plot(time_axis, targets[:, j], 'b-', label='PD Target', linewidth=1.5)
            ax.plot(time_axis, actual[:, j], 'r--', label='Actual', linewidth=1.5, alpha=0.8)
            ax.set_ylabel(f'{joint_names[j]}\n(rad)', fontsize=8)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Compute tracking error
            error = np.abs(targets[:, j] - actual[:, j])
            ax.set_title(f'{joint_names[j]} - Mean Error: {error.mean():.4f} rad, Max: {error.max():.4f} rad', fontsize=9)
        
        axes[-1].set_xlabel('Timestep')
        fig.suptitle(f'PD Target vs Actual Joint Positions (Joints {start_j}-{end_j-1})', fontsize=12)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'joint_tracking_{fig_idx}.png')
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_path}")
    
    # Also save a summary plot showing tracking errors for all joints
    fig, ax = plt.subplots(figsize=(14, 6))
    errors = np.abs(targets - actual)
    mean_errors = errors.mean(axis=0)
    max_errors = errors.max(axis=0)
    
    x = np.arange(num_joints)
    width = 0.35
    ax.bar(x - width/2, mean_errors, width, label='Mean Error', color='blue', alpha=0.7)
    ax.bar(x + width/2, max_errors, width, label='Max Error', color='red', alpha=0.7)
    ax.set_xlabel('Joint')
    ax.set_ylabel('Tracking Error (rad)')
    ax.set_title('Joint Tracking Error Summary')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_', '\n') for n in joint_names], rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'joint_tracking_summary.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
