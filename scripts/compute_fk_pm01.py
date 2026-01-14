"""Compute forward kinematics for PM01 motion data using Isaac Lab.

This script takes a basic motion file (with only root pose and joint angles)
and computes world positions for all tracked bodies using FK.

Usage:
    python scripts/compute_fk_pm01.py --input_file scripts/tennis.npz --output_file scripts/tennis_fk.npz
"""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compute FK for PM01 motion data")
parser.add_argument("--input_file", type=str, required=True, help="Input npz file with basic motion data")
parser.add_argument("--output_file", type=str, required=True, help="Output npz file with full body data")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from whole_body_tracking.robots.pm01 import PM01_CFG

# Body names to track (must match flat_env_cfg.py)
BODY_NAMES = [
    "link_base",
    "link_hip_roll_l",
    "link_knee_pitch_l",
    "link_ankle_roll_l",
    "link_hip_roll_r",
    "link_knee_pitch_r",
    "link_ankle_roll_r",
    "link_torso_yaw",
    "link_shoulder_roll_l",
    "link_elbow_pitch_l",
    "link_elbow_yaw_l",
    "link_shoulder_roll_r",
    "link_elbow_pitch_r",
    "link_elbow_yaw_r",
]


@configclass
class FKSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = PM01_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    # Load input motion
    data = np.load(args_cli.input_file)
    fps = int(data["fps"])
    joint_pos = data["joint_pos"]
    joint_vel = data["joint_vel"]
    body_pos_w_in = data["body_pos_w"]
    body_quat_w_in = data["body_quat_w"]
    body_lin_vel_w_in = data["body_lin_vel_w"]
    body_ang_vel_w_in = data["body_ang_vel_w"]
    
    num_frames = joint_pos.shape[0]
    print(f"[INFO] Loaded {num_frames} frames at {fps} fps")
    
    # Extract root pose from body index 0
    root_pos = body_pos_w_in[:, 0, :]
    root_quat = body_quat_w_in[:, 0, :]
    root_lin_vel = body_lin_vel_w_in[:, 0, :]
    root_ang_vel = body_ang_vel_w_in[:, 0, :]
    
    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = SimulationContext(sim_cfg)
    
    scene_cfg = FKSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    robot: Articulation = scene["robot"]
    
    # Get body indices for tracked bodies
    body_indices = robot.find_bodies(BODY_NAMES, preserve_order=True)[0]
    print(f"[INFO] Tracking {len(body_indices)} bodies")
    print(f"[INFO] Body indices in robot: {body_indices}")
    print(f"[INFO] Total bodies in robot: {robot.num_bodies}")
    
    # Allocate output arrays - use robot's total body count so indices match
    num_bodies = robot.num_bodies
    body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_quat_w[:, :, 0] = 1.0  # Identity quaternion for unused bodies
    body_lin_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    
    print(f"[INFO] Computing FK for {num_frames} frames...")
    
    for i in range(num_frames):
        if i % 50 == 0:
            print(f"  Frame {i}/{num_frames}")
        
        # Set root state
        root_states = robot.data.default_root_state.clone()
        root_states[0, :3] = torch.tensor(root_pos[i], device=sim.device)
        root_states[0, 3:7] = torch.tensor(root_quat[i], device=sim.device)
        root_states[0, 7:10] = torch.tensor(root_lin_vel[i], device=sim.device)
        root_states[0, 10:13] = torch.tensor(root_ang_vel[i], device=sim.device)
        
        # Set joint state
        jp = torch.tensor(joint_pos[i:i+1], device=sim.device)
        jv = torch.tensor(joint_vel[i:i+1], device=sim.device)
        
        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(jp, jv)
        
        # Step to update FK
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())
        
        # Extract body poses - store at the actual body indices
        body_pos_w[i] = robot.data.body_pos_w[0].cpu().numpy()
        body_quat_w[i] = robot.data.body_quat_w[0].cpu().numpy()
        body_lin_vel_w[i] = robot.data.body_lin_vel_w[0].cpu().numpy()
        body_ang_vel_w[i] = robot.data.body_ang_vel_w[0].cpu().numpy()
    
    # Save output
    np.savez(
        args_cli.output_file,
        fps=fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    
    print(f"[INFO] Saved to {args_cli.output_file}")
    print(f"  body_pos_w: {body_pos_w.shape}")
    print(f"  body_quat_w: {body_quat_w.shape}")


if __name__ == "__main__":
    main()
    simulation_app.close()
