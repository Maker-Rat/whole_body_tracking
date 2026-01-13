"""Replay motion on PM01 robot and optionally save video (supports headless rendering).

Usage:
    # Replay from local npz file:
    python scripts/replay_npz_pm01.py --motion_file /path/to/motion.npz

    # Replay from WandB registry:
    python scripts/replay_npz_pm01.py --registry_name your-org/wandb-registry-motions/motion_name

    # Headless mode with video recording:
    python scripts/replay_npz_pm01.py --motion_file motion.npz --headless --video --video_dir ./videos

    # With offscreen rendering for GPU without display:
    python scripts/replay_npz_pm01.py --motion_file motion.npz --headless --video --video_dir ./videos --enable_cameras
"""

import argparse
import os
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay converted motions on PM01.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to local motion .npz file.")
parser.add_argument("--registry_name", type=str, default=None, help="WandB registry name (optional).")
parser.add_argument("--video", action="store_true", default=False, help="Record video of the motion replay.")
parser.add_argument("--video_dir", type=str, default="./videos", help="Directory to save videos.")
parser.add_argument("--video_length", type=int, default=None, help="Max video length in steps (default: full motion).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras for video recording
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from whole_body_tracking.robots.pm01 import PM01_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for replaying motions on PM01."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = PM01_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, motion_file: str, 
                   record_video: bool = False, video_dir: str = "./videos", video_length: int | None = None):
    """Run the simulation and optionally save video."""
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Video recording setup using Isaac Lab's approach
    frames = []
    max_steps = video_length if video_length is not None else motion.time_step_total
    step_count = 0

    print(f"[INFO]: Motion loaded with {motion.time_step_total} frames at {motion.fps} fps")
    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        print(f"[INFO]: Will record video to {video_dir}")

    while simulation_app.is_running():
        time_steps += 1
        step_count += 1
        
        reset_ids = time_steps >= motion.time_step_total
        if reset_ids.any():
            time_steps[reset_ids] = 0
            if record_video:
                # Stop after one full loop when recording
                break

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        # Update camera
        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        # Capture frame for video
        if record_video:
            try:
                # Use rep (replicator) for frame capture in headless mode
                import omni.replicator.core as rep
                from omni.isaac.core.utils.stage import get_current_stage
                
                # Capture from the main viewport
                rgb = rep.orchestrator._default_annotator_registry.get("rgb")
                if rgb:
                    data = rgb.get_data()
                    if data is not None:
                        frames.append(np.array(data))
            except Exception:
                # Alternative: try to get from simulation render
                pass
        
        if record_video and step_count >= max_steps:
            break

    # Save video
    if record_video and frames:
        _save_video(frames, video_dir, int(1.0 / sim_dt))
    elif record_video:
        print("[WARNING]: No frames captured. Try using --enable_cameras flag.")
        # Save joint positions as fallback for offline visualization
        _save_motion_data(motion, video_dir)


def _save_video(frames: list, video_dir: str, fps: int):
    """Save frames as video using imageio."""
    try:
        import imageio
        video_path = os.path.join(video_dir, "replay_pm01.mp4")
        print(f"[INFO]: Saving video with {len(frames)} frames to {video_path}")
        imageio.mimwrite(video_path, frames, fps=fps)
        print(f"[INFO]: Video saved successfully!")
    except ImportError:
        print("[WARNING]: imageio not installed. Install with: pip install imageio imageio-ffmpeg")
        # Save as individual frames instead
        frames_dir = os.path.join(video_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            np.save(os.path.join(frames_dir, f"frame_{i:05d}.npy"), frame)
        print(f"[INFO]: Saved {len(frames)} frames to {frames_dir}/")


def _save_motion_data(motion, video_dir: str):
    """Save motion data for offline visualization."""
    data_path = os.path.join(video_dir, "motion_data.npz")
    np.savez(
        data_path,
        joint_pos=motion.joint_pos.cpu().numpy(),
        body_pos_w=motion._body_pos_w.cpu().numpy(),
        body_quat_w=motion._body_quat_w.cpu().numpy(),
    )
    print(f"[INFO]: Motion data saved to {data_path} for offline visualization")


def main():
    # Determine motion file source
    if args_cli.motion_file:
        motion_file = args_cli.motion_file
    elif args_cli.registry_name:
        import pathlib
        import wandb
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    else:
        raise ValueError("Must provide either --motion_file or --registry_name")

    print(f"[INFO]: Loading motion from {motion_file}")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    run_simulator(sim, scene, motion_file, args_cli.video, args_cli.video_dir, args_cli.video_length)


if __name__ == "__main__":
    main()
    simulation_app.close()
