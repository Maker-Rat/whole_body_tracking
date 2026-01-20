"""Replay motion on PM01 robot.

Usage:
    python scripts/replay_npz_pm01.py --motion_file motion.npz
    python scripts/replay_npz_pm01.py --registry_name your-org/wandb-registry-motions/motion_name
"""

import argparse
import os
import sys
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay converted motions on PM01.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to local motion .npz file.")
parser.add_argument("--registry_name", type=str, default=None, help="WandB registry name.")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
parser.add_argument("--show_markers", action="store_true", help="Show body position markers from motion file.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Direct path to assets to avoid importing whole_body_tracking package
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(SCRIPT_DIR, "..", "source", "whole_body_tracking", "whole_body_tracking", "assets")
ASSET_DIR = os.path.abspath(ASSET_DIR)


# Inline MotionLoader to avoid importing the full package
class MotionLoader:
    def __init__(self, motion_file: str, body_indexes, device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


# Inline PM01 config to avoid importing whole_body_tracking package

# Updated PD parameters and actuator groupings (to match latest pm01.py)
STIFFNESS_HIP_PITCH = 70
STIFFNESS_HIP_ROLL = 50
STIFFNESS_HIP_YAW = 50
STIFFNESS_KNEE_PITCH = 70
STIFFNESS_ANKLE_PITCH = 20
STIFFNESS_ANKLE_ROLL = 20
STIFFNESS_WAIST_YAW = 50
STIFFNESS_HEAD_YAW = 50
STIFFNESS_ARM_ALL = 50

DAMPING_HIP_PITCH = 7.0
DAMPING_HIP_ROLL = 5.0
DAMPING_HIP_YAW = 5.0
DAMPING_KNEE_PITCH = 7.0
DAMPING_ANKLE_PITCH = 0.2
DAMPING_ANKLE_ROLL = 0.2
DAMPING_WAIST_YAW = 5.0
DAMPING_HEAD_YAW = 5.0
DAMPING_ARM_ALL = 5.0

PM01_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/engineai_description/urdf/pm01.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.82),
        joint_pos={
            "j.*_hip_pitch_.*": -0.12,
            "j.*_knee_pitch_.*": 0.24,
            "j.*_ankle_pitch_.*": -0.12,
            "j.*_elbow_pitch_.*": 0.6,
            "j13_shoulder_pitch_l": 0.2,
            "j14_shoulder_roll_l": 0.2,
            "j18_shoulder_pitch_r": 0.2,
            "j19_shoulder_roll_r": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Hip pitch (high torque)
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=["j00_hip_pitch_l", "j06_hip_pitch_r"],
            stiffness=STIFFNESS_HIP_PITCH,
            damping=DAMPING_HIP_PITCH,
        ),
        # Hip roll
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=["j01_hip_roll_l", "j07_hip_roll_r"],
            stiffness=STIFFNESS_HIP_ROLL,
            damping=DAMPING_HIP_ROLL,
        ),
        # Hip yaw
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=["j02_hip_yaw_l", "j08_hip_yaw_r"],
            stiffness=STIFFNESS_HIP_YAW,
            damping=DAMPING_HIP_YAW,
        ),
        # Knee pitch (high torque)
        "knee_pitch": ImplicitActuatorCfg(
            joint_names_expr=["j03_knee_pitch_l", "j09_knee_pitch_r"],
            stiffness=STIFFNESS_KNEE_PITCH,
            damping=DAMPING_KNEE_PITCH,
        ),
        # Ankle pitch
        "ankle_pitch": ImplicitActuatorCfg(
            joint_names_expr=["j04_ankle_pitch_l", "j10_ankle_pitch_r"],
            stiffness=STIFFNESS_ANKLE_PITCH,
            damping=DAMPING_ANKLE_PITCH,
        ),
        # Ankle roll
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=["j05_ankle_roll_l", "j11_ankle_roll_r"],
            stiffness=STIFFNESS_ANKLE_ROLL,
            damping=DAMPING_ANKLE_ROLL,
        ),
        # Waist
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["j12_waist_yaw"],
            stiffness=STIFFNESS_WAIST_YAW,
            damping=DAMPING_WAIST_YAW,
        ),
        # Arms
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "j13_shoulder_pitch_l", "j14_shoulder_roll_l", "j15_shoulder_yaw_l", "j16_elbow_pitch_l", "j17_elbow_yaw_l",
                "j18_shoulder_pitch_r", "j19_shoulder_roll_r", "j20_shoulder_yaw_r", "j21_elbow_pitch_r", "j22_elbow_yaw_r"
            ],
            stiffness=STIFFNESS_ARM_ALL,
            damping=DAMPING_ARM_ALL,
        ),
        # Head
        "head": ImplicitActuatorCfg(
            joint_names_expr=["j23_head_yaw"],
            stiffness=STIFFNESS_HEAD_YAW,
            damping=DAMPING_HEAD_YAW,
        ),
    },
)


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = PM01_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# PM01 tracked body names (14 bodies) - same as in flat_env_cfg.py
PM01_TRACKED_BODY_NAMES = [
    "link_base",              # 0: pelvis/root
    "link_hip_roll_l",        # 1: left hip
    "link_knee_pitch_l",      # 2: left knee
    "link_ankle_roll_l",      # 3: left ankle
    "link_hip_roll_r",        # 4: right hip
    "link_knee_pitch_r",      # 5: right knee
    "link_ankle_roll_r",      # 6: right ankle
    "link_torso_yaw",         # 7: torso
    "link_shoulder_roll_l",   # 8: left shoulder
    "link_elbow_pitch_l",     # 9: left elbow
    "link_elbow_yaw_l",       # 10: left wrist/hand
    "link_shoulder_roll_r",   # 11: right shoulder
    "link_elbow_pitch_r",     # 12: right elbow
    "link_elbow_yaw_r",       # 13: right wrist/hand
]


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, motion_file: str, speed: float = 1.0, show_markers: bool = False):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    # Get body indices for the tracked bodies
    body_ids, body_names = robot.find_bodies(PM01_TRACKED_BODY_NAMES)
    print(f"Tracked body names: {body_names}")
    print(f"Tracked body indices: {body_ids}")

    motion = MotionLoader(
        motion_file,
        body_ids,  # Use actual robot body indices
        sim.device,
    )
    
    # Print motion data shape for debugging
    print(f"Motion file: {motion_file}")
    print(f"Motion body_pos_w shape (raw): {motion._body_pos_w.shape}")
    print(f"Motion total frames: {motion.time_step_total}")
    
    # Create markers if requested
    markers = None
    if show_markers:
        # Use frame markers to show orientation (XYZ axes)
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/BodyMarkers"
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # Smaller frames
        markers = VisualizationMarkers(marker_cfg)
        print(f"Created frame markers for {len(body_ids)} tracked bodies")
    
    time_steps_float = torch.zeros(scene.num_envs, dtype=torch.float32, device=sim.device)
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    while simulation_app.is_running():
        time_steps_float += speed
        time_steps = time_steps_float.long()

        reset_ids = time_steps >= motion.time_step_total
        time_steps_float[reset_ids] = 0
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        # Use first tracked body (link_base at body_ids[0]) for root
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        
        # Update markers if enabled
        if markers is not None:
            # Get body positions and orientations from motion file
            motion_body_pos = motion.body_pos_w[time_steps][0]  # Shape: [num_bodies, 3]
            motion_body_quat = motion.body_quat_w[time_steps][0]  # Shape: [num_bodies, 4]
            # Add env origin offset
            marker_positions = motion_body_pos + scene.env_origins[0]
            # Visualize with position and orientation
            markers.visualize(marker_positions, motion_body_quat)
        
        sim.render()
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
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

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    run_simulator(sim, scene, motion_file, args_cli.speed, args_cli.show_markers)


if __name__ == "__main__":
    main()
    simulation_app.close()
