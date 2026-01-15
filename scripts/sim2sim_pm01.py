"""Sim2Sim script for PM01 motion tracking policy in MuJoCo.

This script runs a trained motion-tracking policy from Isaac Lab in MuJoCo.

Usage:
    python scripts/sim2sim_pm01.py --policy_path logs/rsl_rl/pm01_flat/<run>/model_xxx.pt --motion_file scripts/walk_fk.npz
    python scripts/sim2sim_pm01.py --onnx_path logs/rsl_rl/pm01_flat/<run>/<run>.onnx --motion_file scripts/walk_fk.npz
"""

import argparse
import math
import numpy as np
import torch
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
from collections import deque

# ============== Configuration ==============

class PM01Config:
    """PM01 robot and simulation configuration."""
    
    # Simulation
    dt = 0.001  # MuJoCo timestep (1000 Hz)
    decimation = 20  # Policy runs at 50 Hz (1000/20)
    sim_duration = 60.0  # seconds
    
    # Robot
    num_actions = 24
    mujoco_model_path = None  # Set via command line or default
    
    # PD gains (from pm01.py)
    ARMATURE_HIGH_TORQUE = 0.045325
    ARMATURE_LOW_TORQUE = 0.039175
    NATURAL_FREQ = 10 * 2.0 * np.pi  # 10 Hz
    DAMPING_RATIO = 2.0
    
    # Stiffness and damping
    kp_high = ARMATURE_HIGH_TORQUE * NATURAL_FREQ**2  # ~179
    kp_low = ARMATURE_LOW_TORQUE * NATURAL_FREQ**2    # ~155
    kd_high = 2.0 * DAMPING_RATIO * ARMATURE_HIGH_TORQUE * NATURAL_FREQ  # ~11.4
    kd_low = 2.0 * DAMPING_RATIO * ARMATURE_LOW_TORQUE * NATURAL_FREQ    # ~9.8
    
    # Torque limits
    tau_limit_high = 164.0  # Nm
    tau_limit_low = 61.0    # Nm
    
    # Action scales (from training config) - in Isaac Lab order
    action_scales = {
        'j00_hip_pitch_l': 0.22913229617061878,
        'j01_hip_roll_l': 0.22913229617061878,
        'j03_knee_pitch_l': 0.22913229617061878,
        'j06_hip_pitch_r': 0.22913229617061878,
        'j07_hip_roll_r': 0.22913229617061878,
        'j09_knee_pitch_r': 0.22913229617061878,
        'j02_hip_yaw_l': 0.08405714023137756,
        'j08_hip_yaw_r': 0.08405714023137756,
        'j04_ankle_pitch_l': 0.08405714023137756,
        'j05_ankle_roll_l': 0.08405714023137756,
        'j10_ankle_pitch_r': 0.08405714023137756,
        'j11_ankle_roll_r': 0.08405714023137756,
        'j12_waist_yaw': 0.08405714023137756,
        'j13_shoulder_pitch_l': 0.08405714023137756,
        'j14_shoulder_roll_l': 0.08405714023137756,
        'j15_shoulder_yaw_l': 0.08405714023137756,
        'j16_elbow_pitch_l': 0.08405714023137756,
        'j17_elbow_yaw_l': 0.08405714023137756,
        'j18_shoulder_pitch_r': 0.08405714023137756,
        'j19_shoulder_roll_r': 0.08405714023137756,
        'j20_shoulder_yaw_r': 0.08405714023137756,
        'j21_elbow_pitch_r': 0.08405714023137756,
        'j22_elbow_yaw_r': 0.08405714023137756,
        'j23_head_yaw': 0.08405714023137756,
    }
    
    # Default joint positions (from training config) - in Isaac Lab order
    default_joint_pos = {
        'j00_hip_pitch_l': -0.12,
        'j01_hip_roll_l': 0.0,
        'j02_hip_yaw_l': 0.0,
        'j03_knee_pitch_l': 0.24,
        'j04_ankle_pitch_l': -0.12,
        'j05_ankle_roll_l': 0.0,
        'j06_hip_pitch_r': -0.12,
        'j07_hip_roll_r': 0.0,
        'j08_hip_yaw_r': 0.0,
        'j09_knee_pitch_r': 0.24,
        'j10_ankle_pitch_r': -0.12,
        'j11_ankle_roll_r': 0.0,
        'j12_waist_yaw': 0.0,
        'j13_shoulder_pitch_l': 0.2,
        'j14_shoulder_roll_l': 0.2,
        'j15_shoulder_yaw_l': 0.0,
        'j16_elbow_pitch_l': 0.6,
        'j17_elbow_yaw_l': 0.0,
        'j18_shoulder_pitch_r': 0.2,
        'j19_shoulder_roll_r': -0.2,
        'j20_shoulder_yaw_r': 0.0,
        'j21_elbow_pitch_r': 0.6,
        'j22_elbow_yaw_r': 0.0,
        'j23_head_yaw': 0.0,
    }
    
    # MuJoCo joint names (in URDF order: j00, j01, j02, ..., j23)
    mujoco_joint_names = [
        'j00_hip_pitch_l', 'j01_hip_roll_l', 'j02_hip_yaw_l', 'j03_knee_pitch_l',
        'j04_ankle_pitch_l', 'j05_ankle_roll_l', 'j06_hip_pitch_r', 'j07_hip_roll_r',
        'j08_hip_yaw_r', 'j09_knee_pitch_r', 'j10_ankle_pitch_r', 'j11_ankle_roll_r',
        'j12_waist_yaw', 'j13_shoulder_pitch_l', 'j14_shoulder_roll_l', 'j15_shoulder_yaw_l',
        'j16_elbow_pitch_l', 'j17_elbow_yaw_l', 'j18_shoulder_pitch_r', 'j19_shoulder_roll_r',
        'j20_shoulder_yaw_r', 'j21_elbow_pitch_r', 'j22_elbow_yaw_r', 'j23_head_yaw'
    ]
    
    # Isaac Lab joint names in Isaac Lab's internal order (groups by joint type)
    # This is the order Isaac Lab uses after importing URDF
    isaac_joint_names = [
        'j00_hip_pitch_l',      # 0
        'j06_hip_pitch_r',      # 1
        'j12_waist_yaw',        # 2
        'j01_hip_roll_l',       # 3
        'j07_hip_roll_r',       # 4
        'j13_shoulder_pitch_l', # 5
        'j18_shoulder_pitch_r', # 6
        'j23_head_yaw',         # 7
        'j02_hip_yaw_l',        # 8
        'j08_hip_yaw_r',        # 9
        'j14_shoulder_roll_l',  # 10
        'j19_shoulder_roll_r',  # 11
        'j03_knee_pitch_l',     # 12
        'j09_knee_pitch_r',     # 13
        'j15_shoulder_yaw_l',   # 14
        'j20_shoulder_yaw_r',   # 15
        'j04_ankle_pitch_l',    # 16
        'j10_ankle_pitch_r',    # 17
        'j16_elbow_pitch_l',    # 18
        'j21_elbow_pitch_r',    # 19
        'j05_ankle_roll_l',     # 20
        'j11_ankle_roll_r',     # 21
        'j17_elbow_yaw_l',      # 22
        'j22_elbow_yaw_r',      # 23
    ]
    
    # Mapping from MuJoCo order (j00, j01, ..., j23) to Isaac Lab order
    # MUJOCO_TO_ISAAC[mujoco_idx] = isaac_idx
    # e.g., j00 is at MuJoCo idx 0 and Isaac idx 0
    # e.g., j01 is at MuJoCo idx 1 and Isaac idx 3
    MUJOCO_TO_ISAAC = [
        0,   # j00 -> Isaac idx 0
        3,   # j01 -> Isaac idx 3
        8,   # j02 -> Isaac idx 8
        12,  # j03 -> Isaac idx 12
        16,  # j04 -> Isaac idx 16
        20,  # j05 -> Isaac idx 20
        1,   # j06 -> Isaac idx 1
        4,   # j07 -> Isaac idx 4
        9,   # j08 -> Isaac idx 9
        13,  # j09 -> Isaac idx 13
        17,  # j10 -> Isaac idx 17
        21,  # j11 -> Isaac idx 21
        2,   # j12 -> Isaac idx 2
        5,   # j13 -> Isaac idx 5
        10,  # j14 -> Isaac idx 10
        14,  # j15 -> Isaac idx 14
        18,  # j16 -> Isaac idx 18
        22,  # j17 -> Isaac idx 22
        6,   # j18 -> Isaac idx 6
        11,  # j19 -> Isaac idx 11
        15,  # j20 -> Isaac idx 15
        19,  # j21 -> Isaac idx 19
        23,  # j22 -> Isaac idx 23
        7,   # j23 -> Isaac idx 7
    ]
    
    # Inverse mapping: Isaac Lab order to MuJoCo order
    # ISAAC_TO_MUJOCO[isaac_idx] = mujoco_idx
    ISAAC_TO_MUJOCO = [
        0,   # Isaac idx 0 (j00) -> MuJoCo idx 0
        6,   # Isaac idx 1 (j06) -> MuJoCo idx 6
        12,  # Isaac idx 2 (j12) -> MuJoCo idx 12
        1,   # Isaac idx 3 (j01) -> MuJoCo idx 1
        7,   # Isaac idx 4 (j07) -> MuJoCo idx 7
        13,  # Isaac idx 5 (j13) -> MuJoCo idx 13
        18,  # Isaac idx 6 (j18) -> MuJoCo idx 18
        23,  # Isaac idx 7 (j23) -> MuJoCo idx 23
        2,   # Isaac idx 8 (j02) -> MuJoCo idx 2
        8,   # Isaac idx 9 (j08) -> MuJoCo idx 8
        14,  # Isaac idx 10 (j14) -> MuJoCo idx 14
        19,  # Isaac idx 11 (j19) -> MuJoCo idx 19
        3,   # Isaac idx 12 (j03) -> MuJoCo idx 3
        9,   # Isaac idx 13 (j09) -> MuJoCo idx 9
        15,  # Isaac idx 14 (j15) -> MuJoCo idx 15
        20,  # Isaac idx 15 (j20) -> MuJoCo idx 20
        4,   # Isaac idx 16 (j04) -> MuJoCo idx 4
        10,  # Isaac idx 17 (j10) -> MuJoCo idx 10
        16,  # Isaac idx 18 (j16) -> MuJoCo idx 16
        21,  # Isaac idx 19 (j21) -> MuJoCo idx 21
        5,   # Isaac idx 20 (j05) -> MuJoCo idx 5
        11,  # Isaac idx 21 (j11) -> MuJoCo idx 11
        17,  # Isaac idx 22 (j17) -> MuJoCo idx 17
        22,  # Isaac idx 23 (j22) -> MuJoCo idx 22
    ]
    
    # High torque joints (by joint name)
    high_torque_joints = [
        'j00_hip_pitch_l', 'j01_hip_roll_l', 'j03_knee_pitch_l',
        'j06_hip_pitch_r', 'j07_hip_roll_r', 'j09_knee_pitch_r'
    ]


class MotionLoader:
    """Load motion data from npz file."""
    
    def __init__(self, motion_file: str):
        data = np.load(motion_file)
        self.fps = float(data['fps'])
        self.joint_pos = data['joint_pos']  # [T, num_joints]
        self.joint_vel = data['joint_vel']  # [T, num_joints]
        self.body_pos_w = data['body_pos_w']  # [T, num_bodies, 3]
        self.body_quat_w = data['body_quat_w']  # [T, num_bodies, 4] wxyz
        self.body_lin_vel_w = data['body_lin_vel_w']
        self.body_ang_vel_w = data['body_ang_vel_w']
        self.num_frames = self.joint_pos.shape[0]
        
        print(f"Loaded motion: {self.num_frames} frames at {self.fps} fps")
        print(f"  Joint pos shape: {self.joint_pos.shape}")
        print(f"  Body pos shape: {self.body_pos_w.shape}")


def quat_to_rot_matrix(quat_wxyz):
    """Convert quaternion (wxyz) to rotation matrix."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def quat_mul(q1, q2):
    """Multiply two quaternions (wxyz format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_inv(q):
    """Invert a quaternion (wxyz format)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def transform_point_by_quat(point, quat_wxyz):
    """Transform a point by a quaternion."""
    # Convert point to quaternion
    p = np.array([0, point[0], point[1], point[2]])
    q_inv = quat_inv(quat_wxyz)
    # q * p * q_inv
    result = quat_mul(quat_mul(quat_wxyz, p), q_inv)
    return result[1:4]


def subtract_frame_transforms(t1, q1, t2, q2):
    """
    Compute relative transform of frame 2 in frame 1.
    Returns position and orientation of frame 2 relative to frame 1.
    """
    # Relative orientation: q1_inv * q2
    q1_inv = quat_inv(q1)
    q_rel = quat_mul(q1_inv, q2)
    
    # Relative position: rotate (t2 - t1) into frame 1
    t_diff = t2 - t1
    t_rel = transform_point_by_quat(t_diff, q1_inv)
    
    return t_rel, q_rel


def get_robot_state(data, model, cfg):
    """Extract robot state from MuJoCo data and convert to Isaac Lab order."""
    # Base pose (first 7 elements of qpos: x, y, z, qw, qx, qy, qz)
    base_pos = data.qpos[:3].copy()
    base_quat_wxyz = data.qpos[3:7].copy()  # MuJoCo uses wxyz
    
    # Base velocity
    base_lin_vel = data.qvel[:3].copy()
    base_ang_vel = data.qvel[3:6].copy()
    
    # Transform velocities to base frame
    rot_mat = quat_to_rot_matrix(base_quat_wxyz)
    base_lin_vel_local = rot_mat.T @ base_lin_vel
    base_ang_vel_local = rot_mat.T @ base_ang_vel
    
    # Joint positions and velocities (after the 7 base DOFs) - in MuJoCo order
    joint_pos_mujoco = data.qpos[7:7+cfg.num_actions].copy()
    joint_vel_mujoco = data.qvel[6:6+cfg.num_actions].copy()
    
    # Convert from MuJoCo order to Isaac Lab order
    # mujoco_data[ISAAC_TO_MUJOCO] reorders: result[isaac_idx] = mujoco_data[mujoco_idx]
    joint_pos_isaac = joint_pos_mujoco[cfg.ISAAC_TO_MUJOCO]
    joint_vel_isaac = joint_vel_mujoco[cfg.ISAAC_TO_MUJOCO]
    
    return {
        'base_pos': base_pos,
        'base_quat': base_quat_wxyz,
        'base_lin_vel': base_lin_vel_local,
        'base_ang_vel': base_ang_vel_local,
        'joint_pos': joint_pos_isaac,  # In Isaac Lab order
        'joint_vel': joint_vel_isaac,  # In Isaac Lab order
    }


def build_observation(robot_state, motion, time_step, last_action, cfg):
    """
    Build observation vector matching Isaac Lab training.
    
    Observation order:
    1. command (joint_pos + joint_vel from motion) - 48 dims
    2. motion_anchor_pos_b (relative anchor position) - 3 dims  
    3. motion_anchor_ori_b (relative anchor orientation as 2 cols of rotation matrix) - 6 dims
    4. base_lin_vel - 3 dims
    5. base_ang_vel - 3 dims
    6. joint_pos_rel (joint_pos - default_pos) - 24 dims
    7. joint_vel - 24 dims (assuming no scaling needed)
    8. last_action - 24 dims
    
    Total: 48 + 3 + 6 + 3 + 3 + 24 + 24 + 24 = 135 dims
    """
    obs = []
    
    # Get motion frame
    t = time_step % motion.num_frames
    
    # 1. Command: target joint positions and velocities from motion
    cmd_joint_pos = motion.joint_pos[t]  # [24]
    cmd_joint_vel = motion.joint_vel[t]  # [24]
    obs.append(cmd_joint_pos)
    obs.append(cmd_joint_vel)
    
    # Get anchor body pose from motion (link_torso_yaw)
    # Based on FK analysis: body 7 is link_torso_yaw at z≈1.12m with similar orientation to pelvis
    # Isaac Lab reorders bodies during URDF import, so index != URDF link order
    anchor_idx = 7  # link_torso_yaw in Isaac Lab body ordering
    
    motion_anchor_pos = motion.body_pos_w[t, anchor_idx]
    motion_anchor_quat = motion.body_quat_w[t, anchor_idx]
    
    # Robot anchor pose (using base as approximation - in real impl would need torso body)
    robot_anchor_pos = robot_state['base_pos']
    robot_anchor_quat = robot_state['base_quat']
    
    # 2. motion_anchor_pos_b: relative position of motion anchor in robot anchor frame
    anchor_pos_rel, anchor_quat_rel = subtract_frame_transforms(
        robot_anchor_pos, robot_anchor_quat,
        motion_anchor_pos, motion_anchor_quat
    )
    obs.append(anchor_pos_rel)
    
    # 3. motion_anchor_ori_b: first 2 columns of rotation matrix
    rot_mat = quat_to_rot_matrix(anchor_quat_rel)
    obs.append(rot_mat[:, 0])  # First column
    obs.append(rot_mat[:, 1])  # Second column
    
    # 4. base_lin_vel
    obs.append(robot_state['base_lin_vel'])
    
    # 5. base_ang_vel
    obs.append(robot_state['base_ang_vel'])
    
    # 6. joint_pos_rel (relative to default) - in Isaac Lab order
    default_pos = np.array([cfg.default_joint_pos[name] for name in cfg.isaac_joint_names])
    joint_pos_rel = robot_state['joint_pos'] - default_pos
    obs.append(joint_pos_rel)
    
    # 7. joint_vel
    obs.append(robot_state['joint_vel'])
    
    # 8. last_action
    obs.append(last_action)
    
    return np.concatenate(obs).astype(np.float32)


def pd_control(target_pos_isaac, current_pos_isaac, current_vel_isaac, cfg):
    """
    Compute PD control torques in Isaac Lab order, then convert to MuJoCo order.
    
    Args:
        target_pos_isaac: Target positions in Isaac Lab order (action * scale) 
        current_pos_isaac: Current joint positions in Isaac Lab order
        current_vel_isaac: Current joint velocities in Isaac Lab order
        cfg: Configuration
    
    Returns:
        Torques in MuJoCo order for applying to simulation
    """
    # Get default positions in Isaac Lab order
    default_pos = np.array([cfg.default_joint_pos[name] for name in cfg.isaac_joint_names])
    
    # Target is action * scale + default (all in Isaac Lab order)
    target = target_pos_isaac + default_pos
    
    # Build kp, kd, and tau_limit arrays in Isaac Lab order
    kp = np.zeros(cfg.num_actions)
    kd = np.zeros(cfg.num_actions)
    tau_limit = np.zeros(cfg.num_actions)
    
    for i, name in enumerate(cfg.isaac_joint_names):
        if name in cfg.high_torque_joints:
            kp[i] = cfg.kp_high
            kd[i] = cfg.kd_high
            tau_limit[i] = cfg.tau_limit_high
        else:
            kp[i] = cfg.kp_low
            kd[i] = cfg.kd_low
            tau_limit[i] = cfg.tau_limit_low
    
    # PD control in Isaac Lab order
    tau_isaac = kp * (target - current_pos_isaac) - kd * current_vel_isaac
    
    # Clamp torques
    tau_isaac = np.clip(tau_isaac, -tau_limit, tau_limit)
    
    # Convert torques from Isaac Lab order to MuJoCo order
    # isaac_data[MUJOCO_TO_ISAAC] reorders: result[mujoco_idx] = isaac_data[isaac_idx]
    tau_mujoco = tau_isaac[cfg.MUJOCO_TO_ISAAC]
    
    return tau_mujoco


def load_policy(policy_path=None, onnx_path=None, wandb_path=None):
    """Load policy from PyTorch checkpoint, ONNX, or W&B."""
    
    # Download from W&B if specified
    if wandb_path is not None:
        import wandb
        import os
        
        api = wandb.Api()
        run_path = wandb_path
        
        # Handle model file specification
        if "model" in wandb_path:
            run_path = "/".join(wandb_path.split("/")[:-1])
            model_file = wandb_path.split("/")[-1]
        else:
            # Get latest model
            wandb_run = api.run(run_path)
            files = [f.name for f in wandb_run.files() if "model" in f.name and f.name.endswith(".pt")]
            model_file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        
        wandb_run = api.run(run_path)
        wandb_file = wandb_run.file(model_file)
        
        os.makedirs("./logs/wandb_downloads", exist_ok=True)
        wandb_file.download("./logs/wandb_downloads", replace=True)
        policy_path = f"./logs/wandb_downloads/{model_file}"
        print(f"Downloaded {model_file} from W&B: {run_path}")
    
    if onnx_path is not None:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        
        # Get input names
        input_names = [inp.name for inp in session.get_inputs()]
        print(f"ONNX inputs: {input_names}")
        
        # Check if time_step is required
        needs_time_step = 'time_step' in input_names
        
        def policy_fn(obs, time_step=0):
            obs_np = obs.reshape(1, -1).astype(np.float32)
            if needs_time_step:
                time_step_np = np.array([[time_step]], dtype=np.float32)
                output = session.run(None, {'obs': obs_np, 'time_step': time_step_np})
            else:
                output = session.run(None, {'obs': obs_np})
            return output[0][0]  # actions
        
        print(f"Loaded ONNX policy from {onnx_path}")
        print(f"  Requires time_step: {needs_time_step}")
        return policy_fn
    
    elif policy_path is not None:
        # Load PyTorch model
        checkpoint = torch.load(policy_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Extract actor weights
            actor_weights = {k: v for k, v in state_dict.items() if 'actor' in k}
            
            # Build simple MLP actor: 135 -> 512 -> 256 -> 128 -> 24
            class Actor(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = torch.nn.Sequential(
                        torch.nn.Linear(135, 512),
                        torch.nn.ELU(),
                        torch.nn.Linear(512, 256),
                        torch.nn.ELU(),
                        torch.nn.Linear(256, 128),
                        torch.nn.ELU(),
                        torch.nn.Linear(128, 24),
                    )
                
                def forward(self, x):
                    return self.net(x)
            
            actor = Actor()
            
            # Load weights
            actor_state = {}
            for k, v in actor_weights.items():
                # Map 'actor.0.weight' -> 'net.0.weight'
                new_key = k.replace('actor.', 'net.')
                actor_state[new_key] = v
            
            actor.load_state_dict(actor_state)
            actor.eval()
            
            def policy_fn(obs, time_step=0):
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action = actor(obs_tensor)
                return action.numpy()[0]
            
            print(f"Loaded PyTorch policy from {policy_path}")
            return policy_fn
        else:
            raise ValueError(f"Unknown checkpoint format: {checkpoint.keys()}")
    
    else:
        raise ValueError("Must provide --policy_path, --onnx_path, or --wandb_path")


def find_mujoco_model():
    """Try to find the MuJoCo XML model for PM01."""
    import os
    
    possible_paths = [
        # GMR assets folder
        "/home/marmot/Ritwik/GMR/assets/engineai_pm01/pm_v2.xml",
        # Relative paths
        "../GMR/assets/engineai_pm01/pm_v2.xml",
        "../../GMR/assets/engineai_pm01/pm_v2.xml",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def run_sim2sim(policy_fn, motion, cfg, args):
    """Run the sim2sim simulation."""
    
    # Load MuJoCo model
    if cfg.mujoco_model_path is None:
        cfg.mujoco_model_path = find_mujoco_model()
        if cfg.mujoco_model_path is None:
            raise FileNotFoundError(
                "Could not find MuJoCo XML model for PM01. "
                "Please specify --mujoco_model path"
            )
    
    print(f"Loading MuJoCo model: {cfg.mujoco_model_path}")
    model = mujoco.MjModel.from_xml_path(cfg.mujoco_model_path)
    model.opt.timestep = cfg.dt
    data = mujoco.MjData(model)
    
    # Print joint info
    print(f"\nMuJoCo model has {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")
    
    # Initialize joint positions
    # default_pos is in Isaac Lab order, convert to MuJoCo order for initialization
    # isaac_data[MUJOCO_TO_ISAAC] reorders: result[mujoco_idx] = isaac_data[isaac_idx]
    default_pos_isaac = np.array([cfg.default_joint_pos[name] for name in cfg.isaac_joint_names])
    default_pos_mujoco = default_pos_isaac[cfg.MUJOCO_TO_ISAAC]
    if model.nq >= 7 + cfg.num_actions:
        data.qpos[7:7+cfg.num_actions] = default_pos_mujoco
    
    # Initialize base height
    data.qpos[2] = 0.82  # Base height
    data.qpos[3] = 1.0   # Quaternion w
    
    mujoco.mj_forward(model, data)
    
    # Create viewer
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(model, data)
    
    # Simulation loop
    last_action = np.zeros(cfg.num_actions, dtype=np.float32)
    action_scales = np.array([cfg.action_scales[name] for name in cfg.isaac_joint_names])
    
    step_count = 0
    policy_step = 0
    motion_step = 0
    
    print("\nStarting simulation...")
    print(f"Policy decimation: {cfg.decimation} (policy at {1.0/cfg.dt/cfg.decimation:.1f} Hz)")
    
    try:
        while step_count * cfg.dt < cfg.sim_duration:
            # Run policy at decimated rate
            if step_count % cfg.decimation == 0:
                # Get robot state
                robot_state = get_robot_state(data, model, cfg)
                
                # Build observation
                obs = build_observation(robot_state, motion, motion_step, last_action, cfg)
                
                # Run policy (pass motion time step for phase info)
                action = policy_fn(obs, time_step=motion_step)
                action = np.clip(action, -1.0, 1.0)  # Clip actions
                
                # Scale action
                target_pos = action * action_scales
                last_action = action.copy()
                
                # Debug output
                if hasattr(cfg, 'debug') and cfg.debug and policy_step < 5:
                    print(f"\n=== Debug Step {policy_step} ===")
                    print(f"Obs shape: {obs.shape}, range: [{obs.min():.3f}, {obs.max():.3f}]")
                    print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
                    print(f"Target pos range: [{target_pos.min():.3f}, {target_pos.max():.3f}]")
                    print(f"Base height: {data.qpos[2]:.3f}")
                    # Show first few joint positions vs motion command
                    print(f"Robot joint_pos (first 6): {robot_state['joint_pos'][:6]}")
                    print(f"Motion cmd_pos (first 6): {motion.joint_pos[motion_step, :6]}")
                    print(f"Default pos (first 6): {default_pos_isaac[:6]}")
                
                # Advance motion
                motion_step += 1
                if motion_step >= motion.num_frames:
                    motion_step = 0
                    print(f"Motion looped at step {step_count}")
                
                policy_step += 1
                
                if policy_step % 100 == 0:
                    print(f"Step {policy_step}, time {step_count * cfg.dt:.2f}s, "
                          f"base height: {data.qpos[2]:.3f}")
            
            # PD control - convert MuJoCo positions to Isaac Lab order
            current_pos_mujoco = data.qpos[7:7+cfg.num_actions]
            current_vel_mujoco = data.qvel[6:6+cfg.num_actions]
            # Convert to Isaac Lab order for PD control
            # mujoco_data[ISAAC_TO_MUJOCO] reorders: result[isaac_idx] = mujoco_data[mujoco_idx]
            current_pos_isaac = current_pos_mujoco[cfg.ISAAC_TO_MUJOCO]
            current_vel_isaac = current_vel_mujoco[cfg.ISAAC_TO_MUJOCO]
            # pd_control returns torques in MuJoCo order
            tau = pd_control(target_pos, current_pos_isaac, current_vel_isaac, cfg)
            
            # Debug torques on first few steps
            if hasattr(cfg, 'debug') and cfg.debug and step_count < 100 and step_count % 20 == 0:
                print(f"Torque range: [{tau.min():.1f}, {tau.max():.1f}] Nm")
            
            # Apply torques
            data.ctrl[:cfg.num_actions] = tau
            
            # Step simulation
            mujoco.mj_step(model, data)
            step_count += 1
            
            # Update viewer at ~60 Hz (not every physics step)
            if not args.headless and step_count % 16 == 0:  # 1000 Hz / 16 ≈ 60 Hz
                viewer.sync()
                if not viewer.is_running():
                    break
    
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    if not args.headless:
        viewer.close()
    
    print(f"\nSimulation finished: {step_count} steps, {step_count * cfg.dt:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Sim2Sim for PM01 motion tracking policy")
    parser.add_argument("--policy_path", type=str, default=None, help="Path to PyTorch policy checkpoint")
    parser.add_argument("--onnx_path", type=str, default=None, help="Path to ONNX policy")
    parser.add_argument("--wandb_path", type=str, default=None, help="W&B run path (entity/project/run_id or entity/project/run_id/model_xxx.pt)")
    parser.add_argument("--motion_file", type=str, required=True, help="Path to motion .npz file")
    parser.add_argument("--mujoco_model", type=str, default=None, help="Path to MuJoCo XML model")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration in seconds")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    
    args = parser.parse_args()
    
    # Configuration
    cfg = PM01Config()
    cfg.sim_duration = args.duration
    cfg.debug = args.debug
    if args.mujoco_model:
        cfg.mujoco_model_path = args.mujoco_model
    
    # Load motion
    motion = MotionLoader(args.motion_file)
    
    # Load policy
    policy_fn = load_policy(args.policy_path, args.onnx_path, args.wandb_path)
    
    # Run simulation
    run_sim2sim(policy_fn, motion, cfg, args)


if __name__ == "__main__":
    main()
