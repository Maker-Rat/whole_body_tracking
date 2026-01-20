"""
Sim2Sim script for G1 motion tracking policy in MuJoCo.

This script runs a trained motion-tracking policy from Isaac Lab in MuJoCo for the G1 robot.

Usage:
    python scripts/sim2sim_g1.py --policy_path logs/rsl_rl/g1_flat/<run>/model_xxx.pt --motion_file scripts/walk_fk.npz
    python scripts/sim2sim_g1.py --onnx_path logs/rsl_rl/g1_flat/<run>/<run>.onnx --motion_file scripts/walk_fk.npz
"""

import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# ============== Configuration ==============

class G1Config:
    """G1 robot and simulation configuration."""
    dt = 0.001  # MuJoCo timestep (1000 Hz)
    decimation = 20  # Policy runs at 50 Hz (1000/20 = 50)
    sim_duration = 60.0  # seconds
    mujoco_model_path = None  # Set via command line or default

    # These should match the Isaac Lab training config
    num_actions = 29
    action_scales = np.array([
        0.25, 0.25, 0.25, 0.25,  # hip_yaw, hip_roll, hip_pitch, knee (left)
        0.25, 0.25, 0.25, 0.25,  # hip_yaw, hip_roll, hip_pitch, knee (right)
        0.25, 0.25, 0.25, 0.25,  # ankle_pitch, ankle_roll, waist_roll, waist_pitch
        0.25, 0.25, 0.25, 0.25,  # waist_yaw, shoulder_pitch, shoulder_roll, shoulder_yaw (left)
        0.25, 0.25, 0.25, 0.25   # elbow, wrist_roll, wrist_pitch, wrist_yaw (left)
    ])
    # Default joint positions for G1 (29 DOF, actuator order from XML)
    # Typical standing pose: legs slightly bent, arms neutral, hands zero
    default_joint_pos = np.array([
        # Legs (left)
        -0.312,  # left_hip_pitch_joint
        0.0,     # left_hip_roll_joint
        0.0,     # left_hip_yaw_joint
        0.669,   # left_knee_joint
        -0.363,  # left_ankle_pitch_joint
        0.0,     # left_ankle_roll_joint
        # Legs (right)
        -0.312,  # right_hip_pitch_joint
        0.0,     # right_hip_roll_joint
        0.0,     # right_hip_yaw_joint
        0.669,   # right_knee_joint
        -0.363,  # right_ankle_pitch_joint
        0.0,     # right_ankle_roll_joint
        # Waist/torso
        0.0,     # waist_yaw_joint
        0.0,     # waist_roll_joint
        0.0,     # waist_pitch_joint
        # Left arm
        0.2,     # left_shoulder_pitch_joint
        0.2,     # left_shoulder_roll_joint
        0.0,     # left_shoulder_yaw_joint
        0.6,     # left_elbow_joint
        0.0,     # left_wrist_roll_joint
        0.0,     # left_wrist_pitch_joint
        0.0,     # left_wrist_yaw_joint
        # Left hand (thumb, middle, index)
        0.0,     # left_hand_thumb_0_joint
        0.0,     # left_hand_thumb_1_joint
        0.0,     # left_hand_thumb_2_joint
        0.0,     # left_hand_middle_0_joint
        0.0,     # left_hand_middle_1_joint
        0.0,     # left_hand_index_0_joint
        0.0,     # left_hand_index_1_joint
    ])
    anchor_body_name = "torso_link"

class MotionLoader:
    def __init__(self, motion_file: str):
        data = np.load(motion_file)
        self.fps = float(data['fps'])
        self.joint_pos = data['joint_pos']
        self.joint_vel = data['joint_vel']
        self.body_pos_w = data['body_pos_w']
        self.body_quat_w = data['body_quat_w']
        self.body_lin_vel_w = data['body_lin_vel_w']
        self.body_ang_vel_w = data['body_ang_vel_w']
        self.num_frames = self.joint_pos.shape[0]
        print(f"Loaded motion: {self.num_frames} frames at {self.fps} fps")


def get_robot_state(data, model, cfg):
    base_pos = data.qpos[:3].copy()
    base_quat_wxyz = data.qpos[3:7].copy()
    base_lin_vel = data.qvel[:3].copy()
    base_ang_vel = data.qvel[3:6].copy()
    joint_pos = data.qpos[7:7+cfg.num_actions].copy()
    joint_vel = data.qvel[6:6+cfg.num_actions].copy()
    # Anchor body pose
    anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.anchor_body_name)
    if anchor_body_id >= 0:
        anchor_pos = data.xpos[anchor_body_id].copy()
        anchor_quat = data.xquat[anchor_body_id].copy()
    else:
        anchor_pos = base_pos
        anchor_quat = base_quat_wxyz
    return {
        'base_pos': base_pos,
        'base_quat': base_quat_wxyz,
        'base_lin_vel': base_lin_vel,
        'base_ang_vel': base_ang_vel,
        'joint_pos': joint_pos,
        'joint_vel': joint_vel,
        'anchor_pos': anchor_pos,
        'anchor_quat': anchor_quat,
    }


def build_observation(robot_state, motion, time_step, last_action, cfg):
    t = time_step % motion.num_frames
    obs = []
    # 1. Command: joint_pos + joint_vel from motion (Isaac Lab order, 29+29=58)
    obs.append(motion.joint_pos[t])  # [29]
    obs.append(motion.joint_vel[t])  # [29]

    # 2. motion_anchor_pos_b (relative anchor position) - 3 dims
    # 3. motion_anchor_ori_b (relative anchor orientation as 2 cols of rotation matrix) - 6 dims
    # Use anchor body pose from robot and motion
    # For G1, anchor body index may differ, but let's assume anchor_body_name is correct
    # Get anchor pose from motion (assume anchor index 0 for now, update if needed)
    anchor_idx = 0  # TODO: update if anchor is not index 0 in motion
    motion_anchor_pos = motion.body_pos_w[t, anchor_idx]
    motion_anchor_quat = motion.body_quat_w[t, anchor_idx]
    robot_anchor_pos = robot_state['anchor_pos']
    robot_anchor_quat = robot_state['anchor_quat']

    # Compute relative transform (position and orientation)
    def quat_to_rot_matrix(quat_wxyz):
        w, x, y, z = quat_wxyz
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
    def quat_inv(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])
    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    def transform_point_by_quat(point, quat_wxyz):
        p = np.array([0, point[0], point[1], point[2]])
        q_inv = quat_inv(quat_wxyz)
        result = quat_mul(quat_mul(quat_wxyz, p), q_inv)
        return result[1:4]
    def subtract_frame_transforms(t1, q1, t2, q2):
        q1_inv = quat_inv(q1)
        q_rel = quat_mul(q1_inv, q2)
        t_diff = t2 - t1
        t_rel = transform_point_by_quat(t_diff, q1_inv)
        return t_rel, q_rel

    anchor_pos_rel, anchor_quat_rel = subtract_frame_transforms(
        robot_anchor_pos, robot_anchor_quat,
        motion_anchor_pos, motion_anchor_quat
    )
    obs.append(anchor_pos_rel)  # [3]
    rot_mat = quat_to_rot_matrix(anchor_quat_rel)
    obs.append(rot_mat[:, 0])  # [3]
    obs.append(rot_mat[:, 1])  # [3]

    # 4. base_lin_vel - 3 dims
    obs.append(robot_state['base_lin_vel'])
    # 5. base_ang_vel - 3 dims
    obs.append(robot_state['base_ang_vel'])
    # 6. joint_pos_rel (joint_pos - default_pos) - 29 dims
    obs.append(robot_state['joint_pos'] - cfg.default_joint_pos)
    # 7. joint_vel - 29 dims
    obs.append(robot_state['joint_vel'])
    # 8. last_action - 29 dims
    obs.append(last_action)
    return np.concatenate(obs).astype(np.float32)


def pd_control(target_pos, current_pos, current_vel, cfg):
    kp = 100.0  # Example PD gains
    kd = 5.0
    tau = kp * (target_pos - current_pos) - kd * current_vel
    tau = np.clip(tau, -100, 100)
    return tau


def load_policy(policy_path=None, onnx_path=None, wandb_path=None):
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
            wandb_run = api.run(run_path)
            files = [f.name for f in wandb_run.files() if "model" in f.name and f.name.endswith(".pt")]
            model_file = max(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        wandb_run = api.run(run_path)
        wandb_file = wandb_run.file(model_file)
        os.makedirs("./logs/wandb_downloads", exist_ok=True)
        wandb_file.download("./logs/wandb_downloads", replace=True)
        policy_path = f"./logs/wandb_downloads/{model_file}"
        print(f"Downloaded {model_file} from W&B: {run_path}")
    if onnx_path is not None:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        def policy_fn(obs, time_step=0):
            obs_np = obs.reshape(1, -1).astype(np.float32)
            output = session.run(None, {'obs': obs_np})
            return output[0][0]
        print(f"Loaded ONNX policy from {onnx_path}")
        return policy_fn
    elif policy_path is not None:
        checkpoint = torch.load(policy_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        # Auto-detect input/output sizes from weights
        input_size = state_dict['actor.0.weight'].shape[1]
        output_size = state_dict['actor.6.weight'].shape[0]
        print(f"Detected policy input size: {input_size}, output size: {output_size}")
        class Actor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(input_size, 512),
                    torch.nn.ELU(),
                    torch.nn.Linear(512, 256),
                    torch.nn.ELU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.ELU(),
                    torch.nn.Linear(128, output_size),
                )
            def forward(self, x):
                return self.net(x)
        actor = Actor()
        actor_state = {}
        for k, v in state_dict.items():
            if k.startswith('actor.'):
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
        # Update config sizes globally for rest of script
        G1Config.num_actions = output_size
        G1Config.obs_size = input_size
        return policy_fn
    else:
        raise ValueError("Must provide --policy_path, --onnx_path, or --wandb_path")


def find_mujoco_model():
    import os
    possible_paths = [
        "/home/marmot/Ritwik/GMR/assets/unitree_g1/g1_mocap_29dof_with_hands.xml",
        "../GMR/assets/unitree_g1/g1_mocap_29dof_with_hands.xml",
        "../../GMR/assets/unitree_g1/g1_mocap_29dof_with_hands.xml",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def run_sim2sim(policy_fn, motion, cfg, args):
    if cfg.mujoco_model_path is None:
        cfg.mujoco_model_path = find_mujoco_model()
        if cfg.mujoco_model_path is None:
            raise FileNotFoundError("Could not find MuJoCo XML model for G1. Please specify --mujoco_model path")
    print(f"Loading MuJoCo model: {cfg.mujoco_model_path}")
    model = mujoco.MjModel.from_xml_path(cfg.mujoco_model_path)
    model.opt.timestep = cfg.dt
    data = mujoco.MjData(model)
    print(f"\nMuJoCo model has {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")
    data.qpos[2] = 0.76  # Base height
    data.qpos[3] = 1.0   # Quaternion w
    mujoco.mj_forward(model, data)
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(model, data)
    last_action = np.zeros(cfg.num_actions, dtype=np.float32)
    action_scales = cfg.action_scales
    step_count = 0
    policy_step = 0
    motion_step = 0
    print("\nStarting simulation...")
    print(f"Policy decimation: {cfg.decimation} (policy at {1.0/cfg.dt/cfg.decimation:.1f} Hz)")
    try:
        while step_count * cfg.dt < cfg.sim_duration:
            if step_count % cfg.decimation == 0:
                robot_state = get_robot_state(data, model, cfg)
                obs = build_observation(robot_state, motion, motion_step, last_action, cfg)
                action = policy_fn(obs, time_step=motion_step)
                target_pos = action * action_scales
                last_action = action.copy()
                motion_step += 1
                if motion_step >= motion.num_frames:
                    motion_step = 0
                policy_step += 1
                if policy_step % 100 == 0:
                    print(f"Step {policy_step}, time {step_count * cfg.dt:.2f}s, base height: {data.qpos[2]:.3f}")
            current_pos = data.qpos[7:7+cfg.num_actions]
            current_vel = data.qvel[6:6+cfg.num_actions]
            tau = pd_control(target_pos, current_pos, current_vel, cfg)
            data.ctrl[:cfg.num_actions] = tau
            mujoco.mj_step(model, data)
            step_count += 1
            if not args.headless and step_count % 32 == 0:
                viewer.sync()
                if not viewer.is_running():
                    break
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    if not args.headless:
        viewer.close()
    print(f"\nSimulation finished: {step_count} steps, {step_count * cfg.dt:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Sim2Sim for G1 motion tracking policy")
    parser.add_argument("--policy_path", type=str, default=None, help="Path to PyTorch policy checkpoint")
    parser.add_argument("--onnx_path", type=str, default=None, help="Path to ONNX policy")
    parser.add_argument("--wandb_path", type=str, default=None, help="W&B run path (entity/project/run_id or entity/project/run_id/model_xxx.pt)")
    parser.add_argument("--motion_file", type=str, required=True, help="Path to motion .npz file")
    parser.add_argument("--mujoco_model", type=str, default=None, help="Path to MuJoCo XML model")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration in seconds")
    args = parser.parse_args()
    cfg = G1Config()
    cfg.sim_duration = args.duration
    if args.mujoco_model:
        cfg.mujoco_model_path = args.mujoco_model
    motion = MotionLoader(args.motion_file)
    policy_fn = load_policy(args.policy_path, args.onnx_path, args.wandb_path)
    run_sim2sim(policy_fn, motion, cfg, args)

if __name__ == "__main__":
    main()
