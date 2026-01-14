"""Convert retargeted motion .pkl file to .npz format for PM01.

Usage:
    python scripts/pkl_to_npz_pm01.py --input_file motion.pkl --output_file motion.npz
    python scripts/pkl_to_npz_pm01.py --input_file motion.pkl --inspect
"""

import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser(description="Convert .pkl motion to .npz for PM01")
parser.add_argument("--input_file", type=str, required=True, help="Path to input .pkl file")
parser.add_argument("--output_file", type=str, default=None, help="Path to output .npz file")
parser.add_argument("--fps", type=int, default=50, help="Motion FPS (default: 50)")
parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to convert")
parser.add_argument("--inspect", action="store_true", help="Inspect .pkl structure without converting")

args = parser.parse_args()

if not args.inspect and args.output_file is None:
    parser.error("--output_file is required when not using --inspect")

# GMR to Isaac Lab joint order mapping
GMR_TO_ISAAC = [0, 6, 12, 1, 7, 13, 18, 23, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]


def inspect_pkl(filepath: str):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    print(f"\n=== {filepath} ===\n")
    if isinstance(data, dict):
        print("Type: dict\nKeys:")
        for key in data.keys():
            value = data[key]
            if isinstance(value, np.ndarray):
                print(f"  '{key}': shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, (list, tuple)):
                print(f"  '{key}': {type(value).__name__}, len={len(value)}")
            else:
                print(f"  '{key}': {type(value).__name__} = {value}")
    else:
        print(f"Type: {type(data)}")
    print()


def convert_pkl_to_npz(input_file: str, output_file: str, fps: int, max_frames: int = None):
    with open(input_file, "rb") as f:
        data = pickle.load(f)
    
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data)}")
    
    if 'fps' in data:
        fps = int(data['fps'])
    
    # Extract motion data
    if 'root_pos' in data and 'root_rot' in data and 'dof_pos' in data:
        root_pos = np.array(data['root_pos'])
        root_quat_xyzw = np.array(data['root_rot'])
        root_quat = root_quat_xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        joint_pos = np.array(data['dof_pos'])
    elif 'qpos' in data:
        qpos = np.array(data['qpos'])
        root_pos = qpos[:, :3]
        root_quat = qpos[:, 3:7]
        joint_pos = qpos[:, 7:]
    else:
        raise ValueError(f"Unknown .pkl format. Keys: {data.keys()}")
    
    num_frames = joint_pos.shape[0]
    
    # Limit frames if max_frames specified
    if max_frames is not None and max_frames < num_frames:
        print(f"Limiting from {num_frames} to {max_frames} frames")
        root_pos = root_pos[:max_frames]
        root_quat = root_quat[:max_frames]
        joint_pos = joint_pos[:max_frames]
        num_frames = max_frames
    
    # Reorder joints from GMR to Isaac Lab order
    joint_pos = joint_pos[:, GMR_TO_ISAAC]
    
    # Compute velocities
    dt = 1.0 / fps
    joint_vel = np.gradient(joint_pos, dt, axis=0)
    root_lin_vel = np.gradient(root_pos, dt, axis=0)
    root_ang_vel = _compute_angular_velocity(root_quat, dt)
    
    # Create body arrays (root body only, rest are placeholders)
    num_bodies = 25
    body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    
    body_pos_w[:, 0, :] = root_pos
    body_quat_w[:, 0, :] = root_quat
    body_lin_vel_w[:, 0, :] = root_lin_vel
    body_ang_vel_w[:, 0, :] = root_ang_vel
    body_quat_w[:, 1:, 0] = 1.0  # identity quaternion
    
    np.savez(
        output_file,
        fps=fps,
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    print(f"Saved {output_file}: {num_frames} frames, {joint_pos.shape[1]} joints, {fps} fps")


def _compute_angular_velocity(quats: np.ndarray, dt: float) -> np.ndarray:
    num_frames = quats.shape[0]
    ang_vel = np.zeros((num_frames, 3), dtype=np.float32)
    
    for i in range(1, num_frames - 1):
        q_prev = quats[i - 1]
        q_next = quats[i + 1]
        q_diff = _quat_mul(_quat_conj(q_prev), q_next)
        angle = 2.0 * np.arccos(np.clip(q_diff[0], -1.0, 1.0))
        if abs(angle) > 1e-6:
            axis = q_diff[1:4] / np.sin(angle / 2.0)
            ang_vel[i] = axis * angle / (2.0 * dt)
    
    ang_vel[0] = ang_vel[1]
    ang_vel[-1] = ang_vel[-2]
    return ang_vel


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


if __name__ == "__main__":
    if args.inspect:
        inspect_pkl(args.input_file)
    else:
        convert_pkl_to_npz(args.input_file, args.output_file, args.fps, args.max_frames)
