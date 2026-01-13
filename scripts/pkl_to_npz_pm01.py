"""Convert retargeted motion .pkl file to .npz format for PM01.

This script converts motion data from a .pkl file (typically from motion retargeting)
to the .npz format required by the whole_body_tracking system.

Usage:
    python scripts/pkl_to_npz_pm01.py --input_file motion.pkl --output_file motion.npz --fps 50

Expected .pkl format (from general_motion_retargeting):
    The .pkl should contain a dict with keys like:
    - 'qpos': joint positions per frame (num_frames, num_joints+7) where first 7 are root pos/quat
    - OR 'joint_pos': just joint positions
    - OR 'poses': similar format
    
    Adjust the loading logic below based on your actual .pkl structure.
"""

import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser(description="Convert .pkl motion to .npz for PM01")
parser.add_argument("--input_file", type=str, required=True, help="Path to input .pkl file")
parser.add_argument("--output_file", type=str, default=None, help="Path to output .npz file (required unless --inspect)")
parser.add_argument("--fps", type=int, default=50, help="Motion FPS (default: 50)")
parser.add_argument("--inspect", action="store_true", help="Just inspect the .pkl structure without converting")

args = parser.parse_args()

# Validate arguments
if not args.inspect and args.output_file is None:
    parser.error("--output_file is required when not using --inspect")


def inspect_pkl(filepath: str):
    """Print the structure of a .pkl file."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    print(f"\n=== Inspecting {filepath} ===\n")
    
    if isinstance(data, dict):
        print("Type: dict")
        print("Keys:")
        for key in data.keys():
            value = data[key]
            if isinstance(value, np.ndarray):
                print(f"  '{key}': np.ndarray, shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, (list, tuple)):
                print(f"  '{key}': {type(value).__name__}, len={len(value)}")
                if len(value) > 0 and isinstance(value[0], np.ndarray):
                    print(f"         first element shape: {value[0].shape}")
            else:
                print(f"  '{key}': {type(value).__name__} = {value}")
    elif isinstance(data, (list, tuple)):
        print(f"Type: {type(data).__name__}, len={len(data)}")
        if len(data) > 0:
            print(f"First element type: {type(data[0])}")
            if isinstance(data[0], np.ndarray):
                print(f"First element shape: {data[0].shape}")
    elif isinstance(data, np.ndarray):
        print(f"Type: np.ndarray, shape={data.shape}, dtype={data.dtype}")
    else:
        print(f"Type: {type(data)}")
    
    print("\n=== End inspection ===\n")
    return data


def convert_pkl_to_npz(input_file: str, output_file: str, fps: int):
    """Convert .pkl motion file to .npz format.
    
    This version doesn't require Isaac Sim - it saves joint positions directly
    and creates placeholder body arrays. The replay script will handle FK.
    """
    # Load pkl file
    with open(input_file, "rb") as f:
        data = pickle.load(f)
    
    print(f"[INFO]: Loaded {input_file}")
    
    # Extract motion data based on .pkl structure
    if isinstance(data, dict):
        # Get FPS from file if available
        if 'fps' in data:
            fps = int(data['fps'])
            print(f"[INFO]: Using FPS from file: {fps}")
        
        # Format 1: root_pos, root_rot, dof_pos (your tennis.pkl format)
        if 'root_pos' in data and 'root_rot' in data and 'dof_pos' in data:
            root_pos = np.array(data['root_pos'])
            root_quat = np.array(data['root_rot'])  # Assuming wxyz format
            joint_pos = np.array(data['dof_pos'])
        # Format 2: Dict with 'qpos' key (root pos + quat + joint angles)
        elif 'qpos' in data:
            qpos = np.array(data['qpos'])
            root_pos = qpos[:, :3]
            root_quat = qpos[:, 3:7]
            joint_pos = qpos[:, 7:]
        # Format 3: Separate keys with different naming
        elif 'root_pos' in data and 'joint_pos' in data:
            root_pos = np.array(data['root_pos'])
            root_quat = np.array(data.get('root_quat', data.get('root_ori', data.get('root_rot'))))
            joint_pos = np.array(data['joint_pos'])
        # Format 4: Just joint positions (assume static root)
        elif 'joint_pos' in data or 'poses' in data or 'dof_pos' in data:
            joint_pos = np.array(data.get('joint_pos', data.get('poses', data.get('dof_pos'))))
            num_frames = joint_pos.shape[0]
            root_pos = np.zeros((num_frames, 3))
            root_pos[:, 2] = 0.82  # PM01 standing height
            root_quat = np.zeros((num_frames, 4))
            root_quat[:, 0] = 1.0  # Identity quaternion (wxyz)
        else:
            raise ValueError(f"Unknown .pkl format. Keys: {data.keys()}")
    else:
        raise ValueError(f"Expected dict, got {type(data)}")
    
    num_frames = joint_pos.shape[0]
    print(f"[INFO]: Motion has {num_frames} frames, {joint_pos.shape[1]} joints")
    
    # Compute velocities via finite differences
    dt = 1.0 / fps
    joint_vel = np.gradient(joint_pos, dt, axis=0)
    root_lin_vel = np.gradient(root_pos, dt, axis=0)
    
    # Compute angular velocity from quaternions
    root_ang_vel = _compute_angular_velocity(root_quat, dt)
    
    # For the npz format, we need body_pos_w and body_quat_w
    # Since we can't run FK without Isaac Sim, we'll create placeholder arrays
    # The replay script will use joint_pos directly
    
    # Create body arrays with just the root body (index 0)
    # This is a simplified version - full FK would require the robot model
    num_bodies = 25  # PM01 has 25 bodies (base + 24 joints)
    
    # Initialize body arrays - we'll put root at index 0, rest as placeholders
    body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    
    # Set root body (index 0)
    body_pos_w[:, 0, :] = root_pos
    body_quat_w[:, 0, :] = root_quat
    body_lin_vel_w[:, 0, :] = root_lin_vel
    body_ang_vel_w[:, 0, :] = root_ang_vel
    
    # For other bodies, set identity quaternion
    body_quat_w[:, 1:, 0] = 1.0  # w component
    
    # Save to npz
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
    
    print(f"[INFO]: Saved motion to {output_file}")
    print(f"  - fps: {fps}")
    print(f"  - joint_pos: {joint_pos.shape}")
    print(f"  - body_pos_w: {body_pos_w.shape}")
    print(f"[NOTE]: Body positions are placeholders (root only). The replay script will set joint positions directly.")


def _compute_angular_velocity(quats: np.ndarray, dt: float) -> np.ndarray:
    """Compute angular velocity from quaternion sequence."""
    # Simple finite difference on quaternions, then convert to angular velocity
    # For wxyz quaternions: omega = 2 * q_dot * q_conj
    num_frames = quats.shape[0]
    ang_vel = np.zeros((num_frames, 3), dtype=np.float32)
    
    for i in range(1, num_frames - 1):
        q_prev = quats[i - 1]
        q_next = quats[i + 1]
        
        # Quaternion difference
        q_diff = _quat_mul(_quat_conj(q_prev), q_next)
        
        # Convert to axis-angle and divide by time
        angle = 2.0 * np.arccos(np.clip(q_diff[0], -1.0, 1.0))
        if abs(angle) > 1e-6:
            axis = q_diff[1:4] / np.sin(angle / 2.0)
            ang_vel[i] = axis * angle / (2.0 * dt)
    
    # Copy first and last
    ang_vel[0] = ang_vel[1]
    ang_vel[-1] = ang_vel[-2]
    
    return ang_vel


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication (wxyz format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (wxyz format)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


if __name__ == "__main__":
    if args.inspect:
        inspect_pkl(args.input_file)
    else:
        convert_pkl_to_npz(args.input_file, args.output_file, args.fps)
