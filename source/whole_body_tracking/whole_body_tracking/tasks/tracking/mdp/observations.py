"""Observation functions that support lag for motor and IMU observations."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to ZYX Euler angles (roll, pitch, yaw).
    
    This matches the C++ convention: euler_xyz = CalcRollPitchYawFromRotationMatrix(R)
    where the rotation matrix is constructed from a quaternion.
    
    Args:
        quat: Quaternion tensor of shape (..., 4) in [w, x, y, z] format
        
    Returns:
        Euler angles tensor of shape (..., 3) containing [roll, pitch, yaw] in radians
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # Clamp to avoid numerical issues with asin at singularities
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return torch.stack([roll, pitch, yaw], dim=-1)


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_euler_xyz(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get Euler angles (roll, pitch, yaw) from robot anchor orientation in world frame.
    
    This matches the C++ IMU observation that computes RPY from rotation matrix.
    Returns ZYX Euler angles (which corresponds to roll, pitch, yaw).
    
    Args:
        env: The environment instance.
        command_name: The name of the motion command.
        
    Returns:
        Tensor of shape (num_envs, 3) containing [roll, pitch, yaw] in radians.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # Convert quaternion to Euler angles (roll, pitch, yaw)
    euler_angles = quat_to_euler_xyz(command.robot_anchor_quat_w)
    
    return euler_angles.view(env.num_envs, 3)


def robot_anchor_euler_xyz_from_matrix(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get Euler angles (roll, pitch, yaw) from robot anchor orientation using rotation matrix.
    
    This implementation more closely mirrors the C++ code:
        Eigen::Matrix3d R_real = Eigen::Quaterniond(...).toRotationMatrix();
        Eigen::Vector3d euler_xyz = math::CalcRollPitchYawFromRotationMatrix(R_real);
    
    Args:
        env: The environment instance.
        command_name: The name of the motion command.
        
    Returns:
        Tensor of shape (num_envs, 3) containing [roll, pitch, yaw] in radians.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # Convert quaternion to rotation matrix (similar to C++ .toRotationMatrix())
    rot_matrix = matrix_from_quat(command.robot_anchor_quat_w)  # Shape: (num_envs, 3, 3)
    
    # Extract Euler angles from rotation matrix (ZYX convention = roll, pitch, yaw)
    # For ZYX Euler angles (roll about X, pitch about Y, yaw about Z):
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    
    # Roll (rotation about x-axis)
    roll = torch.atan2(rot_matrix[..., 2, 1], rot_matrix[..., 2, 2])
    
    # Pitch (rotation about y-axis)
    sin_pitch = -rot_matrix[..., 2, 0]
    sin_pitch = torch.clamp(sin_pitch, -1.0, 1.0)  # Clamp for numerical stability
    pitch = torch.asin(sin_pitch)
    
    # Yaw (rotation about z-axis)
    yaw = torch.atan2(rot_matrix[..., 1, 0], rot_matrix[..., 0, 0])
    
    euler_angles = torch.stack([roll, pitch, yaw], dim=-1)
    
    return euler_angles.view(env.num_envs, 3)


# ============================================================================
# Lagged observation functions
# ============================================================================


def robot_anchor_euler_xyz_lagged(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get Euler angles with IMU lag applied.
    
    This function applies the IMU observation lag that was set during environment reset.
    It retrieves euler angles from a history buffer based on the lag timesteps.
    
    Args:
        env: The environment instance.
        command_name: The name of the motion command.
        
    Returns:
        Tensor of shape (num_envs, 3) containing lagged [roll, pitch, yaw] in radians.
    """
    # Get current (non-lagged) observation
    current_euler = robot_anchor_euler_xyz_from_matrix(env, command_name)
    
    # If lag is not initialized, return current observation
    if not hasattr(env, '_imu_lag_timesteps') or not hasattr(env, '_imu_obs_history'):
        return current_euler
    
    # Get angular velocity for the full IMU observation
    current_ang_vel = robot_anchor_ang_vel_w(env, command_name)
    
    # Combine euler angles and angular velocity
    current_imu = torch.cat([current_euler, current_ang_vel], dim=-1)
    
    # Update history buffer (circular buffer)
    max_lag = env._imu_obs_history.shape[1]
    current_idx = env._imu_obs_history_idx % max_lag
    
    # Store current observation in history
    env._imu_obs_history[torch.arange(env.num_envs), current_idx] = current_imu
    
    # Increment index
    env._imu_obs_history_idx = (env._imu_obs_history_idx + 1) % max_lag
    
    # Retrieve lagged observations
    lag_timesteps = env._imu_lag_timesteps
    lagged_idx = (current_idx - lag_timesteps) % max_lag
    lagged_imu = env._imu_obs_history[torch.arange(env.num_envs), lagged_idx]
    
    # Return only the euler angles (first 3 values)
    return lagged_imu[:, :3]


def base_ang_vel_lagged(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get angular velocity with IMU lag applied.
    
    This function applies the IMU observation lag that was set during environment reset.
    
    Args:
        env: The environment instance.
        command_name: The name of the motion command (not used but kept for API consistency).
        
    Returns:
        Tensor of shape (num_envs, 3) containing lagged angular velocity.
    """
    # Get current observation
    current_ang_vel = robot_anchor_ang_vel_w(env, command_name)
    
    # If lag is not initialized, return current observation
    if not hasattr(env, '_imu_lag_timesteps') or not hasattr(env, '_imu_obs_history'):
        return current_ang_vel
    
    # The IMU history already contains ang_vel, just retrieve the lagged values
    max_lag = env._imu_obs_history.shape[1]
    current_idx = env._imu_obs_history_idx % max_lag
    lag_timesteps = env._imu_lag_timesteps
    lagged_idx = (current_idx - lag_timesteps) % max_lag
    lagged_imu = env._imu_obs_history[torch.arange(env.num_envs), lagged_idx]
    
    # Return only the angular velocity (last 3 values)
    return lagged_imu[:, 3:]


def joint_pos_rel_lagged(env: ManagerBasedEnv) -> torch.Tensor:
    """Get joint positions with motor lag applied.
    
    This function applies the motor observation lag that was set during environment reset.
    
    Args:
        env: The environment instance.
        
    Returns:
        Tensor of shape (num_envs, num_joints) containing lagged joint positions.
    """
    # Get current observation
    asset = env.scene["robot"]
    current_joint_pos = asset.data.joint_pos - asset.data.default_joint_pos
    
    # If lag is not initialized, return current observation
    if not hasattr(env, '_motor_lag_timesteps') or not hasattr(env, '_motor_obs_history'):
        return current_joint_pos
    
    # Get current joint velocity
    current_joint_vel = asset.data.joint_vel
    
    # Combine position and velocity
    current_motor = torch.cat([current_joint_pos, current_joint_vel], dim=-1)
    
    # Update history buffer
    max_lag = env._motor_obs_history.shape[1]
    current_idx = env._motor_obs_history_idx % max_lag
    
    env._motor_obs_history[torch.arange(env.num_envs), current_idx] = current_motor
    env._motor_obs_history_idx = (env._motor_obs_history_idx + 1) % max_lag
    
    # Retrieve lagged observations
    lag_timesteps = env._motor_lag_timesteps
    lagged_idx = (current_idx - lag_timesteps) % max_lag
    lagged_motor = env._motor_obs_history[torch.arange(env.num_envs), lagged_idx]
    
    # Return only joint positions (first half)
    num_joints = asset.num_joints
    return lagged_motor[:, :num_joints]


def joint_vel_rel_lagged(env: ManagerBasedEnv) -> torch.Tensor:
    """Get joint velocities with motor lag applied.
    
    This function applies the motor observation lag that was set during environment reset.
    
    Args:
        env: The environment instance.
        
    Returns:
        Tensor of shape (num_envs, num_joints) containing lagged joint velocities.
    """
    # Get current observation
    asset = env.scene["robot"]
    current_joint_vel = asset.data.joint_vel
    
    # If lag is not initialized, return current observation
    if not hasattr(env, '_motor_lag_timesteps') or not hasattr(env, '_motor_obs_history'):
        return current_joint_vel
    
    # The motor history already contains velocity, just retrieve the lagged values
    max_lag = env._motor_obs_history.shape[1]
    current_idx = env._motor_obs_history_idx % max_lag
    lag_timesteps = env._motor_lag_timesteps
    lagged_idx = (current_idx - lag_timesteps) % max_lag
    lagged_motor = env._motor_obs_history[torch.arange(env.num_envs), lagged_idx]
    
    # Return only joint velocities (second half)
    num_joints = asset.num_joints
    return lagged_motor[:, num_joints:]