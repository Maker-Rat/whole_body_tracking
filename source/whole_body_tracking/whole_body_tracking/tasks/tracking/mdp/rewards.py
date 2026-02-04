from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def feet_slip_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize feet slipping when in contact with the ground.

    The penalty is the square-root of the XY speed of each foot, gated by
    whether that foot is currently in contact.  Using sqrt softens the
    gradient for very fast slips so the signal stays learnable.

    Args:
        env: The RL environment.
        sensor_cfg: Configuration for the contact sensor (resolves body_ids
            on the *sensor* side).
        asset_cfg: Configuration for the robot articulation with the same
            foot body names (resolves body_ids on the *articulation* side).
            This is necessary because the sensor's body_ids and the
            articulation's body_ids are independent index spaces.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # Contact mask: shape (num_envs, num_feet)
    # Uses sensor-side body_ids to index into the sensor's force history.
    is_contact = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1) > 1.0

    # Foot XY velocities: shape (num_envs, num_feet, 2)
    # Uses articulation-side body_ids to index into the robot's body states.
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    # sqrt of XY speed, gated by contact, summed over feet
    foot_speed = torch.norm(feet_vel_xy, dim=-1)                          # (num_envs, num_feet)
    penalty = torch.sum(is_contact * torch.sqrt(foot_speed), dim=-1)      # (num_envs,)

    return penalty


def feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long air-time per foot, clamped to [0, 0.5] s.

    Only fires on the *first contact* step so that each stride contributes
    exactly once.  The clamp prevents a single very-long hop from
    dominating the reward signal.

    Args:
        env: The RL environment.
        sensor_cfg: Configuration for the contact sensor with foot body names.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # first_contact: 1 on the step a foot first touches down, 0 otherwise
    first_contact = contact_sensor.compute_first_contact(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    # How long each foot was in the air before that contact
    air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    reward = torch.sum(air_time.clamp(0, 0.5) * first_contact, dim=-1)   # (num_envs,)
    return reward


def feet_contact_forces_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_contact_force: float,
) -> torch.Tensor:
    """Penalize contact forces that exceed a maximum threshold.

    Any per-foot force magnitude above *max_contact_force* contributes to
    the penalty.  The excess is clipped to [0, 350] N to keep gradients
    stable for very hard impacts.

    Args:
        env: The RL environment.
        sensor_cfg: Configuration for the contact sensor with foot body names.
        max_contact_force: Force threshold (N) below which no penalty is
            applied.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Per-foot force magnitudes: (num_envs, num_feet)
    force_magnitudes = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)

    # Excess above threshold, clipped to a safe upper bound
    penalty = torch.sum((force_magnitudes - max_contact_force).clip(0, 350), dim=-1)  # (num_envs,)
    return penalty


def impact_reduction(
    env: ManagerBasedRLEnv,
    sensor_name: str = "contact_forces",
    asset_name: str = "robot",
    body_names: list[str] | None = None,
    delta_v_max_squared: float = 2.0,
) -> torch.Tensor:
    """Penalize changes in vertical (z) foot velocity to reduce impact noise.

    Based on Disney's Olaf paper. Formula: -Σi∈{feet} min(Δv²i,z , Δv²max)
    The saturation prevents large velocity changes during contact resolution
    from destabilizing critic learning.

    Args:
        env: The RL environment.
        sensor_name: Name of the contact sensor asset.
        asset_name: Name of the robot articulation asset.
        body_names: Names of the feet bodies to track. If None, uses all bodies.
        delta_v_max_squared: Maximum squared velocity change threshold (m²/s²).
    """
    asset = env.scene[asset_name]

    # Get body indices
    if body_names is None:
        body_ids = slice(None)
    else:
        body_ids = [i for i, name in enumerate(asset.data.body_names) if name in body_names]

    # Get current foot velocities: (num_envs, num_feet, 3)
    foot_vel_w = asset.data.body_lin_vel_w[:, body_ids]

    # Initialize tracking on first step
    if not hasattr(env, "_last_foot_vel_z"):
        env._last_foot_vel_z = torch.zeros(
            env.num_envs,
            foot_vel_w.shape[1],
            device=env.device,
            dtype=foot_vel_w.dtype,
        )

    # Compute change in vertical velocity: (num_envs, num_feet)
    delta_v_z = foot_vel_w[:, :, 2] - env._last_foot_vel_z
    delta_v_z_squared = delta_v_z**2

    # Apply saturation and sum over feet
    saturated_delta = torch.minimum(
        delta_v_z_squared,
        torch.tensor(delta_v_max_squared, device=env.device, dtype=delta_v_z_squared.dtype),
    )
    total_impact = torch.sum(saturated_delta, dim=1)  # (num_envs,)

    # Update tracking for next step
    env._last_foot_vel_z = foot_vel_w[:, :, 2].clone()

    # Return penalty (positive value to be negated by negative weight, or positive weight for negative contribution)
    return total_impact