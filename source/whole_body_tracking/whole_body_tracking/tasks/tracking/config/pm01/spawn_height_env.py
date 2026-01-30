"""Custom environment wrapper to force spawn height."""

import torch
from isaaclab.envs import ManagerBasedRLEnv


class SpawnHeightEnv(ManagerBasedRLEnv):
    """Wrapper that forces robot spawn height after every reset and step."""
    
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._spawn_height = 0.95  # Desired spawn height
        self._step_counter = 0
        self._enforce_height_next_step = False
        self._enforce_height_next_step_mask = None
        
    def reset(self, seed=None, options=None):
        """Reset with forced spawn height."""
        obs, extras = super().reset(seed=seed, options=options)
        self._enforce_height_next_step = True
        return obs, extras

    def _reset_idx(self, env_ids: torch.Tensor):
        """Override reset to force height after parent reset."""
        super()._reset_idx(env_ids)
        if self._enforce_height_next_step_mask is None or self._enforce_height_next_step_mask.shape[0] != self.num_envs:
            self._enforce_height_next_step_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._enforce_height_next_step_mask[env_ids] = True

    def step(self, action):
        """Override step to maintain spawn height."""
        if self._enforce_height_next_step:
            self._enforce_spawn_height(torch.arange(self.num_envs, device=self.device))
            self._enforce_height_next_step = False
        if self._enforce_height_next_step_mask is not None and self._enforce_height_next_step_mask.any():
            env_ids = torch.where(self._enforce_height_next_step_mask)[0]
            self._enforce_spawn_height(env_ids)
            self._enforce_height_next_step_mask[env_ids] = False
        return super().step(action)
    
    def _enforce_spawn_height(self, env_ids: torch.Tensor):
        """Enforce spawn height by setting root state and syncing with physics."""
        robot = self.scene["robot"]
        if env_ids.device != self.device:
            env_ids = env_ids.to(self.device)
        robot.data.root_state_w[env_ids, 2] = self._spawn_height      # Z position
        robot.data.root_state_w[env_ids, 7:13] = 0.0                  # Linear and angular velocities
        robot.data.root_pos_w[env_ids, 2] = self._spawn_height        # Z position (redundant but safe)
        robot.data.root_lin_vel_w[env_ids, :] = 0.0                   # Linear velocity
        robot.data.root_ang_vel_w[env_ids, :] = 0.0                   # Angular velocity
        robot.data.default_root_state[env_ids, 2] = self._spawn_height
        robot.write_root_state_to_sim(robot.data.root_state_w[env_ids], env_ids=env_ids)