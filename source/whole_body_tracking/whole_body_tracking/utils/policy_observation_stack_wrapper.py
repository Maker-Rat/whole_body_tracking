import numpy as np
from gymnasium import Wrapper
from whole_body_tracking.utils.observation_stack_wrapper import ObservationStackWrapper

class PolicyObservationStackWrapper(Wrapper):
    """
    Gymnasium-style wrapper for stacking policy observations using ObservationStackWrapper.
    Supports environments that return dict observations by stacking a specified key.
    """
    def __init__(self, env, num_stack=10, concat_axis=-1, key="policy"):
        super().__init__(env)
        self.obs_stack = ObservationStackWrapper(num_stack=num_stack, concat_axis=concat_axis)
        self._stacked_obs = None
        self.key = key

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_to_stack = obs[self.key] if isinstance(obs, dict) else obs
        stacked = self.obs_stack.reset(obs_to_stack)
        # Return stacked obs in same structure as input
        if isinstance(obs, dict):
            obs = dict(obs)
            obs[self.key] = stacked
            return obs, info
        else:
            return stacked, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_to_stack = obs[self.key] if isinstance(obs, dict) else obs
        stacked = self.obs_stack.step(obs_to_stack)
        if isinstance(obs, dict):
            obs = dict(obs)
            obs[self.key] = stacked
            return obs, reward, terminated, truncated, info
        else:
            return stacked, reward, terminated, truncated, info
