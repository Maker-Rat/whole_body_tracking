import numpy as np
from collections import deque

class ObservationStackWrapper:
    """
    A simple wrapper for stacking vector observations (not images).
    Usage: Wrap your environment or observation-producing object with this class.
    """
    def __init__(self, num_stack=4, concat_axis=-1):
        self.num_stack = num_stack
        self.concat_axis = concat_axis
        self.stack = None

    def reset(self, obs):
        # obs: initial observation (np.ndarray or torch.Tensor)
        self.stack = deque([obs.copy() for _ in range(self.num_stack)], maxlen=self.num_stack)
        return self._get_stacked_obs()

    def step(self, obs):
        self.stack.append(obs.copy())
        return self._get_stacked_obs()

    def _get_stacked_obs(self):
        # Stack along the specified axis
        if isinstance(self.stack[0], np.ndarray):
            return np.concatenate(list(self.stack), axis=self.concat_axis)
        else:
            # Assume torch.Tensor
            import torch
            return torch.cat(list(self.stack), dim=self.concat_axis)
