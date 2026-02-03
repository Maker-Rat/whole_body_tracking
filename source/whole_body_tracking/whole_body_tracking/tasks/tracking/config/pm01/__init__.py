import gymnasium as gym
from . import agents, flat_env_cfg, rough_env_cfg
from .spawn_height_env import SpawnHeightEnv  # Add this import

##
# Register Gym environments.
##
gym.register(
    id="Tracking-Flat-PM01-v0",
    entry_point=SpawnHeightEnv,  # Changed from "isaaclab.envs:ManagerBasedRLEnv"
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.PM01FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PM01FlatPPORunnerCfg",
    },
)
gym.register(
    id="Tracking-Flat-PM01-Wo-State-Estimation-v0",
    entry_point=SpawnHeightEnv,  # Changed
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.PM01FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PM01FlatPPORunnerCfg",
    },
)
gym.register(
    id="Tracking-Flat-PM01-Low-Freq-v0",
    entry_point=SpawnHeightEnv,  # Changed
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.PM01FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PM01FlatLowFreqPPORunnerCfg",
    },
)

##
# Rough terrain variants
##
gym.register(
    id="Tracking-Rough-PM01-v0",
    entry_point=SpawnHeightEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PM01RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PM01FlatPPORunnerCfg",
    },
)
gym.register(
    id="Tracking-Rough-PM01-Wo-State-Estimation-v0",
    entry_point=SpawnHeightEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PM01RoughWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PM01FlatPPORunnerCfg",
    },
)
gym.register(
    id="Tracking-Rough-PM01-Low-Freq-v0",
    entry_point=SpawnHeightEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PM01RoughLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PM01FlatLowFreqPPORunnerCfg",
    },
)