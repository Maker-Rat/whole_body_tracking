from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen

from whole_body_tracking.robots.pm01 import PM01_ACTION_SCALE, PM01_CFG
from whole_body_tracking.tasks.tracking.config.pm01.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from whole_body_tracking.utils.observation_stack_wrapper import ObservationStackWrapper
import whole_body_tracking.tasks.tracking.mdp as mdp



@configclass
class PM01RoughEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Override terrain to mostly flat with some rough patches
        self.scene.terrain = self.scene.terrain.replace(
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(64.0, 64.0),
                border_width=5.0,
                num_rows=2,
                num_cols=2,
                horizontal_scale=0.2,
                vertical_scale=0.005,
                sub_terrains={
                    # Flat terrain (70% of patches)
                    "flat": terrain_gen.MeshPlaneTerrainCfg(
                        proportion=0.5,
                    ),
                    # Rough patches (30% of patches)
                    "rough": terrain_gen.HfRandomUniformTerrainCfg(
                        proportion=0.5,
                        noise_range=(0.01, 0.04),
                        noise_step=0.01,
                        border_width=0.2
                    ),
                },
            ),
        )

        self.scene.robot = PM01_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = PM01_ACTION_SCALE
        self.commands.motion.anchor_body_name = "link_base"
        # PM01 body names for motion tracking
        # Following the IK config mapping pattern: major links for tracking
        self.commands.motion.body_names = [
            "link_base",              # pelvis/root
            "link_hip_roll_l",        # left hip
            "link_knee_pitch_l",      # left knee
            "link_ankle_roll_l",      # left ankle
            "link_hip_roll_r",        # right hip
            "link_knee_pitch_r",      # right knee
            "link_ankle_roll_r",      # right ankle
            "link_torso_yaw",         # torso
            "link_shoulder_roll_l",   # left shoulder
            "link_elbow_pitch_l",     # left elbow
            "link_elbow_yaw_l",       # left wrist/hand
            "link_shoulder_roll_r",   # right shoulder
            "link_elbow_pitch_r",     # right elbow
            "link_elbow_yaw_r",       # right wrist/hand
        ]

        # Override reward: undesired_contacts for PM01 body names
        # Only allow contacts on feet (ankle_roll) and hands (elbow_yaw)
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        r"^(?!link_ankle_roll_l$)(?!link_ankle_roll_r$)(?!link_elbow_yaw_l$)(?!link_elbow_yaw_r$).+$"
                    ],
                ),
                "threshold": 1.0,
            },
        )

        # Override foot rewards for PM01 body names
        self.rewards.feet_slip = RewTerm(
            func=mdp.feet_slip_penalty,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["link_ankle_roll_l", "link_ankle_roll_r"]
                ),
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["link_ankle_roll_l", "link_ankle_roll_r"]
                ),
            },
        )
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time,
            weight=1.5,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["link_ankle_roll_l", "link_ankle_roll_r"]
                ),
            },
        )
        self.rewards.feet_contact_forces = RewTerm(
            func=mdp.feet_contact_forces_penalty,
            weight=-0.02,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["link_ankle_roll_l", "link_ankle_roll_r"]
                ),
                "max_contact_force": 500.0,
            },
        )

        # Override termination: ee_body_pos for PM01 end-effector body names
        self.terminations.ee_body_pos = DoneTerm(
            func=mdp.bad_motion_body_pos_z_only,
            params={
                "command_name": "motion",
                "threshold": 0.35,
                "body_names": [
                    "link_ankle_roll_l",
                    "link_ankle_roll_r",
                    "link_elbow_yaw_l",
                    "link_elbow_yaw_r",
                ],
            },
        )


@configclass
class PM01RoughWoStateEstimationEnvCfg(PM01RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class PM01RoughLowFreqEnvCfg(PM01RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
