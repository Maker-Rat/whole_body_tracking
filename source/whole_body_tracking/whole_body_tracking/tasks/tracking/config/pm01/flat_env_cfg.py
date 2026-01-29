from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from whole_body_tracking.robots.pm01 import PM01_ACTION_SCALE, PM01_CFG
from whole_body_tracking.tasks.tracking.config.pm01.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from whole_body_tracking.utils.observation_stack_wrapper import ObservationStackWrapper
import whole_body_tracking.tasks.tracking.mdp as mdp



@configclass
class PM01FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

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

        # Override termination: ee_body_pos for PM01 end-effector body names
        self.terminations.ee_body_pos = DoneTerm(
            func=mdp.bad_motion_body_pos_z_only,
            params={
                "command_name": "motion",
                "threshold": 0.25,
                "body_names": [
                    "link_ankle_roll_l",
                    "link_ankle_roll_r",
                    "link_elbow_yaw_l",
                    "link_elbow_yaw_r",
                ],
            },
        )

        # Add observation stacking wrapper (10 frames)
        # self.wrappers = getattr(self, "wrappers", [])
        # self.wrappers.append(
        #     ObservationStackWrapper(
        #         num_stack=10,  # Number of stacked observations (set to 10 for stacking)
        #         concat_axis=-1  # Stack along the last axis (feature dimension)
        #     )
        # )


@configclass
class PM01FlatWoStateEstimationEnvCfg(PM01FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class PM01FlatLowFreqEnvCfg(PM01FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
