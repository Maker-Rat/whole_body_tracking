import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# PM01 motor parameters from MuJoCo XML spec
# High torque motor (164 Nm): hip_pitch, hip_roll, knee_pitch
ARMATURE_HIGH_TORQUE = 0.045325
# Low torque motor (52-61 Nm): hip_yaw, ankle, waist, arms, head
ARMATURE_LOW_TORQUE = 0.039175

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_HIGH_TORQUE = ARMATURE_HIGH_TORQUE * NATURAL_FREQ**2
STIFFNESS_LOW_TORQUE = ARMATURE_LOW_TORQUE * NATURAL_FREQ**2

DAMPING_HIGH_TORQUE = 2.0 * DAMPING_RATIO * ARMATURE_HIGH_TORQUE * NATURAL_FREQ
DAMPING_LOW_TORQUE = 2.0 * DAMPING_RATIO * ARMATURE_LOW_TORQUE * NATURAL_FREQ

PM01_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/engineai_description/urdf/pm01.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.82),  # PM01 default base height
        joint_pos={
            # Leg initial pose from MuJoCo config
            "j.*_hip_pitch_.*": -0.12,
            "j.*_knee_pitch_.*": 0.24,
            "j.*_ankle_pitch_.*": -0.12,
            # Arms relaxed
            "j.*_elbow_pitch_.*": 0.6,
            "j13_shoulder_pitch_l": 0.2,
            "j14_shoulder_roll_l": 0.2,
            "j18_shoulder_pitch_r": 0.2,
            "j19_shoulder_roll_r": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # High torque joints (164 Nm): hip_pitch, hip_roll, knee_pitch
        "legs_high_torque": ImplicitActuatorCfg(
            joint_names_expr=[
                "j00_hip_pitch_l",
                "j01_hip_roll_l",
                "j03_knee_pitch_l",
                "j06_hip_pitch_r",
                "j07_hip_roll_r",
                "j09_knee_pitch_r",
            ],
            effort_limit_sim=164.0,
            velocity_limit_sim=26.3,
            stiffness=STIFFNESS_HIGH_TORQUE,
            damping=DAMPING_HIGH_TORQUE,
            armature=ARMATURE_HIGH_TORQUE,
        ),
        # Low torque hip joints (52 Nm): hip_yaw
        "legs_low_torque": ImplicitActuatorCfg(
            joint_names_expr=[
                "j02_hip_yaw_l",
                "j08_hip_yaw_r",
            ],
            effort_limit_sim=52.0,
            velocity_limit_sim=35.2,
            stiffness=STIFFNESS_LOW_TORQUE,
            damping=DAMPING_LOW_TORQUE,
            armature=ARMATURE_LOW_TORQUE,
        ),
        # Feet (52 Nm): ankle_pitch, ankle_roll
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "j04_ankle_pitch_l",
                "j05_ankle_roll_l",
                "j10_ankle_pitch_r",
                "j11_ankle_roll_r",
            ],
            effort_limit_sim=52.0,
            velocity_limit_sim=35.2,
            stiffness=STIFFNESS_LOW_TORQUE,
            damping=DAMPING_LOW_TORQUE,
            armature=ARMATURE_LOW_TORQUE,
        ),
        # Waist (52 Nm)
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["j12_waist_yaw"],
            effort_limit_sim=52.0,
            velocity_limit_sim=35.2,
            stiffness=STIFFNESS_LOW_TORQUE,
            damping=DAMPING_LOW_TORQUE,
            armature=ARMATURE_LOW_TORQUE,
        ),
        # Arms (52 Nm)
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "j13_shoulder_pitch_l",
                "j14_shoulder_roll_l",
                "j15_shoulder_yaw_l",
                "j16_elbow_pitch_l",
                "j17_elbow_yaw_l",
                "j18_shoulder_pitch_r",
                "j19_shoulder_roll_r",
                "j20_shoulder_yaw_r",
                "j21_elbow_pitch_r",
                "j22_elbow_yaw_r",
            ],
            effort_limit_sim=52.0,
            velocity_limit_sim=35.2,
            stiffness=STIFFNESS_LOW_TORQUE,
            damping=DAMPING_LOW_TORQUE,
            armature=ARMATURE_LOW_TORQUE,
        ),
        # Head (52 Nm)
        "head": ImplicitActuatorCfg(
            joint_names_expr=["j23_head_yaw"],
            effort_limit_sim=52.0,
            velocity_limit_sim=35.2,
            stiffness=STIFFNESS_LOW_TORQUE,
            damping=DAMPING_LOW_TORQUE,
            armature=ARMATURE_LOW_TORQUE,
        ),
    },
)

# Compute action scale for each actuator group
PM01_ACTION_SCALE = {}
for a in PM01_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            PM01_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
