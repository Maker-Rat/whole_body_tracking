import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR


# PM01 motor parameters (updated for new PD config)
# STIFFNESS_HIP_PITCH = 70
# STIFFNESS_HIP_ROLL = 70
# STIFFNESS_HIP_YAW = 50
# STIFFNESS_KNEE_PITCH = 70
# STIFFNESS_ANKLE_PITCH = 20
# STIFFNESS_ANKLE_ROLL = 20
# STIFFNESS_WAIST_YAW = 50
# STIFFNESS_HEAD_YAW = 50
# STIFFNESS_ARM_ALL = 155

# DAMPING_HIP_PITCH = 7.0
# DAMPING_HIP_ROLL = 7.0
# DAMPING_HIP_YAW = 5.0
# DAMPING_KNEE_PITCH = 7.0
# DAMPING_ANKLE_PITCH = 0.2
# DAMPING_ANKLE_ROLL = 0.2
# DAMPING_WAIST_YAW = 5.0
# DAMPING_HEAD_YAW = 5.0
# DAMPING_ARM_ALL = 9.8

STIFFNESS_HIP_PITCH = 179
STIFFNESS_HIP_ROLL = 179
STIFFNESS_HIP_YAW = 155
STIFFNESS_KNEE_PITCH = 179
STIFFNESS_ANKLE_PITCH = 155
STIFFNESS_ANKLE_ROLL = 155
STIFFNESS_WAIST_YAW = 155
STIFFNESS_HEAD_YAW = 155
STIFFNESS_ARM_ALL = 155

# Damping
DAMPING_HIP_PITCH = 11.4
DAMPING_HIP_ROLL = 11.4
DAMPING_HIP_YAW = 9.8
DAMPING_KNEE_PITCH = 11.4
DAMPING_ANKLE_PITCH = 9.8
DAMPING_ANKLE_ROLL = 9.8
DAMPING_WAIST_YAW = 9.8
DAMPING_HEAD_YAW = 9.8
DAMPING_ARM_ALL = 9.8
    
    
PM01_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/engineai_description/urdf/pm01_mod.urdf",
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
        pos=(0.0, 0.0, 0.82),
        joint_pos={
            "j.*_hip_pitch_.*": -0.12,
            "j.*_knee_pitch_.*": 0.24,
            "j.*_ankle_pitch_.*": -0.12,
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
        # Hip pitch (high torque)
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=["j00_hip_pitch_l", "j06_hip_pitch_r"],
            effort_limit_sim=164.0,
            velocity_limit_sim=26.3,
            stiffness=STIFFNESS_HIP_PITCH,
            damping=DAMPING_HIP_PITCH,
        ),
        # Hip roll
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=["j01_hip_roll_l", "j07_hip_roll_r"],
            effort_limit_sim=164.0,
            velocity_limit_sim=26.3,
            stiffness=STIFFNESS_HIP_ROLL,
            damping=DAMPING_HIP_ROLL,
        ),
        # Hip yaw
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=["j02_hip_yaw_l", "j08_hip_yaw_r"],
            effort_limit_sim=61.0,
            velocity_limit_sim=35.2,
            stiffness=STIFFNESS_HIP_YAW,
            damping=DAMPING_HIP_YAW,
        ),
        # Knee pitch (high torque)
        "knee_pitch": ImplicitActuatorCfg(
            joint_names_expr=["j03_knee_pitch_l", "j09_knee_pitch_r"],
            effort_limit_sim=164.0,
            velocity_limit_sim=26.3,
            stiffness=STIFFNESS_KNEE_PITCH,
            damping=DAMPING_KNEE_PITCH,
        ),
        # Ankle pitch
        "ankle_pitch": ImplicitActuatorCfg(
            joint_names_expr=["j04_ankle_pitch_l", "j10_ankle_pitch_r"],
            effort_limit_sim=61.0,
            velocity_limit_sim=35.2,
            stiffness=STIFFNESS_ANKLE_PITCH,
            damping=DAMPING_ANKLE_PITCH,
        ),
        # Ankle roll
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=["j05_ankle_roll_l", "j11_ankle_roll_r"],
            effort_limit_sim=61.0,
            velocity_limit_sim=35.2,
            stiffness=STIFFNESS_ANKLE_ROLL,
            damping=DAMPING_ANKLE_ROLL,
        ),
        # Waist
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["j12_waist_yaw"],
            effort_limit_sim=61.0,
            velocity_limit_sim=35.2,
            stiffness=STIFFNESS_WAIST_YAW,
            damping=DAMPING_WAIST_YAW,
        ),
        # Arms
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "j13_shoulder_pitch_l", "j14_shoulder_roll_l", "j15_shoulder_yaw_l", "j16_elbow_pitch_l", "j17_elbow_yaw_l",
                "j18_shoulder_pitch_r", "j19_shoulder_roll_r", "j20_shoulder_yaw_r", "j21_elbow_pitch_r", "j22_elbow_yaw_r"
            ],
            effort_limit_sim=61.0,
            velocity_limit_sim=35.2,
            stiffness=STIFFNESS_ARM_ALL,
            damping=DAMPING_ARM_ALL,
        ),
        # Head
        "head": ImplicitActuatorCfg(
            joint_names_expr=["j23_head_yaw"],
            effort_limit_sim=61.0,
            velocity_limit_sim=35.2,
            stiffness=STIFFNESS_HEAD_YAW,
            damping=DAMPING_HEAD_YAW,
        ),
    },
)

# Compute action scale for each actuator group (0.25 * effort / stiffness)
PM01_ACTION_SCALE = {}
for group in PM01_CFG.actuators.values():
    e = group.effort_limit_sim
    s = group.stiffness
    names = group.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            PM01_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
