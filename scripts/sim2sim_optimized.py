"""Fixed Sim2Sim script for PM01 - matches Isaac Lab exactly.

CRITICAL FIXES APPLIED:
1. Joint velocity scaling: 0.05 (was missing - causes 20x error!)
2. Torque limits: 61.0 Nm for low-torque joints (was 41.0)
3. Observation clipping: ±100.0 (matches training)
4. NO action clipping (matches training)
5. NaN detection and recovery

Usage:
    python sim2sim_pm01_fixed.py --onnx_path model.onnx --motion_file walk.npz --config rl_basic_param.yaml
"""

import argparse
import numpy as np
import mujoco
import mujoco.viewer
import yaml
from pathlib import Path

# ============== Configuration ==============

class PM01Config:
    """PM01 configuration matching C++ implementation exactly."""
    
    # Simulation
    dt = 0.002  # MuJoCo timestep (500 Hz)
    decimation = 10  # Policy runs at 50 Hz
    sim_duration = 60.0
    
    # Robot
    num_actions = 24
    num_observations = 75
    
    # MuJoCo joint names (URDF order)
    mujoco_joint_names = [
        'j00_hip_pitch_l', 'j01_hip_roll_l', 'j02_hip_yaw_l', 'j03_knee_pitch_l',
        'j04_ankle_pitch_l', 'j05_ankle_roll_l', 'j06_hip_pitch_r', 'j07_hip_roll_r',
        'j08_hip_yaw_r', 'j09_knee_pitch_r', 'j10_ankle_pitch_r', 'j11_ankle_roll_r',
        'j12_waist_yaw', 'j13_shoulder_pitch_l', 'j14_shoulder_roll_l', 'j15_shoulder_yaw_l',
        'j16_elbow_pitch_l', 'j17_elbow_yaw_l', 'j18_shoulder_pitch_r', 'j19_shoulder_roll_r',
        'j20_shoulder_yaw_r', 'j21_elbow_pitch_r', 'j22_elbow_yaw_r', 'j23_head_yaw'
    ]
    
    # Isaac Lab joint order
    isaac_joint_names = [
        'j00_hip_pitch_l', 'j06_hip_pitch_r', 'j12_waist_yaw', 'j01_hip_roll_l',
        'j07_hip_roll_r', 'j13_shoulder_pitch_l', 'j18_shoulder_pitch_r', 'j23_head_yaw',
        'j02_hip_yaw_l', 'j08_hip_yaw_r', 'j14_shoulder_roll_l', 'j19_shoulder_roll_r',
        'j03_knee_pitch_l', 'j09_knee_pitch_r', 'j15_shoulder_yaw_l', 'j20_shoulder_yaw_r',
        'j04_ankle_pitch_l', 'j10_ankle_pitch_r', 'j16_elbow_pitch_l', 'j21_elbow_pitch_r',
        'j05_ankle_roll_l', 'j11_ankle_roll_r', 'j17_elbow_yaw_l', 'j22_elbow_yaw_r'
    ]
    
    # Mapping arrays
    MUJOCO_TO_ISAAC = np.array([
        0, 3, 8, 12, 16, 20, 1, 4, 9, 13, 17, 21,
        2, 5, 10, 14, 18, 22, 6, 11, 15, 19, 23, 7
    ], dtype=np.int32)
    
    ISAAC_TO_MUJOCO = np.array([
        0, 6, 12, 1, 7, 13, 18, 23, 2, 8, 14, 19,
        3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22
    ], dtype=np.int32)
    
    @classmethod
    def from_yaml(cls, yaml_path):
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            params = yaml.safe_load(f)
        
        cfg = cls()
        cfg.action_scales = np.array(params['action_scale'], dtype=np.float32)
        cfg.default_joint_pos = np.array(params['default_joint_q'], dtype=np.float32)
        cfg.joint_kp = np.array(params['joint_kp'], dtype=np.float32)
        cfg.joint_kd = np.array(params['joint_kd'], dtype=np.float32)
        
        # Observation parameters from YAML
        cfg.obs_clip = float(params.get('observation_clip', 100.0))
        
        # CRITICAL: Joint velocity scaling from YAML
        cfg.vel_scale = float(params.get('observation_scale_dof_vel', 0.05))
        
        assert len(cfg.action_scales) == 24
        assert len(cfg.default_joint_pos) == 24
        
        print(f"✓ Loaded config from {yaml_path}")
        print(f"  Obs clip: {cfg.obs_clip}")
        print(f"  Vel scale: {cfg.vel_scale} (CRITICAL - was missing!)")
        print(f"  Actions NOT clipped (matches training)")
        
        return cfg


class MotionLoader:
    """Motion data loader."""
    
    def __init__(self, motion_file: str):
        data = np.load(motion_file)
        self.fps = float(data['fps'])
        self.joint_pos = data['joint_pos'].astype(np.float32)
        self.joint_vel = data['joint_vel'].astype(np.float32)
        self.body_pos_w = data['body_pos_w'].astype(np.float32)
        self.body_quat_w = data['body_quat_w'].astype(np.float32)
        self.num_frames = self.joint_pos.shape[0]
        
        print(f"✓ Loaded motion: {self.num_frames} frames at {self.fps} fps")


class RobotState:
    """Preallocated robot state with NaN detection."""
    
    def __init__(self):
        self.base_pos = np.zeros(3, dtype=np.float32)
        self.base_quat = np.zeros(4, dtype=np.float32)
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.joint_pos_isaac = np.zeros(24, dtype=np.float32)
        self.joint_vel_isaac = np.zeros(24, dtype=np.float32)
        self.joint_pos_mujoco = np.zeros(24, dtype=np.float32)
        self.joint_vel_mujoco = np.zeros(24, dtype=np.float32)
        self.has_nan = False
    
    def update_from_mujoco(self, data, cfg):
        """Update state from MuJoCo data with NaN checking."""
        self.base_pos[:] = data.qpos[:3]
        self.base_quat[:] = data.qpos[3:7]
        
        # Transform velocities to body frame
        world_lin_vel = data.qvel[:3]
        world_ang_vel = data.qvel[3:6]
        
        w, x, y, z = self.base_quat
        rot_mat = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ], dtype=np.float32)
        
        self.base_lin_vel[:] = rot_mat.T @ world_lin_vel
        self.base_ang_vel[:] = rot_mat.T @ world_ang_vel
        
        self.joint_pos_mujoco[:] = data.qpos[7:31]
        self.joint_vel_mujoco[:] = data.qvel[6:30]
        self.joint_pos_isaac[:] = self.joint_pos_mujoco[cfg.ISAAC_TO_MUJOCO]
        self.joint_vel_isaac[:] = self.joint_vel_mujoco[cfg.ISAAC_TO_MUJOCO]
        
        # Check for NaN
        self.has_nan = (
            np.isnan(self.base_pos).any() or
            np.isnan(self.base_quat).any() or
            np.isnan(self.base_lin_vel).any() or
            np.isnan(self.base_ang_vel).any() or
            np.isnan(self.joint_pos_isaac).any() or
            np.isnan(self.joint_vel_isaac).any()
        )


class ObservationBuilder:
    """Observation builder matching training exactly.
    
    From tracking_env_cfg.py:
    - base_ang_vel: noise=Unoise(n_min=-0.2, n_max=0.2)
    - joint_pos: noise=Unoise(n_min=-0.01, n_max=0.01)
    - joint_vel: noise=Unoise(n_min=-0.5, n_max=0.5) with SCALE 0.05
    - actions: no noise
    
    CRITICAL: Joint velocities are scaled by 0.05 in Isaac Lab!
    See: observation_scale_dof_vel: 0.05 in YAML
    
    Note: Noise is applied during TRAINING but NOT during evaluation/deployment
    We don't add noise here since this is deployment
    """
    
    def __init__(self, cfg, include_base_lin_vel=False):
        self.cfg = cfg
        self.include_base_lin_vel = include_base_lin_vel
        self.obs_size = cfg.num_observations + (3 if include_base_lin_vel else 0)
        self.obs = np.zeros(self.obs_size, dtype=np.float32)
        self.joint_pos_rel = np.zeros(24, dtype=np.float32)
        
        # Get velocity scale from YAML
        self.vel_scale = cfg.vel_scale
        
        if include_base_lin_vel:
            print("⚠ ObservationBuilder: Including base_lin_vel (78 dims)")
        else:
            print("✓ ObservationBuilder: Standard config (75 dims)")
        print(f"  Observation clipping: ±{cfg.obs_clip}")
        print(f"  Joint velocity scale: {self.vel_scale} (CRITICAL!)")
        print("  No noise added (deployment mode)")
    
    def build(self, robot_state, last_action):
        """Build observation with clipping (matches training)."""
        idx = 0
        
        if self.include_base_lin_vel:
            self.obs[idx:idx+3] = robot_state.base_lin_vel
            idx += 3
        
        self.obs[idx:idx+3] = robot_state.base_ang_vel
        idx += 3
        
        np.subtract(robot_state.joint_pos_isaac, self.cfg.default_joint_pos, 
                   out=self.joint_pos_rel)
        self.obs[idx:idx+24] = self.joint_pos_rel
        idx += 24
        
        # CRITICAL: Scale joint velocities by 0.05 (matches Isaac Lab training)
        self.obs[idx:idx+24] = robot_state.joint_vel_isaac * self.vel_scale
        idx += 24
        
        self.obs[idx:idx+24] = last_action
        idx += 24
        
        # Clip observations to ±100.0 (from YAML config)
        np.clip(self.obs, -self.cfg.obs_clip, self.cfg.obs_clip, out=self.obs)
        
        return self.obs


class PDController:
    """PD controller matching Isaac Lab pm01.py exactly.
    
    CRITICAL: Torque limits from pm01.py (effort_limit_sim):
    - High torque (hip pitch, hip roll, knee pitch): 164.0 Nm
    - Low torque (all others): 61.0 Nm
    
    NOTE: URDF says 52 Nm but Isaac Lab pm01.py uses 61.0 Nm
    We match Isaac Lab since that's what the policy was trained with!
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.target_isaac = np.zeros(24, dtype=np.float32)
        self.error = np.zeros(24, dtype=np.float32)
        self.tau_isaac = np.zeros(24, dtype=np.float32)
        self.tau_mujoco = np.zeros(24, dtype=np.float32)
        
        # Match Isaac Lab pm01.py: effort_limit_sim=61.0 for low torque
        self.tau_limit = np.full(24, 61.0, dtype=np.float32)
        
        # High torque joints in Isaac order: hip_pitch_l, hip_pitch_r, knee_pitch_l, knee_pitch_r
        # Isaac indices: 0, 1, 12, 13
        high_torque_indices = [0, 1, 12, 13]
        self.tau_limit[high_torque_indices] = 164.0
        
        print("✓ PDController initialized (matching Isaac Lab pm01.py):")
        print(f"  High torque (164.0 Nm): indices {high_torque_indices}")
        print(f"    {[cfg.isaac_joint_names[i] for i in high_torque_indices]}")
        print(f"  Low torque (61.0 Nm): all other joints")
    
    def compute(self, action_isaac, robot_state):
        """Compute PD control torques."""
        # Scale actions and add default offset
        np.multiply(action_isaac, self.cfg.action_scales, out=self.target_isaac)
        np.add(self.target_isaac, self.cfg.default_joint_pos, out=self.target_isaac)
        
        # PD control
        np.subtract(self.target_isaac, robot_state.joint_pos_isaac, out=self.error)
        np.multiply(self.error, self.cfg.joint_kp, out=self.tau_isaac)
        self.tau_isaac -= self.cfg.joint_kd * robot_state.joint_vel_isaac
        
        # Clip torques
        np.clip(self.tau_isaac, -self.tau_limit, self.tau_limit, out=self.tau_isaac)
        
        # Convert to MuJoCo order
        self.tau_mujoco[:] = self.tau_isaac[self.cfg.MUJOCO_TO_ISAAC]
        
        return self.tau_mujoco


def load_onnx_policy(onnx_path):
    """Load ONNX policy."""
    import onnxruntime as ort
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 1
    
    session = ort.InferenceSession(onnx_path, sess_options)
    input_names = [inp.name for inp in session.get_inputs()]
    needs_time_step = 'time_step' in input_names
    
    print(f"✓ Loaded ONNX policy from {onnx_path}")
    print(f"  Requires time_step: {needs_time_step}")
    
    if needs_time_step:
        def policy_fn(obs, time_step=0):
            obs_batch = obs.reshape(1, -1).astype(np.float32)
            time_step_batch = np.array([[time_step]], dtype=np.float32)
            output = session.run(None, {'obs': obs_batch, 'time_step': time_step_batch})
            return output[0][0]
    else:
        def policy_fn(obs, time_step=0):
            obs_batch = obs.reshape(1, -1).astype(np.float32)
            output = session.run(None, {'obs': obs_batch})
            return output[0][0]
    
    return policy_fn


def reset_mujoco_state(data, cfg):
    """Reset MuJoCo state to default."""
    default_pos_mujoco = cfg.default_joint_pos[cfg.MUJOCO_TO_ISAAC]
    data.qpos[7:31] = default_pos_mujoco
    data.qpos[2] = 0.82  # Base height
    data.qpos[3] = 1.0   # Quaternion w
    data.qvel[:] = 0.0   # Zero velocities


def run_sim2sim(policy_fn, motion, cfg, args):
    """Main simulation loop with NaN recovery.
    
    IMPORTANT NOTES:
    - MuJoCo XML has joint damping=0.6 and frictionloss=0.8
    - Isaac Lab sets these to 0 and handles damping via PD controller
    - This script uses PD controller damping only (matches Isaac Lab)
    - If robot behaves differently, the XML damping/friction may need adjustment
    """
    
    if not Path(args.mujoco_model).exists():
        raise FileNotFoundError(f"MuJoCo model not found: {args.mujoco_model}")
    
    print(f"✓ Loading MuJoCo model: {args.mujoco_model}")
    model = mujoco.MjModel.from_xml_path(args.mujoco_model)
    model.opt.timestep = cfg.dt
    data = mujoco.MjData(model)
    
    # CRITICAL: Check MuJoCo model settings
    print(f"\n⚠ MuJoCo Model Check:")
    print(f"  Timestep: {model.opt.timestep} (should be {cfg.dt})")
    if hasattr(model, 'dof_damping') and model.dof_damping is not None:
        print(f"  Joint damping detected: {model.dof_damping[0:6]} (first 6 DoFs)")
        print(f"  ⚠ Isaac Lab uses damping=0! MuJoCo XML has damping=0.6")
        print(f"  This may cause differences - consider setting damping=0 in XML")
    
    # Initialize to default pose
    reset_mujoco_state(data, cfg)
    mujoco.mj_forward(model, data)
    
    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(model, data)
    
    robot_state = RobotState()
    obs_builder = ObservationBuilder(cfg, include_base_lin_vel=args.include_base_lin_vel)
    pd_controller = PDController(cfg)
    last_action = np.zeros(24, dtype=np.float32)
    
    step_count = 0
    policy_step = 0
    motion_step = 0
    sim_time = 0.0
    nan_count = 0
    
    print("\n" + "="*60)
    print("Starting simulation with NaN protection")
    print("="*60)
    print(f"Policy rate: {1.0/cfg.dt/cfg.decimation:.1f} Hz")
    print(f"Observation dims: {obs_builder.obs_size}")
    print(f"Joint velocity scale: {cfg.vel_scale} (CRITICAL!)")
    print("="*60 + "\n")
    
    try:
        while sim_time < cfg.sim_duration:
            if step_count % cfg.decimation == 0:
                robot_state.update_from_mujoco(data, cfg)
                
                # Check for NaN in state
                if robot_state.has_nan:
                    nan_count += 1
                    print(f"\n⚠ NaN detected in robot state at step {policy_step}!")
                    print(f"  Resetting simulation...")
                    
                    # Reset everything
                    reset_mujoco_state(data, cfg)
                    mujoco.mj_forward(model, data)
                    last_action[:] = 0.0
                    sim_time = 0.0
                    step_count = 0
                    policy_step = 0
                    motion_step = 0
                    
                    if nan_count > 5:
                        print("⚠ Too many NaN resets, stopping simulation")
                        break
                    continue
                
                # Build observation
                obs = obs_builder.build(robot_state, last_action)
                
                # Check for NaN in observation
                if np.isnan(obs).any():
                    print(f"⚠ NaN in observation at step {policy_step}! Resetting last_action")
                    last_action[:] = 0.0
                    obs = obs_builder.build(robot_state, last_action)
                
                # Run policy
                action = policy_fn(obs, time_step=motion_step)
                
                # Check for NaN in action
                if np.isnan(action).any():
                    print(f"⚠ NaN in policy output at step {policy_step}! Using zeros")
                    action = np.zeros(24, dtype=np.float32)
                
                # Store action without clipping (matches training)
                last_action[:] = action
                
                # Compute torques
                tau = pd_controller.compute(action, robot_state)
                
                # Check for NaN in torques
                if np.isnan(tau).any():
                    print(f"⚠ NaN in torques at step {policy_step}! Using zeros")
                    tau = np.zeros(24, dtype=np.float32)
                
                data.ctrl[:24] = tau
                
                if policy_step % 100 == 0:
                    height = data.qpos[2]
                    max_tau = np.abs(tau).max()
                    vel_norm = np.linalg.norm(robot_state.base_lin_vel)
                    print(f"Step {policy_step:4d} | Time {sim_time:5.2f}s | "
                          f"Height {height:.3f}m | Vel {vel_norm:.2f}m/s | "
                          f"Max tau {max_tau:.1f}Nm")
                
                motion_step = (motion_step + 1) % motion.num_frames
                policy_step += 1
            
            mujoco.mj_step(model, data)
            step_count += 1
            sim_time = step_count * cfg.dt
            
            if viewer and step_count % 16 == 0:
                viewer.sync()
                if not viewer.is_running():
                    break
    
    except KeyboardInterrupt:
        print("\n⚠ Simulation stopped by user")
    
    if viewer:
        viewer.close()
    
    print(f"\n✓ Simulation finished: {step_count} steps, {sim_time:.2f}s")
    if nan_count > 0:
        print(f"⚠ NaN resets: {nan_count}")


def main():
    parser = argparse.ArgumentParser(description="Fixed Sim2Sim for PM01")
    parser.add_argument("--onnx_path", required=True, help="Path to ONNX policy")
    parser.add_argument("--motion_file", required=True, help="Path to motion .npz file")
    parser.add_argument("--config", required=True, help="Path to rl_basic_param.yaml")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration")
    parser.add_argument("--include_base_lin_vel", action="store_true", 
                       help="Include base_lin_vel in observation (78 dims)")
    parser.add_argument("--debug", action="store_true", help="Print detailed debug info")
    
    args = parser.parse_args()
    
    # Hardcoded MuJoCo model path
    args.mujoco_model = "/media/marmot/606de469-2f76-4155-82bc-e2e657636ad7/Ritwik/GMR/assets/engineai_pm01/pm_v2.xml"
    
    cfg = PM01Config.from_yaml(args.config)
    cfg.sim_duration = args.duration
    
    motion = MotionLoader(args.motion_file)
    policy_fn = load_onnx_policy(args.onnx_path)
    run_sim2sim(policy_fn, motion, cfg, args)


if __name__ == "__main__":
    main()