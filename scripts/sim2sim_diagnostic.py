"""Diagnostic script to compare Isaac Lab vs MuJoCo sim2sim behavior.

This script helps identify discrepancies between the two simulators.
"""

import argparse
import numpy as np
import mujoco
import yaml
from pathlib import Path

def compare_configurations(yaml_path, mujoco_model_path):
    """Compare configuration parameters."""
    
    print("="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)
    
    # Load YAML config
    with open(yaml_path, 'r') as f:
        yaml_params = yaml.safe_load(f)
    
    print("\n1. SIMULATION PARAMETERS:")
    print(f"   control_dt (from YAML): {yaml_params.get('control_dt', 'NOT FOUND')}")
    print(f"   Expected policy rate: 50 Hz (0.02s per step)")
    print(f"   MuJoCo sim2sim uses: dt=0.002, decimation=10 -> 0.02s ✓")
    
    print("\n2. PD GAINS COMPARISON:")
    joint_kp = yaml_params.get('joint_kp', [])
    joint_kd = yaml_params.get('joint_kd', [])
    
    isaac_joint_order = [
        'j00_hip_pitch_l', 'j06_hip_pitch_r', 'j12_waist_yaw', 'j01_hip_roll_l',
        'j07_hip_roll_r', 'j13_shoulder_pitch_l', 'j18_shoulder_pitch_r', 'j23_head_yaw',
    ]
    
    print(f"   YAML Kp (first 8): {joint_kp[:8]}")
    print(f"   YAML Kd (first 8): {joint_kd[:8]}")
    
    # Expected from pm01.py
    expected_kp = [70, 70, 50, 50, 50, 50, 50, 50]  # hip_pitch, hip_pitch, waist, hip_roll, hip_roll, shoulder...
    expected_kd = [7.0, 7.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    
    kp_match = np.allclose(joint_kp[:8], expected_kp)
    kd_match = np.allclose(joint_kd[:8], expected_kd)
    
    print(f"   Matches pm01.py? Kp: {kp_match}, Kd: {kd_match}")
    
    print("\n3. ACTION SCALES:")
    action_scales = yaml_params.get('action_scale', [])
    print(f"   First 6 joints:")
    for i in range(6):
        print(f"     [{i}] {isaac_joint_order[i]:20s}: {action_scales[i]:.6f}")
    
    # Calculate what action scale should be: 0.25 * effort / stiffness
    # From pm01.py:
    # hip_pitch: effort=164, stiffness=70 -> 0.25*164/70 = 0.5857
    # hip_roll: effort=164, stiffness=50 -> 0.25*164/50 = 0.82
    expected_scales = [0.5857, 0.5857, 0.305, 0.82, 0.82, 0.305]
    
    for i in range(6):
        diff = abs(action_scales[i] - expected_scales[i])
        status = "✓" if diff < 0.001 else f"✗ (expected {expected_scales[i]:.4f})"
        print(f"       {status}")
    
    print("\n4. OBSERVATION PARAMETERS:")
    print(f"   observation_clip: {yaml_params.get('observation_clip', 'NOT FOUND')}")
    print(f"   action_clip: {yaml_params.get('action_clip', 'NOT FOUND')} (should NOT be used!)")
    
    print("\n5. DEFAULT JOINT POSITIONS:")
    default_pos = yaml_params.get('default_joint_q', [])
    print(f"   First 6: {default_pos[:6]}")
    
    print("\n" + "="*80)
    print("POTENTIAL ISSUES TO CHECK:")
    print("="*80)
    
    issues = []
    
    # Check 1: Torque limits
    print("\n1. TORQUE LIMITS:")
    print("   In sim2sim: tau_limit_low=41.0, tau_limit_high=164.0")
    print("   From pm01.py: effort_limit should be 61.0 (low) and 164.0 (high)")
    print("   ⚠️  MISMATCH: Low torque should be 61.0, not 41.0!")
    issues.append("Torque limits: sim2sim uses 41.0 but pm01.py uses 61.0 for low-torque joints")
    
    # Check 2: Joint velocity observation
    print("\n2. JOINT VELOCITY OBSERVATION:")
    print("   Isaac Lab uses: joint_vel_rel with scale=0.05")
    print("   From tracking_env_cfg.py: ObsTerm(func=mdp.joint_vel_rel, ...)")
    print("   ⚠️  Check if sim2sim applies velocity scaling!")
    issues.append("Joint velocity: Check if 0.05 scaling is applied")
    
    # Check 3: Gravity compensation
    print("\n3. GRAVITY COMPENSATION:")
    print("   Isaac Lab: ImplicitActuator handles gravity automatically")
    print("   MuJoCo: No built-in gravity compensation in PD control")
    print("   ⚠️  This could cause significant differences!")
    issues.append("Gravity: MuJoCo doesn't compensate, Isaac Lab does")
    
    # Check 4: Damping implementation
    print("\n4. DAMPING IMPLEMENTATION:")
    print("   Isaac Lab: Built into ImplicitActuator")
    print("   MuJoCo sim2sim: tau = kp*(target-pos) - kd*vel")
    print("   Should be equivalent if kd values match")
    
    # Check 5: Action application timing
    print("\n5. ACTION APPLICATION TIMING:")
    print("   Isaac Lab: decimation=4, sim.dt=0.005 -> policy at 50Hz")
    print("   MuJoCo: decimation=10, dt=0.002 -> policy at 50Hz")
    print("   Both run policy at same rate ✓")
    
    # Check 6: Observation noise
    print("\n6. OBSERVATION NOISE:")
    print("   Isaac Lab training: Adds noise to observations")
    print("   Isaac Lab play.py: enable_corruption=True in PolicyCfg")
    print("   MuJoCo sim2sim: No noise added")
    print("   This should be OK for deployment")
    
    print("\n" + "="*80)
    print("SUMMARY OF ISSUES:")
    print("="*80)
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    
    return issues


def check_joint_ordering():
    """Verify joint ordering is correct."""
    
    print("\n" + "="*80)
    print("JOINT ORDERING VERIFICATION")
    print("="*80)
    
    mujoco_order = [
        'j00_hip_pitch_l', 'j01_hip_roll_l', 'j02_hip_yaw_l', 'j03_knee_pitch_l',
        'j04_ankle_pitch_l', 'j05_ankle_roll_l', 'j06_hip_pitch_r', 'j07_hip_roll_r',
        'j08_hip_yaw_r', 'j09_knee_pitch_r', 'j10_ankle_pitch_r', 'j11_ankle_roll_r',
        'j12_waist_yaw', 'j13_shoulder_pitch_l', 'j14_shoulder_roll_l', 'j15_shoulder_yaw_l',
        'j16_elbow_pitch_l', 'j17_elbow_yaw_l', 'j18_shoulder_pitch_r', 'j19_shoulder_roll_r',
        'j20_shoulder_yaw_r', 'j21_elbow_pitch_r', 'j22_elbow_yaw_r', 'j23_head_yaw'
    ]
    
    isaac_order = [
        'j00_hip_pitch_l', 'j06_hip_pitch_r', 'j12_waist_yaw', 'j01_hip_roll_l',
        'j07_hip_roll_r', 'j13_shoulder_pitch_l', 'j18_shoulder_pitch_r', 'j23_head_yaw',
        'j02_hip_yaw_l', 'j08_hip_yaw_r', 'j14_shoulder_roll_l', 'j19_shoulder_roll_r',
        'j03_knee_pitch_l', 'j09_knee_pitch_r', 'j15_shoulder_yaw_l', 'j20_shoulder_yaw_r',
        'j04_ankle_pitch_l', 'j10_ankle_pitch_r', 'j16_elbow_pitch_l', 'j21_elbow_pitch_r',
        'j05_ankle_roll_l', 'j11_ankle_roll_r', 'j17_elbow_yaw_l', 'j22_elbow_yaw_r'
    ]
    
    print("\nMapping from MuJoCo to Isaac Lab:")
    print("(MuJoCo idx -> Isaac idx)")
    
    mujoco_to_isaac = []
    for i, mj_name in enumerate(mujoco_order):
        isaac_idx = isaac_order.index(mj_name)
        mujoco_to_isaac.append(isaac_idx)
        if i < 8:
            print(f"  MuJoCo[{i:2d}] {mj_name:20s} -> Isaac[{isaac_idx:2d}]")
    
    print("\nExpected MUJOCO_TO_ISAAC array:")
    print(f"  {mujoco_to_isaac}")
    
    expected = [0, 3, 8, 12, 16, 20, 1, 4, 9, 13, 17, 21, 2, 5, 10, 14, 18, 22, 6, 11, 15, 19, 23, 7]
    matches = mujoco_to_isaac == expected
    print(f"\nMatches sim2sim script: {matches}")


def suggest_fixes():
    """Suggest potential fixes."""
    
    print("\n" + "="*80)
    print("SUGGESTED FIXES FOR SIM2SIM")
    print("="*80)
    
    print("""
1. FIX TORQUE LIMITS:
   Change in PDController:
   ```python
   self.tau_limit = np.full(24, 61.0, dtype=np.float32)  # Was 41.0
   high_torque_indices = [0, 1, 12, 13]
   self.tau_limit[high_torque_indices] = 164.0
   ```

2. ADD JOINT VELOCITY SCALING:
   In ObservationBuilder.build():
   ```python
   self.obs[idx:idx+24] = robot_state.joint_vel_isaac * 0.05  # Scale by 0.05
   ```

3. CHECK GRAVITY COMPENSATION:
   MuJoCo's implicit integration may handle this differently.
   Try adding gravity compensation term:
   ```python
   # In PDController.compute(), before clipping:
   # tau += gravity_compensation(robot_state)
   ```

4. VERIFY INITIAL STATE:
   Ensure MuJoCo starts at exactly same state as Isaac Lab:
   - Base height: 0.82m
   - Joint positions: from default_joint_q
   - Zero velocities
   
5. CHECK MUJOCO MODEL:
   Verify the MuJoCo XML has:
   - Correct inertias
   - Correct friction coefficients (static=1.0, dynamic=1.0)
   - Correct damping (should be 0 for joints, handled by PD)
   - Correct armature (should be 0 or very small)

6. ADD OBSERVATION NORMALIZATION:
   If the policy was trained with observation normalization,
   you need to apply the same normalization in sim2sim.
   Check if there's a normalizer saved with the policy.
""")


def main():
    parser = argparse.ArgumentParser(description="Diagnose sim2sim issues")
    parser.add_argument("--config", required=True, help="Path to rl_basic_param.yaml")
    parser.add_argument("--mujoco_model", default=None, help="Path to MuJoCo XML")
    
    args = parser.parse_args()
    
    # Run diagnostics
    issues = compare_configurations(args.config, args.mujoco_model)
    check_joint_ordering()
    suggest_fixes()
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("""
1. Apply the torque limit fix (41.0 -> 61.0)
2. Add joint velocity scaling (multiply by 0.05)
3. Check if policy uses observation normalization
4. Compare MuJoCo model parameters with Isaac Lab URDF
5. Test with these fixes and report back!
""")


if __name__ == "__main__":
    main()