
from isaacsim import SimulationApp  # noqa: F401 # isort: skip


import contextlib
import os
import tqdm
from typing import Optional
import inspect
import pinocchio as pin
import sys
import os

# Add scripts directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

from isaaclab.app import AppLauncher 
# Launch the simulator
app_launcher = AppLauncher(
    headless=False, enable_cameras=True, num_envs=1, device="cuda"
)
simulation_app = app_launcher.app

import gymnasium as gym

# Import tasks to register environments with Gymnasium
import isaaclab_eval_tasks.tasks  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import torch
import numpy as np
import random
import argparse as args
from scripts.policies.joints_conversion import remap_policy_joints_to_sim_joints, remap_sim_joints_to_policy_joints
from scripts.io_utils import load_gr1_joints_config
from scripts.config.args import Gr00tN1ClosedLoopArguments

from PIL import Image 

from scripts.robot_joints import JointsAbsPosition


from isaaclab.actuators import ImplicitActuatorCfg




# Load arguments to get paths
args = Gr00tN1ClosedLoopArguments()
gr1_state_joints_config = load_gr1_joints_config(args.state_joints_config_path)
gr1_action_joints_config = load_gr1_joints_config(args.action_joints_config_path)
"""
gr1_action_joints_config: 36
{'left_shoulder_pitch_joint': 0, 'right_shoulder_pitch_joint': 1, 'left_shoulder_roll_joint': 2, 'right_shoulder_roll_joint': 3, 'left_shoulder_yaw_joint': 4, 'right_shoulder_yaw_joint': 5, 'left_elbow_pitch_joint': 6, 'right_elbow_pitch_joint': 7, 'left_wrist_yaw_joint': 8, 'right_wrist_yaw_joint': 9, 'left_wrist_roll_joint': 10, 'right_wrist_roll_joint': 11, 'left_wrist_pitch_joint': 12, 'right_wrist_pitch_joint': 13, 'L_index_proximal_joint': 14, 'L_middle_proximal_joint': 15, 'L_pinky_proximal_joint': 16, 'L_ring_proximal_joint': 17, 'L_thumb_proximal_yaw_joint': 18, 'R_index_proximal_joint': 19, 'R_middle_proximal_joint': 20, 'R_pinky_proximal_joint': 21, 'R_ring_proximal_joint': 22, 'R_thumb_proximal_yaw_joint': 23, 'L_index_intermediate_joint': 24, 'L_middle_intermediate_joint': 25, 'L_pinky_intermediate_joint': 26, 'L_ring_intermediate_joint': 27, 'L_thumb_proximal_pitch_joint': 28, 'R_index_intermediate_joint': 29, 'R_middle_intermediate_joint': 30, 'R_pinky_intermediate_joint': 31, 'R_ring_intermediate_joint': 32, 'R_thumb_proximal_pitch_joint': 33, 'L_thumb_distal_joint': 34, 'R_thumb_distal_joint': 35}

gr1_state_joints_config: 54
{'left_hip_roll_joint': 0, 'right_hip_roll_joint': 1, 'waist_yaw_joint': 2, 'left_hip_yaw_joint': 3, 'right_hip_yaw_joint': 4, 'waist_pitch_joint': 5, 'left_hip_pitch_joint': 6, 'right_hip_pitch_joint': 7, 'waist_roll_joint': 8, 'left_knee_pitch_joint': 9, 'right_knee_pitch_joint': 10, 'head_roll_joint': 11, 'left_shoulder_pitch_joint': 12, 'right_shoulder_pitch_joint': 13, 'left_ankle_pitch_joint': 14, 'right_ankle_pitch_joint': 15, 'head_pitch_joint': 16, 'left_shoulder_roll_joint': 17, 'right_shoulder_roll_joint': 18, 'left_ankle_roll_joint': 19, 'right_ankle_roll_joint': 20, 'head_yaw_joint': 21, 'left_shoulder_yaw_joint': 22, 'right_shoulder_yaw_joint': 23, 'left_elbow_pitch_joint': 24, 'right_elbow_pitch_joint': 25, 'left_wrist_yaw_joint': 26, 'right_wrist_yaw_joint': 27, 'left_wrist_roll_joint': 28, 'right_wrist_roll_joint': 29, 'left_wrist_pitch_joint': 30, 'right_wrist_pitch_joint': 31, 'L_index_proximal_joint': 32, 'L_middle_proximal_joint': 33, 'L_pinky_proximal_joint': 34, 'L_ring_proximal_joint': 35, 'L_thumb_proximal_yaw_joint': 36, 'R_index_proximal_joint': 37, 'R_middle_proximal_joint': 38, 'R_pinky_proximal_joint': 39, 'R_ring_proximal_joint': 40, 'R_thumb_proximal_yaw_joint': 41, 'L_index_intermediate_joint': 42, 'L_middle_intermediate_joint': 43, 'L_pinky_intermediate_joint': 44, 'L_ring_intermediate_joint': 45, 'L_thumb_proximal_pitch_joint': 46, 'R_index_intermediate_joint': 47, 'R_middle_intermediate_joint': 48, 'R_pinky_intermediate_joint': 49, 'R_ring_intermediate_joint': 50, 'R_thumb_proximal_pitch_joint': 51, 'L_thumb_distal_joint': 52, 'R_thumb_distal_joint': 53}
"""

# Set seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

env_name = "Isaac-NutPour-GR1T2-ClosedLoop-v0"
env_cfg = parse_env_cfg(env_name, device="cuda", num_envs=1)
# Disable all recorders
env_cfg.recorders = {}

# Disable terminations to avoid reset env
env_cfg.terminations = {}

env = gym.make(env_name, cfg=env_cfg).unwrapped
# Set seed
env.seed(42)
env.sim.reset()
env.reset(seed=42)

gr00t_joints_config = load_gr1_joints_config(args.gr00t_joints_config_path)
"""
{'left_arm': ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_pitch_joint', 'left_wrist_yaw_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint'], 
 'right_arm': ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_pitch_joint', 'right_wrist_yaw_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint'], 
 'left_hand': ['L_pinky_proximal_joint', 'L_ring_proximal_joint', 'L_middle_proximal_joint', 'L_index_proximal_joint', 'L_thumb_proximal_yaw_joint', 'L_thumb_proximal_pitch_joint'], 
 'right_hand': ['R_pinky_proximal_joint', 'R_ring_proximal_joint', 'R_middle_proximal_joint', 'R_index_proximal_joint', 'R_thumb_proximal_yaw_joint', 'R_thumb_proximal_pitch_joint']}
"""

# replay episode 0
import pandas as pd 
df = pd.read_parquet("/home/minh/Workspace/projects/IsaacLabEvalTasks/dataset/Cut-Pouring-task/Nut-Pouring-task/data/chunk-000/episode_000000.parquet")
# 
states = df["observation.state"] # states.shape: (126,)

with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():

            env.sim.reset()
            env.reset(seed=args.seed)

            print(f"Replaying {len(states)} states...")
            import time
            
            for state in states: 
                # Unflatten state into dictionary expected by remap_policy_joints_to_sim_joints
                # state is likely shape (26,)
                if not isinstance(state, np.ndarray):
                    state = np.array(state)
                    
                policy_joints = {
                    "action.left_arm": state[None, None, 0:7].astype(np.float32),
                    "action.right_arm": state[None, None, 7:14].astype(np.float32),
                    "action.left_hand": state[None, None, 14:20].astype(np.float32),
                    "action.right_hand": state[None, None, 20:26].astype(np.float32),
                }
                
                # Check shapes
                robot_action_sim = remap_policy_joints_to_sim_joints(
                    policy_joints, gr00t_joints_config, gr1_action_joints_config, args.simulation_device
                )

                """
1. The Problem: Format Mismatch
The policy (represented here by the 26-dimensional data extracted from the parquet file) and the simulated robot (the GR1) "speak" different languages regarding joint order and count:

Policy/Dataset: Provides a 26-element vector divided into 4 groups (Left Arm: 7, Right Arm: 7, Left Hand: 6, Right Hand: 6).
Simulator (Isaac Lab): Expects a 36-element tensor for the GR1 robot, where each index must strictly correspond to a specific physical joint (e.g., index 0 might be left_shoulder_pitch_joint).
2. The Solution: 
remap_policy_joints_to_sim_joints
This function acts as a "translator." It takes the raw joint values and uses two configuration maps to move them into the correct slots:

policy_joints
: A dictionary containing the actual joint position values (sliced from the state vector at lines 111-114).
gr00t_joints_config: A reference map that tells the function which joint names correspond to the positions in the policy’s 26-dim output.
gr1_action_joints_config: A reference map that defines the target indices (0-35) for each joint name in the simulator.
args.simulation_device: Ensures the resulting tensor is created on the correct device (e.g., cuda), so it can be passed directly to the simulation engine.
3. What the Result (robot_action_sim) contains
The output is a JointsAbsPosition object. Inside it is a tensor (likely of shape [1, 1, 36]) where:

The 26 joints from your dataset have been moved to their exact required indices in the 36-slot array.
Any joints the simulator expects but the policy doesn't provide (like leg joints or specific torso joints) are typically initialized to zero or a default neutral position.
                """
                # robot_action_sim.joints_pos is (1, 1, 36)
                rollout_action = robot_action_sim.get_joints_pos()
                
                # Check for NaNs or Infs
                if torch.isnan(rollout_action).any() or torch.isinf(rollout_action).any():
                    print("WARNING: rollout_action contains NaNs or Infs!")
                
                # Apply action
                # env.step expects (num_envs, num_actions) -> (1, 36)
                # rollout_action is (1, 1, 36), we need (1, 36)
                action_tensor = rollout_action[:, 0, :]
                env.step(action_tensor)
                
                # Optional: sleep to visualize
                time.sleep(0.02)
    
env.close()
simulation_app.close()
