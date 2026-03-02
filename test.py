
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

env_name = "Isaac-PickPlace-GR1T2-Abs-v0"
env_cfg = parse_env_cfg(env_name, device="cuda", num_envs=1)

# Disable all recorders
env_cfg.recorders = {}

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

# Identify passive joints (unused by action space)
# passive_joints = [name for name in gr1_state_joints_config.keys() if name not in gr1_action_joints_config.keys()]
# print(f"Configuring passive actuator for {len(passive_joints)} joints: {passive_joints}")

# Configure passive actuator for unused joints
# Stiffness/damping high to hold pose
# passive_joint_regex = "(" + "|".join(passive_joints) + ")"
# env_cfg.scene.robot.actuators["passive"] = ImplicitActuatorCfg(
#     joint_names_expr=passive_joint_regex,
#     stiffness=400.0,
#     damping=40.0,
# )

# Manual override for hand stiffness/damping since base config has None
# This prevents the hands from exploding/flopping in simulation
env_cfg.scene.robot.actuators["right-hand"].stiffness = 200.0
env_cfg.scene.robot.actuators["right-hand"].damping = 5.0
env_cfg.scene.robot.actuators["left-hand"].stiffness = 200.0
env_cfg.scene.robot.actuators["left-hand"].damping = 5.0

# Set seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

succeess_term = env_cfg.terminations.success
# Disable terminations to avoid reset env
env_cfg.terminations = {}

# Create environment from loaded config
# Inspect robot config
# Inspect robot config
# print("Robot Config:", env_cfg.scene.robot)
# try:
#     print("\nRobot Actuators:")
#     if hasattr(env_cfg.scene.robot, "actuators"):
#         print("  Keys:", env_cfg.scene.robot.actuators.keys())
#         for key, actuator in env_cfg.scene.robot.actuators.items():
#             print(f"  Actuator '{key}':")
#             print(f"    Joint Names Expr: '{actuator.joint_names_expr}'")
#             # print(f"    Stiffness: {actuator.stiffness}")
#             # print(f"    Damping: {actuator.damping}")
#     else:
#         print("  No actuators defined in env_cfg.scene.robot")
# except Exception as e:
#     print(f"Error printing actuators: {e}")
    
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
df = pd.read_parquet("/home/minh/Workspace/projects/IsaacLabEvalTasks/dataset/generated_dataset_annotated_gr1/data/chunk-000/episode_000001.parquet")
# 
states = df["observation.state"] # states.shape: (126,)

with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():

            env.sim.reset()
            env.reset(seed=args.seed)

            # Teleport robot to start of episode
            robot = env.scene["robot"]
            robot_state = robot.data.joint_pos
            breakpoint()
            model_state = np.zeros((26,))
            model_state[0] = robot_state[0, 12]
            model_state[1] = robot_state[0, 17]
            model_state[2] = robot_state[0, 22]
            model_state[3] = robot_state[0, 24]
            model_state[4] = robot_state[0, 26]
            model_state[5] = robot_state[0, 28]
            model_state[6] = robot_state[0, 30]

            model_state[7] = robot_state[0, 13]
            model_state[8] = robot_state[0, 18]
            model_state[9] = robot_state[0, 23]
            model_state[10] = robot_state[0, 25]
            model_state[11] = robot_state[0, 27]
            model_state[12] = robot_state[0, 29]
            model_state[13] = robot_state[0, 31]
            
            model_state[14] = robot_state[0, 34]
            model_state[15] = robot_state[0, 35]
            model_state[16] = robot_state[0, 33]
            model_state[17] = robot_state[0, 32]
            model_state[18] = robot_state[0, 36]
            model_state[19] = robot_state[0, 46]


            model_state[20] = robot_state[0, 39]
            model_state[21] = robot_state[0, 40]
            model_state[22] = robot_state[0, 38]
            model_state[23] = robot_state[0, 37]
            model_state[24] = robot_state[0, 41]
            model_state[25] = robot_state[0, 51]

            policy_joints = {
                    "action.left_arm": model_state[None, None, 0:7].astype(np.float32),
                    "action.right_arm": model_state[None, None, 7:14].astype(np.float32),
                    "action.left_hand": model_state[None, None, 14:20].astype(np.float32),
                    "action.right_hand": model_state[None, None, 20:26].astype(np.float32),
                }
            robot_action_sim = remap_policy_joints_to_sim_joints( # policy (26, ) -> sim (36,)
                policy_joints, gr00t_joints_config, gr1_action_joints_config, args.simulation_device
            ) # torch.Size([1, 1, 36])
            # config cua model: gr00t_joints_config 
            # config cua action trong sim: gr1_action_joints_config - 36
            rollout_action = robot_action_sim.get_joints_pos() # torch.Size([1, 1, 36])
            # num_robot, time_step, joints
            for i in range(1000):
                env.step(rollout_action[:, 0, :])

            # print(f"Replaying {len(states)} states...")
            # import time
            
            # for state in states: 
            #     # Unflatten state into dictionary expected by remap_policy_joints_to_sim_joints
            #     # state is likely shape (26,)
            #     if not isinstance(state, np.ndarray):
            #         state = np.array(state)
                    
            #     policy_joints = {
            #         "action.left_arm": state[None, None, 0:7].astype(np.float32),
            #         "action.right_arm": state[None, None, 7:14].astype(np.float32),
            #         "action.left_hand": state[None, None, 14:20].astype(np.float32),
            #         "action.right_hand": state[None, None, 20:26].astype(np.float32),
            #     }
                
            #     # Check shapes
            #     # print(f"Left arm shape: {policy_joints['action.left_arm'].shape}")
                
            #     robot_action_sim = remap_policy_joints_to_sim_joints(
            #         policy_joints, gr00t_joints_config, gr1_action_joints_config, args.simulation_device
            #     )
            #     # robot_action_sim.joints_pos is (1, 1, 36)
            #     rollout_action = robot_action_sim.get_joints_pos()
                
            #     # Check for NaNs or Infs
            #     if torch.isnan(rollout_action).any() or torch.isinf(rollout_action).any():
            #         print("WARNING: rollout_action contains NaNs or Infs!")
                
            #     # Apply action
            #     # env.step expects (num_envs, num_actions) -> (1, 36)
            #     # rollout_action is (1, 1, 36), we need (1, 36)
            #     action_tensor = rollout_action[:, 0, :]
            #     env.step(action_tensor)
                
                # Optional: sleep to visualize
                # time.sleep(0.02)
            # Reset environment
            # print("Resetting environment")
            # robot = env.scene["robot"]
            # ego_camera = env.scene["robot_pov_cam"]
            # img = Image.fromarray(ego_camera.data.output["rgb"][0].cpu().numpy())
            # img.save("test.png")
            # print(.shape)
            # robot_state_sim = robot.data.joint_pos
env.close()
simulation_app.close()


