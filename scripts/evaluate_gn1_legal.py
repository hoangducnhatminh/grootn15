# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is a robot policy evaluation script that tests a trained AI model (GR00T N1) by running it in a simulated environment (Isaac Sim). Think of it as a testing program that:

1. Loads a pre-trained AI model (the "policy").
2. Loads a simulated robot (GR00T N1) and a simulated environment (e.g., a task where the robot needs to pick up an object).
3. Connects the AI model to the robot in the simulation.
4. Runs the robot through a series of tasks for a set number of steps (the "rollout").
5. Records videos and collects data on how well the robot performs the tasks.
6. Calculates statistics like success rate and average reward.
7. Prints a summary of the results.

In short, it's a script to see how well the "brain" (AI model) controls the "body" (robot) in a virtual world.

"""

from isaacsim import SimulationApp  # noqa: F401 # isort: skip
from isaaclab.app import AppLauncher  # noqa: F401 # isort: skip

import contextlib
import os
import torch
import tqdm
from typing import Optional

import tyro
from io_utils import VideoWriter

from config.args import Gr00tN1ClosedLoopArguments

args = tyro.cli(Gr00tN1ClosedLoopArguments)

if args.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
    import pinocchio  # noqa: F401

# Launch the simulator
app_launcher = AppLauncher(
    headless=args.headless, enable_cameras=True, num_envs=args.num_envs, device=args.simulation_device
)
simulation_app = app_launcher.app

import gymnasium as gym

from closed_loop_policy import create_sim_environment
from evaluators.gr00t_n1_evaluator import Gr00tN1Evaluator
from policies.gr00t_n1_policy import Gr00tN1Policy
from robot_joints import JointsAbsPosition

from isaaclab.envs import ManagerBasedRLEnvCfg

import isaaclab_eval_tasks.tasks  # noqa: F401


def run_closed_loop_policy(
    args: Gr00tN1ClosedLoopArguments,
    simulation_app: SimulationApp,
    env_cfg: ManagerBasedRLEnvCfg,
    policy: Gr00tN1Policy,
    evaluator: Optional[Gr00tN1Evaluator] = None,
):
    # Extract success checking function
    succeess_term = env_cfg.terminations.success
    # Disable terminations to avoid reset env
    env_cfg.terminations = {}

    # Create environment from loaded config
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    # Set seed
    env.seed(args.seed)

    # Create camera video recorder
    video_writer = None
    # Only record the first environment if multiple envs are running in one episode
    if args.record_camera:
        video_count = 0
        video_fpath = os.path.join(
            args.record_video_output_path, f"{args.task_name}_{args.checkpoint_name}_{video_count}.mp4"
        )
        video_writer = VideoWriter(
            video_fpath,
            # Height, width to be in the order of (width, height) for cv2
            args.original_image_size[:2][::-1],
            fps=20,
        )

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():

            # Terminate the simulation_app if having enough rollouts counted by the evaluator
            # Otherwise, continue the rollout endlessly
            if evaluator is not None and evaluator.num_rollouts >= args.max_num_rollouts:
                break

            # Reset environment
            env.sim.reset()
            env.reset(seed=args.seed)

            robot = env.scene["robot"]
            print(robot.data.joint_pos) # shape: torch.Size([10, 54]) -> TODO: WHY? which means 10 parallel robots each with 54 joints.
            print(policy.gr1_state_joints_config) # {'left_hip_roll_joint': 0, 'right_hip_roll_joint': 1, 'waist_yaw_joint': 2, 'left_hip_yaw_joint': 3, 'right_hip_yaw_joint': 4, 'waist_pitch_joint': 5, 'left_hip_pitch_joint': 6, 'right_hip_pitch_joint': 7, 'waist_roll_joint': 8, 'left_knee_pitch_joint': 9, 'right_knee_pitch_joint': 10, 'head_roll_joint': 11, 'left_shoulder_pitch_joint': 12, 'right_shoulder_pitch_joint': 13, 'left_ankle_pitch_joint': 14, 'right_ankle_pitch_joint': 15, 'head_pitch_joint': 16, 'left_shoulder_roll_joint': 17, 'right_shoulder_roll_joint': 18, 'left_ankle_roll_joint': 19, 'right_ankle_roll_joint': 20, 'head_yaw_joint': 21, 'left_shoulder_yaw_joint': 22, 'right_shoulder_yaw_joint': 23, 'left_elbow_pitch_joint': 24, 'right_elbow_pitch_joint': 25, 'left_wrist_yaw_joint': 26, 'right_wrist_yaw_joint': 27, 'left_wrist_roll_joint': 28, 'right_wrist_roll_joint': 29, 'left_wrist_pitch_joint': 30, 'right_wrist_pitch_joint': 31, 'L_index_proximal_joint': 32, 'L_middle_proximal_joint': 33, 'L_pinky_proximal_joint': 34, 'L_ring_proximal_joint': 35, 'L_thumb_proximal_yaw_joint': 36, 'R_index_proximal_joint': 37, 'R_middle_proximal_joint': 38, 'R_pinky_proximal_joint': 39, 'R_ring_proximal_joint': 40, 'R_thumb_proximal_yaw_joint': 41, 'L_index_intermediate_joint': 42, 'L_middle_intermediate_joint': 43, 'L_pinky_intermediate_joint': 44, 'L_ring_intermediate_joint': 45, 'L_thumb_proximal_pitch_joint': 46, 'R_index_intermediate_joint': 47, 'R_middle_intermediate_joint': 48, 'R_pinky_intermediate_joint': 49, 'R_ring_intermediate_joint': 50, 'R_thumb_proximal_pitch_joint': 51, 'L_thumb_distal_joint': 52, 'R_thumb_distal_joint': 53}
            print(args.simulation_device)


            robot_state_sim = JointsAbsPosition(
                robot.data.joint_pos, policy.gr1_state_joints_config, args.simulation_device
            )
            """
            robot_state_sim:
            JointsAbsPosition(joints_pos=tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                                0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                                0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                                0.0000,  0.0000,  0.0000,  0.0000, -1.5700,  0.0000,  0.0000,  0.0000,
                                                0.0000, -1.5700,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,]]), 
                            joints_order_config={'left_hip_roll_joint': 0, 'right_hip_roll_joint': 1, 'waist_yaw_joint': 2, 'left_hip_yaw_joint': 3, 'right_hip_yaw_joint': 4, 'waist_pitch_joint': 5, 'left_hip_pitch_joint': 6, 'right_hip_pitch_joint': 7, 'waist_roll_joint': 8, 'left_knee_pitch_joint': 9, 'right_knee_pitch_joint': 10, 'head_roll_joint': 11, 'left_shoulder_pitch_joint': 12, 'right_shoulder_pitch_joint': 13, 'left_ankle_pitch_joint': 14, 'right_ankle_pitch_joint': 15, 'head_pitch_joint': 16, 'left_shoulder_roll_joint': 17, 'right_shoulder_roll_joint': 18, 'left_ankle_roll_joint': 19, 'right_ankle_roll_joint': 20, 'head_yaw_joint': 21, 'left_shoulder_yaw_joint': 22, 'right_shoulder_yaw_joint': 23, 'left_elbow_pitch_joint': 24, 'right_elbow_pitch_joint': 25, 'left_wrist_yaw_joint': 26, 'right_wrist_yaw_joint': 27, 'left_wrist_roll_joint': 28, 'right_wrist_roll_joint': 29, 'left_wrist_pitch_joint': 30, 'right_wrist_pitch_joint': 31, 'L_index_proximal_joint': 32, 'L_middle_proximal_joint': 33, 'L_pinky_proximal_joint': 34, 'L_ring_proximal_joint': 35, 'L_thumb_proximal_yaw_joint': 36, 'R_index_proximal_joint': 37, 'R_middle_proximal_joint': 38, 'R_pinky_proximal_joint': 39, 'R_ring_proximal_joint': 40, 'R_thumb_proximal_yaw_joint': 41, 'L_index_intermediate_joint': 42, 'L_middle_intermediate_joint': 43, 'L_pinky_intermediate_joint': 44, 'L_ring_intermediate_joint': 45, 'L_thumb_proximal_pitch_joint': 46, 'R_index_intermediate_joint': 47, 'R_middle_intermediate_joint': 48, 'R_pinky_intermediate_joint': 49, 'R_ring_intermediate_joint': 50, 'R_thumb_proximal_pitch_joint': 51, 'L_thumb_distal_joint': 52, 'R_thumb_distal_joint': 53}, device='cpu')
            """
            
            ego_camera = env.scene["robot_pov_cam"]

            if args.record_camera and video_writer is not None and os.path.exists(video_fpath):
                # Replace the last part of the video file name with the video count
                video_fpath = "_".join(video_fpath.split("_")[:-1]) + f"_{video_count}.mp4"
                video_writer.change_file_path(video_fpath)

            for _ in tqdm.tqdm(range(args.rollout_length)):
                """
                Purpose: This defines how many times the policy will be queried during a single trial (rollout).
                Mechanism: If rollout_length is 100, the robot will make 100 high-level decisions. Because the policy predicts a sequence of actions (a horizon), the total number of simulation steps will actually be rollout_length * num_feedback_actions
                """
                robot_state_sim.set_joints_pos(robot.data.joint_pos)# robot.data.joint_pos is the raw data from the physics engine. t takes the 54 raw simulation joints and reorders or filters them to match the specific format the policy expects (as defined in gr1_state_joints_config).

                robot_action_sim = policy.get_new_goal(robot_state_sim, ego_camera, args.language_instruction) # It returns a tray of predicted future joint positions. VLA models usually predict a "chunk" of future steps (an action horizon) rather than just one single step to ensure smoother motion.
                """
                robot_action_sim
                    JointsAbsPosition(joints_pos=tensor([[[-6.9144e-03,  6.4826e-04, -4.3572e-03,  ...,  6.7974e-01,
                                                      0.0000e+00,  0.0000e+00]]]), 
                            joints_order_config={'left_shoulder_pitch_joint': 0, 'right_shoulder_pitch_joint': 1, 'left_shoulder_roll_joint': 2, 'right_shoulder_roll_joint': 3, 'left_shoulder_yaw_joint': 4, 'right_shoulder_yaw_joint': 5, 'left_elbow_pitch_joint': 6, 'right_elbow_pitch_joint': 7, 'left_wrist_yaw_joint': 8, 'right_wrist_yaw_joint': 9, 'left_wrist_roll_joint': 10, 'right_wrist_roll_joint': 11, 'left_wrist_pitch_joint': 12, 'right_wrist_pitch_joint': 13, 'L_index_proximal_joint': 14, 'L_middle_proximal_joint': 15, 'L_pinky_proximal_joint': 16, 'L_ring_proximal_joint': 17, 'L_thumb_proximal_yaw_joint': 18, 'R_index_proximal_joint': 19, 'R_middle_proximal_joint': 20, 'R_pinky_proximal_joint': 21, 'R_ring_proximal_joint': 22, 'R_thumb_proximal_yaw_joint': 23, 'L_index_intermediate_joint': 24, 'L_middle_intermediate_joint': 25, 'L_pinky_intermediate_joint': 26, 'L_ring_intermediate_joint': 27, 'L_thumb_proximal_pitch_joint': 28, 'R_index_intermediate_joint': 29, 'R_middle_intermediate_joint': 30, 'R_pinky_intermediate_joint': 31, 'R_ring_intermediate_joint': 32, 'R_thumb_proximal_pitch_joint': 33, 'L_thumb_distal_joint': 34, 'R_thumb_distal_joint': 35}, device='cpu')
                """

                rollout_action = robot_action_sim.get_joints_pos(args.simulation_device) # The robot_action_sim object contains the model's prediction. This line converts that prediction into a raw tensor of joint positions that Isaac Lab's environment can understand.
                # rollout_action.shape: torch.Size([10, 16, 36]) - [num_envs, horizon_length, num_joints] which means 10 robots, each given a sequence of 16 future positions for their 36 joints.
                # Number of joints from policy shall match the env action reqs
                assert rollout_action.shape[-1] == env.action_space.shape[1]

                # take only the first num_feedback_actions, the rest are ignored, preventing over memorization
                for i in range(args.num_feedback_actions):
                    assert rollout_action[:, i, :].shape[0] == args.num_envs
                    env.step(rollout_action[:, i, :])

                    if args.record_camera and video_writer is not None:
                        # Only record the first environment if multiple envs are running
                        video_writer.add_image(ego_camera.data.output["rgb"][0])

            if args.record_camera and video_writer is not None:
                video_count += 1

            # Check if rollout was successful
            if evaluator is not None:
                evaluator.evaluate_step(env, succeess_term)
                evaluator.summarize_demos()

    # Log evaluation results to a file
    if evaluator is not None:
        evaluator.maybe_write_eval_file()
    if video_writer is not None:
        video_writer.close()
    env.close()


if __name__ == "__main__":
    print("args", args)

    # model and environment related params
    gr00t_n1_policy = Gr00tN1Policy(args)
    env_cfg = create_sim_environment(args)
    evaluator = Gr00tN1Evaluator(args.checkpoint_name, args.eval_file_path, args.seed)

    # Run the closed loop policy.
    run_closed_loop_policy(
        args=args, simulation_app=simulation_app, env_cfg=env_cfg, policy=gr00t_n1_policy, evaluator=evaluator
    )

    # Close simulation app after rollout is complete
    simulation_app.close()
