# Fixing the GR1 Pick Wheel Replay Script

This guide explains how we debugged and fixed the [test_pick_wheel_replay.py](file:///home/minh/Workspace/projects/IsaacLabEvalTasks/test_pick_wheel_replay.py) script to successfully replay recorded episodes of the GR1 robot picking up a steering wheel.

## 1. The Core Problem: Controller Mismatch

### What happened?
The original `Isaac-PickPlace-GR1T2-Abs-v0` environment is designed for **Inverse Kinematics (IK)** control. This means it expects commands in the form of "move the hand to this [x, y, z] position".

However, our recorded dataset contains **Joint Positions** (angles for each of the 26 motors). When we tried to feed these joint angles directly into an environment expecting IK targets, the robot behaved erratically or did not move correctly because the input format did not match the controller.

### The Fix
We modified the script to **override** the environment's default configuration. We replaced the `upper_body_ik` action type with a `JointPositionActionCfg`.

**Code Change:**
```python
# Define the list of 36 joints the robot expects
joint_names = list(joint_names_dict.keys())

# Create a new Action Configuration using Joint Positions
@configclass
class ActionsCfg:
    gr1_action = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=joint_names, 
        scale=1.0, 
        use_default_offset=False
    )

# Replace the environment's action config with our new one
env_cfg.actions = ActionsCfg()
```

This tells the simulator: *Don't use IK. Just move the motors to exactly these angles.*

## 2. Problem: Robot "Laziness" (Tracking Error) / Vấn đề: Robot bị "lười" (Lỗi bám theo quỹ đạo)

### What happened?
> **English:** Even after fixing the controller, the robot might not have followed the trajectory perfectly. The default stiffness (how hard the motors push to hold a position) of the arm joints was too low for direct replay. The robot arms would "sag" or lag behind the target positions.
>
> **Tiếng Việt:** Ngay cả sau khi đã sửa bộ điều khiển (controller), robot vẫn có thể không đi theo đúng quỹ đạo hoàn hảo. Độ cứng mặc định (stiffness) của các khớp tay quá thấp để thực hiện replay trực tiếp. Điều này khiến tay robot bị "xệ" xuống hoặc bị trễ so với vị trí mục tiêu, giống như cánh tay bị yếu lực.

### The Fix
> **English:** We increased the **stiffness** and **damping** gains for the arm joints. This makes the robot "stiffer" and more precise in tracking the replayed motion.
>
> **Tiếng Việt:** Chúng ta đã tăng các chỉ số **stiffness (độ cứng)** và **damping (giảm chấn)** cho các khớp tay. Việc này giúp robot "cứng cáp" hơn và bám theo chuyển động replay một cách chính xác hơn.

#### Giải thích thuật ngữ (Terminology Explanation):
*   **Stiffness (Độ cứng - $K_p$):** Tưởng tượng khớp robot như một cái lò xo. Stiffness càng cao, lò xo càng cứng. Khi stiffness thấp, tay robot sẽ mềm, dễ bị trọng lực kéo xuống. Khi stiffness cao, tay robot sẽ cố gắng giữ đúng vị trí mục tiêu một cách mạnh mẽ bất chấp ngoại lực.
*   **Damping (Giảm chấn - $K_d$):** Đây là lực cản giúp triệt tiêu dao động. Nếu chỉ có độ cứng (stiffness) cao mà damping thấp, cánh tay robot sẽ bị rung lắc (oscillate) mạnh quanh vị trí mục tiêu mỗi khi di chuyển nhanh (giống như lò xo nảy liên tục). Damping giúp chuyển động mượt mà hơn và dừng lại đúng vị trí mà không bị rung.

**Code Change:**
```python
tuned_joint_names = ["left-arm", "right-arm"]
for joint_name in tuned_joint_names:
    env_cfg.scene.robot.actuators[joint_name].stiffness = 3000  # High stiffness (Độ cứng cao)
    env_cfg.scene.robot.actuators[joint_name].damping = 100     # Moderate damping (Giảm chấn vừa phải)
```

## 3. Problem: Missing the Object

### What happened?
The robot was moving correctly, but it would reach for the wheel and miss it, or the wheel would be slightly out of place. 

The [PickPlace](file:///home/minh/Workspace/projects/IsaacLabEvalTasks/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place/pickplace_gr1t2_env_cfg.py#317-445) environment has a feature called `reset_object`, which **randomizes** the object's starting position slightly (e.g., +/- 1cm) every time the environment resets. This is good for training robust policies but bad for replay, because our recorded episode happened with the object in one specific specific spot. Since our dataset didn't record the object's position, we didn't know where it was.

### The Fix
We **disabled object randomization**. By removing the `reset_object` event, the wheel always spawns at its default, deterministic location `[-0.45, 0.45, 0.9996]`. This happened to match the condition under which the original data was recorded.

**Code Change:**
```python
# Disable object randomization to force deterministic spawn pose
env_cfg.events.reset_object = None
```

## Summary
By matching the **Controller** (Joint Position), **Dynamics** (Stiffness), and **Environment State** (Object Position) of the replay script to the conditions of the original recording, we achieved a successful replay.
