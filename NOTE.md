# 1. Introduction
- provides benchmarking scripts for closed-loop evaluation of manipulation policy (i.e. Isaac GR00T N1) with post-trained checkpoints. 
- These scripts enable developers to load prebuilt Isaac Lab environments and industrial tasks—such as nut pouring and pipe sorting—and run standardized benchmarks to quantitatively assess policy performance.

# 2. Installation
- uv venv -p=3.11
- git clone --recurse-submodules git@github.com:isaac-sim/IsaacLabEvalTasks.git
- Change the submodule URL from SSH to HTTPS:
```
  - git config --file=.gitmodules submodule.submodules/Isaac-GR00T.url https://github.com/NVIDIA/Isaac-GR00T.git
  - git submodule sync
  - git submodule update --init --recursive
```
- Using a python interpreter or conda/virtual env that has Isaac Lab installed, install the library required by GrootN1
```
cd submodules/Isaac-GR00T
uv pip install --upgrade setuptools
uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 torchcodec==0.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install flash_attn==2.8.3 --no-build-isolation
uv pip install 
uv pip install -e .
```
- uv pip install -e source/isaaclab_eval_tasks --python submodules/Isaac-GR00T/.venv/bin/python

- Verify that the GR00T deps are correctly installed by running the following command:
```
python -c "import gr00t; print('gr00t imported successfully')"
```

# 3. Tips & Knowledge:
- `num_feedback_actions` determines the number of feedback actions to execute per inference, and it can be less than `action_horizon`. This option will impact the success rate of evaluation task even with the same checkpoint. How many actions from the predicted sequence you actually execute before asking the AI model for new predictions.
- `action_horizon`: What it is: The number of future actions the AI model predicts in one inference call. Why useful: The model can plan ahead and create smoother, more coherent movements rather than making jerky, one-step-at-a-time decisions.
- `rollout_length` impacts how many batched inference to make before task termination. Normally we set it between 20 to 30 for a faster turnaround. The total number of times you ask the AI model for predictions in one episode.

- `num_envs` decides the number of environments to run in parallel. Too many parallel environments (e.g. >100 on RTX A6000 Ada) will significantly slow down the UI rendering. We recommend to set between 10 to 30 for smooth rendering and efficient benchmarking.


# 4. Evaluation scripts:
```
# Within IsaacLabEvalTasks directory
# Assume the post-trained policy checkpoints are under CKPTS_PATH
# Please use full path, instead of relative path for CKPTS_PATH
# export EVAL_RESULTS_FNAME="./eval_nutpouring.json"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_gn1.py \
    --num_feedback_actions 16 \
    --num_envs 10 \
    --task_name nutpouring \
    --eval_file_path ./eval_nutpouring.json \
    --model_path /home/minh/Workspace/projects/IsaacLabEvalTasks/checkpoints \
    --rollout_length 30 \
    --seed 10 \
    --max_num_rollouts 100
```

5. Install: 


pydantic==2.10.6 numpy==1.26.4
numpydantic==1.7.0 --no-build-isolation
uv pip install pinocchio
uv pip install isaaclab[isaacsim,all] --extra-index-url https://pypi.nvidia.com
transformers==4.45.2 # groot N1 
transformers==4.51.3 # groot N1.5


Sau khi build isaacsim, thi build lai torch
6. 
uv pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
uv pip install diffusers peft==0.18.1

source isaacsim/bin/activate && uv pip install -e source/isaaclab_eval_tasks

python scripts/evaluate_gn1.py --num_feedback_actions 16 --num_envs 5 --task_name nutpouring --eval_file_path ./eval_nutpouring.json --model_path /home/minh/Workspace/projects/IsaacLabEvalTasks/checkpoints/grootn1 --rollout_length 30 --seed 10 --max_num_rollouts 100

python scripts/evaluate_gn1.py --num_feedback_actions 16 --num_envs 5 --task_name pick_wheel --eval_file_path ./pick_wheel.json --model_path /home/minh/Workspace/projects/IsaacLabEvalTasks/submodules/Isaac-GR00T/output_pick_wheel/checkpoint-10000 --rollout_length 30 --seed 10 --max_num_rollouts 100 --data_config fourier_gr1_arms_only --embodiment_tag new_embodiment --language_instruction "Pick up the beaker and tilt it to pour out 1 metallic nut into the bowl. Pick up the bowl and place it on the metallic measuring scale."

python scripts/evaluate_gn1.py --num_feedback_actions 16 --num_envs 5 --task_name pick_wheel --eval_file_path ./pick_wheel.json --model_path /home/minh/Workspace/projects/IsaacLabEvalTasks/submodules/Isaac-GR00T/output_pick_wheel_all_layers_r_8/checkpoint-10000 --rollout_length 30 --seed 10 --max_num_rollouts 100 --data_config fourier_gr1_arms_only --embodiment_tag new_embodiment --language_instruction "Pick up the beaker and tilt it to pour out 1 metallic nut into the bowl. Pick up the bowl and place it on the metallic measuring scale."


python scripts/evaluate_gn1.py --num_feedback_actions 16 --num_envs 5 --task_name pick_wheel --eval_file_path ./pick_wheel_standard_r16.json --model_path /home/minh/Workspace/projects/IsaacLabEvalTasks/submodules/Isaac-GR00T/output_generated_dataset_annotated_gr1_standard_r16/checkpoint-10000 --rollout_length 30 --seed 10 --max_num_rollouts 100 --data_config fourier_gr1_arms_only --embodiment_tag new_embodiment --language_instruction "Pick up the beaker and tilt it to pour out 1 metallic nut into the bowl. Pick up the bowl and place it on the metallic measuring scale."

python scripts/evaluate_gn1.py --num_feedback_actions 16 --num_envs 5 --task_name pick_wheel --eval_file_path ./pick_wheel_robotic_r16.json --model_path /home/minh/Workspace/projects/IsaacLabEvalTasks/submodules/Isaac-GR00T/output_generated_dataset_annotated_gr1_standard_r0/checkpoint-10000 --rollout_length 30 --seed 10 --max_num_rollouts 100 --data_config fourier_gr1_arms_only --embodiment_tag new_embodiment --language_instruction "Pick up the beaker and tilt it to pour out 1 metallic nut into the bowl. Pick up the bowl and place it on the metallic measuring scale."
```

8. Convert hdf5 to lerobot:
python scripts/convert_hdf5_to_lerobot.py --data-root /home/minh/Workspace/projects/IsaacLabEvalTasks/dataset --hdf5-name generated_dataset_annotated_gr1.hdf5


python submodules/Isaac-GR00T/scripts/gr00t_finetune_custom_train_all.py \
    --dataset-path dataset/generated_dataset_annotated_gr1 \
    --lora-mode minimum \
    --lora-rank 16
python submodules/Isaac-GR00T/scripts/gr00t_finetune_custom_train_all.py \
    --dataset-path dataset/generated_dataset_annotated_gr1 \
    --lora-mode standard \
    --lora-rank 16

python submodules/Isaac-GR00T/scripts/gr00t_finetune_custom_train_all.py \
    --dataset-path dataset/generated_dataset_annotated_gr1 \
    --lora-mode robotic \
    --lora-rank 16

python submodules/Isaac-GR00T/scripts/gr00t_finetune_custom_train_all.py \
    --dataset-path dataset/generated_dataset_annotated_gr1 \
    --lora-mode full \
    --lora-rank 16

python submodules/Isaac-GR00T/scripts/gr00t_finetune_custom_train_all.py \
    --dataset-path dataset/generated_dataset_annotated_gr1 \
    --lora-mode robotic_vl \
    --lora-rank 16
```