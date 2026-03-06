#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# script to run PyTorch baseline benchmark (no quantization)

# Exit on any error
set -e

RESULTS_FILE="benchmark_results.json"
WORKSPACE_ROOT=$(pwd)
PERF_SCRIPT="submodules/Isaac-GR00T/deployment_scripts/gr00t_convert_performance.py"
MODEL_PATH="/home/minh/Workspace/projects/IsaacLabEvalTasks/submodules/Isaac-GR00T/output_generated_dataset_annotated_gr1_standard_r0"
DATASET_PATH="/home/minh/Workspace/projects/IsaacLabEvalTasks/dataset/generated_dataset_annotated_gr1_test"

echo "Starting PyTorch Baseline Benchmark..."
echo "Results will be appended to: ${RESULTS_FILE}"

# Run performance measurement in PyTorch mode
# Note: PyTorch mode ignores vit/llm/dit-dtype arguments as it loads the full model
python3 "${PERF_SCRIPT}" \
    --inference-mode=pytorch \
    --calibration-size=10 \
    --json-output "${RESULTS_FILE}" \
    --model-path="${MODEL_PATH}" \
    --embodiment-tag="new_embodiment"

echo "========================================================================"
echo "PyTorch Baseline Complete!"
echo "Check ${RESULTS_FILE} for 'config.inference_mode: pytorch' entry."
echo "========================================================================"
