#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# script to automate benchmarking of GR00T models with different data types

# Exit on any error
set -e

# Define data types to test
VIT_DTYPES=("fp16" "fp8")
LLM_DTYPES=("fp16" "nvfp4" "fp8")
DIT_DTYPES=("fp16" "fp8")

RESULTS_FILE="benchmark_results.json"
WORKSPACE_ROOT=$(pwd)
BUILD_SCRIPT="submodules/Isaac-GR00T/deployment_scripts/build_engine.sh"
PERF_SCRIPT="submodules/Isaac-GR00T/deployment_scripts/gr00t_convert_performance.py"
EXPORT_SCRIPT="submodules/Isaac-GR00T/deployment_scripts/export_onnx.py"
MODEL_PATH="/home/minh/Workspace/projects/IsaacLabEvalTasks/submodules/Isaac-GR00T/output_generated_dataset_annotated_gr1_standard_r0"
DATASET_PATH="/home/minh/Workspace/projects/IsaacLabEvalTasks/dataset/generated_dataset_annotated_gr1_test"

echo "Starting Benchmark Automation..."
echo "Results will be saved to: ${RESULTS_FILE}"

# Clear existing results
if [ -f "${RESULTS_FILE}" ]; then
    rm "${RESULTS_FILE}"
fi

for vit in "${VIT_DTYPES[@]}"; do
    for llm in "${LLM_DTYPES[@]}"; do
        for dit in "${DIT_DTYPES[@]}"; do
            echo "========================================================================"
            echo "Benchmarking Mode: ViT=${vit}, LLM=${llm}, DiT=${dit}"
            echo "========================================================================"

            # 1. Export ONNX if needed (actually we run it every time to be sure, or we can check)
            # Checking logic for speedup:
            NEED_EXPORT=false
            if [ ! -f "gr00t_onnx/eagle2/vit_${vit}.onnx" ] || \
               [ ! -f "gr00t_onnx/eagle2/llm_${llm}.onnx" ] || \
               [ ! -f "gr00t_onnx/action_head/DiT_${dit}.onnx" ]; then
                NEED_EXPORT=true
            fi

            if [ "$NEED_EXPORT" = true ]; then
                echo "[Step 1/3] Exporting ONNX models..."
                python3 "${EXPORT_SCRIPT}" \
                    --model-path="${MODEL_PATH}" \
                    --dataset-path="${DATASET_PATH}" \
                    --vit-dtype="${vit}" \
                    --llm-dtype="${llm}" \
                    --dit-dtype="${dit}" \
                    --calib-size=10 \
                    --embodiment-tag="new_embodiment"
            else
                echo "[Step 1/3] ONNX models already exist, skipping export."
            fi

            # 2. Rebuild engines if needed
            NEED_BUILD=false
            TRT_ENGINE_DIR="gr00t_engine"
            if [ ! -f "${TRT_ENGINE_DIR}/vit_${vit}.engine" ] || \
               [ ! -f "${TRT_ENGINE_DIR}/llm_${llm}.engine" ] || \
               [ ! -f "${TRT_ENGINE_DIR}/DiT_${dit}.engine" ] || \
               [ ! -f "${TRT_ENGINE_DIR}/vlln_vl_self_attention.engine" ] || \
               [ ! -f "${TRT_ENGINE_DIR}/state_encoder.engine" ] || \
               [ ! -f "${TRT_ENGINE_DIR}/action_encoder.engine" ] || \
               [ ! -f "${TRT_ENGINE_DIR}/action_decoder.engine" ]; then
                NEED_BUILD=true
            fi

            if [ "$NEED_BUILD" = true ]; then
                echo "[Step 2/3] Rebuilding TensorRT engines..."
                export VIT_DTYPE=${vit}
                export LLM_DTYPE=${llm}
                export DIT_DTYPE=${dit}
                bash "${BUILD_SCRIPT}"
            else
                echo "[Step 2/3] TensorRT engines already exist, skipping build."
            fi

            # 3. Run performance measurement
            echo "[Step 3/3] Running performance script..."
            python3 "${PERF_SCRIPT}" \
                --inference-mode=tensorrt \
                --vit-dtype="${vit}" \
                --llm-dtype="${llm}" \
                --dit-dtype="${dit}" \
                --calibration-size=10 \
                --json-output "${RESULTS_FILE}" \
                --model-path="${MODEL_PATH}" \
                --embodiment-tag="new_embodiment"

            echo "Done. Cooling down for 30 seconds..."
            sleep 30
        done
    done
done

echo "========================================================================"
echo "Benchmark Automation Complete!"
echo "Final results available in: ${RESULTS_FILE}"
echo "========================================================================"
