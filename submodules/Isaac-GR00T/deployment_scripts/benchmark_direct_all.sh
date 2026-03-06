#!/bin/bash

# bash submodules/Isaac-GR00T/deployment_scripts/benchmark_direct_all.sh
# PYTHONPATH=submodules/Isaac-GR00T/ quant/bin/python3 submodules/Isaac-GR00T/deployment_scripts/gr00t_direct_trt_performance.py \
#     --model-path /path/to/model \
#     --trt-engine-path /path/to/engines \
#     --vit-dtype fp8 --llm-dtype nvfp4 --dit-dtype fp8
    
# Configuration
MODEL_PATH="/home/minh/Workspace/projects/IsaacLabEvalTasks/submodules/Isaac-GR00T/output_generated_dataset_annotated_gr1_standard_r0"
TRT_ENGINE_PATH="/home/minh/Workspace/projects/IsaacLabEvalTasks/gr00t_engine"
PYTHON_BIN="/home/minh/Workspace/projects/IsaacLabEvalTasks/quant/bin/python3"
SCRIPT_PATH="/home/minh/Workspace/projects/IsaacLabEvalTasks/submodules/Isaac-GR00T/deployment_scripts/gr00t_direct_trt_performance.py"
OUTPUT_JSON="/home/minh/Workspace/projects/IsaacLabEvalTasks/submodules/Isaac-GR00T/deployment_scripts/direct_benchmark_results.json"
DATASET_PATH="/home/minh/Workspace/projects/IsaacLabEvalTasks/dataset/generated_dataset_annotated_gr1_test"
DATA_CONFIG="fourier_gr1_arms_only"


# Precisions to test
VIT_DTYPES=("fp16" "fp8")
LLM_DTYPES=("fp16" "fp8" "nvfp4")
DIT_DTYPES=("fp16" "fp8")

# Clear old results
rm -f "$OUTPUT_JSON"

echo "=== Starting Direct TRT Multi-Precision Benchmark ==="
echo "Recording results to $OUTPUT_JSON"

for vit in "${VIT_DTYPES[@]}"; do
    for llm in "${LLM_DTYPES[@]}"; do
        for dit in "${DIT_DTYPES[@]}"; do
            echo "Testing Vit:$vit LLM:$llm DiT:$dit..."
            
            # Using single line to avoid backslash issues in some shells
            PYTHONPATH="submodules/Isaac-GR00T/" "$PYTHON_BIN" "$SCRIPT_PATH" \
                --model-path "$MODEL_PATH" \
                --trt-engine-path "$TRT_ENGINE_PATH" \
                --vit-dtype "$vit" --llm-dtype "$llm" --dit-dtype "$dit" \
                --num-iters 1 \
                --calibration-size 10 \
                --dataset-path "$DATASET_PATH" \
                --data-config "$DATA_CONFIG" \
                --json-output "$OUTPUT_JSON"

            
            # Brief sleep to ensure VRAM is cleared between runs
            sleep 5
        done
    done
done

echo "=== Benchmarking Complete ==="
