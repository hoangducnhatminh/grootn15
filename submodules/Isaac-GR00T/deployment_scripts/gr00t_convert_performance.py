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

import argparse
import os
import time
import threading
import psutil
import json
import subprocess
from functools import partial
from typing import Dict, Tuple, Callable, Any, List

import torch
import numpy as np
from action_head_utils import action_head_pytorch_forward
import sys

# Add the submodule root to sys.path to allow 'import deployment_scripts'
submodule_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if submodule_root not in sys.path:
    sys.path.append(submodule_root)

from trt_model_forward import setup_tensorrt_engines

import gr00t
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values
from gr00t.experiment.data_config import load_data_config


class ResourceTracker:
    """
    Track VRAM and RAM consumption.
    Modified to catch total process VRAM (including TRT) via nvidia-smi.
    """
    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.pid = os.getpid()
        self.peak_ram = 0
        self.vram_peak = 0 # PyTorch managed
        self.proc_vram_peak = 0 # Total Process (from nvidia-smi)

    def __enter__(self):
        # Clear GPU memory stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.vram_baseline = torch.cuda.memory_allocated() / (1024**3)
        
        self.ram_baseline = self.process.memory_info().rss / (1024**3)
        self.peak_ram = self.ram_baseline
        self.proc_vram_peak, _, gt = self.get_nvidia_smi_vram()
        self.gpu_total_peak = gt
        
        self.stop_thread = False
        self.thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.thread.start()
        return self

    def get_nvidia_smi_vram(self):
        try:
            # Query memory per PID and total GPU memory
            cmd_p = "nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits"
            output_p = subprocess.check_output(cmd_p, shell=True).decode('utf-8').strip().split('\n')
            
            proc_mem = 0
            other_mem = 0
            for line in output_p:
                if not line.strip(): continue
                parts = line.split(',')
                if len(parts) != 2: continue
                p, mem = parts
                if int(p.strip()) == self.pid:
                    proc_mem += int(mem.strip()) / 1024.0 # Accumulate just in case
                else:
                    other_mem += int(mem.strip()) / 1024.0
            
            cmd_g = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
            output_g = subprocess.check_output(cmd_g, shell=True).decode('utf-8').strip().split('\n')
            total_gpu_mem = float(output_g[0].strip()) / 1024.0
            
            return proc_mem, other_mem, total_gpu_mem
        except:
            return 0, 0, 0

    def reset_peak(self):
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.peak_ram = self.process.memory_info().rss / (1024**3)
        p, o, gt = self.get_nvidia_smi_vram()
        self.proc_vram_peak = p
        self.gpu_total_peak = gt

    def _monitor_resources(self):
        while not self.stop_thread:
            try:
                # Poll RAM
                curr_ram = self.process.memory_info().rss / (1024**3)
                if curr_ram > self.peak_ram: 
                    self.peak_ram = curr_ram
                
                # Poll Process VRAM and Global VRAM
                curr_proc_vram, _, curr_gpu_total = self.get_nvidia_smi_vram()
                if curr_proc_vram > self.proc_vram_peak:
                    self.proc_vram_peak = curr_proc_vram
                if curr_gpu_total > self.gpu_total_peak:
                    self.gpu_total_peak = curr_gpu_total
            except: 
                break
            time.sleep(self.interval)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_thread = True
        if self.thread: 
            self.thread.join()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
            # Failsafe for final peaks
            p, _, gt = self.get_nvidia_smi_vram()
            self.proc_vram_peak = max(self.proc_vram_peak, p)
            self.gpu_total_peak = max(self.gpu_total_peak, gt)
            
            torch.cuda.empty_cache() 
            torch.cuda.synchronize()
        
        self.ram_cost = self.peak_ram - self.ram_baseline

    def stats(self) -> Dict[str, Dict[str, float]]:
        return {
            "vram": {
                "Total Required (GB)": self.proc_vram_peak,
                "GPU Global Peak (GB)": self.gpu_total_peak,
                "PyTorch Managed (GB)": self.vram_peak,
                "Engines/Workspace (GB)": max(0, self.proc_vram_peak - self.vram_peak)
            },
            "ram": {
                "Peak (GB)": self.peak_ram,
                "Cost (GB)": self.ram_cost
            }
        }


class TimeTracker:
    """
    Track inference latency.
    """
    def __init__(self, iterations: int = 30, warmup: int = 5):
        self.iterations = iterations
        self.warmup = warmup
        self.avg_time = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return self

    def run(self, func: Callable[[], None]) -> float:
        # Warm-up
        for _ in range(self.warmup):
            func()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed runs
        start_time = time.perf_counter()
        for _ in range(self.iterations):
            func()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        self.avg_time = ((time.perf_counter() - start_time) / self.iterations) * 1000
        return self.avg_time

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def compare_predictions(pred_tensorrt: Dict[str, np.ndarray], pred_torch: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Compare similarity and difference between TensorRT and PyTorch predictions.
    """
    assert pred_tensorrt.keys() == pred_torch.keys(), "Prediction keys do not match"
    
    overall_results = {}

    for key in pred_tensorrt.keys():
        tensorrt_tensor = torch.from_numpy(pred_tensorrt[key]).to(torch.float32)
        torch_tensor = torch.from_numpy(pred_torch[key]).to(torch.float32)

        assert tensorrt_tensor.shape == torch_tensor.shape, f"{key} shapes do not match"

        # Cosine Similarity
        flat_tensorrt = tensorrt_tensor.flatten()
        flat_torch = torch_tensor.flatten()
        
        cos_sim = torch.nn.functional.cosine_similarity(
            flat_tensorrt.unsqueeze(0), flat_torch.unsqueeze(0)
        ).item()

        # L1 Distances
        diff = torch.abs(flat_tensorrt - flat_torch)
        l1_mean = diff.mean().item()
        l1_max = diff.max().item()
        l1_min = diff.min().item()

        overall_results[key] = {
            "cosine_similarity": cos_sim,
            "max_difference": l1_max,
            "min_difference": l1_min,
            "mean_difference": l1_mean
        }
    
    return overall_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure GR00T performance and accuracy")
    parser.add_argument("--model-path", type=str, default="/home/minh/Workspace/projects/IsaacLabEvalTasks/submodules/Isaac-GR00T/output_generated_dataset_annotated_gr1_standard_r0", help="Path to the GR00T model")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--data-config", type=str, default="fourier_gr1_arms_only", help="Data config name")
    parser.add_argument("--embodiment-tag", type=str, default="new_embodiment", help="Embodiment tag")
    parser.add_argument("--inference-mode", type=str, choices=["pytorch", "tensorrt"], default="tensorrt", help="Inference mode")
    parser.add_argument("--denoising-steps", type=int, default=4, help="Number of denoising steps")
    workspace_root = os.path.dirname(os.path.dirname(submodule_root))
    parser.add_argument("--trt-engine-path", type=str, default=os.path.join(workspace_root, "gr00t_engine"), help="Path to the TensorRT engine")
    parser.add_argument("--vit-dtype", type=str, choices=["fp16", "fp8"], default="fp8")
    parser.add_argument("--llm-dtype", type=str, choices=["fp16", "nvfp4", "fp8"], default="nvfp4")
    parser.add_argument("--dit-dtype", type=str, choices=["fp16", "fp8"], default="fp8")
    parser.add_argument("--video-backend", type=str, choices=["decord", "torchcodec"], default="decord")
    parser.add_argument("--calibration-size", type=int, default=10, help="Number of samples for accuracy measurement")
    parser.add_argument("--json-output", type=str, default=None, help="Path to save results in JSON format")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data config
    data_config = load_data_config(args.data_config)
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    workspace_root = os.path.dirname(os.path.dirname(submodule_root))
    default_dataset = os.path.join(workspace_root, "dataset/generated_dataset_annotated_gr1_test")
    DATASET_PATH = args.dataset_path if args.dataset_path else default_dataset

    # Initialize Resource Tracker for Memory Measurement
    with ResourceTracker(interval=0.01) as res_tracker:
        # Load dataset
        dataset = LeRobotSingleDataset(
            dataset_path=DATASET_PATH,
            modality_configs=modality_config,
            video_backend=args.video_backend,
            video_backend_kwargs=None,
            transforms=None,
            embodiment_tag=args.embodiment_tag,
        )
        
        vram_start, _, _ = res_tracker.get_nvidia_smi_vram()
        
        # Load Policy
        policy = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=args.denoising_steps,
            device=device
        )
        # Snapshot 1: PyTorch Baseline
        vram_after_pt_load, _, _ = res_tracker.get_nvidia_smi_vram()
        vram_pt_model_cost = vram_after_pt_load - vram_start
        
        # Setup TensorRT if requested
        vram_trt_overhead = 0
        if args.inference_mode == "tensorrt":
            print("\nSwapping PyTorch weights for TensorRT engines...")
            setup_tensorrt_engines(
                policy, args.trt_engine_path, args.vit_dtype, args.llm_dtype, args.dit_dtype
            )
            # Snapshot 2: After TRT swap (PT weights deleted inside setup_tensorrt_engines)
            vram_after_trt_swap, _, _ = res_tracker.get_nvidia_smi_vram()
            # The "Overhead" is the footprint of engines alone
            vram_trt_overhead = vram_after_trt_swap # Absolute footprint after PT delete
        
        # First inference (warmup/peak capture)
        step_data = dataset[0]
        _ = policy.get_action(step_data)

        # Initialize Time Tracker for Latency Measurement
        with TimeTracker() as time_tracker:
            inference_time = time_tracker.run(lambda: policy.get_action(step_data))

        # Accuracy Measurement (if in TensorRT mode, compare against PyTorch)
        accuracy_stats = []
        if args.inference_mode == "tensorrt":
            print(f"\nEvaluating accuracy against PyTorch over {args.calibration_size} samples...")
            
            # Create a PyTorch-only policy for comparison
            policy_torch = Gr00tPolicy(
                model_path=args.model_path,
                embodiment_tag=args.embodiment_tag,
                modality_config=modality_config,
                modality_transform=modality_transform,
                denoising_steps=args.denoising_steps,
                device=device
            )
            vram_accuracy_peak, _, _ = res_tracker.get_nvidia_smi_vram()
            
            # Ensure identical init_actions for consistent comparison
            init_actions = torch.zeros(
                1, policy_torch.model.action_head.action_horizon, policy_torch.model.action_head.action_dim,
                dtype=torch.float16, device=device,
            )
            policy_torch.model.action_head.init_actions = init_actions
            policy.model.action_head.init_actions = init_actions
            
            # PyTorch mode for calculation
            policy_torch.model.action_head.get_action = partial(
                action_head_pytorch_forward, policy_torch.model.action_head
            )

            n_samples = min(args.calibration_size, len(dataset))
            for i in range(n_samples):
                data = dataset[i]
                with torch.no_grad():
                    pred_torch = policy_torch.get_action(data)
                    pred_trt = policy.get_action(data)
                
                sample_diff = compare_predictions(pred_trt, pred_torch)
                accuracy_stats.append(sample_diff)

    # Summarize results
    res_stats = res_tracker.stats()
    
    print(f"\n{'='*20} Convert TensorRT Performance {args.vit_dtype}, {args.llm_dtype}, {args.dit_dtype} {'='*20}")
    print("VRAM Step-by-Step Breakdown:")
    print(f"  1. PyTorch Model Weight Cost:  {max(0, vram_pt_model_cost):.4f} GB")
    
    if args.inference_mode == "tensorrt":
        print(f"  2. TensorRT Engine Cost:       {max(0, vram_trt_overhead):.4f} GB")
        print(f"     (Note: TRT setup deletes the PyTorch weights to save memory)")
        print(f"  3. Accuracy Eval Peak:         {vram_accuracy_peak:.4f} GB")
        print(f"     (Note: Re-loading PyTorch model while TRT engines remain)")
        print(f"  -----------------------------------------------------------")
        print(f"  Absolute Session Peak:         {res_stats['vram']['Total Required (GB)']:.4f} GB")
    else:
        print(f"  Total Session Peak:            {res_stats['vram']['Total Required (GB)']:.4f} GB")

    print(f"\nResource Stats (Summary):")
    print(f"  + GPU Global Peak:     {res_stats['vram']['GPU Global Peak (GB)']:.4f} GB")
    print(f"  + RAM Peak:            {res_stats['ram']['Peak (GB)']:.4f} GB")
    print(f"Inference Time:          {inference_time:.4f} ms")
    
    if accuracy_stats:
        # Average stats across keys and samples
        keys = list(accuracy_stats[0].keys())
        for key in keys:
            avg_cos = np.mean([s[key]['cosine_similarity'] for s in accuracy_stats])
            max_diff = np.max([s[key]['max_difference'] for s in accuracy_stats])
            min_diff = np.min([s[key]['min_difference'] for s in accuracy_stats])
            mean_diff = np.mean([s[key]['mean_difference'] for s in accuracy_stats])
            
            print(f"\nAccuracy Fidelity ({key}) based on calibration_size={args.calibration_size}:")
            print(f"  + Cosine Similarity: {avg_cos:.6f}")
            print(f"  + Maximum Difference: {max_diff:.6f}")
            print(f"  + Minimum Difference: {min_diff:.6f}")
            print(f"  + Mean Difference:    {mean_diff:.6f}")

    print(f"\n{'='*70}")

    if args.json_output:
        results = {
            "config": {
                "inference_mode": args.inference_mode,
                "vit_dtype": args.vit_dtype,
                "llm_dtype": args.llm_dtype,
                "dit_dtype": args.dit_dtype,
                "model_path": args.model_path,
                "calibration_size": args.calibration_size,
                "denoising_steps": args.denoising_steps
            },
            "resources": {
                "total_vram_required_gb": res_stats['vram']['Total Required (GB)'],
                "trt_workspace_gb": res_stats['vram']['Engines/Workspace (GB)'],
                "pytorch_vram_gb": res_stats['vram']['PyTorch Managed (GB)'],
                "peak_ram_gb": res_stats['ram']['Peak (GB)']
            },
            "performance": {
                "inference_time_ms": inference_time
            },
            "accuracy": {}
        }
        
        if accuracy_stats:
            for key in accuracy_stats[0].keys():
                results["accuracy"][key] = {
                    "cosine_similarity": float(np.mean([s[key]['cosine_similarity'] for s in accuracy_stats])),
                    "max_difference": float(np.max([s[key]['max_difference'] for s in accuracy_stats])),
                    "min_difference": float(np.min([s[key]['min_difference'] for s in accuracy_stats])),
                    "mean_difference": float(np.mean([s[key]['mean_difference'] for s in accuracy_stats]))
                }
        
        # Load existing results if file exists
        all_results = []
        if os.path.exists(args.json_output):
            try:
                with open(args.json_output, 'r') as f:
                    all_results = json.load(f)
                    if not isinstance(all_results, list):
                        all_results = [all_results]
            except:
                pass
        
        all_results.append(results)
        
        with open(args.json_output, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"Results saved to {args.json_output}")
