import os
import json
import time
import argparse
import subprocess
import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import load_file
from transformers.feature_extraction_utils import BatchFeature
from functools import partial
from typing import Dict, Any, List


import gr00t
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values, squeeze_dict_values
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config


# Import our custom TRT utilities
import deployment_scripts.trt_torch as trt
from deployment_scripts.trt_model_forward import eagle_tensorrt_forward, action_head_tensorrt_forward
from deployment_scripts.action_head_utils import action_head_pytorch_forward


class ResourceTracker:
    def __init__(self):
        import psutil
        self.process = psutil.Process(os.getpid())
        self.pid = os.getpid()
        self.max_ram_gb = 0
        self.max_vram_gb = 0 # PyTorch-managed
        self.max_proc_vram_gb = 0 # Total Process (from nvidia-smi)
        self.max_other_vram_gb = 0 # Other processes
        self.gpu_total_peak = 0
        
    def start(self):
        torch.cuda.reset_peak_memory_stats()
        self.max_vram_gb = 0
        self.max_ram_gb = 0
        self.max_proc_vram_gb = 0
        self.max_other_vram_gb = 0
        self.gpu_total_peak = 0
        
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
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.max_ram_gb = self.process.memory_info().rss / (1024**3)
        p, o, gt = self.get_nvidia_smi_vram()
        self.max_proc_vram_gb = p
        self.max_other_vram_gb = o
        self.gpu_total_peak = gt

    def update(self):
        ram_gb = self.process.memory_info().rss / (1024**3)
        vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        proc_vram, other_vram, gpu_total = self.get_nvidia_smi_vram()
        
        self.max_ram_gb = max(self.max_ram_gb, ram_gb)
        self.max_vram_gb = max(self.max_vram_gb, vram_gb)
        self.max_proc_vram_gb = max(self.max_proc_vram_gb, proc_vram)
        self.max_other_vram_gb = max(self.max_other_vram_gb, other_vram)
        self.gpu_total_peak = max(self.gpu_total_peak, gpu_total)
        
    def get_stats(self):
        self.update()
        return self.max_ram_gb, self.max_vram_gb, self.max_proc_vram_gb, self.max_other_vram_gb, self.gpu_total_peak

class TimeTracker:
    def __init__(self, name="Inference"):
        self.name = name
        self.times = []
        
    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        self.times.append(end_time - self.start_time)
        
    def stats(self):
        if not self.times: return 0, 0
        return np.mean(self.times) * 1000, np.std(self.times) * 1000

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


class SkeletonModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = nn.Module()
        self.action_head = nn.Module()

def load_glue_weights(model_dir, weight_map_path):
    with open(weight_map_path, 'r') as f:
        weight_map = json.load(f)['weight_map']
    
    needed_keys = {
        "backbone.eagle_model.mlp1.0.weight": "mlp1.0.weight",
        "backbone.eagle_model.mlp1.0.bias": "mlp1.0.bias",
        "backbone.eagle_model.language_model.model.embed_tokens.weight": "embed_tokens.weight",
        "action_head.position_embedding.weight": "action_pos_embed.weight",
        "action_head.future_tokens.weight": "future_tokens.weight"
    }
    
    # Map shard files to keys we need from them
    shard_to_keys = {}
    for full_key, short_name in needed_keys.items():
        shard = weight_map[full_key]
        if shard not in shard_to_keys:
            shard_to_keys[shard] = []
        shard_to_keys[shard].append((full_key, short_name))
        
    weights = {}
    for shard, keys in shard_to_keys.items():
        shard_path = os.path.join(model_dir, shard)
        print(f"Loading selective weights from {shard}...")
        shard_data = load_file(shard_path, device="cpu")
        for full_key, short_name in keys:
            weights[short_name] = shard_data[full_key].cuda().to(torch.float16)
            
    return weights

def main():
    parser = argparse.ArgumentParser(description="GR00T Direct TensorRT Benchmarking")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--trt-engine-path", type=str, required=True)
    parser.add_argument("--vit-dtype", type=str, default="fp8")
    parser.add_argument("--llm-dtype", type=str, default="nvfp4")
    parser.add_argument("--dit-dtype", type=str, default="fp8")
    parser.add_argument("--embodiment", type=str, default="new_embodiment")
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--calibration-size", type=int, default=0, help="Number of samples for accuracy measurement")
    parser.add_argument("--data-config", type=str, default="fourier_gr1_arms_only")
    parser.add_argument("--video-backend", type=str, choices=["decord", "torchcodec"], default="decord")
    args = parser.parse_args()

    res_tracker = ResourceTracker()
    res_tracker.start()

    print(f"--- Direct TensorRT Loading Mode ---")
    
    # 1. Load Configs (Metadata only)
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    # 2. Load "Glue" Weights using Safetensors
    weight_index_path = os.path.join(args.model_path, "model.safetensors.index.json")
    glue_weights = load_glue_weights(args.model_path, weight_index_path)
    
    # 3. Create Skeleton Policy Objects
    # We cheat a bit by using Gr00tPolicy but overriding its model after init if possible,
    # or just manually building the minimal state.
    # Actually, it's easier to use Gr00tPolicy but pass it dummy paths/configs initially.
    # But Gr00tPolicy.__init__ calls self._load_model.
    
    # Let's verify what we need for the forward pass functions
    # eagle_tensorrt_forward needs: self.vit_engine, self.llm_engine, self.embedding_layer, self.mlp1, self.image_token_index, etc.
    
    class LeanBackbone(nn.Module):
        def __init__(self, glue_weights):
            super().__init__()
            self.vit_engine = None
            self.llm_engine = None
            self.num_patches = 256 # Matches vit_fp8.engine position_ids: -1x256
            self.image_token_index = 151669 
            self.embedding_layer = nn.Embedding(151680, 2048)
            self.embedding_layer.weight.data = glue_weights['embed_tokens.weight']
            self.eagle_model = nn.Module()
            self.eagle_model.mlp1 = nn.Sequential(nn.Linear(1152, 2048, bias=True))
            self.eagle_model.mlp1[0].weight.data = glue_weights['mlp1.0.weight']
            self.eagle_model.mlp1[0].bias.data = glue_weights['mlp1.0.bias']
            self.eagle_model.use_pixel_shuffle = False # Adjust if needed
            self.eagle_model.mlp_checkpoint = False

        def set_frozen_modules_to_eval_mode(self):
            pass

    class LeanActionHead(nn.Module):
        def __init__(self, glue_weights):
            super().__init__()
            self.config = nn.Module()
            self.config.action_horizon = 16 
            self.config.action_dim = 32
            self.config.add_pos_embed = True
            self.action_horizon = 16
            self.action_dim = 32
            self.num_inference_timesteps = 4
            self.num_timestep_buckets = 1000
            self.position_embedding = nn.Embedding(1024, 1536)
            self.position_embedding.weight.data = glue_weights['action_pos_embed.weight']
            self.future_tokens = nn.Embedding(32, 1536)
            self.future_tokens.weight.data = glue_weights['future_tokens.weight']

    backbone = LeanBackbone(glue_weights).cuda().to(torch.float16)
    action_head = LeanActionHead(glue_weights).cuda().to(torch.float16)
    
    # Snapshot 1: Weights & Skeleton Load
    vram_weights_baseline, _, _ = res_tracker.get_nvidia_smi_vram()
    
    # 4. Load Engines
    print("Loading TensorRT Engines...")
    backbone.vit_engine = trt.Engine(os.path.join(args.trt_engine_path, f"vit_{args.vit_dtype}.engine"))
    backbone.llm_engine = trt.Engine(os.path.join(args.trt_engine_path, f"llm_{args.llm_dtype}.engine"))
    
    action_head.vlln_vl_self_attention_engine = trt.Engine(os.path.join(args.trt_engine_path, "vlln_vl_self_attention.engine"))
    action_head.state_encoder_engine = trt.Engine(os.path.join(args.trt_engine_path, "state_encoder.engine"))
    action_head.action_encoder_engine = trt.Engine(os.path.join(args.trt_engine_path, "action_encoder.engine"))
    action_head.DiT_engine = trt.Engine(os.path.join(args.trt_engine_path, f"DiT_{args.dit_dtype}.engine"))
    action_head.action_decoder_engine = trt.Engine(os.path.join(args.trt_engine_path, "action_decoder.engine"))

    # Snapshot 2: Engines & Context Load
    vram_after_engines, _, _ = res_tracker.get_nvidia_smi_vram()
    vram_engine_delta = vram_after_engines - vram_weights_baseline

    # 5. Patch Forward Functions
    backbone.forward = partial(eagle_tensorrt_forward, backbone)
    action_head.get_action = partial(action_head_tensorrt_forward, action_head)

    # 6. Setup Dataset / Dummy Input
    # We'll use random input for performance measurement
    # Shapes adjusted to match gr00t_engine
    pixel_values = torch.randn(1, 3, 224, 224).cuda().to(torch.float16)
    input_ids = torch.ones(1, 512).long().cuda() * 100 # Adjusted length
    input_ids[0, 100:100+256] = backbone.image_token_index
    attention_mask = torch.ones(1, 512).long().cuda()
    state = torch.randn(1, 1, 64).cuda().to(torch.float16) # Matches state_encoder -1x1x64
    embodiment_id = torch.tensor([0]).cuda().long()

    vl_input = {
        "eagle_pixel_values": pixel_values,
        "eagle_input_ids": input_ids,
        "eagle_attention_mask": attention_mask,
        "eagle_image_sizes": None
    }
    action_input = nn.Module()
    action_input.state = state
    action_input.embodiment_id = embodiment_id

    # Warmup
    print("Warmup...")
    for _ in range(5):
        bb_out = backbone.forward(vl_input)
        action_head.get_action(bb_out, action_input)

    # res_tracker.reset_peak() # Captured absolute peak including load
    time_tracker = TimeTracker()

    print(f"Running {args.num_iters} iterations...")
    for _ in range(args.num_iters):
        with time_tracker:
            bb_out = backbone.forward(vl_input)
            action_head.get_action(bb_out, action_input)
        res_tracker.update()

    avg_ms, std_ms = time_tracker.stats()
    max_ram, max_vram, max_proc_vram, max_other_vram, gpu_total_peak = res_tracker.get_stats()

    # --- Accuracy Measurement ---
    accuracy_results = {}
    if args.calibration_size > 0:
        print(f"\nEvaluating accuracy against PyTorch over {args.calibration_size} samples...")
        
        # Load dataset if not provided
        if args.dataset_path is None:
            submodule_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            workspace_root = os.path.dirname(os.path.dirname(submodule_root))
            args.dataset_path = os.path.join(workspace_root, "dataset/generated_dataset_annotated_gr1_test")
            
        data_cfg = load_data_config(args.data_config)
        modality_config = data_cfg.modality_config()
        modality_transform = data_cfg.transform()
        
        dataset = LeRobotSingleDataset(
            dataset_path=args.dataset_path,
            modality_configs=modality_config,
            video_backend=args.video_backend,
            video_backend_kwargs=None,
            transforms=None,
            embodiment_tag=args.embodiment,
        )

        # Load PyTorch Policy for comparison
        policy_torch = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=args.embodiment,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=4,
            device="cuda"
        )
        
        # Ensure identical init_actions
        init_actions = torch.zeros(
            1, policy_torch.model.action_head.action_horizon, policy_torch.model.action_head.action_dim,
            dtype=torch.float16, device="cuda",
        )
        policy_torch.model.action_head.init_actions = init_actions
        action_head.init_actions = init_actions
        
        # Swap action head forward to pytorch for the reference policy
        policy_torch.model.action_head.get_action = partial(
            action_head_pytorch_forward, policy_torch.model.action_head
        )

        accuracy_stats = []
        n_samples = min(args.calibration_size, len(dataset))
        
        class ActionInputMap:
            def __init__(self, data):
                self.state = data["state"]
                self.embodiment_id = data["embodiment_id"]

        for i in range(n_samples):
            data = dataset[i]
            with torch.no_grad():
                # 1. TRT Prediction (using lean model)
                # Ensure data is batched before transform so GR00TTransform's collate/processor is called
                batched_data = unsqueeze_dict_values(data)
                transformed_data = modality_transform(batched_data)
                transformed_data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in transformed_data.items()}
                if "eagle_image_sizes" not in transformed_data:
                    transformed_data["eagle_image_sizes"] = None
                
                bb_out_trt = backbone.forward(transformed_data)
                pred_trt_raw = action_head.get_action(bb_out_trt, ActionInputMap(transformed_data))
                
                # Unnormalize TRT prediction to get matching keys and space
                pred_trt_unnorm = policy_torch.unapply_transforms({"action": pred_trt_raw.action_pred.cpu()})
                pred_trt = squeeze_dict_values(pred_trt_unnorm)
                
                # 2. PyTorch Prediction (using full policy)
                pred_torch = policy_torch.get_action(data)
                
                accuracy_stats.append(compare_predictions(pred_trt, pred_torch))

        # Summarize Accuracy
        if accuracy_stats:
            keys = list(accuracy_stats[0].keys())
            for key in keys:
                avg_cos = float(np.mean([s[key]['cosine_similarity'] for s in accuracy_stats]))
                max_diff = float(np.max([s[key]['max_difference'] for s in accuracy_stats]))
                mean_diff = float(np.mean([s[key]['mean_difference'] for s in accuracy_stats]))
                
                accuracy_results[key] = {
                    "cosine_similarity": avg_cos,
                    "max_difference": max_diff,
                    "mean_difference": mean_diff
                }
                print(f"Accuracy Fidelity ({key}): {avg_cos:.6f} CosSim, {max_diff:.6f} MaxDiff")

    print(f"\nResults (Direct TRT):")
    print(f" Latency: {avg_ms:.2f} ms ± {std_ms:.2f}")
    # ... (rest of print statements simplified for brevity in this replacement) ...
    print(f"Resource Breakdown (VRAM):")
    print(f"  1. Weights & Glue Load: {vram_weights_baseline:.4f} GB")
    print(f"  2. TensorRT Engines:    {vram_engine_delta:.4f} GB")
    print(f"  -----------------------------------------------------------")
    print(f"  Total Session Peak:     {max_proc_vram:.2f} GB (Actual footprint)")
    
    print(f"\nOther Stats:")
    print(f" GPU Global Peak:         {gpu_total_peak:.2f} GB")
    print(f" Other VRAM (Background): {max_other_vram:.2f} GB")
    print(f" Max RAM (System):        {max_ram:.2f} GB")

    if args.json_output:
        result = {
            "config": {
                "vit": args.vit_dtype,
                "llm": args.llm_dtype,
                "dit": args.dit_dtype,
                "inference_mode": "direct_tensorrt",
                "calibration_size": args.calibration_size
            },
            "resources": {
                "total_vram_required_gb": round(max_proc_vram, 3),
                "gpu_global_peak_gb": round(gpu_total_peak, 3),
                "trt_workspace_gb": round(max_proc_vram - max_vram, 3),
                "pytorch_vram_gb": round(max_vram, 3),
                "other_vram_gb": round(max_other_vram, 3),
                "max_ram_gb": round(max_ram, 3)
            },
            "performance": {
                "latency_ms": round(avg_ms, 2),
                "latency_std_ms": round(std_ms, 2)
            },
            "accuracy": accuracy_results
        }
        
        all_results = []
        if os.path.exists(args.json_output):
            try:
                with open(args.json_output, 'r') as f:
                    all_results = json.load(f)
                    if not isinstance(all_results, list):
                        all_results = [all_results]
            except:
                pass
        
        all_results.append(result)
        with open(args.json_output, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"Results saved to {args.json_output}")

if __name__ == "__main__":
    main()
