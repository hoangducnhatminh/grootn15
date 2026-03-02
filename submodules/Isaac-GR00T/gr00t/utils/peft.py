import math
import torch
from peft import LoraConfig, get_peft_model


class LoraCategorySpecificLinear(torch.nn.Module):
    def __init__(self, original_module, r=16, alpha=16, dropout=0.0):
        super().__init__()
        self.original_module = original_module
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Shared LoRA for all categories (embodiments)
        in_features = original_module.W.shape[1]
        out_features = original_module.W.shape[2]
        
        self.lora_A = torch.nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = torch.nn.Parameter(torch.zeros(r, out_features))
        
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
        
        self.lora_dropout = torch.nn.Dropout(p=dropout) if dropout > 0 else torch.nn.Identity()
        
        # Freeze base parameters
        self.original_module.requires_grad_(False)

    def forward(self, x, cat_ids):
        # Base forward
        result = self.original_module(x, cat_ids)
        # LoRA forward (shared across embodiments)
        # result: (B, T, out_features)
        # x: (B, T, in_features)
        lora_delta = self.lora_dropout(x) @ (self.lora_A @ self.lora_B) * self.scaling
        return result + lora_delta


def _wrap_forward(model):
    def _forward(inputs):
        backbone_inputs, action_inputs = model.prepare_input(inputs)
        backbone_outputs = model.backbone(backbone_inputs)
        action_head_outputs = model.action_head(backbone_outputs, action_inputs)
        model.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    model.forward = _forward
    return model


def get_lora_model(model, rank=32, lora_alpha=16, lora_dropout=0.1, action_head_only=True, target_prefixes=None):
    from gr00t.model.action_head.flow_matching_action_head import CategorySpecificLinear
    
    standard_targets = []
    custom_targets = []

    # Inspect model structure to find the correct paths
    for name, module in model.named_modules():
        if action_head_only and "action_head" not in name:
            continue

        if target_prefixes is not None:
            # If target_prefixes is provided, use it to filter modules (substring matching)
            if any(prefix in name for prefix in target_prefixes):
                if isinstance(module, (torch.nn.Linear)):
                    standard_targets.append(name)
                elif isinstance(module, CategorySpecificLinear):
                    custom_targets.append(name)
        else:
            # Look for linear layers in attention mechanisms
            if isinstance(module, torch.nn.Linear):
                if any(x in name for x in ["q_proj", "v_proj", "to_q", "to_v", "k_proj", "to_k"]):
                    standard_targets.append(name)
    # Apply Standard PEFT LoRA
    if standard_targets:
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=standard_targets,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Apply Custom LoRA to CategorySpecificLinear layers
    for name in custom_targets:
        parts = name.split(".")
        child_name = parts[-1]
        parent_name = ".".join(parts[:-1])
        try:
            parent = model.get_submodule(parent_name)
            original_module = getattr(parent, child_name)
            
            # Wrap it
            new_module = LoraCategorySpecificLinear(
                original_module, r=rank, alpha=lora_alpha, dropout=lora_dropout
            )
            setattr(parent, child_name, new_module)
            print(f"Applied Custom LoRA to: {name}")
        except Exception as e:
            print(f"Failed to apply Custom LoRA to {name}: {e}")

    model.print_trainable_parameters()

    model = _wrap_forward(model)

    return model
