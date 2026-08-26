"""Shared video/action LoRA for exact-condition FlashWAM adaptation.

The released Flash-WAM transformer has one set of transformer blocks for
video and action tokens.  V0H deliberately used a video-only execution gate;
this module is a separate, always-on wrapper for V0J so the old V0H path and
its artifacts remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from experiments.goal1_exact_condition import tensor_hash
from experiments.video_mode_lora import (
    _resolve_parent,
    _set_child,
    resolved_video_lora_targets,
)


class JointLoRALinear(nn.Module):
    """Frozen linear plus a zero-up-initialized residual on every call."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"JointLoRALinear requires Linear, got {type(base).__name__}")
        if int(rank) <= 0:
            raise ValueError("Joint LoRA rank must be positive")
        if float(dropout) != 0.0:
            raise ValueError("V0J fixes Joint LoRA dropout to zero")
        self.base = base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.scaling = self.alpha / float(self.rank)
        # Keep the adapter master parameters and their optimizer state in FP32.
        # The released Transformer is BF16, so the residual branch explicitly
        # casts its input and returns to the base output dtype below.
        factory_kwargs = {"device": base.weight.device, "dtype": torch.float32}
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, **factory_kwargs))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, **factory_kwargs))
        self.forward_call_count = 0
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_call_count += 1
        base_output = self.base(inputs)
        residual = F.linear(F.linear(inputs.float(), self.lora_A), self.lora_B)
        return base_output + (residual * self.scaling).to(dtype=base_output.dtype)


@dataclass(frozen=True)
class JointLoRAModule:
    name: str
    block_index: int
    projection_type: str
    input_features: int
    output_features: int
    rank: int
    alpha: float
    dropout: float
    trainable_parameter_count: int
    base_weight_hash: str
    base_bias_hash: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "block_index": self.block_index,
            "projection_type": self.projection_type,
            "input_features": self.input_features,
            "output_features": self.output_features,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "trainable_parameter_count": self.trainable_parameter_count,
            "base_weight_hash": self.base_weight_hash,
            "base_bias_hash": self.base_bias_hash,
        }


@dataclass(frozen=True)
class JointLoRAInfo:
    rank: int
    alpha: float
    dropout: float
    block_indices: tuple[int, ...]
    modules: tuple[JointLoRAModule, ...]
    trainable_parameter_count: int
    base_parameter_hashes: dict[str, str]

    @property
    def parameter_dtype(self) -> str:
        return "torch.float32"

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "block_indices": list(self.block_indices),
            "trainable_parameter_count": self.trainable_parameter_count,
            "parameter_dtype": self.parameter_dtype,
            "modules": [item.to_dict() for item in self.modules],
            "base_parameter_hashes": dict(self.base_parameter_hashes),
        }


def attach_joint_lora(
    transformer: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    block_indices: tuple[int, ...] = (26, 27, 28, 29),
) -> JointLoRAInfo:
    """Attach the requested shared-block targets and freeze the base."""

    if any(isinstance(module, (JointLoRALinear,)) for module in transformer.modules()):
        raise ValueError("Joint LoRA is already attached")
    transformer.requires_grad_(False)
    modules: list[JointLoRAModule] = []
    base_hashes: dict[str, str] = {}
    for block_index, name, projection_type in resolved_video_lora_targets(block_indices):
        parent, leaf, base = _resolve_parent(transformer, name)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"V0J target {name} is not Linear: {type(base).__name__}")
        base_weight_hash = str(tensor_hash(base.weight))
        base_bias_hash = str(tensor_hash(base.bias)) if base.bias is not None else None
        wrapper = JointLoRALinear(
            base,
            rank=int(rank),
            alpha=float(alpha),
            dropout=float(dropout),
        )
        _set_child(parent, leaf, wrapper)
        base_hashes[f"{name}.weight"] = base_weight_hash
        if base_bias_hash is not None:
            base_hashes[f"{name}.bias"] = base_bias_hash
        modules.append(
            JointLoRAModule(
                name=name,
                block_index=int(block_index),
                projection_type=projection_type,
                input_features=int(base.in_features),
                output_features=int(base.out_features),
                rank=int(rank),
                alpha=float(alpha),
                dropout=float(dropout),
                trainable_parameter_count=int(rank) * (base.in_features + base.out_features),
                base_weight_hash=base_weight_hash,
                base_bias_hash=base_bias_hash,
            )
        )
    return JointLoRAInfo(
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
        block_indices=tuple(int(item) for item in block_indices),
        modules=tuple(modules),
        trainable_parameter_count=sum(item.trainable_parameter_count for item in modules),
        base_parameter_hashes=base_hashes,
    )


def joint_lora_state_dict(transformer: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: parameter.detach().cpu()
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad and (name.endswith(".lora_A") or name.endswith(".lora_B"))
    }
    if not state:
        raise ValueError("no Joint LoRA parameters are attached")
    return state


def joint_lora_trainable_names(transformer: nn.Module) -> list[str]:
    return [
        name
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad and (name.endswith(".lora_A") or name.endswith(".lora_B"))
    ]


def joint_lora_base_parameter_hashes(transformer: nn.Module) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, module in transformer.named_modules():
        if isinstance(module, JointLoRALinear):
            hashes[f"{name}.weight"] = str(tensor_hash(module.base.weight))
            if module.base.bias is not None:
                hashes[f"{name}.bias"] = str(tensor_hash(module.base.bias))
    return hashes


def joint_lora_call_counts(transformer: nn.Module) -> dict[str, int]:
    total = 0
    for module in transformer.modules():
        if isinstance(module, JointLoRALinear):
            total += int(module.forward_call_count)
    return {"total_forward_calls": total}


__all__ = [
    "JointLoRAInfo",
    "JointLoRALinear",
    "attach_joint_lora",
    "joint_lora_base_parameter_hashes",
    "joint_lora_call_counts",
    "joint_lora_state_dict",
    "joint_lora_trainable_names",
]
