"""Small native LoRA wrapper with an explicit video-mode execution gate.

This module intentionally does not depend on PEFT.  The released LingBot
transformer is a native Wan module with two transformer modes sharing blocks.
The wrapper keeps the original ``nn.Linear`` frozen and adds a low-rank
residual only while :func:`video_mode_lora_scope` is active.  Action-mode
calls therefore take the exact frozen base path.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

from experiments.goal1_exact_condition import tensor_hash


_VIDEO_MODE_ACTIVE: ContextVar[bool] = ContextVar(
    "waopd_video_mode_lora_active", default=False
)


@contextmanager
def video_mode_lora_scope(active: bool) -> Iterator[None]:
    """Enable/disable all attached video-mode LoRA residuals in this scope."""

    token = _VIDEO_MODE_ACTIVE.set(bool(active))
    try:
        yield
    finally:
        _VIDEO_MODE_ACTIVE.reset(token)


def video_mode_lora_active() -> bool:
    return bool(_VIDEO_MODE_ACTIVE.get())


class VideoModeLoRALinear(nn.Module):
    """Frozen linear plus a zero-up-initialized, gated LoRA residual."""

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
            raise TypeError(f"VideoModeLoRALinear requires Linear, got {type(base).__name__}")
        if int(rank) <= 0:
            raise ValueError("LoRA rank must be positive")
        if float(dropout) != 0.0:
            raise ValueError("V0H fixes LoRA dropout to zero")
        self.base = base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.scaling = self.alpha / float(self.rank)
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, base.in_features, **factory_kwargs)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, self.rank, **factory_kwargs)
        )
        self.forward_call_count = 0
        self.active_call_count = 0
        self.bypass_call_count = 0
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_call_count += 1
        base_output = self.base(inputs)
        if not video_mode_lora_active():
            self.bypass_call_count += 1
            return base_output
        self.active_call_count += 1
        residual = F.linear(F.linear(inputs, self.lora_A), self.lora_B)
        return base_output + residual * self.scaling


@dataclass(frozen=True)
class VideoModeLoRAModule:
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
class VideoModeLoRAInfo:
    rank: int
    alpha: float
    dropout: float
    block_indices: tuple[int, ...]
    modules: tuple[VideoModeLoRAModule, ...]
    trainable_parameter_count: int
    base_parameter_hashes: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "block_indices": list(self.block_indices),
            "trainable_parameter_count": self.trainable_parameter_count,
            "modules": [item.to_dict() for item in self.modules],
            "base_parameter_hashes": dict(self.base_parameter_hashes),
        }


def _get_child(module: nn.Module, part: str) -> nn.Module:
    if part.isdigit():
        return module[int(part)]  # type: ignore[index]
    return getattr(module, part)


def _set_child(module: nn.Module, part: str, value: nn.Module) -> None:
    if part.isdigit():
        module[int(part)] = value  # type: ignore[index]
    else:
        setattr(module, part, value)


def _resolve_parent(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str, nn.Module]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = _get_child(parent, part)
    leaf = parts[-1]
    return parent, leaf, _get_child(parent, leaf)


def resolved_video_lora_targets(block_indices: tuple[int, ...]) -> list[tuple[int, str, str]]:
    """Return the resolved semantic target list for this Wan source tree."""

    targets: list[tuple[int, str, str]] = []
    for block_index in block_indices:
        for attention_name in ("attn1", "attn2"):
            for suffix, projection_type in (
                ("to_q", "attention_q"),
                ("to_k", "attention_k"),
                ("to_v", "attention_v"),
                ("to_out.0", "attention_o"),
            ):
                targets.append(
                    (
                        int(block_index),
                        f"blocks.{block_index}.{attention_name}.{suffix}",
                        projection_type,
                    )
                )
        targets.extend(
            [
                (
                    int(block_index),
                    f"blocks.{block_index}.ffn.net.0.proj",
                    "mlp_up",
                ),
                (
                    int(block_index),
                    f"blocks.{block_index}.ffn.net.2",
                    "mlp_down",
                ),
            ]
        )
    return targets


def attach_video_mode_lora(
    transformer: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    block_indices: tuple[int, ...] = (26, 27, 28, 29),
) -> VideoModeLoRAInfo:
    """Freeze the transformer and attach exactly the V0H target modules."""

    if any(isinstance(module, VideoModeLoRALinear) for module in transformer.modules()):
        raise ValueError("video-mode LoRA is already attached")
    transformer.requires_grad_(False)
    modules: list[VideoModeLoRAModule] = []
    base_hashes: dict[str, str] = {}
    for block_index, name, projection_type in resolved_video_lora_targets(block_indices):
        parent, leaf, base = _resolve_parent(transformer, name)
        if not isinstance(base, nn.Linear):
            raise TypeError(
                f"V0H target {name} is not Linear: {type(base).__name__}"
            )
        base_weight_hash = str(tensor_hash(base.weight))
        base_bias_hash = str(tensor_hash(base.bias)) if base.bias is not None else None
        wrapper = VideoModeLoRALinear(
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
            VideoModeLoRAModule(
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
    trainable_count = sum(item.trainable_parameter_count for item in modules)
    return VideoModeLoRAInfo(
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
        block_indices=tuple(int(item) for item in block_indices),
        modules=tuple(modules),
        trainable_parameter_count=int(trainable_count),
        base_parameter_hashes=base_hashes,
    )


def video_mode_lora_state_dict(transformer: nn.Module) -> dict[str, torch.Tensor]:
    """Return only trainable LoRA tensors, suitable for a compact checkpoint."""

    state: dict[str, torch.Tensor] = {}
    for name, parameter in transformer.named_parameters():
        if parameter.requires_grad and (name.endswith(".lora_A") or name.endswith(".lora_B")):
            state[name] = parameter.detach().cpu()
    if not state:
        raise ValueError("no video-mode LoRA parameters are attached")
    return state


def video_mode_lora_trainable_names(transformer: nn.Module) -> list[str]:
    return [
        name
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad and (name.endswith(".lora_A") or name.endswith(".lora_B"))
    ]


def video_mode_lora_base_parameter_hashes(transformer: nn.Module) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, module in transformer.named_modules():
        if isinstance(module, VideoModeLoRALinear):
            prefix = name
            hashes[f"{prefix}.weight"] = str(tensor_hash(module.base.weight))
            if module.base.bias is not None:
                hashes[f"{prefix}.bias"] = str(tensor_hash(module.base.bias))
    return hashes


def video_mode_lora_gate_counts(transformer: nn.Module) -> dict[str, int]:
    total = active = bypass = 0
    for module in transformer.modules():
        if isinstance(module, VideoModeLoRALinear):
            total += int(getattr(module, "forward_call_count", 0))
            active += int(getattr(module, "active_call_count", 0))
            bypass += int(getattr(module, "bypass_call_count", 0))
    return {
        "total_forward_calls": total,
        "active_forward_calls": active,
        "bypass_forward_calls": bypass,
    }


__all__ = [
    "VideoModeLoRAInfo",
    "VideoModeLoRALinear",
    "attach_video_mode_lora",
    "resolved_video_lora_targets",
    "video_mode_lora_active",
    "video_mode_lora_base_parameter_hashes",
    "video_mode_lora_gate_counts",
    "video_mode_lora_scope",
    "video_mode_lora_state_dict",
    "video_mode_lora_trainable_names",
]
