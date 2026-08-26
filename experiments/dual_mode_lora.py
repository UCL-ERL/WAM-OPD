"""Two full-capacity LoRA banks with explicit video/action execution gates.

The released Flash-WAM transformer reuses the same Wan blocks for video and
action calls.  A single always-on LoRA therefore couples two different
execution modes.  This module keeps two independent LoRA banks on every
selected ``nn.Linear`` and activates exactly one bank per scoped Student call.

Both banks use the same target modules and rank.  Consequently, each bank has
the same parameter capacity as the corresponding JointLoRA; mode isolation
does not reduce the capacity of either stage.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
from typing import Iterator, Literal, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from experiments.goal1_exact_condition import tensor_hash
from experiments.video_mode_lora import (
    _resolve_parent,
    _set_child,
    resolved_video_lora_targets,
)


AdapterMode = Literal["none", "video", "action"]
AdapterBank = Literal["video", "action", "both"]
DUAL_MODE_LORA_CONTRACT_SCHEMA = "waopd_dual_mode_lora_contract_v1"

_VALID_MODES = frozenset({"none", "video", "action"})
_VALID_BANKS = frozenset({"video", "action", "both"})
_DUAL_MODE: ContextVar[AdapterMode] = ContextVar(
    "waopd_dual_mode_lora_mode", default="none"
)


def _checked_mode(mode: str) -> AdapterMode:
    normalized = str(mode)
    if normalized not in _VALID_MODES:
        raise ValueError(f"unsupported dual-mode LoRA mode: {mode!r}")
    return normalized  # type: ignore[return-value]


def _checked_bank(bank: str) -> AdapterBank:
    normalized = str(bank)
    if normalized not in _VALID_BANKS:
        raise ValueError(f"unsupported dual-mode LoRA bank: {bank!r}")
    return normalized  # type: ignore[return-value]


@contextmanager
def dual_mode_lora_scope(mode: AdapterMode | str) -> Iterator[None]:
    """Activate exactly one LoRA bank, restoring the prior mode on exit."""

    token = _DUAL_MODE.set(_checked_mode(str(mode)))
    try:
        yield
    finally:
        _DUAL_MODE.reset(token)


def dual_mode_lora_active_mode() -> AdapterMode:
    return _DUAL_MODE.get()


class DualModeLoRALinear(nn.Module):
    """Frozen Linear plus independent, zero-up video and action residuals."""

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
            raise TypeError(
                f"DualModeLoRALinear requires Linear, got {type(base).__name__}"
            )
        if int(rank) <= 0:
            raise ValueError("dual-mode LoRA rank must be positive")
        if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
            raise ValueError("dual-mode LoRA alpha must be finite and positive")
        if float(dropout) != 0.0:
            raise ValueError("dual-mode LoRA fixes dropout to zero")

        self.base = base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.scaling = self.alpha / float(self.rank)
        # Keep trainable/master weights in FP32.  The frozen Wan base remains
        # BF16; only the low-rank residual is computed in FP32 and cast back.
        # This avoids BF16 parameter quantization and BF16 Adam moment state.
        factory_kwargs = {"device": base.weight.device, "dtype": torch.float32}

        self.video_lora_A = nn.Parameter(
            torch.empty(self.rank, base.in_features, **factory_kwargs)
        )
        self.video_lora_B = nn.Parameter(
            torch.zeros(base.out_features, self.rank, **factory_kwargs)
        )
        self.action_lora_A = nn.Parameter(
            torch.empty(self.rank, base.in_features, **factory_kwargs)
        )
        self.action_lora_B = nn.Parameter(
            torch.zeros(base.out_features, self.rank, **factory_kwargs)
        )
        nn.init.kaiming_uniform_(self.video_lora_A, a=5**0.5)
        nn.init.kaiming_uniform_(self.action_lora_A, a=5**0.5)

        self.forward_call_count = 0
        self.video_call_count = 0
        self.action_call_count = 0
        self.bypass_call_count = 0

    def _residual(
        self,
        inputs: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
    ) -> torch.Tensor:
        residual = F.linear(F.linear(inputs.to(dtype=lora_A.dtype), lora_A), lora_B)
        return (residual * self.scaling).to(dtype=inputs.dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_call_count += 1
        base_output = self.base(inputs)
        mode = dual_mode_lora_active_mode()
        if mode == "video":
            self.video_call_count += 1
            return base_output + self._residual(
                inputs, self.video_lora_A, self.video_lora_B
            )
        if mode == "action":
            self.action_call_count += 1
            return base_output + self._residual(
                inputs, self.action_lora_A, self.action_lora_B
            )
        self.bypass_call_count += 1
        return base_output


@dataclass(frozen=True)
class DualModeLoRAModule:
    name: str
    block_index: int
    projection_type: str
    input_features: int
    output_features: int
    rank: int
    alpha: float
    dropout: float
    parameter_dtype: str
    trainable_parameter_count_per_bank: int
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
            "parameter_dtype": self.parameter_dtype,
            "trainable_parameter_count_per_bank": (
                self.trainable_parameter_count_per_bank
            ),
            "trainable_parameter_count": 2
            * self.trainable_parameter_count_per_bank,
            "base_weight_hash": self.base_weight_hash,
            "base_bias_hash": self.base_bias_hash,
        }


@dataclass(frozen=True)
class DualModeLoRAInfo:
    rank: int
    alpha: float
    dropout: float
    parameter_dtype: str
    block_indices: tuple[int, ...]
    modules: tuple[DualModeLoRAModule, ...]
    trainable_parameter_count_per_bank: int
    trainable_parameter_count: int
    base_parameter_hashes: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "parameter_dtype": self.parameter_dtype,
            "block_indices": list(self.block_indices),
            "trainable_parameter_count_per_bank": (
                self.trainable_parameter_count_per_bank
            ),
            "trainable_parameter_count": self.trainable_parameter_count,
            "modules": [item.to_dict() for item in self.modules],
            "base_parameter_hashes": dict(self.base_parameter_hashes),
        }


def attach_dual_mode_lora(
    transformer: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    block_indices: tuple[int, ...] = (26, 27, 28, 29),
) -> DualModeLoRAInfo:
    """Freeze the transformer and attach both banks to the V0J targets."""

    if any(isinstance(module, DualModeLoRALinear) for module in transformer.modules()):
        raise ValueError("dual-mode LoRA is already attached")
    normalized_blocks = tuple(int(item) for item in block_indices)
    if not normalized_blocks:
        raise ValueError("dual-mode LoRA block_indices must not be empty")
    if len(set(normalized_blocks)) != len(normalized_blocks):
        raise ValueError("dual-mode LoRA block_indices must be unique")
    if any(item < 0 for item in normalized_blocks):
        raise ValueError("dual-mode LoRA block indices must be nonnegative")
    if int(rank) <= 0:
        raise ValueError("dual-mode LoRA rank must be positive")
    if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
        raise ValueError("dual-mode LoRA alpha must be finite and positive")
    if float(dropout) != 0.0:
        raise ValueError("dual-mode LoRA fixes dropout to zero")

    targets = resolved_video_lora_targets(normalized_blocks)
    target_names = [name for _block, name, _projection in targets]
    if not targets or len(set(target_names)) != len(target_names):
        raise ValueError("dual-mode LoRA resolved an empty or duplicate target list")

    # Resolve and validate every target before freezing or replacing anything.
    resolved: list[
        tuple[int, str, str, nn.Module, str, nn.Linear, str, str | None]
    ] = []
    for block_index, name, projection_type in targets:
        parent, leaf, base = _resolve_parent(transformer, name)
        if not isinstance(base, nn.Linear):
            raise TypeError(
                f"dual-mode LoRA target {name} is not Linear: {type(base).__name__}"
            )
        resolved.append(
            (
                int(block_index),
                name,
                projection_type,
                parent,
                leaf,
                base,
                str(tensor_hash(base.weight)),
                str(tensor_hash(base.bias)) if base.bias is not None else None,
            )
        )

    original_requires_grad = [
        (parameter, bool(parameter.requires_grad))
        for parameter in transformer.parameters()
    ]
    replaced: list[tuple[nn.Module, str, nn.Linear]] = []
    modules: list[DualModeLoRAModule] = []
    base_hashes: dict[str, str] = {}
    try:
        transformer.requires_grad_(False)
        for (
            block_index,
            name,
            projection_type,
            parent,
            leaf,
            base,
            base_weight_hash,
            base_bias_hash,
        ) in resolved:
            wrapper = DualModeLoRALinear(
                base,
                rank=int(rank),
                alpha=float(alpha),
                dropout=float(dropout),
            )
            _set_child(parent, leaf, wrapper)
            replaced.append((parent, leaf, base))
            base_hashes[f"{name}.weight"] = base_weight_hash
            if base_bias_hash is not None:
                base_hashes[f"{name}.bias"] = base_bias_hash
            modules.append(
                DualModeLoRAModule(
                    name=name,
                    block_index=int(block_index),
                    projection_type=projection_type,
                    input_features=int(base.in_features),
                    output_features=int(base.out_features),
                    rank=int(rank),
                    alpha=float(alpha),
                    dropout=float(dropout),
                    parameter_dtype=str(wrapper.video_lora_A.dtype),
                    trainable_parameter_count_per_bank=int(rank)
                    * (int(base.in_features) + int(base.out_features)),
                    base_weight_hash=base_weight_hash,
                    base_bias_hash=base_bias_hash,
                )
            )
    except BaseException:
        for parent, leaf, base in reversed(replaced):
            _set_child(parent, leaf, base)
        for parameter, requires_grad in original_requires_grad:
            parameter.requires_grad_(requires_grad)
        raise

    per_bank = sum(item.trainable_parameter_count_per_bank for item in modules)
    return DualModeLoRAInfo(
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
        parameter_dtype="torch.float32",
        block_indices=normalized_blocks,
        modules=tuple(modules),
        trainable_parameter_count_per_bank=int(per_bank),
        trainable_parameter_count=int(2 * per_bank),
        base_parameter_hashes=base_hashes,
    )


def _bank_suffixes(bank: AdapterBank | str) -> tuple[str, ...]:
    selected = _checked_bank(str(bank))
    video = (".video_lora_A", ".video_lora_B")
    action = (".action_lora_A", ".action_lora_B")
    if selected == "video":
        return video
    if selected == "action":
        return action
    return video + action


def dual_mode_lora_named_parameters(
    transformer: nn.Module, *, bank: AdapterBank | str = "both"
) -> list[tuple[str, nn.Parameter]]:
    suffixes = _bank_suffixes(bank)
    parameters = [
        (name, parameter)
        for name, parameter in transformer.named_parameters()
        if name.endswith(suffixes)
    ]
    if not parameters:
        raise ValueError(f"no dual-mode LoRA parameters found for bank={bank!r}")
    return parameters


def select_dual_mode_lora_trainable_bank(
    transformer: nn.Module, *, bank: AdapterBank | str
) -> list[tuple[str, nn.Parameter]]:
    """Enable gradients for exactly the selected bank(s) and return them."""

    selected_bank = _checked_bank(str(bank))
    all_parameters = dual_mode_lora_named_parameters(transformer, bank="both")
    selected_suffixes = _bank_suffixes(selected_bank)
    selected: list[tuple[str, nn.Parameter]] = []
    for name, parameter in all_parameters:
        enabled = name.endswith(selected_suffixes)
        parameter.requires_grad_(enabled)
        parameter.grad = None
        if enabled:
            selected.append((name, parameter))
    if not selected:
        raise ValueError(f"dual-mode LoRA selected no parameters for bank={bank!r}")
    return selected


def dual_mode_lora_state_dict(
    transformer: nn.Module, *, bank: AdapterBank | str = "both"
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in dual_mode_lora_named_parameters(
            transformer, bank=bank
        )
    }


def load_dual_mode_lora_state_dict(
    transformer: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    bank: AdapterBank | str = "both",
) -> None:
    """Atomically load selected banks after validating and converting every tensor."""

    expected = dict(dual_mode_lora_named_parameters(transformer, bank=bank))
    provided_names = set(state)
    expected_names = set(expected)
    missing = sorted(expected_names - provided_names)
    unexpected = sorted(provided_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            "dual-mode LoRA checkpoint key mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    converted: dict[str, torch.Tensor] = {}
    for name, parameter in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"dual-mode LoRA tensor {name} is not a Tensor")
        if not torch.is_floating_point(value):
            raise TypeError(f"dual-mode LoRA tensor {name} is not floating point")
        if tuple(value.shape) != tuple(parameter.shape):
            raise ValueError(
                f"dual-mode LoRA shape mismatch for {name}: "
                f"checkpoint={tuple(value.shape)}, expected={tuple(parameter.shape)}"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"dual-mode LoRA tensor {name} is nonfinite")
        candidate = value.to(
            device=parameter.device, dtype=parameter.dtype
        ).detach().clone()
        if not bool(torch.isfinite(candidate).all().item()):
            raise ValueError(
                f"dual-mode LoRA tensor {name} became nonfinite after dtype conversion"
            )
        converted[name] = candidate

    original = {
        name: parameter.detach().clone() for name, parameter in expected.items()
    }
    try:
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(converted[name])
    except BaseException:
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(original[name])
        raise


def dual_mode_lora_contract(info: DualModeLoRAInfo) -> dict[str, object]:
    """Build the identity contract that must accompany every dual-bank checkpoint."""

    return {
        "schema": DUAL_MODE_LORA_CONTRACT_SCHEMA,
        "adapter_kind": "dual_lora",
        "rank": int(info.rank),
        "alpha": float(info.alpha),
        "dropout": float(info.dropout),
        "parameter_dtype": str(info.parameter_dtype),
        "block_indices": list(info.block_indices),
        "target_names": [item.name for item in info.modules],
        "trainable_parameter_count_per_bank": int(
            info.trainable_parameter_count_per_bank
        ),
        "base_parameter_hashes": dict(info.base_parameter_hashes),
    }


def validate_dual_mode_lora_contract(
    info: DualModeLoRAInfo, contract: Mapping[str, object]
) -> None:
    expected = dual_mode_lora_contract(info)
    mismatches = {
        key: {"checkpoint": contract.get(key), "expected": value}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    unexpected = sorted(set(contract) - set(expected))
    if mismatches or unexpected:
        raise ValueError(
            "dual-mode LoRA checkpoint contract mismatch: "
            f"mismatches={mismatches}, unexpected={unexpected}"
        )


def load_dual_mode_lora_checkpoint(
    transformer: nn.Module,
    info: DualModeLoRAInfo,
    payload: Mapping[str, object],
) -> None:
    """Validate checkpoint identity and atomically load both adapter banks."""

    if payload.get("adapter_kind") != "dual_lora":
        raise ValueError(
            "dual-mode LoRA runtime received checkpoint with "
            f"adapter_kind={payload.get('adapter_kind')!r}"
        )
    contract = payload.get("adapter_contract")
    if not isinstance(contract, Mapping):
        raise TypeError("dual-mode LoRA checkpoint lacks adapter_contract")
    validate_dual_mode_lora_contract(info, contract)
    state = payload.get("adapter_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("dual-mode LoRA checkpoint lacks adapter_state_dict")
    load_dual_mode_lora_state_dict(transformer, state, bank="both")  # type: ignore[arg-type]


def dual_mode_lora_base_parameter_hashes(
    transformer: nn.Module,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, module in transformer.named_modules():
        if isinstance(module, DualModeLoRALinear):
            hashes[f"{name}.weight"] = str(tensor_hash(module.base.weight))
            if module.base.bias is not None:
                hashes[f"{name}.bias"] = str(tensor_hash(module.base.bias))
    return hashes


def dual_mode_lora_gate_counts(transformer: nn.Module) -> dict[str, int]:
    total = video = action = bypass = 0
    for module in transformer.modules():
        if isinstance(module, DualModeLoRALinear):
            total += int(module.forward_call_count)
            video += int(module.video_call_count)
            action += int(module.action_call_count)
            bypass += int(module.bypass_call_count)
    return {
        "total_forward_calls": total,
        "video_forward_calls": video,
        "action_forward_calls": action,
        "bypass_forward_calls": bypass,
    }


__all__ = [
    "AdapterBank",
    "AdapterMode",
    "DUAL_MODE_LORA_CONTRACT_SCHEMA",
    "DualModeLoRAInfo",
    "DualModeLoRALinear",
    "attach_dual_mode_lora",
    "dual_mode_lora_active_mode",
    "dual_mode_lora_base_parameter_hashes",
    "dual_mode_lora_contract",
    "dual_mode_lora_gate_counts",
    "dual_mode_lora_named_parameters",
    "dual_mode_lora_scope",
    "dual_mode_lora_state_dict",
    "load_dual_mode_lora_state_dict",
    "load_dual_mode_lora_checkpoint",
    "select_dual_mode_lora_trainable_bank",
    "validate_dual_mode_lora_contract",
]
