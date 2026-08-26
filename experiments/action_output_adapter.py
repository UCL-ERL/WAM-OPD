"""Shared low-rank residual adapter for Flash-WAM action endpoints.

The released transformer remains frozen.  The adapter wraps only
``action_proj_out``, which is not used by the video output path.  Its learned
per-channel gate starts at exactly zero, so attaching the module preserves the
released Student output before the first optimizer step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ActionOutputAdapterInfo:
    rank: int
    input_features: int
    output_features: int
    trainable_parameter_count: int
    initialization: str = "gated_random"


class ActionOutputResidualAdapter(nn.Module):
    """Frozen linear projection plus a gated low-rank residual branch."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        initialization: str = "gated_random",
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not isinstance(base, nn.Linear):
            raise TypeError(
                "action output adapter requires torch.nn.Linear, got "
                f"{type(base).__name__}"
            )

        self.rank = int(rank)
        if initialization not in ("gated_random", "zero_up"):
            raise ValueError(
                "initialization must be gated_random or zero_up, got "
                f"{initialization!r}"
            )
        self.initialization = initialization
        self.base = base.requires_grad_(False)
        factory_kwargs = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }
        self.down = nn.Linear(
            base.in_features,
            self.rank,
            bias=False,
            **factory_kwargs,
        )
        self.up = nn.Linear(
            self.rank,
            base.out_features,
            bias=False,
            **factory_kwargs,
        )
        if initialization == "gated_random":
            self.gate = nn.Parameter(
                torch.zeros(base.out_features, **factory_kwargs)
            )
        else:
            nn.init.zeros_(self.up.weight)
            self.gate = nn.Parameter(
                torch.ones(base.out_features, **factory_kwargs),
                requires_grad=False,
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        residual = self.up(self.down(inputs))
        return base_output + residual * self.gate


def attach_action_output_adapter(
    transformer: nn.Module,
    *,
    rank: int,
    initialization: str = "gated_random",
) -> ActionOutputAdapterInfo:
    """Freeze ``transformer`` and attach one shared action-output adapter."""

    projection = getattr(transformer, "action_proj_out", None)
    if isinstance(projection, ActionOutputResidualAdapter):
        raise ValueError("action output adapter is already attached")
    if not isinstance(projection, nn.Linear):
        raise TypeError(
            "transformer.action_proj_out must be torch.nn.Linear before "
            f"attachment, got {type(projection).__name__}"
        )

    transformer.requires_grad_(False)
    adapter = ActionOutputResidualAdapter(
        projection,
        rank=rank,
        initialization=initialization,
    )
    transformer.action_proj_out = adapter
    trainable_count = sum(
        parameter.numel()
        for parameter in adapter.parameters()
        if parameter.requires_grad
    )
    return ActionOutputAdapterInfo(
        rank=int(rank),
        input_features=projection.in_features,
        output_features=projection.out_features,
        trainable_parameter_count=trainable_count,
        initialization=initialization,
    )


def action_output_adapter_state_dict(
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    """Return only trainable adapter tensors with transformer-qualified keys."""

    adapter = getattr(transformer, "action_proj_out", None)
    if not isinstance(adapter, ActionOutputResidualAdapter):
        raise ValueError("action output adapter is not attached")
    return {
        f"action_proj_out.{name}": parameter.detach().cpu()
        for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    }
