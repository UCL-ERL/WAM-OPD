"""Isolated zero-init residual adapter for the released video output head.

The adapter wraps only ``WanTransformer3DModel.proj_out``.  The action output
head, transformer blocks, text conditioning and cache implementation remain
frozen.  With ``zero_up`` initialization the attached module is bitwise
equivalent to the released projection before the first optimizer step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class VideoOutputAdapterInfo:
    rank: int
    input_features: int
    output_features: int
    trainable_parameter_count: int
    initialization: str


class VideoOutputResidualAdapter(nn.Module):
    """Frozen video projection plus a low-rank residual correction."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        initialization: str = "zero_up",
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(
                "video output adapter requires torch.nn.Linear, got "
                f"{type(base).__name__}"
            )
        if int(rank) <= 0:
            raise ValueError("rank must be positive")
        if initialization not in ("zero_up", "gated_random"):
            raise ValueError(f"unsupported initialization: {initialization!r}")
        self.rank = int(rank)
        self.initialization = initialization
        self.base = base.requires_grad_(False)
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.down = nn.Linear(base.in_features, self.rank, bias=False, **factory_kwargs)
        self.up = nn.Linear(self.rank, base.out_features, bias=False, **factory_kwargs)
        if initialization == "zero_up":
            nn.init.zeros_(self.up.weight)
            self.gate = nn.Parameter(
                torch.ones(base.out_features, **factory_kwargs), requires_grad=False
            )
        else:
            self.gate = nn.Parameter(torch.zeros(base.out_features, **factory_kwargs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        residual = self.up(self.down(inputs))
        return base_output + residual * self.gate


def attach_video_output_adapter(
    transformer: nn.Module,
    *,
    rank: int = 8,
    initialization: str = "zero_up",
) -> VideoOutputAdapterInfo:
    """Freeze the transformer and attach exactly one video-local adapter."""

    projection = getattr(transformer, "proj_out", None)
    if isinstance(projection, VideoOutputResidualAdapter):
        raise ValueError("video output adapter is already attached")
    if not isinstance(projection, nn.Linear):
        raise TypeError(
            "transformer.proj_out must be torch.nn.Linear before attachment, "
            f"got {type(projection).__name__}"
        )
    transformer.requires_grad_(False)
    adapter = VideoOutputResidualAdapter(
        projection, rank=int(rank), initialization=initialization
    )
    transformer.proj_out = adapter
    trainable_count = sum(
        parameter.numel()
        for parameter in adapter.parameters()
        if parameter.requires_grad
    )
    return VideoOutputAdapterInfo(
        rank=int(rank),
        input_features=projection.in_features,
        output_features=projection.out_features,
        trainable_parameter_count=trainable_count,
        initialization=initialization,
    )


def video_output_adapter_state_dict(transformer: nn.Module) -> dict[str, torch.Tensor]:
    adapter = getattr(transformer, "proj_out", None)
    if not isinstance(adapter, VideoOutputResidualAdapter):
        raise ValueError("video output adapter is not attached")
    return {
        f"proj_out.{name}": parameter.detach().cpu()
        for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    }


__all__ = [
    "VideoOutputAdapterInfo",
    "VideoOutputResidualAdapter",
    "attach_video_output_adapter",
    "video_output_adapter_state_dict",
]
