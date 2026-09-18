"""Opt-in synchronous data-parallel helpers for success-path pilots.

The formal trainer remains single-process by default.  This module is only
activated by an explicitly constructed context after ``torch.distributed``
has been initialized.  Gradients are summed (not averaged) because the
single-GPU success-path loop accumulates one weighted loss per sample before
each optimizer step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class SuccessPathDistributedContext:
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("distributed world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("distributed rank is outside world_size")
        if self.world_size > 1:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "multi-rank success-path context requires an initialized "
                    "torch.distributed process group"
                )
            if dist.get_rank() != self.rank or dist.get_world_size() != self.world_size:
                raise RuntimeError("distributed context does not match process group")

    @classmethod
    def from_initialized_process_group(cls) -> "SuccessPathDistributedContext":
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed process group is not initialized")
        return cls(rank=int(dist.get_rank()), world_size=int(dist.get_world_size()))

    def local_indices(self, global_batch_size: int) -> tuple[int, ...]:
        if global_batch_size <= 0:
            raise ValueError("global batch size must be positive")
        return tuple(range(self.rank, int(global_batch_size), self.world_size))

    def sum_gradients(self, parameters: Sequence[torch.nn.Parameter]) -> None:
        if self.world_size == 1:
            return
        if not parameters:
            raise ValueError("distributed gradient synchronization needs parameters")
        device = parameters[0].device
        local_presence = torch.tensor(
            [parameter.grad is not None for parameter in parameters],
            device=device,
            dtype=torch.int32,
        )
        dist.all_reduce(local_presence, op=dist.ReduceOp.MAX)
        for index, parameter in enumerate(parameters):
            if int(local_presence[index].item()) == 0:
                parameter.grad = None
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)

    def sum_scalars(
        self,
        values: Sequence[float],
        *,
        device: torch.device,
    ) -> tuple[float, ...]:
        tensor = torch.tensor(list(values), device=device, dtype=torch.float64)
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tuple(float(value) for value in tensor.cpu().tolist())

    def gather_indexed_rows(
        self,
        rows: Sequence[tuple[int, Mapping[str, Any]]],
    ) -> list[dict[str, Any]]:
        local_rows = [(int(index), dict(row)) for index, row in rows]
        if self.world_size == 1:
            gathered = [local_rows]
        else:
            gathered: list[list[tuple[int, dict[str, Any]]] | None] = [
                None for _ in range(self.world_size)
            ]
            dist.all_gather_object(gathered, local_rows)
        flattened = [item for rank_rows in gathered if rank_rows for item in rank_rows]
        indices = [index for index, _row in flattened]
        if sorted(indices) != list(range(len(indices))):
            raise RuntimeError("distributed sample rows do not form one global batch")
        return [row for _index, row in sorted(flattened, key=lambda item: item[0])]

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()
