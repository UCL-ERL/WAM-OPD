"""Native LingBot-VA Student-only runtime for D3 evaluation.

The class below shares the released native Student implementation with the
Goal-D1 runner but intentionally does not construct a Teacher transformer.
It is used for reload and teacher-free behavioral evaluation after a bounded
pilot; importing this module must therefore never load a Teacher checkpoint.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from experiments.waopd_native_closed_loop_runner import NativeModelRuntime


class NativeStudentOnlyRuntime(NativeModelRuntime):
    """Native VA_Server with no Teacher object or Teacher checkpoint load."""

    def __init__(
        self,
        *,
        student_checkpoint: Path,
        device: str,
        save_root: Path,
        enable_offload: bool = True,
        official_offload_parity: bool = False,
    ) -> None:
        from wan_va.configs import VA_CONFIGS
        from wan_va.utils import init_logger
        from wan_va.wan_va_server import VA_Server

        device_obj = torch.device(device)
        if device_obj.type != "cuda":
            raise ValueError("native Student-only runtime requires CUDA")
        init_logger()
        config = deepcopy(VA_CONFIGS["robotwin"])
        config.wan22_pretrained_model_name_or_path = str(student_checkpoint)
        config.save_root = str(save_root)
        config.local_rank = int(device_obj.index or 0)
        config.rank = 0
        config.world_size = 1
        config.infer_mode = "server"
        config.enable_offload = bool(enable_offload)
        config.num_inference_steps = 1
        config.action_num_inference_steps = 1
        self.server = VA_Server(config)
        self.device = device_obj
        self.dtype = self.server.dtype
        self.student_checkpoint = str(student_checkpoint)
        # NativeModelRuntime's inherited reset/history helpers consult this
        # flag even though the Student-only constructor does not load a
        # Teacher.  Keep the choice explicit so V0F can use the official
        # CPU-offload semantics without changing older callers.
        self.official_offload_parity = bool(official_offload_parity)
        self.teacher_transformer = None
        self.teacher = None

    def load_action_adapter(
        self,
        state_path: Path,
        *,
        rank: int = 8,
    ) -> None:
        from experiments.action_output_adapter import attach_action_output_adapter

        attach_action_output_adapter(
            self.server.transformer,
            rank=int(rank),
            initialization="zero_up",
        )
        state = torch.load(
            state_path.expanduser().resolve(),
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(state, dict) and "adapter_state_dict" in state:
            state = state["adapter_state_dict"]
        if not isinstance(state, dict):
            raise TypeError(f"adapter checkpoint is not a state dict: {state_path}")
        self.server.transformer.load_state_dict(state, strict=False)


def adapter_parameter_hashes(runtime: NativeStudentOnlyRuntime) -> dict[str, str]:
    """Hash the reloadable action-local parameters without loading a Teacher."""

    from experiments.goal1_exact_condition import tensor_hash

    adapter = getattr(runtime.server.transformer, "action_proj_out", None)
    if adapter is None or not hasattr(adapter, "named_parameters"):
        return {}
    return {
        str(name): str(tensor_hash(parameter))
        for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    }


__all__ = ["NativeStudentOnlyRuntime", "adapter_parameter_hashes"]
