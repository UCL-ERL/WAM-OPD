"""Native Goal-V0 video-labeling, exact-condition, and bounded-trainer core.

The module deliberately keeps the Teacher out of environment execution.  It
uses the released native LingBot runtime, a single isolated ``proj_out``
adapter on the Student video output, and semantic history reconstruction for
each context.  No RLinf adaptation is imported.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from experiments.goal1_exact_condition import (
    ConditionContractError,
    ConditionFingerprint,
    PreparedPlan,
    build_condition_fingerprint,
    cache_valid_length,
    capture_prepared_plan,
    fingerprint_mismatches,
    grid_token_positions,
    prepare_plan_input,
    sequence_hash,
    stable_hash,
    tensor_diff,
    tensor_hash,
)
from experiments.video_output_adapter import (
    VideoOutputResidualAdapter,
    attach_video_output_adapter,
    video_output_adapter_state_dict,
)
from experiments.video_mode_lora import (
    attach_video_mode_lora,
    video_mode_lora_base_parameter_hashes,
    video_mode_lora_scope,
    video_mode_lora_state_dict,
)
from experiments.dual_mode_lora import (
    attach_dual_mode_lora,
    dual_mode_lora_base_parameter_hashes,
    dual_mode_lora_contract,
    dual_mode_lora_named_parameters,
    dual_mode_lora_scope,
    dual_mode_lora_state_dict,
    load_dual_mode_lora_checkpoint,
    select_dual_mode_lora_trainable_bank,
)
from experiments.joint_lora import (
    attach_joint_lora,
    joint_lora_base_parameter_hashes,
    joint_lora_state_dict,
)
from experiments.waopd_native_closed_loop_runner import (
    HistoryInput,
    NativeClosedLoopError,
    NativeModelRuntime,
    _action_condition,
    _fingerprint,
    _normalization_metadata,
    _valid_action_mask,
)


FROZEN_SAVED_INITIAL_LATENT_MODE = "frozen_saved_initial_latent_v1"
RETENTION_INITIAL_LATENT_MODE = FROZEN_SAVED_INITIAL_LATENT_MODE
TARGET_INITIAL_LATENT_MODE = FROZEN_SAVED_INITIAL_LATENT_MODE
VIDEO_CONSISTENCY_SIGMA_DATA = 0.5


def flow_x0_prediction(
    noisy_state: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma: float | torch.Tensor,
) -> torch.Tensor:
    """Map a Flow Matching velocity to its pseudo-clean endpoint."""

    if tuple(noisy_state.shape) != tuple(velocity.shape):
        raise ConditionContractError("flow x0 state/velocity shape mismatch")
    if isinstance(sigma, torch.Tensor):
        sigma_view = _per_frame_sigma_view(sigma, noisy_state)
        return noisy_state - sigma_view * velocity
    resolved_sigma = float(sigma)
    if not np.isfinite(resolved_sigma) or not 0.0 <= resolved_sigma <= 1.0:
        raise ValueError(f"flow sigma must be finite in [0, 1], got {sigma!r}")
    return noisy_state - resolved_sigma * velocity


def _per_frame_sigma_view(
    sigma: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    if reference.ndim < 3:
        raise ValueError("flow state must expose a frame dimension")
    values = sigma.detach().to(device=reference.device, dtype=reference.dtype)
    if values.ndim != 1 or int(values.numel()) != int(reference.shape[2]):
        raise ValueError("flow sigma tensor must be 1-D and match state frames")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("flow sigma tensor contains nonfinite values")
    if bool(((values < 0.0) | (values > 1.0)).any().item()):
        raise ValueError("flow sigma tensor must lie in [0, 1]")
    return values.reshape(1, 1, int(values.numel()), *([1] * (reference.ndim - 3)))


def video_consistency_map(
    noisy_state: torch.Tensor,
    x0: torch.Tensor,
    *,
    sigma: float | torch.Tensor,
    sigma_data: float = VIDEO_CONSISTENCY_SIGMA_DATA,
) -> torch.Tensor:
    """Apply Flash-WAM's LCM/Karras boundary map to one video endpoint."""

    if tuple(noisy_state.shape) != tuple(x0.shape):
        raise ConditionContractError("video consistency state/x0 shape mismatch")
    resolved_sigma_data = float(sigma_data)
    if not np.isfinite(resolved_sigma_data) or resolved_sigma_data <= 0.0:
        raise ValueError("video sigma_data must be finite and positive")
    if isinstance(sigma, torch.Tensor):
        resolved_sigma = _per_frame_sigma_view(sigma, noisy_state)
        denominator = resolved_sigma.square() + resolved_sigma_data**2
        c_skip = resolved_sigma_data**2 / denominator
        c_out = resolved_sigma * resolved_sigma_data / denominator.sqrt()
        return c_skip * noisy_state + c_out * x0
    resolved_sigma = float(sigma)
    if not np.isfinite(resolved_sigma) or not 0.0 <= resolved_sigma <= 1.0:
        raise ValueError(f"video sigma must be finite in [0, 1], got {sigma!r}")
    denominator = resolved_sigma**2 + resolved_sigma_data**2
    c_skip = resolved_sigma_data**2 / denominator
    c_out = resolved_sigma * resolved_sigma_data / denominator**0.5
    return c_skip * noisy_state + c_out * x0


def scheduler_sigma_timestep(
    scheduler: object, requested_sigma: float
) -> tuple[torch.Tensor, float]:
    """Resolve a requested flow sigma through the active native scheduler."""

    sigma = float(requested_sigma)
    if not np.isfinite(sigma) or sigma <= 0.0 or sigma > 1.0:
        raise ValueError(f"flow sigma must be finite in (0, 1], got {sigma!r}")
    config = getattr(scheduler, "config", None)
    train_steps = int(getattr(config, "num_train_timesteps", 1000))
    if train_steps <= 0:
        raise ValueError("scheduler num_train_timesteps must be positive")
    scheduler.set_timesteps(train_steps)
    timesteps = getattr(scheduler, "timesteps", None)
    sigmas = getattr(scheduler, "sigmas", None)
    if not isinstance(timesteps, torch.Tensor) or not isinstance(sigmas, torch.Tensor):
        raise TypeError("native flow scheduler must expose Tensor timesteps/sigmas")
    usable_sigmas = sigmas[: int(timesteps.numel())]
    if usable_sigmas.numel() != timesteps.numel() or usable_sigmas.numel() == 0:
        raise ValueError("native flow scheduler has incompatible timesteps/sigmas")
    native_min = float(usable_sigmas.detach().float().cpu().min().item())
    native_max = float(usable_sigmas.detach().float().cpu().max().item())
    boundary_tolerance = 1e-6
    if sigma < native_min - boundary_tolerance or sigma > native_max + boundary_tolerance:
        raise ValueError(
            f"requested sigma {sigma} is outside the native scheduler range "
            f"[{native_min}, {native_max}]"
        )
    index = int(torch.argmin((usable_sigmas.float().cpu() - sigma).abs()).item())
    resolved_sigma = float(usable_sigmas[index].detach().float().cpu().item())
    return timesteps[index].detach().clone(), resolved_sigma


def _finite_stats(delta: torch.Tensor) -> dict[str, float]:
    values = delta.detach().float()
    return {
        "max_abs": float(values.abs().max().item()) if values.numel() else 0.0,
        "mean_abs": float(values.abs().mean().item()) if values.numel() else 0.0,
        "rmse": float(values.square().mean().sqrt().item()) if values.numel() else 0.0,
    }


def _history_from_context(context: dict[str, Any]) -> list[HistoryInput]:
    history: list[HistoryInput] = []
    expected = 0
    for index, row in enumerate(context.get("history", [])):
        frame_st_id = int(row["frame_st_id"])
        if frame_st_id != expected:
            raise ConditionContractError(
                f"V0 history chronology mismatch at {index}: expected {expected}, got {frame_st_id}"
            )
        latent = row["latent"]
        action = row["action"]
        if not isinstance(latent, torch.Tensor) or not isinstance(action, torch.Tensor):
            raise ConditionContractError("V0 history latent/action must be tensors")
        record = HistoryInput(
            frame_st_id=frame_st_id,
            latent=latent,
            action=action,
            observations=deepcopy(row.get("observations", [])),
        )
        history.append(record)
        expected += int(latent.shape[2])
    if expected != int(context["frame_st_id"]):
        raise ConditionContractError(
            f"V0 history ends at {expected}, context frame_st_id={context['frame_st_id']}"
        )
    return history


def _history_hashes(history: Iterable[HistoryInput]) -> tuple[str, str, str]:
    records = list(history)
    return (
        sequence_hash(
            {"frame_st_id": row.frame_st_id, "latent": row.latent, "action": row.action}
            for row in records
        ),
        sequence_hash(
            {"frame_st_id": row.frame_st_id, "latent": row.latent}
            for row in records
        ),
        sequence_hash(
            {"frame_st_id": row.frame_st_id, "action": row.action}
            for row in records
        ),
    )


def _context_initial_latent(runtime: NativeModelRuntime, context: dict[str, Any]) -> torch.Tensor:
    retention_mode = context.get("_v0_retention_initial_latent_mode")
    target_mode = context.get("_v0_target_initial_latent_mode")
    if retention_mode is not None and target_mode is not None:
        raise ConditionContractError("multiple V0 saved initial-latent modes are active")
    saved_mode = retention_mode if retention_mode is not None else target_mode
    if saved_mode is not None:
        if str(saved_mode) != FROZEN_SAVED_INITIAL_LATENT_MODE:
            raise ConditionContractError(
                f"unknown V0 saved initial-latent mode: {saved_mode!r}"
            )
        if retention_mode is not None and str(context.get("split")) != "retention":
            raise ConditionContractError(
                "V0 retention initial-latent mode used by a non-retention context"
            )
        if target_mode is not None:
            schema = str(context.get("schema"))
            split = str(context.get("split"))
            supported_target_context = (
                schema == "waopd_goal_v0l_student_public_history_context_v1"
                and split == "v0l_occupancy"
            ) or (
                schema == "waopd_goal_v0n_student_public_history_context_v1"
                and split in {"train", "target_heldout"}
            ) or (
                schema == "waopd_goal_v0m_student_public_history_context_v1"
                and split in {"target_train", "target_validation"}
            ) or (
                schema == "waopd_video_trajectory_context_v1"
                and split == "train"
            )
            if not supported_target_context:
                raise ConditionContractError(
                    "V0 target initial-latent mode used by an unsupported context"
                )
        saved = context.get("initial_latent")
        if not isinstance(saved, torch.Tensor):
            raise ConditionContractError("V0 context lacks saved initial_latent")
        saved = saved.to(device=runtime.device, dtype=runtime.dtype)
        if not bool(torch.isfinite(saved).all().item()):
            raise ConditionContractError("V0 saved initial_latent is nonfinite")
        # Preserve official prompt/text setup while avoiding a redundant VAE
        # replay that is not bitwise-stable across fresh runtime processes.
        with runtime._auxiliary_compute_scope(text_encoder=True):
            runtime.server.infer({"reset": True, "prompt": str(context["prompt"])})
        runtime.server.init_latent = saved
        return saved.detach().clone()

    initial_latent = runtime.reset(str(context["prompt"]), context["initial_observation"])
    saved = context.get("initial_latent")
    if not isinstance(saved, torch.Tensor):
        raise ConditionContractError("V0 context lacks saved initial_latent")
    saved = saved.to(device=runtime.device, dtype=runtime.dtype)
    if not torch.equal(initial_latent, saved):
        stats = _finite_stats(initial_latent - saved)
        raise ConditionContractError(
            f"V0 initial latent replay mismatch: {json.dumps(stats, sort_keys=True)}"
        )
    return initial_latent


@dataclass
class V0ActionForward:
    endpoint: torch.Tensor
    initial_velocity: torch.Tensor
    action_input_noise: torch.Tensor
    action_timestep: torch.Tensor
    valid_mask: torch.Tensor
    token_positions: tuple[int, ...]
    cache_valid_length: int
    fingerprint: ConditionFingerprint


@dataclass
class V0StudentVideoForward:
    plan: PreparedPlan
    action: V0ActionForward
    raw_video_noise: torch.Tensor
    raw_action_noise: torch.Tensor
    future_mask: torch.Tensor


@dataclass
class V0SigmaForward:
    """One exact-condition Student action prediction at a resolved sigma."""

    requested_sigma: float
    sigma: float
    timestep: torch.Tensor
    noisy_state: torch.Tensor
    velocity: torch.Tensor
    x0_prediction: torch.Tensor
    valid_mask: torch.Tensor | None = None
    token_positions: tuple[int, ...] = ()
    cache_valid_length: int | None = None
    fingerprint: ConditionFingerprint | None = None


@dataclass
class V0VideoSigmaForward:
    """Video flow endpoint and modality-specific consistency prediction."""

    requested_sigma: float
    sigma: float
    timestep: torch.Tensor
    noisy_state: torch.Tensor
    velocity: torch.Tensor
    x0_hat: torch.Tensor
    consistency_prediction: torch.Tensor


@dataclass
class V0ExplicitVideoForward:
    """One video vector-field/consistency evaluation at an explicit state."""

    timestep: torch.Tensor
    noisy_state: torch.Tensor
    velocity: torch.Tensor
    x0_prediction: torch.Tensor
    consistency_prediction: torch.Tensor


@dataclass
class V0ExplicitActionForward:
    """One hierarchical action evaluation conditioned on an explicit plan."""

    timestep: torch.Tensor
    noisy_state: torch.Tensor
    velocity: torch.Tensor
    x0_prediction: torch.Tensor
    valid_mask: torch.Tensor
    token_positions: tuple[int, ...]
    cache_valid_length: int


class NativeV0VideoRuntime(NativeModelRuntime):
    """Native Student+Teacher runtime with one video-local Student adapter."""

    def __init__(
        self,
        *,
        student_checkpoint: Path,
        teacher_transformer: Path | None,
        device: str,
        save_root: Path,
        enable_offload: bool = True,
        official_offload_parity: bool = False,
        adapter_rank: int = 8,
        adapter_state: Path | None = None,
        adapter_kind: str = "output",
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        lora_block_indices: tuple[int, ...] = (26, 27, 28, 29),
    ) -> None:
        super().__init__(
            student_checkpoint=student_checkpoint,
            teacher_transformer=teacher_transformer,
            device=device,
            save_root=save_root,
            enable_offload=enable_offload,
            official_offload_parity=official_offload_parity,
        )
        self.adapter_kind = str(adapter_kind)
        if self.adapter_kind == "none":
            self.adapter_info = None
        elif self.adapter_kind == "output" and adapter_state is None:
            self.adapter_info = attach_video_output_adapter(
                self.server.transformer,
                rank=int(adapter_rank),
                initialization="zero_up",
            )
        elif self.adapter_kind == "output":
            self.adapter_info = attach_video_output_adapter(
                self.server.transformer,
                rank=int(adapter_rank),
                initialization="zero_up",
            )
            state = torch.load(
                adapter_state.expanduser().resolve(), map_location="cpu", weights_only=True
            )
            if isinstance(state, dict) and "adapter_state_dict" in state:
                state = state["adapter_state_dict"]
            if not isinstance(state, dict):
                raise TypeError(f"V0 adapter checkpoint is not a state dict: {adapter_state}")
            self.server.transformer.load_state_dict(state, strict=False)
        elif self.adapter_kind == "video_lora":
            self.adapter_info = attach_video_mode_lora(
                self.server.transformer,
                rank=int(adapter_rank),
                alpha=float(lora_alpha),
                dropout=float(lora_dropout),
                block_indices=tuple(int(item) for item in lora_block_indices),
            )
            if adapter_state is not None:
                state = torch.load(
                    adapter_state.expanduser().resolve(), map_location="cpu", weights_only=True
                )
                if isinstance(state, dict) and "adapter_state_dict" in state:
                    state = state["adapter_state_dict"]
                if not isinstance(state, dict):
                    raise TypeError(f"V0H LoRA checkpoint is not a state dict: {adapter_state}")
                self.server.transformer.load_state_dict(state, strict=False)
        elif self.adapter_kind == "joint_lora":
            self.adapter_info = attach_joint_lora(
                self.server.transformer,
                rank=int(adapter_rank),
                alpha=float(lora_alpha),
                dropout=float(lora_dropout),
                block_indices=tuple(int(item) for item in lora_block_indices),
            )
            if adapter_state is not None:
                state = torch.load(
                    adapter_state.expanduser().resolve(), map_location="cpu", weights_only=True
                )
                if isinstance(state, dict) and "adapter_state_dict" in state:
                    state = state["adapter_state_dict"]
                if not isinstance(state, dict):
                    raise TypeError(f"V0J Joint LoRA checkpoint is not a state dict: {adapter_state}")
                self.server.transformer.load_state_dict(state, strict=False)
        elif self.adapter_kind == "dual_lora":
            self.adapter_info = attach_dual_mode_lora(
                self.server.transformer,
                rank=int(adapter_rank),
                alpha=float(lora_alpha),
                dropout=float(lora_dropout),
                block_indices=tuple(int(item) for item in lora_block_indices),
            )
            if adapter_state is not None:
                payload = torch.load(
                    adapter_state.expanduser().resolve(),
                    map_location="cpu",
                    weights_only=True,
                )
                if not isinstance(payload, dict):
                    raise TypeError(
                        f"dual-mode LoRA checkpoint is not a mapping: {adapter_state}"
                    )
                load_dual_mode_lora_checkpoint(
                    self.server.transformer, self.adapter_info, payload
                )
        else:
            raise ValueError(f"unsupported V0 adapter_kind={self.adapter_kind!r}")
        self.server.transformer.eval()
        if self.teacher is not None:
            self.teacher.eval()
        self.trainable = [
            (str(name), parameter)
            for name, parameter in self.server.transformer.named_parameters()
            if parameter.requires_grad
        ]
        if not self.trainable and self.adapter_kind != "none":
            raise NativeClosedLoopError("V0 video adapter has no trainable parameters")
        self.adapter_parameter_names = [name for name, _ in self.trainable]

    def _student_video_call_scope(self):  # type: ignore[no-untyped-def]
        if self.adapter_kind == "video_lora":
            return video_mode_lora_scope(True)
        if self.adapter_kind == "dual_lora":
            return dual_mode_lora_scope("video")
        return super()._student_video_call_scope()

    def _student_action_call_scope(self):  # type: ignore[no-untyped-def]
        if self.adapter_kind == "video_lora":
            return video_mode_lora_scope(False)
        if self.adapter_kind == "dual_lora":
            return dual_mode_lora_scope("action")
        return super()._student_action_call_scope()

    def close(self) -> None:
        if getattr(self, "teacher", None) is not None:
            del self.teacher
        del self.server
        torch.cuda.empty_cache()

    def parameter_hashes(self) -> dict[str, str]:
        return {name: str(tensor_hash(parameter)) for name, parameter in self.trainable}

    def base_parameter_hashes(self) -> dict[str, str]:
        if self.adapter_kind == "video_lora":
            return video_mode_lora_base_parameter_hashes(self.server.transformer)
        if self.adapter_kind == "joint_lora":
            return joint_lora_base_parameter_hashes(self.server.transformer)
        if self.adapter_kind == "dual_lora":
            return dual_mode_lora_base_parameter_hashes(self.server.transformer)
        return {}

    def adapter_trainable(
        self, bank: str = "both"
    ) -> list[tuple[str, torch.nn.Parameter]]:
        """Return an explicit optimizer parameter set for one adapter bank."""

        if self.adapter_kind == "dual_lora":
            return dual_mode_lora_named_parameters(
                self.server.transformer, bank=bank
            )
        if bank != "both":
            raise ValueError(
                f"adapter_kind={self.adapter_kind!r} does not expose bank={bank!r}"
            )
        return list(self.trainable)

    def select_adapter_trainable_bank(
        self, bank: str
    ) -> list[tuple[str, torch.nn.Parameter]]:
        """Freeze the inactive dual bank and expose one optimizer-safe list."""

        if self.adapter_kind != "dual_lora":
            raise ValueError(
                "bank selection requires adapter_kind='dual_lora', got "
                f"{self.adapter_kind!r}"
            )
        selected = select_dual_mode_lora_trainable_bank(
            self.server.transformer, bank=bank
        )
        self.trainable = list(selected)
        self.adapter_parameter_names = [name for name, _ in selected]
        return list(selected)

    def adapter_contract(self) -> dict[str, object]:
        if self.adapter_kind == "dual_lora":
            return dual_mode_lora_contract(self.adapter_info)
        if self.adapter_kind == "joint_lora":
            return self.adapter_info.to_dict()
        raise NativeClosedLoopError(
            f"adapter_kind={self.adapter_kind!r} has no adapter contract"
        )

    def adapter_state(self) -> dict[str, torch.Tensor]:
        if self.adapter_kind == "output":
            return video_output_adapter_state_dict(self.server.transformer)
        if self.adapter_kind == "video_lora":
            return video_mode_lora_state_dict(self.server.transformer)
        if self.adapter_kind == "joint_lora":
            return joint_lora_state_dict(self.server.transformer)
        if self.adapter_kind == "dual_lora":
            return dual_mode_lora_state_dict(self.server.transformer, bank="both")
        raise NativeClosedLoopError("adapter_state requested from no-adapter runtime")

    def _replay_history(
        self,
        *,
        model: object,
        cache_name: str,
        history: Iterable[HistoryInput],
    ) -> None:
        """Rebuild semantic cache with LoRA active only on Student video calls."""

        super()._replay_history(model=model, cache_name=cache_name, history=history)

    def _prepare_context(self, context: dict[str, Any]) -> tuple[torch.Tensor, list[HistoryInput]]:
        initial_latent = _context_initial_latent(self, context)
        history = _history_from_context(context)
        self.server.frame_st_id = int(context["frame_st_id"])
        self._replay_history(
            model=self.server.transformer,
            cache_name=self.server.cache_name,
            history=history,
        )
        return initial_latent, history

    def _prepare_teacher_context(
        self, context: dict[str, Any], cache_name: str
    ) -> tuple[torch.Tensor, list[HistoryInput]]:
        initial_latent = _context_initial_latent(self, context)
        history = _history_from_context(context)
        self._replay_history(model=self.teacher, cache_name=cache_name, history=history)
        return initial_latent, history

    def _future_mask(self, plan: torch.Tensor, frame_st_id: int) -> torch.Tensor:
        mask = torch.ones_like(plan, dtype=torch.bool)
        if int(frame_st_id) == 0:
            mask[:, :, 0:1] = False
        if int(mask.sum()) == 0:
            raise ConditionContractError("V0 future video mask is empty")
        return mask

    def _video_plan_student(self, context: dict[str, Any]) -> tuple[PreparedPlan, torch.Tensor, torch.Tensor]:
        """Run the exact one-step Student video solver in normal grad mode."""

        from wan_va.utils import data_seq_to_patch

        server = self.server
        initial_latent, _history = self._prepare_context(context)
        frame_st_id = int(context["frame_st_id"])
        video_noise = context["epsilon_v"].to(device=self.device, dtype=self.dtype)
        server.scheduler.set_timesteps(1)
        video_timesteps = F.pad(server.scheduler.timesteps, (0, 1), value=0)
        latents = video_noise.clone()
        capture: PreparedPlan | None = None
        prepared_plan_for_action: torch.Tensor | None = None
        prepared_timestep_for_action: torch.Tensor | None = None
        for index, timestep in enumerate(video_timesteps):
            last_step = index == len(video_timesteps) - 1
            model_input, candidate = prepare_plan_input(
                server,
                latents,
                frame_st_id=frame_st_id,
                init_latent=initial_latent,
                already_prepared=False,
                latent_t=timestep,
                preserve_grad=True,
            )
            if last_step:
                # Candidate is detached provenance, while model_input retains
                # the adapter graph used by the differentiable forward.
                capture = candidate
                # _repeat_input_for_cfg mutates the input dictionary by
                # replacing its tensors with CFG-batched copies.  Preserve
                # the actual batch-1 canonical action-consumer input before
                # that helper is called.
                prepared_plan_for_action = model_input["latent_res_lst"][
                    "noisy_latents"
                ]
                prepared_timestep_for_action = model_input["latent_res_lst"][
                    "timesteps"
                ].detach().clone()
            with self._student_video_call_scope():
                velocity = server.transformer(
                    server._repeat_input_for_cfg(model_input["latent_res_lst"]),
                    update_cache=1 if last_step else 0,
                    cache_name=server.cache_name,
                    action_mode=False,
                )
            if not last_step:
                velocity = data_seq_to_patch(
                    server.job_config.patch_size,
                    velocity,
                    server.job_config.frame_chunk_size,
                    server.latent_height,
                    server.latent_width,
                    batch_size=2 if server.use_cfg else 1,
                )
                if server.job_config.guidance_scale > 1:
                    velocity = velocity[1:] + server.job_config.guidance_scale * (
                        velocity[:1] - velocity[1:]
                    )
                else:
                    velocity = velocity[:1]
                latents = server.scheduler.step(velocity, timestep, latents)
            latent_cond = (
                initial_latent[:, :, 0:1].to(dtype=latents.dtype)
                if frame_st_id == 0
                else None
            )
            if latent_cond is not None:
                latents = torch.cat([latent_cond, latents[:, :, 1:]], dim=2)
        if capture is None:
            raise NativeClosedLoopError("V0 Student video forward did not expose a plan")
        if prepared_plan_for_action is None or prepared_timestep_for_action is None:
            raise NativeClosedLoopError("V0 Student video forward lost canonical plan input")
        # Use the pre-CFG, batch-1 tensor actually passed to the native
        # preparation helper.  It retains the adapter graph for L_AA and is
        # not the batch-2 replacement installed by _repeat_input_for_cfg.
        prepared_plan = prepared_plan_for_action
        if frame_st_id > 0 and not torch.equal(prepared_plan.detach(), latents.detach()):
            raise ConditionContractError("V0 nonzero Student plan was changed before action call")
        if frame_st_id == 0 and not torch.equal(
            prepared_plan.detach()[:, :, 0:1], initial_latent.detach()[:, :, 0:1]
        ):
            first_slice_delta = (
                prepared_plan.detach()[:, :, 0:1].float()
                - initial_latent.detach()[:, :, 0:1].float()
            )
            raise ConditionContractError(
                "V0 frame0 Student plan lost initial condition: "
                + json.dumps(
                    {
                        "context_id": str(context.get("context_id")),
                        "prepared_dtype": str(prepared_plan.dtype),
                        "initial_dtype": str(initial_latent.dtype),
                        "prepared_device": str(prepared_plan.device),
                        "initial_device": str(initial_latent.device),
                        "max_abs": float(first_slice_delta.abs().max().item()),
                        "mean_abs": float(first_slice_delta.abs().mean().item()),
                        "prepared_hash": tensor_hash(prepared_plan[:, :, 0:1]),
                        "initial_hash": tensor_hash(initial_latent[:, :, 0:1]),
                    },
                    sort_keys=True,
                )
            )
        differentiable_plan = PreparedPlan(
            raw_z_s=latents,
            prepared_z_s=prepared_plan,
            prepared_z_s_timestep=prepared_timestep_for_action,
            latent_cond_applied=frame_st_id == 0,
            latent_cond=initial_latent[:, :, 0:1] if frame_st_id == 0 else None,
        )
        return differentiable_plan, video_noise, initial_latent

    def _action_endpoint_after_plan(
        self,
        context: dict[str, Any],
        plan: torch.Tensor,
        *,
        require_grad: bool,
    ) -> V0ActionForward:
        """Consume one already-prepared plan with the frozen Student action branch."""

        server = self.server
        initial_latent, history = self._prepare_context(context)
        del initial_latent
        frame_st_id = int(context["frame_st_id"])
        plan_input, capture = prepare_plan_input(
            server,
            plan,
            frame_st_id=frame_st_id,
            already_prepared=True,
            latent_t=0,
            preserve_grad=require_grad,
        )
        if not torch.equal(capture.prepared_z_s.detach(), plan.detach()):
            raise ConditionContractError("V0 already-prepared plan was modified")
        video_call = lambda: server.transformer(
            server._repeat_input_for_cfg(plan_input["latent_res_lst"]),
            update_cache=1,
            cache_name=server.cache_name,
            action_mode=False,
        )
        if require_grad:
            with self._student_video_call_scope():
                video_call()
        else:
            with torch.no_grad():
                with self._student_video_call_scope():
                    video_call()

        epsilon_a = context["epsilon_a"].to(device=self.device, dtype=self.dtype)
        server.action_scheduler.set_timesteps(1)
        action_timesteps = F.pad(server.action_scheduler.timesteps, (0, 1), value=0)
        action_cond = _action_condition(server, frame_st_id, dtype=self.dtype)
        actions = epsilon_a.clone()
        if action_cond is not None:
            actions[:, :, 0:1] = action_cond
        actions[:, ~server.action_mask] *= 0
        valid_mask = _valid_action_mask(server, actions, frame_st_id)
        cache_length = cache_valid_length(server.transformer, server.cache_name)
        action_branch_input: dict[str, Any] | None = None
        initial_velocity: torch.Tensor | None = None
        for index, timestep in enumerate(action_timesteps):
            last_step = index == len(action_timesteps) - 1
            model_input = server._prepare_latent_input(
                None,
                actions,
                timestep,
                timestep,
                None,
                action_cond,
                frame_st_id=frame_st_id,
            )
            if action_branch_input is None:
                action_branch_input = {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in model_input["action_res_lst"].items()
                }
            output_call = lambda: server.transformer(
                server._repeat_input_for_cfg(model_input["action_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=server.cache_name,
                action_mode=True,
            )
            if require_grad:
                with self._student_action_call_scope():
                    output = output_call()
            else:
                with torch.no_grad():
                    with self._student_action_call_scope():
                        output = output_call()
            if not last_step:
                velocity = rearrange(
                    output,
                    "b (f n) c -> b c f n 1",
                    f=server.job_config.frame_chunk_size,
                )[:1]
                if initial_velocity is None:
                    # FlashWAM uses one action-flow step.  This is the exact
                    # vector field evaluated at its deployment state x_t.
                    initial_velocity = velocity
                actions = server.action_scheduler.step(velocity, timestep, actions)
                if action_cond is not None:
                    actions[:, :, 0:1] = action_cond
        if action_branch_input is None:
            raise NativeClosedLoopError("V0 action branch did not expose input")
        if initial_velocity is None:
            raise NativeClosedLoopError("V0 action branch did not expose an initial velocity")
        endpoint = actions.clone()
        if action_cond is not None:
            endpoint[:, :, 0:1] = action_cond
        endpoint[:, ~server.action_mask] *= 0
        action_token_positions = tuple(
            int(item)
            for item in action_branch_input["grid_id"].detach().reshape(-1).cpu().tolist()
        )
        fingerprint = _fingerprint(
            runtime=self,
            owner="student",
            history=history,
            prompt_ids=tuple(int(x) for x in context["prompt_token_ids"]),
            plan=PreparedPlan(
                raw_z_s=plan,
                prepared_z_s=plan,
                prepared_z_s_timestep=capture.prepared_z_s_timestep,
                latent_cond_applied=False,
            ),
            action_noise=action_branch_input["noisy_latents"],
            action_timestep=action_branch_input["timesteps"],
            mask=valid_mask,
            frame_st_id=frame_st_id,
            token_positions=action_token_positions,
            cache_valid_length=cache_length,
        )
        return V0ActionForward(
            endpoint=endpoint,
            initial_velocity=initial_velocity,
            action_input_noise=action_branch_input["noisy_latents"],
            action_timestep=action_branch_input["timesteps"],
            valid_mask=valid_mask,
            token_positions=action_token_positions,
            cache_valid_length=cache_length,
            fingerprint=fingerprint,
        )

    def student_video_forward(
        self,
        context: dict[str, Any],
        *,
        detach_plan_for_action: bool = False,
    ) -> V0StudentVideoForward:
        plan, video_noise, _initial_latent = self._video_plan_student(context)
        # The video-plan forward populated the cache.  Rebuild semantic
        # history and inject the exact prepared plan through the same action
        # consumer so a_self has a clean, auditable cache chronology.
        self.server.transformer.clear_cache(self.server.cache_name)
        action_plan = (
            plan.prepared_z_s.detach()
            if detach_plan_for_action
            else plan.prepared_z_s
        )
        action = self._action_endpoint_after_plan(context, action_plan, require_grad=True)
        self.server.transformer.clear_cache(self.server.cache_name)
        future_mask = self._future_mask(plan.prepared_z_s, int(context["frame_st_id"]))
        return V0StudentVideoForward(
            plan=plan,
            action=action,
            raw_video_noise=video_noise,
            raw_action_noise=context["epsilon_a"].to(device=self.device, dtype=self.dtype),
            future_mask=future_mask,
        )

    def student_action_on_plan(
        self, context: dict[str, Any], plan: torch.Tensor, *, require_grad: bool
    ) -> V0ActionForward:
        try:
            return self._action_endpoint_after_plan(context, plan, require_grad=require_grad)
        finally:
            self.server.transformer.clear_cache(self.server.cache_name)

    def student_video_consistency_at_state(
        self,
        context: dict[str, Any],
        state: torch.Tensor,
        *,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
        require_grad: bool,
    ) -> V0ExplicitVideoForward:
        """Evaluate the online/temporarily-swapped Student on an explicit state."""

        from wan_va.utils import data_seq_to_patch

        server = self.server
        try:
            initial_latent, _history = self._prepare_context(context)
            frame_st_id = int(context["frame_st_id"])
            model_input, _capture = prepare_plan_input(
                server,
                state.to(device=self.device, dtype=self.dtype).clone(),
                frame_st_id=frame_st_id,
                init_latent=initial_latent,
                already_prepared=False,
                latent_t=timestep.to(device=self.device),
                preserve_grad=False,
            )
            canonical_state = model_input["latent_res_lst"]["noisy_latents"].clone()
            canonical_timestep = model_input["latent_res_lst"][
                "timesteps"
            ].detach().clone()

            def video_call() -> torch.Tensor:
                return server.transformer(
                    server._repeat_input_for_cfg(model_input["latent_res_lst"]),
                    update_cache=0,
                    cache_name=server.cache_name,
                    action_mode=False,
                )

            if require_grad:
                with self._student_video_call_scope():
                    velocity_seq = video_call()
            else:
                with torch.no_grad():
                    with self._student_video_call_scope():
                        velocity_seq = video_call()
            velocity = data_seq_to_patch(
                server.job_config.patch_size,
                velocity_seq,
                server.job_config.frame_chunk_size,
                server.latent_height,
                server.latent_width,
                batch_size=2 if server.use_cfg else 1,
            )
            if server.job_config.guidance_scale > 1:
                velocity = velocity[1:] + server.job_config.guidance_scale * (
                    velocity[:1] - velocity[1:]
                )
            else:
                velocity = velocity[:1]
            if tuple(velocity.shape) != tuple(canonical_state.shape):
                raise ConditionContractError(
                    "explicit Student video velocity/state shape mismatch"
                )
            x0_prediction = flow_x0_prediction(
                canonical_state, velocity, sigma=sigma
            )
            return V0ExplicitVideoForward(
                timestep=canonical_timestep,
                noisy_state=canonical_state,
                velocity=velocity,
                x0_prediction=x0_prediction,
                consistency_prediction=video_consistency_map(
                    canonical_state, x0_prediction, sigma=sigma
                ),
            )
        finally:
            server.transformer.clear_cache(server.cache_name)

    def teacher_video_velocity_at_state(
        self,
        context: dict[str, Any],
        state: torch.Tensor,
        *,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
    ) -> V0ExplicitVideoForward:
        """Evaluate the frozen Teacher video field at an explicit native state."""

        from wan_va.utils import data_seq_to_patch

        if self.teacher is None:
            raise NativeClosedLoopError("Teacher video bridge requires a Teacher")
        server = self.server
        cache_name = "coherent_tt_teacher_video_bridge"
        try:
            initial_latent, _history = self._prepare_teacher_context(
                context, cache_name
            )
            model_input, _capture = prepare_plan_input(
                server,
                state.to(device=self.device, dtype=self.dtype).clone(),
                frame_st_id=int(context["frame_st_id"]),
                init_latent=initial_latent,
                already_prepared=False,
                latent_t=timestep.to(device=self.device),
                preserve_grad=False,
            )
            canonical_state = model_input["latent_res_lst"]["noisy_latents"].clone()
            canonical_timestep = model_input["latent_res_lst"][
                "timesteps"
            ].detach().clone()
            with torch.inference_mode():
                velocity_seq = self.teacher(
                    server._repeat_input_for_cfg(model_input["latent_res_lst"]),
                    update_cache=0,
                    cache_name=cache_name,
                    action_mode=False,
                )
            velocity = data_seq_to_patch(
                server.job_config.patch_size,
                velocity_seq,
                server.job_config.frame_chunk_size,
                server.latent_height,
                server.latent_width,
                batch_size=2 if server.use_cfg else 1,
            )
            if server.job_config.guidance_scale > 1:
                velocity = velocity[1:] + server.job_config.guidance_scale * (
                    velocity[:1] - velocity[1:]
                )
            else:
                velocity = velocity[:1]
            if tuple(velocity.shape) != tuple(canonical_state.shape):
                raise ConditionContractError(
                    "explicit Teacher video velocity/state shape mismatch"
                )
            x0_prediction = flow_x0_prediction(
                canonical_state, velocity, sigma=sigma
            )
            return V0ExplicitVideoForward(
                timestep=canonical_timestep,
                noisy_state=canonical_state,
                velocity=velocity,
                x0_prediction=x0_prediction,
                consistency_prediction=video_consistency_map(
                    canonical_state, x0_prediction, sigma=sigma
                ),
            )
        finally:
            try:
                self.teacher.clear_cache(cache_name)
            except (KeyError, TypeError, AttributeError):
                pass

    def student_action_consistency_at_state(
        self,
        context: dict[str, Any],
        plan: torch.Tensor,
        plan_timestep: torch.Tensor,
        state: torch.Tensor,
        *,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
        require_grad: bool,
    ) -> V0ExplicitActionForward:
        """Evaluate Student action on ``detach(z_T)`` at an explicit state."""

        server = self.server
        try:
            _initial_latent, _history = self._prepare_context(context)
            frame_st_id = int(context["frame_st_id"])
            detached_plan = plan.detach().to(device=self.device, dtype=self.dtype)
            plan_input, capture = prepare_plan_input(
                server,
                detached_plan,
                frame_st_id=frame_st_id,
                already_prepared=True,
                latent_t=plan_timestep.to(device=self.device),
                preserve_grad=False,
            )
            if not torch.equal(capture.prepared_z_s.detach(), detached_plan):
                raise ConditionContractError(
                    "explicit Student action path modified Teacher z_T"
                )

            def plan_call() -> torch.Tensor:
                return server.transformer(
                    server._repeat_input_for_cfg(plan_input["latent_res_lst"]),
                    update_cache=1,
                    cache_name=server.cache_name,
                    action_mode=False,
                )

            if require_grad:
                with self._student_video_call_scope():
                    plan_call()
            else:
                with torch.no_grad():
                    with self._student_video_call_scope():
                        plan_call()
            cache_length = cache_valid_length(
                server.transformer, server.cache_name
            )
            action_cond = _action_condition(server, frame_st_id, dtype=self.dtype)
            action_state = state.to(device=self.device, dtype=self.dtype).clone()
            if action_cond is not None:
                action_state[:, :, 0:1] = action_cond
            action_state[:, ~server.action_mask] *= 0
            valid_mask = _valid_action_mask(server, action_state, frame_st_id)
            model_input = server._prepare_latent_input(
                None,
                action_state,
                timestep.to(device=self.device),
                timestep.to(device=self.device),
                None,
                action_cond,
                frame_st_id=frame_st_id,
            )
            action_input = model_input["action_res_lst"]
            canonical_state = action_input["noisy_latents"].clone()
            canonical_timestep = action_input["timesteps"].detach().clone()
            token_positions = tuple(
                int(item)
                for item in action_input["grid_id"].detach().reshape(-1).cpu().tolist()
            )

            def action_call() -> torch.Tensor:
                return server.transformer(
                    server._repeat_input_for_cfg(action_input),
                    update_cache=0,
                    cache_name=server.cache_name,
                    action_mode=True,
                )

            if require_grad:
                with self._student_action_call_scope():
                    output = action_call()
            else:
                with torch.no_grad():
                    with self._student_action_call_scope():
                        output = action_call()
            velocity = rearrange(
                output,
                "b (f n) c -> b c f n 1",
                f=server.job_config.frame_chunk_size,
            )[:1]
            if tuple(velocity.shape) != tuple(canonical_state.shape):
                raise ConditionContractError(
                    "explicit Student action velocity/state shape mismatch"
                )
            x0_prediction = flow_x0_prediction(
                canonical_state, velocity, sigma=sigma
            )
            if action_cond is not None:
                x0_prediction[:, :, 0:1] = action_cond
            x0_prediction[:, ~server.action_mask] *= 0
            return V0ExplicitActionForward(
                timestep=canonical_timestep,
                noisy_state=canonical_state,
                velocity=velocity,
                x0_prediction=x0_prediction,
                valid_mask=valid_mask,
                token_positions=token_positions,
                cache_valid_length=int(cache_length),
            )
        finally:
            server.transformer.clear_cache(server.cache_name)

    def teacher_action_velocity_at_state(
        self,
        context: dict[str, Any],
        plan: torch.Tensor,
        plan_timestep: torch.Tensor,
        state: torch.Tensor,
        *,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
    ) -> V0ExplicitActionForward:
        """Evaluate frozen Teacher action on the same exact ``z_T`` condition."""

        if self.teacher is None:
            raise NativeClosedLoopError("Teacher action bridge requires a Teacher")
        server = self.server
        cache_name = "coherent_tt_teacher_action_bridge"
        try:
            _initial_latent, _history = self._prepare_teacher_context(
                context, cache_name
            )
            frame_st_id = int(context["frame_st_id"])
            detached_plan = plan.detach().to(device=self.device, dtype=self.dtype)
            plan_input, capture = prepare_plan_input(
                server,
                detached_plan,
                frame_st_id=frame_st_id,
                already_prepared=True,
                latent_t=plan_timestep.to(device=self.device),
                preserve_grad=False,
            )
            if not torch.equal(capture.prepared_z_s.detach(), detached_plan):
                raise ConditionContractError(
                    "explicit Teacher action path modified Teacher z_T"
                )
            with torch.inference_mode():
                self.teacher(
                    server._repeat_input_for_cfg(plan_input["latent_res_lst"]),
                    update_cache=1,
                    cache_name=cache_name,
                    action_mode=False,
                )
            cache_length = cache_valid_length(self.teacher, cache_name)
            action_cond = _action_condition(server, frame_st_id, dtype=self.dtype)
            action_state = state.to(device=self.device, dtype=self.dtype).clone()
            if action_cond is not None:
                action_state[:, :, 0:1] = action_cond
            action_state[:, ~server.action_mask] *= 0
            valid_mask = _valid_action_mask(server, action_state, frame_st_id)
            model_input = server._prepare_latent_input(
                None,
                action_state,
                timestep.to(device=self.device),
                timestep.to(device=self.device),
                None,
                action_cond,
                frame_st_id=frame_st_id,
            )
            action_input = model_input["action_res_lst"]
            canonical_state = action_input["noisy_latents"].clone()
            canonical_timestep = action_input["timesteps"].detach().clone()
            token_positions = tuple(
                int(item)
                for item in action_input["grid_id"].detach().reshape(-1).cpu().tolist()
            )
            with torch.inference_mode():
                output = self.teacher(
                    server._repeat_input_for_cfg(action_input),
                    update_cache=0,
                    cache_name=cache_name,
                    action_mode=True,
                )
            velocity = rearrange(
                output,
                "b (f n) c -> b c f n 1",
                f=server.job_config.frame_chunk_size,
            )[:1]
            if tuple(velocity.shape) != tuple(canonical_state.shape):
                raise ConditionContractError(
                    "explicit Teacher action velocity/state shape mismatch"
                )
            x0_prediction = flow_x0_prediction(
                canonical_state, velocity, sigma=sigma
            )
            if action_cond is not None:
                x0_prediction[:, :, 0:1] = action_cond
            x0_prediction[:, ~server.action_mask] *= 0
            return V0ExplicitActionForward(
                timestep=canonical_timestep,
                noisy_state=canonical_state,
                velocity=velocity,
                x0_prediction=x0_prediction,
                valid_mask=valid_mask,
                token_positions=token_positions,
                cache_valid_length=int(cache_length),
            )
        finally:
            try:
                self.teacher.clear_cache(cache_name)
            except (KeyError, TypeError, AttributeError):
                pass

    def student_video_x0_at_sigma(
        self,
        context: dict[str, Any],
        teacher_endpoint: torch.Tensor,
        *,
        sigma: float,
        require_grad: bool,
    ) -> V0VideoSigmaForward:
        """Evaluate video endpoint and Karras map on a Teacher-endpoint re-noise."""

        from wan_va.utils import data_seq_to_patch

        server = self.server
        try:
            initial_latent, _history = self._prepare_context(context)
            frame_st_id = int(context["frame_st_id"])
            epsilon = context["epsilon_v"].to(device=self.device, dtype=self.dtype)
            target = teacher_endpoint.to(device=self.device, dtype=self.dtype)
            if target.shape != epsilon.shape:
                raise ConditionContractError("video sigma target/noise shape mismatch")
            timestep, resolved_sigma = scheduler_sigma_timestep(
                server.scheduler, float(sigma)
            )
            timestep = timestep.to(device=self.device)
            noisy = (1.0 - resolved_sigma) * target + resolved_sigma * epsilon
            model_input, _capture = prepare_plan_input(
                server,
                noisy,
                frame_st_id=frame_st_id,
                init_latent=initial_latent,
                already_prepared=False,
                latent_t=timestep,
                preserve_grad=require_grad,
            )
            canonical_noisy = model_input["latent_res_lst"]["noisy_latents"].clone()
            canonical_timesteps = model_input["latent_res_lst"][
                "timesteps"
            ].detach().clone()

            def video_call() -> torch.Tensor:
                return server.transformer(
                    server._repeat_input_for_cfg(model_input["latent_res_lst"]),
                    update_cache=0,
                    cache_name=server.cache_name,
                    action_mode=False,
                )

            if require_grad:
                with self._student_video_call_scope():
                    velocity_seq = video_call()
            else:
                with torch.no_grad():
                    with self._student_video_call_scope():
                        velocity_seq = video_call()
            velocity = data_seq_to_patch(
                server.job_config.patch_size,
                velocity_seq,
                server.job_config.frame_chunk_size,
                server.latent_height,
                server.latent_width,
                batch_size=2 if server.use_cfg else 1,
            )
            if server.job_config.guidance_scale > 1:
                velocity = velocity[1:] + server.job_config.guidance_scale * (
                    velocity[:1] - velocity[1:]
                )
            else:
                velocity = velocity[:1]
            if velocity.shape != canonical_noisy.shape:
                raise ConditionContractError("video sigma velocity/state shape mismatch")
            x0_hat = flow_x0_prediction(
                canonical_noisy,
                velocity,
                sigma=resolved_sigma,
            )
            consistency_prediction = video_consistency_map(
                canonical_noisy,
                x0_hat,
                sigma=resolved_sigma,
            )
            return V0VideoSigmaForward(
                requested_sigma=float(sigma),
                sigma=resolved_sigma,
                timestep=canonical_timesteps,
                noisy_state=canonical_noisy,
                velocity=velocity,
                x0_hat=x0_hat,
                consistency_prediction=consistency_prediction,
            )
        finally:
            server.transformer.clear_cache(server.cache_name)

    def student_action_x0_at_sigma(
        self,
        context: dict[str, Any],
        plan: torch.Tensor,
        teacher_endpoint: torch.Tensor,
        *,
        sigma: float,
        require_grad: bool,
    ) -> V0SigmaForward:
        """Evaluate action flow at sigma while consuming the exact Student plan."""

        server = self.server
        try:
            _initial_latent, history = self._prepare_context(context)
            frame_st_id = int(context["frame_st_id"])
            plan_input, capture = prepare_plan_input(
                server,
                plan,
                frame_st_id=frame_st_id,
                already_prepared=True,
                latent_t=0,
                preserve_grad=require_grad,
            )
            if not torch.equal(capture.prepared_z_s.detach(), plan.detach()):
                raise ConditionContractError("sigma action path modified Student z_S")

            def plan_call() -> torch.Tensor:
                return server.transformer(
                    server._repeat_input_for_cfg(plan_input["latent_res_lst"]),
                    update_cache=1,
                    cache_name=server.cache_name,
                    action_mode=False,
                )

            if require_grad:
                with self._student_video_call_scope():
                    plan_call()
            else:
                with torch.no_grad():
                    with self._student_video_call_scope():
                        plan_call()

            epsilon = context["epsilon_a"].to(device=self.device, dtype=self.dtype)
            target = teacher_endpoint.to(device=self.device, dtype=self.dtype)
            if target.shape != epsilon.shape:
                raise ConditionContractError("action sigma target/noise shape mismatch")
            timestep, resolved_sigma = scheduler_sigma_timestep(
                server.action_scheduler, float(sigma)
            )
            timestep = timestep.to(device=self.device)
            action_cond = _action_condition(server, frame_st_id, dtype=self.dtype)
            noisy = (1.0 - resolved_sigma) * target + resolved_sigma * epsilon
            if action_cond is not None:
                noisy[:, :, 0:1] = action_cond
            noisy[:, ~server.action_mask] *= 0
            valid_mask = _valid_action_mask(server, noisy, frame_st_id)
            cache_length = cache_valid_length(server.transformer, server.cache_name)
            model_input = server._prepare_latent_input(
                None,
                noisy,
                timestep,
                timestep,
                None,
                action_cond,
                frame_st_id=frame_st_id,
            )
            canonical_noisy = model_input["action_res_lst"]["noisy_latents"].clone()
            action_timesteps = model_input["action_res_lst"]["timesteps"].detach().clone()
            token_positions = tuple(
                int(item)
                for item in model_input["action_res_lst"]["grid_id"]
                .detach()
                .reshape(-1)
                .cpu()
                .tolist()
            )

            def action_call() -> torch.Tensor:
                return server.transformer(
                    server._repeat_input_for_cfg(model_input["action_res_lst"]),
                    update_cache=0,
                    cache_name=server.cache_name,
                    action_mode=True,
                )

            if require_grad:
                with self._student_action_call_scope():
                    output = action_call()
            else:
                with torch.no_grad():
                    with self._student_action_call_scope():
                        output = action_call()
            velocity = rearrange(
                output,
                "b (f n) c -> b c f n 1",
                f=server.job_config.frame_chunk_size,
            )[:1]
            if velocity.shape != canonical_noisy.shape:
                raise ConditionContractError("action sigma velocity/state shape mismatch")
            x0_prediction = flow_x0_prediction(
                canonical_noisy,
                velocity,
                sigma=resolved_sigma,
            )
            if action_cond is not None:
                x0_prediction[:, :, 0:1] = action_cond
            x0_prediction[:, ~server.action_mask] *= 0
            fingerprint = _fingerprint(
                runtime=self,
                owner="student",
                history=history,
                prompt_ids=tuple(int(x) for x in context["prompt_token_ids"]),
                plan=PreparedPlan(
                    raw_z_s=plan,
                    prepared_z_s=plan,
                    prepared_z_s_timestep=capture.prepared_z_s_timestep,
                    latent_cond_applied=False,
                ),
                action_noise=canonical_noisy,
                action_timestep=action_timesteps,
                mask=valid_mask,
                frame_st_id=frame_st_id,
                token_positions=token_positions,
                cache_valid_length=cache_length,
            )
            return V0SigmaForward(
                requested_sigma=float(sigma),
                sigma=resolved_sigma,
                timestep=action_timesteps,
                noisy_state=canonical_noisy,
                velocity=velocity,
                x0_prediction=x0_prediction,
                valid_mask=valid_mask,
                token_positions=token_positions,
                cache_valid_length=cache_length,
                fingerprint=fingerprint,
            )
        finally:
            server.transformer.clear_cache(server.cache_name)

    def teacher_video_label(self, context: dict[str, Any]) -> dict[str, Any]:
        teacher_cache = f"v0_teacher_{context['context_id']}"
        try:
            initial_latent, _history = self._prepare_teacher_context(context, teacher_cache)
            epsilon_v = context["epsilon_v"].to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                plan = self._teacher_video_plan(
                    frame_st_id=int(context["frame_st_id"]),
                    initial_latent=initial_latent,
                    video_noise=epsilon_v,
                    cache_name=teacher_cache,
                )
            if plan.prepared_z_s.shape != context["prepared_z_s"].to(self.device).shape:
                raise ConditionContractError("V0 Teacher latent shape is not Student-consumable")
            if int(context["frame_st_id"]) > 0 and plan.latent_cond_applied:
                raise ConditionContractError("V0 Teacher applied episode initial latent at nonzero frame")
            return {
                "schema": "waopd_goal_v0_teacher_video_label_v1",
                "artifact_kind": "v0_teacher_video_label",
                "training_started": False,
                "context_id": str(context["context_id"]),
                "unit_id": str(context["unit_id"]),
                "task": str(context["task"]),
                "seed": int(context["seed"]),
                "frame_st_id": int(context["frame_st_id"]),
                "macro_id": int(context["macro_id"]),
                "epsilon_v": epsilon_v.detach().cpu(),
                "epsilon_a": context["epsilon_a"].detach().cpu(),
                "epsilon_v_hash": tensor_hash(epsilon_v),
                "epsilon_a_hash": tensor_hash(context["epsilon_a"]),
                "raw_z_s": context["raw_z_s"],
                "prepared_z_s": context["prepared_z_s"],
                "prepared_z_s_hash": tensor_hash(context["prepared_z_s"]),
                "teacher_z_t": plan.prepared_z_s.detach().cpu(),
                "teacher_z_t_hash": tensor_hash(plan.prepared_z_s),
                "teacher_z_t_timestep": plan.prepared_z_s_timestep.detach().cpu(),
                "teacher_z_t_timestep_hash": tensor_hash(plan.prepared_z_s_timestep),
                "teacher_latent_cond_applied": bool(plan.latent_cond_applied),
                "student_plan_directly_consumable": True,
                "translator_or_reencode": False,
                "cache_owner": "teacher_transformer_only",
                "macro_boundary": [1.0, 0.0],
            }
        finally:
            self.teacher.clear_cache(teacher_cache)


def save_teacher_label(path: Path, label: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(label, path)


def video_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 1e-3,
) -> torch.Tensor:
    if tuple(prediction.shape) != tuple(target.shape) or tuple(mask.shape) != tuple(prediction.shape):
        raise ConditionContractError("V0 video loss shape/mask mismatch")
    if int(mask.sum()) == 0:
        raise ConditionContractError("V0 video loss mask is empty")
    return F.smooth_l1_loss(prediction.float()[mask], target.float()[mask], beta=float(beta))


def action_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 1e-3,
) -> torch.Tensor:
    if tuple(prediction.shape) != tuple(target.shape) or tuple(mask.shape) != tuple(prediction.shape):
        raise ConditionContractError("V0 action loss shape/mask mismatch")
    if int(mask.sum()) == 0:
        raise ConditionContractError("V0 action loss mask is empty")
    return F.smooth_l1_loss(prediction.float()[mask], target.float()[mask], beta=float(beta))


def action_velocity_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE between Teacher and Student vector fields at one shared action state."""

    if tuple(prediction.shape) != tuple(target.shape) or tuple(mask.shape) != tuple(prediction.shape):
        raise ConditionContractError("V0 action velocity loss shape/mask mismatch")
    if int(mask.sum()) == 0:
        raise ConditionContractError("V0 action velocity loss mask is empty")
    return F.mse_loss(prediction.float()[mask], target.float()[mask])


def adapter_parameter_hashes(runtime: NativeV0VideoRuntime) -> dict[str, str]:
    return runtime.parameter_hashes()


__all__ = [
    "NativeV0VideoRuntime",
    "V0ActionForward",
    "V0StudentVideoForward",
    "V0SigmaForward",
    "V0VideoSigmaForward",
    "VIDEO_CONSISTENCY_SIGMA_DATA",
    "action_huber_loss",
    "action_velocity_mse_loss",
    "adapter_parameter_hashes",
    "flow_x0_prediction",
    "save_teacher_label",
    "scheduler_sigma_timestep",
    "video_consistency_map",
    "video_huber_loss",
]
