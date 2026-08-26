"""Exact-condition contract for Flash-WAM 1v/1a action labeling.

This module is deliberately independent of the model implementation.  It owns
the provenance object and the checks that are shared by deployment, Teacher
labeling, and replay.  Model-specific code may construct an input dictionary,
but it must pass the resulting tensor through :func:`capture_prepared_plan` or
use :func:`prepare_plan_input`; neither helper hides a second initial-condition
application.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Iterable, Mapping

import torch


CANONICAL_SCHEMA_VERSION = "flashwam_canonical_action_context_v2"
FINGERPRINT_SCHEMA_VERSION = "flashwam_condition_fingerprint_v1"
GOAL1_PRODUCTION_SCHEMA_VERSION = "flashwam_goal1_production_record_v1"

# ``state_key`` and ``label_key`` belong to the later Stage-G dataset
# manifest.  A Goal-1 canonical record is a condition record, not yet a
# Stage-G training row, so those fields are intentionally optional here.  The
# policy is serialized into every production record so a loader cannot infer
# optionality from missing keys and accidentally accept a legacy artifact.
GOAL1_PRODUCTION_FIELD_POLICY = {
    "artifact_kind": {
        "required": True,
        "type": "string",
        "allowed": [
            "goal1_canonical_action_context",
            "stage_g_teacher_bridge_label",
            "stage_m_live_teacher_bridge_label",
        ],
    },
    "state_key": {
        "required": False,
        "type": "string",
        "reason": "only required for a Stage-G label manifest",
    },
    "label_key": {
        "required": False,
        "type": "string",
        "reason": "only required for a Stage-G label manifest",
    },
}


class ConditionContractError(RuntimeError):
    """Raised when a label cannot be proven to use the canonical condition."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def tensor_hash(value: torch.Tensor | None) -> str | None:
    """Hash shape, dtype, and exact storage bytes without NumPy conversion.

    The byte path is intentional: NumPy does not support every model dtype
    used by Flash-WAM (notably bfloat16), while the action contract requires
    the actual tensor rather than a seed or a rounded summary.
    """

    if value is None:
        return None
    tensor = value.detach().contiguous().cpu()
    header = _json_bytes(
        {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    )
    return hashlib.sha256(header + tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_hashes(values: Mapping[str, torch.Tensor | None]) -> dict[str, str | None]:
    return {name: tensor_hash(value) for name, value in values.items()}


def sequence_hash(values: Iterable[Any]) -> str:
    normalized = []
    for value in values:
        if isinstance(value, torch.Tensor):
            normalized.append({"tensor_hash": tensor_hash(value)})
        elif isinstance(value, Mapping):
            normalized.append(
                {
                    str(key): (
                        {"tensor_hash": tensor_hash(item)}
                        if isinstance(item, torch.Tensor)
                        else item
                    )
                    for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                }
            )
        else:
            normalized.append(value)
    return stable_hash(normalized)


def tensor_stats(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    x = value.detach().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "sha256": tensor_hash(value),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def tensor_diff(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | bool]:
    if tuple(left.shape) != tuple(right.shape):
        return {"shape_equal": False}
    delta = right.detach().float() - left.detach().float()
    return {
        "shape_equal": True,
        "bitwise_equal": bool(torch.equal(left, right)),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "first_slice_mean_abs": float(delta[..., 0, :, :].abs().mean().item()),
        "rest_mean_abs": float(
            delta[..., 1:, :, :].abs().mean().item()
            if delta.shape[-3] > 1
            else 0.0
        ),
    }


@dataclass
class PreparedPlan:
    """The exact video-plan tensor and metadata seen by the video branch."""

    raw_z_s: torch.Tensor
    prepared_z_s: torch.Tensor
    prepared_z_s_timestep: torch.Tensor
    latent_cond_applied: bool
    latent_cond: torch.Tensor | None = None

    @property
    def raw_hash(self) -> str:
        return str(tensor_hash(self.raw_z_s))

    @property
    def prepared_hash(self) -> str:
        return str(tensor_hash(self.prepared_z_s))


def capture_prepared_plan(
    raw_z_s: torch.Tensor,
    prepared_z_s: torch.Tensor,
    prepared_z_s_timestep: torch.Tensor,
    *,
    frame_st_id: int,
    latent_cond: torch.Tensor | None,
    frame_zero_already_prepared: bool = False,
) -> PreparedPlan:
    """Capture a model input already produced by the native deployment loop."""

    raw = raw_z_s.detach().clone()
    prepared = prepared_z_s.detach().clone()
    cond = latent_cond.detach().clone() if latent_cond is not None else None
    if int(frame_st_id) > 0 and not torch.equal(raw, prepared):
        raise ConditionContractError(
            "nonzero frame_st_id prepared plan differs from raw Student plan; "
            "an undeclared overwrite occurred"
        )
    if int(frame_st_id) == 0 and not frame_zero_already_prepared:
        if cond is None:
            raise ConditionContractError(
                "frame_st_id=0 requires the native initial latent condition"
            )
        if not torch.equal(prepared[:, :, 0:1], cond[:, :, 0:1]):
            raise ConditionContractError(
                "frame_st_id=0 prepared first slice is not the native initial condition"
            )
    return PreparedPlan(
        raw_z_s=raw,
        prepared_z_s=prepared,
        prepared_z_s_timestep=prepared_z_s_timestep.detach().clone(),
        latent_cond_applied=cond is not None,
        latent_cond=cond,
    )


def prepare_plan_input(
    server: Any,
    raw_z_s: torch.Tensor,
    *,
    frame_st_id: int,
    init_latent: torch.Tensor | None = None,
    already_prepared: bool = False,
    latent_t: Any = 0,
    preserve_grad: bool = False,
) -> tuple[dict[str, Any], PreparedPlan]:
    """Prepare one plan exactly once, or inject a previously prepared plan.

    ``already_prepared=True`` intentionally calls the native helper with both
    condition arguments set to ``None`` and verifies that the helper is an
    identity on the supplied plan.  This protects Teacher/replay from the
    original hidden overwrite while retaining the native grid/timestep setup.
    """

    # Deployment/replay defaults to a detached provenance tensor.  The V0
    # differentiable Student video path explicitly opts into preserving the
    # graph from its isolated video-local adapter through the native plan
    # preparation; no existing inference caller changes behavior.
    raw = raw_z_s.clone() if preserve_grad else raw_z_s.detach().clone()
    if already_prepared:
        latent_cond = None
    else:
        latent_cond = (
            init_latent[:, :, 0:1].to(dtype=raw.dtype, device=raw.device)
            if int(frame_st_id) == 0
            else None
        )
    model_input = server._prepare_latent_input(
        raw.clone(),
        None,
        latent_t,
        latent_t,
        latent_cond,
        None,
        frame_st_id=int(frame_st_id),
    )
    latent_input = model_input["latent_res_lst"]
    prepared = (
        latent_input["noisy_latents"].clone()
        if preserve_grad
        else latent_input["noisy_latents"].detach().clone()
    )
    if already_prepared and not torch.equal(prepared, raw):
        raise ConditionContractError(
            "already_prepared plan was modified by native input preparation"
        )
    capture = capture_prepared_plan(
        raw,
        prepared,
        latent_input["timesteps"],
        frame_st_id=int(frame_st_id),
        latent_cond=latent_cond,
        frame_zero_already_prepared=already_prepared,
    )
    return model_input, capture


@dataclass
class ConditionFingerprint:
    schema_version: str = FINGERPRINT_SCHEMA_VERSION
    checkpoint_owner: str = "unknown"
    history_hash: str | None = None
    prompt_hash: str | None = None
    observation_hash: str | None = None
    model_action_history_hash: str | None = None
    plan_hash: str | None = None
    plan_timestep_hash: str | None = None
    action_noise_hash: str | None = None
    action_timestep_hash: str | None = None
    mask_hash: str | None = None
    normalization_hash: str | None = None
    frame_st_id: int = 0
    token_positions: tuple[int, ...] = ()
    cache_valid_length: int = 0
    sigma_start: float = 1.0
    sigma_end: float = 0.0
    declared_dtype_device_conversion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["token_positions"] = list(self.token_positions)
        return value

    def semantic_dict(self) -> dict[str, Any]:
        value = self.to_dict()
        value.pop("checkpoint_owner", None)
        value.pop("declared_dtype_device_conversion", None)
        return value


def build_condition_fingerprint(
    *,
    checkpoint_owner: str,
    history_hash: str,
    prompt_hash: str,
    observation_hash: str,
    model_action_history_hash: str,
    prepared_plan: torch.Tensor,
    prepared_plan_timestep: torch.Tensor,
    action_base_noise: torch.Tensor,
    action_timestep: torch.Tensor,
    mask: torch.Tensor,
    normalization_metadata: Mapping[str, Any],
    frame_st_id: int,
    token_positions: Iterable[int],
    cache_valid_length: int,
    sigma_start: float,
    sigma_end: float,
) -> ConditionFingerprint:
    return ConditionFingerprint(
        checkpoint_owner=checkpoint_owner,
        history_hash=history_hash,
        prompt_hash=prompt_hash,
        observation_hash=observation_hash,
        model_action_history_hash=model_action_history_hash,
        plan_hash=str(tensor_hash(prepared_plan)),
        plan_timestep_hash=str(tensor_hash(prepared_plan_timestep)),
        action_noise_hash=str(tensor_hash(action_base_noise)),
        action_timestep_hash=str(tensor_hash(action_timestep)),
        mask_hash=str(tensor_hash(mask)),
        normalization_hash=stable_hash(dict(normalization_metadata)),
        frame_st_id=int(frame_st_id),
        token_positions=tuple(int(x) for x in token_positions),
        cache_valid_length=int(cache_valid_length),
        sigma_start=float(sigma_start),
        sigma_end=float(sigma_end),
    )


def fingerprint_mismatches(
    left: ConditionFingerprint,
    right: ConditionFingerprint,
) -> dict[str, tuple[Any, Any]]:
    left_values = left.semantic_dict()
    right_values = right.semantic_dict()
    return {
        key: (left_values.get(key), right_values.get(key))
        for key in sorted(set(left_values) | set(right_values))
        if left_values.get(key) != right_values.get(key)
    }


def assert_fingerprint_match(
    expected: ConditionFingerprint,
    actual: ConditionFingerprint,
    *,
    label: str = "Teacher label",
) -> None:
    mismatches = fingerprint_mismatches(expected, actual)
    if mismatches:
        raise ConditionContractError(
            f"{label} rejected: semantic condition fingerprint mismatch: "
            f"{json.dumps(mismatches, sort_keys=True, default=str)}"
        )


def cache_valid_length(transformer: Any, cache_name: str) -> int:
    """Return the first attention block's semantic cache length."""

    blocks = getattr(transformer, "blocks", None)
    if not blocks:
        return 0
    attention = getattr(getattr(blocks[0], "attn1", None), "attn_caches", None)
    if not isinstance(attention, dict) or cache_name not in attention:
        return 0
    cache = attention[cache_name]
    if not isinstance(cache, dict) or cache.get("mask") is None:
        return 0
    return int(cache["mask"].sum().item())


def grid_token_positions(model_input: Mapping[str, Any], branch: str) -> tuple[int, ...]:
    grid = model_input[branch]["grid_id"].detach().reshape(-1).cpu().tolist()
    return tuple(int(item) for item in grid)


@dataclass
class CanonicalActionContextV2:
    schema_version: str = CANONICAL_SCHEMA_VERSION
    behavior_checkpoint_id: str = ""
    task: str = ""
    seed: int = 0
    episode_id: str = ""
    prefix_id: str = ""
    frame_st_id: int = 0
    prompt_token_ids: tuple[int, ...] = ()
    observation_history: str = ""
    model_format_action_history: str = ""
    history_hash: str = ""
    executed_physical_actions: Any = field(default_factory=list)
    raw_z_s: torch.Tensor | None = None
    prepared_z_s: torch.Tensor | None = None
    prepared_z_s_timestep: torch.Tensor | None = None
    latent_cond_applied: bool = False
    action_base_noise: torch.Tensor | None = None
    sigma_start: float = 1.0
    sigma_end: float = 0.0
    valid_action_mask: torch.Tensor | None = None
    normalization_metadata: dict[str, Any] = field(default_factory=dict)
    action_token_positions: tuple[int, ...] = ()
    cache_valid_length: int = 0
    tensor_hashes: dict[str, str | None] = field(default_factory=dict)
    student_fingerprint: ConditionFingerprint | None = None
    teacher_fingerprint: ConditionFingerprint | None = None

    def __post_init__(self) -> None:
        if self.schema_version != CANONICAL_SCHEMA_VERSION:
            raise ConditionContractError(
                f"unsupported canonical schema: {self.schema_version!r}"
            )
        if int(self.frame_st_id) < 0:
            raise ConditionContractError("frame_st_id must be non-negative")
        if int(self.frame_st_id) > 0 and self.latent_cond_applied:
            raise ConditionContractError(
                "nonzero context cannot declare episode initial latent application"
            )
        if self.prepared_z_s is None or self.action_base_noise is None:
            raise ConditionContractError(
                "canonical context must save prepared_z_s and action_base_noise"
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["prompt_token_ids"] = list(self.prompt_token_ids)
        if self.student_fingerprint is not None:
            value["student_fingerprint"] = self.student_fingerprint.to_dict()
        if self.teacher_fingerprint is not None:
            value["teacher_fingerprint"] = self.teacher_fingerprint.to_dict()
        return value


def build_canonical_context(
    *,
    behavior_checkpoint_id: str,
    task: str,
    seed: int,
    episode_id: str,
    prefix_id: str,
    frame_st_id: int,
    prompt_token_ids: Iterable[int],
    observation_history: str,
    model_format_action_history: str,
    history_hash: str = "",
    executed_physical_actions: Any,
    prepared_plan: PreparedPlan,
    action_base_noise: torch.Tensor,
    valid_action_mask: torch.Tensor,
    sigma_start: float,
    sigma_end: float,
    normalization_metadata: Mapping[str, Any],
) -> CanonicalActionContextV2:
    return CanonicalActionContextV2(
        behavior_checkpoint_id=str(behavior_checkpoint_id),
        task=str(task),
        seed=int(seed),
        episode_id=str(episode_id),
        prefix_id=str(prefix_id),
        frame_st_id=int(frame_st_id),
        prompt_token_ids=tuple(int(item) for item in prompt_token_ids),
        observation_history=str(observation_history),
        model_format_action_history=str(model_format_action_history),
        history_hash=str(history_hash),
        executed_physical_actions=executed_physical_actions,
        raw_z_s=prepared_plan.raw_z_s,
        prepared_z_s=prepared_plan.prepared_z_s,
        prepared_z_s_timestep=prepared_plan.prepared_z_s_timestep,
        latent_cond_applied=prepared_plan.latent_cond_applied,
        action_base_noise=action_base_noise.detach().clone(),
        sigma_start=float(sigma_start),
        sigma_end=float(sigma_end),
        valid_action_mask=valid_action_mask.detach().clone(),
        normalization_metadata=dict(normalization_metadata),
        tensor_hashes=tensor_hashes(
            {
                "raw_z_s": prepared_plan.raw_z_s,
                "prepared_z_s": prepared_plan.prepared_z_s,
                "prepared_z_s_timestep": prepared_plan.prepared_z_s_timestep,
                "action_base_noise": action_base_noise,
                "valid_action_mask": valid_action_mask,
            }
        ),
    )


def assert_canonical_plan_injection(
    canonical: CanonicalActionContextV2,
    injected_prepared_plan: torch.Tensor,
) -> None:
    if canonical.prepared_z_s is None or not torch.equal(
        canonical.prepared_z_s, injected_prepared_plan
    ):
        raise ConditionContractError("injected plan is not exact canonical prepared_z_s")


def assert_cache_semantics(
    student: ConditionFingerprint,
    teacher: ConditionFingerprint,
) -> None:
    """Check semantic chronology without requiring KV values to match."""

    for field_name in (
        "history_hash",
        "prompt_hash",
        "observation_hash",
        "model_action_history_hash",
        "frame_st_id",
        "token_positions",
        "cache_valid_length",
    ):
        if getattr(student, field_name) != getattr(teacher, field_name):
            raise ConditionContractError(
                f"cache semantic chronology mismatch in {field_name}: "
                f"student={getattr(student, field_name)!r} "
                f"teacher={getattr(teacher, field_name)!r}"
            )
    if student.checkpoint_owner == teacher.checkpoint_owner:
        raise ConditionContractError("Teacher/Student cache owners must be distinct")
