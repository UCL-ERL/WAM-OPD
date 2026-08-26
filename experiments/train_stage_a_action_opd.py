"""Config-driven per-task Stage-A action OPD trainer.

This is a deliberately narrow orchestration layer over the existing native D3
Student-occupancy replay and action-forward code.  It does not own Teacher
labeling, simulator rollouts, or a second model implementation.  The only
optimized parameters are a rank-32, zero-up residual adapter around
``action_proj_out``.

Invocation::

    python -m experiments.train_stage_a_action_opd --config CONFIG.json

The JSON config is fail-closed and records the fixed Stage-A contract explicitly::

    {
      "schema": "flashwam_stage_a_action_opd_v1",
      "task_id": "open_microwave",
      "task_config": "demo_clean",
      "artifacts": ["/absolute/label0.pt", "/absolute/label1.pt"],
      "output": "/absolute/stage_a_delta.pt",
      "adapter": {
        "kind": "action_output_residual",
        "target": "action_proj_out",
        "rank": 32,
        "initialization": "zero_up"
      },
      "loss": {
        "target_key": "teacher_bridge_model_action",
        "target_kind": "teacher_bridge_endpoint_on_student_occupancy",
        "retention_weight": 0.0
      },
      "training": {
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "gradient_clip": 2.0,
        "seed": 20260816,
        "device": "cuda"
      }
    }

Teacher labels may contain the full bridge endpoint.  Loss is computed only on
the intersection of ``valid_action_mask`` and the physical
``executed_action_mask``.  The terminal low-level action remains included;
every later action position is excluded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from experiments.action_output_adapter import ActionOutputResidualAdapter
from experiments.waopd_native_action_opd import (
    NativeActionEndpointTrainer,
    load_labels,
)


CONFIG_SCHEMA = "flashwam_stage_a_action_opd_v1"
EXPECTED_TASK_CONFIG = "demo_clean"
TARGET_KEY = "teacher_bridge_model_action"
TEACHER_TARGET = "teacher_bridge_endpoint_on_student_occupancy"
ADAPTER_KIND = "action_output_residual"
ADAPTER_TARGET = "action_proj_out"
ADAPTER_RANK = 32
ADAPTER_INITIALIZATION = "zero_up"
EPOCHS = 3
TEACHER_ARTIFACT_KIND = "waopd_d3_student_occupancy_teacher_bridge_label"


@dataclass(frozen=True)
class StageAConfig:
    config_path: Path
    task_id: str
    task_config: str
    artifacts: tuple[Path, ...]
    output: Path
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    seed: int
    device: str


def _require_object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    where: str,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing or unknown:
        raise ValueError(
            f"{where} keys mismatch: missing={missing} unknown={unknown}"
        )


def _resolve_path(raw: Any, *, base: Path, where: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{where} must be a non-empty path string")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_stage_a_config(path: str | Path) -> StageAConfig:
    """Load one strict per-task Stage-A config and resolve its paths."""

    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text())
    payload = _require_object(payload, "config")
    _require_exact_keys(
        payload,
        required={
            "schema",
            "task_id",
            "task_config",
            "artifacts",
            "output",
            "adapter",
            "loss",
            "training",
        },
        where="config",
    )
    if payload["schema"] != CONFIG_SCHEMA:
        raise ValueError(f"schema must be {CONFIG_SCHEMA}")
    task_id = payload["task_id"]
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be a non-empty string")
    if payload["task_config"] != EXPECTED_TASK_CONFIG:
        raise ValueError(
            f"task_config must be {EXPECTED_TASK_CONFIG!r}, got "
            f"{payload['task_config']!r}"
        )

    adapter = _require_object(payload["adapter"], "adapter")
    _require_exact_keys(
        adapter,
        required={"kind", "target", "rank", "initialization"},
        where="adapter",
    )
    expected_adapter = {
        "kind": ADAPTER_KIND,
        "target": ADAPTER_TARGET,
        "rank": ADAPTER_RANK,
        "initialization": ADAPTER_INITIALIZATION,
    }
    if dict(adapter) != expected_adapter:
        raise ValueError(
            "adapter must be the fixed Stage-A rank-32 zero-up "
            f"action_proj_out adapter: expected={expected_adapter} got={dict(adapter)}"
        )

    loss = _require_object(payload["loss"], "loss")
    _require_exact_keys(
        loss,
        required={"target_key", "target_kind", "retention_weight"},
        where="loss",
    )
    if loss["target_key"] != TARGET_KEY:
        raise ValueError(f"loss.target_key must be {TARGET_KEY}")
    if loss["target_kind"] != TEACHER_TARGET:
        raise ValueError(f"loss.target_kind must be {TEACHER_TARGET}")
    if float(loss["retention_weight"]) != 0.0:
        raise ValueError("Stage A forbids retention; retention_weight must be 0.0")

    training = _require_object(payload["training"], "training")
    _require_exact_keys(
        training,
        required={
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip",
            "seed",
            "device",
        },
        where="training",
    )
    if isinstance(training["epochs"], bool) or int(training["epochs"]) != EPOCHS:
        raise ValueError(f"training.epochs must be fixed at {EPOCHS}")
    batch_size = training["batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("training.batch_size must be a positive integer")
    learning_rate = float(training["learning_rate"])
    gradient_clip = float(training["gradient_clip"])
    weight_decay = float(training["weight_decay"])
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("training.learning_rate must be finite and positive")
    if not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError("training.gradient_clip must be finite and positive")
    if weight_decay != 0.0:
        raise ValueError("Stage A uses no parameter anchor; weight_decay must be 0.0")
    seed = training["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("training.seed must be an integer")
    device = training["device"]
    if not isinstance(device, str) or not device.strip():
        raise ValueError("training.device must be a non-empty string")

    raw_artifacts = payload["artifacts"]
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("artifacts must be a non-empty list")
    base = config_path.parent
    artifacts = tuple(
        _resolve_path(item, base=base, where=f"artifacts[{index}]")
        for index, item in enumerate(raw_artifacts)
    )
    if len(set(artifacts)) != len(artifacts):
        raise ValueError("artifacts contains duplicate paths")
    missing_artifacts = [str(item) for item in artifacts if not item.is_file()]
    if missing_artifacts:
        raise FileNotFoundError(f"missing Stage-A artifacts: {missing_artifacts}")
    output = _resolve_path(payload["output"], base=base, where="output")
    if output in set(artifacts):
        raise ValueError("output must not overwrite an input artifact")
    if output.suffix != ".pt":
        raise ValueError("output must have a .pt suffix")

    return StageAConfig(
        config_path=config_path,
        task_id=task_id,
        task_config=EXPECTED_TASK_CONFIG,
        artifacts=artifacts,
        output=output,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        seed=seed,
        device=device,
    )


def terminal_execution_mask(
    replay: Mapping[str, Any],
    *,
    frame_count: int,
    horizon: int,
) -> torch.Tensor:
    """Validate and return the exact physical-action prefix for one macro."""

    raw_mask = replay.get("executed_action_mask")
    if not isinstance(raw_mask, (list, tuple, torch.Tensor)):
        raise ValueError("executed_action_mask is required")
    mask = torch.as_tensor(raw_mask)
    if mask.dtype != torch.bool:
        raise ValueError("executed_action_mask must have bool dtype")
    if tuple(mask.shape) != (frame_count, horizon):
        raise ValueError(
            "executed_action_mask shape must match action time axes: "
            f"expected={(frame_count, horizon)} got={tuple(mask.shape)}"
        )

    start_frame = replay.get("start_frame")
    if isinstance(start_frame, bool) or not isinstance(start_frame, int):
        raise ValueError("start_frame must be an integer")
    if start_frame < 0 or start_frame >= frame_count:
        raise ValueError(
            f"start_frame must be in [0, {frame_count}), got {start_frame}"
        )
    terminal_reached = replay.get("terminal_reached")
    if not isinstance(terminal_reached, bool):
        raise ValueError("terminal_reached must be a bool")
    horizon_reached = replay.get("horizon_reached", False)
    if not isinstance(horizon_reached, bool):
        raise ValueError("horizon_reached must be a bool")
    terminal_position = replay.get("terminal_action_position")
    action_steps = replay.get("action_steps")
    if isinstance(action_steps, bool) or not isinstance(action_steps, int):
        raise ValueError("action_steps must be an integer")

    flat_expected = torch.zeros(frame_count * horizon, dtype=torch.bool)
    first_executable = start_frame * horizon
    if terminal_reached:
        if (
            not isinstance(terminal_position, (list, tuple))
            or len(terminal_position) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in terminal_position)
        ):
            raise ValueError(
                "terminal_reached requires integer terminal_action_position [frame, horizon]"
            )
        terminal_frame, terminal_horizon = terminal_position
        if not (start_frame <= terminal_frame < frame_count):
            raise ValueError("terminal_action_position frame is outside the executable prefix")
        if not (0 <= terminal_horizon < horizon):
            raise ValueError("terminal_action_position horizon is out of range")
        terminal_flat = terminal_frame * horizon + terminal_horizon
        # Include the first action which triggers native success.
        flat_expected[first_executable : terminal_flat + 1] = True
    elif horizon_reached:
        if terminal_position is not None:
            raise ValueError(
                "terminal_action_position must be null when terminal_reached is false"
            )
        executable_positions = frame_count * horizon - first_executable
        if action_steps <= 0 or action_steps > executable_positions:
            raise ValueError(
                "horizon-limited action_steps must select a nonempty executable prefix"
            )
        flat_expected[first_executable : first_executable + action_steps] = True
    else:
        if terminal_position is not None:
            raise ValueError(
                "terminal_action_position must be null when terminal_reached is false"
            )
        flat_expected[first_executable:] = True

    expected = flat_expected.view(frame_count, horizon)
    if not torch.equal(mask.cpu(), expected):
        raise ValueError(
            "executed_action_mask is inconsistent with start/terminal metadata"
        )
    if action_steps != int(mask.sum().item()):
        raise ValueError(
            "action_steps does not equal executed_action_mask.sum(): "
            f"{action_steps} != {int(mask.sum().item())}"
        )
    if not bool(mask.any()):
        raise ValueError(
            "artifact contains no executed action; it is a post-success context"
        )
    return mask


def stage_a_loss_mask(
    replay: Mapping[str, Any],
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Intersect model-valid channels with the physically executed prefix."""

    valid = replay.get("valid_action_mask")
    if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool:
        raise ValueError("valid_action_mask must be a bool tensor")
    if tuple(valid.shape) != tuple(reference.shape):
        raise ValueError(
            "valid_action_mask shape differs from endpoint: "
            f"{tuple(valid.shape)} != {tuple(reference.shape)}"
        )
    if reference.ndim != 5 or reference.shape[0] != 1 or reference.shape[-1] != 1:
        raise ValueError(
            "Stage-A endpoint must have shape [1, channels, frames, horizon, 1]"
        )
    executed = terminal_execution_mask(
        replay,
        frame_count=int(reference.shape[2]),
        horizon=int(reference.shape[3]),
    ).to(device=reference.device)
    valid = valid.to(device=reference.device)
    valid_by_position = valid.any(dim=(0, 1, 4))
    invalid_executed = executed & ~valid_by_position
    if bool(invalid_executed.any()):
        raise ValueError(
            "executed_action_mask selects a position excluded by valid_action_mask"
        )
    effective = valid & executed[None, None, :, :, None]
    if not bool(effective.any()):
        raise ValueError("effective Stage-A action loss mask is empty")
    return effective


def stage_a_endpoint_loss(
    endpoint: torch.Tensor,
    replay: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Huber/RMSE to the full Teacher bridge endpoint on executed positions."""

    target = replay.get(TARGET_KEY)
    if not isinstance(target, torch.Tensor):
        raise ValueError(f"missing Teacher endpoint tensor: {TARGET_KEY}")
    if tuple(target.shape) != tuple(endpoint.shape):
        raise ValueError(
            f"{TARGET_KEY} shape differs from endpoint: "
            f"{tuple(target.shape)} != {tuple(endpoint.shape)}"
        )
    if not bool(torch.isfinite(target).all()):
        raise ValueError(f"{TARGET_KEY} contains NaN or Inf")
    mask = stage_a_loss_mask(replay, reference=endpoint)
    target = target.to(device=endpoint.device, dtype=endpoint.dtype)
    delta = endpoint.float()[mask] - target.float()[mask]
    loss = F.smooth_l1_loss(
        endpoint.float()[mask],
        target.float()[mask],
        beta=1e-3,
    )
    rmse = delta.square().mean().sqrt()
    return loss, rmse, int(mask.sum().item())


def validate_stage_a_replays(
    replays: Sequence[Mapping[str, Any]],
    config: StageAConfig,
) -> None:
    """Apply Stage-A constraints after native schema-v4 label loading."""

    if not replays:
        raise ValueError("Stage A requires at least one replay")
    errors: list[str] = []
    terminal_rows_by_collection: dict[str, list[tuple[int, tuple[int, int]]]] = {}
    frames_by_collection: dict[str, list[int]] = {}
    terminal_signature_by_context: dict[tuple[str, int], set[tuple[Any, ...]]] = {}

    for index, replay in enumerate(replays):
        prefix = f"artifact[{index}]"
        if str(replay.get("task_id", "")) != config.task_id:
            errors.append(
                f"{prefix} task_id={replay.get('task_id')!r}, expected {config.task_id!r}"
            )
        if replay.get("task_config") != EXPECTED_TASK_CONFIG:
            errors.append(
                f"{prefix} task_config={replay.get('task_config')!r}, "
                f"expected {EXPECTED_TASK_CONFIG!r}"
            )
        if replay.get("artifact_kind") != TEACHER_ARTIFACT_KIND:
            errors.append(
                f"{prefix} artifact_kind must be {TEACHER_ARTIFACT_KIND}"
            )
        if replay.get("teacher_target_kind") != TEACHER_TARGET:
            errors.append(
                f"{prefix} teacher_target_kind must be {TEACHER_TARGET}"
            )
        if replay.get("opd_target_key", TARGET_KEY) != TARGET_KEY:
            errors.append(f"{prefix} opd_target_key must be {TARGET_KEY}")
        if not isinstance(replay.get(TARGET_KEY), torch.Tensor):
            errors.append(f"{prefix} missing {TARGET_KEY}")
        retention_fields = sorted(
            str(key) for key in replay if str(key).startswith("retention")
        )
        if retention_fields:
            errors.append(f"{prefix} contains retention fields: {retention_fields}")

        collection_id = replay.get("collection_id")
        if not isinstance(collection_id, str) or not collection_id.strip():
            errors.append(f"{prefix} collection_id must be a non-empty string")
            collection_id = f"invalid-{index}"
        frame_st_id = replay.get("frame_st_id")
        if isinstance(frame_st_id, bool) or not isinstance(frame_st_id, int):
            errors.append(f"{prefix} frame_st_id must be an integer")
            frame_st_id = -1
        frames_by_collection.setdefault(collection_id, []).append(frame_st_id)

        target = replay.get(TARGET_KEY)
        if isinstance(target, torch.Tensor):
            try:
                physical_mask = terminal_execution_mask(
                    replay,
                    frame_count=int(target.shape[2]),
                    horizon=int(target.shape[3]),
                )
                _ = stage_a_loss_mask(replay, reference=target)
                terminal = bool(replay.get("terminal_reached"))
                signature = (
                    terminal,
                    tuple(replay.get("terminal_action_position") or ()),
                    int(replay.get("start_frame", -1)),
                    int(replay.get("action_steps", -1)),
                    tuple(physical_mask.flatten().tolist()),
                )
                terminal_signature_by_context.setdefault(
                    (collection_id, frame_st_id), set()
                ).add(signature)
                if terminal:
                    position = tuple(int(item) for item in replay["terminal_action_position"])
                    terminal_rows_by_collection.setdefault(collection_id, []).append(
                        (frame_st_id, position)
                    )
            except (IndexError, TypeError, ValueError) as error:
                errors.append(f"{prefix} terminal mask invalid: {error}")

    for (collection_id, frame_st_id), signatures in sorted(
        terminal_signature_by_context.items()
    ):
        if len(signatures) != 1:
            errors.append(
                "terminal metadata differs across labels for the same context: "
                f"collection={collection_id} frame_st_id={frame_st_id}"
            )
    for collection_id, terminal_rows in sorted(terminal_rows_by_collection.items()):
        terminal_frames = {frame for frame, _position in terminal_rows}
        terminal_positions = {position for _frame, position in terminal_rows}
        if len(terminal_frames) != 1 or len(terminal_positions) != 1:
            errors.append(
                f"collection {collection_id} has inconsistent terminal contexts"
            )
            continue
        terminal_frame = next(iter(terminal_frames))
        later_frames = [
            frame
            for frame in frames_by_collection.get(collection_id, ())
            if frame > terminal_frame
        ]
        if later_frames:
            errors.append(
                f"collection {collection_id} contains post-success contexts: "
                f"terminal_frame={terminal_frame} later={sorted(set(later_frames))}"
            )

    if errors:
        raise ValueError("invalid Stage-A replay set: " + "; ".join(errors))


def epoch_batches(
    replays: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    seed: int,
    epochs: int = EPOCHS,
) -> Iterable[tuple[int, int, list[Mapping[str, Any]]]]:
    """Yield deterministic shuffled batches, covering every replay each epoch."""

    if epochs != EPOCHS:
        raise ValueError(f"Stage A requires exactly {EPOCHS} epochs")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not replays:
        raise ValueError("replays must be non-empty")
    for epoch in range(epochs):
        indices = list(range(len(replays)))
        random.Random(seed + epoch).shuffle(indices)
        for batch_index, start in enumerate(range(0, len(indices), batch_size)):
            yield (
                epoch,
                batch_index,
                [replays[index] for index in indices[start : start + batch_size]],
            )


@torch.no_grad()
def evaluate_stage_a(
    trainer: NativeActionEndpointTrainer,
    replays: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    # Gradients through the adapter do not require train mode.  Keep the
    # frozen deployment model in eval mode so dropout or any other
    # mode-dependent layer cannot change the Student condition.
    trainer.server.transformer.eval()
    losses: list[float] = []
    rmses: list[float] = []
    effective_elements = 0
    for replay in replays:
        endpoint, _state = trainer.endpoint(dict(replay), require_grad=False)
        loss, rmse, count = stage_a_endpoint_loss(endpoint, replay)
        losses.append(float(loss.item()))
        rmses.append(float(rmse.item()))
        effective_elements += count
    return {
        "sample_count": len(replays),
        "effective_element_count": effective_elements,
        "huber_mean": sum(losses) / len(losses),
        "huber_max": max(losses),
        "rmse_mean": sum(rmses) / len(rmses),
        "rmse_max": max(rmses),
    }


def train(config: StageAConfig) -> dict[str, Any]:
    """Run the fixed Stage-A vertical slice and save one deployable delta."""

    started = time.monotonic()
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    replays = load_labels(config.artifacts)
    validate_stage_a_replays(replays, config)
    config.output.parent.mkdir(parents=True, exist_ok=True)

    trainer = NativeActionEndpointTrainer(
        student=Path(str(replays[0]["student_checkpoint"])),
        device=config.device,
        save_root=config.output.parent / f"{config.output.stem}_native_runtime",
        enable_offload=True,
        adapter_rank=ADAPTER_RANK,
    )
    adapter = trainer.server.transformer.action_proj_out
    if not isinstance(adapter, ActionOutputResidualAdapter):
        raise RuntimeError("native trainer did not attach an action output adapter")
    trainable = trainer.trainable
    trainable_names = [name for name, _parameter in trainable]
    if not trainable or any(
        not name.startswith("action_proj_out.") or ".base." in name
        for name in trainable_names
    ):
        raise RuntimeError(
            "Stage A exposed parameters outside the action_proj_out adapter: "
            f"{trainable_names}"
        )
    if adapter.rank != ADAPTER_RANK or adapter.initialization != ADAPTER_INITIALIZATION:
        raise RuntimeError("attached action adapter does not match Stage-A contract")
    if bool(torch.count_nonzero(adapter.up.weight.detach()).item()):
        raise RuntimeError("Stage-A zero-up adapter is not exactly zero initialized")
    trainable_parameter_count = sum(
        parameter.numel() for _name, parameter in trainable
    )
    if trainable_parameter_count > 5_000_000:
        raise RuntimeError("Stage-A action output adapter exceeds 5M parameters")

    parity_indices = sorted(
        {
            0,
            len(replays) // 3,
            (2 * len(replays)) // 3,
            len(replays) - 1,
        }
    )
    parity_replays = [replays[index] for index in parity_indices]
    trainer.server.transformer.eval()
    zero_init_parity = trainer.zero_init_parity(
        parity_replays,
        max_rows=len(parity_replays),
        tolerance=1e-3,
    )
    if not bool(zero_init_parity["pass"]):
        trainer.close()
        raise RuntimeError(
            "Stage-A zero-init endpoint parity failed before optimizer creation: "
            f"{zero_init_parity}"
        )
    initial_adapter_state = {
        name: tensor.clone() for name, tensor in trainer.adapter_state().items()
    }
    # Offline loss is only a feedback signal, never checkpoint selection.
    # Evaluate the same horizon-spanning representatives before and after
    # training instead of adding two expensive full replay passes.
    initial_eval = evaluate_stage_a(trainer, parity_replays)

    optimizer = torch.optim.AdamW(
        [parameter for _name, parameter in trainable],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    epoch_metrics: list[dict[str, Any]] = []
    optimizer_steps = 0
    for epoch in range(EPOCHS):
        if trainer.server.transformer.training:
            raise RuntimeError("Stage-A transformer must remain in eval mode")
        sample_losses: list[float] = []
        sample_rmses: list[float] = []
        grad_norms: list[float] = []
        sample_count = 0
        for yielded_epoch, _batch_index, batch in epoch_batches(
            replays,
            batch_size=config.batch_size,
            seed=config.seed,
            epochs=EPOCHS,
        ):
            if yielded_epoch != epoch:
                continue
            optimizer.zero_grad(set_to_none=True)
            # Each sample contributes 1 / actual_batch_size, including the
            # final short batch.  This is true batch-mean gradient
            # accumulation, not sequential optimizer updates.
            batch_denominator = float(len(batch))
            for replay in batch:
                endpoint, _state = trainer.endpoint(
                    dict(replay),
                    require_grad=True,
                )
                loss, rmse, _count = stage_a_endpoint_loss(endpoint, replay)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"non-finite Stage-A loss at epoch {epoch + 1}"
                    )
                (loss / batch_denominator).backward()
                sample_losses.append(float(loss.detach().item()))
                sample_rmses.append(float(rmse.detach().item()))
                sample_count += 1
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _name, parameter in trainable],
                config.gradient_clip,
            )
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(
                    f"non-finite Stage-A gradient at epoch {epoch + 1}"
                )
            optimizer.step()
            optimizer_steps += 1
            grad_norms.append(float(grad_norm.detach().item()))
        if sample_count != len(replays):
            raise RuntimeError(
                f"epoch {epoch + 1} consumed {sample_count} replays, expected {len(replays)}"
            )
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "sample_count": sample_count,
                "huber_mean": sum(sample_losses) / sample_count,
                "rmse_mean": sum(sample_rmses) / sample_count,
                "grad_norm_mean": sum(grad_norms) / len(grad_norms),
                "optimizer_steps": len(grad_norms),
            }
        )

    final_eval = evaluate_stage_a(trainer, parity_replays)
    state_dict = trainer.adapter_state()
    changed_parameter_names = sorted(
        name
        for name, tensor in state_dict.items()
        if not torch.equal(tensor, initial_adapter_state[name])
    )
    if not changed_parameter_names:
        raise RuntimeError("Stage-A completed without changing any adapter parameter")
    checkpoint = {
        "format": "flashwam_action_output_adapter_v1",
        "base_checkpoint": str(replays[0]["student_checkpoint"]),
        "trainable_scope": "action_output_lora",
        "state_dict": state_dict,
        # The prototype deployment server consumes ``state_dict`` while the
        # native Student-only runtime consumes ``adapter_state_dict``.  Keep
        # one checkpoint compatible with both evaluators.
        "adapter_state_dict": state_dict,
        "adapter": {
            "kind": ADAPTER_KIND,
            "target": ADAPTER_TARGET,
            "rank": ADAPTER_RANK,
            "input_features": adapter.base.in_features,
            "output_features": adapter.base.out_features,
            "initialization": ADAPTER_INITIALIZATION,
            "zero_initialized_up": True,
        },
        "training": {
            "schema": CONFIG_SCHEMA,
            "config": str(config.config_path),
            "task_id": config.task_id,
            "task_config": config.task_config,
            "epochs": EPOCHS,
            "batch_size": config.batch_size,
            "gradient_accumulation": "mean_over_actual_batch_size",
            "optimizer_steps": optimizer_steps,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "gradient_clip": config.gradient_clip,
            "seed": config.seed,
            "target_key": TARGET_KEY,
            "target_kind": TEACHER_TARGET,
            "retention_weight": 0.0,
            "artifacts": [str(item) for item in config.artifacts],
            "terminal_mask": "valid_action_mask AND executed_action_mask",
            "zero_init_parity": zero_init_parity,
            "offline_eval_scope": {
                "kind": "horizon_spanning_representatives",
                "artifact_indices": parity_indices,
            },
        },
    }
    torch.save(checkpoint, config.output)
    reloaded = torch.load(config.output, map_location="cpu", weights_only=False)
    if reloaded.get("format") != checkpoint["format"]:
        raise RuntimeError("checkpoint reload format mismatch")
    for name, tensor in state_dict.items():
        if not torch.equal(tensor, reloaded["state_dict"][name]):
            raise RuntimeError(f"checkpoint reload mismatch: {name}")
        if not torch.equal(tensor, reloaded["adapter_state_dict"][name]):
            raise RuntimeError(f"native checkpoint reload mismatch: {name}")

    wall_seconds = time.monotonic() - started
    metrics = {
        "schema": "flashwam_stage_a_action_opd_metrics_v1",
        "checkpoint": str(config.output),
        "checkpoint_reload_verified": True,
        "task_id": config.task_id,
        "task_config": config.task_config,
        "artifact_count": len(replays),
        "epochs": EPOCHS,
        "batch_size": config.batch_size,
        "optimizer_steps": optimizer_steps,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_parameter_names": trainable_names,
        "changed_parameter_names": changed_parameter_names,
        "zero_init_parity": zero_init_parity,
        "offline_eval_artifact_indices": parity_indices,
        "initial_eval": initial_eval,
        "epoch_metrics": epoch_metrics,
        "final_eval": final_eval,
        "wall_seconds": wall_seconds,
        "cuda_peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else 0.0
        ),
    }
    metrics_path = config.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    trainer.close()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_stage_a_config(args.config))


if __name__ == "__main__":
    main()
