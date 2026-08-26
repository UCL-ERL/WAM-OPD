"""Iterative environment-on-policy distillation for RoboTwin.

The Student is the only policy that controls the simulator.  The coherent TT
path labels each Student-visited history with ``z_T`` and ``a_T(h_S, z_T)``,
then trains a shared JointLoRA with native random Teacher bridges, an
adapter-only EMA target, and multiple AdamW steps over one bounded inner
epoch.  Legacy endpoint and fixed-sigma objectives remain explicit separate
code paths for artifact compatibility.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import socket
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from experiments.dual_mode_lora import load_dual_mode_lora_checkpoint
from experiments.opd_task_specs import (
    TASK_SPECS,
    require_training_task_config,
    resolve_task_chunks,
)
from experiments.train_joint_teacher_trajectory_opd import (
    _outcome,
    _setup_task_with_locked_prompt,
    _worker_progress,
)
from experiments.train_video_trajectory_opd import (
    SCHEMA as TRAJECTORY_SCHEMA,
    NativeStudentVideoLabelRuntime,
    action_execution_mask,
    build_trajectory_artifact,
    capture_student_context,
    materialize_context,
    video_execution_mask,
)
from experiments.waopd_native_closed_loop_runner import (
    ActionSolve,
    LockedNoiseBank,
    NativeClosedLoopError,
    run_live_episode,
    tensor_hash,
)
from experiments.waopd_v0_video_opd import (
    NativeV0VideoRuntime,
    action_velocity_mse_loss,
    video_consistency_map,
)


CHECKPOINT_SCHEMA = "waopd_iterative_on_policy_flow_checkpoint_v1"
SUMMARY_SCHEMA = "waopd_iterative_on_policy_flow_summary_v1"
DUAL_CHECKPOINT_SCHEMA = "waopd_iterative_on_policy_dual_bank_checkpoint_v1"
DUAL_SUMMARY_SCHEMA = "waopd_iterative_on_policy_dual_bank_summary_v1"
COLLECT_SUMMARY_SCHEMA = "waopd_iterative_on_policy_collect_summary_v1"
ROLLOUT_BUNDLE_SCHEMA = "waopd_iterative_on_policy_dual_rollout_bundle_v1"
BRANCH_SUMMARY_SCHEMA = "waopd_iterative_on_policy_dual_branch_summary_v1"
TRAJECTORY_UPDATE_SUMMARY_SCHEMA = (
    "waopd_iterative_on_policy_trajectory_update_summary_v1"
)
LOSS_REDUCTION_MEAN_ALL = "mean_all_labels"
LOSS_REDUCTION_MEAN_TRAJECTORIES = "mean_trajectories_mean_labels"
LOSS_REDUCTION_MEAN_TASKS = "mean_tasks_mean_trajectories_mean_labels"
OBJECTIVE_ENDPOINT = "endpoint"
OBJECTIVE_MULTI_SIGMA_X0 = "multi_sigma_x0"
OBJECTIVE_COHERENT_TT_CONSISTENCY = "coherent_tt_consistency"
COHERENT_TT_VARIANT_BASELINE = "baseline"
COHERENT_TT_VARIANT_SUCCESS_PATH_V1 = "success_path_v1"
SUCCESS_PATH_PROGRESS_SCHEMA = "waopd_success_path_v1_resume_progress_v1"
SUCCESS_PATH_RESUME_CONTRACT_SCHEMA = (
    "waopd_success_path_v1_exact_resume_contract_v2"
)
SUCCESS_PATH_EXACT_IDENTITY_SCHEMA = (
    "waopd_success_path_v1_exact_input_identity_v1"
)
SUCCESS_PATH_COMMIT_SCHEMA = "waopd_success_path_v1_atomic_commit_v1"
SUCCESS_PATH_WRITER_LOCK_SCHEMA = "waopd_success_path_v1_writer_lock_v1"
SUCCESS_PATH_FINALIZATION_SCHEMA = "waopd_success_path_v1_finalization_v1"
FUNCTIONAL_ACCEPTANCE_HELDOUT_NONREGRESSION = (
    "heldout_modality_nonregression_p95_v1"
)
SHARED_TRANSFORMER_BLOCK_COUNT = 30


def sample_native_consistency_indices(
    frame_count: int,
    *,
    generator: torch.Generator,
    num_train_timesteps: int = 1000,
    stride: int = 500,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample Flash-WAM-style per-frame native start/end grid indices."""

    if int(frame_count) <= 0:
        raise ValueError("frame_count must be positive")
    if int(num_train_timesteps) <= 0:
        raise ValueError("num_train_timesteps must be positive")
    if int(stride) <= 0:
        raise ValueError("consistency stride must be positive")
    start = torch.randint(
        low=0,
        high=int(num_train_timesteps),
        size=(int(frame_count),),
        generator=generator,
        dtype=torch.long,
    )
    end = (start + int(stride)).clamp(max=int(num_train_timesteps) - 1)
    return start, end


def _frame_sigma_view(sigma: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    values = sigma.detach().to(device=reference.device, dtype=reference.dtype)
    if values.ndim != 1 or int(values.numel()) != int(reference.shape[2]):
        raise ValueError("per-frame sigma must be a 1-D tensor matching state frames")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("per-frame sigma contains nonfinite values")
    if bool(((values < 0.0) | (values > 1.0)).any().item()):
        raise ValueError("per-frame sigma must lie in [0, 1]")
    return values.reshape(1, 1, int(values.numel()), *([1] * (reference.ndim - 3)))


def teacher_euler_bridge(
    start_state: torch.Tensor,
    teacher_velocity: torch.Tensor,
    *,
    sigma_start: torch.Tensor,
    sigma_end: torch.Tensor,
) -> torch.Tensor:
    """Apply one frozen-Teacher Euler jump on a per-frame flow schedule."""

    if tuple(start_state.shape) != tuple(teacher_velocity.shape):
        raise ValueError("Teacher Euler state/velocity shapes differ")
    if start_state.ndim < 3:
        raise ValueError("Teacher Euler state must expose a frame dimension")
    start_view = _frame_sigma_view(sigma_start, start_state)
    end_view = _frame_sigma_view(sigma_end, start_state)
    if bool((end_view > start_view).any().item()):
        raise ValueError("Teacher Euler bridge must move toward lower noise")
    return start_state + teacher_velocity * (end_view - start_view)


class LoRAEMAState:
    """Adapter-only EMA target with transactional in-place model swapping."""

    def __init__(self, state: Mapping[str, torch.Tensor], *, decay: float) -> None:
        if not math.isfinite(float(decay)) or not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must be finite in [0, 1)")
        if not state:
            raise ValueError("EMA state must not be empty")
        self.decay = float(decay)
        self._state = {
            str(name): value.detach().clone()
            for name, value in state.items()
        }
        if len(self._state) != len(state):
            raise ValueError("EMA state keys must be unique strings")
        self.committed_updates = 0

    @classmethod
    def from_online(
        cls, online_state: Mapping[str, torch.Tensor], *, decay: float
    ) -> "LoRAEMAState":
        return cls(online_state, decay=float(decay))

    def _validate_live_state(self, live_state: Mapping[str, torch.Tensor]) -> None:
        if set(live_state) != set(self._state):
            raise ValueError("EMA/live adapter parameter keys differ")
        for name, target in self._state.items():
            live = live_state[name]
            if tuple(live.shape) != tuple(target.shape):
                raise ValueError(f"EMA/live shape differs for {name}")
            if live.dtype != target.dtype or live.device != target.device:
                raise ValueError(f"EMA/live dtype or device differs for {name}")

    def target_state(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().clone()
            for name, value in self._state.items()
        }

    def after_committed_step(
        self,
        online_state: Mapping[str, torch.Tensor],
        *,
        committed: bool,
    ) -> None:
        self._validate_live_state(online_state)
        if not bool(committed):
            return
        with torch.no_grad():
            for name, target in self._state.items():
                target.mul_(self.decay).add_(
                    online_state[name].detach(), alpha=1.0 - self.decay
                )
        self.committed_updates += 1

    @contextmanager
    def use_target(self, live_state: Mapping[str, torch.Tensor]):  # type: ignore[no-untyped-def]
        self._validate_live_state(live_state)
        online = {
            name: value.detach().clone()
            for name, value in live_state.items()
        }
        try:
            with torch.no_grad():
                for name, value in live_state.items():
                    value.copy_(self._state[name])
            yield
        finally:
            with torch.no_grad():
                for name, value in live_state.items():
                    value.copy_(online[name])


def student_action_on_detached_teacher_plan(
    action_forward: Any,
    *,
    context: Mapping[str, Any],
    teacher_plan: torch.Tensor,
    teacher_action: torch.Tensor,
    sigma: float,
    require_grad: bool,
) -> Any:
    """Call the Student action path on the exact stop-gradient Teacher plan."""

    return action_forward(
        context,
        teacher_plan.detach(),
        teacher_action,
        sigma=float(sigma),
        require_grad=bool(require_grad),
    )


def _checkpoint_objective(checkpoint: Mapping[str, Any]) -> str:
    """Read objective identity while preserving legacy endpoint checkpoints."""

    value = checkpoint.get("objective")
    if value is None:
        loss = checkpoint.get("loss")
        if isinstance(loss, Mapping):
            value = loss.get("objective")
    objective = OBJECTIVE_ENDPOINT if value is None else str(value)
    if objective not in {
        OBJECTIVE_ENDPOINT,
        OBJECTIVE_MULTI_SIGMA_X0,
        OBJECTIVE_COHERENT_TT_CONSISTENCY,
    }:
        raise ValueError(f"initial_checkpoint has unknown objective={objective!r}")
    return objective


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone().cpu()


def _atomic_temp_prefix(path: Path) -> str:
    return f".{path.name}.atomic-"


def _atomic_writer_temp_prefix(path: Path) -> str:
    return f"{_atomic_temp_prefix(path)}{os.getpid()}-"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Publish bytes through a same-directory fsynced atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp_path = tempfile.mkstemp(
        prefix=_atomic_writer_temp_prefix(path),
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(raw_temp_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one torch checkpoint without exposing a partial final path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp_path = tempfile.mkstemp(
        prefix=_atomic_writer_temp_prefix(path),
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(raw_temp_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _cleanup_atomic_temps(path: Path) -> list[Path]:
    """Remove only temp files created for this exact committed path."""

    if not path.parent.is_dir():
        return []
    removed: list[Path] = []
    prefix = _atomic_temp_prefix(path)
    for candidate in path.parent.iterdir():
        if not (
            candidate.name.startswith(prefix)
            and candidate.name.endswith(".tmp")
            and (candidate.is_file() or candidate.is_symlink())
        ):
            continue
        writer_token = candidate.name[len(prefix) :].split("-", 1)[0]
        try:
            writer_pid = int(writer_token)
        except ValueError:
            continue
        if writer_pid <= 0:
            continue
        try:
            os.kill(writer_pid, 0)
            writer_is_alive = True
        except ProcessLookupError:
            writer_is_alive = False
        except PermissionError:
            writer_is_alive = True
        if not writer_is_alive:
            candidate.unlink()
            removed.append(candidate)
    if removed:
        _fsync_directory(path.parent)
    return removed


def _success_path_writer_contract(
    *,
    config: Mapping[str, Any],
    student: Path,
    teacher: Path,
    output_dir: Path,
    task_contract_hash: str,
) -> dict[str, Any]:
    path_identity = {
        "trajectory_artifacts": [
            str(Path(value).expanduser().resolve())
            for value in config["trajectory_artifacts"]
        ],
        "student": str(student.expanduser().resolve()),
        "teacher": str(teacher.expanduser().resolve()),
    }
    return {
        "schema": SUCCESS_PATH_WRITER_LOCK_SCHEMA,
        "output_dir": str(output_dir.expanduser().resolve()),
        "task_contract_hash": str(task_contract_hash),
        "resume_contract": _success_path_resume_contract(
            config,
            exact_identity=path_identity,
        ),
    }


def _writer_pid_is_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_writer_lock(path: Path) -> tuple[dict[str, Any], os.stat_result]:
    if path.is_symlink():
        raise NativeClosedLoopError(
            f"success-path writer lock must not be a symlink: {path}"
        )
    try:
        stat_result = path.stat(follow_symlinks=False)
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeClosedLoopError(
            f"success-path writer lock is malformed: {path}"
        ) from error
    if not isinstance(loaded, dict):
        raise NativeClosedLoopError(
            f"success-path writer lock is not a mapping: {path}"
        )
    return loaded, stat_result


def _acquire_success_path_writer_lock(
    *,
    output_dir: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".success_path_writer.lock"
    guard_path = output_dir / ".success_path_writer.guard"
    host = socket.gethostname()
    contract_copy = deepcopy(dict(contract))
    contract_hash = _stable_hash(contract_copy)
    guard_flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        guard_flags |= os.O_CLOEXEC
    guard_descriptor = os.open(guard_path, guard_flags, 0o600)
    try:
        try:
            fcntl.flock(
                guard_descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise NativeClosedLoopError(
                "success-path output has a live writer"
            ) from error
        _cleanup_atomic_temps(lock_path)
        if lock_path.exists() or lock_path.is_symlink():
            loaded, _observed_stat = _read_writer_lock(lock_path)
            if loaded.get("schema") != SUCCESS_PATH_WRITER_LOCK_SCHEMA:
                raise NativeClosedLoopError(
                    "success-path writer lock schema mismatch"
                )
            if str(loaded.get("host", "")) != host:
                raise NativeClosedLoopError(
                    "success-path writer lock belongs to another host; "
                    "owner death cannot be verified"
                )
            raw_pid = loaded.get("pid")
            if isinstance(raw_pid, bool) or not isinstance(raw_pid, int):
                raise NativeClosedLoopError(
                    "success-path writer lock PID is malformed"
                )
            if _writer_pid_is_alive(int(raw_pid)):
                raise NativeClosedLoopError(
                    "success-path output has a live writer"
                )
            if (
                loaded.get("contract") != contract_copy
                or str(loaded.get("contract_hash", "")) != contract_hash
            ):
                raise NativeClosedLoopError(
                    "stale success-path writer lock belongs to a different contract"
                )
        owner = {
            "schema": SUCCESS_PATH_WRITER_LOCK_SCHEMA,
            "pid": int(os.getpid()),
            "host": host,
            "process_started_at": datetime.now(timezone.utc).isoformat(
                timespec="microseconds"
            ),
            "owner_token": secrets.token_hex(16),
            "contract": contract_copy,
            "contract_hash": contract_hash,
        }
        _write_json(lock_path, owner)
        owner_with_guard = dict(owner)
        owner_with_guard["_guard_descriptor"] = int(guard_descriptor)
        return owner_with_guard
    except BaseException:
        try:
            fcntl.flock(guard_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(guard_descriptor)
        raise


def _release_success_path_writer_lock(
    *,
    output_dir: Path,
    owner: Mapping[str, Any],
) -> None:
    lock_path = output_dir / ".success_path_writer.lock"
    raw_guard_descriptor = owner.get("_guard_descriptor")
    if isinstance(raw_guard_descriptor, bool) or not isinstance(
        raw_guard_descriptor, int
    ):
        raise NativeClosedLoopError(
            "success-path writer lock has no kernel guard descriptor"
        )
    guard_descriptor = int(raw_guard_descriptor)
    try:
        loaded, _ = _read_writer_lock(lock_path)
        if (
            loaded.get("owner_token") != owner.get("owner_token")
            or loaded.get("pid") != owner.get("pid")
            or loaded.get("host") != owner.get("host")
        ):
            raise NativeClosedLoopError(
                "success-path writer lock ownership changed before release"
            )
        lock_path.unlink()
        _fsync_directory(output_dir)
    finally:
        try:
            fcntl.flock(guard_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(guard_descriptor)


@contextmanager
def _success_path_output_lock(
    *,
    output_dir: Path,
    contract: Mapping[str, Any],
):  # type: ignore[no-untyped-def]
    owner = _acquire_success_path_writer_lock(
        output_dir=output_dir,
        contract=contract,
    )
    try:
        yield owner
    finally:
        _release_success_path_writer_lock(
            output_dir=output_dir,
            owner=owner,
        )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, payload)


def _stable_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            byte_count += len(block)
    return digest.hexdigest(), int(byte_count)


def _path_content_identity(
    path: Path,
    *,
    _active_directories: set[Path] | None = None,
) -> dict[str, Any]:
    """Hash one file/tree by bytes and stable relative entry names."""

    requested = path.expanduser()
    try:
        resolved = requested.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"exact-resume input does not exist: {path}") from error
    if resolved.is_file():
        digest, byte_count = _sha256_file(resolved)
        return {
            "path": str(resolved),
            "kind": "file",
            "sha256": digest,
            "bytes": int(byte_count),
            "files": 1,
        }
    if not resolved.is_dir():
        raise ValueError(f"exact-resume input is not a file or directory: {resolved}")

    active = set() if _active_directories is None else _active_directories
    if resolved in active:
        raise ValueError(
            f"exact-resume input tree contains a symlink cycle: {resolved}"
        )
    active.add(resolved)
    entries: list[dict[str, Any]] = []
    byte_count = 0
    file_count = 0
    try:
        for child in sorted(resolved.iterdir(), key=lambda item: item.name):
            if child.is_symlink():
                target_text = os.readlink(child)
                target_identity = _path_content_identity(
                    child,
                    _active_directories=active,
                )
                entries.append(
                    {
                        "name": child.name,
                        "kind": "symlink",
                        "target": target_text,
                        "target_kind": target_identity["kind"],
                        "sha256": target_identity["sha256"],
                        "bytes": int(target_identity["bytes"]),
                        "files": int(target_identity["files"]),
                    }
                )
                byte_count += int(target_identity["bytes"])
                file_count += int(target_identity["files"])
            elif child.is_file():
                digest, child_bytes = _sha256_file(child)
                entries.append(
                    {
                        "name": child.name,
                        "kind": "file",
                        "sha256": digest,
                        "bytes": int(child_bytes),
                        "files": 1,
                    }
                )
                byte_count += int(child_bytes)
                file_count += 1
            elif child.is_dir():
                child_identity = _path_content_identity(
                    child,
                    _active_directories=active,
                )
                entries.append(
                    {
                        "name": child.name,
                        "kind": "directory",
                        "sha256": child_identity["sha256"],
                        "bytes": int(child_identity["bytes"]),
                        "files": int(child_identity["files"]),
                    }
                )
                byte_count += int(child_identity["bytes"])
                file_count += int(child_identity["files"])
            else:
                raise ValueError(
                    f"exact-resume input tree contains a special file: {child}"
                )
    finally:
        active.remove(resolved)
    return {
        "path": str(resolved),
        "kind": "directory",
        "sha256": _stable_hash(
            {
                "schema": "waopd_stable_tree_digest_v1",
                "entries": entries,
            }
        ),
        "bytes": int(byte_count),
        "files": int(file_count),
    }


def _success_path_exact_identity(
    *,
    artifact_paths: Sequence[Path],
    student: Path,
    teacher: Path,
) -> dict[str, Any]:
    identity = {
        "schema": SUCCESS_PATH_EXACT_IDENTITY_SCHEMA,
        "trajectory_artifacts": [
            _path_content_identity(path) for path in artifact_paths
        ],
        "student": _path_content_identity(student),
        "teacher": _path_content_identity(teacher),
    }
    identity["identity_hash"] = _stable_hash(identity)
    return identity


def _validate_success_path_artifacts_unchanged(
    identity: Mapping[str, Any], artifact_paths: Sequence[Path]
) -> None:
    expected = identity.get("trajectory_artifacts")
    observed = [_path_content_identity(path) for path in artifact_paths]
    if expected != observed:
        raise NativeClosedLoopError(
            "success-path trajectory artifact content changed while loading"
        )


def _success_path_resume_contract(
    config: Mapping[str, Any],
    *,
    exact_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze every training choice that must not change across exact resume."""

    return {
        "schema": SUCCESS_PATH_RESUME_CONTRACT_SCHEMA,
        "exact_input_identity": deepcopy(dict(exact_identity)),
        "collection_group_id": str(config["collection_group_id"]),
        "trajectory_artifacts": list(config["trajectory_artifacts"]),
        "adapter": {
            "kind": str(config["adapter_kind"]),
            "rank": int(config["adapter_rank"]),
            "alpha": float(config["lora_alpha"]),
            "dropout": float(config["lora_dropout"]),
            "block_indices": list(config["lora_block_indices"]),
            "seed": int(config["adapter_seed"]),
        },
        "optimizer": {
            "kind": str(config["optimizer_kind"]),
            "bank": str(config["trainable_bank"]),
            "learning_rate": float(config["learning_rate"]),
            "max_grad_norm": float(config["max_grad_norm"]),
        },
        "loss": {
            "pseudo_huber_c": float(config["pseudo_huber_c"]),
            "video_weight": float(config["video_weight"]),
            "action_weight": float(config["action_weight"]),
            "action_fm_weight": float(config["action_fm_weight"]),
            "action_velocity_weight": float(config["action_velocity_weight"]),
            "reduction": str(config["loss_reduction"]),
            "retention_weight": float(config["retention_weight"]),
        },
        "schedule": {
            "inner_epochs_total": int(config["inner_epochs"]),
            "effective_batch_size": int(config["effective_batch_size"]),
            "consistency_seed": int(config["consistency_seed"]),
            "consistency_noise_source": str(config["consistency_noise_source"]),
            "calibration_anchors_per_trajectory": int(
                config["calibration_anchors_per_trajectory"]
            ),
            "max_train_labels_per_trajectory": config[
                "success_path_max_train_labels_per_trajectory"
            ],
        },
    }


def _success_path_adamw_group_contract(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact AdamW param-group defaults used by this trainer."""

    probe = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        [probe],
        lr=float(config["learning_rate"]),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    group = deepcopy(optimizer.state_dict()["param_groups"][0])
    group.pop("params")
    return group


def _validate_success_path_resume_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    expected_task_contract_hash: str,
    expected_behavior_policy_version: str,
    expected_round_id: int,
    expected_parameter_names: Sequence[str] | None = None,
    expected_exact_identity: Mapping[str, Any] | None = None,
    expected_adapter_contract: Mapping[str, Any] | None = None,
    expected_base_parameter_hashes: Mapping[str, str] | None = None,
    expected_checkpoint_role: str = "success_path_epoch",
) -> dict[str, Any]:
    """Validate one fail-closed success-path epoch/final checkpoint."""

    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("success-path resume checkpoint schema mismatch")
    if str(checkpoint.get("task", "")) != str(config["task"]) or str(
        checkpoint.get("task_config", "")
    ) != str(config["task_config"]):
        raise ValueError("success-path resume task condition mismatch")
    loaded_task_hash = checkpoint.get("task_contract_hash")
    if str(loaded_task_hash or "") != str(expected_task_contract_hash):
        raise ValueError("success-path resume task contract mismatch")
    loaded_task_entries = checkpoint.get("task_entries")
    if not isinstance(loaded_task_entries, Sequence) or isinstance(
        loaded_task_entries, (str, bytes)
    ):
        raise ValueError("success-path resume has no task entries")
    if _stable_hash(loaded_task_entries) != str(loaded_task_hash):
        raise ValueError("success-path resume checkpoint task entries are corrupt")
    if expected_checkpoint_role not in {
        "success_path_epoch",
        "success_path_final",
    }:
        raise ValueError("unsupported success-path checkpoint role")
    if checkpoint.get("checkpoint_role") != expected_checkpoint_role:
        raise ValueError(
            f"success-path checkpoint role mismatch: {expected_checkpoint_role}"
        )
    loaded_finalization = checkpoint.get("success_path_finalization")
    loaded_finalization_hash = checkpoint.get(
        "success_path_finalization_hash"
    )
    if expected_checkpoint_role == "success_path_final":
        if not isinstance(loaded_finalization, Mapping):
            raise ValueError(
                "legacy success-path final checkpoint has no atomic "
                "finalization payload"
            )
        if (
            loaded_finalization.get("success_path_finalization_schema")
            != SUCCESS_PATH_FINALIZATION_SCHEMA
            or "checkpoint_sha256" in loaded_finalization
            or str(loaded_finalization_hash or "")
            != _stable_hash(dict(loaded_finalization))
        ):
            raise ValueError(
                "success-path final checkpoint finalization payload is corrupt"
            )
    elif loaded_finalization is not None or loaded_finalization_hash is not None:
        raise ValueError(
            "success-path epoch checkpoint contains finalization payload"
        )
    if checkpoint.get("adapter_kind") != "joint_lora":
        raise ValueError("success-path resume adapter kind mismatch")
    if _checkpoint_objective(checkpoint) != OBJECTIVE_COHERENT_TT_CONSISTENCY:
        raise ValueError("success-path resume objective mismatch")
    if checkpoint.get("coherent_tt_variant") != (
        COHERENT_TT_VARIANT_SUCCESS_PATH_V1
    ):
        raise ValueError("success-path resume variant mismatch")
    if int(checkpoint.get("adapter_seed", -1)) != int(config["adapter_seed"]):
        raise ValueError("success-path resume adapter seed mismatch")
    if checkpoint.get("optimizer_kind") != config["optimizer_kind"]:
        raise ValueError("success-path resume optimizer kind mismatch")
    if checkpoint.get("optimizer_bank") != config["trainable_bank"]:
        raise ValueError("success-path resume optimizer bank mismatch")
    if int(checkpoint.get("round_id", -1)) != int(expected_round_id):
        raise ValueError("success-path resume round mismatch")
    if str(checkpoint.get("policy_version_before", "")) != str(
        expected_behavior_policy_version
    ):
        raise ValueError("success-path resume behavior policy mismatch")
    if not str(checkpoint.get("policy_version_after", "")):
        raise ValueError("success-path resume has no adapted policy identity")

    loaded_identity = checkpoint.get("success_path_exact_identity")
    if not isinstance(loaded_identity, Mapping):
        raise ValueError(
            "legacy success-path checkpoint has no exact content identity"
        )
    loaded_identity = dict(loaded_identity)
    loaded_identity_hash = loaded_identity.get("identity_hash")
    identity_without_hash = dict(loaded_identity)
    identity_without_hash.pop("identity_hash", None)
    if (
        loaded_identity.get("schema") != SUCCESS_PATH_EXACT_IDENTITY_SCHEMA
        or str(loaded_identity_hash or "") != _stable_hash(identity_without_hash)
        or str(checkpoint.get("success_path_exact_identity_hash", ""))
        != str(loaded_identity_hash or "")
    ):
        raise ValueError("success-path checkpoint exact content identity is corrupt")
    if expected_exact_identity is not None and loaded_identity != dict(
        expected_exact_identity
    ):
        raise ValueError("success-path checkpoint exact input identity mismatch")

    loaded_contract = checkpoint.get("success_path_resume_contract")
    expected_contract = _success_path_resume_contract(
        config,
        exact_identity=loaded_identity,
    )
    if not isinstance(loaded_contract, Mapping) or dict(loaded_contract) != (
        expected_contract
    ):
        raise ValueError("success-path resume training contract mismatch")
    if str(checkpoint.get("success_path_resume_contract_hash", "")) != (
        _stable_hash(expected_contract)
    ):
        raise ValueError("success-path resume training contract hash mismatch")

    loaded_adapter_contract = checkpoint.get("adapter_contract")
    if not isinstance(loaded_adapter_contract, Mapping):
        raise ValueError("success-path checkpoint has no adapter contract")
    loaded_adapter_contract = dict(loaded_adapter_contract)
    if str(checkpoint.get("adapter_contract_hash", "")) != _stable_hash(
        loaded_adapter_contract
    ):
        raise ValueError("success-path checkpoint adapter contract hash mismatch")
    if expected_adapter_contract is not None and loaded_adapter_contract != dict(
        expected_adapter_contract
    ):
        raise ValueError("success-path live adapter contract mismatch")
    contract_base_hashes = loaded_adapter_contract.get("base_parameter_hashes")
    loaded_base_hashes = checkpoint.get("base_parameter_hashes")
    if (
        not isinstance(contract_base_hashes, Mapping)
        or not contract_base_hashes
        or not isinstance(loaded_base_hashes, Mapping)
        or dict(contract_base_hashes) != dict(loaded_base_hashes)
        or str(checkpoint.get("base_parameter_hashes_hash", ""))
        != _stable_hash(dict(loaded_base_hashes))
    ):
        raise ValueError(
            "success-path checkpoint base parameter identity mismatch"
        )
    if expected_base_parameter_hashes is not None and dict(
        loaded_base_hashes
    ) != dict(expected_base_parameter_hashes):
        raise ValueError("success-path live base parameter identity mismatch")

    parameter_names = checkpoint.get("optimizer_parameter_names")
    if not isinstance(parameter_names, Sequence) or isinstance(
        parameter_names, (str, bytes)
    ) or not parameter_names:
        raise ValueError("success-path resume optimizer manifest is missing")
    if expected_parameter_names is not None and list(parameter_names) != list(
        expected_parameter_names
    ):
        raise ValueError("success-path resume optimizer parameter manifest mismatch")
    if checkpoint.get("optimizer_parameter_dtypes") != ["torch.float32"]:
        raise ValueError("success-path resume optimizer parameters are not FP32")
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("success-path resume has no optimizer state")
    raw_param_groups = optimizer_state.get("param_groups")
    if not isinstance(raw_param_groups, list) or len(raw_param_groups) != 1:
        raise ValueError(
            "success-path resume optimizer parameter groups are malformed"
        )
    raw_param_group = raw_param_groups[0]
    if not isinstance(raw_param_group, Mapping):
        raise ValueError(
            "success-path resume optimizer parameter group is malformed"
        )
    raw_group_parameters = raw_param_group.get("params")
    if not isinstance(raw_group_parameters, list) or raw_group_parameters != list(
        range(len(parameter_names))
    ):
        raise ValueError(
            "success-path resume optimizer parameter ordering mismatch"
        )
    loaded_adamw_group = {
        str(name): deepcopy(value)
        for name, value in raw_param_group.items()
        if name != "params"
    }
    if loaded_adamw_group != _success_path_adamw_group_contract(config):
        raise ValueError(
            "success-path resume optimizer parameter-group contract mismatch"
        )
    raw_optimizer_state = optimizer_state.get("state")
    if not isinstance(raw_optimizer_state, Mapping) or not raw_optimizer_state:
        raise ValueError("success-path resume optimizer state is malformed")
    optimizer_state_dtypes = sorted(
        {
            str(value.dtype)
            for state in raw_optimizer_state.values()
            if isinstance(state, Mapping)
            for value in state.values()
            if isinstance(value, torch.Tensor)
        }
    )
    if checkpoint.get("optimizer_state_dtypes") != optimizer_state_dtypes:
        raise ValueError("success-path resume optimizer dtype metadata mismatch")
    if any(value != "torch.float32" for value in optimizer_state_dtypes):
        raise ValueError("success-path resume optimizer state is not FP32")
    adapter_state = checkpoint.get("adapter_state_dict")
    if not isinstance(adapter_state, Mapping) or not adapter_state:
        raise ValueError("success-path resume has no adapter state")
    if list(adapter_state) != list(parameter_names) or any(
        not isinstance(adapter_state[name], torch.Tensor)
        or adapter_state[name].dtype != torch.float32
        for name in parameter_names
    ):
        raise ValueError("success-path resume adapter manifest mismatch")
    if checkpoint.get("ema_adapter_state_dict") is not None or int(
        checkpoint.get("ema_updates", 0)
    ) != 0:
        raise ValueError("success_path_v1 resume must not contain EMA state")

    completed = checkpoint.get("completed_inner_epochs")
    if isinstance(completed, bool) or not isinstance(completed, int):
        raise ValueError("success-path resume completed epoch is invalid")
    if expected_checkpoint_role == "success_path_epoch":
        if not 1 <= int(completed) < int(config["inner_epochs"]):
            raise ValueError("success-path resume checkpoint is already complete")
    elif int(completed) != int(config["inner_epochs"]):
        raise ValueError("success-path final checkpoint epoch is incomplete")
    progress = checkpoint.get("success_path_progress")
    if not isinstance(progress, Mapping):
        raise ValueError("success-path resume checkpoint has no progress")
    if progress.get("schema") != SUCCESS_PATH_PROGRESS_SCHEMA or int(
        progress.get("completed_inner_epochs", -1)
    ) != int(completed):
        raise ValueError("success-path resume progress identity mismatch")
    generator_state = checkpoint.get("consistency_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("success-path resume checkpoint has no generator state")
    global_step = checkpoint.get("global_optimizer_step")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or int(global_step) <= 0
    ):
        raise ValueError("success-path resume global optimizer step is invalid")
    expected_state_ids = set(raw_group_parameters)
    if set(raw_optimizer_state) != expected_state_ids:
        raise ValueError(
            "success-path resume optimizer state does not cover its manifest"
        )
    required_state_keys = {"step", "exp_avg", "exp_avg_sq"}
    for parameter_id, parameter_name in enumerate(parameter_names):
        parameter_state = raw_optimizer_state[parameter_id]
        if not isinstance(parameter_state, Mapping) or set(parameter_state) != (
            required_state_keys
        ):
            raise ValueError(
                "success-path resume AdamW parameter state is malformed"
            )
        step = parameter_state["step"]
        exp_avg = parameter_state["exp_avg"]
        exp_avg_sq = parameter_state["exp_avg_sq"]
        parameter = adapter_state[parameter_name]
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or not bool(torch.isfinite(step).all().item())
            or float(step.item()) != float(global_step)
        ):
            raise ValueError(
                "success-path resume AdamW step/global step mismatch"
            )
        if (
            not isinstance(exp_avg, torch.Tensor)
            or not isinstance(exp_avg_sq, torch.Tensor)
            or exp_avg.shape != parameter.shape
            or exp_avg_sq.shape != parameter.shape
        ):
            raise ValueError(
                "success-path resume AdamW moment shape mismatch"
            )
    progress_steps = progress.get("steps")
    if not isinstance(progress_steps, list) or len(progress_steps) != int(global_step):
        raise ValueError("success-path resume progress/optimizer step mismatch")

    return {
        "completed_inner_epochs": int(completed),
        "behavior_policy_version": str(expected_behavior_policy_version),
        "consistency_generator_state": generator_state.detach().clone().cpu(),
        "success_path_progress": deepcopy(dict(progress)),
        "global_optimizer_step": int(global_step),
    }


def _adapter_state_policy_version(state: Mapping[str, Any]) -> str:
    if not state:
        raise NativeClosedLoopError("dual adapter state is empty")
    tensor_hashes: dict[str, str] = {}
    for raw_name, value in state.items():
        name = str(raw_name)
        if name != raw_name:
            raise TypeError("dual adapter state keys must be strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"dual adapter state {name} is not a Tensor")
        tensor_hashes[name] = str(tensor_hash(value))
    return _stable_hash(tensor_hashes)


def _policy_version(runtime: NativeV0VideoRuntime) -> str:
    if runtime.adapter_kind == "dual_lora":
        return _adapter_state_policy_version(runtime.adapter_state())
    return _stable_hash(runtime.parameter_hashes())


def _validate_checkpoint_task_contract(
    checkpoint: Mapping[str, Any],
    *,
    expected_schema: str,
    expected_hash: str,
    multi_task: bool,
) -> None:
    """Bind every dual checkpoint to its exact rollout/seed/prompt contract."""

    loaded_hash = checkpoint.get("task_contract_hash")
    strict = expected_schema == DUAL_CHECKPOINT_SCHEMA or bool(multi_task)
    if loaded_hash is None:
        if strict:
            raise ValueError("initial_checkpoint has no task contract")
        return
    if str(loaded_hash) != str(expected_hash):
        raise ValueError("initial_checkpoint task contract mismatch")


def _validate_dual_optimizer_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
) -> None:
    """Validate raw dual-bank optimizer dtypes before PyTorch can cast them."""

    parameter_dtypes = checkpoint.get("optimizer_parameter_dtypes")
    if parameter_dtypes != ["torch.float32"]:
        raise ValueError(
            "initial_checkpoint dual optimizer parameters are not declared FP32"
        )
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("initial_checkpoint has no optimizer_state_dict")
    raw_state = optimizer_state.get("state")
    if not isinstance(raw_state, Mapping):
        raise ValueError("initial_checkpoint optimizer state is malformed")
    raw_state_dtypes = sorted(
        {
            str(value.dtype)
            for state in raw_state.values()
            if isinstance(state, Mapping)
            for value in state.values()
            if isinstance(value, torch.Tensor)
        }
    )
    if checkpoint.get("optimizer_state_dtypes") != raw_state_dtypes:
        raise ValueError(
            "initial_checkpoint optimizer state dtype metadata mismatch"
        )
    if any(value != "torch.float32" for value in raw_state_dtypes):
        raise ValueError("initial_checkpoint dual optimizer state is not FP32")


def _rollout_collection_contract(
    task_entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "task": str(entry["task"]),
            "task_config": str(entry["task_config"]),
            "chunks": int(entry["chunks"]),
            "max_control_steps": int(
                TASK_SPECS[str(entry["task"])].max_control_steps
            ),
            "rollouts": deepcopy(list(entry["rollouts"])),
        }
        for entry in task_entries
    ]


TERMINAL_LABEL_CONTRACT = {
    "stop_on_success": True,
    "success_triggering_macro": "included_with_terminal_execution_mask",
    "post_success_labels": 0,
    "execution_mask_fields": [
        "executed_action_mask",
        "terminal_reached",
        "terminal_action_position",
        "horizon_reached",
    ],
}


def _validate_rollout_bundle(
    bundle: Mapping[str, Any],
    *,
    expected_task_entries: Sequence[Mapping[str, Any]],
    expected_task_contract_hash: str,
    expected_student: Path,
) -> None:
    """Validate the immutable policy/data boundary used by branch updates."""

    if bundle.get("schema") != ROLLOUT_BUNDLE_SCHEMA:
        raise ValueError("rollout_bundle schema mismatch")
    if bundle.get("adapter_kind") != "dual_lora":
        raise ValueError("rollout_bundle must contain a dual_lora policy")
    adapter_contract = bundle.get("adapter_contract")
    if not isinstance(adapter_contract, Mapping):
        raise ValueError("rollout_bundle has no adapter contract")
    adapter_state = bundle.get("adapter_state_dict")
    if not isinstance(adapter_state, Mapping):
        raise ValueError("rollout_bundle has no adapter state")
    computed_policy_version = _adapter_state_policy_version(adapter_state)
    if str(bundle.get("behavior_policy_version")) != computed_policy_version:
        raise NativeClosedLoopError(
            "rollout_bundle policy hash differs from its exact adapter state"
        )
    adapter_seed = bundle.get("adapter_seed")
    if (
        not isinstance(adapter_seed, int)
        or isinstance(adapter_seed, bool)
        or adapter_seed < 0
    ):
        raise ValueError("rollout_bundle adapter_seed is invalid")

    task_contract = bundle.get("task_contract")
    if not isinstance(task_contract, Mapping):
        raise ValueError("rollout_bundle has no task contract")
    task_entries = task_contract.get("task_entries")
    if not isinstance(task_entries, Sequence) or isinstance(
        task_entries, (str, bytes)
    ):
        raise ValueError("rollout_bundle task entries are malformed")
    task_entries_copy = deepcopy(list(task_entries))
    computed_task_hash = _stable_hash(task_entries_copy)
    if str(task_contract.get("task_contract_hash")) != computed_task_hash:
        raise ValueError("rollout_bundle internal task contract hash mismatch")
    if computed_task_hash != str(expected_task_contract_hash):
        raise ValueError("rollout_bundle task contract mismatch")
    if task_entries_copy != deepcopy(list(expected_task_entries)):
        raise ValueError("rollout_bundle task entries differ from branch config")
    expected_collection_contract = _rollout_collection_contract(
        expected_task_entries
    )
    collection_contract = task_contract.get("collection_contract")
    if collection_contract != expected_collection_contract:
        raise ValueError("rollout_bundle collection horizon contract mismatch")
    if bundle.get("terminal_label_contract") != TERMINAL_LABEL_CONTRACT:
        raise ValueError("rollout_bundle terminal label contract mismatch")
    if Path(str(bundle.get("base_student_checkpoint", ""))).expanduser().resolve() != (
        expected_student.expanduser().resolve()
    ):
        raise ValueError("rollout_bundle base Student checkpoint mismatch")

    trajectories = bundle.get("trajectories")
    outcomes = bundle.get("outcomes")
    if not isinstance(trajectories, Sequence) or isinstance(
        trajectories, (str, bytes)
    ):
        raise ValueError("rollout_bundle trajectories are malformed")
    if not isinstance(outcomes, Sequence) or isinstance(outcomes, (str, bytes)):
        raise ValueError("rollout_bundle outcomes are malformed")
    if not trajectories or len(trajectories) != len(outcomes):
        raise ValueError("rollout_bundle trajectory/outcome counts differ")
    policy_version = str(bundle["behavior_policy_version"])
    round_id = int(bundle.get("round_id", -1))
    if round_id < 0:
        raise ValueError("rollout_bundle round_id is invalid")
    for index, trajectory in enumerate(trajectories):
        if not isinstance(trajectory, Mapping):
            raise ValueError(f"rollout_bundle trajectory[{index}] is malformed")
        if str(trajectory.get("behavior_policy_version")) != policy_version:
            raise NativeClosedLoopError(
                "rollout_bundle trajectory policy differs from adapter state"
            )
        if int(trajectory.get("round_id", -1)) != round_id:
            raise ValueError("rollout_bundle trajectory round mismatch")
        if int(trajectory.get("success_post_label_count", -1)) != 0:
            raise ValueError("rollout_bundle contains post-success labels")
        labels = trajectory.get("labels")
        if not isinstance(labels, Sequence) or not labels:
            raise ValueError("rollout_bundle trajectory has no labels")
        for label in labels:
            if not isinstance(label, Mapping):
                raise ValueError("rollout_bundle label is malformed")
            if str(label.get("behavior_policy_version")) != policy_version:
                raise NativeClosedLoopError(
                    "rollout_bundle label policy differs from adapter state"
                )
            if int(label.get("round_id", -1)) != round_id:
                raise ValueError("rollout_bundle label round mismatch")
            missing_terminal_fields = sorted(
                set(TERMINAL_LABEL_CONTRACT["execution_mask_fields"])
                - set(label)
            )
            if missing_terminal_fields:
                raise ValueError(
                    "rollout_bundle label lacks terminal execution mask fields: "
                    f"{missing_terminal_fields}"
                )
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, Mapping):
            raise ValueError(f"rollout_bundle outcome[{index}] is malformed")
        if str(outcome.get("policy_version")) != policy_version:
            raise NativeClosedLoopError(
                "rollout_bundle outcome policy differs from adapter state"
            )
        if int(outcome.get("round_id", -1)) != round_id:
            raise ValueError("rollout_bundle outcome round mismatch")


def _build_rollout_bundle(
    *,
    runtime: NativeV0VideoRuntime,
    config: Mapping[str, Any],
    student: Path,
    task_contract_hash: str,
    policy_version: str,
    round_id: int,
    global_optimizer_step: int,
    trajectories: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if runtime.adapter_kind != "dual_lora":
        raise ValueError("collect-once rollout bundles require dual_lora")
    adapter_state = runtime.adapter_state()
    if _adapter_state_policy_version(adapter_state) != str(policy_version):
        raise NativeClosedLoopError(
            "collected policy hash differs from exact dual adapter state"
        )
    task_entries = deepcopy(list(config["task_entries"]))
    if _stable_hash(task_entries) != str(task_contract_hash):
        raise ValueError("resolved task contract changed before bundle save")
    collection_contract = _rollout_collection_contract(task_entries)
    bundle = {
        "schema": ROLLOUT_BUNDLE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "adapter_kind": "dual_lora",
        "adapter_contract": runtime.adapter_contract(),
        "adapter_state_dict": adapter_state,
        "behavior_policy_version": str(policy_version),
        "base_student_checkpoint": str(student.expanduser().resolve()),
        "adapter_seed": int(config["adapter_seed"]),
        "task_contract": {
            "task": str(config["task"]),
            "task_config": str(config["task_config"]),
            "task_entries": task_entries,
            "task_contract_hash": str(task_contract_hash),
            "collection_contract": collection_contract,
        },
        "round_id": int(round_id),
        "global_optimizer_step": int(global_optimizer_step),
        "trajectories": deepcopy(list(trajectories)),
        "outcomes": deepcopy(list(outcomes)),
        "environment_execution": "SS",
        "teacher_role": "label_only",
        "teacher_controls_environment": False,
        "post_success_labels": 0,
        "terminal_label_contract": deepcopy(TERMINAL_LABEL_CONTRACT),
    }
    _validate_rollout_bundle(
        bundle,
        expected_task_entries=task_entries,
        expected_task_contract_hash=str(task_contract_hash),
        expected_student=student,
    )
    return bundle


def _load_rollout_bundle(
    path: Path,
    *,
    expected_task_entries: Sequence[Mapping[str, Any]],
    expected_task_contract_hash: str,
    expected_student: Path,
) -> dict[str, Any]:
    # Trajectories contain simulator observations (including NumPy values), so
    # this trusted local training artifact is intentionally not weights-only.
    loaded = torch.load(
        path.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    if not isinstance(loaded, dict):
        raise TypeError("rollout_bundle is not a mapping")
    _validate_rollout_bundle(
        loaded,
        expected_task_entries=expected_task_entries,
        expected_task_contract_hash=expected_task_contract_hash,
        expected_student=expected_student,
    )
    return loaded


def _load_trajectory_artifacts(
    paths: Sequence[Path],
    *,
    expected_task_entries: Sequence[Mapping[str, Any]],
    expected_student: Path,
    expected_teacher: Path | None = None,
    expected_adapter_seed: int | None = None,
    expected_collection_group_id: str | None = None,
    require_coherent_collection_contract: bool = False,
) -> list[dict[str, Any]]:
    """Load one exact Student-occupancy artifact per configured rollout."""

    expected_order: list[tuple[str, str, int, str]] = []
    expected_contracts: dict[tuple[str, str, int, str], dict[str, int]] = {}
    for entry in expected_task_entries:
        for rollout in entry["rollouts"]:
            key = (
                str(entry["task"]),
                str(entry["task_config"]),
                int(rollout["seed"]),
                str(rollout["prompt"]),
            )
            expected_order.append(key)
            expected_contracts[key] = {
                "chunks": int(entry["chunks"]),
                "max_control_steps": int(
                    TASK_SPECS[str(entry["task"])].max_control_steps
                ),
                "rollout_id": (
                    None
                    if "rollout_id" not in rollout
                    else int(rollout["rollout_id"])
                ),
                "dataset_role": rollout.get("role"),
            }
    if not expected_order:
        raise ValueError("trajectory_update task contract has no rollouts")

    resolved_paths = [path.expanduser().resolve() for path in paths]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("trajectory_artifacts contains duplicate paths")
    if len(resolved_paths) != len(expected_order):
        raise ValueError(
            "trajectory artifact count differs from configured rollouts: "
            f"{len(resolved_paths)} != {len(expected_order)}"
        )

    expected_keys = set(expected_order)
    loaded_by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    policy_versions: set[str] = set()
    round_ids: set[int] = set()
    collection_ids: set[str] = set()
    rollout_ids: set[int] = set()
    for path in resolved_paths:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict):
            raise TypeError(f"trajectory artifact is not a mapping: {path}")
        if loaded.get("schema") != TRAJECTORY_SCHEMA:
            raise ValueError(f"trajectory artifact schema mismatch: {path}")
        key = (
            str(loaded.get("task", "")),
            str(loaded.get("task_config", "")),
            int(loaded.get("seed", -1)),
            str(loaded.get("prompt", "")),
        )
        if key not in expected_keys:
            raise ValueError(
                f"trajectory task/seed/prompt differs from config: {path}"
            )
        if key in loaded_by_key:
            raise ValueError(f"duplicate trajectory contract entry: {key}")
        if loaded.get("history_owner") != "current_student_on_policy":
            raise ValueError("trajectory history is not owned by current Student")
        if loaded.get("environment_execution") != "SS":
            raise ValueError("trajectory environment execution is not SS")
        if loaded.get("teacher_controls_environment") is not False:
            raise ValueError("trajectory Teacher must not control the environment")
        if int(loaded.get("fresh_passes_allowed", -1)) != 1:
            raise ValueError("trajectory freshness contract is not one-pass")
        if int(loaded.get("success_post_label_count", -1)) != 0:
            raise ValueError("trajectory contains post-success labels")

        policy_version = str(loaded.get("behavior_policy_version", ""))
        round_id = int(loaded.get("round_id", -1))
        rollout_id = int(loaded.get("rollout_id", -1))
        collection_id = str(loaded.get("collection_id", ""))
        if not policy_version or round_id < 0 or rollout_id < 0 or not collection_id:
            raise ValueError("trajectory policy/round/rollout identity is malformed")
        if collection_id in collection_ids:
            raise ValueError("trajectory collection_id is duplicated")
        collection_ids.add(collection_id)
        if rollout_id in rollout_ids:
            raise ValueError("trajectory global rollout_id is duplicated")
        rollout_ids.add(rollout_id)

        expected_contract = expected_contracts[key]
        expected_rollout_id = expected_contract["rollout_id"]
        if (
            expected_rollout_id is not None
            and rollout_id != int(expected_rollout_id)
        ):
            raise ValueError("trajectory global rollout_id differs from config")
        expected_role = expected_contract["dataset_role"]
        if expected_role is not None and str(
            loaded.get("dataset_role", "")
        ) != str(expected_role):
            raise ValueError("trajectory dataset role differs from config")

        if require_coherent_collection_contract:
            if not expected_collection_group_id:
                raise ValueError(
                    "coherent trajectory merge requires collection_group_id"
                )
            if str(loaded.get("collection_group_id", "")) != str(
                expected_collection_group_id
            ):
                raise ValueError("trajectory collection group differs")
            if int(loaded.get("adapter_seed", -1)) != int(
                expected_adapter_seed
            ):
                raise ValueError("trajectory adapter seed differs")
            if str(loaded.get("objective", "")) != (
                OBJECTIVE_COHERENT_TT_CONSISTENCY
            ):
                raise ValueError("trajectory coherent objective differs")
            artifact_student = Path(
                str(loaded.get("base_student_checkpoint", ""))
            ).expanduser().resolve()
            if artifact_student != expected_student.expanduser().resolve():
                raise ValueError("trajectory base Student checkpoint differs")
            if expected_teacher is None:
                raise ValueError(
                    "coherent trajectory merge requires a frozen Teacher"
                )
            artifact_teacher = Path(
                str(loaded.get("teacher_transformer", ""))
            ).expanduser().resolve()
            if artifact_teacher != expected_teacher.expanduser().resolve():
                raise ValueError("trajectory Teacher checkpoint differs")

        labels = loaded.get("labels")
        history = loaded.get("history")
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise ValueError("trajectory labels are malformed")
        if not labels:
            raise ValueError("trajectory has no labels")
        if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
            raise ValueError("trajectory history is malformed")
        baseline = loaded.get("baseline_episode")
        if not isinstance(baseline, Mapping):
            raise ValueError("trajectory baseline episode is malformed")
        baseline_key = (
            str(baseline.get("task", "")),
            str(baseline.get("task_config", "")),
            int(baseline.get("seed", -1)),
            str(baseline.get("prompt", "")),
        )
        if baseline_key != key:
            raise ValueError("trajectory baseline task/seed/prompt differs")
        if str(baseline.get("arm", "")) != "SS":
            raise ValueError("trajectory baseline was not executed by SS")
        if baseline.get("shared_noise_across_arms") is not True:
            raise ValueError("trajectory baseline did not use shared arm noise")
        baseline_student = Path(
            str(baseline.get("student_checkpoint", ""))
        ).expanduser().resolve()
        if baseline_student != expected_student.expanduser().resolve():
            raise ValueError("trajectory baseline Student checkpoint differs")
        if int(baseline.get("chunks_requested", -1)) != int(
            expected_contract["chunks"]
        ):
            raise ValueError("trajectory collection chunk horizon differs")
        if int(baseline.get("max_control_steps", -1)) != int(
            expected_contract["max_control_steps"]
        ):
            raise ValueError("trajectory collection control horizon differs")
        physical_rows = baseline.get("chunks")
        if not isinstance(physical_rows, Sequence) or isinstance(
            physical_rows, (str, bytes)
        ):
            raise ValueError("trajectory physical chunks are malformed")
        if len(physical_rows) != len(labels):
            raise ValueError("trajectory physical chunk/label counts differ")
        if len(history) != len(labels):
            raise ValueError("trajectory history/label counts differ")

        terminal_indices: list[int] = []
        cumulative_control_steps = 0
        for label_index, label in enumerate(labels):
            if not isinstance(label, Mapping):
                raise ValueError("trajectory label is malformed")
            _validate_label(label)
            if str(label.get("collection_id", "")) != collection_id:
                raise ValueError("trajectory label collection identity differs")
            if str(label.get("behavior_policy_version", "")) != policy_version:
                raise NativeClosedLoopError(
                    "trajectory label policy differs from behavior policy"
                )
            if int(label.get("round_id", -1)) != round_id or int(
                label.get("rollout_id", -1)
            ) != rollout_id:
                raise ValueError("trajectory label round/rollout identity differs")
            if require_coherent_collection_contract:
                if str(label.get("collection_group_id", "")) != str(
                    expected_collection_group_id
                ):
                    raise ValueError("trajectory label collection group differs")
                if int(label.get("adapter_seed", -1)) != int(
                    expected_adapter_seed
                ):
                    raise ValueError("trajectory label adapter seed differs")
            if int(label.get("history_prefix_length", -1)) != label_index:
                raise ValueError("trajectory label history chronology differs")
            if label_index > len(history):
                raise ValueError("trajectory label history prefix exceeds history")
            missing_terminal_fields = sorted(
                set(TERMINAL_LABEL_CONTRACT["execution_mask_fields"]) - set(label)
            )
            if missing_terminal_fields:
                raise ValueError(
                    "trajectory label lacks terminal execution mask fields: "
                    f"{missing_terminal_fields}"
                )
            physical = physical_rows[label_index]
            if not isinstance(physical, Mapping):
                raise ValueError("trajectory physical chunk is malformed")
            for label_field, physical_field in {
                "macro_id": "chunk_id",
                "frame_st_id": "frame_st_id",
                "start_frame": "start_frame",
                "action_steps": "action_steps",
                "terminal_reached": "terminal_reached",
                "terminal_action_position": "terminal_action_position",
                "horizon_reached": "horizon_reached",
            }.items():
                if label.get(label_field) != physical.get(physical_field):
                    raise NativeClosedLoopError(
                        "trajectory label differs from physical chunk at "
                        f"{label_field}"
                    )
            label_mask = label["executed_action_mask"]
            physical_mask = physical.get("executed_action_mask")
            try:
                label_mask_tensor = torch.as_tensor(label_mask, dtype=torch.bool)
                physical_mask_tensor = torch.as_tensor(
                    physical_mask, dtype=torch.bool
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "trajectory execution mask is malformed"
                ) from error
            if not torch.equal(label_mask_tensor, physical_mask_tensor):
                raise NativeClosedLoopError(
                    "trajectory label differs from physical chunk at "
                    "executed_action_mask"
                )
            cumulative_control_steps += int(physical["action_steps"])
            if int(label.get("cumulative_control_steps", -1)) != int(
                cumulative_control_steps
            ):
                raise NativeClosedLoopError(
                    "trajectory cumulative controls differ from physical chunks"
                )
            if bool(label["terminal_reached"]):
                terminal_indices.append(label_index)
        if terminal_indices and terminal_indices != [len(labels) - 1]:
            raise ValueError("trajectory has labels after terminal success")

        policy_versions.add(policy_version)
        round_ids.add(round_id)
        loaded_by_key[key] = loaded

    missing_keys = [key for key in expected_order if key not in loaded_by_key]
    if missing_keys:
        raise ValueError(f"trajectory artifacts do not cover task contract: {missing_keys}")
    if len(policy_versions) != 1:
        raise NativeClosedLoopError(
            "trajectory artifacts were collected by different Student policies"
        )
    if len(round_ids) != 1:
        raise ValueError("trajectory artifacts were collected in different rounds")
    return [loaded_by_key[key] for key in expected_order]


def _masked_pseudo_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    c: float,
) -> torch.Tensor:
    """FlashWAM's exact pseudo-Huber objective on selected elements."""

    if tuple(prediction.shape) != tuple(target.shape):
        raise NativeClosedLoopError("pseudo-Huber prediction/target shapes differ")
    if tuple(mask.shape) != tuple(prediction.shape) or mask.dtype != torch.bool:
        raise NativeClosedLoopError("pseudo-Huber mask shape/dtype mismatch")
    if int(mask.sum().item()) <= 0:
        raise NativeClosedLoopError("pseudo-Huber mask is empty")
    if not math.isfinite(float(c)) or float(c) <= 0.0:
        raise ValueError("pseudo-Huber c must be finite and positive")
    diff = prediction.float() - target.detach().float()
    values = torch.sqrt(diff.square() + float(c) ** 2) - float(c)
    return values[mask].mean()


def _trust_region_sgd_learning_rate(
    *, configured_learning_rate: float, gradient_norm: float, max_update_norm: float
) -> float:
    """Scale plain SGD so its parameter-space update stays within an L2 ball."""

    for name, value in {
        "configured_learning_rate": configured_learning_rate,
        "gradient_norm": gradient_norm,
        "max_update_norm": max_update_norm,
    }.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    proposed = float(configured_learning_rate) * float(gradient_norm)
    return float(configured_learning_rate) * min(
        1.0, float(max_update_norm) / proposed
    )


def _sample_optimization_weights(
    trajectories: Sequence[Mapping[str, Any]], *, loss_reduction: str
) -> list[float]:
    """Return one optimization weight per flattened label.

    ``mean_trajectories_mean_labels`` gives every fresh rollout total weight
    ``1 / G`` regardless of terminal truncation or failure-horizon length.
    """

    if not trajectories:
        raise NativeClosedLoopError("fresh on-policy round has no trajectories")
    label_counts = [len(trajectory["labels"]) for trajectory in trajectories]
    if any(count <= 0 for count in label_counts):
        raise NativeClosedLoopError("fresh on-policy trajectory has no labels")
    if loss_reduction == LOSS_REDUCTION_MEAN_ALL:
        weight = 1.0 / float(sum(label_counts))
        return [weight for count in label_counts for _ in range(count)]
    if loss_reduction == LOSS_REDUCTION_MEAN_TRAJECTORIES:
        trajectory_weight = 1.0 / float(len(trajectories))
        return [
            trajectory_weight / float(count)
            for count in label_counts
            for _ in range(count)
        ]
    if loss_reduction == LOSS_REDUCTION_MEAN_TASKS:
        task_counts: dict[str, int] = {}
        trajectory_tasks: list[str] = []
        for trajectory in trajectories:
            task = str(trajectory.get("task", "")).strip()
            if not task:
                raise NativeClosedLoopError(
                    "task-balanced fresh trajectory has no task"
                )
            trajectory_tasks.append(task)
            task_counts[task] = task_counts.get(task, 0) + 1
        task_weight = 1.0 / float(len(task_counts))
        return [
            task_weight / float(task_counts[task]) / float(count)
            for task, count in zip(trajectory_tasks, label_counts, strict=True)
            for _ in range(count)
        ]
    raise ValueError(f"unsupported loss_reduction: {loss_reduction!r}")


class NativeOnPolicyJointLabelRuntime(NativeStudentVideoLabelRuntime):
    """Execute SS and collect coherent ``z_T, a_T(h_S, z_T)`` supervision."""

    def _on_teacher_video_plan_label(
        self,
        *,
        frame_st_id: int,
        teacher_plan: Any,
        student_solve: ActionSolve,
        action_noise: torch.Tensor,
    ) -> None:
        if not self._label_active:
            raise NativeClosedLoopError("Teacher action label arrived outside collection")
        if not self.teacher_video_labels:
            raise NativeClosedLoopError("Teacher action label has no video label")
        label = self.teacher_video_labels[-1]
        if int(label["frame_st_id"]) != frame_st_id:
            raise NativeClosedLoopError("Teacher video/action frame boundary mismatch")

        # NativeStudentVideoLabelRuntime deliberately invokes this hook while
        # the exact Teacher video prediction remains in the Teacher cache.
        # Do not clear that prediction or inject z_S: the action target must be
        # the coherent Teacher conditional a_T(h_S, z_T).
        teacher_solve = self._teacher_action(
            frame_st_id=frame_st_id,
            action_noise=action_noise,
            cache_name=self._teacher_history_cache,
            plan=teacher_plan,
            arm="TT",
        )
        if teacher_solve.initial_velocity is None:
            raise NativeClosedLoopError("Teacher initial action velocity is missing")
        if not bool(torch.isfinite(teacher_solve.initial_velocity).all().item()):
            raise NativeClosedLoopError("Teacher initial action velocity is nonfinite")
        if teacher_solve.plan.prepared_hash != teacher_plan.prepared_hash:
            raise NativeClosedLoopError("Teacher action did not consume Teacher z_T")

        exact_pairs = {
            "action_input_noise": (
                student_solve.action_noise,
                teacher_solve.action_noise,
            ),
            "action_timestep": (
                student_solve.action_timestep,
                teacher_solve.action_timestep,
            ),
            "valid_action_mask": (student_solve.mask, teacher_solve.mask),
        }
        for name, (student_value, teacher_value) in exact_pairs.items():
            if not torch.equal(student_value, teacher_value):
                raise NativeClosedLoopError(
                    f"Teacher/Student same-state contract differs at {name}"
                )
        if student_solve.action_token_positions != teacher_solve.action_token_positions:
            raise NativeClosedLoopError("Teacher/Student action token positions differ")
        if int(student_solve.cache_valid_length) != int(
            teacher_solve.cache_valid_length
        ):
            raise NativeClosedLoopError("Teacher/Student action cache lengths differ")

        label.update(
            {
                "teacher_action": _cpu_tensor(teacher_solve.model_action),
                "teacher_action_initial_velocity": _cpu_tensor(
                    teacher_solve.initial_velocity
                ),
                "teacher_action_input_noise": _cpu_tensor(
                    teacher_solve.action_noise
                ),
                "teacher_action_timestep": _cpu_tensor(
                    teacher_solve.action_timestep
                ),
                "teacher_action_valid_mask": _cpu_tensor(teacher_solve.mask),
                "teacher_action_token_positions": tuple(
                    int(value) for value in teacher_solve.action_token_positions
                ),
                "teacher_action_cache_valid_length": int(
                    teacher_solve.cache_valid_length
                ),
                "teacher_action_consumed_plan_hash": str(
                    teacher_plan.prepared_hash
                ),
                "student_action_input_noise": _cpu_tensor(
                    student_solve.action_noise
                ),
                "student_action_timestep": _cpu_tensor(
                    student_solve.action_timestep
                ),
                "student_action_token_positions": tuple(
                    int(value) for value in student_solve.action_token_positions
                ),
                "student_action_cache_valid_length": int(
                    student_solve.cache_valid_length
                ),
                "teacher_role": "label_only",
                "teacher_controls_environment": False,
                "environment_execution": "SS",
                "action_target_condition": "teacher_on_teacher_z_t",
            }
        )


def _validate_label(label: Mapping[str, Any]) -> None:
    if str(label.get("action_target_condition")) != "teacher_on_teacher_z_t":
        raise NativeClosedLoopError("saved action target is not coherent TT")
    teacher_plan_hash = tensor_hash(label["teacher_z_t"])
    if str(label["teacher_action_consumed_plan_hash"]) != str(teacher_plan_hash):
        raise NativeClosedLoopError("saved Teacher action plan hash is not Teacher z_T")
    if bool(label.get("teacher_controls_environment", True)):
        raise NativeClosedLoopError("Teacher action is marked as environment control")
    if str(label.get("environment_execution")) != "SS":
        raise NativeClosedLoopError("rollout is not marked as SS execution")
    if not torch.equal(
        label["student_action_input_noise"], label["teacher_action_input_noise"]
    ):
        raise NativeClosedLoopError("saved Teacher/Student action states differ")
    if not torch.equal(
        label["student_action_timestep"], label["teacher_action_timestep"]
    ):
        raise NativeClosedLoopError("saved Teacher/Student action timesteps differ")


def _collect_rollout(
    *,
    runtime: NativeOnPolicyJointLabelRuntime,
    slot: Mapping[str, Any],
    task: str,
    task_config: str,
    prompt: str,
    chunks: int,
    max_control_steps: int,
    policy_version: str,
    round_id: int,
    rollout_id: int,
    collection_group_id: str | None,
    adapter_seed: int,
    student_checkpoint: Path,
    teacher_transformer: Path,
    objective: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    worker = slot["worker"]
    events: list[dict[str, Any]] = []
    runtime.begin_label_episode()
    try:
        noise_bank = LockedNoiseBank(
            task=task,
            seed=int(slot["seed"]),
            device=runtime.device,
            dtype=runtime.dtype,
        )
        episode = run_live_episode(
            runtime=runtime,
            task_env=slot["task_env"],
            worker=worker,
            parent_snapshot=slot["parent_snapshot"],
            initial_observation=slot["initial_observation"],
            initial_eef_pose=slot["initial_eef_pose"],
            format_obs=slot["format_obs"],
            add_init_pose=slot["add_init_pose"],
            task=task,
            task_config=task_config,
            seed=int(slot["seed"]),
            prompt=prompt,
            arm="SS",
            chunks=int(chunks),
            noise_bank=noise_bank,
            macro_callback=lambda event: events.append(capture_student_context(event)),
            stop_on_success=True,
            max_control_steps=int(max_control_steps),
            shared_noise_across_arms=True,
        )
        teacher_labels = runtime.end_label_episode()
    except BaseException:
        if runtime._label_active:
            runtime.end_label_episode()
        raise

    trajectory = build_trajectory_artifact(
        task=task,
        task_config=task_config,
        seed=int(slot["seed"]),
        prompt=prompt,
        initial_observation=slot["initial_observation"],
        episode=episode,
        events=events,
        teacher_labels=teacher_labels,
    )
    collection_id = (
        f"{task}_round{round_id:02d}_rollout{rollout_id:02d}"
    )
    trajectory.update(
        {
            "collection_id": collection_id,
            "round_id": int(round_id),
            "rollout_id": int(rollout_id),
            "behavior_policy_version": str(policy_version),
            "collection_group_id": collection_group_id,
            "adapter_seed": int(adapter_seed),
            "base_student_checkpoint": str(
                student_checkpoint.expanduser().resolve()
            ),
            "teacher_transformer": str(
                teacher_transformer.expanduser().resolve()
            ),
            "objective": str(objective),
            "history_owner": "current_student_on_policy",
            "environment_execution": "SS",
            "teacher_controls_environment": False,
            "dataset_role": str(slot.get("dataset_role", "train")),
            "fresh_passes_allowed": 1,
        }
    )
    for label in trajectory["labels"]:
        label.update(
            {
                "collection_id": collection_id,
                "round_id": int(round_id),
                "rollout_id": int(rollout_id),
                "behavior_policy_version": str(policy_version),
                "collection_group_id": collection_group_id,
                "adapter_seed": int(adapter_seed),
            }
        )
        _validate_label(label)

    progress = _worker_progress(
        worker=worker,
        task_env=slot["task_env"],
        parent_snapshot=slot["parent_snapshot"],
        task=task,
    )
    outcome = _outcome(episode, progress)
    outcome.update(
        {
            "task": str(task),
            "seed": int(slot["seed"]),
            "round_id": int(round_id),
            "rollout_id": int(rollout_id),
            "policy_version": str(policy_version),
            "collection_group_id": collection_group_id,
            "adapter_seed": int(adapter_seed),
            "labels": int(len(trajectory["labels"])),
        }
    )
    return trajectory, outcome


def _validated_sigmas(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    sigmas = tuple(float(value) for value in values)
    if not sigmas:
        raise ValueError(f"{name} must not be empty")
    if len(set(sigmas)) != len(sigmas):
        raise ValueError(f"{name} must contain unique values")
    if any(
        not math.isfinite(value) or value <= 0.0 or value > 1.0
        for value in sigmas
    ):
        raise ValueError(f"{name} values must be finite in (0, 1]")
    return sigmas


def _stratified_calibration_indices(label_count: int, requested: int) -> tuple[int, ...]:
    """Choose interior early/middle/late anchors while retaining train anchors."""

    count = int(label_count)
    anchors = int(requested)
    if count <= 1:
        raise NativeClosedLoopError(
            "functional line search requires at least two labels per trajectory"
        )
    if anchors <= 0:
        raise ValueError("calibration_anchors_per_trajectory must be positive")
    anchors = min(anchors, count - 1)
    if anchors == 1:
        return (count // 2,)
    low = 1 if count > 2 else 0
    high = count - 2 if count > 2 else count - 1
    selected = {int(round(value)) for value in np.linspace(low, high, num=anchors)}
    for index in range(low, high + 1):
        if len(selected) >= anchors:
            break
        selected.add(index)
    if len(selected) < anchors:
        for index in range(count):
            if len(selected) >= anchors:
                break
            selected.add(index)
    result = tuple(sorted(selected))
    if len(result) != anchors or len(result) >= count:
        raise NativeClosedLoopError("could not form a nonempty train/calibration split")
    return result


def _partition_consistency_trajectories(
    trajectories: Sequence[Mapping[str, Any]], *, anchors_per_trajectory: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    for trajectory in trajectories:
        labels = list(trajectory["labels"])
        calibration_indices = set(
            _stratified_calibration_indices(len(labels), anchors_per_trajectory)
        )
        train_labels = [
            label for index, label in enumerate(labels) if index not in calibration_indices
        ]
        calibration_labels = [
            label for index, label in enumerate(labels) if index in calibration_indices
        ]
        if not train_labels or not calibration_labels:
            raise NativeClosedLoopError("empty multi-sigma train/calibration partition")
        train.append({**trajectory, "labels": train_labels})
        calibration.append({**trajectory, "labels": calibration_labels})
    return train, calibration


def _multi_sigma_anchor_forward(
    *,
    runtime: NativeV0VideoRuntime,
    trajectory: Mapping[str, Any],
    label: Mapping[str, Any],
    sigma_values: Sequence[float],
    pseudo_huber_c: float,
    require_grad: bool,
    capture_outputs: bool,
    backward_scales: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Compute one anchor, optionally streaming each loss through backward."""

    requested_sigmas = tuple(float(value) for value in sigma_values)
    if not requested_sigmas:
        raise ValueError("multi-sigma anchor requires at least one sigma")
    stream_backward = backward_scales is not None
    if stream_backward:
        if not require_grad:
            raise ValueError("streamed backward requires gradients")
        if capture_outputs:
            raise ValueError("streamed backward cannot capture calibration outputs")
        assert backward_scales is not None
        video_backward_scale, action_backward_scale = (
            float(backward_scales[0]),
            float(backward_scales[1]),
        )
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (video_backward_scale, action_backward_scale)
        ):
            raise ValueError("streamed backward scales must be finite and positive")
        video_backward_scale /= float(len(requested_sigmas))
        action_backward_scale /= float(len(requested_sigmas))
    else:
        video_backward_scale = 0.0
        action_backward_scale = 0.0

    _validate_label(label)
    context = materialize_context(trajectory, label)
    target_plan = label["teacher_z_t"].to(
        device=runtime.device, dtype=runtime.dtype
    )
    target_action = label["teacher_action"].to(
        device=runtime.device, dtype=runtime.dtype
    )
    video_mask = video_execution_mask(target_plan, label)
    video_losses: list[torch.Tensor] = []
    action_losses: list[torch.Tensor] = []
    output_rows: list[dict[str, Any]] = []

    for requested_sigma in requested_sigmas:
        video = runtime.student_video_x0_at_sigma(
            context,
            target_plan,
            sigma=float(requested_sigma),
            require_grad=require_grad,
        )
        # Video and action use different native scheduler shifts, so the same
        # requested noise mass can resolve to slightly different grid values.
        # Each modality must use its own resolved sigma; only the deployment
        # endpoint request is required to close to sigma=1 in both grids.
        if abs(float(requested_sigma) - 1.0) <= 1e-7:
            if abs(float(video.sigma) - 1.0) > 1e-7:
                raise NativeClosedLoopError(
                    "sigma=1 request does not close on both native grids"
                )
            expected_video_noise = context["epsilon_v"].to(
                device=runtime.device, dtype=runtime.dtype
            )
            if not torch.equal(
                video.noisy_state.detach()[video_mask],
                expected_video_noise.detach()[video_mask],
            ):
                raise NativeClosedLoopError(
                    "sigma=1 video state does not close to deployment noise"
                )
        video_target = video_consistency_map(
            video.noisy_state,
            target_plan,
            sigma=float(video.sigma),
        )
        video_loss = _masked_pseudo_huber_loss(
            video.consistency_prediction,
            video_target,
            video_mask,
            c=float(pseudo_huber_c),
        )
        output_row: dict[str, Any] | None = None
        if capture_outputs:
            output_row = {
                "requested_sigma": float(requested_sigma),
                "video_sigma": float(video.sigma),
                "video_prediction": video.consistency_prediction.detach()
                .float()[video_mask]
                .cpu(),
                "video_target": video_target.detach().float()[video_mask].cpu(),
            }
        if stream_backward:
            if not bool(torch.isfinite(video_loss).item()):
                raise FloatingPointError(
                    f"non-finite video loss at requested sigma {requested_sigma}"
                )
            (video_loss * video_backward_scale).backward()
            video_losses.append(video_loss.detach())
        else:
            video_losses.append(video_loss)
        del video_loss, video_target, video

        action = runtime.student_action_x0_at_sigma(
            context,
            target_plan.detach(),
            target_action,
            sigma=float(requested_sigma),
            require_grad=require_grad,
        )
        if abs(float(requested_sigma) - 1.0) <= 1e-7:
            if abs(float(action.sigma) - 1.0) > 1e-7:
                raise NativeClosedLoopError(
                    "sigma=1 request does not close on both native grids"
                )
            exact_action_state = {
                "input_noise": (
                    action.noisy_state.detach(),
                    label["teacher_action_input_noise"].to(
                        device=runtime.device, dtype=runtime.dtype
                    ),
                ),
                "timestep": (
                    action.timestep.detach(),
                    label["teacher_action_timestep"].to(
                        device=runtime.device, dtype=action.timestep.dtype
                    ),
                ),
                "valid_mask": (
                    action.valid_mask,
                    label["teacher_action_valid_mask"].to(
                        device=runtime.device, dtype=torch.bool
                    ),
                ),
            }
            for name, (student_value, teacher_value) in exact_action_state.items():
                if not torch.equal(student_value, teacher_value):
                    raise NativeClosedLoopError(
                        f"sigma=1 action deployment contract differs at {name}"
                    )
            if tuple(action.token_positions) != tuple(
                int(value) for value in label["teacher_action_token_positions"]
            ):
                raise NativeClosedLoopError("sigma=1 action token positions differ")
            if int(action.cache_valid_length) != int(
                label["teacher_action_cache_valid_length"]
            ):
                raise NativeClosedLoopError("sigma=1 action cache length differs")
        if action.valid_mask is None:
            raise NativeClosedLoopError("sigma action forward has no valid mask")
        action_mask = action_execution_mask(action.valid_mask, label)
        action_loss = _masked_pseudo_huber_loss(
            action.x0_prediction,
            target_action,
            action_mask,
            c=float(pseudo_huber_c),
        )
        if stream_backward:
            if not bool(torch.isfinite(action_loss).item()):
                raise FloatingPointError(
                    f"non-finite action loss at requested sigma {requested_sigma}"
                )
            (action_loss * action_backward_scale).backward()
            action_losses.append(action_loss.detach())
        else:
            action_losses.append(action_loss)
        if capture_outputs:
            assert output_row is not None
            output_row.update(
                {
                    "action_sigma": float(action.sigma),
                    "action_prediction": action.x0_prediction.detach()
                    .float()[action_mask]
                    .cpu(),
                    "action_target": target_action.detach()
                    .float()[action_mask]
                    .cpu(),
                }
            )
            output_rows.append(
                output_row
            )
        del action_loss, action
    return {
        "video_loss": torch.stack(video_losses).mean(),
        "action_loss": torch.stack(action_losses).mean(),
        "outputs": output_rows,
    }


def _evaluate_multi_sigma_calibration(
    *,
    runtime: NativeV0VideoRuntime,
    trajectories: Sequence[Mapping[str, Any]],
    sigma_values: Sequence[float],
    pseudo_huber_c: float,
    loss_reduction: str,
    video_weight: float,
    action_weight: float,
    capture_outputs: bool = True,
) -> dict[str, Any]:
    samples = [
        (trajectory, label)
        for trajectory in trajectories
        for label in trajectory["labels"]
    ]
    weights = _sample_optimization_weights(
        trajectories, loss_reduction=loss_reduction
    )
    if len(samples) != len(weights):
        raise NativeClosedLoopError("calibration weights do not match labels")
    video_total = 0.0
    action_total = 0.0
    outputs: list[dict[str, Any]] = []
    for (trajectory, label), weight in zip(samples, weights, strict=True):
        row = _multi_sigma_anchor_forward(
            runtime=runtime,
            trajectory=trajectory,
            label=label,
            sigma_values=sigma_values,
            pseudo_huber_c=pseudo_huber_c,
            require_grad=False,
            capture_outputs=bool(capture_outputs),
        )
        video_total += float(row["video_loss"].item()) * float(weight)
        action_total += float(row["action_loss"].item()) * float(weight)
        if capture_outputs:
            outputs.extend(row["outputs"])
    total = float(video_weight) * video_total + float(action_weight) * action_total
    if not math.isfinite(total):
        raise FloatingPointError("nonfinite multi-sigma calibration loss")
    return {
        "loss": total,
        "video_loss": video_total,
        "action_loss": action_total,
        "outputs": outputs,
    }


def _functional_closure_stats(
    before: Mapping[str, Any], after: Mapping[str, Any], *, modality: str
) -> dict[str, float]:
    if modality not in {"video", "action"}:
        raise ValueError(f"unsupported closure modality={modality!r}")
    before_rows = list(before["outputs"])
    after_rows = list(after["outputs"])
    if len(before_rows) != len(after_rows) or not before_rows:
        raise NativeClosedLoopError("functional closure rows are missing or misaligned")
    values: list[float] = []
    prediction_key = f"{modality}_prediction"
    target_key = f"{modality}_target"
    for old, new in zip(before_rows, after_rows, strict=True):
        for sigma_key in ("requested_sigma", "video_sigma", "action_sigma"):
            if float(old[sigma_key]) != float(new[sigma_key]):
                raise NativeClosedLoopError(
                    f"functional closure {sigma_key} order changed"
                )
        if not torch.equal(old[target_key], new[target_key]):
            raise NativeClosedLoopError("functional closure target changed")
        numerator = (new[prediction_key] - old[prediction_key]).norm()
        denominator = (old[target_key] - old[prediction_key]).norm().clamp(min=1e-12)
        value = float((numerator / denominator).item())
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite functional closure ratio")
        values.append(value)
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "median": float(torch.quantile(tensor, 0.5).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def _functional_candidate_gate(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    closure: Mapping[str, Mapping[str, float]],
    p95_max: float,
) -> dict[str, Any]:
    """Accept a resolved joint step without assuming a continuous BF16 response."""

    if not math.isfinite(float(p95_max)) or float(p95_max) <= 0.0:
        raise ValueError("functional closure p95 cap must be finite and positive")
    loss_keys = ("loss", "video_loss", "action_loss")
    for source_name, source in (("before", before), ("after", after)):
        for key in loss_keys:
            if key not in source or not math.isfinite(float(source[key])):
                raise FloatingPointError(
                    f"functional candidate has nonfinite {source_name}.{key}"
                )

    reasons: list[str] = []
    if float(after["loss"]) >= float(before["loss"]):
        reasons.append("combined_loss_not_improved")
    for modality in ("video", "action"):
        loss_key = f"{modality}_loss"
        if float(after[loss_key]) > float(before[loss_key]):
            reasons.append(f"{modality}_loss_regressed")
        stats = closure.get(modality)
        if stats is None:
            raise NativeClosedLoopError(
                f"functional candidate is missing {modality} closure"
            )
        median = float(stats["median"])
        p95 = float(stats["p95"])
        if not math.isfinite(median) or not math.isfinite(p95):
            raise FloatingPointError(
                f"functional candidate has nonfinite {modality} closure"
            )
        if median <= 0.0:
            reasons.append(f"{modality}_functionally_zero")
        if p95 > float(p95_max):
            reasons.append(f"{modality}_p95_overshoot")
    return {"accepted": not reasons, "rejection_reasons": reasons}


def _update_round_multi_sigma_x0(
    *,
    runtime: NativeV0VideoRuntime,
    optimizer: torch.optim.Optimizer,
    trajectories: Sequence[Mapping[str, Any]],
    round_id: int,
    video_weight: float,
    action_weight: float,
    pseudo_huber_c: float,
    max_grad_norm: float,
    loss_reduction: str,
    trainable_bank: str,
    optimizer_kind: str,
    sigma_values: Sequence[float],
    line_search_sigma_values: Sequence[float],
    line_search_update_norms: Sequence[float],
    calibration_anchors_per_trajectory: int,
    functional_closure_p95_max: float,
) -> dict[str, Any]:
    """One fresh-round normalized-SGD update selected on held-out anchors."""

    if runtime.adapter_kind != "joint_lora" or trainable_bank != "both":
        raise ValueError("multi_sigma_x0 requires one joint shared LoRA bank")
    if optimizer_kind != "functional_sgd" or not isinstance(
        optimizer, torch.optim.SGD
    ):
        raise ValueError("multi_sigma_x0 requires optimizer_kind='functional_sgd'")
    sigmas = _validated_sigmas(sigma_values, name="sigma_values")
    calibration_sigmas = _validated_sigmas(
        line_search_sigma_values, name="line_search_sigma_values"
    )
    candidate_norms = tuple(
        sorted({float(value) for value in line_search_update_norms}, reverse=True)
    )
    if not candidate_norms or any(
        not math.isfinite(value) or value <= 0.0 for value in candidate_norms
    ):
        raise ValueError("line_search_update_norms must be finite and positive")
    if (
        not math.isfinite(float(functional_closure_p95_max))
        or float(functional_closure_p95_max) <= 0.0
    ):
        raise ValueError("functional_closure_p95_max must be finite and positive")
    if 1.0 not in sigmas or 1.0 not in calibration_sigmas:
        raise ValueError("multi-sigma training and line search must include sigma=1")
    if float(video_weight) <= 0.0 or float(action_weight) <= 0.0:
        raise ValueError("multi_sigma_x0 requires positive joint video/action weights")

    current_policy_version = _policy_version(runtime)
    if any(
        str(trajectory["behavior_policy_version"]) != current_policy_version
        for trajectory in trajectories
    ):
        raise NativeClosedLoopError(
            "fresh trajectory policy version differs from the current Student"
        )
    if any(
        int(label["round_id"]) != int(round_id)
        for trajectory in trajectories
        for label in trajectory["labels"]
    ):
        raise NativeClosedLoopError("stale label entered the current round update")
    train_trajectories, calibration_trajectories = (
        _partition_consistency_trajectories(
            trajectories,
            anchors_per_trajectory=int(calibration_anchors_per_trajectory),
        )
    )
    train_samples = [
        (trajectory, label)
        for trajectory in train_trajectories
        for label in trajectory["labels"]
    ]
    train_weights = _sample_optimization_weights(
        train_trajectories, loss_reduction=loss_reduction
    )
    parameters = [parameter for _, parameter in runtime.trainable]
    if not parameters or any(parameter.dtype != torch.float32 for parameter in parameters):
        raise NativeClosedLoopError(
            "multi_sigma_x0 requires nonempty FP32 JointLoRA parameters"
        )
    optimizer.zero_grad(set_to_none=True)
    video_objective = 0.0
    action_objective = 0.0
    sample_rows: list[dict[str, Any]] = []
    for (trajectory, label), weight in zip(
        train_samples, train_weights, strict=True
    ):
        row = _multi_sigma_anchor_forward(
            runtime=runtime,
            trajectory=trajectory,
            label=label,
            sigma_values=sigmas,
            pseudo_huber_c=pseudo_huber_c,
            require_grad=True,
            capture_outputs=False,
            backward_scales=(
                float(weight) * float(video_weight),
                float(weight) * float(action_weight),
            ),
        )
        weighted_loss_value = (
            float(video_weight) * float(row["video_loss"].item())
            + float(action_weight) * float(row["action_loss"].item())
        ) * float(weight)
        if not math.isfinite(weighted_loss_value):
            raise FloatingPointError("nonfinite multi-sigma training loss")
        video_value = float(row["video_loss"].detach().item())
        action_value = float(row["action_loss"].detach().item())
        video_objective += video_value * float(weight)
        action_objective += action_value * float(weight)
        sample_rows.append(
            {
                "task": str(trajectory["task"]),
                "collection_id": str(label["collection_id"]),
                "macro_id": int(label["macro_id"]),
                "optimization_weight": float(weight),
                "video_loss": video_value,
                "action_loss": action_value,
            }
        )
    gradient_norm_pre_clip = torch.nn.utils.clip_grad_norm_(
        parameters, max_norm=float(max_grad_norm)
    )
    if not bool(torch.isfinite(gradient_norm_pre_clip).item()):
        raise FloatingPointError("nonfinite multi-sigma JointLoRA gradient")
    gradient_norm = float(
        sum(
            parameter.grad.detach().float().square().sum()
            for parameter in parameters
            if parameter.grad is not None
        )
        .sqrt()
        .item()
    )
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise NativeClosedLoopError("multi-sigma update produced no finite gradient")

    calibration_before = _evaluate_multi_sigma_calibration(
        runtime=runtime,
        trajectories=calibration_trajectories,
        sigma_values=calibration_sigmas,
        pseudo_huber_c=pseudo_huber_c,
        loss_reduction=loss_reduction,
        video_weight=video_weight,
        action_weight=action_weight,
    )
    calibration_before_log = {
        "loss": float(calibration_before["loss"]),
        "video_loss": float(calibration_before["video_loss"]),
        "action_loss": float(calibration_before["action_loss"]),
    }
    print(
        "FUNCTIONAL_CALIBRATION_BEFORE "
        + json.dumps(calibration_before_log, sort_keys=True),
        flush=True,
    )
    parameters_before = [parameter.detach().clone() for parameter in parameters]
    line_search_rows: list[dict[str, Any]] = []
    selected_norm: float | None = None
    selected_after: dict[str, Any] | None = None
    selected_closure: dict[str, dict[str, float]] | None = None
    try:
        for candidate_norm in candidate_norms:
            with torch.no_grad():
                for parameter, before in zip(
                    parameters, parameters_before, strict=True
                ):
                    if parameter.grad is None:
                        parameter.copy_(before)
                    else:
                        parameter.copy_(
                            before
                            - float(candidate_norm)
                            * parameter.grad.detach()
                            / float(gradient_norm)
                        )
            actual_candidate_norm = float(
                sum(
                    (parameter.detach() - before).float().square().sum()
                    for parameter, before in zip(
                        parameters, parameters_before, strict=True
                    )
                )
                .sqrt()
                .item()
            )
            if abs(actual_candidate_norm - float(candidate_norm)) > max(
                1e-6, float(candidate_norm) * 1e-3
            ):
                raise NativeClosedLoopError(
                    "functional candidate actual update norm differs from requested norm"
                )
            calibration_after = _evaluate_multi_sigma_calibration(
                runtime=runtime,
                trajectories=calibration_trajectories,
                sigma_values=calibration_sigmas,
                pseudo_huber_c=pseudo_huber_c,
                loss_reduction=loss_reduction,
                video_weight=video_weight,
                action_weight=action_weight,
            )
            closure = {
                modality: _functional_closure_stats(
                    calibration_before, calibration_after, modality=modality
                )
                for modality in ("video", "action")
            }
            gate = _functional_candidate_gate(
                before=calibration_before,
                after=calibration_after,
                closure=closure,
                p95_max=float(functional_closure_p95_max),
            )
            accepted = bool(gate["accepted"])
            candidate_row = {
                "candidate_update_norm": float(candidate_norm),
                "actual_candidate_update_norm": float(actual_candidate_norm),
                "calibration_loss": float(calibration_after["loss"]),
                "video_loss": float(calibration_after["video_loss"]),
                "action_loss": float(calibration_after["action_loss"]),
                "closure": closure,
                "accepted": bool(accepted),
                "rejection_reasons": list(gate["rejection_reasons"]),
            }
            line_search_rows.append(candidate_row)
            print(
                "FUNCTIONAL_LINE_SEARCH_CANDIDATE "
                + json.dumps(candidate_row, sort_keys=True),
                flush=True,
            )
            if accepted:
                selected_norm = float(candidate_norm)
                selected_after = calibration_after
                selected_closure = closure
                break
    finally:
        with torch.no_grad():
            for parameter, before in zip(parameters, parameters_before, strict=True):
                parameter.copy_(before)
    if selected_norm is None or selected_after is None or selected_closure is None:
        failure = {
            "calibration_before": calibration_before_log,
            "candidates": line_search_rows,
        }
        raise NativeClosedLoopError(
            "functional line search found no update satisfying the frozen contract: "
            + json.dumps(failure, sort_keys=True)
        )

    configured_learning_rates = {float(group["lr"]) for group in optimizer.param_groups}
    if len(configured_learning_rates) != 1:
        raise NativeClosedLoopError("optimizer parameter groups use different LRs")
    configured_learning_rate = configured_learning_rates.pop()
    effective_learning_rate = float(selected_norm) / float(gradient_norm)
    for group in optimizer.param_groups:
        group["lr"] = effective_learning_rate
    optimizer.step()
    for group in optimizer.param_groups:
        group["lr"] = configured_learning_rate
    actual_update_norm = float(
        sum(
            (parameter.detach() - before).float().square().sum()
            for parameter, before in zip(parameters, parameters_before, strict=True)
        )
        .sqrt()
        .item()
    )
    if abs(actual_update_norm - selected_norm) > max(1e-6, selected_norm * 1e-3):
        raise NativeClosedLoopError(
            "normalized SGD actual update norm differs from selected candidate"
        )
    if not all(bool(torch.isfinite(parameter).all().item()) for parameter in parameters):
        raise FloatingPointError("nonfinite JointLoRA parameter after functional SGD")
    return {
        "round_id": int(round_id),
        "objective": OBJECTIVE_MULTI_SIGMA_X0,
        "trainable_bank": "both",
        "optimizer_kind": "functional_sgd",
        "functional_acceptance": FUNCTIONAL_ACCEPTANCE_HELDOUT_NONREGRESSION,
        "optimizer_steps_this_round": 1,
        "fresh_trajectories": int(len(trajectories)),
        "fresh_samples": int(
            sum(len(trajectory["labels"]) for trajectory in trajectories)
        ),
        "train_samples": int(len(train_samples)),
        "calibration_samples": int(
            sum(len(trajectory["labels"]) for trajectory in calibration_trajectories)
        ),
        "sigma_values": list(sigmas),
        "line_search_sigma_values": list(calibration_sigmas),
        "loss_reduction": str(loss_reduction),
        "video_loss_objective_mean": float(video_objective),
        "action_loss_objective_mean": float(action_objective),
        "gradient_norm_pre_clip": float(gradient_norm_pre_clip.item()),
        "gradient_norm_post_clip": float(gradient_norm),
        "configured_learning_rate": float(configured_learning_rate),
        "effective_learning_rate": float(effective_learning_rate),
        "selected_update_norm": float(selected_norm),
        "actual_update_norm": float(actual_update_norm),
        "calibration_before": {
            key: float(calibration_before[key])
            for key in ("loss", "video_loss", "action_loss")
        },
        "calibration_after": {
            key: float(selected_after[key])
            for key in ("loss", "video_loss", "action_loss")
        },
        "selected_closure": selected_closure,
        "line_search": line_search_rows,
        "samples": sample_rows,
    }


def _scheduler_train_steps(scheduler: object) -> int:
    config = getattr(scheduler, "config", None)
    value = getattr(config, "num_train_timesteps", None)
    if value is None:
        value = getattr(scheduler, "num_train_timesteps", None)
    if value is None or int(value) <= 0:
        raise NativeClosedLoopError(
            "native scheduler does not expose positive num_train_timesteps"
        )
    return int(value)


def _native_schedule_pair(
    scheduler: object,
    *,
    frame_count: int,
    stride: int,
    generator: torch.Generator,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    train_steps = _scheduler_train_steps(scheduler)
    scheduler.set_timesteps(train_steps)
    timesteps = getattr(scheduler, "timesteps", None)
    sigmas = getattr(scheduler, "sigmas", None)
    if not isinstance(timesteps, torch.Tensor) or not isinstance(sigmas, torch.Tensor):
        raise NativeClosedLoopError("native scheduler lacks Tensor timesteps/sigmas")
    if int(timesteps.numel()) != train_steps or int(sigmas.numel()) < train_steps:
        raise NativeClosedLoopError("native scheduler grid length differs from training grid")
    usable_sigmas = sigmas[:train_steps]
    if train_steps > 1 and not bool(
        (usable_sigmas[:-1] >= usable_sigmas[1:]).all().item()
    ):
        raise NativeClosedLoopError("native scheduler sigma grid is not descending")
    start_index, end_index = sample_native_consistency_indices(
        int(frame_count),
        generator=generator,
        num_train_timesteps=train_steps,
        stride=int(stride),
    )
    return {
        "start_index": start_index,
        "end_index": end_index,
        "start_timestep": timesteps[start_index].detach().clone().to(device),
        "end_timestep": timesteps[end_index].detach().clone().to(device),
        "start_sigma": usable_sigmas[start_index].detach().clone().to(device),
        "end_sigma": usable_sigmas[end_index].detach().clone().to(device),
    }


def _flow_noised_state(
    clean: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    if tuple(clean.shape) != tuple(noise.shape):
        raise NativeClosedLoopError("clean/noise state shapes differ")
    sigma_view = _frame_sigma_view(sigma, clean)
    return (1.0 - sigma_view) * clean + sigma_view * noise


def _runtime_live_adapter_state(
    runtime: NativeV0VideoRuntime,
) -> dict[str, torch.Tensor]:
    state = {str(name): parameter for name, parameter in runtime.trainable}
    if not state:
        raise NativeClosedLoopError("runtime has no live adapter state")
    if any(parameter.dtype != torch.float32 for parameter in state.values()):
        raise NativeClosedLoopError("coherent TT requires FP32 adapter parameters")
    return state


def _trajectory_distinct_epoch_batches(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    generator: torch.Generator,
) -> list[list[tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    """Consume every label once while keeping at most one anchor/trajectory/batch."""

    if int(batch_size) <= 0:
        raise ValueError("effective batch size must be positive")
    queues: list[list[Mapping[str, Any]]] = []
    for trajectory in trajectories:
        labels = list(trajectory["labels"])
        if not labels:
            raise NativeClosedLoopError("training trajectory has no labels")
        order = torch.randperm(len(labels), generator=generator).tolist()
        queues.append([labels[int(index)] for index in order])
    batches: list[list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = []
    cursor = 0
    while any(queues):
        active = [index for index, queue in enumerate(queues) if queue]
        if not active:
            break
        chosen: list[int] = []
        for offset in range(len(queues)):
            candidate = (cursor + offset) % len(queues)
            if queues[candidate]:
                chosen.append(candidate)
            if len(chosen) >= int(batch_size):
                break
        if not chosen:
            raise NativeClosedLoopError("trajectory batcher made no progress")
        batch = [
            (trajectories[index], queues[index].pop())
            for index in chosen
        ]
        if len({id(trajectory) for trajectory, _label in batch}) != len(batch):
            raise NativeClosedLoopError("effective batch repeats a trajectory")
        batches.append(batch)
        cursor = (chosen[-1] + 1) % len(queues)
    expected = sum(len(trajectory["labels"]) for trajectory in trajectories)
    if sum(len(batch) for batch in batches) != expected:
        raise NativeClosedLoopError("trajectory batcher did not consume one epoch")
    return batches


def _trajectory_equal_epoch_batch_scales(
    trajectories: Sequence[Mapping[str, Any]],
    batches: Sequence[Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]],
) -> list[list[float]]:
    """Give every trajectory equal total weight while keeping mean step scale one."""

    if not trajectories or not batches:
        raise NativeClosedLoopError("trajectory-balanced epoch must not be empty")
    label_counts = {id(trajectory): len(trajectory["labels"]) for trajectory in trajectories}
    if len(label_counts) != len(trajectories) or any(
        count <= 0 for count in label_counts.values()
    ):
        raise NativeClosedLoopError(
            "trajectory-balanced epoch requires distinct nonempty trajectories"
        )
    step_count = float(len(batches))
    trajectory_count = float(len(trajectories))
    observed = {key: 0 for key in label_counts}
    result: list[list[float]] = []
    for batch in batches:
        scales: list[float] = []
        for trajectory, _label in batch:
            key = id(trajectory)
            if key not in label_counts:
                raise NativeClosedLoopError(
                    "trajectory-balanced batch contains an unknown trajectory"
                )
            observed[key] += 1
            scales.append(
                step_count / (trajectory_count * float(label_counts[key]))
            )
        result.append(scales)
    if observed != label_counts:
        raise NativeClosedLoopError(
            "trajectory-balanced batch coverage differs from trajectory labels"
        )
    return result


def _assert_exact_explicit_state(
    *,
    name: str,
    teacher_state: torch.Tensor,
    student_state: torch.Tensor,
) -> None:
    if not torch.equal(teacher_state.detach(), student_state.detach()):
        raise NativeClosedLoopError(f"{name} Teacher/Student start states differ")


def _coherent_tt_calibration_indices(
    label_count: int, requested: int
) -> tuple[int, ...]:
    """Select deterministic whole-trajectory calibration anchors."""

    count = int(label_count)
    anchors = int(requested)
    if count <= 0:
        raise NativeClosedLoopError("calibration trajectory has no labels")
    if anchors <= 0:
        raise ValueError("calibration_anchors_per_trajectory must be positive")
    if count > 1 and anchors < 2:
        raise ValueError(
            "coherent TT calibration needs at least two anchors to include early/late"
        )
    anchors = min(anchors, count)
    if anchors == count:
        return tuple(range(count))
    selected = tuple(
        int(round(value))
        for value in np.linspace(0, count - 1, num=anchors)
    )
    if (
        len(selected) != anchors
        or len(set(selected)) != anchors
        or selected[0] != 0
        or selected[-1] != count - 1
    ):
        raise NativeClosedLoopError(
            "could not form deterministic early/late calibration anchors"
        )
    return selected


def _coherent_tt_calibration_samples(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    anchors_per_trajectory: int,
) -> tuple[
    list[tuple[Mapping[str, Any], Mapping[str, Any]]],
    list[float],
    dict[int, list[int]],
]:
    """Select anchors without moving any trajectory across dataset roles."""

    samples: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    weighted_views: list[dict[str, Any]] = []
    anchor_indices: dict[int, list[int]] = {}
    for trajectory in trajectories:
        labels = list(trajectory["labels"])
        indices = _coherent_tt_calibration_indices(
            len(labels), int(anchors_per_trajectory)
        )
        selected_labels = [labels[index] for index in indices]
        weighted_views.append({**trajectory, "labels": selected_labels})
        samples.extend((trajectory, label) for label in selected_labels)
        anchor_indices[int(trajectory["seed"])] = list(indices)
    weights = _sample_optimization_weights(
        weighted_views,
        loss_reduction=LOSS_REDUCTION_MEAN_TRAJECTORIES,
    )
    return samples, weights, anchor_indices


def _prepare_coherent_tt_calibration_cases(
    *,
    runtime: NativeV0VideoRuntime,
    cases: Sequence[Mapping[str, Any]],
    live_state: Mapping[str, torch.Tensor],
    target_ema: LoRAEMAState,
) -> list[dict[str, Any]]:
    """Cache each frozen Teacher bridge and round-start EMA target once."""

    if not cases:
        raise NativeClosedLoopError("coherent TT calibration has no cases")
    prepared: list[dict[str, Any]] = []
    with torch.no_grad():
        for case in cases:
            trajectory = case["trajectory"]
            label = case["label"]
            video_schedule = case["video_schedule"]
            action_schedule = case["action_schedule"]
            context = materialize_context(trajectory, label)
            teacher_plan = label["teacher_z_t"].to(
                device=runtime.device, dtype=runtime.dtype
            )
            teacher_plan_timestep = label["teacher_z_t_timestep"].to(
                device=runtime.device
            )
            teacher_action = label["teacher_action"].to(
                device=runtime.device, dtype=runtime.dtype
            )
            epsilon_v = context["epsilon_v"].to(
                device=runtime.device, dtype=runtime.dtype
            )
            epsilon_a = context["epsilon_a"].to(
                device=runtime.device, dtype=runtime.dtype
            )
            video_start = _flow_noised_state(
                teacher_plan, epsilon_v, video_schedule["start_sigma"]
            )
            action_start = _flow_noised_state(
                teacher_action, epsilon_a, action_schedule["start_sigma"]
            )
            teacher_video = runtime.teacher_video_velocity_at_state(
                context,
                video_start,
                timestep=video_schedule["start_timestep"],
                sigma=video_schedule["start_sigma"],
            )
            teacher_action_forward = runtime.teacher_action_velocity_at_state(
                context,
                teacher_plan.detach(),
                teacher_plan_timestep,
                action_start,
                timestep=action_schedule["start_timestep"],
                sigma=action_schedule["start_sigma"],
            )
            prepared.append(
                {
                    **case,
                    "context": context,
                    "teacher_plan": teacher_plan.detach(),
                    "teacher_plan_timestep": teacher_plan_timestep.detach(),
                    "teacher_action": teacher_action.detach(),
                    "video_start": video_start.detach(),
                    "action_start": action_start.detach(),
                    "teacher_video_state": teacher_video.noisy_state.detach().clone(),
                    "teacher_action_state": (
                        teacher_action_forward.noisy_state.detach().clone()
                    ),
                    "bridged_video": teacher_euler_bridge(
                        teacher_video.noisy_state,
                        teacher_video.velocity,
                        sigma_start=video_schedule["start_sigma"],
                        sigma_end=video_schedule["end_sigma"],
                    ).detach(),
                    "bridged_action": teacher_euler_bridge(
                        teacher_action_forward.noisy_state,
                        teacher_action_forward.velocity,
                        sigma_start=action_schedule["start_sigma"],
                        sigma_end=action_schedule["end_sigma"],
                    ).detach(),
                    "action_fm_target": (epsilon_a - teacher_action).detach(),
                }
            )

        with target_ema.use_target(live_state):
            for case in prepared:
                video_schedule = case["video_schedule"]
                action_schedule = case["action_schedule"]
                target_video = runtime.student_video_consistency_at_state(
                    case["context"],
                    case["bridged_video"],
                    timestep=video_schedule["end_timestep"],
                    sigma=video_schedule["end_sigma"],
                    require_grad=False,
                )
                target_action = runtime.student_action_consistency_at_state(
                    case["context"],
                    case["teacher_plan"].detach(),
                    case["teacher_plan_timestep"],
                    case["bridged_action"],
                    timestep=action_schedule["end_timestep"],
                    sigma=action_schedule["end_sigma"],
                    require_grad=False,
                )
                case["target_video_prediction"] = (
                    target_video.consistency_prediction.detach().clone()
                )
                case["target_action_prediction"] = (
                    target_action.x0_prediction.detach().clone()
                )
                case["target_action_valid_mask"] = (
                    target_action.valid_mask.detach().clone()
                )
                case["target_action_token_positions"] = target_action.token_positions
                case["target_action_cache_valid_length"] = (
                    target_action.cache_valid_length
                )
                del case["bridged_video"], case["bridged_action"]
    return prepared


def _evaluate_coherent_tt_calibration(
    *,
    runtime: NativeV0VideoRuntime,
    cases: Sequence[Mapping[str, Any]],
    pseudo_huber_c: float,
    video_weight: float,
    action_weight: float,
    action_fm_weight: float,
) -> dict[str, float]:
    """Evaluate cached frozen targets using only the current online Student."""

    if not cases:
        raise NativeClosedLoopError("coherent TT calibration has no cases")
    video_total = 0.0
    action_total = 0.0
    action_fm_total = 0.0
    weight_total = 0.0
    for case in cases:
        trajectory = case["trajectory"]
        label = case["label"]
        video_schedule = case["video_schedule"]
        action_schedule = case["action_schedule"]
        weight = float(case["weight"])
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("coherent TT calibration weight must be finite and positive")

        online_video = runtime.student_video_consistency_at_state(
            case["context"],
            case["video_start"],
            timestep=video_schedule["start_timestep"],
            sigma=video_schedule["start_sigma"],
            require_grad=False,
        )
        _assert_exact_explicit_state(
            name="calibration video",
            teacher_state=case["teacher_video_state"],
            student_state=online_video.noisy_state,
        )
        video_mask = video_execution_mask(online_video.noisy_state, label)
        video_loss = _masked_pseudo_huber_loss(
            online_video.consistency_prediction,
            case["target_video_prediction"],
            video_mask,
            c=float(pseudo_huber_c),
        )

        online_action = runtime.student_action_consistency_at_state(
            case["context"],
            case["teacher_plan"].detach(),
            case["teacher_plan_timestep"],
            case["action_start"],
            timestep=action_schedule["start_timestep"],
            sigma=action_schedule["start_sigma"],
            require_grad=False,
        )
        _assert_exact_explicit_state(
            name="calibration action",
            teacher_state=case["teacher_action_state"],
            student_state=online_action.noisy_state,
        )
        if not torch.equal(online_action.valid_mask, case["target_action_valid_mask"]):
            raise NativeClosedLoopError("calibration online/EMA action masks differ")
        if online_action.token_positions != case["target_action_token_positions"]:
            raise NativeClosedLoopError(
                "calibration online/EMA action token positions differ"
            )
        if (
            online_action.cache_valid_length
            != case["target_action_cache_valid_length"]
        ):
            raise NativeClosedLoopError(
                "calibration online/EMA action cache lengths differ"
            )
        action_mask = action_execution_mask(online_action.valid_mask, label)
        action_loss = _masked_pseudo_huber_loss(
            online_action.x0_prediction,
            case["target_action_prediction"],
            action_mask,
            c=float(pseudo_huber_c),
        )
        action_fm_loss = action_velocity_mse_loss(
            online_action.velocity,
            case["action_fm_target"],
            action_mask,
        )
        values = (video_loss, action_loss, action_fm_loss)
        if not all(bool(torch.isfinite(value).item()) for value in values):
            raise FloatingPointError("nonfinite coherent TT calibration loss")
        video_total += float(video_loss.item()) * weight
        action_total += float(action_loss.item()) * weight
        action_fm_total += float(action_fm_loss.item()) * weight
        weight_total += weight

    if not math.isclose(weight_total, 1.0, rel_tol=1e-6, abs_tol=1e-8):
        raise NativeClosedLoopError(
            f"coherent TT calibration weights sum to {weight_total}, expected 1"
        )
    total = (
        float(video_weight) * video_total
        + float(action_weight) * action_total
        + float(action_fm_weight) * action_fm_total
    )
    if not math.isfinite(total):
        raise FloatingPointError("nonfinite coherent TT total calibration loss")
    return {
        "loss": total,
        "video_loss": video_total,
        "action_loss": action_total,
        "action_fm_loss": action_fm_total,
    }


def _assert_success_path_action_contract(
    *,
    runtime: NativeV0VideoRuntime,
    action: Any,
    label: Mapping[str, Any],
) -> None:
    """Keep the action solver state fixed while changing only its plan to z_S."""

    exact_state = {
        "input_noise": (
            action.action_input_noise.detach(),
            label["teacher_action_input_noise"].to(
                device=runtime.device, dtype=runtime.dtype
            ),
        ),
        "timestep": (
            action.action_timestep.detach(),
            label["teacher_action_timestep"].to(
                device=runtime.device, dtype=action.action_timestep.dtype
            ),
        ),
        "valid_mask": (
            action.valid_mask,
            label["teacher_action_valid_mask"].to(
                device=runtime.device, dtype=torch.bool
            ),
        ),
    }
    for name, (student_value, teacher_value) in exact_state.items():
        if not torch.equal(student_value, teacher_value):
            raise NativeClosedLoopError(
                f"success-path action same-state contract differs at {name}"
            )
    if tuple(action.token_positions) != tuple(
        int(value) for value in label["teacher_action_token_positions"]
    ):
        raise NativeClosedLoopError(
            "success-path action token positions differ from the coherent label"
        )
    if int(action.cache_valid_length) != int(
        label["teacher_action_cache_valid_length"]
    ):
        raise NativeClosedLoopError(
            "success-path action cache length differs from the coherent label"
        )


def _prepare_success_path_calibration_cases(
    *,
    runtime: NativeV0VideoRuntime,
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize immutable direct Teacher targets for fixed pre/post scoring."""

    if not cases:
        raise NativeClosedLoopError("success-path calibration has no cases")
    prepared: list[dict[str, Any]] = []
    for case in cases:
        trajectory = case["trajectory"]
        label = case["label"]
        _validate_label(label)
        context = materialize_context(trajectory, label)
        target_plan = label["teacher_z_t"].to(
            device=runtime.device, dtype=runtime.dtype
        )
        target_action = label["teacher_action"].to(
            device=runtime.device, dtype=runtime.dtype
        )
        epsilon_a = context["epsilon_a"].to(
            device=runtime.device, dtype=runtime.dtype
        )
        if tuple(epsilon_a.shape) != tuple(target_action.shape):
            raise NativeClosedLoopError(
                "success-path calibration action noise/target shapes differ"
            )
        prepared.append(
            {
                "trajectory": trajectory,
                "label": label,
                "context": context,
                "target_plan": target_plan,
                "target_action": target_action,
                "action_fm_target": epsilon_a - target_action,
                "weight": float(case["weight"]),
            }
        )
    return prepared


def _evaluate_success_path_calibration(
    *,
    runtime: NativeV0VideoRuntime,
    cases: Sequence[Mapping[str, Any]],
    pseudo_huber_c: float,
    video_weight: float,
    action_weight: float,
    action_fm_weight: float,
) -> dict[str, float]:
    """Score the exact deployment graph on fixed histories, targets, and noise."""

    if not cases:
        raise NativeClosedLoopError("success-path calibration has no cases")
    totals = {"video": 0.0, "action": 0.0, "action_fm": 0.0}
    weight_total = 0.0
    with torch.no_grad():
        for case in cases:
            weight = float(case["weight"])
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(
                    "success-path calibration weight must be finite and positive"
                )
            forward = runtime.student_video_forward(
                case["context"], detach_plan_for_action=True
            )
            _assert_success_path_action_contract(
                runtime=runtime, action=forward.action, label=case["label"]
            )
            video_mask = video_execution_mask(
                forward.plan.prepared_z_s, case["label"]
            )
            action_mask = action_execution_mask(
                forward.action.valid_mask, case["label"]
            )
            video_loss = _masked_pseudo_huber_loss(
                forward.plan.prepared_z_s,
                case["target_plan"],
                video_mask,
                c=float(pseudo_huber_c),
            )
            action_loss = _masked_pseudo_huber_loss(
                forward.action.endpoint,
                case["target_action"],
                action_mask,
                c=float(pseudo_huber_c),
            )
            action_fm_loss = action_velocity_mse_loss(
                forward.action.initial_velocity,
                case["action_fm_target"],
                action_mask,
            )
            values = (video_loss, action_loss, action_fm_loss)
            if not all(bool(torch.isfinite(value).item()) for value in values):
                raise FloatingPointError("nonfinite success-path calibration loss")
            totals["video"] += float(video_loss.item()) * weight
            totals["action"] += float(action_loss.item()) * weight
            totals["action_fm"] += float(action_fm_loss.item()) * weight
            weight_total += weight
            del forward, video_loss, action_loss, action_fm_loss

    if not math.isclose(weight_total, 1.0, rel_tol=1e-6, abs_tol=1e-8):
        raise NativeClosedLoopError(
            f"success-path calibration weights sum to {weight_total}, expected 1"
        )
    total = (
        float(video_weight) * totals["video"]
        + float(action_weight) * totals["action"]
        + float(action_fm_weight) * totals["action_fm"]
    )
    if not math.isfinite(total):
        raise FloatingPointError("nonfinite success-path total calibration loss")
    return {
        "loss": float(total),
        "video_loss": float(totals["video"]),
        "action_loss": float(totals["action"]),
        "action_fm_loss": float(totals["action_fm"]),
    }


def _update_round_success_path_tt(
    *,
    runtime: NativeV0VideoRuntime,
    optimizer: torch.optim.Optimizer,
    trajectories: Sequence[Mapping[str, Any]],
    round_id: int,
    video_weight: float,
    action_weight: float,
    action_fm_weight: float,
    pseudo_huber_c: float,
    max_grad_norm: float,
    effective_batch_size: int,
    inner_epochs: int,
    consistency_seed: int,
    calibration_anchors_per_trajectory: int,
    max_train_labels_per_trajectory: int | None = None,
    epoch_checkpoint_callback: Callable[
        [int, int, Mapping[str, Any]], str | None
    ]
    | None = None,
    success_path_resume_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train z_S and a_S(h_S, stopgrad(z_S)) against coherent direct targets."""

    if runtime.adapter_kind != "joint_lora":
        raise ValueError("success_path_v1 requires JointLoRA")
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("success_path_v1 requires AdamW")
    if int(inner_epochs) <= 0:
        raise ValueError("success_path_v1 inner_epochs must be positive")
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in (video_weight, action_weight, action_fm_weight)
    ):
        raise ValueError("success_path_v1 loss weights must be finite and positive")
    if int(calibration_anchors_per_trajectory) < 2:
        raise ValueError(
            "success_path_v1 calibration needs at least two early/late anchors"
        )
    if (
        max_train_labels_per_trajectory is not None
        and int(max_train_labels_per_trajectory) <= 0
    ):
        raise ValueError("max_train_labels_per_trajectory must be positive")

    completed_inner_epochs = 0
    resume_progress: Mapping[str, Any] | None = None
    resume_generator_state: torch.Tensor | None = None
    behavior_policy_version = _policy_version(runtime)
    if success_path_resume_state is not None:
        raw_completed = success_path_resume_state.get("completed_inner_epochs")
        if isinstance(raw_completed, bool) or not isinstance(raw_completed, int):
            raise ValueError("success-path resume completed_inner_epochs is invalid")
        completed_inner_epochs = int(raw_completed)
        if not 1 <= completed_inner_epochs < int(inner_epochs):
            raise ValueError(
                "success-path resume must stop before the total inner_epochs target"
            )
        behavior_policy_version = str(
            success_path_resume_state.get("behavior_policy_version", "")
        )
        if not behavior_policy_version:
            raise ValueError("success-path resume has no behavior policy version")
        raw_generator_state = success_path_resume_state.get(
            "consistency_generator_state"
        )
        if not isinstance(raw_generator_state, torch.Tensor):
            raise ValueError("success-path resume has no generator state")
        resume_generator_state = raw_generator_state.detach().clone().cpu()
        raw_progress = success_path_resume_state.get("success_path_progress")
        if not isinstance(raw_progress, Mapping):
            raise ValueError("success-path resume has no progress mapping")
        if raw_progress.get("schema") != SUCCESS_PATH_PROGRESS_SCHEMA:
            raise ValueError("success-path resume progress schema mismatch")
        if int(raw_progress.get("completed_inner_epochs", -1)) != (
            completed_inner_epochs
        ):
            raise ValueError("success-path resume epoch/progress mismatch")
        if str(raw_progress.get("behavior_policy_version", "")) != (
            behavior_policy_version
        ):
            raise ValueError("success-path resume behavior policy mismatch")
        resume_progress = raw_progress

    current_policy_version = _policy_version(runtime)
    if success_path_resume_state is None and (
        current_policy_version != behavior_policy_version
    ):
        raise NativeClosedLoopError(
            "success-path starting policy identity is inconsistent"
        )
    if any(
        str(trajectory["behavior_policy_version"]) != behavior_policy_version
        for trajectory in trajectories
    ):
        raise NativeClosedLoopError(
            "success-path data were not collected by the starting Student"
        )
    if any(
        int(label["round_id"]) != int(round_id)
        for trajectory in trajectories
        for label in trajectory["labels"]
    ):
        raise NativeClosedLoopError("stale label entered success_path_v1")

    train_trajectories = [
        trajectory
        for trajectory in trajectories
        if str(trajectory.get("dataset_role")) == "train"
    ]
    calibration_trajectories = [
        trajectory
        for trajectory in trajectories
        if str(trajectory.get("dataset_role")) == "calibration"
    ]
    unknown_roles = [
        str(trajectory.get("dataset_role"))
        for trajectory in trajectories
        if str(trajectory.get("dataset_role")) not in {"train", "calibration"}
    ]
    if unknown_roles:
        raise ValueError(
            f"success_path_v1 trajectories have unknown roles: {unknown_roles}"
        )
    if not train_trajectories or not calibration_trajectories:
        raise ValueError("success_path_v1 requires train and calibration trajectories")
    for trajectory in trajectories:
        for label in trajectory["labels"]:
            _validate_label(label)

    if max_train_labels_per_trajectory is not None:
        limit = int(max_train_labels_per_trajectory)
        train_trajectories = [
            {**trajectory, "labels": list(trajectory["labels"])[:limit]}
            for trajectory in train_trajectories
        ]
    if any(not trajectory["labels"] for trajectory in train_trajectories):
        raise NativeClosedLoopError("success_path_v1 has an empty train trajectory")

    live_state = _runtime_live_adapter_state(runtime)
    initial_state = {
        name: parameter.detach().clone() for name, parameter in live_state.items()
    }
    generator = torch.Generator(device="cpu").manual_seed(int(consistency_seed))
    if resume_generator_state is not None:
        generator.set_state(resume_generator_state)
    (
        calibration_samples,
        calibration_weights,
        calibration_anchor_indices,
    ) = _coherent_tt_calibration_samples(
        calibration_trajectories,
        anchors_per_trajectory=int(calibration_anchors_per_trajectory),
    )
    calibration_cases = _prepare_success_path_calibration_cases(
        runtime=runtime,
        cases=[
            {"trajectory": trajectory, "label": label, "weight": weight}
            for (trajectory, label), weight in zip(
                calibration_samples, calibration_weights, strict=True
            )
        ],
    )
    calibration_at_invocation_start = _evaluate_success_path_calibration(
        runtime=runtime,
        cases=calibration_cases,
        pseudo_huber_c=float(pseudo_huber_c),
        video_weight=float(video_weight),
        action_weight=float(action_weight),
        action_fm_weight=float(action_fm_weight),
    )
    if resume_progress is None:
        calibration_before = calibration_at_invocation_start
        calibration_history = [{"epoch": 0, **calibration_before}]
        step_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        epoch_rows: list[dict[str, Any]] = []
        totals = {"video": 0.0, "action": 0.0, "action_fm": 0.0}
    else:
        raw_calibration_history = resume_progress.get("calibration_history")
        raw_step_rows = resume_progress.get("steps")
        raw_sample_rows = resume_progress.get("samples")
        raw_epoch_rows = resume_progress.get("epoch_metrics")
        raw_totals = resume_progress.get("loss_totals")
        if not all(
            isinstance(value, list)
            for value in (
                raw_calibration_history,
                raw_step_rows,
                raw_sample_rows,
                raw_epoch_rows,
            )
        ) or not isinstance(raw_totals, Mapping):
            raise ValueError("success-path resume progress is malformed")
        calibration_history = deepcopy(raw_calibration_history)
        step_rows = deepcopy(raw_step_rows)
        sample_rows = deepcopy(raw_sample_rows)
        epoch_rows = deepcopy(raw_epoch_rows)
        totals = {
            name: float(raw_totals[name])
            for name in ("video", "action", "action_fm")
        }
        if [int(row.get("epoch", -1)) for row in epoch_rows] != list(
            range(1, completed_inner_epochs + 1)
        ):
            raise ValueError("success-path resume epoch metrics are incomplete")
        if [int(row.get("epoch", -1)) for row in calibration_history] != list(
            range(0, completed_inner_epochs + 1)
        ):
            raise ValueError("success-path resume calibration history is incomplete")
        if len(step_rows) != int(
            success_path_resume_state.get("global_optimizer_step", len(step_rows))
        ):
            raise ValueError("success-path resume optimizer step count mismatch")
        stored_calibration = calibration_history[-1]
        for name in ("loss", "video_loss", "action_loss", "action_fm_loss"):
            if not math.isclose(
                float(calibration_at_invocation_start[name]),
                float(stored_calibration[name]),
                rel_tol=1e-6,
                abs_tol=1e-8,
            ):
                raise NativeClosedLoopError(
                    "success-path resume calibration does not match checkpoint"
                )
        calibration_before = calibration_history[0]

    optimizer_steps_at_invocation_start = len(step_rows)

    for epoch_id in range(completed_inner_epochs + 1, int(inner_epochs) + 1):
        batches = _trajectory_distinct_epoch_batches(
            train_trajectories,
            batch_size=int(effective_batch_size),
            generator=generator,
        )
        batch_scales = _trajectory_equal_epoch_batch_scales(
            train_trajectories, batches
        )
        epoch_totals = {"video": 0.0, "action": 0.0, "action_fm": 0.0}
        epoch_step_start = len(step_rows)
        for batch, sample_scales in zip(batches, batch_scales, strict=True):
            step_id = len(step_rows)
            optimizer.zero_grad(set_to_none=True)
            batch_totals = {"video": 0.0, "action": 0.0, "action_fm": 0.0}
            batch_weight = 0.0
            for (trajectory, label), sample_scale in zip(
                batch, sample_scales, strict=True
            ):
                context = materialize_context(trajectory, label)
                target_plan = label["teacher_z_t"].to(
                    device=runtime.device, dtype=runtime.dtype
                )
                target_action = label["teacher_action"].to(
                    device=runtime.device, dtype=runtime.dtype
                )
                epsilon_a = context["epsilon_a"].to(
                    device=runtime.device, dtype=runtime.dtype
                )
                forward = runtime.student_video_forward(
                    context, detach_plan_for_action=True
                )
                _assert_success_path_action_contract(
                    runtime=runtime, action=forward.action, label=label
                )
                video_mask = video_execution_mask(
                    forward.plan.prepared_z_s, label
                )
                action_mask = action_execution_mask(
                    forward.action.valid_mask, label
                )
                video_loss = _masked_pseudo_huber_loss(
                    forward.plan.prepared_z_s,
                    target_plan,
                    video_mask,
                    c=float(pseudo_huber_c),
                )
                action_loss = _masked_pseudo_huber_loss(
                    forward.action.endpoint,
                    target_action,
                    action_mask,
                    c=float(pseudo_huber_c),
                )
                action_fm_loss = action_velocity_mse_loss(
                    forward.action.initial_velocity,
                    epsilon_a - target_action,
                    action_mask,
                )
                combined = (
                    float(video_weight) * video_loss
                    + float(action_weight) * action_loss
                    + float(action_fm_weight) * action_fm_loss
                ) * float(sample_scale)
                if not bool(torch.isfinite(combined).item()):
                    raise FloatingPointError("nonfinite success_path_v1 loss")
                combined.backward()
                values = {
                    "video": float(video_loss.detach().item()),
                    "action": float(action_loss.detach().item()),
                    "action_fm": float(action_fm_loss.detach().item()),
                }
                for key, value in values.items():
                    weighted = value * float(sample_scale)
                    totals[key] += weighted
                    epoch_totals[key] += weighted
                    batch_totals[key] += weighted
                batch_weight += float(sample_scale)
                sample_rows.append(
                    {
                        "epoch": int(epoch_id),
                        "step_id": int(step_id),
                        "task": str(trajectory["task"]),
                        "seed": int(trajectory["seed"]),
                        "collection_id": str(label["collection_id"]),
                        "macro_id": int(label["macro_id"]),
                        "video_loss": values["video"],
                        "action_loss": values["action"],
                        "action_fm_loss": values["action_fm"],
                        "optimization_weight": float(sample_scale),
                        "action_condition": "student_z_s_detached",
                    }
                )
                del (
                    forward,
                    video_loss,
                    action_loss,
                    action_fm_loss,
                    combined,
                )

            parameters = list(live_state.values())
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, max_norm=float(max_grad_norm)
            )
            if (
                not bool(torch.isfinite(gradient_norm).item())
                or float(gradient_norm.item()) <= 0.0
            ):
                raise FloatingPointError(
                    "nonfinite or zero success_path_v1 gradient"
                )
            optimizer.step()
            if not all(
                bool(torch.isfinite(value).all().item()) for value in parameters
            ):
                raise FloatingPointError(
                    "nonfinite success_path_v1 adapter parameter"
                )
            step_rows.append(
                {
                    "epoch": int(epoch_id),
                    "step_id": int(step_id),
                    "batch_size": int(len(batch)),
                    "batch_weight": float(batch_weight),
                    "trajectory_seeds": [
                        int(trajectory["seed"]) for trajectory, _label in batch
                    ],
                    "video_loss": batch_totals["video"] / float(batch_weight),
                    "action_loss": batch_totals["action"] / float(batch_weight),
                    "action_fm_loss": batch_totals["action_fm"]
                    / float(batch_weight),
                    "gradient_norm_pre_clip": float(gradient_norm.item()),
                }
            )

        epoch_steps = len(step_rows) - epoch_step_start
        if epoch_steps <= 0:
            raise NativeClosedLoopError("success_path_v1 epoch made no update")
        calibration = _evaluate_success_path_calibration(
            runtime=runtime,
            cases=calibration_cases,
            pseudo_huber_c=float(pseudo_huber_c),
            video_weight=float(video_weight),
            action_weight=float(action_weight),
            action_fm_weight=float(action_fm_weight),
        )
        calibration_history.append({"epoch": int(epoch_id), **calibration})
        epoch_rows.append(
            {
                "epoch": int(epoch_id),
                "optimizer_steps": int(epoch_steps),
                "train_samples": int(
                    sum(len(trajectory["labels"]) for trajectory in train_trajectories)
                ),
                "video_loss_mean": epoch_totals["video"] / float(epoch_steps),
                "action_loss_mean": epoch_totals["action"] / float(epoch_steps),
                "action_fm_loss_mean": epoch_totals["action_fm"]
                / float(epoch_steps),
                "calibration": calibration,
                "checkpoint": None,
            }
        )
        runtime._coherent_tt_generator_state = generator.get_state().clone()
        progress = {
            "schema": SUCCESS_PATH_PROGRESS_SCHEMA,
            "completed_inner_epochs": int(epoch_id),
            "behavior_policy_version": behavior_policy_version,
            "calibration_history": deepcopy(calibration_history),
            "epoch_metrics": deepcopy(epoch_rows),
            "steps": deepcopy(step_rows),
            "samples": deepcopy(sample_rows),
            "loss_totals": deepcopy(totals),
        }
        # The final epoch is committed by checkpoint_trajectory_update.pt.
        # Keeping the last resumable epoch strictly before completion avoids
        # an unusable "epoch" checkpoint if the process dies while publishing
        # the final checkpoint/summary pair; recovery deterministically replays
        # only the last epoch from this predecessor.
        if (
            epoch_checkpoint_callback is not None
            and int(epoch_id) < int(inner_epochs)
        ):
            epoch_rows[-1]["checkpoint"] = epoch_checkpoint_callback(
                int(epoch_id), int(len(step_rows)), progress
            )

    if len(step_rows) <= 1:
        raise NativeClosedLoopError(
            "success_path_v1 did not perform multiple optimizer steps"
        )
    changed_a = any(
        name.endswith(".lora_A")
        and not torch.equal(parameter.detach(), initial_state[name])
        for name, parameter in live_state.items()
    )
    changed_b = any(
        name.endswith(".lora_B")
        and not torch.equal(parameter.detach(), initial_state[name])
        for name, parameter in live_state.items()
    )
    if not changed_a or not changed_b:
        raise NativeClosedLoopError(
            "success_path_v1 did not change both JointLoRA factors"
        )
    runtime._coherent_tt_generator_state = generator.get_state().clone()
    calibration_after = calibration_history[-1]
    final_progress = {
        "schema": SUCCESS_PATH_PROGRESS_SCHEMA,
        "completed_inner_epochs": int(inner_epochs),
        "behavior_policy_version": behavior_policy_version,
        "calibration_history": deepcopy(calibration_history),
        "epoch_metrics": deepcopy(epoch_rows),
        "steps": deepcopy(step_rows),
        "samples": deepcopy(sample_rows),
        "loss_totals": deepcopy(totals),
    }
    return {
        "round_id": int(round_id),
        "objective": OBJECTIVE_COHERENT_TT_CONSISTENCY,
        "coherent_tt_variant": COHERENT_TT_VARIANT_SUCCESS_PATH_V1,
        "optimizer_kind": "adamw",
        "optimizer_steps_this_round": int(len(step_rows)),
        "optimizer_steps_this_invocation": int(
            len(step_rows) - optimizer_steps_at_invocation_start
        ),
        "inner_epochs": int(inner_epochs),
        "effective_batch_size": int(effective_batch_size),
        "fresh_trajectories": int(len(trajectories)),
        "train_trajectories": int(len(train_trajectories)),
        "calibration_trajectories": int(len(calibration_trajectories)),
        "calibration_samples": int(len(calibration_cases)),
        "calibration_anchors_per_trajectory": int(
            calibration_anchors_per_trajectory
        ),
        "calibration_anchor_indices": calibration_anchor_indices,
        "train_samples_per_epoch": int(
            sum(len(trajectory["labels"]) for trajectory in train_trajectories)
        ),
        "train_samples": int(len(sample_rows)),
        "max_train_labels_per_trajectory": max_train_labels_per_trajectory,
        "consistency_seed": int(consistency_seed),
        "action_condition": "student_z_s_detached",
        "teacher_targets": "artifact_z_t_and_a_t_given_h_s_z_t",
        "video_loss_mean": totals["video"] / float(len(step_rows)),
        "action_loss_mean": totals["action"] / float(len(step_rows)),
        "action_fm_loss_mean": totals["action_fm"] / float(len(step_rows)),
        "calibration_before": {
            key: float(calibration_before[key])
            for key in ("loss", "video_loss", "action_loss", "action_fm_loss")
        },
        "calibration_after": {
            key: float(calibration_after[key])
            for key in ("loss", "video_loss", "action_loss", "action_fm_loss")
        },
        "calibration_history": calibration_history,
        "lora_A_changed": bool(changed_a),
        "lora_B_changed": bool(changed_b),
        "epoch_metrics": epoch_rows,
        "steps": step_rows,
        "samples": sample_rows,
        "_success_path_progress": final_progress,
    }


def _update_round_coherent_tt_consistency(
    *,
    runtime: NativeV0VideoRuntime,
    optimizer: torch.optim.Optimizer,
    trajectories: Sequence[Mapping[str, Any]],
    round_id: int,
    video_weight: float,
    action_weight: float,
    action_fm_weight: float,
    pseudo_huber_c: float,
    max_grad_norm: float,
    effective_batch_size: int,
    inner_epochs: int,
    ema_decay: float,
    video_stride: int,
    action_stride: int,
    consistency_seed: int,
    calibration_anchors_per_trajectory: int,
) -> dict[str, Any]:
    """One bounded inner epoch of environment-on-policy coherent TT CD."""

    if runtime.adapter_kind != "joint_lora":
        raise ValueError("coherent TT consistency requires JointLoRA")
    if runtime.teacher is None:
        raise ValueError("coherent TT consistency requires a live frozen Teacher")
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("coherent TT consistency requires AdamW")
    if int(inner_epochs) != 1:
        raise ValueError("the first coherent TT baseline fixes inner_epochs=1")
    if any(float(value) <= 0.0 for value in (video_weight, action_weight, action_fm_weight)):
        raise ValueError("coherent TT loss weights must be positive")
    if any(int(value) <= 0 for value in (video_stride, action_stride)):
        raise ValueError("coherent TT native strides must be positive")
    if int(calibration_anchors_per_trajectory) < 2:
        raise ValueError(
            "coherent TT calibration needs at least two anchors to include early/late"
        )

    current_policy_version = _policy_version(runtime)
    if any(
        str(trajectory["behavior_policy_version"]) != current_policy_version
        for trajectory in trajectories
    ):
        raise NativeClosedLoopError(
            "fresh trajectory policy version differs from current Student"
        )
    if any(
        int(label["round_id"]) != int(round_id)
        for trajectory in trajectories
        for label in trajectory["labels"]
    ):
        raise NativeClosedLoopError("stale label entered coherent TT update")
    train_trajectories = [
        trajectory
        for trajectory in trajectories
        if str(trajectory.get("dataset_role")) == "train"
    ]
    calibration_trajectories = [
        trajectory
        for trajectory in trajectories
        if str(trajectory.get("dataset_role")) == "calibration"
    ]
    unknown_roles = [
        str(trajectory.get("dataset_role"))
        for trajectory in trajectories
        if str(trajectory.get("dataset_role")) not in {"train", "calibration"}
    ]
    if unknown_roles:
        raise ValueError(f"coherent TT trajectories have unknown roles: {unknown_roles}")
    if not train_trajectories or not calibration_trajectories:
        raise ValueError("coherent TT requires whole train and calibration trajectories")

    for trajectory in trajectories:
        for label in trajectory["labels"]:
            _validate_label(label)
    live_state = _runtime_live_adapter_state(runtime)
    ema = LoRAEMAState.from_online(live_state, decay=float(ema_decay))
    calibration_target_ema = LoRAEMAState.from_online(
        live_state, decay=float(ema_decay)
    )
    generator = torch.Generator(device="cpu").manual_seed(int(consistency_seed))
    calibration_generator = torch.Generator(device="cpu")
    calibration_generator.set_state(generator.get_state().clone())
    (
        calibration_samples,
        calibration_weights,
        calibration_anchor_indices,
    ) = _coherent_tt_calibration_samples(
        calibration_trajectories,
        anchors_per_trajectory=int(calibration_anchors_per_trajectory),
    )
    calibration_cases: list[dict[str, Any]] = []
    for (trajectory, label), weight in zip(
        calibration_samples, calibration_weights, strict=True
    ):
        teacher_plan = label["teacher_z_t"]
        teacher_action = label["teacher_action"]
        calibration_cases.append(
            {
                "trajectory": trajectory,
                "label": label,
                "video_schedule": _native_schedule_pair(
                    runtime.server.scheduler,
                    frame_count=int(teacher_plan.shape[2]),
                    stride=int(video_stride),
                    generator=calibration_generator,
                    device=runtime.device,
                ),
                "action_schedule": _native_schedule_pair(
                    runtime.server.action_scheduler,
                    frame_count=int(teacher_action.shape[2]),
                    stride=int(action_stride),
                    generator=calibration_generator,
                    device=runtime.device,
                ),
                "weight": float(weight),
            }
        )
    prepared_calibration_cases = _prepare_coherent_tt_calibration_cases(
        runtime=runtime,
        cases=calibration_cases,
        live_state=live_state,
        target_ema=calibration_target_ema,
    )
    calibration_before = _evaluate_coherent_tt_calibration(
        runtime=runtime,
        cases=prepared_calibration_cases,
        pseudo_huber_c=float(pseudo_huber_c),
        video_weight=float(video_weight),
        action_weight=float(action_weight),
        action_fm_weight=float(action_fm_weight),
    )
    batches = _trajectory_distinct_epoch_batches(
        train_trajectories,
        batch_size=int(effective_batch_size),
        generator=generator,
    )
    batch_scales = _trajectory_equal_epoch_batch_scales(
        train_trajectories,
        batches,
    )
    step_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    total_video = 0.0
    total_action = 0.0
    total_action_fm = 0.0
    total_samples = 0

    for step_id, (batch, sample_scales) in enumerate(
        zip(batches, batch_scales, strict=True)
    ):
        optimizer.zero_grad(set_to_none=True)
        batch_video = 0.0
        batch_action = 0.0
        batch_action_fm = 0.0
        batch_weight = 0.0
        for (trajectory, label), sample_scale in zip(
            batch, sample_scales, strict=True
        ):
            context = materialize_context(trajectory, label)
            teacher_plan = label["teacher_z_t"].to(
                device=runtime.device, dtype=runtime.dtype
            )
            teacher_plan_timestep = label["teacher_z_t_timestep"].to(
                device=runtime.device
            )
            teacher_action = label["teacher_action"].to(
                device=runtime.device, dtype=runtime.dtype
            )
            epsilon_v = context["epsilon_v"].to(
                device=runtime.device, dtype=runtime.dtype
            )
            epsilon_a = context["epsilon_a"].to(
                device=runtime.device, dtype=runtime.dtype
            )
            video_schedule = _native_schedule_pair(
                runtime.server.scheduler,
                frame_count=int(teacher_plan.shape[2]),
                stride=int(video_stride),
                generator=generator,
                device=runtime.device,
            )
            action_schedule = _native_schedule_pair(
                runtime.server.action_scheduler,
                frame_count=int(teacher_action.shape[2]),
                stride=int(action_stride),
                generator=generator,
                device=runtime.device,
            )
            video_start = _flow_noised_state(
                teacher_plan, epsilon_v, video_schedule["start_sigma"]
            )
            action_start = _flow_noised_state(
                teacher_action, epsilon_a, action_schedule["start_sigma"]
            )
            teacher_video = runtime.teacher_video_velocity_at_state(
                context,
                video_start,
                timestep=video_schedule["start_timestep"],
                sigma=video_schedule["start_sigma"],
            )
            teacher_action_forward = runtime.teacher_action_velocity_at_state(
                context,
                teacher_plan.detach(),
                teacher_plan_timestep,
                action_start,
                timestep=action_schedule["start_timestep"],
                sigma=action_schedule["start_sigma"],
            )
            bridged_video = teacher_euler_bridge(
                teacher_video.noisy_state,
                teacher_video.velocity,
                sigma_start=video_schedule["start_sigma"],
                sigma_end=video_schedule["end_sigma"],
            )
            bridged_action = teacher_euler_bridge(
                teacher_action_forward.noisy_state,
                teacher_action_forward.velocity,
                sigma_start=action_schedule["start_sigma"],
                sigma_end=action_schedule["end_sigma"],
            )

            with ema.use_target(live_state):
                target_video = runtime.student_video_consistency_at_state(
                    context,
                    bridged_video,
                    timestep=video_schedule["end_timestep"],
                    sigma=video_schedule["end_sigma"],
                    require_grad=False,
                )
                target_action = runtime.student_action_consistency_at_state(
                    context,
                    teacher_plan.detach(),
                    teacher_plan_timestep,
                    bridged_action,
                    timestep=action_schedule["end_timestep"],
                    sigma=action_schedule["end_sigma"],
                    require_grad=False,
                )

            online_video = runtime.student_video_consistency_at_state(
                context,
                video_start,
                timestep=video_schedule["start_timestep"],
                sigma=video_schedule["start_sigma"],
                require_grad=True,
            )
            _assert_exact_explicit_state(
                name="video",
                teacher_state=teacher_video.noisy_state,
                student_state=online_video.noisy_state,
            )
            video_mask = video_execution_mask(online_video.noisy_state, label)
            video_loss = _masked_pseudo_huber_loss(
                online_video.consistency_prediction,
                target_video.consistency_prediction.detach(),
                video_mask,
                c=float(pseudo_huber_c),
            )
            if not bool(torch.isfinite(video_loss).item()):
                raise FloatingPointError("nonfinite coherent TT video loss")
            (float(video_weight) * float(sample_scale) * video_loss).backward()

            online_action = runtime.student_action_consistency_at_state(
                context,
                teacher_plan.detach(),
                teacher_plan_timestep,
                action_start,
                timestep=action_schedule["start_timestep"],
                sigma=action_schedule["start_sigma"],
                require_grad=True,
            )
            _assert_exact_explicit_state(
                name="action",
                teacher_state=teacher_action_forward.noisy_state,
                student_state=online_action.noisy_state,
            )
            if not torch.equal(online_action.valid_mask, target_action.valid_mask):
                raise NativeClosedLoopError("online/EMA action masks differ")
            if online_action.token_positions != target_action.token_positions:
                raise NativeClosedLoopError("online/EMA action token positions differ")
            if online_action.cache_valid_length != target_action.cache_valid_length:
                raise NativeClosedLoopError("online/EMA action cache lengths differ")
            action_mask = action_execution_mask(online_action.valid_mask, label)
            action_loss = _masked_pseudo_huber_loss(
                online_action.x0_prediction,
                target_action.x0_prediction.detach(),
                action_mask,
                c=float(pseudo_huber_c),
            )
            action_fm_target = epsilon_a - teacher_action
            action_fm_loss = action_velocity_mse_loss(
                online_action.velocity,
                action_fm_target,
                action_mask,
            )
            action_combined = (
                float(action_weight) * action_loss
                + float(action_fm_weight) * action_fm_loss
            ) * float(sample_scale)
            if not bool(torch.isfinite(action_combined).item()):
                raise FloatingPointError("nonfinite coherent TT action loss")
            action_combined.backward()

            video_value = float(video_loss.detach().item())
            action_value = float(action_loss.detach().item())
            action_fm_value = float(action_fm_loss.detach().item())
            batch_video += video_value * float(sample_scale)
            batch_action += action_value * float(sample_scale)
            batch_action_fm += action_fm_value * float(sample_scale)
            batch_weight += float(sample_scale)
            total_video += video_value * float(sample_scale)
            total_action += action_value * float(sample_scale)
            total_action_fm += action_fm_value * float(sample_scale)
            total_samples += 1
            sample_rows.append(
                {
                    "step_id": int(step_id),
                    "task": str(trajectory["task"]),
                    "seed": int(trajectory["seed"]),
                    "collection_id": str(label["collection_id"]),
                    "macro_id": int(label["macro_id"]),
                    "video_start_indices": video_schedule["start_index"].tolist(),
                    "video_end_indices": video_schedule["end_index"].tolist(),
                    "action_start_indices": action_schedule["start_index"].tolist(),
                    "action_end_indices": action_schedule["end_index"].tolist(),
                    "video_loss": video_value,
                    "action_loss": action_value,
                    "action_fm_loss": action_fm_value,
                    "optimization_weight": float(sample_scale),
                }
            )
            del (
                target_video,
                target_action,
                online_video,
                online_action,
                teacher_video,
                teacher_action_forward,
                video_loss,
                action_loss,
                action_fm_loss,
                action_combined,
            )

        parameters = list(live_state.values())
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=float(max_grad_norm)
        )
        if not bool(torch.isfinite(gradient_norm).item()) or float(gradient_norm) <= 0.0:
            raise FloatingPointError("nonfinite or zero coherent TT gradient")
        optimizer.step()
        ema.after_committed_step(live_state, committed=True)
        if not all(bool(torch.isfinite(value).all().item()) for value in parameters):
            raise FloatingPointError("nonfinite coherent TT adapter parameter")
        step_rows.append(
            {
                "step_id": int(step_id),
                "batch_size": int(len(batch)),
                "batch_weight": float(batch_weight),
                "trajectory_seeds": [int(item[0]["seed"]) for item in batch],
                "video_loss": batch_video / float(batch_weight),
                "action_loss": batch_action / float(batch_weight),
                "action_fm_loss": batch_action_fm / float(batch_weight),
                "gradient_norm_pre_clip": float(gradient_norm.item()),
            }
        )

    if len(step_rows) <= 1:
        raise NativeClosedLoopError("coherent TT did not perform multiple optimizer steps")
    changed_a = any(
        name.endswith(".lora_A")
        and not torch.equal(
            parameter.detach(), calibration_target_ema._state[name]
        )
        for name, parameter in live_state.items()
    )
    changed_b = any(
        name.endswith(".lora_B")
        and not torch.equal(
            parameter.detach(), calibration_target_ema._state[name]
        )
        for name, parameter in live_state.items()
    )
    if not changed_a or not changed_b:
        raise NativeClosedLoopError(
            "coherent TT multi-step update did not change both LoRA factors"
        )
    calibration_after = _evaluate_coherent_tt_calibration(
        runtime=runtime,
        cases=prepared_calibration_cases,
        pseudo_huber_c=float(pseudo_huber_c),
        video_weight=float(video_weight),
        action_weight=float(action_weight),
        action_fm_weight=float(action_fm_weight),
    )
    runtime._coherent_tt_ema_state = ema
    runtime._coherent_tt_generator_state = generator.get_state().clone()
    return {
        "round_id": int(round_id),
        "objective": OBJECTIVE_COHERENT_TT_CONSISTENCY,
        "optimizer_kind": "adamw",
        "optimizer_steps_this_round": int(len(step_rows)),
        "inner_epochs": 1,
        "effective_batch_size": int(effective_batch_size),
        "fresh_trajectories": int(len(trajectories)),
        "train_trajectories": int(len(train_trajectories)),
        "calibration_trajectories": int(len(calibration_trajectories)),
        "calibration_samples": int(len(prepared_calibration_cases)),
        "calibration_anchors_per_trajectory": int(
            calibration_anchors_per_trajectory
        ),
        "calibration_anchor_indices": calibration_anchor_indices,
        "train_samples": int(total_samples),
        "ema_decay": float(ema_decay),
        "ema_updates": int(ema.committed_updates),
        "video_stride": int(video_stride),
        "action_stride": int(action_stride),
        "consistency_seed": int(consistency_seed),
        "video_loss_mean": total_video / float(len(step_rows)),
        "action_loss_mean": total_action / float(len(step_rows)),
        "action_fm_loss_mean": total_action_fm / float(len(step_rows)),
        "calibration_before": {
            key: float(calibration_before[key])
            for key in ("loss", "video_loss", "action_loss", "action_fm_loss")
        },
        "calibration_after": {
            key: float(calibration_after[key])
            for key in ("loss", "video_loss", "action_loss", "action_fm_loss")
        },
        "lora_A_changed": bool(changed_a),
        "lora_B_changed": bool(changed_b),
        "steps": step_rows,
        "samples": sample_rows,
    }


def _update_round(
    *,
    runtime: NativeV0VideoRuntime,
    optimizer: torch.optim.Optimizer,
    trajectories: Sequence[Mapping[str, Any]],
    round_id: int,
    video_weight: float,
    action_weight: float,
    action_velocity_weight: float,
    pseudo_huber_c: float,
    max_grad_norm: float,
    loss_reduction: str = LOSS_REDUCTION_MEAN_ALL,
    trainable_bank: str = "both",
    optimizer_kind: str = "adamw",
    max_update_norm: float | None = None,
    objective: str = OBJECTIVE_ENDPOINT,
    sigma_values: Sequence[float] = (),
    line_search_sigma_values: Sequence[float] = (1.0,),
    line_search_update_norms: Sequence[float] = (),
    calibration_anchors_per_trajectory: int = 0,
    functional_closure_p95_max: float = 0.50,
    action_fm_weight: float = 0.0,
    effective_batch_size: int = 4,
    inner_epochs: int = 1,
    ema_decay: float = 0.995,
    consistency_video_stride: int = 500,
    consistency_action_stride: int = 500,
    consistency_seed: int = 0,
    coherent_tt_variant: str = COHERENT_TT_VARIANT_BASELINE,
    success_path_max_train_labels_per_trajectory: int | None = None,
    epoch_checkpoint_callback: Callable[
        [int, int, Mapping[str, Any]], str | None
    ]
    | None = None,
    success_path_resume_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if str(objective) == OBJECTIVE_COHERENT_TT_CONSISTENCY:
        if trainable_bank != "both" or optimizer_kind != "adamw":
            raise ValueError(
                "coherent TT consistency requires both-bank AdamW"
            )
        if float(action_velocity_weight) != 0.0:
            raise ValueError(
                "coherent TT consistency does not use legacy action_velocity_weight"
            )
        if str(coherent_tt_variant) == COHERENT_TT_VARIANT_SUCCESS_PATH_V1:
            return _update_round_success_path_tt(
                runtime=runtime,
                optimizer=optimizer,
                trajectories=trajectories,
                round_id=round_id,
                video_weight=video_weight,
                action_weight=action_weight,
                action_fm_weight=action_fm_weight,
                pseudo_huber_c=pseudo_huber_c,
                max_grad_norm=max_grad_norm,
                effective_batch_size=effective_batch_size,
                inner_epochs=inner_epochs,
                consistency_seed=consistency_seed,
                calibration_anchors_per_trajectory=(
                    calibration_anchors_per_trajectory
                ),
                max_train_labels_per_trajectory=(
                    success_path_max_train_labels_per_trajectory
                ),
                epoch_checkpoint_callback=epoch_checkpoint_callback,
                success_path_resume_state=success_path_resume_state,
            )
        if str(coherent_tt_variant) != COHERENT_TT_VARIANT_BASELINE:
            raise ValueError(
                f"unsupported coherent_tt_variant={coherent_tt_variant!r}"
            )
        return _update_round_coherent_tt_consistency(
            runtime=runtime,
            optimizer=optimizer,
            trajectories=trajectories,
            round_id=round_id,
            video_weight=video_weight,
            action_weight=action_weight,
            action_fm_weight=action_fm_weight,
            pseudo_huber_c=pseudo_huber_c,
            max_grad_norm=max_grad_norm,
            effective_batch_size=effective_batch_size,
            inner_epochs=inner_epochs,
            ema_decay=ema_decay,
            video_stride=consistency_video_stride,
            action_stride=consistency_action_stride,
            consistency_seed=consistency_seed,
            calibration_anchors_per_trajectory=calibration_anchors_per_trajectory,
        )
    if str(objective) == OBJECTIVE_MULTI_SIGMA_X0:
        return _update_round_multi_sigma_x0(
            runtime=runtime,
            optimizer=optimizer,
            trajectories=trajectories,
            round_id=round_id,
            video_weight=video_weight,
            action_weight=action_weight,
            pseudo_huber_c=pseudo_huber_c,
            max_grad_norm=max_grad_norm,
            loss_reduction=loss_reduction,
            trainable_bank=trainable_bank,
            optimizer_kind=optimizer_kind,
            sigma_values=sigma_values,
            line_search_sigma_values=line_search_sigma_values,
            line_search_update_norms=line_search_update_norms,
            calibration_anchors_per_trajectory=calibration_anchors_per_trajectory,
            functional_closure_p95_max=functional_closure_p95_max,
        )
    if str(objective) != OBJECTIVE_ENDPOINT:
        raise ValueError(f"unsupported objective={objective!r}")
    if trainable_bank not in {"both", "video", "action"}:
        raise ValueError(f"unsupported trainable_bank={trainable_bank!r}")
    if optimizer_kind not in {"adamw", "trust_region_sgd"}:
        raise ValueError(f"unsupported optimizer_kind={optimizer_kind!r}")
    if optimizer_kind == "trust_region_sgd":
        if max_update_norm is None or not math.isfinite(float(max_update_norm)):
            raise ValueError("trust_region_sgd requires finite max_update_norm")
        if float(max_update_norm) <= 0.0:
            raise ValueError("max_update_norm must be positive")
    elif max_update_norm is not None:
        raise ValueError("max_update_norm is only valid for trust_region_sgd")

    samples = [
        (trajectory, label)
        for trajectory in trajectories
        for label in trajectory["labels"]
    ]
    if not samples:
        raise NativeClosedLoopError("fresh on-policy round has no labels")
    if any(int(label["round_id"]) != int(round_id) for _, label in samples):
        raise NativeClosedLoopError("stale label entered the current round update")
    current_policy_version = _policy_version(runtime)
    if any(
        str(trajectory["behavior_policy_version"]) != current_policy_version
        for trajectory in trajectories
    ):
        raise NativeClosedLoopError(
            "fresh trajectory policy version differs from the current Student"
        )
    optimization_weights = _sample_optimization_weights(
        trajectories, loss_reduction=str(loss_reduction)
    )
    if len(optimization_weights) != len(samples):
        raise NativeClosedLoopError("loss weights do not match fresh labels")

    parameters = [parameter for _, parameter in runtime.trainable]
    if not parameters:
        raise NativeClosedLoopError("round update has no trainable parameters")
    if runtime.adapter_kind == "dual_lora":
        selected_names = [name for name, _ in runtime.trainable]
        expected_names = [
            name for name, _ in runtime.adapter_trainable(trainable_bank)
        ]
        if selected_names != expected_names:
            raise NativeClosedLoopError(
                "runtime trainable manifest differs from selected dual bank"
            )
        if any(parameter.dtype != torch.float32 for parameter in parameters):
            raise NativeClosedLoopError("dual-bank trainable parameters are not FP32")
    optimizer.zero_grad(set_to_none=True)
    objective_totals = {"video": 0.0, "action": 0.0, "action_velocity": 0.0}
    macro_totals = {"video": 0.0, "action": 0.0, "action_velocity": 0.0}
    sample_rows: list[dict[str, Any]] = []

    for (trajectory, label), optimization_weight in zip(
        samples, optimization_weights, strict=True
    ):
        _validate_label(label)
        context = materialize_context(trajectory, label)
        saved_student_plan = label["student_z_s"].to(
            device=runtime.device, dtype=runtime.dtype
        )
        target_plan = label["teacher_z_t"].to(
            device=runtime.device, dtype=runtime.dtype
        )
        target_action = label["teacher_action"].to(
            device=runtime.device, dtype=runtime.dtype
        )
        target_velocity = label["teacher_action_initial_velocity"].to(
            device=runtime.device, dtype=runtime.dtype
        )

        prediction_plan: torch.Tensor | None = None
        prediction_action: Any | None = None
        if trainable_bank == "both":
            forward = runtime.student_video_forward(
                context, detach_plan_for_action=True
            )
            prediction_plan = forward.plan.prepared_z_s
            prediction_action = forward.action
        elif trainable_bank == "video":
            try:
                plan, _video_noise, _initial_latent = runtime._video_plan_student(
                    context
                )
                prediction_plan = plan.prepared_z_s
            finally:
                runtime.server.transformer.clear_cache(runtime.server.cache_name)
        else:
            prediction_action = runtime.student_action_on_plan(
                context, saved_student_plan, require_grad=True
            )

        if prediction_plan is not None and not torch.equal(
            prediction_plan.detach(), saved_student_plan
        ):
            raise NativeClosedLoopError(
                "Student plan changed before its fresh round was consumed"
            )
        if prediction_action is not None:
            exact_action_state = {
                "input_noise": (
                    prediction_action.action_input_noise.detach(),
                    label["teacher_action_input_noise"].to(
                        device=runtime.device, dtype=runtime.dtype
                    ),
                ),
                "timestep": (
                    prediction_action.action_timestep.detach(),
                    label["teacher_action_timestep"].to(
                        device=runtime.device,
                        dtype=prediction_action.action_timestep.dtype,
                    ),
                ),
                "valid_mask": (
                    prediction_action.valid_mask,
                    label["teacher_action_valid_mask"].to(
                        device=runtime.device, dtype=torch.bool
                    ),
                ),
            }
            for name, (student_value, teacher_value) in exact_action_state.items():
                if not torch.equal(student_value, teacher_value):
                    raise NativeClosedLoopError(
                        f"training same-state action contract differs at {name}"
                    )
            if tuple(prediction_action.token_positions) != tuple(
                int(value) for value in label["teacher_action_token_positions"]
            ):
                raise NativeClosedLoopError("training action token positions differ")
            if int(prediction_action.cache_valid_length) != int(
                label["teacher_action_cache_valid_length"]
            ):
                raise NativeClosedLoopError("training action cache length differs")

        zero = torch.zeros((), device=runtime.device, dtype=torch.float32)
        if prediction_plan is None:
            video_loss = zero
        else:
            video_mask = video_execution_mask(prediction_plan, label)
            video_loss = _masked_pseudo_huber_loss(
                prediction_plan,
                target_plan,
                video_mask,
                c=float(pseudo_huber_c),
            )
        if prediction_action is None:
            action_loss = zero
            action_velocity_loss = zero
        else:
            action_mask = action_execution_mask(
                prediction_action.valid_mask, label
            )
            action_loss = _masked_pseudo_huber_loss(
                prediction_action.endpoint,
                target_action,
                action_mask,
                c=float(pseudo_huber_c),
            )
            action_velocity_loss = action_velocity_mse_loss(
                prediction_action.initial_velocity,
                target_velocity,
                action_mask,
            )
        loss = (
            float(video_weight) * video_loss
            + float(action_weight) * action_loss
            + float(action_velocity_weight) * action_velocity_loss
        ) * float(optimization_weight)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("nonfinite on-policy OPD loss")
        loss.backward()
        video_loss_value = float(video_loss.detach().item())
        action_loss_value = float(action_loss.detach().item())
        action_velocity_loss_value = float(action_velocity_loss.detach().item())
        objective_totals["video"] += video_loss_value * float(
            optimization_weight
        )
        objective_totals["action"] += action_loss_value * float(
            optimization_weight
        )
        objective_totals["action_velocity"] += action_velocity_loss_value * float(
            optimization_weight
        )
        macro_totals["video"] += video_loss_value
        macro_totals["action"] += action_loss_value
        macro_totals["action_velocity"] += action_velocity_loss_value
        sample_rows.append(
            {
                "task": str(trajectory["task"]),
                "collection_id": str(label["collection_id"]),
                "macro_id": int(label["macro_id"]),
                "optimization_weight": float(optimization_weight),
                "video_loss": video_loss_value,
                "action_loss": action_loss_value,
                "action_velocity_loss": action_velocity_loss_value,
            }
        )

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters, max_norm=float(max_grad_norm)
    )
    if not bool(torch.isfinite(gradient_norm).item()):
        raise FloatingPointError("nonfinite Joint LoRA gradient")
    if float(gradient_norm.item()) <= 0.0:
        raise NativeClosedLoopError("fresh on-policy round produced zero gradient")

    post_clip_gradient_norm = float(
        sum(
            parameter.grad.detach().float().square().sum()
            for parameter in parameters
            if parameter.grad is not None
        )
        .sqrt()
        .item()
    )
    if not math.isfinite(post_clip_gradient_norm) or post_clip_gradient_norm <= 0.0:
        raise FloatingPointError("nonfinite or zero post-clip gradient norm")
    if runtime.adapter_kind == "dual_lora" and trainable_bank in {"video", "action"}:
        inactive_bank = "action" if trainable_bank == "video" else "video"
        inactive_gradients = [
            name
            for name, parameter in runtime.adapter_trainable(inactive_bank)
            if parameter.grad is not None
        ]
        if inactive_gradients:
            raise NativeClosedLoopError(
                f"inactive {inactive_bank} bank received gradients: {inactive_gradients}"
            )

    configured_learning_rates = {
        float(group["lr"]) for group in optimizer.param_groups
    }
    if len(configured_learning_rates) != 1:
        raise NativeClosedLoopError("optimizer parameter groups use different LRs")
    configured_learning_rate = configured_learning_rates.pop()
    effective_learning_rate = configured_learning_rate
    if optimizer_kind == "trust_region_sgd":
        effective_learning_rate = _trust_region_sgd_learning_rate(
            configured_learning_rate=configured_learning_rate,
            gradient_norm=post_clip_gradient_norm,
            max_update_norm=float(max_update_norm),
        )
        for group in optimizer.param_groups:
            group["lr"] = float(effective_learning_rate)

    parameters_before = [parameter.detach().clone() for parameter in parameters]

    # The only optimizer step in this fresh round.
    optimizer.step()
    actual_update_norm = float(
        sum(
            (parameter.detach().float() - before.float()).square().sum()
            for parameter, before in zip(parameters, parameters_before, strict=True)
        )
        .sqrt()
        .item()
    )
    actual_update_max_abs = max(
        float((parameter.detach().float() - before.float()).abs().max().item())
        for parameter, before in zip(parameters, parameters_before, strict=True)
    )
    for group in optimizer.param_groups:
        group["lr"] = float(configured_learning_rate)
    if optimizer_kind == "trust_region_sgd" and actual_update_norm > float(
        max_update_norm
    ) * 1.0001:
        raise NativeClosedLoopError(
            "actual SGD update exceeded the parameter-space trust region"
        )
    if not all(bool(torch.isfinite(parameter).all().item()) for parameter in parameters):
        raise FloatingPointError("nonfinite Joint LoRA parameter after update")
    return {
        "round_id": int(round_id),
        "trainable_bank": str(trainable_bank),
        "optimizer_kind": str(optimizer_kind),
        "optimizer_steps_this_round": 1,
        "fresh_samples": int(len(samples)),
        "fresh_trajectories": int(len(trajectories)),
        "loss_reduction": str(loss_reduction),
        "trajectory_label_counts": [
            {
                "task": str(trajectory["task"]),
                "collection_id": str(trajectory["labels"][0]["collection_id"]),
                "labels": int(len(trajectory["labels"])),
                "optimization_weight_sum": float(
                    sum(
                        weight
                        for weight, (sample_trajectory, _label) in zip(
                            optimization_weights, samples, strict=True
                        )
                        if sample_trajectory is trajectory
                    )
                ),
            }
            for trajectory in trajectories
        ],
        # Keep the legacy macro means comparable to the first pilot while
        # logging the actual trajectory-balanced optimization objective.
        "video_loss_mean": macro_totals["video"] / float(len(samples)),
        "action_loss_mean": macro_totals["action"] / float(len(samples)),
        "action_velocity_loss_mean": macro_totals["action_velocity"]
        / float(len(samples)),
        "video_loss_objective_mean": objective_totals["video"],
        "action_loss_objective_mean": objective_totals["action"],
        "action_velocity_loss_objective_mean": objective_totals[
            "action_velocity"
        ],
        "gradient_norm_pre_clip": float(gradient_norm.item()),
        "gradient_norm_post_clip": post_clip_gradient_norm,
        "configured_learning_rate": float(configured_learning_rate),
        "effective_learning_rate": float(effective_learning_rate),
        "max_update_norm": (
            None if max_update_norm is None else float(max_update_norm)
        ),
        "actual_update_norm": actual_update_norm,
        "actual_update_max_abs": actual_update_max_abs,
        "samples": sample_rows,
    }


def _save_checkpoint(
    *,
    path: Path,
    runtime: NativeV0VideoRuntime,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    round_id: int,
    global_optimizer_step: int,
    policy_version_before: str,
    policy_version_after: str,
    checkpoint_role: str | None = None,
    completed_inner_epochs: int | None = None,
    success_path_progress: Mapping[str, Any] | None = None,
    success_path_exact_identity: Mapping[str, Any] | None = None,
    success_path_finalization: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    adapter_kind = str(config["adapter_kind"])
    optimizer_state_dtypes = sorted(
        {
            str(value.dtype)
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        }
    )
    if adapter_kind in {"dual_lora", "joint_lora"} and any(
        value != "torch.float32" for value in optimizer_state_dtypes
    ):
        raise NativeClosedLoopError(
            f"adapter optimizer state is not FP32: {optimizer_state_dtypes}"
        )
    adapter_contract = runtime.adapter_contract()
    raw_contract_base_hashes = adapter_contract.get("base_parameter_hashes")
    base_parameter_hashes = (
        dict(raw_contract_base_hashes)
        if isinstance(raw_contract_base_hashes, Mapping)
        else {}
    )
    objective = str(config["objective"])
    coherent_tt_variant = str(
        config.get("coherent_tt_variant", COHERENT_TT_VARIANT_BASELINE)
    )
    ema_state = getattr(runtime, "_coherent_tt_ema_state", None)
    generator_state = getattr(runtime, "_coherent_tt_generator_state", None)
    if (
        objective == OBJECTIVE_COHERENT_TT_CONSISTENCY
        and coherent_tt_variant == COHERENT_TT_VARIANT_BASELINE
    ):
        if not isinstance(ema_state, LoRAEMAState):
            raise NativeClosedLoopError(
                "coherent TT checkpoint is missing adapter EMA state"
            )
        if not isinstance(generator_state, torch.Tensor):
            raise NativeClosedLoopError(
                "coherent TT checkpoint is missing sampler generator state"
            )
    success_path_contract = None
    if (
        objective == OBJECTIVE_COHERENT_TT_CONSISTENCY
        and coherent_tt_variant == COHERENT_TT_VARIANT_SUCCESS_PATH_V1
        and checkpoint_role in {"success_path_epoch", "success_path_final"}
    ):
        if completed_inner_epochs is None or not isinstance(
            success_path_progress, Mapping
        ):
            raise NativeClosedLoopError(
                "success-path checkpoint is missing exact-resume progress"
            )
        if int(success_path_progress.get("completed_inner_epochs", -1)) != int(
            completed_inner_epochs
        ):
            raise NativeClosedLoopError(
                "success-path checkpoint epoch/progress mismatch"
            )
        if not isinstance(success_path_exact_identity, Mapping):
            raise NativeClosedLoopError(
                "success-path checkpoint is missing exact input identity"
            )
        if checkpoint_role == "success_path_final":
            if not isinstance(success_path_finalization, Mapping):
                raise NativeClosedLoopError(
                    "success-path final checkpoint is missing finalization payload"
                )
            if (
                success_path_finalization.get(
                    "success_path_finalization_schema"
                )
                != SUCCESS_PATH_FINALIZATION_SCHEMA
                or "checkpoint_sha256" in success_path_finalization
                or success_path_finalization.get(
                    "commit_recovered_from_final_checkpoint"
                )
                is not False
            ):
                raise NativeClosedLoopError(
                    "success-path finalization payload contract mismatch"
                )
        elif success_path_finalization is not None:
            raise NativeClosedLoopError(
                "success-path epoch checkpoint must not contain finalization payload"
            )
        adapter_base_hashes = adapter_contract.get("base_parameter_hashes")
        cached_base_hashes = getattr(
            runtime, "_success_path_base_parameter_hashes", None
        )
        if isinstance(cached_base_hashes, Mapping):
            live_base_parameter_hashes = dict(cached_base_hashes)
        else:
            live_base_parameter_hashes = runtime.base_parameter_hashes()
            runtime._success_path_base_parameter_hashes = deepcopy(
                dict(live_base_parameter_hashes)
            )
        if (
            not isinstance(adapter_base_hashes, Mapping)
            or not adapter_base_hashes
            or dict(adapter_base_hashes) != dict(live_base_parameter_hashes)
        ):
            raise NativeClosedLoopError(
                "success-path live adapter/base parameter contract mismatch"
            )
        base_parameter_hashes = dict(live_base_parameter_hashes)
        success_path_contract = _success_path_resume_contract(
            config,
            exact_identity=success_path_exact_identity,
        )
    payload = {
            "schema": (
                DUAL_CHECKPOINT_SCHEMA
                if adapter_kind == "dual_lora"
                else CHECKPOINT_SCHEMA
            ),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "checkpoint_role": checkpoint_role,
            "adapter_kind": adapter_kind,
            "adapter_contract": adapter_contract,
            "adapter_contract_hash": _stable_hash(adapter_contract),
            "base_parameter_hashes": dict(base_parameter_hashes),
            "base_parameter_hashes_hash": _stable_hash(base_parameter_hashes),
            "adapter_state_dict": runtime.adapter_state(),
            "ema_adapter_state_dict": (
                {
                    name: value.detach().cpu()
                    for name, value in ema_state.target_state().items()
                }
                if isinstance(ema_state, LoRAEMAState)
                else None
            ),
            "ema_decay": (
                float(ema_state.decay)
                if isinstance(ema_state, LoRAEMAState)
                else None
            ),
            "ema_updates": (
                int(ema_state.committed_updates)
                if isinstance(ema_state, LoRAEMAState)
                else 0
            ),
            "consistency_generator_state": (
                generator_state.detach().cpu()
                if isinstance(generator_state, torch.Tensor)
                else None
            ),
            "objective": objective,
            "coherent_tt_variant": coherent_tt_variant,
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_kind": str(config["optimizer_kind"]),
            "optimizer_bank": str(config["trainable_bank"]),
            "optimizer_parameter_names": list(runtime.adapter_parameter_names),
            "optimizer_parameter_dtypes": sorted(
                {str(parameter.dtype) for _, parameter in runtime.trainable}
            ),
            "optimizer_state_dtypes": optimizer_state_dtypes,
            "round_id": int(round_id),
            "global_optimizer_step": int(global_optimizer_step),
            "completed_inner_epochs": completed_inner_epochs,
            "success_path_progress": (
                deepcopy(dict(success_path_progress))
                if isinstance(success_path_progress, Mapping)
                else None
            ),
            "success_path_resume_contract": success_path_contract,
            "success_path_resume_contract_hash": (
                _stable_hash(success_path_contract)
                if success_path_contract is not None
                else None
            ),
            "success_path_exact_identity": (
                deepcopy(dict(success_path_exact_identity))
                if isinstance(success_path_exact_identity, Mapping)
                else None
            ),
            "success_path_exact_identity_hash": (
                str(success_path_exact_identity.get("identity_hash", ""))
                if isinstance(success_path_exact_identity, Mapping)
                else None
            ),
            "success_path_finalization": (
                deepcopy(dict(success_path_finalization))
                if isinstance(success_path_finalization, Mapping)
                else None
            ),
            "success_path_finalization_hash": (
                _stable_hash(dict(success_path_finalization))
                if isinstance(success_path_finalization, Mapping)
                else None
            ),
            "policy_version_before": str(policy_version_before),
            "policy_version_after": str(policy_version_after),
            "task": str(config["task"]),
            "task_config": str(config["task_config"]),
            "task_entries": deepcopy(list(config["task_entries"])),
            "task_contract_hash": _stable_hash(config["task_entries"]),
            "adapter_seed": int(config["adapter_seed"]),
            "loss": {
                "objective": objective,
                "video": (
                    "masked_pseudo_huber_deployment_z_s_to_teacher_z_t"
                    if coherent_tt_variant
                    == COHERENT_TT_VARIANT_SUCCESS_PATH_V1
                    else "masked_pseudo_huber_teacher_bridge_to_ema_video_karras"
                    if objective == OBJECTIVE_COHERENT_TT_CONSISTENCY
                    else (
                        "masked_pseudo_huber_multi_sigma_x0"
                        if objective == OBJECTIVE_MULTI_SIGMA_X0
                        else "masked_pseudo_huber_endpoint"
                    )
                ),
                "action": (
                    "masked_pseudo_huber_deployment_action_on_detached_z_s_to_teacher_action"
                    if coherent_tt_variant
                    == COHERENT_TT_VARIANT_SUCCESS_PATH_V1
                    else "masked_pseudo_huber_teacher_bridge_to_ema_x_minus_sigma_v_on_detached_z_T"
                    if objective == OBJECTIVE_COHERENT_TT_CONSISTENCY
                    else (
                        "masked_pseudo_huber_multi_sigma_x0_on_detached_z_T"
                        if objective == OBJECTIVE_MULTI_SIGMA_X0
                        else "masked_pseudo_huber_endpoint_on_detached_z_T"
                    )
                ),
                "action_velocity": (
                    "none"
                    if objective in {
                        OBJECTIVE_MULTI_SIGMA_X0,
                        OBJECTIVE_COHERENT_TT_CONSISTENCY,
                    }
                    else "masked_mse_same_state"
                ),
                "action_fm": (
                    "masked_mse_epsilon_minus_teacher_action"
                    if objective == OBJECTIVE_COHERENT_TT_CONSISTENCY
                    else "none"
                ),
                "sigma_values": list(config["sigma_values"]),
                "line_search_sigma_values": list(
                    config["line_search_sigma_values"]
                ),
                "pseudo_huber_c": float(config["pseudo_huber_c"]),
                "video_weight": float(config["video_weight"]),
                "action_weight": float(config["action_weight"]),
                "action_velocity_weight": float(
                    config["action_velocity_weight"]
                ),
                "action_fm_weight": float(config["action_fm_weight"]),
                "action_target_condition": "teacher_on_teacher_z_t",
                "student_action_condition": (
                    "student_on_detached_student_z_s"
                    if coherent_tt_variant
                    == COHERENT_TT_VARIANT_SUCCESS_PATH_V1
                    else "student_on_teacher_z_t"
                    if objective == OBJECTIVE_COHERENT_TT_CONSISTENCY
                    else None
                ),
                "inner_epochs": int(config["inner_epochs"]),
                "effective_batch_size": int(config["effective_batch_size"]),
                "consistency_video_stride": int(
                    config["consistency_video_stride"]
                ),
                "consistency_action_stride": int(
                    config["consistency_action_stride"]
                ),
                "consistency_noise_source": config[
                    "consistency_noise_source"
                ],
                "reduction": str(config["loss_reduction"]),
                "retention_weight": 0.0,
            },
            "max_update_norm": config.get("max_update_norm"),
            "parent_checkpoint": config.get("initial_checkpoint"),
            "parent_rollout_bundle": config.get("rollout_bundle"),
            "resumed_optimizer_state": bool(config["resume_optimizer_state"]),
        }
    if checkpoint_role == "success_path_final":
        if not isinstance(success_path_exact_identity, Mapping):
            raise NativeClosedLoopError(
                "success-path final checkpoint has no exact identity"
            )
        _validate_success_path_finalization_payload(
            checkpoint=payload,
            checkpoint_path=path,
            config=config,
            exact_identity=success_path_exact_identity,
            task_contract_hash=str(payload["task_contract_hash"]),
            behavior_policy_version=str(policy_version_before),
            round_id=int(round_id),
        )
    _atomic_torch_save(path, payload)


def _fresh_optimizer(
    runtime: NativeV0VideoRuntime, config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    """Construct a branch-local optimizer; bundle optimizer state is forbidden."""

    parameters = [parameter for _, parameter in runtime.trainable]
    if not parameters:
        raise NativeClosedLoopError("optimizer has no selected adapter parameters")
    if str(config["optimizer_kind"]) == "adamw":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config["learning_rate"]),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
    else:
        optimizer = torch.optim.SGD(
            parameters,
            lr=float(config["learning_rate"]),
            momentum=float(config["sgd_momentum"]),
            weight_decay=0.0,
        )
    if optimizer.state:
        raise NativeClosedLoopError("fresh optimizer unexpectedly has state")
    return optimizer


def _configure_teacher_solver_from_config(
    runtime: NativeV0VideoRuntime,
    config: Mapping[str, Any],
) -> None:
    """Apply the normalized Teacher-only solver geometry to one runtime."""

    runtime.configure_teacher_solver(
        video_steps=int(config["teacher_video_steps"]),
        video_exec_steps=config["teacher_video_exec_steps"],
        action_steps=int(config["teacher_action_steps"]),
    )


def _configure_cuda_memory_limit(config: Mapping[str, Any]) -> None:
    """Bound this process's allocator so a colocated job keeps headroom."""

    fraction = config.get("cuda_memory_fraction")
    if fraction is None:
        return
    device = torch.device(str(config.get("device", "cuda:0")))
    if device.type != "cuda":
        raise ValueError("cuda_memory_fraction requires a CUDA device")
    torch.cuda.set_per_process_memory_fraction(float(fraction), device=device)


def _json_compatible_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact mapping representation emitted by ``_write_json``."""

    loaded = json.loads(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    )
    if not isinstance(loaded, dict):
        raise NativeClosedLoopError("JSON-compatible payload is not a mapping")
    return loaded


def _validate_success_path_finalization_payload(
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    config: Mapping[str, Any],
    exact_identity: Mapping[str, Any],
    task_contract_hash: str,
    behavior_policy_version: str,
    round_id: int,
) -> dict[str, Any]:
    """Validate and return the summary base embedded in a final checkpoint."""

    raw_finalization = checkpoint.get("success_path_finalization")
    if not isinstance(raw_finalization, Mapping):
        raise NativeClosedLoopError(
            "success-path final checkpoint has no recoverable finalization payload"
        )
    if str(checkpoint.get("success_path_finalization_hash", "")) != (
        _stable_hash(dict(raw_finalization))
    ):
        raise NativeClosedLoopError(
            "success-path finalization payload hash mismatch"
        )
    finalization = _json_compatible_mapping(raw_finalization)
    if "checkpoint_sha256" in finalization:
        raise NativeClosedLoopError(
            "success-path finalization payload is self-referential"
        )

    expected_fields = {
        "schema": TRAJECTORY_UPDATE_SUMMARY_SCHEMA,
        "status": "PASS",
        "run_mode": "trajectory_update",
        "adapter_kind": str(config["adapter_kind"]),
        "trainable_bank": str(config["trainable_bank"]),
        "adapter_seed": int(config["adapter_seed"]),
        "coherent_tt_variant": COHERENT_TT_VARIANT_SUCCESS_PATH_V1,
        "teacher_loaded": False,
        "teacher_target_source": "trajectory_artifacts",
        "teacher_controls_environment": False,
        "environment_execution": "SS",
        "collection_group_id": config["collection_group_id"],
        "behavior_policy_version": str(behavior_policy_version),
        "policy_version_after": str(checkpoint["policy_version_after"]),
        "task_contract_hash": str(task_contract_hash),
        "round_id": int(round_id),
        "global_optimizer_step": int(checkpoint["global_optimizer_step"]),
        "success_path_commit_schema": SUCCESS_PATH_COMMIT_SCHEMA,
        "success_path_finalization_schema": SUCCESS_PATH_FINALIZATION_SCHEMA,
        "commit_recovered_from_final_checkpoint": False,
        "success_path_exact_identity_hash": str(
            exact_identity.get("identity_hash", "")
        ),
        "adapter_contract_hash": str(
            checkpoint.get("adapter_contract_hash", "")
        ),
        "base_parameter_hashes_hash": str(
            checkpoint.get("base_parameter_hashes_hash", "")
        ),
    }
    if any(
        finalization.get(name) != value
        for name, value in expected_fields.items()
    ):
        raise NativeClosedLoopError(
            "success-path finalization summary/checkpoint contract mismatch"
        )
    if finalization.get("success_path_exact_identity") != dict(
        exact_identity
    ):
        raise NativeClosedLoopError(
            "success-path finalization exact input identity mismatch"
        )

    raw_artifacts = exact_identity.get("trajectory_artifacts")
    raw_teacher = exact_identity.get("teacher")
    if (
        not isinstance(raw_artifacts, list)
        or not all(isinstance(value, Mapping) for value in raw_artifacts)
        or not isinstance(raw_teacher, Mapping)
    ):
        raise NativeClosedLoopError(
            "success-path finalization exact path identity is malformed"
        )
    expected_artifacts = [str(value.get("path", "")) for value in raw_artifacts]
    if finalization.get("trajectory_artifacts") != expected_artifacts:
        raise NativeClosedLoopError(
            "success-path finalization trajectory paths mismatch"
        )
    try:
        finalization_teacher = Path(
            str(finalization["teacher_transformer"])
        ).expanduser().resolve()
        identity_teacher = Path(str(raw_teacher["path"])).expanduser().resolve()
        finalization_checkpoint = Path(
            str(finalization["checkpoint"])
        ).expanduser().resolve()
    except (KeyError, TypeError, ValueError) as error:
        raise NativeClosedLoopError(
            "success-path finalization path metadata is malformed"
        ) from error
    if finalization_teacher != identity_teacher:
        raise NativeClosedLoopError(
            "success-path finalization Teacher path mismatch"
        )
    if finalization_checkpoint != checkpoint_path.expanduser().resolve():
        raise NativeClosedLoopError(
            "success-path finalization checkpoint path mismatch"
        )

    resumed = config.get("initial_checkpoint") is not None
    if (
        finalization.get("fresh_optimizer") is not (not resumed)
        or finalization.get("resumed_optimizer_state") is not resumed
    ):
        raise NativeClosedLoopError(
            "success-path finalization optimizer-resume mode mismatch"
        )
    expected_initial_checkpoint = (
        None
        if not resumed
        else str(
            Path(str(config["initial_checkpoint"]))
            .expanduser()
            .resolve()
        )
    )
    if finalization.get("initial_checkpoint") != expected_initial_checkpoint:
        raise NativeClosedLoopError(
            "success-path finalization parent checkpoint mismatch"
        )

    starting_step = finalization.get("starting_global_optimizer_step")
    steps_this_run = finalization.get("optimizer_steps_this_run")
    global_step = checkpoint.get("global_optimizer_step")
    if (
        isinstance(starting_step, bool)
        or not isinstance(starting_step, int)
        or int(starting_step) < 0
        or isinstance(steps_this_run, bool)
        or not isinstance(steps_this_run, int)
        or int(steps_this_run) <= 0
        or int(starting_step) + int(steps_this_run) != int(global_step)
    ):
        raise NativeClosedLoopError(
            "success-path finalization optimizer progress mismatch"
        )
    update = finalization.get("update")
    if not isinstance(update, Mapping) or any(
        update.get(name) != value
        for name, value in {
            "round_id": int(round_id),
            "objective": OBJECTIVE_COHERENT_TT_CONSISTENCY,
            "coherent_tt_variant": COHERENT_TT_VARIANT_SUCCESS_PATH_V1,
            "optimizer_steps_this_round": int(global_step),
            "optimizer_steps_this_invocation": int(steps_this_run),
        }.items()
    ):
        raise NativeClosedLoopError(
            "success-path finalization update/progress mismatch"
        )
    return finalization


def _guard_success_path_committed_output(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    exact_identity: Mapping[str, Any],
    task_contract_hash: str,
    behavior_policy_version: str,
    round_id: int,
) -> dict[str, Any] | None:
    """Validate a commit or recover its summary from one atomic final file."""

    checkpoint_path = output_dir / "checkpoint_trajectory_update.pt"
    summary_path = output_dir / "summary.json"
    if not checkpoint_path.exists() and not summary_path.exists():
        return None
    if checkpoint_path.exists() and not summary_path.exists():
        try:
            checkpoint_hash_before, _ = _sha256_file(checkpoint_path)
            loaded = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(loaded, Mapping):
                raise TypeError("final checkpoint is not a mapping")
            _validate_success_path_resume_checkpoint(
                loaded,
                config=config,
                expected_task_contract_hash=str(task_contract_hash),
                expected_behavior_policy_version=behavior_policy_version,
                expected_round_id=round_id,
                expected_exact_identity=exact_identity,
                expected_checkpoint_role="success_path_final",
            )
            finalization = _validate_success_path_finalization_payload(
                checkpoint=loaded,
                checkpoint_path=checkpoint_path,
                config=config,
                exact_identity=exact_identity,
                task_contract_hash=str(task_contract_hash),
                behavior_policy_version=behavior_policy_version,
                round_id=round_id,
            )
            checkpoint_hash_after, _ = _sha256_file(checkpoint_path)
            if checkpoint_hash_after != checkpoint_hash_before:
                raise NativeClosedLoopError(
                    "success-path final checkpoint changed during recovery"
                )
        except Exception as error:
            raise NativeClosedLoopError(
                "success-path final checkpoint exists without a summary commit "
                f"and is invalid: {checkpoint_path}"
            ) from error
        recovered_summary = deepcopy(finalization)
        recovered_summary.update(
            {
                "checkpoint_sha256": checkpoint_hash_after,
                "commit_recovered_from_final_checkpoint": True,
            }
        )
        if summary_path.exists():
            raise NativeClosedLoopError(
                "success-path summary appeared during finalization recovery"
            )
        _write_json(summary_path, recovered_summary)
        return recovered_summary
    if summary_path.exists() and not checkpoint_path.exists():
        raise NativeClosedLoopError(
            "success-path summary commit exists without its final checkpoint"
        )

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeClosedLoopError(
            "success-path summary is not a valid atomic commit marker"
        ) from error
    if not isinstance(summary, Mapping):
        raise NativeClosedLoopError(
            "success-path summary commit marker is not a mapping"
        )
    expected_summary_fields = {
        "schema": TRAJECTORY_UPDATE_SUMMARY_SCHEMA,
        "status": "PASS",
        "success_path_commit_schema": SUCCESS_PATH_COMMIT_SCHEMA,
        "task_contract_hash": str(task_contract_hash),
        "behavior_policy_version": str(behavior_policy_version),
        "round_id": int(round_id),
        "success_path_exact_identity_hash": str(
            exact_identity.get("identity_hash", "")
        ),
    }
    if any(
        summary.get(name) != value
        for name, value in expected_summary_fields.items()
    ):
        raise NativeClosedLoopError(
            "success-path summary atomic commit contract mismatch"
        )
    if summary.get("success_path_exact_identity") != dict(exact_identity):
        raise NativeClosedLoopError(
            "success-path summary exact input identity mismatch"
        )
    try:
        summary_checkpoint = Path(str(summary["checkpoint"])).expanduser().resolve()
    except (KeyError, TypeError, ValueError) as error:
        raise NativeClosedLoopError(
            "success-path summary checkpoint path is malformed"
        ) from error
    if summary_checkpoint != checkpoint_path.resolve():
        raise NativeClosedLoopError(
            "success-path summary points to a different final checkpoint"
        )
    observed_checkpoint_hash, _ = _sha256_file(checkpoint_path)
    if str(summary.get("checkpoint_sha256", "")) != observed_checkpoint_hash:
        raise NativeClosedLoopError(
            "success-path committed checkpoint content hash mismatch"
        )
    try:
        loaded = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(loaded, Mapping):
            raise TypeError("final checkpoint is not a mapping")
        _validate_success_path_resume_checkpoint(
            loaded,
            config=config,
            expected_task_contract_hash=str(task_contract_hash),
            expected_behavior_policy_version=behavior_policy_version,
            expected_round_id=round_id,
            expected_exact_identity=exact_identity,
            expected_checkpoint_role="success_path_final",
        )
        finalization = _validate_success_path_finalization_payload(
            checkpoint=loaded,
            checkpoint_path=checkpoint_path,
            config=config,
            exact_identity=exact_identity,
            task_contract_hash=str(task_contract_hash),
            behavior_policy_version=behavior_policy_version,
            round_id=round_id,
        )
        checkpoint_hash_after, _ = _sha256_file(checkpoint_path)
        if checkpoint_hash_after != observed_checkpoint_hash:
            raise NativeClosedLoopError(
                "success-path final checkpoint changed during commit validation"
            )
    except Exception as error:
        raise NativeClosedLoopError(
            "success-path committed checkpoint failed load/contract validation"
        ) from error
    expected_summary = deepcopy(finalization)
    recovered = summary.get("commit_recovered_from_final_checkpoint")
    if not isinstance(recovered, bool):
        raise NativeClosedLoopError(
            "success-path summary recovery audit field is malformed"
        )
    expected_summary.update(
        {
            "checkpoint_sha256": observed_checkpoint_hash,
            "commit_recovered_from_final_checkpoint": recovered,
        }
    )
    if dict(summary) != expected_summary:
        raise NativeClosedLoopError(
            "success-path summary/finalization payload mismatch"
        )
    raise FileExistsError(
        "success-path trajectory_update output is already complete and committed"
    )


def _pre_update_solver_closure(
    runtime: NativeV0VideoRuntime,
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the real deployment/sigma=1 closure before any optimizer step."""

    if not trajectories:
        raise NativeClosedLoopError("pre-update solver closure has no trajectory")
    labels = list(trajectories[0].get("labels", []))
    if not labels:
        raise NativeClosedLoopError("pre-update solver closure has no label")
    selected = [labels[0]]
    if len(labels) > 1:
        selected.append(labels[-1])

    # Import lazily because the standalone verifier imports this trainer's
    # data-contract helpers.  At runtime all trainer symbols are already
    # defined, and the verifier only executes the two selected real anchors.
    from experiments.verify_multi_sigma_solver_closure import _anchor_closure

    anchors = [
        _anchor_closure(runtime, trajectories[0], label) for label in selected
    ]
    passed = all(str(anchor.get("status")) == "PASS" for anchor in anchors)
    return {
        "schema": "waopd_pre_update_solver_closure_v1",
        "status": "PASS" if passed else "FAIL",
        "trajectory_collection_id": str(
            trajectories[0].get("collection_id", "unknown")
        ),
        "anchors": anchors,
    }


def _make_teacher_free_branch_runtime(
    *,
    bundle: Mapping[str, Any],
    config: Mapping[str, Any],
    student: Path,
    save_root: Path,
) -> NativeV0VideoRuntime:
    contract = bundle["adapter_contract"]
    if not isinstance(contract, Mapping):
        raise ValueError("rollout_bundle has no adapter contract")
    try:
        rank = int(contract["rank"])
        alpha = float(contract["alpha"])
        dropout = float(contract["dropout"])
        block_indices = tuple(int(value) for value in contract["block_indices"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("rollout_bundle adapter contract is malformed") from error
    runtime = NativeV0VideoRuntime(
        student_checkpoint=student,
        teacher_transformer=None,
        device=str(config.get("device", "cuda:0")),
        save_root=save_root,
        enable_offload=bool(config.get("enable_offload", True)),
        official_offload_parity=bool(
            config.get("official_offload_parity", True)
        ),
        adapter_rank=rank,
        adapter_state=None,
        adapter_kind="dual_lora",
        lora_alpha=alpha,
        lora_dropout=dropout,
        lora_block_indices=block_indices,
    )
    try:
        # Load the already-validated in-memory payload.  Passing the full
        # trajectory bundle through NativeV0VideoRuntime's weights-only path
        # would reject legitimate NumPy simulator observations.
        load_dual_mode_lora_checkpoint(
            runtime.server.transformer, runtime.adapter_info, bundle
        )
        runtime.select_adapter_trainable_bank(str(config["trainable_bank"]))
        if runtime.adapter_contract() != dict(contract):
            raise NativeClosedLoopError(
                "branch runtime adapter contract differs after exact restore"
            )
        restored_policy = _policy_version(runtime)
        if restored_policy != str(bundle["behavior_policy_version"]):
            raise NativeClosedLoopError(
                "branch runtime policy hash differs after exact bundle restore"
            )
    except BaseException:
        runtime.close()
        raise
    return runtime


def _run_branch_update(
    *,
    config: Mapping[str, Any],
    student: Path,
    output_dir: Path,
    task_contract_hash: str,
) -> dict[str, Any]:
    """Consume one collect-once bundle with one teacher-free optimizer step."""

    bundle_path = Path(str(config["rollout_bundle"])).expanduser().resolve()
    bundle = _load_rollout_bundle(
        bundle_path,
        expected_task_entries=config["task_entries"],
        expected_task_contract_hash=task_contract_hash,
        expected_student=student,
    )
    # The branch restores the exact adapter tensors from collection.  Keep
    # checkpoint metadata tied to that same initialization identity instead
    # of inheriting a branch config default derived from the rollout seed.
    branch_config = dict(config)
    branch_config["adapter_seed"] = int(bundle["adapter_seed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = _make_teacher_free_branch_runtime(
        bundle=bundle,
        config=branch_config,
        student=student,
        save_root=output_dir / "native_save",
    )
    try:
        optimizer = _fresh_optimizer(runtime, branch_config)
        if optimizer.state:
            raise NativeClosedLoopError("branch optimizer is not fresh")
        policy_before = _policy_version(runtime)
        if policy_before != str(bundle["behavior_policy_version"]):
            raise NativeClosedLoopError(
                "bundle policy hash changed before branch update"
            )
        round_id = int(bundle["round_id"])
        update = _update_round(
            runtime=runtime,
            optimizer=optimizer,
            trajectories=bundle["trajectories"],
            round_id=round_id,
            video_weight=float(branch_config["video_weight"]),
            action_weight=float(branch_config["action_weight"]),
            action_velocity_weight=float(
                branch_config["action_velocity_weight"]
            ),
            pseudo_huber_c=float(branch_config["pseudo_huber_c"]),
            max_grad_norm=float(branch_config["max_grad_norm"]),
            loss_reduction=str(branch_config["loss_reduction"]),
            trainable_bank=str(branch_config["trainable_bank"]),
            optimizer_kind=str(branch_config["optimizer_kind"]),
            max_update_norm=branch_config["max_update_norm"],
            objective=str(branch_config["objective"]),
            sigma_values=branch_config["sigma_values"],
            line_search_sigma_values=branch_config["line_search_sigma_values"],
            line_search_update_norms=branch_config["line_search_update_norms"],
            calibration_anchors_per_trajectory=int(
                branch_config["calibration_anchors_per_trajectory"]
            ),
            functional_closure_p95_max=float(
                branch_config["functional_closure_p95_max"]
            ),
        )
        if int(update["optimizer_steps_this_round"]) != 1:
            raise NativeClosedLoopError(
                "branch update did not perform exactly one optimizer step"
            )
        policy_after = _policy_version(runtime)
        if policy_after == policy_before:
            raise NativeClosedLoopError("branch update did not change the adapter")
        global_optimizer_step = int(bundle.get("global_optimizer_step", 0)) + 1
        checkpoint_path = output_dir / (
            f"checkpoint_branch_{branch_config['trainable_bank']}.pt"
        )
        _save_checkpoint(
            path=checkpoint_path,
            runtime=runtime,
            optimizer=optimizer,
            config=branch_config,
            round_id=round_id,
            global_optimizer_step=global_optimizer_step,
            policy_version_before=policy_before,
            policy_version_after=policy_after,
        )
        summary = {
            "schema": BRANCH_SUMMARY_SCHEMA,
            "status": "PASS",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_mode": "branch_update",
            "adapter_kind": "dual_lora",
            "trainable_bank": str(branch_config["trainable_bank"]),
            "adapter_seed": int(branch_config["adapter_seed"]),
            "teacher_loaded": False,
            "fresh_optimizer": True,
            "rollout_bundle": str(bundle_path),
            "behavior_policy_version": policy_before,
            "policy_version_after": policy_after,
            "task_contract_hash": str(task_contract_hash),
            "round_id": round_id,
            "global_optimizer_step": global_optimizer_step,
            "checkpoint": str(checkpoint_path),
            "collection_outcomes": deepcopy(list(bundle["outcomes"])),
            "update": update,
        }
        _write_json(output_dir / "summary.json", summary)
        return summary
    finally:
        runtime.close()


def _run_trajectory_update(
    *,
    config: Mapping[str, Any],
    student: Path,
    teacher: Path | None,
    output_dir: Path,
    task_contract_hash: str,
    _success_path_writer_lock_held: bool = False,
) -> dict[str, Any]:
    """Merge exact Student-occupancy artifacts into one training update."""

    if _stable_hash(config["task_entries"]) != str(task_contract_hash):
        raise ValueError("trajectory_update task contract hash changed")
    coherent_update = (
        config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY
    )
    success_path_update = (
        coherent_update
        and config["coherent_tt_variant"]
        == COHERENT_TT_VARIANT_SUCCESS_PATH_V1
    )
    live_teacher_update = coherent_update and not success_path_update
    if coherent_update and teacher is None:
        raise ValueError(
            "coherent trajectory_update requires the artifact Teacher identity"
        )
    if success_path_update and not _success_path_writer_lock_held:
        if teacher is None:
            raise ValueError("success-path writer contract requires a Teacher")
        writer_contract = _success_path_writer_contract(
            config=config,
            student=student,
            teacher=teacher,
            output_dir=output_dir,
            task_contract_hash=str(task_contract_hash),
        )
        with _success_path_output_lock(
            output_dir=output_dir,
            contract=writer_contract,
        ):
            return _run_trajectory_update(
                config=config,
                student=student,
                teacher=teacher,
                output_dir=output_dir,
                task_contract_hash=str(task_contract_hash),
                _success_path_writer_lock_held=True,
            )
    artifact_paths = [
        Path(value).expanduser().resolve()
        for value in config["trajectory_artifacts"]
    ]
    success_path_exact_identity: Mapping[str, Any] | None = None
    if success_path_update:
        if teacher is None:
            raise ValueError("success-path exact identity requires a Teacher")
        success_path_exact_identity = _success_path_exact_identity(
            artifact_paths=artifact_paths,
            student=student,
            teacher=teacher,
        )
    trajectories = _load_trajectory_artifacts(
        artifact_paths,
        expected_task_entries=config["task_entries"],
        expected_student=student,
        expected_teacher=teacher,
        expected_adapter_seed=int(config["adapter_seed"]),
        expected_collection_group_id=config["collection_group_id"],
        require_coherent_collection_contract=(
            config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY
        ),
    )
    if success_path_exact_identity is not None:
        _validate_success_path_artifacts_unchanged(
            success_path_exact_identity,
            artifact_paths,
        )
    behavior_policy_versions = {
        str(trajectory["behavior_policy_version"])
        for trajectory in trajectories
    }
    round_ids = {int(trajectory["round_id"]) for trajectory in trajectories}
    if len(behavior_policy_versions) != 1 or len(round_ids) != 1:
        raise NativeClosedLoopError(
            "trajectory_update artifacts do not share one policy and round"
        )
    behavior_policy_version = behavior_policy_versions.pop()
    round_id = round_ids.pop()

    output_dir.mkdir(parents=True, exist_ok=True)
    if success_path_update:
        committed_paths = [
            output_dir / "checkpoint_trajectory_update.pt",
            output_dir / "summary.json",
            *[
                output_dir
                / "epoch_checkpoints"
                / f"checkpoint_epoch_{epoch_id:02d}.pt"
                for epoch_id in range(1, int(config["inner_epochs"]) + 1)
            ],
        ]
        for committed_path in committed_paths:
            _cleanup_atomic_temps(committed_path)
        if success_path_exact_identity is None:
            raise NativeClosedLoopError(
                "success-path exact input identity was not constructed"
            )
        recovered_summary = _guard_success_path_committed_output(
            output_dir=output_dir,
            config=config,
            exact_identity=success_path_exact_identity,
            task_contract_hash=str(task_contract_hash),
            behavior_policy_version=behavior_policy_version,
            round_id=round_id,
        )
        if recovered_summary is not None:
            return recovered_summary
    resume_checkpoint_path = (
        Path(str(config["initial_checkpoint"])).expanduser().resolve()
        if config["initial_checkpoint"] is not None
        else None
    )
    resume_checkpoint: Mapping[str, Any] | None = None
    success_path_resume_state: Mapping[str, Any] | None = None
    if resume_checkpoint_path is not None:
        loaded = torch.load(
            resume_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(loaded, Mapping):
            raise TypeError("success-path resume checkpoint is not a mapping")
        resume_checkpoint = loaded
        success_path_resume_state = _validate_success_path_resume_checkpoint(
            resume_checkpoint,
            config=config,
            expected_task_contract_hash=str(task_contract_hash),
            expected_behavior_policy_version=behavior_policy_version,
            expected_round_id=round_id,
            expected_exact_identity=success_path_exact_identity,
        )
        expected_parent = (
            output_dir
            / "epoch_checkpoints"
            / (
                "checkpoint_epoch_"
                f"{int(success_path_resume_state['completed_inner_epochs']):02d}.pt"
            )
        ).resolve()
        if resume_checkpoint_path != expected_parent:
            raise ValueError(
                "success-path resume checkpoint must belong to this output_dir"
            )

    if success_path_update:
        completed_epochs = (
            int(success_path_resume_state["completed_inner_epochs"])
            if success_path_resume_state is not None
            else 0
        )
        for epoch_id in range(1, int(config["inner_epochs"]) + 1):
            epoch_path = (
                output_dir
                / "epoch_checkpoints"
                / f"checkpoint_epoch_{epoch_id:02d}.pt"
            )
            if epoch_id <= completed_epochs and not epoch_path.is_file():
                raise FileNotFoundError(
                    f"completed success-path checkpoint is missing: {epoch_path}"
                )
            if epoch_id > completed_epochs and epoch_path.exists():
                raise FileExistsError(
                    f"refusing to overwrite success-path checkpoint: {epoch_path}"
                )

    _configure_cuda_memory_limit(config)
    np.random.seed(int(config["adapter_seed"]))
    torch.manual_seed(int(config["adapter_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["adapter_seed"]))
    runtime = NativeV0VideoRuntime(
        student_checkpoint=student,
        teacher_transformer=teacher if live_teacher_update else None,
        device=str(config.get("device", "cuda:0")),
        save_root=output_dir / "native_save",
        enable_offload=bool(config.get("enable_offload", True)),
        official_offload_parity=bool(
            config.get("official_offload_parity", True)
        ),
        adapter_rank=int(config["adapter_rank"]),
        adapter_state=resume_checkpoint_path,
        adapter_kind=str(config["adapter_kind"]),
        lora_alpha=float(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        lora_block_indices=tuple(
            int(value) for value in config["lora_block_indices"]
        ),
    )
    try:
        if live_teacher_update:
            _configure_teacher_solver_from_config(runtime, config)
        if config["adapter_kind"] == "dual_lora":
            runtime.select_adapter_trainable_bank(str(config["trainable_bank"]))
        live_adapter_contract: Mapping[str, Any] | None = None
        live_base_parameter_hashes: Mapping[str, str] | None = None
        if success_path_update:
            live_adapter_contract = runtime.adapter_contract()
            live_base_parameter_hashes = runtime.base_parameter_hashes()
            if live_adapter_contract.get("base_parameter_hashes") != dict(
                live_base_parameter_hashes
            ):
                raise NativeClosedLoopError(
                    "success-path runtime adapter/base contract is inconsistent"
                )
            runtime._success_path_base_parameter_hashes = deepcopy(
                dict(live_base_parameter_hashes)
            )
        optimizer = _fresh_optimizer(runtime, config)
        policy_at_invocation_start = _policy_version(runtime)
        if resume_checkpoint is None:
            if optimizer.state:
                raise NativeClosedLoopError(
                    "trajectory_update optimizer is not fresh"
                )
            if policy_at_invocation_start != behavior_policy_version:
                raise NativeClosedLoopError(
                    "reconstructed Student policy differs from trajectory "
                    "behavior policy"
                )
            starting_global_optimizer_step = 0
        else:
            success_path_resume_state = _validate_success_path_resume_checkpoint(
                resume_checkpoint,
                config=config,
                expected_task_contract_hash=str(task_contract_hash),
                expected_behavior_policy_version=behavior_policy_version,
                expected_round_id=round_id,
                expected_parameter_names=runtime.adapter_parameter_names,
                expected_exact_identity=success_path_exact_identity,
                expected_adapter_contract=live_adapter_contract,
                expected_base_parameter_hashes=live_base_parameter_hashes,
            )
            if policy_at_invocation_start != str(
                resume_checkpoint["policy_version_after"]
            ):
                raise NativeClosedLoopError(
                    "loaded success-path policy differs from resume checkpoint"
                )
            optimizer.load_state_dict(
                deepcopy(dict(resume_checkpoint["optimizer_state_dict"]))
            )
            if not optimizer.state:
                raise NativeClosedLoopError(
                    "success-path resumed optimizer state is empty"
                )
            loaded_state_dtypes = {
                str(value.dtype)
                for state in optimizer.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            }
            if any(value != "torch.float32" for value in loaded_state_dtypes):
                raise NativeClosedLoopError(
                    "success-path resumed optimizer state is not FP32"
                )
            starting_global_optimizer_step = int(
                success_path_resume_state["global_optimizer_step"]
            )
        policy_before = behavior_policy_version

        solver_closure_path: Path | None = None
        if bool(config["pre_update_solver_closure"]):
            solver_closure = _pre_update_solver_closure(runtime, trajectories)
            solver_closure_path = output_dir / "solver_closure.json"
            _write_json(solver_closure_path, solver_closure)
            if solver_closure["status"] != "PASS":
                raise NativeClosedLoopError(
                    "trajectory_update pre-update solver closure failed"
                )

        epoch_checkpoint_callback = None
        if success_path_update:
            def _save_success_path_epoch(
                epoch_id: int,
                optimizer_steps: int,
                progress: Mapping[str, Any],
            ) -> str:
                path = (
                    output_dir
                    / "epoch_checkpoints"
                    / f"checkpoint_epoch_{int(epoch_id):02d}.pt"
                )
                if path.exists():
                    raise FileExistsError(
                        f"refusing to overwrite success-path checkpoint: {path}"
                    )
                checkpoint_progress = deepcopy(dict(progress))
                epoch_metrics = checkpoint_progress.get("epoch_metrics")
                if not isinstance(epoch_metrics, list) or not epoch_metrics:
                    raise NativeClosedLoopError(
                        "success-path checkpoint progress has no epoch metrics"
                    )
                epoch_metrics[-1]["checkpoint"] = str(path)
                _save_checkpoint(
                    path=path,
                    runtime=runtime,
                    optimizer=optimizer,
                    config=config,
                    round_id=round_id,
                    global_optimizer_step=int(optimizer_steps),
                    policy_version_before=policy_before,
                    policy_version_after=_policy_version(runtime),
                    checkpoint_role="success_path_epoch",
                    completed_inner_epochs=int(epoch_id),
                    success_path_progress=checkpoint_progress,
                    success_path_exact_identity=(
                        success_path_exact_identity
                    ),
                )
                return str(path)

            epoch_checkpoint_callback = _save_success_path_epoch

        update = _update_round(
            runtime=runtime,
            optimizer=optimizer,
            trajectories=trajectories,
            round_id=round_id,
            video_weight=float(config["video_weight"]),
            action_weight=float(config["action_weight"]),
            action_velocity_weight=float(config["action_velocity_weight"]),
            pseudo_huber_c=float(config["pseudo_huber_c"]),
            max_grad_norm=float(config["max_grad_norm"]),
            loss_reduction=str(config["loss_reduction"]),
            trainable_bank=str(config["trainable_bank"]),
            optimizer_kind=str(config["optimizer_kind"]),
            max_update_norm=config["max_update_norm"],
            objective=str(config["objective"]),
            sigma_values=config["sigma_values"],
            line_search_sigma_values=config["line_search_sigma_values"],
            line_search_update_norms=config["line_search_update_norms"],
            calibration_anchors_per_trajectory=int(
                config["calibration_anchors_per_trajectory"]
            ),
            functional_closure_p95_max=float(
                config["functional_closure_p95_max"]
            ),
            action_fm_weight=float(config["action_fm_weight"]),
            effective_batch_size=int(config["effective_batch_size"]),
            inner_epochs=int(config["inner_epochs"]),
            ema_decay=float(config["ema_decay"]),
            consistency_video_stride=int(
                config["consistency_video_stride"]
            ),
            consistency_action_stride=int(
                config["consistency_action_stride"]
            ),
            consistency_seed=int(config["consistency_seed"]),
            coherent_tt_variant=str(config["coherent_tt_variant"]),
            success_path_max_train_labels_per_trajectory=config[
                "success_path_max_train_labels_per_trajectory"
            ],
            epoch_checkpoint_callback=epoch_checkpoint_callback,
            success_path_resume_state=success_path_resume_state,
        )
        final_success_path_progress: Mapping[str, Any] | None = None
        raw_success_path_progress = update.pop("_success_path_progress", None)
        optimizer_steps_this_round = int(
            update["optimizer_steps_this_round"]
        )
        if success_path_update:
            if not isinstance(raw_success_path_progress, Mapping):
                raise NativeClosedLoopError(
                    "success-path update did not return exact-resume progress"
                )
            final_success_path_progress = raw_success_path_progress
            optimizer_steps_this_run = int(
                update["optimizer_steps_this_invocation"]
            )
            expected_global_step = (
                int(starting_global_optimizer_step)
                + int(optimizer_steps_this_run)
            )
            if optimizer_steps_this_round != expected_global_step:
                raise NativeClosedLoopError(
                    "success-path cumulative optimizer step mismatch: "
                    f"update={optimizer_steps_this_round}, "
                    f"expected={expected_global_step}"
                )
            global_optimizer_step = int(optimizer_steps_this_round)
        else:
            if raw_success_path_progress is not None:
                raise NativeClosedLoopError(
                    "non-success-path update returned success-path progress"
                )
            optimizer_steps_this_run = int(optimizer_steps_this_round)
            global_optimizer_step = int(optimizer_steps_this_round)
        if coherent_update:
            if optimizer_steps_this_round <= 1:
                raise NativeClosedLoopError(
                    "coherent trajectory_update did not perform multiple steps"
                )
        elif optimizer_steps_this_round != 1:
            raise NativeClosedLoopError(
                "trajectory_update did not perform exactly one optimizer step"
            )
        policy_after = _policy_version(runtime)
        if policy_after == policy_at_invocation_start:
            raise NativeClosedLoopError(
                "trajectory_update did not change the adapter"
            )

        checkpoint_path = output_dir / "checkpoint_trajectory_update.pt"
        summary_path = output_dir / "summary.json"
        if success_path_update and (
            checkpoint_path.exists() or summary_path.exists()
        ):
            raise FileExistsError(
                "refusing to overwrite completed success-path output"
            )
        summary = {
            "schema": TRAJECTORY_UPDATE_SUMMARY_SCHEMA,
            "status": "PASS",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_mode": "trajectory_update",
            "adapter_kind": str(config["adapter_kind"]),
            "trainable_bank": str(config["trainable_bank"]),
            "adapter_seed": int(config["adapter_seed"]),
            "coherent_tt_variant": str(config["coherent_tt_variant"]),
            "teacher_loaded": bool(runtime.teacher is not None),
            "teacher_transformer": (
                str(teacher) if coherent_update and teacher is not None else None
            ),
            "teacher_target_source": (
                "trajectory_artifacts" if success_path_update else "live_teacher"
            ),
            "teacher_controls_environment": False,
            "environment_execution": "SS",
            "fresh_optimizer": resume_checkpoint is None,
            "resumed_optimizer_state": resume_checkpoint is not None,
            "initial_checkpoint": (
                str(resume_checkpoint_path)
                if resume_checkpoint_path is not None
                else None
            ),
            "collection_group_id": config["collection_group_id"],
            "trajectory_artifacts": [str(path) for path in artifact_paths],
            "behavior_policy_version": policy_before,
            "policy_version_after": policy_after,
            "task_contract_hash": str(task_contract_hash),
            "round_id": round_id,
            "starting_global_optimizer_step": int(
                starting_global_optimizer_step
            ),
            "global_optimizer_step": global_optimizer_step,
            "optimizer_steps_this_run": optimizer_steps_this_run,
            "checkpoint": str(checkpoint_path),
            "solver_closure": (
                None if solver_closure_path is None else str(solver_closure_path)
            ),
            "update": update,
        }
        if success_path_update:
            if success_path_exact_identity is None:
                raise NativeClosedLoopError(
                    "success-path summary is missing exact input identity"
                )
            if not isinstance(live_adapter_contract, Mapping) or not isinstance(
                live_base_parameter_hashes, Mapping
            ):
                raise NativeClosedLoopError(
                    "success-path summary is missing live model identity"
                )
            summary.update(
                {
                    "success_path_commit_schema": SUCCESS_PATH_COMMIT_SCHEMA,
                    "success_path_finalization_schema": (
                        SUCCESS_PATH_FINALIZATION_SCHEMA
                    ),
                    "commit_recovered_from_final_checkpoint": False,
                    "success_path_exact_identity": deepcopy(
                        dict(success_path_exact_identity)
                    ),
                    "success_path_exact_identity_hash": str(
                        success_path_exact_identity["identity_hash"]
                    ),
                    "adapter_contract_hash": _stable_hash(
                        live_adapter_contract
                    ),
                    "base_parameter_hashes_hash": _stable_hash(
                        live_base_parameter_hashes
                    ),
                }
            )
        _save_checkpoint(
            path=checkpoint_path,
            runtime=runtime,
            optimizer=optimizer,
            config=config,
            round_id=round_id,
            global_optimizer_step=global_optimizer_step,
            policy_version_before=policy_before,
            policy_version_after=policy_after,
            checkpoint_role=(
                "success_path_final" if success_path_update else None
            ),
            completed_inner_epochs=(
                int(config["inner_epochs"]) if success_path_update else None
            ),
            success_path_progress=final_success_path_progress,
            success_path_exact_identity=(
                success_path_exact_identity if success_path_update else None
            ),
            success_path_finalization=(
                summary if success_path_update else None
            ),
        )
        if success_path_update:
            checkpoint_sha256, _ = _sha256_file(checkpoint_path)
            summary["checkpoint_sha256"] = checkpoint_sha256
        if success_path_update and summary_path.exists():
            raise FileExistsError(
                f"refusing to overwrite success-path summary: {summary_path}"
            )
        _write_json(summary_path, summary)
        return summary
    finally:
        runtime.close()


def _normalize_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(raw)
    config["objective"] = str(config.get("objective", OBJECTIVE_ENDPOINT))
    if config["objective"] not in {
        OBJECTIVE_ENDPOINT,
        OBJECTIVE_MULTI_SIGMA_X0,
        OBJECTIVE_COHERENT_TT_CONSISTENCY,
    }:
        raise ValueError(f"unsupported objective: {config['objective']!r}")
    config["coherent_tt_variant"] = str(
        config.get("coherent_tt_variant", COHERENT_TT_VARIANT_BASELINE)
    )
    if config["coherent_tt_variant"] not in {
        COHERENT_TT_VARIANT_BASELINE,
        COHERENT_TT_VARIANT_SUCCESS_PATH_V1,
    }:
        raise ValueError(
            "unsupported coherent_tt_variant: "
            f"{config['coherent_tt_variant']!r}"
        )
    if (
        config["objective"] != OBJECTIVE_COHERENT_TT_CONSISTENCY
        and config["coherent_tt_variant"] != COHERENT_TT_VARIANT_BASELINE
    ):
        raise ValueError(
            "coherent_tt_variant is only valid for coherent_tt_consistency"
        )
    config["run_mode"] = str(config.get("run_mode", "iterative"))
    if config["run_mode"] not in {
        "iterative",
        "collect",
        "branch_update",
        "trajectory_update",
    }:
        raise ValueError(f"unsupported run_mode: {config['run_mode']!r}")
    default_rounds = 1 if config["run_mode"] != "iterative" else 4
    config["rounds"] = int(config.get("rounds", default_rounds))
    if config["rounds"] <= 0:
        raise ValueError("rounds must be positive")
    if config["run_mode"] != "iterative" and config["rounds"] != 1:
        raise ValueError(f"{config['run_mode']} requires rounds=1")
    if config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY:
        if config["run_mode"] not in {
            "iterative",
            "collect",
            "trajectory_update",
        }:
            raise ValueError(
                "coherent_tt_consistency supports iterative, collect, or "
                "trajectory_update run modes"
            )
        if config["rounds"] != 1:
            raise ValueError("coherent_tt_consistency requires rounds=1")
    raw_collection_group_id = config.get("collection_group_id")
    config["collection_group_id"] = (
        None
        if raw_collection_group_id in (None, "")
        else str(raw_collection_group_id).strip()
    )
    if (
        config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY
        and config["run_mode"] in {"collect", "trajectory_update"}
        and not config["collection_group_id"]
    ):
        raise ValueError(
            "parallel coherent collect/trajectory_update requires "
            "collection_group_id"
        )
    config["teacher_video_steps"] = int(config.get("teacher_video_steps", 25))
    if config["teacher_video_steps"] <= 0:
        raise ValueError("teacher_video_steps must be positive")
    raw_teacher_video_exec_steps = config.get("teacher_video_exec_steps")
    if raw_teacher_video_exec_steps is None:
        config["teacher_video_exec_steps"] = None
    else:
        teacher_video_exec_steps = int(raw_teacher_video_exec_steps)
        config["teacher_video_exec_steps"] = (
            None if teacher_video_exec_steps == -1 else teacher_video_exec_steps
        )
    if config["teacher_video_exec_steps"] is not None and not (
        1
        <= int(config["teacher_video_exec_steps"])
        <= int(config["teacher_video_steps"])
    ):
        raise ValueError(
            "teacher_video_exec_steps must lie in [1, teacher_video_steps] "
            "or be -1/None"
        )
    config["teacher_action_steps"] = int(config.get("teacher_action_steps", 50))
    if config["teacher_action_steps"] <= 0:
        raise ValueError("teacher_action_steps must be positive")
    config["pre_update_solver_closure"] = bool(
        config.get("pre_update_solver_closure", False)
    )
    if (
        config["pre_update_solver_closure"]
        and config["objective"] != OBJECTIVE_MULTI_SIGMA_X0
    ):
        raise ValueError(
            "pre_update_solver_closure requires objective='multi_sigma_x0'"
        )
    raw_cuda_memory_fraction = config.get("cuda_memory_fraction")
    config["cuda_memory_fraction"] = (
        None
        if raw_cuda_memory_fraction is None
        else float(raw_cuda_memory_fraction)
    )
    if config["cuda_memory_fraction"] is not None and not (
        0.0 < float(config["cuda_memory_fraction"]) <= 1.0
    ):
        raise ValueError("cuda_memory_fraction must lie in (0, 1]")
    adapter_default = (
        "dual_lora"
        if config["run_mode"] in {"collect", "branch_update"}
        else "joint_lora"
    )
    config["adapter_kind"] = str(config.get("adapter_kind", adapter_default))
    if config["adapter_kind"] not in {"joint_lora", "dual_lora"}:
        raise ValueError(
            f"unsupported adapter_kind: {config['adapter_kind']!r}"
        )
    if (
        config["run_mode"] == "branch_update"
        and config["adapter_kind"] != "dual_lora"
    ):
        raise ValueError("branch_update requires adapter_kind='dual_lora'")
    config["adapter_rank"] = int(config.get("adapter_rank", 8))
    if config["adapter_rank"] <= 0:
        raise ValueError("adapter_rank must be positive")
    config["lora_alpha"] = float(config.get("lora_alpha", 8.0))
    if not math.isfinite(config["lora_alpha"]) or config["lora_alpha"] <= 0.0:
        raise ValueError("lora_alpha must be finite and positive")
    config["lora_dropout"] = float(config.get("lora_dropout", 0.0))
    if config["lora_dropout"] != 0.0:
        raise ValueError("the active LoRA runtime requires lora_dropout=0")
    raw_block_indices = config.get("lora_block_indices", [26, 27, 28, 29])
    if not isinstance(raw_block_indices, Sequence) or isinstance(
        raw_block_indices, (str, bytes)
    ):
        raise TypeError("lora_block_indices must be a non-empty sequence")
    block_indices = [int(value) for value in raw_block_indices]
    if not block_indices:
        raise ValueError("lora_block_indices must not be empty")
    if len(set(block_indices)) != len(block_indices):
        raise ValueError("lora_block_indices must be unique")
    if any(
        index < 0 or index >= SHARED_TRANSFORMER_BLOCK_COUNT
        for index in block_indices
    ):
        raise ValueError(
            "lora_block_indices must address FlashWAM shared blocks 0-29"
        )
    config["lora_block_indices"] = block_indices
    config["trainable_bank"] = str(
        config.get(
            "trainable_bank",
            "both" if config["adapter_kind"] == "joint_lora" else "action",
        )
    )
    if config["adapter_kind"] == "joint_lora":
        if config["trainable_bank"] != "both":
            raise ValueError("JointLoRA requires trainable_bank='both'")
    elif config["trainable_bank"] not in {"video", "action"}:
        raise ValueError("dual_lora requires trainable_bank video or action")
    config["optimizer_kind"] = str(
        config.get(
            "optimizer_kind",
            (
                "functional_sgd"
                if config["objective"] == OBJECTIVE_MULTI_SIGMA_X0
                else "adamw"
            )
            if config["adapter_kind"] == "joint_lora"
            else "trust_region_sgd",
        )
    )
    if config["optimizer_kind"] not in {
        "adamw",
        "trust_region_sgd",
        "functional_sgd",
    }:
        raise ValueError(
            f"unsupported optimizer_kind: {config['optimizer_kind']!r}"
        )
    if (
        config["adapter_kind"] == "dual_lora"
        and config["optimizer_kind"] != "trust_region_sgd"
    ):
        raise ValueError("dual-bank pilot requires trust_region_sgd")
    if config["objective"] == OBJECTIVE_MULTI_SIGMA_X0:
        if config["adapter_kind"] != "joint_lora":
            raise ValueError("multi_sigma_x0 requires adapter_kind='joint_lora'")
        if config["trainable_bank"] != "both":
            raise ValueError("multi_sigma_x0 requires trainable_bank='both'")
        if config["optimizer_kind"] != "functional_sgd":
            raise ValueError("multi_sigma_x0 requires optimizer_kind='functional_sgd'")
        if config["lora_block_indices"] != list(
            range(SHARED_TRANSFORMER_BLOCK_COUNT)
        ):
            raise ValueError("multi_sigma_x0 requires all shared blocks 0-29")
    elif config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY:
        if config["adapter_kind"] != "joint_lora":
            raise ValueError(
                "coherent_tt_consistency requires adapter_kind='joint_lora'"
            )
        if config["trainable_bank"] != "both":
            raise ValueError(
                "coherent_tt_consistency requires trainable_bank='both'"
            )
        if config["optimizer_kind"] != "adamw":
            raise ValueError("coherent_tt_consistency requires optimizer_kind='adamw'")
        if config["adapter_rank"] != 8:
            raise ValueError("the first coherent TT baseline fixes adapter_rank=8")
        if config["lora_block_indices"] != list(
            range(SHARED_TRANSFORMER_BLOCK_COUNT)
        ):
            raise ValueError(
                "coherent_tt_consistency requires all shared blocks 0-29"
            )
    elif config["optimizer_kind"] == "functional_sgd":
        raise ValueError("functional_sgd is only valid for multi_sigma_x0")

    raw_tasks = config.get("tasks")
    is_multi_task = raw_tasks is not None
    if is_multi_task:
        if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, (str, bytes)):
            raise TypeError("tasks must be a non-empty sequence of mappings")
        if not raw_tasks:
            raise ValueError("tasks must not be empty")
        task_sources = []
        for value in raw_tasks:
            if not isinstance(value, Mapping):
                raise TypeError("every tasks entry must be a mapping")
            task_sources.append(dict(value))
    else:
        if "task" not in config:
            raise ValueError("single-task config requires task")
        task_sources = [dict(config)]

    task_entries: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for source in task_sources:
        task = str(source["task"]).strip()
        if not task:
            raise ValueError("task must not be empty")
        if task in seen_tasks:
            raise ValueError(f"duplicate task entry: {task}")
        seen_tasks.add(task)
        task_config = require_training_task_config(
            str(source.get("task_config", config.get("task_config", "demo_clean")))
        )
        chunks_override: int | None = None
        if "chunks" in source:
            chunks_override = int(source["chunks"])
        elif not is_multi_task and "chunks" in config:
            chunks_override = int(config["chunks"])
        chunks = resolve_task_chunks(task, chunks_override)

        common_prompt_value = source.get(
            "prompt", config.get("prompt") if not is_multi_task else None
        )
        common_prompt = (
            None
            if common_prompt_value in (None, "")
            else str(common_prompt_value).strip()
        )
        gate_value = source.get(
            "gate_json", config.get("gate_json") if not is_multi_task else None
        )
        gate_json = None if gate_value in (None, "") else str(gate_value)

        raw_rollouts = source.get("rollouts")
        rollouts: list[dict[str, Any]] = []
        if raw_rollouts is not None:
            if not isinstance(raw_rollouts, Sequence) or isinstance(
                raw_rollouts, (str, bytes)
            ):
                raise TypeError("rollouts must be a non-empty sequence of mappings")
            if not raw_rollouts:
                raise ValueError(f"task {task} has no rollouts")
            for raw_rollout in raw_rollouts:
                if not isinstance(raw_rollout, Mapping):
                    raise TypeError("every rollout entry must be a mapping")
                prompt_value = raw_rollout.get("prompt", common_prompt)
                prompt = (
                    None
                    if prompt_value in (None, "")
                    else str(prompt_value).strip()
                )
                rollout = {"seed": int(raw_rollout["seed"]), "prompt": prompt}
                if "rollout_id" in raw_rollout:
                    rollout["rollout_id"] = int(raw_rollout["rollout_id"])
                if "role" in raw_rollout:
                    role = str(raw_rollout["role"])
                    if role not in {"train", "calibration"}:
                        raise ValueError(
                            f"unsupported rollout role for task {task}: {role!r}"
                        )
                    rollout["role"] = role
                rollouts.append(rollout)
        else:
            rollout_count = int(
                source.get(
                    "rollouts_per_round",
                    config.get(
                        "rollouts_per_task",
                        config.get("rollouts_per_round", 2),
                    ),
                )
            )
            if rollout_count <= 0:
                raise ValueError("rollouts_per_round must be positive")
            seed_values = source.get(
                "rollout_seeds",
                config.get("rollout_seeds", []) if not is_multi_task else [],
            )
            seeds = [int(value) for value in seed_values]
            if not seeds:
                if "seed" in source:
                    base_seed = int(source["seed"])
                elif "seed" in config:
                    base_seed = int(config["seed"])
                else:
                    raise ValueError(f"task {task} requires seed or rollouts")
                seeds = [base_seed + index for index in range(rollout_count)]
            if len(seeds) != rollout_count:
                raise ValueError(
                    f"task {task} rollout_seeds must match rollouts_per_round"
                )
            prompts = source.get("prompts", {})
            if not isinstance(prompts, Mapping):
                raise TypeError("prompts must map rollout seed to prompt")
            for seed in seeds:
                prompt_value = prompts.get(str(seed), prompts.get(seed, common_prompt))
                prompt = (
                    None
                    if prompt_value in (None, "")
                    else str(prompt_value).strip()
                )
                rollouts.append({"seed": int(seed), "prompt": prompt})

        rollout_seeds = [int(rollout["seed"]) for rollout in rollouts]
        if len(set(rollout_seeds)) != len(rollout_seeds):
            raise ValueError(f"task {task} rollout seeds must be unique")
        if any(rollout["prompt"] is None for rollout in rollouts) and gate_json is None:
            raise ValueError(
                f"task {task} requires a prompt per rollout or gate_json"
            )
        task_entries.append(
            {
                "task": task,
                "task_config": task_config,
                "chunks": int(chunks),
                "rollouts": rollouts,
                "gate_json": gate_json,
            }
        )

    config["task_entries"] = task_entries
    require_explicit_rollout_ids = (
        config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY
        and config["run_mode"] in {"collect", "trajectory_update"}
    )
    all_rollouts = [
        rollout
        for entry in task_entries
        for rollout in entry["rollouts"]
    ]
    if require_explicit_rollout_ids and any(
        "rollout_id" not in rollout for rollout in all_rollouts
    ):
        raise ValueError(
            "parallel coherent collect/trajectory_update requires an explicit "
            "global rollout_id on every rollout"
        )
    used_rollout_ids = {
        int(rollout["rollout_id"])
        for rollout in all_rollouts
        if "rollout_id" in rollout
    }
    if len(used_rollout_ids) != sum(
        1 for rollout in all_rollouts if "rollout_id" in rollout
    ):
        raise ValueError("rollout_id must be globally unique")
    if any(rollout_id < 0 for rollout_id in used_rollout_ids):
        raise ValueError("rollout_id must be nonnegative")
    next_rollout_id = 0
    for rollout in all_rollouts:
        if "rollout_id" in rollout:
            continue
        while next_rollout_id in used_rollout_ids:
            next_rollout_id += 1
        rollout["rollout_id"] = int(next_rollout_id)
        used_rollout_ids.add(next_rollout_id)
        next_rollout_id += 1
    config["task"] = task_entries[0]["task"] if len(task_entries) == 1 else "multi_task"
    config["task_config"] = "demo_clean"
    config["rollouts_per_round"] = sum(
        len(entry["rollouts"]) for entry in task_entries
    )
    if config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY:
        rollout_roles = [
            rollout.get("role")
            for entry in task_entries
            for rollout in entry["rollouts"]
        ]
        if any(role not in {"train", "calibration"} for role in rollout_roles):
            raise ValueError(
                "coherent_tt_consistency requires explicit train/calibration "
                "role on every rollout"
            )
        if (
            config["run_mode"] != "collect"
            and ("train" not in rollout_roles or "calibration" not in rollout_roles)
        ):
            raise ValueError(
                "coherent_tt_consistency requires both train and calibration rollouts"
            )
    if "adapter_seed" not in config and (
        len(task_entries) > 1
        or (
            config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY
            and config["run_mode"] in {"collect", "trajectory_update"}
        )
    ):
        raise ValueError(
            "multi-task or sharded coherent config requires explicit adapter_seed"
        )
    legacy_adapter_seed = int(task_entries[-1]["rollouts"][-1]["seed"])
    config["adapter_seed"] = int(
        config.get("adapter_seed", legacy_adapter_seed)
    )
    if config["adapter_seed"] < 0:
        raise ValueError("adapter_seed must be non-negative")
    if len(task_entries) == 1:
        config["chunks"] = int(task_entries[0]["chunks"])
        config["rollout_seeds"] = [
            int(rollout["seed"]) for rollout in task_entries[0]["rollouts"]
        ]
    positive_defaults = {
        "learning_rate": (
            5e-6
            if config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY
            else (1e-4 if config["optimizer_kind"] == "adamw" else 1.0)
        ),
        "pseudo_huber_c": 0.001,
        "max_grad_norm": (
            2.0
            if config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY
            else 1.0
        ),
    }
    for key, default in positive_defaults.items():
        config[key] = float(config.get(key, default))
        if not math.isfinite(config[key]) or config[key] <= 0.0:
            raise ValueError(f"{key} must be finite and positive")
    weight_defaults = {
        "video_weight": (
            1.0 if config["trainable_bank"] in {"both", "video"} else 0.0
        ),
        "action_weight": (
            1.0 if config["trainable_bank"] in {"both", "action"} else 0.0
        ),
        "action_velocity_weight": (
            0.01
            if config["trainable_bank"] == "both"
            and config["objective"] == OBJECTIVE_ENDPOINT
            else 0.0
        ),
    }
    for key, default in weight_defaults.items():
        config[key] = float(config.get(key, default))
        if not math.isfinite(config[key]) or config[key] < 0.0:
            raise ValueError(f"{key} must be finite and nonnegative")
    config["action_fm_weight"] = float(
        config.get(
            "action_fm_weight",
            0.2
            if config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY
            else 0.0,
        )
    )
    if not math.isfinite(config["action_fm_weight"]) or config[
        "action_fm_weight"
    ] < 0.0:
        raise ValueError("action_fm_weight must be finite and nonnegative")
    if config["trainable_bank"] == "video":
        if config["video_weight"] <= 0.0:
            raise ValueError("video bank requires positive video_weight")
        if config["action_weight"] != 0.0 or config["action_velocity_weight"] != 0.0:
            raise ValueError("video-only stage requires zero action weights")
    elif config["trainable_bank"] == "action":
        if config["video_weight"] != 0.0:
            raise ValueError("action-only stage requires video_weight=0")
        if config["action_weight"] <= 0.0 and config["action_velocity_weight"] <= 0.0:
            raise ValueError("action bank requires a positive action objective")
    elif config["video_weight"] <= 0.0 or config["action_weight"] <= 0.0:
        raise ValueError("joint stage requires positive video/action weights")
    if (
        config["objective"] == OBJECTIVE_MULTI_SIGMA_X0
        and config["action_velocity_weight"] != 0.0
    ):
        raise ValueError("multi_sigma_x0 requires action_velocity_weight=0")
    if config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY:
        if config["action_velocity_weight"] != 0.0:
            raise ValueError(
                "coherent_tt_consistency requires action_velocity_weight=0"
            )
        if config["action_fm_weight"] <= 0.0:
            raise ValueError(
                "coherent_tt_consistency requires positive action_fm_weight"
            )
    elif config["action_fm_weight"] != 0.0:
        raise ValueError(
            "action_fm_weight is only valid for coherent_tt_consistency"
        )
    raw_max_update_norm = config.get("max_update_norm")
    if config["optimizer_kind"] == "trust_region_sgd":
        config["max_update_norm"] = float(
            0.003 if raw_max_update_norm is None else raw_max_update_norm
        )
        if (
            not math.isfinite(config["max_update_norm"])
            or config["max_update_norm"] <= 0.0
        ):
            raise ValueError("max_update_norm must be finite and positive")
        config["sgd_momentum"] = float(config.get("sgd_momentum", 0.0))
        if config["sgd_momentum"] != 0.0:
            raise ValueError(
                "parameter-space trust bound currently requires sgd_momentum=0"
            )
    else:
        if raw_max_update_norm is not None:
            raise ValueError("max_update_norm is only valid for trust_region_sgd")
        config["max_update_norm"] = None
        config["sgd_momentum"] = 0.0
    if config["objective"] == OBJECTIVE_MULTI_SIGMA_X0:
        config["sigma_values"] = list(
            _validated_sigmas(
                config.get("sigma_values", [1.0, 0.5, 0.25]),
                name="sigma_values",
            )
        )
        config["line_search_sigma_values"] = list(
            _validated_sigmas(
                config.get("line_search_sigma_values", [1.0]),
                name="line_search_sigma_values",
            )
        )
        if 1.0 not in config["sigma_values"] or 1.0 not in config[
            "line_search_sigma_values"
        ]:
            raise ValueError("multi_sigma_x0 sigma contracts must include 1.0")
        raw_candidate_norms = config.get(
            "line_search_update_norms", [0.03, 0.01, 0.003, 0.001, 0.0003]
        )
        if not isinstance(raw_candidate_norms, Sequence) or isinstance(
            raw_candidate_norms, (str, bytes)
        ):
            raise TypeError("line_search_update_norms must be a sequence")
        candidate_norms = sorted(
            {float(value) for value in raw_candidate_norms}, reverse=True
        )
        if not candidate_norms or any(
            not math.isfinite(value) or value <= 0.0 for value in candidate_norms
        ):
            raise ValueError("line_search_update_norms must be finite and positive")
        config["line_search_update_norms"] = candidate_norms
        config["calibration_anchors_per_trajectory"] = int(
            config.get("calibration_anchors_per_trajectory", 3)
        )
        if config["calibration_anchors_per_trajectory"] <= 0:
            raise ValueError("calibration_anchors_per_trajectory must be positive")
        obsolete_closure_keys = {
            "functional_closure_min",
            "functional_closure_max",
        }.intersection(config)
        if obsolete_closure_keys:
            raise ValueError(
                "fixed positive median closure bands are invalid for the BF16 "
                "functional staircase; remove "
                + ", ".join(sorted(obsolete_closure_keys))
            )
        config["functional_acceptance"] = str(
            config.get(
                "functional_acceptance",
                FUNCTIONAL_ACCEPTANCE_HELDOUT_NONREGRESSION,
            )
        )
        if (
            config["functional_acceptance"]
            != FUNCTIONAL_ACCEPTANCE_HELDOUT_NONREGRESSION
        ):
            raise ValueError(
                "multi_sigma_x0 requires functional_acceptance="
                f"{FUNCTIONAL_ACCEPTANCE_HELDOUT_NONREGRESSION!r}"
            )
        config["functional_closure_p95_max"] = float(
            config.get("functional_closure_p95_max", 0.50)
        )
        if (
            not math.isfinite(config["functional_closure_p95_max"])
            or config["functional_closure_p95_max"] <= 0.0
        ):
            raise ValueError(
                "functional_closure_p95_max must be finite and positive"
            )
    elif config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY:
        config["sigma_values"] = []
        config["line_search_sigma_values"] = []
        config["line_search_update_norms"] = []
        config["calibration_anchors_per_trajectory"] = int(
            config.get("calibration_anchors_per_trajectory", 5)
        )
        if config["calibration_anchors_per_trajectory"] < 2:
            raise ValueError(
                "coherent TT calibration needs at least two anchors to include "
                "early/late"
            )
        config["functional_acceptance"] = None
        config["functional_closure_p95_max"] = 0.0
    else:
        config["sigma_values"] = []
        config["line_search_sigma_values"] = []
        config["line_search_update_norms"] = []
        config["calibration_anchors_per_trajectory"] = 0
        config["functional_acceptance"] = None
        config["functional_closure_p95_max"] = 0.0
    if config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY:
        config["inner_epochs"] = int(config.get("inner_epochs", 1))
        if (
            config["coherent_tt_variant"] == COHERENT_TT_VARIANT_BASELINE
            and config["inner_epochs"] != 1
        ):
            raise ValueError("the first coherent TT baseline fixes inner_epochs=1")
        if config["inner_epochs"] <= 0:
            raise ValueError("inner_epochs must be positive")
        raw_max_train_labels = config.get(
            "success_path_max_train_labels_per_trajectory"
        )
        config["success_path_max_train_labels_per_trajectory"] = (
            None
            if raw_max_train_labels in (None, "")
            else int(raw_max_train_labels)
        )
        if (
            config["success_path_max_train_labels_per_trajectory"] is not None
            and config["success_path_max_train_labels_per_trajectory"] <= 0
        ):
            raise ValueError(
                "success_path_max_train_labels_per_trajectory must be positive"
            )
        if (
            config["coherent_tt_variant"] == COHERENT_TT_VARIANT_BASELINE
            and config["success_path_max_train_labels_per_trajectory"] is not None
        ):
            raise ValueError(
                "success_path_max_train_labels_per_trajectory requires "
                "coherent_tt_variant='success_path_v1'"
            )
        config["effective_batch_size"] = int(
            config.get("effective_batch_size", 4)
        )
        if config["effective_batch_size"] <= 0:
            raise ValueError("effective_batch_size must be positive")
        config["ema_decay"] = float(config.get("ema_decay", 0.995))
        if not math.isfinite(config["ema_decay"]) or not (
            0.0 <= config["ema_decay"] < 1.0
        ):
            raise ValueError("ema_decay must be finite in [0, 1)")
        config["consistency_video_stride"] = int(
            config.get("consistency_video_stride", 500)
        )
        config["consistency_action_stride"] = int(
            config.get("consistency_action_stride", 500)
        )
        if min(
            config["consistency_video_stride"],
            config["consistency_action_stride"],
        ) <= 0:
            raise ValueError("consistency strides must be positive")
        config["consistency_seed"] = int(
            config.get("consistency_seed", config["adapter_seed"] + 7919)
        )
        if config["consistency_seed"] < 0:
            raise ValueError("consistency_seed must be nonnegative")
        config["consistency_noise_source"] = str(
            config.get("consistency_noise_source", "artifact_epsilon")
        )
        if config["consistency_noise_source"] != "artifact_epsilon":
            raise ValueError(
                "the first coherent TT baseline requires artifact_epsilon noise"
            )
    else:
        config["inner_epochs"] = 1
        config["effective_batch_size"] = 1
        config["ema_decay"] = 0.0
        config["consistency_video_stride"] = 0
        config["consistency_action_stride"] = 0
        config["consistency_seed"] = 0
        config["consistency_noise_source"] = None
        config["success_path_max_train_labels_per_trajectory"] = None
    if float(config.get("retention_weight", 0.0)) != 0.0:
        raise ValueError("this pilot requires retention_weight=0")
    config["retention_weight"] = 0.0
    config["loss_reduction"] = str(
        config.get(
            "loss_reduction",
            (
                LOSS_REDUCTION_MEAN_TRAJECTORIES
                if config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY
                else (
                    LOSS_REDUCTION_MEAN_TASKS
                    if len(task_entries) > 1
                    else LOSS_REDUCTION_MEAN_ALL
                )
            ),
        )
    )
    if config["loss_reduction"] not in {
        LOSS_REDUCTION_MEAN_ALL,
        LOSS_REDUCTION_MEAN_TRAJECTORIES,
        LOSS_REDUCTION_MEAN_TASKS,
    }:
        raise ValueError(
            f"unsupported loss_reduction: {config['loss_reduction']!r}"
        )
    if (
        config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY
        and config["loss_reduction"] != LOSS_REDUCTION_MEAN_TRAJECTORIES
    ):
        raise ValueError(
            "coherent_tt_consistency requires trajectory-balanced reduction"
        )
    initial_checkpoint = config.get("initial_checkpoint")
    config["initial_checkpoint"] = (
        None if initial_checkpoint in (None, "") else str(initial_checkpoint)
    )
    rollout_bundle = config.get("rollout_bundle")
    config["rollout_bundle"] = (
        None if rollout_bundle in (None, "") else str(rollout_bundle)
    )
    raw_trajectory_artifacts = config.get("trajectory_artifacts", [])
    if not isinstance(raw_trajectory_artifacts, Sequence) or isinstance(
        raw_trajectory_artifacts, (str, bytes)
    ):
        raise TypeError("trajectory_artifacts must be a sequence of paths")
    config["trajectory_artifacts"] = [
        str(value) for value in raw_trajectory_artifacts
    ]
    if len(set(config["trajectory_artifacts"])) != len(
        config["trajectory_artifacts"]
    ):
        raise ValueError("trajectory_artifacts paths must be unique")
    config["resume_optimizer_state"] = bool(
        config.get("resume_optimizer_state", False)
    )
    if config["resume_optimizer_state"] and config["initial_checkpoint"] is None:
        raise ValueError("resume_optimizer_state requires initial_checkpoint")
    if config["run_mode"] == "branch_update":
        if config["rollout_bundle"] is None:
            raise ValueError("branch_update requires rollout_bundle")
        if config["initial_checkpoint"] is not None:
            raise ValueError(
                "branch_update restores policy only from rollout_bundle"
            )
        if config["resume_optimizer_state"]:
            raise ValueError("branch_update requires a fresh optimizer")
    if config["run_mode"] == "trajectory_update":
        if not config["trajectory_artifacts"]:
            raise ValueError("trajectory_update requires trajectory_artifacts")
        if config["rollout_bundle"] is not None:
            raise ValueError("trajectory_update does not consume rollout_bundle")
        exact_success_path_resume = (
            config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY
            and config["coherent_tt_variant"]
            == COHERENT_TT_VARIANT_SUCCESS_PATH_V1
        )
        if (
            config["initial_checkpoint"] is not None
            and not exact_success_path_resume
        ):
            raise ValueError(
                "trajectory_update only supports exact resume for "
                "coherent_tt_consistency/success_path_v1"
            )
        if (
            config["initial_checkpoint"] is not None
            and not config["resume_optimizer_state"]
        ):
            raise ValueError(
                "success_path_v1 trajectory resume requires "
                "resume_optimizer_state=true"
            )
        if config["resume_optimizer_state"] and not exact_success_path_resume:
            raise ValueError("trajectory_update requires a fresh optimizer")
    if config["run_mode"] == "collect" and config["resume_optimizer_state"]:
        raise ValueError("collect does not consume optimizer state")
    if (
        config["run_mode"] == "collect"
        and config["adapter_kind"] == "joint_lora"
        and config["rollout_bundle"] is not None
    ):
        raise ValueError(
            "joint_lora collect writes exact trajectory artifacts, not a "
            "dual_lora branch rollout_bundle"
        )
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = _normalize_config(
        json.loads(config_path.read_text(encoding="utf-8"))
    )

    project_root = Path(config["project_root"]).expanduser().resolve()
    workspace_root = Path(__file__).resolve().parents[1]
    robotwin_root = project_root / "third_party" / "RoboTwin-lingbot-native"
    lingbot_root = project_root / "third_party" / "lingbot-va"
    sys.path[:0] = [
        str(workspace_root),
        str(project_root / "src"),
        str(project_root),
        str(robotwin_root),
        str(lingbot_root),
    ]
    os.environ["ROBOTWIN_ROOT"] = str(robotwin_root)
    run_mode = str(config["run_mode"])
    # The simulator setup path normally establishes this working directory
    # before model construction.  A teacher-free branch skips simulator setup,
    # so reproduce the same runtime bootstrap explicitly.
    if run_mode in {"branch_update", "trajectory_update"}:
        os.chdir(robotwin_root)
    task = str(config["task"])
    task_config = str(config["task_config"])
    student = Path(config["student"]).expanduser().resolve()
    teacher: Path | None = None
    needs_teacher = run_mode != "branch_update" and (
        run_mode != "trajectory_update"
        or config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY
    )
    if needs_teacher:
        teacher_value = config.get("teacher_transformer")
        if teacher_value in (None, ""):
            raise ValueError(f"{run_mode} requires teacher_transformer")
        teacher = Path(str(teacher_value)).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (
        run_mode == "collect"
        and config["adapter_kind"] == "dual_lora"
        and config["rollout_bundle"] is None
    ):
        config["rollout_bundle"] = str(output_dir / "rollout_bundle.pt")

    resolved_task_entries: list[dict[str, Any]] = []
    for raw_entry in config["task_entries"]:
        entry = deepcopy(dict(raw_entry))
        gate_prompt: str | None = None
        if entry["gate_json"] is not None:
            gate = json.loads(
                Path(entry["gate_json"])
                .expanduser()
                .resolve()
                .read_text(encoding="utf-8")
            )
            gate_prompt = str(gate["prompt"]).strip()
        resolved_rollouts: list[dict[str, Any]] = []
        for rollout in entry["rollouts"]:
            prompt_value = rollout.get("prompt") or gate_prompt
            prompt = "" if prompt_value is None else str(prompt_value).strip()
            if not prompt:
                raise ValueError(
                    f"task {entry['task']} seed {rollout['seed']} has no locked prompt"
                )
            resolved_rollout = {
                "seed": int(rollout["seed"]),
                "prompt": prompt,
                "rollout_id": int(rollout["rollout_id"]),
            }
            if "role" in rollout:
                resolved_rollout["role"] = str(rollout["role"])
            resolved_rollouts.append(resolved_rollout)
        entry["rollouts"] = resolved_rollouts
        resolved_task_entries.append(entry)
    config["task_entries"] = resolved_task_entries
    task_contract_hash = _stable_hash(resolved_task_entries)

    if run_mode == "branch_update":
        summary = _run_branch_update(
            config=config,
            student=student,
            output_dir=output_dir,
            task_contract_hash=task_contract_hash,
        )
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return 0
    if run_mode == "trajectory_update":
        summary = _run_trajectory_update(
            config=config,
            student=student,
            teacher=teacher,
            output_dir=output_dir,
            task_contract_hash=task_contract_hash,
        )
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return 0

    initial_checkpoint_path = (
        Path(config["initial_checkpoint"]).expanduser().resolve()
        if config["initial_checkpoint"] is not None
        else None
    )
    initial_checkpoint: Mapping[str, Any] | None = None
    if initial_checkpoint_path is not None:
        loaded = torch.load(
            initial_checkpoint_path, map_location="cpu", weights_only=True
        )
        if not isinstance(loaded, Mapping):
            raise TypeError("initial_checkpoint is not a checkpoint mapping")
        expected_checkpoint_schema = (
            DUAL_CHECKPOINT_SCHEMA
            if config["adapter_kind"] == "dual_lora"
            else CHECKPOINT_SCHEMA
        )
        if loaded.get("schema") != expected_checkpoint_schema:
            raise ValueError("initial_checkpoint schema mismatch")
        if loaded.get("adapter_kind") != config["adapter_kind"]:
            raise ValueError("initial_checkpoint adapter kind mismatch")
        if _checkpoint_objective(loaded) != config["objective"]:
            raise ValueError("initial_checkpoint objective mismatch")
        if str(loaded.get("task")) != task or str(
            loaded.get("task_config")
        ) != task_config:
            raise ValueError("initial_checkpoint task condition mismatch")
        _validate_checkpoint_task_contract(
            loaded,
            expected_schema=expected_checkpoint_schema,
            expected_hash=task_contract_hash,
            multi_task=len(resolved_task_entries) > 1,
        )
        if not isinstance(loaded.get("adapter_state_dict"), Mapping):
            raise ValueError("initial_checkpoint has no adapter_state_dict")
        if bool(config["resume_optimizer_state"]):
            loaded_optimizer_kind = loaded.get("optimizer_kind")
            loaded_optimizer_bank = loaded.get("optimizer_bank")
            if expected_checkpoint_schema == CHECKPOINT_SCHEMA:
                # Checkpoints predating the explicit optimizer identity fields
                # were JointLoRA AdamW checkpoints over the single shared bank.
                loaded_optimizer_kind = loaded_optimizer_kind or "adamw"
                loaded_optimizer_bank = loaded_optimizer_bank or "both"
            if loaded_optimizer_kind != config["optimizer_kind"]:
                raise ValueError("initial_checkpoint optimizer kind mismatch")
            if loaded_optimizer_bank != config["trainable_bank"]:
                raise ValueError("initial_checkpoint optimizer bank mismatch")
            if expected_checkpoint_schema == DUAL_CHECKPOINT_SCHEMA:
                _validate_dual_optimizer_checkpoint_metadata(loaded)
        initial_checkpoint = loaded

    from experiments.robotwin_persistent_physics_worker import (
        PersistentNativePhysicsWorker,
    )

    slots: list[dict[str, Any]] = []
    runtime: NativeOnPolicyJointLabelRuntime | None = None
    round_rows: list[dict[str, Any]] = []
    metrics_path = output_dir / "round_metrics.jsonl"
    consumed_rounds: set[int] = set()
    try:
        # Fork every simulator worker before model/CUDA initialization.
        for entry in resolved_task_entries:
            entry_task = str(entry["task"])
            entry_task_config = str(entry["task_config"])
            entry_chunks = int(entry["chunks"])
            entry_max_control_steps = int(
                TASK_SPECS[entry_task].max_control_steps
            )
            for task_rollout_id, rollout in enumerate(entry["rollouts"]):
                seed = int(rollout["seed"])
                prompt = str(rollout["prompt"])
                (
                    task_env,
                    initial_observation,
                    format_obs,
                    add_init_pose,
                    parent_snapshot,
                    initial_eef_pose,
                    parent_snapshot_hash,
                ) = _setup_task_with_locked_prompt(
                    project_root=project_root,
                    task=entry_task,
                    seed=seed,
                    task_config=entry_task_config,
                    prompt=prompt,
                )
                # A persistent planner contains RNG/cache state that simulator
                # snapshot restore does not own.  Fork one never-before-stepped
                # worker per round up front (before CUDA) and consume it once.
                workers: list[Any] = []
                worker_initial_hashes: list[str] = []
                for _round_id in range(int(config["rounds"])):
                    worker = PersistentNativePhysicsWorker(
                        task_env=task_env,
                        prompt=prompt,
                        initial_eef_pose=initial_eef_pose,
                        format_obs=None,
                        materialize_renderer=None,
                        worker_mode="parent_render_bridge",
                    )
                    root = worker.snapshot()
                    workers.append(worker)
                    worker_initial_hashes.append(str(root["simulator_sha256"]))
                if len(set(worker_initial_hashes)) != 1:
                    raise NativeClosedLoopError(
                        "fresh round workers do not share the same simulator root"
                    )
                slots.append(
                    {
                        "rollout_id": int(rollout["rollout_id"]),
                        "task_rollout_id": int(task_rollout_id),
                        "task": entry_task,
                        "task_config": entry_task_config,
                        "chunks": entry_chunks,
                        "max_control_steps": entry_max_control_steps,
                        "prompt": prompt,
                        "seed": seed,
                        "dataset_role": str(rollout.get("role", "train")),
                        "task_env": task_env,
                        "initial_observation": initial_observation,
                        "format_obs": format_obs,
                        "add_init_pose": add_init_pose,
                        "parent_snapshot": parent_snapshot,
                        "parent_snapshot_hash": parent_snapshot_hash,
                        "initial_eef_pose": initial_eef_pose,
                        "workers": workers,
                        "worker_initial_hash": worker_initial_hashes[0],
                    }
                )

        # Simulator setup mutates global RNG.  Reset the adapter RNG only after
        # every worker has been forked so specialist/shared initialization is
        # independent of slot order and uses the same low-rank basis.
        _configure_cuda_memory_limit(config)
        np.random.seed(int(config["adapter_seed"]))
        torch.manual_seed(int(config["adapter_seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config["adapter_seed"]))

        runtime = NativeOnPolicyJointLabelRuntime(
            student_checkpoint=student,
            teacher_transformer=teacher,
            device=str(config.get("device", "cuda:0")),
            save_root=output_dir / "native_save",
            enable_offload=bool(config.get("enable_offload", True)),
            official_offload_parity=bool(
                config.get("official_offload_parity", True)
            ),
            adapter_rank=int(config.get("adapter_rank", 8)),
            adapter_state=initial_checkpoint_path,
            adapter_kind=str(config["adapter_kind"]),
            lora_alpha=float(config.get("lora_alpha", 8.0)),
            lora_dropout=float(config["lora_dropout"]),
            lora_block_indices=tuple(
                int(value)
                for value in config["lora_block_indices"]
            ),
        )
        _configure_teacher_solver_from_config(runtime, config)
        if config["adapter_kind"] == "dual_lora":
            runtime.select_adapter_trainable_bank(str(config["trainable_bank"]))
        optimizer = _fresh_optimizer(runtime, config)
        if initial_checkpoint is not None:
            expected_policy = str(initial_checkpoint["policy_version_after"])
            if _policy_version(runtime) != expected_policy:
                raise NativeClosedLoopError(
                    "initial checkpoint policy hash differs after runtime load"
                )
        if bool(config["resume_optimizer_state"]):
            checkpoint_parameter_names = initial_checkpoint.get(  # type: ignore[union-attr]
                "optimizer_parameter_names"
            )
            if list(checkpoint_parameter_names or []) != list(
                runtime.adapter_parameter_names
            ):
                raise ValueError(
                    "initial_checkpoint optimizer parameter manifest mismatch"
                )
            optimizer_state = initial_checkpoint.get("optimizer_state_dict")  # type: ignore[union-attr]
            if not isinstance(optimizer_state, Mapping):
                raise ValueError("initial_checkpoint has no optimizer_state_dict")
            optimizer.load_state_dict(deepcopy(dict(optimizer_state)))
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = float(config["learning_rate"])
                parameter_group["weight_decay"] = 0.0
            if config["adapter_kind"] == "dual_lora":
                state_dtypes = {
                    str(value.dtype)
                    for state in optimizer.state.values()
                    for value in state.values()
                    if isinstance(value, torch.Tensor)
                }
                if any(value != "torch.float32" for value in state_dtypes):
                    raise ValueError(
                        f"resumed dual optimizer state is not FP32: {state_dtypes}"
                    )
        starting_global_optimizer_step = (
            int(initial_checkpoint.get("global_optimizer_step", 0))
            if initial_checkpoint is not None
            else 0
        )
        global_optimizer_step = int(starting_global_optimizer_step)

        with metrics_path.open("w", encoding="utf-8") as metrics_handle:
            for round_id in range(int(config["rounds"])):
                if round_id in consumed_rounds:
                    raise NativeClosedLoopError("round package was already consumed")
                policy_before = _policy_version(runtime)
                trajectories: list[dict[str, Any]] = []
                outcomes: list[dict[str, Any]] = []
                trajectory_paths: list[Path] = []
                for slot in slots:
                    round_slot = dict(slot)
                    round_slot["worker"] = slot["workers"][round_id]
                    trajectory, outcome = _collect_rollout(
                        runtime=runtime,
                        slot=round_slot,
                        task=str(slot["task"]),
                        task_config=str(slot["task_config"]),
                        prompt=str(slot["prompt"]),
                        chunks=int(slot["chunks"]),
                        max_control_steps=int(slot["max_control_steps"]),
                        policy_version=policy_before,
                        round_id=round_id,
                        rollout_id=int(slot["rollout_id"]),
                        collection_group_id=config["collection_group_id"],
                        adapter_seed=int(config["adapter_seed"]),
                        student_checkpoint=student,
                        teacher_transformer=teacher,
                        objective=str(config["objective"]),
                    )
                    slot["workers"][round_id].close()
                    slot["workers"][round_id] = None
                    artifact_path = (
                        output_dir
                        / (
                            f"{slot['task']}_round_{round_id:02d}_"
                            f"rollout_{slot['task_rollout_id']:02d}.pt"
                        )
                    )
                    torch.save(trajectory, artifact_path)
                    trajectory_paths.append(artifact_path)
                    trajectories.append(trajectory)
                    outcomes.append(outcome)

                if run_mode == "collect":
                    bundle_path: Path | None = None
                    if config["adapter_kind"] == "dual_lora":
                        bundle = _build_rollout_bundle(
                            runtime=runtime,
                            config=config,
                            student=student,
                            task_contract_hash=task_contract_hash,
                            policy_version=policy_before,
                            round_id=round_id,
                            global_optimizer_step=global_optimizer_step,
                            trajectories=trajectories,
                            outcomes=outcomes,
                        )
                        bundle_path = Path(
                            str(config["rollout_bundle"])
                        ).expanduser().resolve()
                        bundle_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(bundle, bundle_path)
                    summary = {
                        "schema": (
                            DUAL_SUMMARY_SCHEMA
                            if config["adapter_kind"] == "dual_lora"
                            else COLLECT_SUMMARY_SCHEMA
                        ),
                        "status": "PASS",
                        "created_at": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                        "run_mode": "collect",
                        "adapter_kind": str(config["adapter_kind"]),
                        "teacher_loaded": True,
                        "teacher_controls_environment": False,
                        "behavior_policy_version": policy_before,
                        "collection_group_id": config["collection_group_id"],
                        "adapter_seed": int(config["adapter_seed"]),
                        "base_student_checkpoint": str(student),
                        "teacher_transformer": str(teacher),
                        "task_contract_hash": task_contract_hash,
                        "rollout_bundle": (
                            None if bundle_path is None else str(bundle_path)
                        ),
                        "trajectory_artifacts": [
                            str(path) for path in trajectory_paths
                        ],
                        "rollouts": int(len(trajectories)),
                        "labels": int(
                            sum(len(trajectory["labels"]) for trajectory in trajectories)
                        ),
                        "collection_outcomes": outcomes,
                    }
                    _write_json(output_dir / "summary.json", summary)
                    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
                    return 0

                solver_closure_path: Path | None = None
                if bool(config["pre_update_solver_closure"]):
                    solver_closure = _pre_update_solver_closure(runtime, trajectories)
                    solver_closure_path = (
                        output_dir
                        / f"solver_closure_round_{round_id:02d}.json"
                    )
                    _write_json(solver_closure_path, solver_closure)
                    if solver_closure["status"] != "PASS":
                        raise NativeClosedLoopError(
                            "pre-update multi-sigma solver closure failed"
                        )

                update = _update_round(
                    runtime=runtime,
                    optimizer=optimizer,
                    trajectories=trajectories,
                    round_id=round_id,
                    video_weight=float(config["video_weight"]),
                    action_weight=float(config["action_weight"]),
                    action_velocity_weight=float(
                        config["action_velocity_weight"]
                    ),
                    pseudo_huber_c=float(config["pseudo_huber_c"]),
                    max_grad_norm=float(config["max_grad_norm"]),
                    loss_reduction=str(config["loss_reduction"]),
                    trainable_bank=str(config["trainable_bank"]),
                    optimizer_kind=str(config["optimizer_kind"]),
                    max_update_norm=config["max_update_norm"],
                    objective=str(config["objective"]),
                    sigma_values=config["sigma_values"],
                    line_search_sigma_values=config["line_search_sigma_values"],
                    line_search_update_norms=config["line_search_update_norms"],
                    calibration_anchors_per_trajectory=int(
                        config["calibration_anchors_per_trajectory"]
                    ),
                    functional_closure_p95_max=float(
                        config["functional_closure_p95_max"]
                    ),
                    action_fm_weight=float(config["action_fm_weight"]),
                    effective_batch_size=int(config["effective_batch_size"]),
                    inner_epochs=int(config["inner_epochs"]),
                    ema_decay=float(config["ema_decay"]),
                    consistency_video_stride=int(
                        config["consistency_video_stride"]
                    ),
                    consistency_action_stride=int(
                        config["consistency_action_stride"]
                    ),
                    consistency_seed=int(config["consistency_seed"]),
                    coherent_tt_variant=str(config["coherent_tt_variant"]),
                    success_path_max_train_labels_per_trajectory=config[
                        "success_path_max_train_labels_per_trajectory"
                    ],
                )
                consumed_rounds.add(round_id)
                optimizer_steps_this_round = int(
                    update["optimizer_steps_this_round"]
                )
                global_optimizer_step += optimizer_steps_this_round
                if config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY:
                    if optimizer_steps_this_round <= 1:
                        raise NativeClosedLoopError(
                            "coherent TT round did not perform multiple steps"
                        )
                elif optimizer_steps_this_round != 1:
                    raise NativeClosedLoopError(
                        "legacy round did not perform exactly one step"
                    )
                policy_after = _policy_version(runtime)
                if policy_after == policy_before:
                    raise NativeClosedLoopError("optimizer step did not change the adapter")
                checkpoint_path = output_dir / f"checkpoint_round_{round_id:02d}.pt"
                _save_checkpoint(
                    path=checkpoint_path,
                    runtime=runtime,
                    optimizer=optimizer,
                    config=config,
                    round_id=round_id,
                    global_optimizer_step=global_optimizer_step,
                    policy_version_before=policy_before,
                    policy_version_after=policy_after,
                )
                row = {
                    "round_id": int(round_id),
                    "solver_closure": (
                        str(solver_closure_path)
                        if solver_closure_path is not None
                        else None
                    ),
                    "policy_version_before": policy_before,
                    "policy_version_after": policy_after,
                    "global_optimizer_step": int(global_optimizer_step),
                    "checkpoint": str(checkpoint_path),
                    "collection_outcomes": outcomes,
                    "update": update,
                }
                round_rows.append(row)
                metrics_handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                metrics_handle.flush()
                _write_json(output_dir / "summary.partial.json", {
                    "schema": (
                        DUAL_SUMMARY_SCHEMA
                        if config["adapter_kind"] == "dual_lora"
                        else SUMMARY_SCHEMA
                    ),
                    "status": "RUNNING",
                    "objective": str(config["objective"]),
                    "completed_rounds": int(round_id + 1),
                    "rounds": round_rows,
                })

        summary = {
            "schema": (
                DUAL_SUMMARY_SCHEMA
                if config["adapter_kind"] == "dual_lora"
                else SUMMARY_SCHEMA
            ),
            "status": "PASS",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "method": (
                "iterative on-policy mode-isolated distillation"
                if config["adapter_kind"] == "dual_lora"
                else (
                    "environment-on-policy coherent TT forward consistency "
                    "distillation"
                    if config["objective"] == OBJECTIVE_COHERENT_TT_CONSISTENCY
                    else (
                        "environment-on-policy denoising-hybrid multi-sigma x0 "
                        "consistency distillation"
                        if config["objective"] == OBJECTIVE_MULTI_SIGMA_X0
                        else (
                            "iterative on-policy endpoint distillation with "
                            "same-state velocity auxiliary"
                        )
                    )
                )
            ),
            "adapter_kind": str(config["adapter_kind"]),
            "objective": str(config["objective"]),
            "trainable_bank": str(config["trainable_bank"]),
            "optimizer_kind": str(config["optimizer_kind"]),
            "task": task,
            "tasks": [entry["task"] for entry in resolved_task_entries],
            "task_config": task_config,
            "task_contract_hash": task_contract_hash,
            "adapter_seed": int(config["adapter_seed"]),
            "rounds_completed": int(len(round_rows)),
            "optimizer_steps_this_run": int(
                sum(
                    int(row["update"]["optimizer_steps_this_round"])
                    for row in round_rows
                )
            ),
            "rollouts_per_round": int(len(slots)),
            "rollout_contracts": [
                {
                    "task": str(slot["task"]),
                    "seed": int(slot["seed"]),
                    "prompt": str(slot["prompt"]),
                    "chunks": int(slot["chunks"]),
                    "max_control_steps": int(slot["max_control_steps"]),
                    "dataset_role": str(slot["dataset_role"]),
                }
                for slot in slots
            ],
            "global_optimizer_steps": int(global_optimizer_step),
            "teacher_controls_environment": False,
            "retention_weight": 0.0,
            "replay": False,
            "initial_student": str(student),
            "initial_adapter_checkpoint": (
                str(initial_checkpoint_path)
                if initial_checkpoint_path is not None
                else None
            ),
            "starting_global_optimizer_step": int(
                starting_global_optimizer_step
            ),
            "final_checkpoint": round_rows[-1]["checkpoint"],
            "config": config,
            "rounds": round_rows,
        }
        _write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return 0
    finally:
        if runtime is not None:
            runtime.close()
        for slot in slots:
            try:
                for worker in slot["workers"]:
                    if worker is not None:
                        worker.close(force=sys.exc_info()[0] is not None)
            finally:
                slot["task_env"].close_env()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
