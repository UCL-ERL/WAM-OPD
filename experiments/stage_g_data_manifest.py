"""Policy provenance and matched-budget contracts for Stage G.

This module deliberately contains no model or simulator code.  It gives the
expensive rollout/Teacher-label pipeline stable identifiers, rejects duplicate
labels, enforces task-balanced training rows, and verifies that Static and
Iterative Adapter-OPD candidates used the same declared budget.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch


SCHEMA = "flashwam_stage_g_candidate_manifest_v2"
TEACHER_TARGET = "teacher_bridge_pathwise_macro_endpoint"
RETENTION_TARGET = "released_student_endpoint"


def _sha256_parts(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return _sha256_parts(
        (
            str(value.dtype).encode("ascii"),
            json.dumps(list(value.shape)).encode("ascii"),
            raw,
        )
    )


def build_policy_version(
    base_checkpoint: str | Path,
    policy_delta: str | Path | None = None,
) -> str:
    base = Path(base_checkpoint).resolve()
    if not base.exists():
        raise FileNotFoundError(base)
    base_identity = _sha256_parts((str(base).encode("utf-8"),))[:16]
    if policy_delta is None:
        return f"released-{base_identity}"
    delta = Path(policy_delta).resolve()
    if not delta.is_file():
        raise FileNotFoundError(delta)
    return f"adapter-{base_identity}-{file_sha256(delta)[:24]}"


def build_state_key(
    *,
    task_id: str,
    task_config: str,
    env_seed: int,
    frame_st_id: int,
    prompt: str,
    replay_context_path: str | Path,
    student_plan: torch.Tensor,
) -> str:
    context = Path(replay_context_path).resolve()
    if not context.is_file():
        raise FileNotFoundError(context)
    digest = _sha256_parts(
        (
            str(task_id).encode("utf-8"),
            str(task_config).encode("utf-8"),
            str(int(env_seed)).encode("ascii"),
            str(int(frame_st_id)).encode("ascii"),
            str(prompt).encode("utf-8"),
            file_sha256(context).encode("ascii"),
            tensor_sha256(student_plan).encode("ascii"),
        )
    )
    return f"state-{digest}"


def build_label_key(
    *,
    state_key: str,
    policy_version: str,
    action_base_noise: torch.Tensor,
    teacher_transformer: str | Path,
) -> str:
    digest = _sha256_parts(
        (
            str(state_key).encode("ascii"),
            str(policy_version).encode("ascii"),
            tensor_sha256(action_base_noise).encode("ascii"),
            str(Path(teacher_transformer).resolve()).encode("utf-8"),
            TEACHER_TARGET.encode("ascii"),
        )
    )
    return f"label-{digest}"


def build_rollout_key(
    *,
    task_id: str,
    env_seed: int,
    policy_version: str,
    replay_context_path: str | Path,
) -> str:
    context = Path(replay_context_path).resolve()
    if not context.is_file():
        raise FileNotFoundError(context)
    digest = _sha256_parts(
        (
            str(task_id).encode("utf-8"),
            str(int(env_seed)).encode("ascii"),
            str(policy_version).encode("ascii"),
            file_sha256(context).encode("ascii"),
        )
    )
    return f"rollout-{digest}"


def task_balanced_rows(
    rows: list[dict[str, Any]],
    *,
    per_task_count: int,
) -> list[dict[str, Any]]:
    if per_task_count <= 0:
        raise ValueError("per_task_count must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    if not grouped:
        raise ValueError("no rows to balance")
    for task, task_rows in grouped.items():
        task_rows.sort(
            key=lambda row: (
                int(row.get("round_index", 0)),
                str(row.get("label_key", "")),
            )
        )
        if len(task_rows) < per_task_count:
            raise ValueError(
                f"task {task} has {len(task_rows)} rows, "
                f"needs {per_task_count}"
            )

    selected = []
    for index in range(per_task_count):
        for task in sorted(grouped):
            selected.append(grouped[task][index])
    return selected


def two_round_task_balanced_rows(
    rows: list[dict[str, Any]],
    *,
    per_task_count_per_round: int,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row.get("round_index", -1))].append(row)
    if set(grouped) != {0, 1}:
        raise ValueError(
            f"positive row spec must contain rounds 0 and 1, got {sorted(grouped)}"
        )
    selected = []
    for round_index in (0, 1):
        selected.extend(
            task_balanced_rows(
                grouped[round_index],
                per_task_count=per_task_count_per_round,
            )
        )
    return selected


def validate_candidate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    candidate_kind = payload.get("candidate_kind")
    if candidate_kind not in ("static", "iterative"):
        errors.append("candidate_kind must be static or iterative")

    adapter = payload.get("adapter")
    if not isinstance(adapter, dict):
        errors.append("adapter must be an object")
    elif (
        adapter.get("kind") != "action_output_residual"
        or int(adapter.get("rank", 0)) <= 0
        or int(adapter.get("parameter_cap", 0)) > 5_000_000
    ):
        errors.append("adapter violates the shared action-output Stage G contract")

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("rows must be a non-empty list")
        rows = []
    required = {
        "task_id",
        "policy_version",
        "round_index",
        "label_key",
        "rollout_key",
        "classification",
        "target_kind",
        "artifact",
    }
    label_keys = []
    policy_versions: set[str] = set()
    round_indices: set[int] = set()
    positive_task_counts: Counter[str] = Counter()
    positive_rollout_keys: set[str] = set()
    positive_round_task_counts: dict[int, Counter[str]] = defaultdict(Counter)
    rollout_round_task_keys: dict[int, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    round_policy_versions: dict[int, set[str]] = defaultdict(set)
    teacher_queries = 0
    for index, row in enumerate(rows):
        missing = sorted(required - set(row)) if isinstance(row, dict) else sorted(required)
        if missing:
            errors.append(f"row {index} missing fields: {missing}")
            continue
        classification = row["classification"]
        target_kind = row["target_kind"]
        round_index = int(row["round_index"])
        if round_index < 0:
            errors.append(f"row {index} round_index must be non-negative")
        if classification == "positive":
            task = str(row["task_id"])
            policy = str(row["policy_version"])
            rollout_key = str(row["rollout_key"])
            positive_task_counts[task] += 1
            positive_round_task_counts[round_index][task] += 1
            rollout_round_task_keys[round_index][task].add(rollout_key)
            positive_rollout_keys.add(rollout_key)
            policy_versions.add(policy)
            round_indices.add(round_index)
            round_policy_versions[round_index].add(policy)
            if target_kind != TEACHER_TARGET:
                errors.append(f"row {index} positive target must be {TEACHER_TARGET}")
            teacher_queries += 1
        elif classification == "retention":
            if target_kind != RETENTION_TARGET:
                errors.append(f"row {index} retention target must be {RETENTION_TARGET}")
        else:
            errors.append(f"row {index} has non-trainable classification {classification!r}")
        label_keys.append(str(row["label_key"]))

    duplicates = sorted(
        key for key, count in Counter(label_keys).items() if count > 1
    )
    if duplicates:
        errors.append(f"duplicate label_key values: {duplicates[:4]}")
    if positive_task_counts and len(set(positive_task_counts.values())) != 1:
        errors.append(
            "positive rows are not task-balanced: "
            f"{dict(sorted(positive_task_counts.items()))}"
        )
    expected_tasks = set(positive_task_counts)
    for round_index, counts in sorted(positive_round_task_counts.items()):
        if set(counts) != expected_tasks or len(set(counts.values())) != 1:
            errors.append(
                f"round {round_index} positive rows are not task-balanced: "
                f"{dict(sorted(counts.items()))}"
            )
    rollout_round_task_counts = {
        round_index: {
            task: len(keys) for task, keys in sorted(task_keys.items())
        }
        for round_index, task_keys in sorted(rollout_round_task_keys.items())
    }
    for round_index, counts in rollout_round_task_counts.items():
        if set(counts) != expected_tasks or len(set(counts.values())) != 1:
            errors.append(
                f"round {round_index} positive rollouts are not task-balanced: "
                f"{counts}"
            )
    if round_indices != {0, 1}:
        errors.append(
            f"candidate must contain positive rows from rounds 0 and 1, got "
            f"{sorted(round_indices)}"
        )
    if any(len(policies) != 1 for policies in round_policy_versions.values()):
        errors.append(
            "each collection round must contain exactly one positive policy_version"
        )
    if candidate_kind == "static" and len(policy_versions) != 1:
        errors.append("static candidate must use one policy_version across both rounds")
    if candidate_kind == "iterative" and len(policy_versions) != 2:
        errors.append(
            "iterative candidate must use distinct policy versions in rounds 0 and 1"
        )

    budget = payload.get("budget")
    if not isinstance(budget, dict):
        errors.append("budget must be an object")
        budget = {}
    if int(budget.get("teacher_queries", -1)) != teacher_queries:
        errors.append(
            "budget.teacher_queries does not equal the number of positive "
            f"Teacher targets: declared={budget.get('teacher_queries')} actual={teacher_queries}"
        )
    actual_environment_episodes = len(positive_rollout_keys)
    if int(budget.get("environment_episodes", -1)) != actual_environment_episodes:
        errors.append(
            "budget.environment_episodes does not equal unique positive rollouts: "
            f"declared={budget.get('environment_episodes')} "
            f"actual={actual_environment_episodes}"
        )
    if int(budget.get("optimizer_steps", 0)) <= 0:
        errors.append("budget.optimizer_steps must be positive")

    return {
        "valid": not errors,
        "errors": errors,
        "candidate_kind": candidate_kind,
        "row_count": len(rows),
        "teacher_queries": teacher_queries,
        "environment_episodes": actual_environment_episodes,
        "policy_versions": sorted(policy_versions),
        "round_indices": sorted(round_indices),
        "positive_task_counts": dict(sorted(positive_task_counts.items())),
        "positive_round_task_counts": {
            str(round_index): dict(sorted(counts.items()))
            for round_index, counts in sorted(positive_round_task_counts.items())
        },
        "rollout_round_task_counts": {
            str(round_index): counts
            for round_index, counts in rollout_round_task_counts.items()
        },
    }


def matched_budget_errors(
    static: dict[str, Any],
    iterative: dict[str, Any],
) -> list[str]:
    errors = []
    for name, payload in (("static", static), ("iterative", iterative)):
        result = validate_candidate_manifest(payload)
        if not result["valid"]:
            errors.extend(f"{name}: {item}" for item in result["errors"])
    if static.get("adapter") != iterative.get("adapter"):
        errors.append("adapter configurations do not match")
    static_budget = static.get("budget", {})
    iterative_budget = iterative.get("budget", {})
    for key in ("environment_episodes", "teacher_queries", "optimizer_steps"):
        if static_budget.get(key) != iterative_budget.get(key):
            errors.append(
                f"budget.{key} mismatch: static={static_budget.get(key)} "
                f"iterative={iterative_budget.get(key)}"
            )
    static_counts = validate_candidate_manifest(static)["positive_task_counts"]
    iterative_counts = validate_candidate_manifest(iterative)["positive_task_counts"]
    if static_counts != iterative_counts:
        errors.append(
            "positive per-task label budgets do not match: "
            f"static={static_counts} iterative={iterative_counts}"
        )
    static_result = validate_candidate_manifest(static)
    iterative_result = validate_candidate_manifest(iterative)
    for key in ("positive_round_task_counts", "rollout_round_task_counts"):
        if static_result[key] != iterative_result[key]:
            errors.append(
                f"{key} do not match: static={static_result[key]} "
                f"iterative={iterative_result[key]}"
            )
    return errors
