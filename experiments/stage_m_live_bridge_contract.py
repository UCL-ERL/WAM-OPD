"""Fail-closed data contracts for live/on-policy Stage-M Bridge labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

try:
    from .stage_g_data_manifest import file_sha256
    from .stage_h_context_contract import formatted_observation_sha256
except ImportError:  # Direct execution from the experiments/ script directory.
    from stage_g_data_manifest import file_sha256
    from stage_h_context_contract import formatted_observation_sha256


LIVE_CONTEXT_SCHEMA = "flashwam_stage_m_live_context_v1"
LIVE_LABEL_KIND = "stage_m_live_teacher_bridge_label"


def _digest_value(digest: Any, value: object) -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(value, sort_keys=True).encode("utf-8"))
        return
    if isinstance(value, Path):
        _digest_value(digest, str(value))
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value):
            if key == "semantic_sha256":
                continue
            _digest_value(digest, str(key))
            _digest_value(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(len(value).to_bytes(8, "little"))
        for item in value:
            _digest_value(digest, item)
        return
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(np.asarray(value))
    digest.update(b"array\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())


def semantic_sha256(value: object) -> str:
    digest = hashlib.sha256()
    _digest_value(digest, value)
    return digest.hexdigest()


def _validate_chunk_sequence(chunks: object) -> tuple[list[dict[str, object]], int]:
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("live context chunks must be a non-empty list")
    frame_st_id = 0
    validated = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"chunk {index} must be a dictionary")
        if int(chunk.get("frame_st_id", -1)) != frame_st_id:
            raise ValueError(
                "non-contiguous live context: "
                f"chunk={index} expected frame_st_id={frame_st_id} "
                f"actual={chunk.get('frame_st_id')}"
            )
        observations = chunk.get("observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError(f"chunk {index} has no observations")
        if not all(isinstance(item, dict) for item in observations):
            raise ValueError(f"chunk {index} observations must be dictionaries")
        action = np.asarray(chunk.get("env_action"))
        if action.ndim != 3 or action.shape[0] != 16 or action.shape[2] != 16:
            raise ValueError(
                f"chunk {index} action has unexpected shape {action.shape}"
            )
        frame_st_id += int(action.shape[1])
        validated.append(chunk)
    return validated, frame_st_id


def build_live_context(
    *,
    live_context_id: str,
    task: str,
    task_config: str,
    seed: int,
    prompt: str,
    model_name: str,
    policy_version: str,
    policy_delta_path: str | Path | None,
    initial_observation: dict[str, object],
    chunks: list[dict[str, object]],
    prefix_runtime_sha256: Iterable[str],
    prefix_runtime_components: Iterable[dict[str, str]],
    simulator_snapshot_sha256: str,
    start_observation_sha256: str,
    simulator_step_count: int,
    capture_started_unix_ns: int,
) -> dict[str, object]:
    chunks, frame_st_id = _validate_chunk_sequence(chunks)
    runtime_hashes = list(prefix_runtime_sha256)
    runtime_components = list(prefix_runtime_components)
    if len(runtime_hashes) != len(chunks):
        raise ValueError("prefix runtime hash count must match chunk count")
    if len(runtime_components) != len(chunks):
        raise ValueError("prefix runtime component count must match chunk count")
    observation_hashes = [
        [formatted_observation_sha256(item) for item in chunk["observations"]]
        for chunk in chunks
    ]
    payload: dict[str, object] = {
        "schema": LIVE_CONTEXT_SCHEMA,
        "artifact_kind": "stage_m_live_student_context",
        "simulation_only": True,
        "live_context_id": str(live_context_id),
        "task": str(task),
        "task_config": str(task_config),
        "seed": int(seed),
        "prompt": str(prompt),
        "model_name": str(model_name),
        "policy_version": str(policy_version),
        "policy_delta_path": (
            str(Path(policy_delta_path).resolve()) if policy_delta_path else None
        ),
        "initial_observation": initial_observation,
        "initial_observation_sha256": formatted_observation_sha256(
            initial_observation
        ),
        "chunks": chunks,
        "chunk_observation_sha256": observation_hashes,
        "prefix_runtime_sha256": runtime_hashes,
        "prefix_runtime_components": runtime_components,
        "frame_st_id": frame_st_id,
        "simulator_snapshot_sha256": str(simulator_snapshot_sha256),
        "start_observation_sha256": str(start_observation_sha256),
        "simulator_step_count": int(simulator_step_count),
        "capture_started_unix_ns": int(capture_started_unix_ns),
    }
    payload["semantic_sha256"] = semantic_sha256(payload)
    return payload


def validate_live_context(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("live context must be a dictionary")
    if payload.get("schema") != LIVE_CONTEXT_SCHEMA:
        raise ValueError(f"live context schema must be {LIVE_CONTEXT_SCHEMA}")
    if payload.get("artifact_kind") != "stage_m_live_student_context":
        raise ValueError("live context artifact_kind mismatch")
    required_strings = (
        "live_context_id",
        "task",
        "task_config",
        "prompt",
        "model_name",
        "policy_version",
        "simulator_snapshot_sha256",
        "start_observation_sha256",
    )
    for key in required_strings:
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"live context {key} must be a non-empty string")
    chunks, frame_st_id = _validate_chunk_sequence(payload.get("chunks"))
    if int(payload.get("frame_st_id", -1)) != frame_st_id:
        raise ValueError("live context frame_st_id mismatch")
    runtime_hashes = payload.get("prefix_runtime_sha256")
    components = payload.get("prefix_runtime_components")
    if not isinstance(runtime_hashes, list) or len(runtime_hashes) != len(chunks):
        raise ValueError("live context runtime hash count mismatch")
    if not isinstance(components, list) or len(components) != len(chunks):
        raise ValueError("live context runtime component count mismatch")
    expected_observation_hashes = [
        [formatted_observation_sha256(item) for item in chunk["observations"]]
        for chunk in chunks
    ]
    if payload.get("chunk_observation_sha256") != expected_observation_hashes:
        raise ValueError("live context chunk observation hashes mismatch")
    initial = payload.get("initial_observation")
    if not isinstance(initial, dict):
        raise ValueError("live context initial observation must be a dictionary")
    if payload.get("initial_observation_sha256") != formatted_observation_sha256(
        initial
    ):
        raise ValueError("live context initial observation hash mismatch")
    expected_semantic = semantic_sha256(payload)
    if payload.get("semantic_sha256") != expected_semantic:
        raise ValueError(
            "live context semantic_sha256 mismatch: "
            f"expected={expected_semantic} actual={payload.get('semantic_sha256')}"
        )
    return payload


def validate_live_bridge_label(
    *,
    context: dict[str, object],
    context_path: str | Path,
    label: object,
) -> dict[str, object]:
    validate_live_context(context)
    if not isinstance(label, dict):
        raise ValueError("live Bridge label must be a dictionary")
    expected = {
        "artifact_kind": LIVE_LABEL_KIND,
        "live_context_id": context["live_context_id"],
        "replay_context_sha256": file_sha256(context_path),
        "replay_context_semantic_sha256": context["semantic_sha256"],
        "task_id": context["task"],
        "task_config": context["task_config"],
        "env_seed": context["seed"],
        "frame_st_id": context["frame_st_id"],
        "prompt": context["prompt"],
        "policy_version": context["policy_version"],
    }
    for key, value in expected.items():
        if label.get(key) != value:
            raise ValueError(
                f"live Bridge label {key} mismatch: "
                f"expected={value!r} actual={label.get(key)!r}"
            )
    for key in ("student_env_action", "teacher_bridge_env_action"):
        if key not in label:
            raise ValueError(f"live Bridge label lacks {key}")
        shape = tuple(np.asarray(label[key]).shape)
        if shape != (16, 2, 16):
            raise ValueError(f"live Bridge label {key} shape is {shape}")
    return label


def build_teacher_bridge_command(
    *,
    python: Path,
    script: Path,
    project_root: Path,
    student: Path,
    teacher_transformer: Path,
    context: Path,
    label: Path,
    runtime_audit: Path,
    live_context_id: str,
    diffusion_seed: int,
    context_chunks: int,
    teacher_gpu: int,
) -> tuple[list[str], dict[str, str]]:
    if int(teacher_gpu) not in (6, 7):
        raise ValueError("Stage M Teacher GPU must be 6 or 7")
    command = [
        str(python),
        str(script),
        "--student",
        str(student),
        "--teacher-transformer",
        str(teacher_transformer),
        "--replay-context",
        str(context),
        "--context-chunks",
        str(int(context_chunks)),
        "--seed",
        str(int(diffusion_seed)),
        "--student-action-steps",
        "1",
        "--stage-m-live-context-id",
        str(live_context_id),
        "--save-actions",
        str(label),
        "--runtime-state-audit-output",
        str(runtime_audit),
    ]
    environment = {
        "CUDA_VISIBLE_DEVICES": str(int(teacher_gpu)),
        "PROJECT_ROOT": str(project_root),
    }
    return command, environment
