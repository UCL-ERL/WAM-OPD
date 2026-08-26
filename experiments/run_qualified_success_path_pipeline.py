"""Validate and run the fixed pipeline for a qualified OPD task.

The task-specific input is a JSON manifest.  The state machine and all
research-facing invariants live here so that a new qualified task cannot
silently change the 8/4/2/6 split, the three-epoch recipe, checkpoint
selection, held-out isolation, or the optional-evaluation boundary.

The controller is deliberately fail-closed.  It never deletes output and it
never retries an ambiguous/partial phase.  Run the controller itself in one
tmux session when asynchronous execution is desired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Mapping, Sequence


PIPELINE_SCHEMA = "waopd_qualified_success_path_pipeline_v1"
CONTRACT_SCHEMA = "waopd_qualified_success_path_contract_receipt_v1"
STATE_SCHEMA = "waopd_qualified_success_path_state_v1"
FINAL_SCHEMA = "waopd_qualified_success_path_final_summary_v1"
GATE_SCHEMA = "flashwam_stage_g_gap_gate_decision_v1"

SPLITS = {
    "train": [0, 8],
    "calibration": [8, 12],
    "screening": [12, 14],
    "heldout": [14, 20],
}
ROLES = ["train"] * 8 + ["calibration"] * 4 + ["screening"] * 2 + ["heldout"] * 6
SELECTION_RULE = [
    "adapted_success_count_desc",
    "sum_paired_max_ordinal_delta_desc",
    "stage_improvement_minus_regression_desc",
    "fixed_calibration_total_loss_asc",
    "epoch_asc",
]
STRICT_GATE_THRESHOLDS = {
    "expected_pairs": 8,
    "min_teacher_successes": 6,
    "min_net_advantage": 3,
    "require_teacher_only_gt_student_only": True,
}
RECIPE = {
    "objective": "coherent_tt_consistency",
    "coherent_tt_variant": "success_path_v1",
    "rounds": 1,
    "teacher_video_steps": 25,
    "teacher_video_exec_steps": None,
    "teacher_action_steps": 50,
    "adapter_seed": 20260820,
    "adapter_kind": "joint_lora",
    "trainable_bank": "both",
    "adapter_rank": 8,
    "lora_alpha": 8.0,
    "lora_dropout": 0.0,
    "optimizer_kind": "adamw",
    "learning_rate": 2e-5,
    "video_weight": 1.0,
    "action_weight": 1.0,
    "action_fm_weight": 0.2,
    "action_velocity_weight": 0.0,
    "effective_batch_size": 4,
    "calibration_anchors_per_trajectory": 2,
    "loss_reduction": "mean_trajectories_mean_labels",
    "retention_weight": 0.0,
    "pre_update_solver_closure": False,
}

CORE_STAGES = ("collection", "update", "screening", "heldout")
TERMINAL_CORE_STAGE = "heldout"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def _path(value: Any, label: str, *, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"{label} must be an absolute path")
    result = Path(value).expanduser().resolve()
    if must_exist and not result.exists():
        raise FileNotFoundError(f"{label} is missing: {result}")
    return result


def _ssd_path(value: Any, label: str, *, must_exist: bool = False) -> Path:
    result = _path(value, label, must_exist=must_exist)
    if not str(result).startswith("/ssd/data/"):
        raise ValueError(f"{label} must be under /ssd/data: {result}")
    return result


def _validate_gate(path: Path, task: str) -> dict[str, Any]:
    decision = _read_json(path)
    expected = {
        "schema": GATE_SCHEMA,
        "status": "PASS",
        "task_id": task,
        "episodes": 8,
        "thresholds": STRICT_GATE_THRESHOLDS,
    }
    for key, value in expected.items():
        _equal(decision.get(key), value, f"qualification.{key}")
    return decision


def _validate_records(split: Mapping[str, Any], task: str) -> list[dict[str, Any]]:
    _equal(split.get("task"), task, "split.task")
    _equal(split.get("binding_status"), "BOUND", "split.binding_status")
    _equal(split.get("split_indices"), SPLITS, "split.split_indices")
    raw_records = _require_list(split.get("episode_records"), "split.episode_records")
    if len(raw_records) != 20:
        raise ValueError("split must contain exactly 20 episode_records")
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_records):
        row = _require_mapping(raw, f"split.episode_records[{index}]")
        _equal(set(row), {"episode", "seed", "instruction", "role"}, f"split record {index} fields")
        _equal(row.get("episode"), index, f"split record {index}.episode")
        _equal(row.get("role"), ROLES[index], f"split record {index}.role")
        if not isinstance(row.get("seed"), int):
            raise TypeError(f"split record {index}.seed must be an integer")
        if not isinstance(row.get("instruction"), str) or not str(row["instruction"]).strip():
            raise ValueError(f"split record {index}.instruction is empty")
        if any("success" in str(key).lower() for key in row):
            raise ValueError(f"split record {index} contains outcome data")
        records.append(dict(row))
    seeds = [int(row["seed"]) for row in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError("split seeds must be unique")
    _equal(split.get("accepted_seed_list"), seeds, "split.accepted_seed_list")
    projection = [
        {key: row[key] for key in ("episode", "seed", "instruction")}
        for row in records
    ]
    _equal(
        split.get("ordered_metadata_projection_sha256"),
        _json_sha256(projection),
        "split.ordered_metadata_projection_sha256",
    )
    return records


def _rollout(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rollout_id": int(row["episode"]),
        "seed": int(row["seed"]),
        "prompt": str(row["instruction"]),
        "role": str(row["role"]),
    }


def _validate_binding(
    config: Mapping[str, Any],
    *,
    split_path: Path,
    split_sha: str,
    metadata_sha: str,
    label: str,
) -> None:
    _equal(config.get("source_binding_status"), "BOUND", f"{label}.source_binding_status")
    _equal(
        Path(str(config.get("split_manifest", ""))).expanduser().resolve(),
        split_path,
        f"{label}.split_manifest",
    )
    _equal(config.get("split_manifest_sha256"), split_sha, f"{label}.split_manifest_sha256")
    _equal(config.get("split_indices"), SPLITS, f"{label}.split_indices")
    _equal(
        config.get("ordered_metadata_projection_sha256"),
        metadata_sha,
        f"{label}.ordered_metadata_projection_sha256",
    )


def _validate_recipe(
    config: Mapping[str, Any],
    *,
    task: str,
    task_config: str,
    chunks: int,
    project_root: Path,
    label: str,
) -> None:
    expected = {
        **RECIPE,
        "project_root": str(project_root),
        "task": task,
        "task_config": task_config,
        "chunks": chunks,
    }
    for key, value in expected.items():
        _equal(config.get(key), value, f"{label}.{key}")
    _equal(config.get("lora_block_indices"), list(range(30)), f"{label}.lora_block_indices")
    for key in ("student", "teacher_transformer", "output_dir"):
        _ssd_path(config.get(key), f"{label}.{key}")


def _validate_gpu_policy(raw: Any) -> dict[str, Any]:
    policy = dict(_require_mapping(raw, "gpu_policy"))
    expected = {
        "stable_gpu_count": 4,
        "world_size_change_boundary": "epoch",
        "allow_mid_epoch_elastic_resize": False,
    }
    for key, value in expected.items():
        _equal(policy.get(key), value, f"gpu_policy.{key}")
    allowed = _require_list(policy.get("allowed_gpu_ids"), "gpu_policy.allowed_gpu_ids")
    if (
        len(allowed) < int(policy["stable_gpu_count"])
        or len(set(allowed)) != len(allowed)
        or any(isinstance(value, bool) or not isinstance(value, int) or value not in range(8) for value in allowed)
    ):
        raise ValueError(
            "gpu_policy.allowed_gpu_ids must contain at least four unique ids in [0,7]"
        )
    burst_max = policy.get("burst_max_gpu_count")
    if (
        isinstance(burst_max, bool)
        or not isinstance(burst_max, int)
        or burst_max < int(policy["stable_gpu_count"])
        or burst_max > len(allowed)
    ):
        raise ValueError(
            "gpu_policy.burst_max_gpu_count must be between stable_gpu_count and the allowed GPU count"
        )
    allow_existing = policy.get("allow_existing_compute_processes", False)
    if not isinstance(allow_existing, bool):
        raise TypeError("gpu_policy.allow_existing_compute_processes must be boolean")
    policy["allow_existing_compute_processes"] = allow_existing
    graphics_exclusive = policy.get(
        "require_graphics_exclusive_for_rollout", True
    )
    if not isinstance(graphics_exclusive, bool):
        raise TypeError(
            "gpu_policy.require_graphics_exclusive_for_rollout must be boolean"
        )
    policy["require_graphics_exclusive_for_rollout"] = graphics_exclusive
    thresholds = _require_mapping(policy.get("min_free_mib"), "gpu_policy.min_free_mib")
    for phase in ("collection", "update", "evaluation"):
        value = thresholds.get(phase)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"gpu_policy.min_free_mib.{phase} must be positive")
    train_world_size = policy.get("train_world_size")
    if train_world_size not in (1, 4):
        raise ValueError("gpu_policy.train_world_size must be 1 or 4")
    return policy


def validate_contract(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _read_json(manifest_path)
    _equal(manifest.get("schema"), PIPELINE_SCHEMA, "manifest.schema")
    task = manifest.get("task")
    if not isinstance(task, str) or not re.fullmatch(r"[a-z0-9_]+", task):
        raise ValueError("manifest.task must be a lowercase task id")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-z0-9_.-]+", run_id):
        raise ValueError("manifest.run_id is invalid")

    workspace = _path(manifest.get("workspace"), "manifest.workspace")
    project_root = _path(manifest.get("project_root"), "manifest.project_root")
    python_bin = _path(manifest.get("python_bin"), "manifest.python_bin")
    formal_root = _ssd_path(manifest.get("formal_root"), "manifest.formal_root")
    collection_root = _ssd_path(manifest.get("collection_root"), "manifest.collection_root")
    if formal_root == collection_root:
        raise ValueError("formal_root and collection_root must be distinct")

    decision_path = _ssd_path(
        manifest.get("qualification_decision"),
        "manifest.qualification_decision",
        must_exist=True,
    )
    _validate_gate(decision_path, task)

    split_path = _path(manifest.get("split_manifest"), "manifest.split_manifest")
    split = _read_json(split_path)
    records = _validate_records(split, task)
    split_sha = _sha256(split_path)
    metadata_sha = str(split["ordered_metadata_projection_sha256"])
    task_config = str(split.get("task_config"))

    training_path = _path(manifest.get("training_config"), "manifest.training_config")
    training = _read_json(training_path)
    chunks = int(training.get("chunks", -1))
    if chunks <= 0:
        raise ValueError("training.chunks must be positive")
    _validate_binding(
        training,
        split_path=split_path,
        split_sha=split_sha,
        metadata_sha=metadata_sha,
        label="training",
    )
    _validate_recipe(
        training,
        task=task,
        task_config=task_config,
        chunks=chunks,
        project_root=project_root,
        label="training",
    )
    _equal(training.get("run_mode"), "trajectory_update", "training.run_mode")
    _equal(training.get("inner_epochs"), 3, "training.inner_epochs")
    _equal(training.get("initial_checkpoint"), None, "training.initial_checkpoint")
    _equal(training.get("resume_optimizer_state"), False, "training.resume_optimizer_state")
    _equal(training.get("rollouts"), [_rollout(row) for row in records[:12]], "training.rollouts")
    _equal(
        Path(str(training.get("output_dir", ""))).expanduser().resolve(),
        formal_root / "update",
        "training.output_dir",
    )
    artifacts = _require_list(training.get("trajectory_artifacts"), "training.trajectory_artifacts")
    if len(artifacts) != 12:
        raise ValueError("training must bind exactly 12 trajectory artifacts")
    for index, artifact in enumerate(artifacts):
        path = _ssd_path(artifact, f"training.trajectory_artifacts[{index}]")
        if not str(path).startswith(f"{collection_root}/"):
            raise ValueError(f"trajectory artifact is outside collection_root: {path}")

    collection_paths = [
        _path(value, f"manifest.collection_configs[{index}]")
        for index, value in enumerate(
            _require_list(manifest.get("collection_configs"), "manifest.collection_configs")
        )
    ]
    if len(collection_paths) != 4:
        raise ValueError("pipeline requires exactly four collection configs")
    collected: list[dict[str, Any]] = []
    output_roots: set[Path] = set()
    for shard, path in enumerate(collection_paths):
        config = _read_json(path)
        label = f"collection_{shard:02d}"
        _validate_binding(
            config,
            split_path=split_path,
            split_sha=split_sha,
            metadata_sha=metadata_sha,
            label=label,
        )
        _validate_recipe(
            config,
            task=task,
            task_config=task_config,
            chunks=chunks,
            project_root=project_root,
            label=label,
        )
        _equal(config.get("run_mode"), "collect", f"{label}.run_mode")
        _equal(config.get("inner_epochs"), 1, f"{label}.inner_epochs")
        expected_rollouts = [_rollout(records[index]) for index in range(shard, 12, 4)]
        _equal(config.get("rollouts"), expected_rollouts, f"{label}.rollouts")
        output = _ssd_path(config.get("output_dir"), f"{label}.output_dir")
        if not str(output).startswith(f"{collection_root}/") or output in output_roots:
            raise ValueError(f"{label}.output_dir is not a unique collection shard")
        output_roots.add(output)
        collected.extend(expected_rollouts)
    _equal(
        sorted(collected, key=lambda row: int(row["rollout_id"])),
        [_rollout(row) for row in records[:12]],
        "collection union",
    )

    protocol_path = _path(manifest.get("eval_protocol"), "manifest.eval_protocol")
    protocol = _read_json(protocol_path)
    _validate_binding(
        protocol,
        split_path=split_path,
        split_sha=split_sha,
        metadata_sha=metadata_sha,
        label="protocol",
    )
    expected_protocol = {
        "task": task,
        "task_config": task_config,
        "chunks": chunks,
        "epoch_ids": [1, 2, 3],
        "selection_rule": SELECTION_RULE,
        "training_config": str(training_path),
        "training_summary": str(formal_root / "update" / "summary.json"),
        "output_root": str(formal_root / "eval"),
    }
    for key, value in expected_protocol.items():
        _equal(protocol.get(key), value, f"protocol.{key}")
    screening = _require_mapping(protocol.get("screening"), "protocol.screening")
    heldout = _require_mapping(protocol.get("heldout"), "protocol.heldout")
    screen_rows = [
        {"seed": row["seed"], "instruction": row["instruction"]}
        for row in records[12:14]
    ]
    heldout_rows = [
        {"seed": row["seed"], "instruction": row["instruction"]}
        for row in records[14:20]
    ]
    _equal(screening.get("episode_records"), screen_rows, "protocol.screening.episode_records")
    _equal(heldout.get("episode_records"), heldout_rows, "protocol.heldout.episode_records")
    screen_banks = _require_list(screening.get("noise_base_seeds"), "protocol.screening.noise_base_seeds")
    heldout_banks = _require_list(heldout.get("noise_base_seeds"), "protocol.heldout.noise_base_seeds")
    if len(screen_banks) != 2 or len(set(screen_banks)) != 2:
        raise ValueError("screening must use exactly two distinct noise banks")
    if len(heldout_banks) != 2 or len(set(heldout_banks)) != 2:
        raise ValueError("heldout must use exactly two distinct noise banks")
    if set(screen_banks) & set(heldout_banks):
        raise ValueError("screening and heldout noise banks overlap")
    if {row["seed"] for row in screen_rows} & {row["seed"] for row in heldout_rows}:
        raise ValueError("screening and heldout episode seeds overlap")
    expected_heldout_pairs = [
        {"noise_base_seed": bank, "seed": row["seed"]}
        for bank in heldout_banks
        for row in heldout_rows
    ]
    _equal(heldout.get("record_pairs"), expected_heldout_pairs, "protocol.heldout.record_pairs")

    optional_eval = dict(_require_mapping(manifest.get("optional_eval"), "manifest.optional_eval"))
    if optional_eval.get("enabled") not in (True, False):
        raise TypeError("optional_eval.enabled must be boolean")
    _equal(optional_eval.get("after_stage"), TERMINAL_CORE_STAGE, "optional_eval.after_stage")
    _equal(optional_eval.get("selection_input"), False, "optional_eval.selection_input")
    argv = _require_list(optional_eval.get("argv", []), "optional_eval.argv")
    if optional_eval["enabled"]:
        if not argv or not all(isinstance(value, str) and value for value in argv):
            raise ValueError("enabled optional_eval requires a non-empty argv")
    elif argv:
        raise ValueError("disabled optional_eval must have an empty argv")

    gpu_policy = _validate_gpu_policy(manifest.get("gpu_policy"))
    execution = _require_mapping(manifest.get("execution"), "manifest.execution")
    backend = execution.get("training_backend")
    if backend not in ("joint_lora_single_process_v1", "joint_lora_ddp_v1"):
        raise ValueError(
            "execution.training_backend must be a certified JointLoRA backend"
        )
    expected_world_size = 1 if backend == "joint_lora_single_process_v1" else 4
    _equal(
        gpu_policy.get("train_world_size"),
        expected_world_size,
        "gpu_policy.train_world_size for execution.training_backend",
    )
    update_argv = _require_list(execution.get("update_argv"), "execution.update_argv")
    if not update_argv or not all(isinstance(value, str) and value for value in update_argv):
        raise ValueError("execution.update_argv must be a non-empty argv")

    return {
        "schema": CONTRACT_SCHEMA,
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "task": task,
        "run_id": run_id,
        "workspace": str(workspace),
        "python_bin": str(python_bin),
        "formal_root": str(formal_root),
        "collection_root": str(collection_root),
        "split_counts": {"train": 8, "calibration": 4, "screening": 2, "heldout": 6},
        "epochs": 3,
        "adapter": {"kind": "joint_lora", "rank": 8, "layers": list(range(30))},
        "effective_batch_size": 4,
        "heldout_exact_pairs": 12,
        "gpu_policy": gpu_policy,
        "training_backend": backend,
        "optional_eval_enabled": bool(optional_eval["enabled"]),
        "optional_eval_selection_input": False,
    }


def _pass_json(path: Path) -> bool:
    return path.is_file() and _read_json(path).get("status") == "PASS"


def _phase_status(manifest: Mapping[str, Any]) -> dict[str, Any]:
    formal_root = Path(str(manifest["formal_root"]))
    collection_configs = [
        _read_json(Path(value)) for value in manifest["collection_configs"]
    ]
    collection_summaries = [
        Path(str(config["output_dir"])) / "summary.json" for config in collection_configs
    ]
    collection_passes = [_pass_json(path) for path in collection_summaries]
    collection_done = all(collection_passes)
    collection_started = any(path.exists() for path in collection_summaries)
    collection_started = collection_started or any(
        (Path(str(config["output_dir"])) / "collect.log").exists()
        for config in collection_configs
    )
    update_summary = formal_root / "update" / "summary.json"
    checkpoints = [
        formal_root / "update" / "epoch_checkpoints" / f"checkpoint_epoch_{epoch:02d}.pt"
        for epoch in (1, 2)
    ]
    checkpoints.append(formal_root / "update" / "checkpoint_trajectory_update.pt")
    update_done = _pass_json(update_summary) and all(path.is_file() for path in checkpoints)
    update_started = (
        update_summary.exists()
        or any(path.exists() for path in checkpoints)
        or (formal_root / "logs" / "update.log").exists()
    )
    selection = formal_root / "eval" / "screen" / "selection.json"
    screening_done = _pass_json(selection)
    screening_started = selection.exists() or (
        formal_root / "eval" / "logs" / "screening.log"
    ).exists()
    heldout = formal_root / "eval" / "heldout" / "summary.json"
    heldout_done = _pass_json(heldout)
    heldout_started = heldout.exists() or (
        formal_root / "eval" / "logs" / "heldout.log"
    ).exists()
    optional = _require_mapping(manifest["optional_eval"], "optional_eval")
    optional_receipt = formal_root / "optional_eval" / "summary.json"
    optional_done = (not bool(optional["enabled"])) or _pass_json(optional_receipt)
    optional_started = optional_receipt.exists() or (
        formal_root / "optional_eval" / "optional_eval.log"
    ).exists()
    return {
        "collection": {
            "done": collection_done,
            "started": collection_started,
            "receipts": [str(path) for path in collection_summaries],
        },
        "update": {
            "done": update_done,
            "started": update_started,
            "summary": str(update_summary),
            "checkpoints": [str(path) for path in checkpoints],
        },
        "screening": {
            "done": screening_done,
            "started": screening_started,
            "selection": str(selection),
        },
        "heldout": {
            "done": heldout_done,
            "started": heldout_started,
            "summary": str(heldout),
        },
        "optional_eval": {
            "enabled": bool(optional["enabled"]),
            "done": optional_done,
            "started": optional_started,
            "summary": str(optional_receipt),
        },
    }


def _next_stage(status: Mapping[str, Any]) -> str | None:
    for stage in (*CORE_STAGES, "optional_eval"):
        phase = _require_mapping(status[stage], stage)
        if bool(phase["done"]):
            continue
        if bool(phase.get("started")):
            raise RuntimeError(
                f"{stage} has partial or non-PASS output; refusing automatic rerun"
            )
        if not bool(phase["done"]):
            return stage
    return None


def _gpu_rows() -> list[dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows: list[dict[str, int]] = []
    for line in result.stdout.splitlines():
        raw_index, raw_free = line.split(",", maxsplit=1)
        rows.append({"index": int(raw_index.strip()), "free_mib": int(raw_free.strip())})
    return rows


def _gpu_has_process(index: int) -> bool:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return any(line.strip().isdigit() for line in result.stdout.splitlines())


def _parse_pmon_graphics_processes(
    output: str, *, expected_gpu: int
) -> list[dict[str, Any]]:
    """Return only C+G processes from one ``nvidia-smi pmon`` snapshot.

    RoboTwin renderers also create tiny secondary G-only contexts on GPU 0.
    A bounded first/second-get_obs probe showed that those contexts do not
    prevent a new rollout on GPU 0.  Existing C+G work on the target GPU is
    the unsafe overlay: it can starve SAPIEN/Vulkan camera capture.
    """

    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 3:
            raise RuntimeError(f"unexpected nvidia-smi pmon row: {line!r}")
        try:
            gpu = int(fields[0])
        except ValueError as exc:
            raise RuntimeError(
                f"unexpected nvidia-smi pmon row: {line!r}"
            ) from exc
        if gpu != expected_gpu:
            raise RuntimeError(
                f"nvidia-smi pmon returned GPU {gpu} for requested GPU {expected_gpu}"
            )
        process_type = fields[2]
        if fields[1] == "-" and process_type == "-":
            continue
        try:
            pid = int(fields[1])
        except ValueError as exc:
            raise RuntimeError(
                f"unexpected nvidia-smi pmon row: {line!r}"
            ) from exc
        if "C" in process_type and "G" in process_type:
            rows.append(
                {
                    "gpu": gpu,
                    "pid": pid,
                    "type": process_type,
                    "command": fields[-1] if len(fields) >= 10 else None,
                }
            )
    return rows


def _gpu_graphics_processes(index: int) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["nvidia-smi", "pmon", "-i", str(index), "-c", "1"],
        check=True,
        text=True,
        capture_output=True,
    )
    return _parse_pmon_graphics_processes(
        result.stdout, expected_gpu=index
    )


def _allocate_gpus(manifest: Mapping[str, Any], stage: str) -> list[int]:
    policy = _require_mapping(manifest["gpu_policy"], "gpu_policy")
    threshold_key = "evaluation" if stage in ("screening", "heldout", "optional_eval") else stage
    minimum = int(_require_mapping(policy["min_free_mib"], "gpu_policy.min_free_mib")[threshold_key])
    allowed = set(int(value) for value in policy["allowed_gpu_ids"])
    allow_existing = bool(policy.get("allow_existing_compute_processes", False))
    rollout_stage = stage in (
        "collection",
        "screening",
        "heldout",
        "optional_eval",
    )
    graphics_exclusive = bool(
        policy.get("require_graphics_exclusive_for_rollout", True)
    )
    available = [
        row["index"]
        for row in _gpu_rows()
        if row["index"] in allowed
        and row["free_mib"] >= minimum
        and (allow_existing or not _gpu_has_process(row["index"]))
        and (
            not rollout_stage
            or not graphics_exclusive
            or not _gpu_graphics_processes(row["index"])
        )
    ]
    required = int(policy["stable_gpu_count"])
    maximum = int(policy["burst_max_gpu_count"])
    if stage == "update":
        required = int(policy["train_world_size"])
        maximum = required
    elif stage == "collection":
        maximum = min(4, maximum)
    if len(available) < required:
        raise RuntimeError(
            f"{stage} requires {required} eligible GPUs with >= {minimum} MiB"
            f" and rollout graphics exclusivity={graphics_exclusive};"
            f" found {available}"
        )
    return available[:maximum]


def _runtime_env(manifest: Mapping[str, Any], gpus: Sequence[int]) -> dict[str, str]:
    workspace = str(manifest["workspace"])
    project_root = str(manifest["project_root"])
    robotwin_root = f"{project_root}/third_party/RoboTwin-lingbot-native"
    env = dict(os.environ)
    env.update(
        ROBOTWIN_ROOT=robotwin_root,
        PYTHONPATH=":".join(
            (
                workspace,
                f"{project_root}/src",
                project_root,
                f"{project_root}/third_party/lingbot-va",
                robotwin_root,
            )
        ),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=",".join(str(value) for value in gpus),
        PYTHONUNBUFFERED="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        TF_CPP_MIN_LOG_LEVEL="2",
    )
    return env


def _run_logged(argv: Sequence[str], *, log_path: Path, env: Mapping[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.run(list(argv), check=True, stdout=handle, stderr=subprocess.STDOUT, env=dict(env))


def _run_stage(manifest: Mapping[str, Any], stage: str, *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        policy = _require_mapping(manifest["gpu_policy"], "gpu_policy")
        allowed = [int(value) for value in policy["allowed_gpu_ids"]]
        count = (
            int(policy["train_world_size"])
            if stage == "update"
            else min(4, int(policy["burst_max_gpu_count"]))
            if stage == "collection"
            else int(policy["burst_max_gpu_count"])
        )
        gpus = allowed[:count]
    else:
        gpus = _allocate_gpus(manifest, stage)
    workspace = Path(str(manifest["workspace"]))
    python_bin = str(manifest["python_bin"])
    formal_root = Path(str(manifest["formal_root"]))
    protocol = str(manifest["eval_protocol"])
    env = _runtime_env(manifest, gpus)
    commands: list[tuple[list[str], Path]] = []
    if stage == "collection":
        configs = [str(value) for value in manifest["collection_configs"]]
        if len(gpus) < len(configs):
            raise RuntimeError("collection v1 requires four simultaneous GPU shards")
        processes: list[tuple[subprocess.Popen[Any], IO[str], Path]] = []
        for index, config in enumerate(configs):
            shard_env = _runtime_env(manifest, [gpus[index]])
            command = [
                python_bin,
                "-u",
                str(workspace / "experiments" / "train_iterative_on_policy_flow_opd.py"),
                "--config",
                config,
            ]
            log = Path(str(_read_json(Path(config))["output_dir"])) / "collect.log"
            commands.append((command, log))
            if not dry_run:
                # Collection processes are independent; launch all before waiting.
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open("a", encoding="utf-8")
                try:
                    process = subprocess.Popen(
                        command,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        env=shard_env,
                    )
                except BaseException:
                    handle.close()
                    raise
                processes.append((process, handle, log))
        if not dry_run:
            failures: list[dict[str, Any]] = []
            for process, handle, log in processes:
                try:
                    returncode = process.wait()
                finally:
                    handle.close()
                if returncode != 0:
                    failures.append({"pid": process.pid, "returncode": returncode, "log": str(log)})
            if failures:
                raise RuntimeError(f"collection shard failures: {failures}")
    elif stage == "update":
        execution = _require_mapping(manifest["execution"], "execution")
        argv = [str(value) for value in execution["update_argv"]]
        commands.append((argv, formal_root / "logs" / "update.log"))
        if not dry_run:
            _run_logged(argv, log_path=commands[0][1], env=env)
    elif stage in ("screening", "heldout"):
        command_name = "screen" if stage == "screening" else "heldout"
        argv = [
            python_bin,
            "-u",
            "-m",
            "experiments.run_handover_mic_success_path_eval",
            command_name,
            "--protocol",
            protocol,
            "--gpus",
            ",".join(str(value) for value in gpus),
        ]
        commands.append((argv, formal_root / "eval" / "logs" / f"{stage}.log"))
        if not dry_run:
            _run_logged(argv, log_path=commands[0][1], env=env)
    elif stage == "optional_eval":
        optional = _require_mapping(manifest["optional_eval"], "optional_eval")
        argv = [str(value) for value in optional["argv"]]
        commands.append((argv, formal_root / "optional_eval" / "optional_eval.log"))
        if not dry_run:
            _run_logged(argv, log_path=commands[0][1], env=env)
    else:
        raise ValueError(f"unsupported stage: {stage}")
    return {
        "stage": stage,
        "gpus": gpus,
        "commands": [command for command, _log in commands],
        "logs": [str(log) for _command, log in commands],
        "dry_run": bool(dry_run),
    }


def _finalize(manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    formal_root = Path(str(manifest["formal_root"]))
    training = _read_json(formal_root / "update" / "summary.json")
    selection = _read_json(formal_root / "eval" / "screen" / "selection.json")
    heldout = _read_json(formal_root / "eval" / "heldout" / "summary.json")
    optional = _require_mapping(manifest["optional_eval"], "optional_eval")
    optional_summary = None
    optional_path = formal_root / "optional_eval" / "summary.json"
    if optional_path.is_file():
        optional_summary = _read_json(optional_path)
    result = {
        "schema": FINAL_SCHEMA,
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": manifest["task"],
        "run_id": manifest["run_id"],
        "contract_receipt": str(formal_root / "pipeline" / "contract.json"),
        "training_summary": str(formal_root / "update" / "summary.json"),
        "selection_receipt": str(formal_root / "eval" / "screen" / "selection.json"),
        "heldout_summary": str(formal_root / "eval" / "heldout" / "summary.json"),
        "selected_epoch": selection.get("selected_epoch"),
        "selected_checkpoint": selection.get("selected_checkpoint"),
        "optimizer_steps": training.get("global_optimizer_step"),
        "heldout": heldout,
        "optional_eval": {
            "enabled": bool(optional["enabled"]),
            "selection_input": False,
            "summary": optional_summary,
        },
        "protocol": {
            "epochs": contract["epochs"],
            "split_counts": contract["split_counts"],
            "heldout_exact_pairs": contract["heldout_exact_pairs"],
            "gpu_policy": contract["gpu_policy"],
        },
    }
    _write_json(formal_root / "pipeline" / "final_summary.json", result)
    return result


def _status_payload(manifest_path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    phases = _phase_status(manifest)
    blocker = None
    try:
        next_stage = _next_stage(phases)
    except RuntimeError as exc:
        next_stage = None
        blocker = str(exc)
    return {
        "schema": STATE_SCHEMA,
        "status": (
            "BLOCKED"
            if blocker is not None
            else "COMPLETE"
            if next_stage is None
            else "IN_PROGRESS"
        ),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": manifest["task"],
        "run_id": manifest["run_id"],
        "manifest_sha256": contract["manifest_sha256"],
        "phases": phases,
        "next_stage": next_stage,
        "blocker": blocker,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "status", "run"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    contract = validate_contract(manifest_path)
    manifest = _read_json(manifest_path)
    formal_root = Path(str(manifest["formal_root"]))
    if args.command == "validate":
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0

    if not args.dry_run:
        _write_json(formal_root / "pipeline" / "contract.json", contract)
    state = _status_payload(manifest_path, contract)
    if args.command == "status":
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0

    if state["status"] == "BLOCKED":
        raise RuntimeError(str(state["blocker"]))

    while state["next_stage"] is not None:
        stage = str(state["next_stage"])
        launch = _run_stage(manifest, stage, dry_run=bool(args.dry_run))
        if args.dry_run:
            print(json.dumps({"contract": contract, "state": state, "launch": launch}, indent=2, sort_keys=True))
            return 0
        state = _status_payload(manifest_path, contract)
        _write_json(formal_root / "pipeline" / "state.json", state)
        if state["status"] == "BLOCKED":
            raise RuntimeError(str(state["blocker"]))
        if state["next_stage"] == stage:
            raise RuntimeError(f"{stage} exited without a complete PASS receipt")
    result = _finalize(manifest, contract)
    _write_json(formal_root / "pipeline" / "state.json", state)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
