"""Bind one qualified RoboTwin task to the frozen success-path pipeline.

The binder copies only outcome-free metadata from the canonical twenty-record
Teacher sweep.  It writes the 8/4/2/6 split, four collection shards, the
exactly-three-epoch JointLoRA config, the exact-paired evaluation protocol,
and the generic pipeline manifest.  All large outputs remain under /ssd/data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from experiments.opd_task_specs import TASK_SPECS, resolve_task_chunks
from experiments.paths import (
    CONFIG_ROOT,
    OUTPUT_ROOT,
    PYTHON_BIN,
    REPO_ROOT,
    SOURCE_SWEEP,
    STUDENT_ROOT,
    TEACHER_ROOT,
    WAVE_RL_ROOT,
)
from experiments.stage_h_task_progress import SUPPORTED_TASKS


WORKSPACE = REPO_ROOT
PROJECT_ROOT = WAVE_RL_ROOT
STUDENT = STUDENT_ROOT
TEACHER = TEACHER_ROOT
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


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rollout(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rollout_id": row["episode"],
        "seed": row["seed"],
        "prompt": row["instruction"],
        "role": row["role"],
    }


def _load_records(source: Path) -> list[dict[str, Any]]:
    payload = _read(source)
    if payload.get("task_config") != "demo_clean" or payload.get("instruction_type") != "seen":
        raise ValueError(f"source protocol mismatch: {source}")
    raw = payload.get("episode_records")
    accepted = payload.get("accepted_seed_list")
    if payload.get("total_num") != 20 or not isinstance(raw, list) or len(raw) != 20:
        raise ValueError("source must contain exactly twenty accepted episodes")
    seeds = [int(row["seed"]) for row in raw]
    if accepted != seeds or len(set(seeds)) != 20:
        raise ValueError("source accepted seeds are not twenty unique ordered records")
    records: list[dict[str, Any]] = []
    for index, (row, role) in enumerate(zip(raw, ROLES, strict=True)):
        instruction = row.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"source record {index} lacks an instruction")
        records.append(
            {
                "episode": index,
                "seed": int(row["seed"]),
                "instruction": instruction,
                "role": role,
            }
        )
    return records


def _training_template(
    *, task: str, chunks: int, collection_root: Path, formal_root: Path
) -> dict[str, Any]:
    return {
        "project_root": str(PROJECT_ROOT),
        "run_mode": "trajectory_update",
        "objective": "coherent_tt_consistency",
        "coherent_tt_variant": "success_path_v1",
        "rounds": 1,
        "collection_group_id": f"{task}_coherent_tt_formal_round0_20260825",
        "task": task,
        "task_config": "demo_clean",
        "chunks": chunks,
        "student": str(STUDENT),
        "teacher_transformer": str(TEACHER / "transformer"),
        "teacher_video_steps": 25,
        "teacher_video_exec_steps": None,
        "teacher_action_steps": 50,
        "output_dir": str(formal_root / "update"),
        "device": "cuda:0",
        "enable_offload": True,
        "official_offload_parity": True,
        "adapter_seed": 20260820,
        "adapter_kind": "joint_lora",
        "trainable_bank": "both",
        "adapter_rank": 8,
        "lora_alpha": 8.0,
        "lora_dropout": 0.0,
        "lora_block_indices": list(range(30)),
        "optimizer_kind": "adamw",
        "learning_rate": 2e-5,
        "max_grad_norm": 2.0,
        "pseudo_huber_c": 0.001,
        "video_weight": 1.0,
        "action_weight": 1.0,
        "action_fm_weight": 0.2,
        "action_velocity_weight": 0.0,
        "ema_decay": 0.995,
        "effective_batch_size": 4,
        "inner_epochs": 3,
        "consistency_video_stride": 500,
        "consistency_action_stride": 500,
        "consistency_seed": 20268739,
        "consistency_noise_source": "artifact_epsilon",
        "calibration_anchors_per_trajectory": 2,
        "loss_reduction": "mean_trajectories_mean_labels",
        "retention_weight": 0.0,
        "pre_update_solver_closure": False,
        "initial_checkpoint": None,
        "resume_optimizer_state": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--qualification-decision", type=Path, required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--noise-banks", type=int, nargs=4, required=True)
    parser.add_argument("--run-date", default="20260825")
    parser.add_argument("--collection-min-free-mib", type=int, default=65000)
    parser.add_argument("--allow-existing-compute-processes", action="store_true")
    args = parser.parse_args()

    task = str(args.task)
    if task not in SUPPORTED_TASKS or task not in TASK_SPECS:
        raise ValueError(f"task lacks formal progress/horizon support: {task}")
    gpu_ids = [int(value) for value in str(args.gpu_ids).split(",")]
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4 or any(value not in range(8) for value in gpu_ids):
        raise ValueError("--gpu-ids must contain four unique ids in [0,7]")
    screen_banks = list(args.noise_banks[:2])
    heldout_banks = list(args.noise_banks[2:])
    if len(set(args.noise_banks)) != 4:
        raise ValueError("screening and heldout noise banks must be distinct")

    decision = args.qualification_decision.resolve()
    if not decision.is_file():
        raise FileNotFoundError(decision)
    decision_payload = _read(decision)
    if decision_payload.get("status") != "PASS" or decision_payload.get("task_id") != task:
        raise ValueError("qualification decision is not a matching PASS")

    source = (
        SOURCE_SWEEP
        / task
        / "seed_0/stseed-10000/metrics"
        / task
        / "res.json"
    )
    source_task = PROJECT_ROOT / f"third_party/RoboTwin-lingbot-native/envs/{task}.py"
    for path in (source, source_task, PYTHON_BIN, STUDENT, TEACHER / "transformer"):
        if not path.exists():
            raise FileNotFoundError(path)

    records = _load_records(source)
    projection = [
        {key: row[key] for key in ("episode", "seed", "instruction")}
        for row in records
    ]
    projection_sha = _json_sha256(projection)
    chunks = resolve_task_chunks(task)
    max_control_steps = TASK_SPECS[task].max_control_steps
    date = str(args.run_date)
    formal_root = OUTPUT_ROOT / f"{task}_success_path_v1_formal_{date}"
    collection_root = OUTPUT_ROOT / f"{task}_coherent_tt_formal_round0_{date}"
    if formal_root.exists() or collection_root.exists():
        raise FileExistsError("refusing to bind over an existing formal or collection root")

    stem = f"{task}_success_path_v1_formal"
    split_path = CONFIG_ROOT / f"{stem}_split_{date}.json"
    training_path = CONFIG_ROOT / f"{stem}_8train4calib_{date}.json"
    protocol_path = CONFIG_ROOT / f"{task}_success_path_v1_eval_protocol_{date}.json"
    collection_paths = [
        CONFIG_ROOT / f"{task}_coherent_tt_formal_collect_shard_{shard:02d}_{date}.json"
        for shard in range(4)
    ]
    manifest_path = CONFIG_ROOT / f"{task}_qualified_pipeline_v1_{date}.json"
    for path in [split_path, training_path, protocol_path, manifest_path, *collection_paths]:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite config: {path}")

    split = {
        "schema": f"waopd_{task}_success_path_formal_split_v1",
        "task": task,
        "task_config": "demo_clean",
        "instruction_type": "seen",
        "binding_status": "BOUND",
        "selection_rule": "fresh_metadata_records_positional_split_8_4_2_6",
        "source_res_jsons": [str(source)],
        "source_res_json_sha256s": [_sha256(source)],
        "source_task_py": str(source_task),
        "source_task_py_sha256": _sha256(source_task),
        "ordered_metadata_projection_sha256": projection_sha,
        "split_indices": SPLITS,
        "accepted_seed_list": [row["seed"] for row in records],
        "episode_records": records,
    }
    _write(split_path, split)
    split_sha = _sha256(split_path)

    training = _training_template(
        task=task,
        chunks=chunks,
        collection_root=collection_root,
        formal_root=formal_root,
    )
    training.update(
        source_binding_status="BOUND",
        split_manifest=str(split_path),
        split_manifest_sha256=split_sha,
        split_indices=SPLITS,
        ordered_metadata_projection_sha256=projection_sha,
        rollouts=[_rollout(row) for row in records[:12]],
        trajectory_artifacts=[
            str(
                collection_root
                / f"collect_shard_{rollout_id % 4:02d}"
                / f"{task}_round_00_rollout_{rollout_id // 4:02d}.pt"
            )
            for rollout_id in range(12)
        ],
    )
    _write(training_path, training)

    for shard, path in enumerate(collection_paths):
        config = dict(training)
        config.pop("trajectory_artifacts")
        config.update(
            run_mode="collect",
            inner_epochs=1,
            rollouts=[_rollout(records[index]) for index in range(shard, 12, 4)],
            output_dir=str(collection_root / f"collect_shard_{shard:02d}"),
        )
        _write(path, config)

    screening = [
        {"seed": row["seed"], "instruction": row["instruction"]}
        for row in records[12:14]
    ]
    heldout = [
        {"seed": row["seed"], "instruction": row["instruction"]}
        for row in records[14:20]
    ]
    protocol = {
        "schema": f"waopd_{task}_success_path_eval_protocol_v1",
        "task": task,
        "task_config": "demo_clean",
        "chunks": chunks,
        "max_control_steps": max_control_steps,
        "source_binding_status": "BOUND",
        "workspace": str(WORKSPACE),
        "project_root": str(PROJECT_ROOT),
        "python_bin": str(PYTHON_BIN),
        "student": str(STUDENT),
        "training_config": str(training_path),
        "training_summary": str(formal_root / "update/summary.json"),
        "output_root": str(formal_root / "eval"),
        "split_manifest": str(split_path),
        "split_manifest_sha256": split_sha,
        "split_indices": SPLITS,
        "ordered_metadata_projection_sha256": projection_sha,
        "epoch_ids": [1, 2, 3],
        "selection_rule": SELECTION_RULE,
        "screening": {
            "noise_base_seeds": screen_banks,
            "episode_records": screening,
            "record_pairs": [
                {"noise_base_seed": screen_banks[0], "seed": screening[0]["seed"]}
            ],
        },
        "heldout": {
            "noise_base_seeds": heldout_banks,
            "episode_records": heldout,
            "record_pairs": [
                {"noise_base_seed": bank, "seed": row["seed"]}
                for bank in heldout_banks
                for row in heldout
            ],
        },
    }
    _write(protocol_path, protocol)

    manifest = {
        "schema": "waopd_qualified_success_path_pipeline_v1",
        "task": task,
        "run_id": f"{task.replace('_', '-')}-success-path-v1-{date}",
        "workspace": str(WORKSPACE),
        "project_root": str(PROJECT_ROOT),
        "python_bin": str(PYTHON_BIN),
        "formal_root": str(formal_root),
        "collection_root": str(collection_root),
        "qualification_decision": str(decision),
        "split_manifest": str(split_path),
        "training_config": str(training_path),
        "collection_configs": [str(path) for path in collection_paths],
        "eval_protocol": str(protocol_path),
        "gpu_policy": {
            "stable_gpu_count": 4,
            "burst_max_gpu_count": 4,
            "train_world_size": 1,
            "world_size_change_boundary": "epoch",
            "allow_mid_epoch_elastic_resize": False,
            "allow_existing_compute_processes": bool(
                args.allow_existing_compute_processes
            ),
            # SAPIEN/Vulkan ray-tracing camera capture can starve indefinitely
            # when another RoboTwin C+G process already owns the same GPU.
            # Compute-only overlay remains allowed for model update.
            "require_graphics_exclusive_for_rollout": True,
            "allowed_gpu_ids": gpu_ids,
            "min_free_mib": {
                "collection": args.collection_min_free_mib,
                "update": 70000,
                "evaluation": 45000,
            },
        },
        "execution": {
            "training_backend": "joint_lora_single_process_v1",
            "update_argv": [
                str(PYTHON_BIN),
                "-u",
                str(WORKSPACE / "experiments/train_iterative_on_policy_flow_opd.py"),
                "--config",
                str(training_path),
            ],
        },
        "optional_eval": {
            "enabled": False,
            "after_stage": "heldout",
            "selection_input": False,
            "argv": [],
        },
    }
    _write(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "BOUND",
                "task": task,
                "records": 20,
                "chunks": chunks,
                "gpu_ids": gpu_ids,
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
