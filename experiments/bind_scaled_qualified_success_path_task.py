"""Bind a larger qualified RoboTwin cohort to the success-path pipeline.

This binder is the scaled counterpart of ``bind_qualified_success_path_task``.
It preserves the frozen JointLoRA recipe while allowing the number of
train/calibration records and collection workers to scale independently from
the legacy 8/4/four-shard contract.  Only outcome-free seed/instruction
metadata is projected into the formal split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.bind_qualified_success_path_task import (
    CONFIG_ROOT,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    PYTHON_BIN,
    SELECTION_RULE,
    STUDENT,
    TEACHER,
    WORKSPACE,
    _json_sha256,
    _read,
    _rollout,
    _sha256,
    _training_template,
    _write,
)
from experiments.opd_task_specs import TASK_SPECS, resolve_task_chunks
from experiments.stage_h_task_progress import SUPPORTED_TASKS


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _load_metadata(
    source: Path,
    *,
    task: str,
    roles: list[str],
) -> list[dict[str, Any]]:
    payload = _read(source)
    if payload.get("status") != "PASS":
        raise ValueError("metadata source is not PASS")
    if payload.get("task") not in (None, task):
        raise ValueError("metadata source task mismatch")
    if payload.get("task_config") != "demo_clean":
        raise ValueError("metadata source task_config mismatch")
    if payload.get("instruction_type") != "seen":
        raise ValueError("metadata source instruction_type mismatch")
    raw = payload.get("episode_records")
    if not isinstance(raw, list) or len(raw) != len(roles):
        raise ValueError(
            f"metadata source must contain exactly {len(roles)} episode_records"
        )
    accepted = payload.get("accepted_seed_list")
    seeds = [int(row["seed"]) for row in raw]
    if accepted != seeds or len(set(seeds)) != len(seeds):
        raise ValueError("metadata source accepted seeds are not unique and ordered")
    records: list[dict[str, Any]] = []
    for episode, (raw_row, role) in enumerate(zip(raw, roles, strict=True)):
        instruction = raw_row.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"metadata record {episode} lacks an instruction")
        records.append(
            {
                "episode": episode,
                "seed": int(raw_row["seed"]),
                "instruction": instruction.strip(),
                "role": role,
            }
        )
    return records


def _indices(
    train: int, calibration: int, screening: int, heldout: int
) -> dict[str, list[int]]:
    train_end = train
    calibration_end = train_end + calibration
    screening_end = calibration_end + screening
    heldout_end = screening_end + heldout
    return {
        "train": [0, train_end],
        "calibration": [train_end, calibration_end],
        "screening": [calibration_end, screening_end],
        "heldout": [screening_end, heldout_end],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--metadata-source", type=Path, required=True)
    parser.add_argument("--qualification-decision", type=Path, required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--noise-banks", type=int, nargs=4, required=True)
    parser.add_argument("--train-count", type=_positive, default=24)
    parser.add_argument("--calibration-count", type=_positive, default=12)
    parser.add_argument("--screening-count", type=_positive, default=4)
    parser.add_argument("--heldout-count", type=_positive, default=6)
    parser.add_argument("--collection-workers-per-gpu", type=_positive, default=2)
    parser.add_argument("--run-date", default="20260825")
    args = parser.parse_args()

    task = str(args.task)
    if task not in SUPPORTED_TASKS or task not in TASK_SPECS:
        raise ValueError(f"task lacks formal progress/horizon support: {task}")
    gpu_ids = [int(value) for value in str(args.gpu_ids).split(",")]
    if len(gpu_ids) not in (2, 4) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpu-ids must contain two or four unique GPU ids")
    if any(value not in range(8) for value in gpu_ids):
        raise ValueError("GPU ids must lie in [0,7]")
    if len(set(args.noise_banks)) != 4:
        raise ValueError("screening and heldout noise banks must be distinct")

    counts = {
        "train": args.train_count,
        "calibration": args.calibration_count,
        "screening": args.screening_count,
        "heldout": args.heldout_count,
    }
    split_indices = _indices(**counts)
    roles = [
        role
        for role in ("train", "calibration", "screening", "heldout")
        for _ in range(counts[role])
    ]
    metadata_source = args.metadata_source.expanduser().resolve()
    decision = args.qualification_decision.expanduser().resolve()
    if not metadata_source.is_file() or not decision.is_file():
        raise FileNotFoundError("metadata source or qualification decision is missing")
    decision_payload = _read(decision)
    if decision_payload.get("status") != "PASS" or decision_payload.get("task_id") != task:
        raise ValueError("qualification decision is not a matching PASS")
    records = _load_metadata(metadata_source, task=task, roles=roles)

    date = str(args.run_date)
    formal_root = OUTPUT_ROOT / f"{task}_success_path_v1_scaled_formal_{date}"
    collection_root = OUTPUT_ROOT / f"{task}_coherent_tt_scaled_formal_round0_{date}"
    canary_root = OUTPUT_ROOT / f"{task}_dual_lane_collection_canary_{date}"
    if formal_root.exists() or collection_root.exists() or canary_root.exists():
        raise FileExistsError("refusing to bind over an existing output root")

    chunks = resolve_task_chunks(task)
    max_control_steps = TASK_SPECS[task].max_control_steps
    collection_count = counts["train"] + counts["calibration"]
    shard_count = len(gpu_ids) * args.collection_workers_per_gpu
    stem = f"{task}_success_path_v1_scaled_formal"
    split_path = CONFIG_ROOT / f"{stem}_split_{date}.json"
    training_path = CONFIG_ROOT / (
        f"{stem}_{counts['train']}train{counts['calibration']}calib_{date}.json"
    )
    protocol_path = CONFIG_ROOT / f"{task}_success_path_v1_scaled_eval_protocol_{date}.json"
    collection_paths = [
        CONFIG_ROOT / f"{task}_coherent_tt_scaled_collect_shard_{shard:02d}_{date}.json"
        for shard in range(shard_count)
    ]
    canary_paths = [
        CONFIG_ROOT / f"{task}_dual_lane_canary_{lane:02d}_{date}.json"
        for lane in range(shard_count)
    ]
    manifest_path = CONFIG_ROOT / f"{task}_scaled_qualified_pipeline_v1_{date}.json"
    for path in [
        split_path,
        training_path,
        protocol_path,
        manifest_path,
        *collection_paths,
        *canary_paths,
    ]:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite config: {path}")

    projection = [
        {key: row[key] for key in ("episode", "seed", "instruction")}
        for row in records
    ]
    projection_sha = _json_sha256(projection)
    split = {
        "schema": f"waopd_{task}_scaled_success_path_formal_split_v1",
        "task": task,
        "task_config": "demo_clean",
        "instruction_type": "seen",
        "binding_status": "BOUND",
        "selection_rule": "expert_feasible_metadata_positional_split",
        "source_metadata": str(metadata_source),
        "source_metadata_sha256": _sha256(metadata_source),
        "split_counts": counts,
        "split_indices": split_indices,
        "ordered_metadata_projection_sha256": projection_sha,
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
        collection_group_id=f"{task}_coherent_tt_scaled_formal_round0_{date}",
        source_binding_status="BOUND",
        split_manifest=str(split_path),
        split_manifest_sha256=split_sha,
        split_counts=counts,
        split_indices=split_indices,
        ordered_metadata_projection_sha256=projection_sha,
        rollouts=[_rollout(row) for row in records[:collection_count]],
        trajectory_artifacts=[
            str(
                collection_root
                / f"collect_shard_{rollout_id % shard_count:02d}"
                / f"{task}_round_00_rollout_{rollout_id // shard_count:02d}.pt"
            )
            for rollout_id in range(collection_count)
        ],
    )
    _write(training_path, training)

    for shard, path in enumerate(collection_paths):
        config = dict(training)
        config.pop("trajectory_artifacts")
        config.update(
            run_mode="collect",
            inner_epochs=1,
            rollouts=[
                _rollout(records[index])
                for index in range(shard, collection_count, shard_count)
            ],
            output_dir=str(collection_root / f"collect_shard_{shard:02d}"),
        )
        _write(path, config)

    for lane, path in enumerate(canary_paths):
        config = dict(training)
        config.pop("trajectory_artifacts")
        config.update(
            run_mode="collect",
            inner_epochs=1,
            collection_group_id=f"{task}_dual_lane_canary_{date}",
            rollouts=[_rollout(records[lane])],
            output_dir=str(canary_root / f"lane_{lane:02d}"),
        )
        _write(path, config)

    screen_start, screen_end = split_indices["screening"]
    heldout_start, heldout_end = split_indices["heldout"]
    screening = [
        {"seed": row["seed"], "instruction": row["instruction"]}
        for row in records[screen_start:screen_end]
    ]
    heldout = [
        {"seed": row["seed"], "instruction": row["instruction"]}
        for row in records[heldout_start:heldout_end]
    ]
    screen_banks = list(args.noise_banks[:2])
    heldout_banks = list(args.noise_banks[2:])
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
        "split_counts": counts,
        "split_indices": split_indices,
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
        "schema": "waopd_scaled_qualified_success_path_pipeline_v1",
        "task": task,
        "run_id": f"{task.replace('_', '-')}-scaled-success-path-v1-{date}",
        "workspace": str(WORKSPACE),
        "project_root": str(PROJECT_ROOT),
        "python_bin": str(PYTHON_BIN),
        "formal_root": str(formal_root),
        "collection_root": str(collection_root),
        "canary_root": str(canary_root),
        "qualification_decision": str(decision),
        "split_manifest": str(split_path),
        "training_config": str(training_path),
        "collection_configs": [str(path) for path in collection_paths],
        "canary_configs": [str(path) for path in canary_paths],
        "eval_protocol": str(protocol_path),
        "split_counts": counts,
        "gpu_policy": {
            "stable_gpu_count": len(gpu_ids),
            "burst_max_gpu_count": len(gpu_ids),
            "train_world_size": 1,
            "collection_workers_per_gpu": args.collection_workers_per_gpu,
            "allowed_gpu_ids": gpu_ids,
            "min_free_mib": {
                "collection": 70000,
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
    }
    _write(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "BOUND",
                "task": task,
                "records": len(records),
                "split_counts": counts,
                "chunks": chunks,
                "collection_shards": shard_count,
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
