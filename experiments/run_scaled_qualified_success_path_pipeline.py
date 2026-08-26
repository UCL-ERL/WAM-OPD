"""Run a scaled qualified success-path pipeline with bounded GPU concurrency.

The scientific protocol is fixed by the bound manifest.  Collection shard
count is independent from physical GPU count: a canary chooses one or two
simultaneous workers per GPU, while formal artifacts and split membership are
unchanged.  Completed PASS phases are resumable; partial formal phases are
never retried automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO, Mapping, Sequence

from experiments import run_qualified_success_path_pipeline as base


PIPELINE_SCHEMA = "waopd_scaled_qualified_success_path_pipeline_v1"
CONTRACT_SCHEMA = "waopd_scaled_qualified_success_path_contract_v1"
STATE_SCHEMA = "waopd_scaled_qualified_success_path_state_v1"
CORE_STAGES = ("collection", "update", "screening", "heldout")


def _expected_roles(counts: Mapping[str, int]) -> list[str]:
    return [
        role
        for role in ("train", "calibration", "screening", "heldout")
        for _ in range(int(counts[role]))
    ]


def _split_bounds(counts: Mapping[str, int]) -> dict[str, list[int]]:
    cursor = 0
    result: dict[str, list[int]] = {}
    for role in ("train", "calibration", "screening", "heldout"):
        start = cursor
        cursor += int(counts[role])
        result[role] = [start, cursor]
    return result


def validate_contract(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = base._read_json(manifest_path)
    base._equal(manifest.get("schema"), PIPELINE_SCHEMA, "manifest.schema")
    task = str(manifest.get("task", ""))
    if not task:
        raise ValueError("manifest.task is empty")
    decision = base._ssd_path(
        manifest.get("qualification_decision"),
        "manifest.qualification_decision",
        must_exist=True,
    )
    base._validate_gate(decision, task)

    counts = dict(base._require_mapping(manifest.get("split_counts"), "split_counts"))
    if set(counts) != {"train", "calibration", "screening", "heldout"}:
        raise ValueError("split_counts fields mismatch")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts.values()):
        raise ValueError("all split counts must be positive integers")
    split_indices = _split_bounds(counts)
    split_path = base._path(manifest.get("split_manifest"), "split_manifest")
    split = base._read_json(split_path)
    base._equal(split.get("task"), task, "split.task")
    base._equal(split.get("binding_status"), "BOUND", "split.binding_status")
    base._equal(split.get("split_counts"), counts, "split.split_counts")
    base._equal(split.get("split_indices"), split_indices, "split.split_indices")
    records = base._require_list(split.get("episode_records"), "split.episode_records")
    roles = _expected_roles(counts)
    if len(records) != len(roles):
        raise ValueError("split record count mismatch")
    seeds: list[int] = []
    normalized_records: list[dict[str, Any]] = []
    for index, (raw, role) in enumerate(zip(records, roles, strict=True)):
        row = dict(base._require_mapping(raw, f"record[{index}]"))
        base._equal(set(row), {"episode", "seed", "instruction", "role"}, f"record[{index}].fields")
        base._equal(row["episode"], index, f"record[{index}].episode")
        base._equal(row["role"], role, f"record[{index}].role")
        if not isinstance(row["seed"], int) or not isinstance(row["instruction"], str) or not row["instruction"].strip():
            raise ValueError(f"record[{index}] seed/instruction invalid")
        if any("success" in str(key).lower() for key in row):
            raise ValueError(f"record[{index}] leaks outcome data")
        seeds.append(int(row["seed"]))
        normalized_records.append(row)
    if len(set(seeds)) != len(seeds):
        raise ValueError("split seeds must be unique")
    base._equal(split.get("accepted_seed_list"), seeds, "split.accepted_seed_list")

    formal_root = base._ssd_path(manifest.get("formal_root"), "formal_root")
    collection_root = base._ssd_path(manifest.get("collection_root"), "collection_root")
    training_path = base._path(manifest.get("training_config"), "training_config")
    training = base._read_json(training_path)
    collection_count = counts["train"] + counts["calibration"]
    expected_rollouts = [base._rollout(row) for row in normalized_records[:collection_count]]
    base._equal(training.get("rollouts"), expected_rollouts, "training.rollouts")
    base._equal(training.get("split_counts"), counts, "training.split_counts")
    base._equal(training.get("split_indices"), split_indices, "training.split_indices")
    base._equal(training.get("inner_epochs"), 3, "training.inner_epochs")
    base._equal(training.get("run_mode"), "trajectory_update", "training.run_mode")
    base._equal(Path(str(training.get("output_dir"))).resolve(), formal_root / "update", "training.output_dir")
    base._validate_recipe(
        training,
        task=task,
        task_config="demo_clean",
        chunks=int(training["chunks"]),
        project_root=Path(str(manifest["project_root"])),
        label="training",
    )

    configs = [
        base._path(value, f"collection_configs[{index}]")
        for index, value in enumerate(base._require_list(manifest.get("collection_configs"), "collection_configs"))
    ]
    policy = dict(base._require_mapping(manifest.get("gpu_policy"), "gpu_policy"))
    stable_gpu_count = int(policy.get("stable_gpu_count", 0))
    if stable_gpu_count not in (2, 4):
        raise ValueError("gpu_policy.stable_gpu_count must be two or four")
    if len(configs) < stable_gpu_count or len(configs) % stable_gpu_count:
        raise ValueError(
            "collection config count must be a positive multiple of stable_gpu_count"
        )
    artifacts = base._require_list(training.get("trajectory_artifacts"), "trajectory_artifacts")
    if len(artifacts) != collection_count:
        raise ValueError("trajectory artifact count mismatch")
    collected: list[dict[str, Any]] = []
    for shard, path in enumerate(configs):
        config = base._read_json(path)
        expected = [
            base._rollout(normalized_records[index])
            for index in range(shard, collection_count, len(configs))
        ]
        base._equal(config.get("rollouts"), expected, f"collection[{shard}].rollouts")
        base._equal(config.get("run_mode"), "collect", f"collection[{shard}].run_mode")
        output = Path(str(config.get("output_dir"))).resolve()
        base._equal(output, collection_root / f"collect_shard_{shard:02d}", f"collection[{shard}].output_dir")
        collected.extend(expected)
    base._equal(sorted(collected, key=lambda row: row["rollout_id"]), expected_rollouts, "collection union")

    protocol_path = base._path(manifest.get("eval_protocol"), "eval_protocol")
    protocol = base._read_json(protocol_path)
    base._equal(protocol.get("task"), task, "protocol.task")
    base._equal(protocol.get("split_counts"), counts, "protocol.split_counts")
    base._equal(protocol.get("split_indices"), split_indices, "protocol.split_indices")
    base._equal(protocol.get("epoch_ids"), [1, 2, 3], "protocol.epoch_ids")
    base._equal(protocol.get("selection_rule"), base.SELECTION_RULE, "protocol.selection_rule")
    screen_start, screen_end = split_indices["screening"]
    heldout_start, heldout_end = split_indices["heldout"]
    expected_screen = [
        {"seed": row["seed"], "instruction": row["instruction"]}
        for row in normalized_records[screen_start:screen_end]
    ]
    expected_heldout = [
        {"seed": row["seed"], "instruction": row["instruction"]}
        for row in normalized_records[heldout_start:heldout_end]
    ]
    screening = base._require_mapping(protocol.get("screening"), "protocol.screening")
    heldout = base._require_mapping(protocol.get("heldout"), "protocol.heldout")
    base._equal(screening.get("episode_records"), expected_screen, "screening.episode_records")
    base._equal(heldout.get("episode_records"), expected_heldout, "heldout.episode_records")
    screen_banks = base._require_list(screening.get("noise_base_seeds"), "screening.noise_base_seeds")
    heldout_banks = base._require_list(heldout.get("noise_base_seeds"), "heldout.noise_base_seeds")
    if len(screen_banks) != 2 or len(heldout_banks) != 2 or set(screen_banks) & set(heldout_banks):
        raise ValueError("screening/heldout banks must be two-by-two and disjoint")

    base._equal(policy.get("train_world_size"), 1, "gpu_policy.train_world_size")
    allowed = base._require_list(policy.get("allowed_gpu_ids"), "gpu_policy.allowed_gpu_ids")
    if len(allowed) != stable_gpu_count or len(set(allowed)) != stable_gpu_count:
        raise ValueError("allowed_gpu_ids must match stable_gpu_count and be unique")
    workers = int(policy.get("collection_workers_per_gpu", 0))
    if workers not in (1, 2):
        raise ValueError("collection_workers_per_gpu must be one or two")
    canary_configs = base._require_list(manifest.get("canary_configs"), "canary_configs")
    if len(canary_configs) != stable_gpu_count * workers:
        raise ValueError(
            "canary config count must equal stable_gpu_count times workers per GPU"
        )

    return {
        "schema": CONTRACT_SCHEMA,
        "status": "PASS",
        "manifest": str(manifest_path),
        "manifest_sha256": base._sha256(manifest_path),
        "task": task,
        "split_counts": counts,
        "chunks": int(training["chunks"]),
        "collection_shards": len(configs),
        "configured_collection_workers_per_gpu": workers,
        "heldout_exact_pairs": len(expected_heldout) * len(heldout_banks),
    }


def _summary_pass(path: Path) -> bool:
    return path.is_file() and base._read_json(path).get("status") == "PASS"


def _phase_status(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    formal_root = Path(str(manifest["formal_root"]))
    collection_summaries = [
        Path(str(base._read_json(Path(value))["output_dir"])) / "summary.json"
        for value in manifest["collection_configs"]
    ]
    checkpoints = [
        formal_root / "update/epoch_checkpoints" / f"checkpoint_epoch_{epoch:02d}.pt"
        for epoch in (1, 2)
    ] + [formal_root / "update/checkpoint_trajectory_update.pt"]
    update_summary = formal_root / "update/summary.json"
    screening = formal_root / "eval/screen/selection.json"
    heldout = formal_root / "eval/heldout/summary.json"
    return {
        "collection": {
            "done": all(_summary_pass(path) for path in collection_summaries),
            "started": any(path.exists() or (path.parent / "collect.log").exists() for path in collection_summaries),
            "receipts": [str(path) for path in collection_summaries],
        },
        "update": {
            "done": _summary_pass(update_summary) and all(path.is_file() for path in checkpoints),
            "started": update_summary.exists() or (formal_root / "logs/update.log").exists(),
            "summary": str(update_summary),
        },
        "screening": {
            "done": _summary_pass(screening),
            "started": screening.exists() or (formal_root / "eval/logs/screening.log").exists(),
            "selection": str(screening),
        },
        "heldout": {
            "done": _summary_pass(heldout),
            "started": heldout.exists() or (formal_root / "eval/logs/heldout.log").exists(),
            "summary": str(heldout),
        },
    }


def _eligible_gpus(manifest: Mapping[str, Any], stage: str) -> list[int]:
    policy = base._require_mapping(manifest["gpu_policy"], "gpu_policy")
    threshold_key = "evaluation" if stage in ("screening", "heldout") else stage
    minimum = int(base._require_mapping(policy["min_free_mib"], "min_free_mib")[threshold_key])
    allowed = [int(value) for value in policy["allowed_gpu_ids"]]
    rows = {row["index"]: row["free_mib"] for row in base._gpu_rows()}
    available = [gpu for gpu in allowed if rows.get(gpu, 0) >= minimum]
    required = 1 if stage == "update" else int(policy["stable_gpu_count"])
    if len(available) < required:
        raise RuntimeError(f"{stage} requires {required} eligible GPUs >= {minimum} MiB; found {available}")
    return available[:required]


def _launch_configs(
    manifest: Mapping[str, Any],
    configs: Sequence[str],
    *,
    workers_per_gpu: int,
) -> list[dict[str, Any]]:
    gpus = _eligible_gpus(manifest, "collection")
    capacity = len(gpus) * workers_per_gpu
    failures: list[dict[str, Any]] = []
    for batch_start in range(0, len(configs), capacity):
        batch = list(configs[batch_start : batch_start + capacity])
        processes: list[tuple[subprocess.Popen[Any], IO[str], Path]] = []
        for position, config_value in enumerate(batch):
            config_path = Path(config_value)
            config = base._read_json(config_path)
            gpu = gpus[position % len(gpus)]
            env = base._runtime_env(manifest, [gpu])
            command = [
                str(manifest["python_bin"]),
                "-u",
                str(Path(str(manifest["workspace"])) / "experiments/train_iterative_on_policy_flow_opd.py"),
                "--config",
                str(config_path),
            ]
            log = Path(str(config["output_dir"])) / "collect.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a", encoding="utf-8")
            try:
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
            except BaseException:
                handle.close()
                raise
            processes.append((process, handle, log))
        for process, handle, log in processes:
            try:
                returncode = process.wait()
            finally:
                handle.close()
            if returncode != 0 or not _summary_pass(log.parent / "summary.json"):
                failures.append(
                    {"pid": process.pid, "returncode": returncode, "log": str(log)}
                )
        if failures:
            break
    return failures


def run_canary(
    manifest: Mapping[str, Any], *, workers_per_gpu: int = 2
) -> dict[str, Any]:
    if workers_per_gpu not in (1, 2):
        raise ValueError("canary workers_per_gpu must be one or two")
    canary_root = Path(str(manifest["canary_root"]))
    decision_path = canary_root / "decision.json"
    if decision_path.exists():
        raise FileExistsError(f"canary decision already exists: {decision_path}")
    failures = _launch_configs(
        manifest,
        [str(value) for value in manifest["canary_configs"]],
        workers_per_gpu=workers_per_gpu,
    )
    selected = workers_per_gpu if not failures else 1
    decision = {
        "schema": "waopd_collection_concurrency_canary_v1",
        "status": "PASS" if not failures else (
            "FALLBACK" if workers_per_gpu == 2 else "FAIL"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attempted_workers_per_gpu": workers_per_gpu,
        "selected_workers_per_gpu": selected,
        "failures": failures,
    }
    base._write_json(decision_path, decision)
    return decision


def _run_stage(manifest: Mapping[str, Any], stage: str) -> dict[str, Any]:
    formal_root = Path(str(manifest["formal_root"]))
    if stage == "collection":
        decision_path = Path(str(manifest["canary_root"])) / "decision.json"
        if not decision_path.is_file():
            raise FileNotFoundError("collection concurrency canary decision is missing")
        decision = base._read_json(decision_path)
        workers = int(decision["selected_workers_per_gpu"])
        failures = _launch_configs(
            manifest,
            [str(value) for value in manifest["collection_configs"]],
            workers_per_gpu=workers,
        )
        if failures:
            raise RuntimeError(f"formal collection failures: {failures}")
        return {"stage": stage, "workers_per_gpu": workers}

    gpus = _eligible_gpus(manifest, stage)
    env = base._runtime_env(manifest, gpus)
    if stage == "update":
        argv = [str(value) for value in manifest["execution"]["update_argv"]]
        log = formal_root / "logs/update.log"
    else:
        argv = [
            str(manifest["python_bin"]),
            "-u",
            "-m",
            "experiments.run_handover_mic_success_path_eval",
            "screen" if stage == "screening" else "heldout",
            "--protocol",
            str(manifest["eval_protocol"]),
            "--gpus",
            ",".join(str(value) for value in gpus),
        ]
        log = formal_root / f"eval/logs/{stage}.log"
    base._run_logged(argv, log_path=log, env=env)
    return {"stage": stage, "gpus": gpus, "log": str(log)}


def run_pipeline(manifest_path: Path) -> dict[str, Any]:
    contract = validate_contract(manifest_path)
    manifest = base._read_json(manifest_path)
    formal_root = Path(str(manifest["formal_root"]))
    base._write_json(formal_root / "contract_receipt.json", contract)
    while True:
        status = _phase_status(manifest)
        next_stage: str | None = None
        for stage in CORE_STAGES:
            if status[stage]["done"]:
                continue
            if status[stage]["started"]:
                raise RuntimeError(f"{stage} has partial output; refusing automatic rerun")
            next_stage = stage
            break
        state = {
            "schema": STATE_SCHEMA,
            "status": "RUNNING" if next_stage else "PASS",
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "next_stage": next_stage,
            "phases": status,
        }
        base._write_json(formal_root / "pipeline_state.json", state)
        if next_stage is None:
            return state
        _run_stage(manifest, next_stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "canary", "status", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--manifest", type=Path, required=True)
        if command == "canary":
            child.add_argument(
                "--workers-per-gpu",
                type=int,
                choices=(1, 2),
                default=2,
            )
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    if args.command == "validate":
        result = validate_contract(manifest_path)
    else:
        manifest = base._read_json(manifest_path)
        if args.command == "canary":
            validate_contract(manifest_path)
            result = run_canary(
                manifest, workers_per_gpu=args.workers_per_gpu
            )
        elif args.command == "status":
            validate_contract(manifest_path)
            result = _phase_status(manifest)
        else:
            result = run_pipeline(manifest_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
