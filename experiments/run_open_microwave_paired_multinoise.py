"""Run exact Released-vs-OPD task pairs over several locked noise banks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parse_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(set(result)) != len(result):
        raise ValueError(f"expected non-empty unique integer list, got {value!r}")
    return result


def _episode_is_complete(
    path: Path,
    *,
    task: str,
    task_config: str,
    seed: int,
    prompt: str,
    chunks: int,
    max_control_steps: int | None,
    noise_base_seed: int,
    student: Path,
    adapter: Path | None,
) -> bool:
    if not path.is_file():
        return False
    try:
        row = _read_json(path)
    except (OSError, ValueError, TypeError):
        return False
    expected = {
        "status": "PASS",
        "task": task,
        "task_config": task_config,
        "seed": int(seed),
        "prompt": prompt,
        "chunks": int(chunks),
        "max_control_steps": max_control_steps,
        "student_checkpoint": str(student.expanduser().resolve()),
        "adapter_state": (
            None if adapter is None else str(adapter.expanduser().resolve())
        ),
    }
    if any(row.get(key) != value for key, value in expected.items()):
        return False
    return (
        row.get("noise_contract", {}).get("student_base_seed")
        == int(noise_base_seed)
    )


def _run_policy(
    *,
    args: argparse.Namespace,
    gpu: int,
    seed: int,
    prompt: str,
    noise_base_seed: int,
    label: str,
    adapter: Path | None,
    output: Path,
) -> None:
    if _episode_is_complete(
        output,
        task=str(args.task),
        task_config=str(args.task_config),
        seed=seed,
        prompt=prompt,
        chunks=int(args.chunks),
        max_control_steps=args.max_control_steps,
        noise_base_seed=noise_base_seed,
        student=args.student,
        adapter=adapter,
    ):
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python_bin),
        "-u",
        "-m",
        "experiments.waopd_v0j_teacher_free_behavior",
        "--task",
        str(args.task),
        "--task-config",
        str(args.task_config),
        "--seed",
        str(seed),
        "--chunks",
        str(args.chunks),
        "--noise-base-seed",
        str(noise_base_seed),
        "--prompt",
        prompt,
        "--student",
        str(args.student),
        "--project-root",
        str(args.project_root),
        "--device",
        "cuda:0",
        "--enable-offload",
        "--official-offload-parity",
        "--output",
        str(output),
    ]
    if args.max_control_steps is not None:
        command.extend(["--max-control-steps", str(args.max_control_steps)])
    if adapter is not None:
        command.extend(["--adapter-state", str(adapter), "--adapter-kind", "joint_lora"])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with (output.parent / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=args.workspace,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not _episode_is_complete(
        output,
        task=str(args.task),
        task_config=str(args.task_config),
        seed=seed,
        prompt=prompt,
        chunks=int(args.chunks),
        max_control_steps=args.max_control_steps,
        noise_base_seed=noise_base_seed,
        student=args.student,
        adapter=adapter,
    ):
        raise RuntimeError(
            f"{label} failed for seed={seed} bank={noise_base_seed} gpu={gpu}; "
            f"see {output.parent / 'run.log'}"
        )


def _validate_pair(
    *,
    released_path: Path,
    opd_path: Path,
    seed: int,
    prompt: str,
    noise_base_seed: int,
    task: str,
    task_config: str,
    chunks: int,
    max_control_steps: int | None,
    student: Path,
    adapter: Path,
) -> dict[str, Any]:
    released = _read_json(released_path)
    opd = _read_json(opd_path)
    for label, row in (("released", released), ("opd", opd)):
        expected = {
            "status": "PASS",
            "task": task,
            "task_config": task_config,
            "seed": seed,
            "prompt": prompt,
            "chunks": int(chunks),
            "max_control_steps": max_control_steps,
            "student_checkpoint": str(student.expanduser().resolve()),
            "runtime_nfe": {"video": 1, "action": 1},
            "teacher_loaded": False,
            "teacher_called": False,
            "training_started": False,
        }
        for key, expected_value in expected.items():
            if row.get(key) != expected_value:
                raise RuntimeError(
                    f"{label} {key} mismatch for seed={seed} bank={noise_base_seed}: "
                    f"{row.get(key)!r} != {expected_value!r}"
                )
        if row.get("noise_contract", {}).get("student_base_seed") != noise_base_seed:
            raise RuntimeError(f"{label} noise bank metadata mismatch")
    if (
        released.get("adapter_state") is not None
        or opd.get("adapter_state") != str(adapter.expanduser().resolve())
    ):
        raise RuntimeError("Released/OPD adapter assignment is inverted or missing")
    for key in ("prompt_hash", "initial_snapshot_sha256", "student_checkpoint_sha256"):
        if released.get(key) != opd.get(key):
            raise RuntimeError(f"pair differs at {key}: seed={seed} bank={noise_base_seed}")
    released_noise = released["noise_contract"]
    opd_noise = opd["noise_contract"]
    common_chunks = min(
        len(released_noise["chunk_action_noise_hashes"]),
        len(opd_noise["chunk_action_noise_hashes"]),
    )
    if common_chunks < 1:
        raise RuntimeError("pair has no common policy-noise prefix")
    for key in ("chunk_action_noise_hashes", "chunk_video_noise_hashes"):
        if released_noise[key][:common_chunks] != opd_noise[key][:common_chunks]:
            raise RuntimeError(
                f"pair differs at {key} on common prefix: seed={seed} bank={noise_base_seed}"
            )
    released_success = bool(released["success"])
    opd_success = bool(opd["success"])
    return {
        "schema": "waopd_released_opd_pair_v1",
        "status": "PASS",
        "task": task,
        "task_config": task_config,
        "seed": seed,
        "prompt": prompt,
        "noise_base_seed": noise_base_seed,
        "common_noise_chunks": common_chunks,
        "released_success": released_success,
        "opd_success": opd_success,
        "rescue": (not released_success) and opd_success,
        "regression": released_success and (not opd_success),
        "released_episode": str(released_path),
        "opd_episode": str(opd_path),
    }


def _run_gpu_queue(
    *, args: argparse.Namespace, gpu: int, units: list[tuple[int, int, str]]
) -> list[dict[str, Any]]:
    rows = []
    for noise_base_seed, seed, prompt in units:
        unit_root = args.output_root / f"bank_{noise_base_seed}" / f"seed_{seed}"
        released_path = unit_root / "released" / "episode.json"
        opd_path = unit_root / "opd" / "episode.json"
        _run_policy(
            args=args,
            gpu=gpu,
            seed=seed,
            prompt=prompt,
            noise_base_seed=noise_base_seed,
            label="released",
            adapter=None,
            output=released_path,
        )
        _run_policy(
            args=args,
            gpu=gpu,
            seed=seed,
            prompt=prompt,
            noise_base_seed=noise_base_seed,
            label="opd",
            adapter=args.adapter,
            output=opd_path,
        )
        row = _validate_pair(
            released_path=released_path,
            opd_path=opd_path,
            seed=seed,
            prompt=prompt,
            noise_base_seed=noise_base_seed,
            task=str(args.task),
            task_config=str(args.task_config),
            chunks=int(args.chunks),
            max_control_steps=args.max_control_steps,
            student=args.student,
            adapter=args.adapter,
        )
        row["gpu"] = gpu
        _write_json(unit_root / "pair.json", row)
        rows.append(row)
    return rows


def _summarize(
    rows: list[dict[str, Any]], *, task: str, task_config: str
) -> dict[str, Any]:
    by_bank: dict[str, dict[str, int]] = {}
    for row in rows:
        key = str(row["noise_base_seed"])
        bucket = by_bank.setdefault(
            key,
            {"pairs": 0, "released_success": 0, "opd_success": 0, "rescues": 0, "regressions": 0},
        )
        bucket["pairs"] += 1
        bucket["released_success"] += int(row["released_success"])
        bucket["opd_success"] += int(row["opd_success"])
        bucket["rescues"] += int(row["rescue"])
        bucket["regressions"] += int(row["regression"])
    return {
        "schema": "waopd_paired_multinoise_summary_v1",
        "status": "PASS",
        "task": task,
        "task_config": task_config,
        "pairs": len(rows),
        "released_success": sum(int(row["released_success"]) for row in rows),
        "opd_success": sum(int(row["opd_success"]) for row in rows),
        "rescues": sum(int(row["rescue"]) for row in rows),
        "regressions": sum(int(row["regression"]) for row in rows),
        "paired_net_gain": sum(int(row["rescue"]) - int(row["regression"]) for row in rows),
        "by_bank": by_bank,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="open_microwave")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--chunks", type=int, default=48)
    parser.add_argument("--max-control-steps", type=int, default=1500)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--gpus", default="2,3,4,5,6,7")
    parser.add_argument("--noise-base-seeds", default="2026080401,2026081801,2026081802")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = _read_json(args.manifest.expanduser().resolve())
    if args.chunks < 1:
        raise ValueError("--chunks must be positive")
    if args.max_control_steps is not None and args.max_control_steps < 1:
        raise ValueError("--max-control-steps must be positive")
    manifest_task = manifest.get("task")
    if manifest_task is not None and manifest_task != args.task:
        raise ValueError(
            f"manifest task differs from --task: {manifest_task!r} != {args.task!r}"
        )
    if (
        manifest.get("task_config") != args.task_config
        or manifest.get("instruction_type") != "seen"
    ):
        raise ValueError(
            "manifest task_config/instruction_type differs from requested "
            f"{args.task_config!r}/'seen'"
        )
    records = manifest.get("episode_records")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest has no episode_records")
    accepted = [int(item) for item in manifest.get("accepted_seed_list", [])]
    record_seeds = [int(item["seed"]) for item in records]
    if record_seeds != accepted:
        raise ValueError("episode_records order differs from accepted_seed_list")

    gpus = _parse_ints(args.gpus)
    noise_base_seeds = _parse_ints(args.noise_base_seeds)
    units = [
        (bank, int(record["seed"]), str(record["instruction"]))
        for bank in noise_base_seeds
        for record in records
    ]
    if args.max_pairs is not None:
        if args.max_pairs < 1:
            raise ValueError("--max-pairs must be positive")
        units = units[: args.max_pairs]
    run_config = {
        "schema": "waopd_paired_multinoise_run_v1",
        "task": str(args.task),
        "task_config": str(args.task_config),
        "chunks": int(args.chunks),
        "max_control_steps": (
            None
            if args.max_control_steps is None
            else int(args.max_control_steps)
        ),
        "manifest": str(args.manifest.expanduser().resolve()),
        "student": str(args.student.expanduser().resolve()),
        "adapter": str(args.adapter.expanduser().resolve()),
        "gpus": gpus,
        "noise_base_seeds": noise_base_seeds,
        "pair_units": len(units),
    }
    if args.dry_run:
        print(json.dumps(run_config, indent=2, sort_keys=True))
        return 0
    args.output_root = args.output_root.expanduser().resolve()
    _write_json(args.output_root / "run_config.json", run_config)
    queues = [[] for _ in gpus]
    for index, unit in enumerate(units):
        queues[index % len(gpus)].append(unit)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(_run_gpu_queue, args=args, gpu=gpu, units=queue)
            for gpu, queue in zip(gpus, queues, strict=True)
            if queue
        ]
        for future in futures:
            rows.extend(future.result())
    rows.sort(key=lambda row: (int(row["noise_base_seed"]), int(row["seed"])))
    summary = _summarize(
        rows, task=str(args.task), task_config=str(args.task_config)
    )
    summary["run_config"] = run_config
    _write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
