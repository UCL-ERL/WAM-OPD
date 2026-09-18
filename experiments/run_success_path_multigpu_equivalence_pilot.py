"""Run an isolated Single-GPU or two-rank success-path equivalence pilot."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist

from experiments.success_path_distributed import SuccessPathDistributedContext
from experiments import train_iterative_on_policy_flow_opd as trainer


PILOT_SCHEMA = "waopd_success_path_multigpu_equivalence_pilot_v1"


def _stable_tensor_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _state_l2(state: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for value in state.values():
        total += float(value.detach().double().square().sum().item())
    return math.sqrt(total)


def _optimizer_moments(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    totals = {"exp_avg": 0.0, "exp_avg_sq": 0.0}
    steps: set[float] = set()
    for values in optimizer.state.values():
        for name in totals:
            value = values.get(name)
            if isinstance(value, torch.Tensor):
                totals[name] += float(value.detach().double().square().sum().item())
        step = values.get("step")
        if isinstance(step, torch.Tensor):
            steps.add(float(step.item()))
        elif step is not None:
            steps.add(float(step))
    return {
        "exp_avg_l2": math.sqrt(totals["exp_avg"]),
        "exp_avg_sq_l2": math.sqrt(totals["exp_avg_sq"]),
        "optimizer_state_steps_min": min(steps) if steps else 0.0,
        "optimizer_state_steps_max": max(steps) if steps else 0.0,
    }


def _artifact_snapshot(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "size": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
        }
        for path in paths
    ]


def _cpu_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = deepcopy(optimizer.state_dict())
    for values in state["state"].values():
        for name, value in list(values.items()):
            if isinstance(value, torch.Tensor):
                values[name] = value.detach().cpu()
    return state


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "distributed"), required=True)
    parser.add_argument("--train-trajectories", type=int, default=20)
    parser.add_argument("--calibration-trajectories", type=int, default=2)
    parser.add_argument("--labels-per-train-trajectory", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    distributed_mode = args.mode == "distributed"
    if distributed_mode:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        if (output_dir / "result.json").exists():
            raise FileExistsError(f"pilot result already exists: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.config.expanduser().resolve()
    config = trainer._normalize_config(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    if config["run_mode"] != "trajectory_update":
        raise ValueError("pilot requires a trajectory_update config")
    if config["coherent_tt_variant"] != trainer.COHERENT_TT_VARIANT_SUCCESS_PATH_V1:
        raise ValueError("pilot requires success_path_v1")

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
    os.chdir(robotwin_root)

    student = Path(config["student"]).expanduser().resolve()
    teacher = Path(config["teacher_transformer"]).expanduser().resolve()
    artifact_paths = [
        Path(value).expanduser().resolve() for value in config["trajectory_artifacts"]
    ]
    artifacts_before = _artifact_snapshot(artifact_paths)
    task_contract_hash = trainer._stable_hash(config["task_entries"])
    trajectories = trainer._load_trajectory_artifacts(
        artifact_paths,
        expected_task_entries=config["task_entries"],
        expected_student=student,
        expected_teacher=teacher,
        expected_adapter_seed=int(config["adapter_seed"]),
        expected_collection_group_id=config["collection_group_id"],
        require_coherent_collection_contract=True,
    )
    train = [item for item in trajectories if item.get("dataset_role") == "train"]
    calibration = [
        item for item in trajectories if item.get("dataset_role") == "calibration"
    ]
    if len(train) < args.train_trajectories or len(calibration) < args.calibration_trajectories:
        raise ValueError("fixed package does not contain the requested pilot subset")
    selected_train = [
        {
            **item,
            "labels": list(item["labels"])[: args.labels_per_train_trajectory],
        }
        for item in train[: args.train_trajectories]
    ]
    selected = selected_train + calibration[: args.calibration_trajectories]
    round_ids = {int(item["round_id"]) for item in selected}
    if len(round_ids) != 1:
        raise ValueError("pilot subset spans multiple rounds")

    np.random.seed(int(config["adapter_seed"]))
    torch.manual_seed(int(config["adapter_seed"]))
    torch.cuda.manual_seed_all(int(config["adapter_seed"]))
    device = torch.device("cuda", local_rank)
    runtime = trainer.NativeV0VideoRuntime(
        student_checkpoint=student,
        teacher_transformer=None,
        device=str(device),
        save_root=output_dir / f"rank_{rank:02d}" / "native_save",
        enable_offload=bool(config.get("enable_offload", True)),
        official_offload_parity=bool(config.get("official_offload_parity", True)),
        adapter_rank=int(config["adapter_rank"]),
        adapter_state=None,
        adapter_kind=str(config["adapter_kind"]),
        lora_alpha=float(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        lora_block_indices=tuple(int(value) for value in config["lora_block_indices"]),
    )
    try:
        # Initialize NCCL only after model construction.  The model loader may
        # otherwise create DTensor parameters, while the adapter identity hash
        # intentionally operates on ordinary local tensors.
        if distributed_mode:
            dist.init_process_group(backend="nccl")
            context = SuccessPathDistributedContext.from_initialized_process_group()
        else:
            context = SuccessPathDistributedContext()
        if context.rank != rank or context.world_size != world_size:
            raise RuntimeError("torchrun environment and process group disagree")
        context.barrier()
        if trainer._policy_version(runtime) != str(selected[0]["behavior_policy_version"]):
            raise RuntimeError("pilot runtime does not match fixed package policy")
        optimizer = trainer._fresh_optimizer(runtime, config)
        live_state = trainer._runtime_live_adapter_state(runtime)
        initial_state = {
            name: value.detach().clone() for name, value in live_state.items()
        }
        optimizer_trace: list[dict[str, Any]] = []

        def trace_step(
            step_id: int,
            current_optimizer: torch.optim.Optimizer,
            current_state: Mapping[str, torch.Tensor],
        ) -> None:
            torch.cuda.synchronize(device)
            delta = {
                name: current_state[name].detach() - initial_state[name]
                for name in current_state
            }
            optimizer_trace.append(
                {
                    "step_id": int(step_id),
                    "learning_rates": [
                        float(group["lr"]) for group in current_optimizer.param_groups
                    ],
                    "adapter_l2": _state_l2(current_state),
                    "adapter_delta_l2": _state_l2(delta),
                    **_optimizer_moments(current_optimizer),
                }
            )

        context.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        update = trainer._update_round_success_path_tt(
            runtime=runtime,
            optimizer=optimizer,
            trajectories=selected,
            round_id=round_ids.pop(),
            video_weight=float(config["video_weight"]),
            action_weight=float(config["action_weight"]),
            action_fm_weight=float(config["action_fm_weight"]),
            pseudo_huber_c=float(config["pseudo_huber_c"]),
            max_grad_norm=float(config["max_grad_norm"]),
            effective_batch_size=int(config["effective_batch_size"]),
            inner_epochs=1,
            consistency_seed=int(config["consistency_seed"]),
            calibration_anchors_per_trajectory=int(
                config["calibration_anchors_per_trajectory"]
            ),
            max_train_labels_per_trajectory=args.labels_per_train_trajectory,
            distributed_context=context,
            optimizer_step_callback=trace_step,
        )
        context.barrier()
        torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - started
        artifacts_after = _artifact_snapshot(artifact_paths)
        if artifacts_after != artifacts_before:
            raise RuntimeError("read-only fixed package changed during pilot")

        rank_payload = {
            "optimizer_trace": optimizer_trace,
            "final_adapter_hash": _stable_tensor_hash(live_state),
            "elapsed_seconds": float(elapsed_seconds),
        }
        if context.world_size > 1:
            rank_payloads: list[dict[str, Any] | None] = [
                None for _ in range(context.world_size)
            ]
            dist.all_gather_object(rank_payloads, rank_payload)
        else:
            rank_payloads = [rank_payload]

        if context.rank == 0:
            adapter_state = {
                name: value.detach().cpu() for name, value in live_state.items()
            }
            state_path = output_dir / "final_state.pt"
            temporary_state = state_path.with_suffix(f".tmp.{os.getpid()}.pt")
            torch.save(
                {
                    "adapter_state": adapter_state,
                    "optimizer_state": _cpu_optimizer_state(optimizer),
                },
                temporary_state,
            )
            os.replace(temporary_state, state_path)
            sample_ids = [
                {
                    key: row[key]
                    for key in ("epoch", "step_id", "seed", "collection_id", "macro_id")
                }
                for row in update["samples"]
            ]
            result = {
                "schema": PILOT_SCHEMA,
                "status": "PASS",
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "mode": args.mode,
                "world_size": int(context.world_size),
                "global_effective_batch_size": int(config["effective_batch_size"]),
                "optimizer_steps": int(update["optimizer_steps_this_round"]),
                "sample_ids": sample_ids,
                "steps": update["steps"],
                "optimizer_trace": optimizer_trace,
                "rank_payloads": rank_payloads,
                "elapsed_seconds": float(elapsed_seconds),
                "peak_memory_mib": float(
                    torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                ),
                "final_adapter_hash": rank_payload["final_adapter_hash"],
                "fixed_package": {
                    "source_config": str(config_path),
                    "source_config_sha256": hashlib.sha256(
                        config_path.read_bytes()
                    ).hexdigest(),
                    "artifacts": artifacts_before,
                    "selected_train_trajectories": int(len(selected_train)),
                    "selected_calibration_trajectories": int(
                        len(selected) - len(selected_train)
                    ),
                    "labels_per_train_trajectory": int(
                        args.labels_per_train_trajectory
                    ),
                    "read_only_unchanged": True,
                },
                "state_path": str(state_path),
            }
            _write_json(output_dir / "result.json", result)
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
    finally:
        runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
