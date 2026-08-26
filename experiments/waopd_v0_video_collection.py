"""Collect fresh native Student occupancy contexts for Goal V0.

This collector executes only the released Student in the environment.  It
does not load a Teacher and does not train.  The actual common video/action
noise tensors and the Student's prepared plan are saved at each macro boundary
through the callback seam in ``run_live_episode``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch

from experiments.goal1_exact_condition import tensor_hash
from experiments.waopd_native_closed_loop_runner import (
    HistoryInput,
    NativeClosedLoopError,
    _initialize_task_local_success_state,
    _serialize_fingerprint,
    run_live_episode,
)
from experiments.waopd_native_student_only import NativeStudentOnlyRuntime


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class V0CommonNoiseBank:
    """One actual epsilon pair keyed without arm/family/checkpoint identity."""

    schema = "waopd_v0_common_noise_v1"

    def __init__(
        self,
        *,
        protocol_id: str,
        unit_id: str,
        task: str,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        gpu_visibility_amendment_sha256: str | None = None,
        schema: str = "waopd_v0_common_noise_v1",
    ) -> None:
        self.protocol_id = str(protocol_id)
        self.unit_id = str(unit_id)
        self.task = str(task)
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype
        self.gpu_visibility_amendment_sha256 = gpu_visibility_amendment_sha256
        self.schema = str(schema)
        self._cache: dict[tuple[int, tuple[int, ...], tuple[int, ...]], dict[str, torch.Tensor]] = {}

    def _generator(self, *, macro_id: int, stream: str) -> torch.Generator:
        if stream not in {"epsilon_v", "epsilon_a"}:
            raise ValueError(stream)
        payload = {
            "protocol_id": self.protocol_id,
            "unit_id": self.unit_id,
            "macro_id": int(macro_id),
            "stream": stream,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
        return generator

    def pair(
        self,
        *,
        family: str,
        chunk_id: int,
        video_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
    ) -> dict[str, torch.Tensor]:
        # ``family`` is accepted only for the backwards-compatible runner
        # seam.  It is deliberately absent from the derivation key.
        if str(family) not in {"student", "teacher"}:
            raise ValueError(f"unsupported model family: {family!r}")
        key = (int(chunk_id), tuple(video_shape), tuple(action_shape))
        if key not in self._cache:
            video = torch.randn(
                video_shape,
                generator=self._generator(macro_id=int(chunk_id), stream="epsilon_v"),
                device="cpu",
                dtype=self.dtype,
            )
            action = torch.randn(
                action_shape,
                generator=self._generator(macro_id=int(chunk_id), stream="epsilon_a"),
                device="cpu",
                dtype=self.dtype,
            )
            self._cache[key] = {"video": video, "action": action}
        return {
            name: value.to(device=self.device, dtype=self.dtype).detach().clone()
            for name, value in self._cache[key].items()
        }

    def key(self, macro_id: int, stream: str) -> dict[str, Any]:
        if stream not in {"epsilon_v", "epsilon_a"}:
            raise ValueError(stream)
        return {
            "schema": self.schema,
            "protocol_id": self.protocol_id,
            "unit_id": self.unit_id,
            "macro_id": int(macro_id),
            "stream": stream,
        }


def _serialize_history(history: list[HistoryInput]) -> list[dict[str, Any]]:
    return [
        {
            "frame_st_id": int(record.frame_st_id),
            "latent": record.latent.detach().cpu(),
            "action": record.action.detach().cpu(),
            "observations": deepcopy(record.observations),
        }
        for record in history
    ]


def serialize_context(
    event: dict[str, Any],
    *,
    split: str,
    unit_id: str,
    bank: V0CommonNoiseBank,
    context_schema: str = "waopd_goal_v0_student_occupancy_context_v1",
    artifact_kind: str = "v0_student_occupancy_video_context",
    protocol_sha256: str | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    plan = event["plan"]
    solve = event["solve"]
    video_noise = event["video_base_noise"].detach().cpu()
    action_noise = event["action_base_noise"].detach().cpu()
    history = _serialize_history(event["history"])
    frame_st_id = int(event["frame_st_id"])
    context_id = f"{unit_id}_macro{int(event['chunk_id']):02d}"
    return {
        "schema": str(context_schema),
        "artifact_kind": str(artifact_kind),
        "training_started": False,
        "split": str(split),
        "unit_id": str(unit_id),
        "context_id": context_id,
        "task": str(event["task"]),
        "task_config": str(event["task_config"]),
        "seed": int(event["seed"]),
        "prompt": str(event["prompt"]),
        "prompt_token_ids": tuple(int(x) for x in event["prompt_token_ids"]),
        "frame_st_id": frame_st_id,
        "macro_id": int(event["chunk_id"]),
        "macro_boundary": [1.0, 0.0],
        "initial_observation": deepcopy(event["initial_observation"]),
        "initial_latent": event["initial_latent"].detach().cpu(),
        "history": history,
        "raw_z_s": plan.raw_z_s.detach().cpu(),
        "prepared_z_s": plan.prepared_z_s.detach().cpu(),
        "prepared_z_s_timestep": plan.prepared_z_s_timestep.detach().cpu(),
        "latent_cond_applied": bool(plan.latent_cond_applied),
        "raw_z_s_hash": tensor_hash(plan.raw_z_s),
        "prepared_z_s_hash": tensor_hash(plan.prepared_z_s),
        "prepared_z_s_timestep_hash": tensor_hash(plan.prepared_z_s_timestep),
        "epsilon_v": video_noise,
        "epsilon_a": action_noise,
        "epsilon_v_hash": tensor_hash(video_noise),
        "epsilon_a_hash": tensor_hash(action_noise),
        "noise_keys": {
            "epsilon_v": bank.key(int(event["chunk_id"]), "epsilon_v"),
            "epsilon_a": bank.key(int(event["chunk_id"]), "epsilon_a"),
        },
        "student_model_action": solve.model_action.detach().cpu(),
        "student_action_input_noise": solve.action_noise.detach().cpu(),
        "student_action_timestep": solve.action_timestep.detach().cpu(),
        "student_valid_action_mask": solve.mask.detach().cpu(),
        "student_env_action": np.asarray(solve.env_action),
        "student_action_input_noise_hash": tensor_hash(solve.action_noise),
        "student_action_timestep_hash": tensor_hash(solve.action_timestep),
        "student_valid_action_mask_hash": tensor_hash(solve.mask),
        "student_action_token_positions": tuple(int(x) for x in solve.action_token_positions),
        "student_cache_valid_length": int(solve.cache_valid_length),
        "student_plan_owner": str(solve.plan_owner),
        "student_cache_owner": str(solve.cache_name),
        "student_fingerprint": _serialize_fingerprint(event.get("student_fingerprint")),
        "teacher_fingerprint_at_collection": _serialize_fingerprint(event.get("teacher_fingerprint")),
        "history_semantics": {
            "student_executes_environment": True,
            "teacher_labels_only": True,
            "teacher_actions_executed": False,
            "history_rebuilt_from_semantic_records": True,
        },
        "physical_gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_visibility_amendment_sha256": bank.gpu_visibility_amendment_sha256,
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": manifest_sha256,
        "official_offload_parity": bool(getattr(event, "official_offload_parity", False))
        if not isinstance(event, dict)
        else bool(event.get("official_offload_parity", False)),
    }


def _validate_gpu_visibility() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [item.strip() for item in visible.split(",") if item.strip()]
    if not tokens or any(item not in {"5", "6", "7"} for item in tokens):
        raise RuntimeError(
            "V0F GPU policy requires CUDA_VISIBLE_DEVICES containing only physical 5/6/7; "
            f"got {visible!r}"
        )


def _load_partition(path: Path, split: str, tasks: set[str]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload.get("partitions", [])
        if row.get("split") == split and row.get("task") in tasks
    ]
    if not rows:
        raise ValueError(f"partition has no rows for {split}/{sorted(tasks)}")
    return rows


def _setup_task(
    *,
    project_root: Path,
    task: str,
    seed: int,
    task_config: str = "demo_randomized",
) -> tuple[object, dict[str, Any], object, object, object, np.ndarray, dict[str, Any], str]:
    robotwin_root = project_root / "third_party/RoboTwin-lingbot-native"
    sys.path[:0] = [
        str(project_root / "src"),
        str(project_root),
        str(robotwin_root),
        str(project_root / "third_party/lingbot-va"),
    ]
    os.environ["ROBOTWIN_ROOT"] = str(robotwin_root)
    os.chdir(robotwin_root)
    from evaluation.robotwin.eval_polict_client_openpi import add_init_pose, class_decorator, format_obs
    from experiments.prototype_stage1_fixed_action_robotwin import build_task_args, install_enhanced_determinism
    from experiments.robotwin_sim_snapshot import capture_simulator_snapshot, simulator_state_sha256

    install_enhanced_determinism()
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    task_args = build_task_args(robotwin_root, task, str(task_config))
    task_env = class_decorator(task)
    task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **task_args)
    _initialize_task_local_success_state(task_env, task)
    from description.utils.generate_episode_instructions import generate_episode_descriptions

    descriptions = generate_episode_descriptions(
        task,
        [task_env.info["info"]],
        max_descriptions=100,
    )
    if not descriptions or not descriptions[0].get("seen"):
        raise NativeClosedLoopError(f"no released seen instruction for {task}/{seed}")
    prompt = str(np.random.choice(descriptions[0]["seen"]))
    task_env.set_instruction(prompt)
    initial_raw = task_env.get_obs()
    initial_observation = format_obs(initial_raw, prompt)
    initial_eef_pose = np.asarray(
        initial_raw["endpose"]["left_endpose"]
        + [initial_raw["endpose"]["left_gripper"]]
        + initial_raw["endpose"]["right_endpose"]
        + [initial_raw["endpose"]["right_gripper"]],
        dtype=np.float64,
    )
    parent_snapshot = capture_simulator_snapshot(task_env, capture_cuda_rng=True)
    parent_snapshot_hash = simulator_state_sha256(parent_snapshot)
    return (
        task_env,
        initial_observation,
        format_obs,
        add_init_pose,
        parent_snapshot,
        initial_eef_pose,
        {"prompt": prompt, "parent_snapshot_sha256": parent_snapshot_hash},
        task,
    )


def collect_one(
    *,
    runtime: NativeStudentOnlyRuntime,
    project_root: Path,
    protocol_id: str,
    protocol_sha256: str | None,
    manifest_sha256: str | None,
    row: dict[str, Any],
    output_dir: Path,
    chunks: int,
    context_schema: str,
    artifact_kind: str,
    noise_schema: str,
) -> dict[str, Any]:
    task = str(row["task"])
    seed = int(row["seed"])
    split = str(row["split"])
    unit_id = str(row["unit_id"])
    (
        task_env,
        initial_observation,
        format_obs,
        add_init_pose,
        parent_snapshot,
        initial_eef_pose,
        setup_meta,
        _task,
    ) = _setup_task(project_root=project_root, task=task, seed=seed)
    from experiments.robotwin_persistent_physics_worker import PersistentNativePhysicsWorker

    bank = V0CommonNoiseBank(
        protocol_id=protocol_id,
        unit_id=unit_id,
        task=task,
        seed=seed,
        device=runtime.device,
        dtype=runtime.dtype,
        gpu_visibility_amendment_sha256=str(row.get("gpu_visibility_amendment_sha256"))
        if row.get("gpu_visibility_amendment_sha256")
        else None,
        schema=noise_schema,
    )
    contexts: list[dict[str, Any]] = []

    def callback(event: dict[str, Any]) -> None:
        event["official_offload_parity"] = bool(getattr(runtime, "official_offload_parity", False))
        contexts.append(
            serialize_context(
                event,
                split=split,
                unit_id=unit_id,
                bank=bank,
                context_schema=context_schema,
                artifact_kind=artifact_kind,
                protocol_sha256=protocol_sha256,
                manifest_sha256=manifest_sha256,
            )
        )

    worker = PersistentNativePhysicsWorker(
        task_env=task_env,
        prompt=setup_meta["prompt"],
        initial_eef_pose=initial_eef_pose,
        format_obs=None,
        materialize_renderer=None,
        worker_mode="parent_render_bridge",
    )
    try:
        result = run_live_episode(
            runtime=runtime,
            task_env=task_env,
            worker=worker,
            parent_snapshot=parent_snapshot,
            initial_observation=initial_observation,
            initial_eef_pose=initial_eef_pose,
            format_obs=format_obs,
            add_init_pose=add_init_pose,
            task=task,
            task_config="demo_randomized",
            seed=seed,
            prompt=setup_meta["prompt"],
            arm="SS",
            chunks=int(chunks),
            noise_bank=bank,
            macro_callback=callback,
            stop_on_success=False,
        )
    finally:
        worker.close(force=True)
        task_env.close_env()
    if not contexts:
        raise NativeClosedLoopError(f"Student produced no context for {unit_id}")
    payload = {
        "schema": "waopd_goal_v0_student_episode_v1",
        "training_started": False,
        "unit_id": unit_id,
        "split": split,
        "task": task,
        "seed": seed,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": manifest_sha256,
        "setup_snapshot_sha256": setup_meta["parent_snapshot_sha256"],
        "student_episode": result,
        "contexts": contexts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{unit_id}.pt"
    torch.save(payload, path)
    row_out = {
        "unit_id": unit_id,
        "split": split,
        "task": task,
        "seed": seed,
        "status": "PASS",
        "path": str(path),
        "contexts": len(contexts),
        "frame_st_ids": [int(item["frame_st_id"]) for item in contexts],
        "setup_snapshot_sha256": setup_meta["parent_snapshot_sha256"],
        "episode_success": bool(result.get("success")),
        "policy_inference_started": True,
        "training_started": False,
    }
    return row_out


def main() -> int:
    parser = argparse.ArgumentParser()
    workspace_root = Path(__file__).resolve().parents[1]
    artifact_root = Path(
        os.environ.get("WAM_OPD_ARTIFACT_ROOT", workspace_root / ".artifacts")
    )
    parser.add_argument("--workspace-root", type=Path, default=workspace_root)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(
            os.environ.get(
                "WAVE_RL_ROOT",
                os.environ.get("PROJECT_ROOT", workspace_root.parent / "wave-rl"),
            )
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("target_train", "target_validation", "retention"), required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--student",
        type=Path,
        default=Path(
            os.environ.get(
                "WAM_OPD_STUDENT_ROOT",
                artifact_root / "models" / "FlashWAM-RoboTwin",
            )
        ),
    )
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--protocol-name", default="goal_v0_training_protocol.json")
    parser.add_argument("--manifest-name", default="goal_v0_data_partition_manifest.json")
    parser.add_argument("--output-root", default="student_contexts")
    parser.add_argument("--official-offload-parity", action="store_true")
    parser.add_argument("--summary-name", default=None)
    args = parser.parse_args()
    _validate_gpu_visibility()
    # The runtime is constructed before _setup_task is called.  Install the
    # native LingBot import roots here, rather than relying on the later
    # simulator setup helper to do it after model construction.
    project_root = args.project_root.expanduser().resolve()
    robotwin_root = project_root / "third_party/RoboTwin-lingbot-native"
    sys.path[:0] = [
        str(args.workspace_root.expanduser().resolve()),
        str(project_root / "src"),
        str(project_root),
        str(robotwin_root),
        str(project_root / "third_party/lingbot-va"),
    ]
    os.environ["ROBOTWIN_ROOT"] = str(robotwin_root)
    run_dir = args.run_dir.expanduser().resolve()
    protocol_path = run_dir / args.protocol_name
    manifest_path = run_dir / args.manifest_name
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_id = str(
        protocol.get("protocol_id")
        or protocol.get("parent", {}).get("hashes", {}).get("goal_d2_scientific_adjudication")
    )
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    is_v0f = str(args.protocol_name).startswith("goal_v0f_")
    amendment_path = run_dir / (
        "goal_v0f_gpu_visibility_amendment.json"
        if is_v0f
        else "goal_v0_gpu_visibility_amendment.json"
    )
    amendment_hash = hashlib.sha256(amendment_path.read_bytes()).hexdigest() if amendment_path.is_file() else None
    rows = _load_partition(manifest_path, args.split, set(args.tasks))
    output_dir = run_dir / args.output_root / args.split
    runtime = NativeStudentOnlyRuntime(
        student_checkpoint=args.student.expanduser().resolve(),
        device=args.device,
        save_root=run_dir / "native_save" / args.split,
        enable_offload=True,
        official_offload_parity=bool(args.official_offload_parity),
    )
    summaries = []
    try:
        for row in rows:
            row = dict(row)
            row["gpu_visibility_amendment_sha256"] = amendment_hash
            summaries.append(
                collect_one(
                    runtime=runtime,
                    project_root=args.project_root.expanduser().resolve(),
                    protocol_id=protocol_id,
                    protocol_sha256=protocol_sha256,
                    manifest_sha256=manifest_sha256,
                    row=row,
                    output_dir=output_dir,
                    chunks=int(args.chunks),
                    context_schema=(
                        "waopd_goal_v0f_student_occupancy_context_v1"
                        if is_v0f
                        else "waopd_goal_v0_student_occupancy_context_v1"
                    ),
                    artifact_kind=(
                        "v0f_student_occupancy_video_context"
                        if is_v0f
                        else "v0_student_occupancy_video_context"
                    ),
                    noise_schema=(
                        "waopd_v0f_common_noise_v1"
                        if is_v0f
                        else "waopd_v0_common_noise_v1"
                    ),
                )
            )
            print(json.dumps(summaries[-1], sort_keys=True), flush=True)
    finally:
        del runtime
        torch.cuda.empty_cache()
    summary_prefix = "goal_v0f" if is_v0f else "goal_v0"
    summary_path = run_dir / (
        str(args.summary_name)
        if args.summary_name
        else f"{summary_prefix}_{args.split}_collection_summary.json"
    )
    summary_path.write_text(
        json.dumps(
            {
                "schema": "waopd_goal_v0_collection_summary_v1",
                "training_started": False,
                "split": args.split,
                "tasks": list(args.tasks),
                "protocol_id": protocol_id,
                "protocol_sha256": protocol_sha256,
                "manifest_sha256": manifest_sha256,
                "official_offload_parity": bool(args.official_offload_parity),
                "rows": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if all(row["status"] == "PASS" for row in summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
