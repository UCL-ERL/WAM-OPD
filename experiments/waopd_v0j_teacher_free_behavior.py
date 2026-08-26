"""Teacher-free closed-loop evaluation for JointLoRA and dual-bank LoRA.

This is intentionally a narrow evaluator for the post-V0J behavioral
amendment.  It uses the native Student runtime, never constructs a Teacher,
and can load a checked adapter state on top of the immutable released Student.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Callable, Mapping

import numpy as np
import torch

from experiments.dual_mode_lora import dual_mode_lora_gate_counts
from experiments.waopd_native_closed_loop_runner import (
    LockedNoiseBank,
    _initialize_task_local_success_state,
    run_live_episode,
)
from experiments.waopd_v0_video_opd import NativeV0VideoRuntime


def _sha256(path: Path) -> str:
    if path.is_dir():
        files = []
        for item in sorted(item for item in path.rglob("*") if item.is_file()):
            files.append(
                {
                    "relative_path": str(item.relative_to(path)),
                    "size": item.stat().st_size,
                    "sha256": _sha256(item),
                }
            )
        return hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _student_checkpoint_identity(path: Path, output: Path) -> tuple[str, str]:
    """Use the frozen tree manifest when this evaluator runs under the V0J run root.

    The released Student is a HuggingFace directory, not a single checkpoint
    file.  The first behavioral attempt completed its rollout but failed only
    while serializing ``sha256(directory)``.  The frozen protocol already
    contains the authoritative aggregate tree hash, so consume that identity
    here and retain a generic directory-tree fallback for standalone use.
    """
    resolved = path.expanduser().resolve()
    for ancestor in (output.expanduser().resolve().parent, *output.expanduser().resolve().parents):
        protocol_path = ancestor / "goal_v0j_behavioral_eval_protocol.json"
        if not protocol_path.is_file():
            continue
        try:
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        protocol_root = Path(protocol.get("student_checkpoint", "")).expanduser().resolve()
        protocol_hash = protocol.get("student_checkpoint_tree_sha256")
        if protocol_root == resolved and isinstance(protocol_hash, str) and protocol_hash:
            return protocol_hash, "frozen_behavior_protocol_tree_manifest"
    return _sha256(resolved), "runtime_directory_tree_manifest"


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AdapterLoadSpec:
    kind: str
    rank: int
    alpha: float
    dropout: float
    block_indices: tuple[int, ...]
    contract: dict[str, Any] | None
    state_dict: dict[str, torch.Tensor] | None


def _resolve_adapter_kind(
    adapter_state: Path | None,
    explicit_kind: str | None,
    *,
    payload: object | None = None,
) -> str:
    """Resolve checkpoint type without silently treating dual banks as JointLoRA."""

    if adapter_state is None:
        if explicit_kind is not None:
            raise ValueError("--adapter-kind requires --adapter-state")
        return "none"
    resolved = adapter_state.expanduser().resolve()
    if payload is None:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    declared_kind = (
        payload.get("adapter_kind") if isinstance(payload, Mapping) else None
    )
    if declared_kind is None:
        state = (
            payload.get("adapter_state_dict", payload)
            if isinstance(payload, Mapping)
            else None
        )
        if not isinstance(state, Mapping):
            raise ValueError("metadata-less adapter checkpoint is not a state dict")
        names = [str(name) for name in state]
        has_dual_keys = any(
            name.endswith(
                (
                    ".video_lora_A",
                    ".video_lora_B",
                    ".action_lora_A",
                    ".action_lora_B",
                )
            )
            for name in names
        )
        has_joint_keys = any(
            name.endswith((".lora_A", ".lora_B")) for name in names
        )
        if has_dual_keys:
            raise ValueError(
                "metadata-less checkpoint contains dual-bank keys; "
                "pass a contracted dual checkpoint"
            )
        if not has_joint_keys:
            raise ValueError(
                "metadata-less checkpoint has no recognized JointLoRA keys"
            )
        if explicit_kind not in {None, "joint_lora"}:
            raise ValueError(
                "metadata-less JointLoRA checkpoint conflicts with "
                f"--adapter-kind {explicit_kind!r}"
            )
        # Legacy V0J files were plain JointLoRA state dicts without metadata.
        return "joint_lora"
    if explicit_kind is not None:
        if explicit_kind not in {"joint_lora", "dual_lora"}:
            raise ValueError(f"unsupported adapter kind: {explicit_kind!r}")
        if declared_kind is not None and declared_kind != explicit_kind:
            raise ValueError(
                "adapter kind override differs from checkpoint metadata: "
                f"override={explicit_kind!r}, checkpoint={declared_kind!r}"
            )
        return explicit_kind
    if declared_kind not in {"joint_lora", "dual_lora"}:
        raise ValueError(f"unsupported checkpoint adapter kind: {declared_kind!r}")
    return str(declared_kind)


def _resolve_adapter_load_spec(
    adapter_state: Path | None, explicit_kind: str | None
) -> AdapterLoadSpec:
    """Reconstruct the exact adapter topology declared by a checkpoint."""

    defaults = {
        "rank": 8,
        "alpha": 8.0,
        "dropout": 0.0,
        "block_indices": (26, 27, 28, 29),
    }
    if adapter_state is None:
        kind = _resolve_adapter_kind(None, explicit_kind)
        return AdapterLoadSpec(kind=kind, contract=None, state_dict=None, **defaults)

    resolved = adapter_state.expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    kind = _resolve_adapter_kind(
        resolved, explicit_kind, payload=payload
    )
    if not isinstance(payload, Mapping):
        raise ValueError("adapter checkpoint is not a mapping")
    raw_state = payload.get("adapter_state_dict", payload)
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise ValueError("adapter checkpoint has no adapter state dict")
    state_dict: dict[str, torch.Tensor] = {}
    for name, value in raw_state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"adapter tensor {name!r} is not a Tensor")
        state_dict[str(name)] = value

    raw_contract = payload.get("adapter_contract")
    if raw_contract is None:
        return AdapterLoadSpec(
            kind=kind, contract=None, state_dict=state_dict, **defaults
        )
    if not isinstance(raw_contract, Mapping):
        raise ValueError("adapter_contract is not a mapping")
    contract = dict(raw_contract)
    contract_kind = contract.get("adapter_kind")
    if contract_kind is not None and contract_kind != kind:
        raise ValueError(
            "adapter contract kind differs from checkpoint kind: "
            f"{contract_kind!r} != {kind!r}"
        )
    rank = contract.get("rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("adapter_contract.rank must be a positive integer")
    alpha = contract.get("alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("adapter_contract.alpha must be numeric")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("adapter_contract.alpha must be finite and positive")
    dropout = contract.get("dropout")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError("adapter_contract.dropout must be numeric")
    dropout = float(dropout)
    if dropout != 0.0:
        raise ValueError("behavior evaluation requires zero adapter dropout")
    raw_blocks = contract.get("block_indices")
    if not isinstance(raw_blocks, (list, tuple)) or not raw_blocks:
        raise ValueError("adapter_contract.block_indices must be non-empty")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw_blocks):
        raise ValueError("adapter_contract.block_indices must contain integers")
    block_indices = tuple(int(item) for item in raw_blocks)
    if len(set(block_indices)) != len(block_indices):
        raise ValueError("adapter_contract.block_indices contains duplicates")
    return AdapterLoadSpec(
        kind=kind,
        rank=int(rank),
        alpha=alpha,
        dropout=dropout,
        block_indices=block_indices,
        contract=contract,
        state_dict=state_dict,
    )


def _validate_loaded_adapter(
    runtime: NativeV0VideoRuntime, spec: AdapterLoadSpec
) -> None:
    """Reject partial ``strict=False`` loads before an episode can run."""

    if spec.state_dict is None:
        return
    if spec.contract is not None and runtime.adapter_contract() != spec.contract:
        raise RuntimeError("runtime adapter contract differs from checkpoint")
    loaded = runtime.adapter_state()
    expected_names = set(spec.state_dict)
    loaded_names = set(loaded)
    if loaded_names != expected_names:
        raise RuntimeError(
            "runtime adapter parameter names differ from checkpoint: "
            f"missing={sorted(expected_names - loaded_names)}, "
            f"unexpected={sorted(loaded_names - expected_names)}"
        )
    mismatched = [
        name
        for name in sorted(expected_names)
        if not torch.equal(
            loaded[name].detach().cpu(), spec.state_dict[name].detach().cpu()
        )
    ]
    if mismatched:
        raise RuntimeError(
            f"runtime adapter tensors differ from checkpoint: {mismatched}"
        )


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def _worker_progress(
    *, worker: object, task_env: object, parent_snapshot: dict[str, Any], task: str
) -> dict[str, Any]:
    """Read continuous task progress from the worker's final simulator state."""

    from experiments.robotwin_sim_snapshot import restore_simulator_snapshot
    from experiments.stage_h_task_progress import collect_task_progress

    end_snapshot = worker.snapshot()["snapshot"]
    restore_simulator_snapshot(task_env, end_snapshot)
    try:
        return dict(collect_task_progress(task, task_env))
    finally:
        restore_simulator_snapshot(task_env, parent_snapshot)


def run_one(
    *,
    task: str,
    task_config: str,
    seed: int,
    chunks: int,
    max_control_steps: int | None = None,
    noise_base_seed: int = 2026080401,
    student: Path,
    output: Path,
    project_root: Path,
    device: str,
    enable_offload: bool,
    official_offload_parity: bool,
    adapter_state: Path | None,
    adapter_kind_override: str | None = None,
    prompt_override: str | None = None,
    macro_callback: Callable[[dict[str, Any]], None] | None = None,
    observation_callback: Callable[[dict[str, Any]], None] | None = None,
    stop_on_success: bool = True,
) -> dict[str, Any]:
    workspace_root = Path(__file__).resolve().parents[1]
    project_root = project_root.expanduser().resolve()
    robotwin_root = project_root / "third_party/RoboTwin-lingbot-native"
    sys.path[:0] = [
        str(workspace_root),
        str(project_root / "src"),
        str(project_root),
        str(project_root / "third_party/lingbot-va"),
        str(robotwin_root),
    ]
    os.environ["ROBOTWIN_ROOT"] = str(robotwin_root)
    os.chdir(robotwin_root)

    from evaluation.robotwin.eval_polict_client_openpi import (
        add_init_pose,
        class_decorator,
        format_obs,
    )
    from experiments.prototype_stage1_fixed_action_robotwin import (
        build_task_args,
        install_enhanced_determinism,
    )
    from experiments.robotwin_persistent_physics_worker import (
        PersistentNativePhysicsWorker,
    )
    from experiments.robotwin_sim_snapshot import (
        capture_simulator_snapshot,
        simulator_state_sha256,
    )

    install_enhanced_determinism()
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    task_env = class_decorator(task)
    task_args = build_task_args(robotwin_root, task, task_config)
    task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **task_args)
    _initialize_task_local_success_state(task_env, task)
    from description.utils.generate_episode_instructions import (
        generate_episode_descriptions,
    )

    if prompt_override is None:
        descriptions = generate_episode_descriptions(
            task, [task_env.info["info"]], max_descriptions=100
        )
        if not descriptions or not descriptions[0].get("seen"):
            raise RuntimeError("released seen instruction generation returned no prompt")
        prompt = str(np.random.choice(descriptions[0]["seen"]))
    else:
        prompt = str(prompt_override).strip()
        if not prompt:
            raise ValueError("prompt_override must not be empty")
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
    noise_bank = LockedNoiseBank(
        task=task,
        seed=seed,
        device=torch.device(device),
        dtype=torch.bfloat16,
        student_base_seed=int(noise_base_seed),
    )

    adapter_spec = _resolve_adapter_load_spec(adapter_state, adapter_kind_override)
    adapter_kind = adapter_spec.kind
    adapter_contract = adapter_spec.contract
    runtime = NativeV0VideoRuntime(
        student_checkpoint=student.expanduser().resolve(),
        teacher_transformer=None,
        device=device,
        save_root=output.parent / "native_save",
        enable_offload=enable_offload,
        official_offload_parity=official_offload_parity,
        adapter_rank=adapter_spec.rank,
        adapter_state=adapter_state,
        adapter_kind=adapter_kind,
        lora_alpha=adapter_spec.alpha,
        lora_dropout=adapter_spec.dropout,
        lora_block_indices=adapter_spec.block_indices,
    )
    _validate_loaded_adapter(runtime, adapter_spec)
    del adapter_spec
    runtime_nfe = {
        "video": int(runtime.server.job_config.num_inference_steps),
        "action": int(runtime.server.job_config.action_num_inference_steps),
    }
    if runtime_nfe != {"video": 1, "action": 1}:
        raise RuntimeError(f"behavior evaluator requires 1v/1a, got {runtime_nfe}")
    loaded_adapter_hashes = runtime.parameter_hashes()
    worker = PersistentNativePhysicsWorker(
        task_env=task_env,
        prompt=prompt,
        initial_eef_pose=initial_eef_pose,
        format_obs=None,
        materialize_renderer=None,
        worker_mode="parent_render_bridge",
    )
    progress: dict[str, Any] | None = None
    adapter_gate_counts: dict[str, int] | None = None
    try:
        episode = run_live_episode(
            runtime=runtime,
            task_env=task_env,
            worker=worker,
            parent_snapshot=parent_snapshot,
            initial_observation=initial_observation,
            initial_eef_pose=initial_eef_pose,
            format_obs=format_obs,
            add_init_pose=add_init_pose,
            task=task,
            task_config=task_config,
            seed=seed,
            prompt=prompt,
            arm="SS",
            chunks=chunks,
            max_control_steps=max_control_steps,
            noise_bank=noise_bank,
            macro_callback=macro_callback,
            observation_callback=observation_callback,
            stop_on_success=bool(stop_on_success),
        )
        progress = _worker_progress(
            worker=worker,
            task_env=task_env,
            parent_snapshot=parent_snapshot,
            task=task,
        )
        if adapter_kind == "dual_lora":
            adapter_gate_counts = dual_mode_lora_gate_counts(
                runtime.server.transformer
            )
            if adapter_gate_counts["video_forward_calls"] <= 0:
                raise RuntimeError("dual behavior eval never activated video bank")
            if adapter_gate_counts["action_forward_calls"] <= 0:
                raise RuntimeError("dual behavior eval never activated action bank")
            if adapter_gate_counts["bypass_forward_calls"] != 0:
                raise RuntimeError("dual behavior eval bypassed both adapter banks")
    finally:
        worker.close(force=True)
        task_env.close_env()
        runtime.close()
        torch.cuda.empty_cache()

    student_checkpoint_sha256, student_checkpoint_hash_source = _student_checkpoint_identity(
        student, output
    )
    result = {
        "schema": "waopd_v0j_teacher_free_behavior_episode_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "PASS",
        "backend": "native_lingbot_student_only_not_rlinf_adaptation",
        "task": task,
        "task_config": task_config,
        "seed": int(seed),
        "chunks": int(chunks),
        "max_control_steps": (
            None if max_control_steps is None else int(max_control_steps)
        ),
        "success": bool(episode["success"]),
        "progress": progress,
        "episode": episode,
        "student_checkpoint": str(student.expanduser().resolve()),
        "student_checkpoint_sha256": student_checkpoint_sha256,
        "student_checkpoint_hash_source": student_checkpoint_hash_source,
        "adapter_state": str(adapter_state.expanduser().resolve()) if adapter_state else None,
        "adapter_state_sha256": _sha256(adapter_state.expanduser().resolve()) if adapter_state else None,
        "adapter_kind": adapter_kind,
        "adapter_contract": adapter_contract,
        "teacher_loaded": False,
        "teacher_called": False,
        "teacher_transformer": None,
        "training_started": False,
        "runtime_nfe": runtime_nfe,
        "prompt": prompt,
        "prompt_hash": _stable_hash(prompt),
        "initial_snapshot_sha256": parent_snapshot_hash,
        "noise_contract": {
            "source": "LockedNoiseBank",
            "task": task,
            "seed": int(seed),
            "student_base_seed": int(noise_base_seed),
            "chunks": int(chunks),
            "chunk_action_noise_hashes": [
                row["action_base_noise_hash"] for row in episode["chunks"]
            ],
            "chunk_video_noise_hashes": [
                row["video_base_noise_hash"] for row in episode["chunks"]
            ],
        },
        "adapter_parameter_hashes": loaded_adapter_hashes,
        "adapter_gate_counts": adapter_gate_counts,
    }
    _write(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default="demo_randomized")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--chunks", type=int, required=True)
    parser.add_argument("--max-control-steps", type=int)
    parser.add_argument("--noise-base-seed", type=int, default=2026080401)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--enable-offload", action="store_true")
    parser.add_argument("--official-offload-parity", action="store_true")
    parser.add_argument("--adapter-state", type=Path)
    parser.add_argument("--adapter-kind", choices=("joint_lora", "dual_lora"))
    parser.add_argument("--prompt")
    args = parser.parse_args()
    try:
        result = run_one(
            task=args.task,
            task_config=args.task_config,
            seed=args.seed,
            chunks=args.chunks,
            max_control_steps=args.max_control_steps,
            noise_base_seed=args.noise_base_seed,
            student=args.student,
            output=args.output.expanduser().resolve(),
            project_root=args.project_root,
            device=args.device,
            enable_offload=bool(args.enable_offload),
            official_offload_parity=bool(args.official_offload_parity),
            adapter_state=args.adapter_state,
            adapter_kind_override=args.adapter_kind,
            prompt_override=args.prompt,
        )
    except Exception as exc:
        result = {
            "schema": "waopd_v0j_teacher_free_behavior_episode_v1",
            "status": "BLOCKED",
            "backend": "native_lingbot_student_only_not_rlinf_adaptation",
            "reason": f"Student-only behavior exception: {type(exc).__name__}: {exc}",
            "teacher_loaded": False,
            "teacher_called": False,
            "training_started": False,
        }
        _write(args.output.expanduser().resolve(), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
