"""One-shot paired Teacher-trajectory OPD for an Easy RoboTwin task.

The collector executes one successful TT trajectory and stores, at every
visited macro state, the Teacher video plan and Teacher action produced from
that exact plan/history/noise pair.  The trainer reuses all pre-success labels
offline and optimizes one shared Joint LoRA checkpoint with two losses:

* Student video plan -> frozen Teacher video plan;
* Student action on the detached Teacher plan -> frozen Teacher action.

The success-triggering action is included by the physical execution mask and
no post-success macro is collected.  Retention is deliberately disabled.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.opd_task_specs import (
    TASK_SPECS,
    require_training_task_config,
    resolve_task_chunks,
)
from experiments.train_stage_a_action_opd import terminal_execution_mask
from experiments.waopd_native_closed_loop_runner import (
    LockedNoiseBank,
    NativeClosedLoopError,
    _initialize_task_local_success_state,
    run_live_episode,
)
from experiments.waopd_v0_video_opd import (
    NativeV0VideoRuntime,
    action_huber_loss,
    video_huber_loss,
)


TRAJECTORY_SCHEMA = "waopd_paired_teacher_trajectory_v1"
CONTEXT_SCHEMA = "waopd_video_trajectory_context_v1"
CHECKPOINT_SCHEMA = "waopd_paired_teacher_joint_lora_v1"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone().cpu()


def _gate_episode(gate: Mapping[str, Any], arm: str) -> dict[str, Any]:
    matches = [row for row in gate.get("episodes", []) if row.get("arm") == arm]
    if len(matches) != 1:
        raise ValueError(f"gate must contain exactly one {arm} episode")
    return dict(matches[0])


def validate_gate_contract(
    gate: Mapping[str, Any],
    *,
    task: str,
    task_config: str,
    seed: int,
    chunks: int,
    max_control_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require the already-run exact common-noise causal gate."""

    expected = {
        "task": str(task),
        "task_config": str(task_config),
        "seed": int(seed),
        "chunks_requested": int(chunks),
        "max_control_steps": int(max_control_steps),
    }
    mismatches = {
        key: {"expected": value, "actual": gate.get(key)}
        for key, value in expected.items()
        if gate.get(key) != value
    }
    if mismatches:
        raise ValueError(f"gate/config mismatch: {mismatches}")
    if gate.get("status") != "PASS":
        raise ValueError("common-noise gate did not pass")
    if gate.get("training_started") is not False:
        raise ValueError("gate is contaminated by training")
    if gate.get("shared_noise_across_arms") is not True:
        raise ValueError("gate did not use exact shared noise across arms")
    if list(gate.get("arms", [])) != ["SS", "ST", "TS", "TT"]:
        raise ValueError("gate did not run the canonical SS/ST/TS/TT arms")
    ss = _gate_episode(gate, "SS")
    tt = _gate_episode(gate, "TT")
    if bool(ss.get("success")):
        raise ValueError("gate SS already succeeds; rescue experiment is not causal")
    if not bool(tt.get("success")):
        raise ValueError("gate TT failed; no successful paired Teacher trajectory exists")
    return ss, tt


def capture_teacher_target(event: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the complete paired TT macro target before physical execution."""

    if str(event.get("arm")) != "TT":
        raise NativeClosedLoopError(f"paired collector expected TT, got {event.get('arm')!r}")
    plan = event["plan"]
    solve = event["solve"]
    if str(solve.plan_owner) != "teacher":
        raise NativeClosedLoopError("TT action did not consume a Teacher-owned plan")
    if solve.plan.prepared_hash != plan.prepared_hash:
        raise NativeClosedLoopError("TT callback plan/action pair is not canonical")
    return {
        "macro_id": int(event["chunk_id"]),
        "frame_st_id": int(event["frame_st_id"]),
        "prompt_token_ids": tuple(int(value) for value in event["prompt_token_ids"]),
        "initial_latent": _cpu_tensor(event["initial_latent"]),
        "epsilon_v": _cpu_tensor(event["video_base_noise"]),
        "epsilon_a": _cpu_tensor(event["action_base_noise"]),
        "teacher_z_t": _cpu_tensor(plan.prepared_z_s),
        "teacher_z_t_timestep": _cpu_tensor(plan.prepared_z_s_timestep),
        "teacher_action": _cpu_tensor(solve.model_action),
        "teacher_action_input_noise": _cpu_tensor(solve.action_noise),
        "teacher_action_timestep": _cpu_tensor(solve.action_timestep),
        "teacher_action_valid_mask": _cpu_tensor(solve.mask).to(dtype=torch.bool),
        "teacher_action_token_positions": tuple(
            int(value) for value in solve.action_token_positions
        ),
        "teacher_cache_valid_length": int(solve.cache_valid_length),
    }


def build_teacher_trajectory(
    *,
    task: str,
    task_config: str,
    seed: int,
    prompt: str,
    initial_observation: Mapping[str, Any],
    episode: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Join pre-execution TT targets with post-execution terminal masks."""

    require_training_task_config(task_config)
    physical_rows = list(episode["chunks"])
    if not bool(episode.get("success")):
        raise NativeClosedLoopError("Teacher trajectory failed; training is not allowed")
    if len(targets) != len(physical_rows):
        raise NativeClosedLoopError(
            "Teacher target/physical-row counts differ: "
            f"{len(targets)}/{len(physical_rows)}"
        )
    if not targets:
        raise NativeClosedLoopError("Teacher trajectory produced no macro")

    labels: list[dict[str, Any]] = []
    cumulative_steps = 0
    for index, (target_value, physical) in enumerate(
        zip(targets, physical_rows, strict=True)
    ):
        if {
            index,
            int(target_value["macro_id"]),
            int(physical["chunk_id"]),
        } != {index}:
            raise NativeClosedLoopError("Teacher trajectory macro chronology mismatch")
        if int(target_value["frame_st_id"]) != int(physical["frame_st_id"]):
            raise NativeClosedLoopError("Teacher trajectory frame chronology mismatch")
        cumulative_steps += int(physical["action_steps"])
        label = {
            **dict(target_value),
            "history_prefix_length": int(index),
            "start_frame": int(physical["start_frame"]),
            "action_steps": int(physical["action_steps"]),
            "executed_action_mask": physical["executed_action_mask"],
            "terminal_reached": bool(physical["terminal_reached"]),
            "terminal_action_position": physical["terminal_action_position"],
            "horizon_reached": bool(physical.get("horizon_reached", False)),
            "task_success": bool(physical.get("task_success", False)),
            "eval_success": bool(physical.get("eval_success", False)),
            "cumulative_control_steps": int(cumulative_steps),
        }
        execution = terminal_execution_mask(
            label,
            frame_count=int(target_value["teacher_action"].shape[2]),
            horizon=int(target_value["teacher_action"].shape[3]),
        )
        if int(execution.sum()) != int(physical["action_steps"]):
            raise NativeClosedLoopError("physical action count differs from terminal mask")
        labels.append(label)

    success_rows = [
        index
        for index, row in enumerate(labels)
        if row["terminal_reached"] or row["task_success"] or row["eval_success"]
    ]
    if success_rows != [len(labels) - 1]:
        raise NativeClosedLoopError(
            f"Teacher labels do not stop exactly at first success: {success_rows}"
        )
    return {
        "schema": TRAJECTORY_SCHEMA,
        "task": str(task),
        "task_config": str(task_config),
        "seed": int(seed),
        "prompt": str(prompt),
        "initial_observation": deepcopy(dict(initial_observation)),
        "initial_latent": targets[0]["initial_latent"],
        "prompt_token_ids": targets[0]["prompt_token_ids"],
        "history": list(episode["history"]),
        "labels": labels,
        "teacher_outcome": {
            "success": True,
            "chunks_completed": int(episode["chunks_completed"]),
            "control_steps": int(episode["control_steps"]),
        },
        "success_trigger_label_included": True,
        "success_post_label_count": 0,
        "retention_weight": 0.0,
        "training_started": False,
    }


def materialize_teacher_context(
    trajectory: Mapping[str, Any], label: Mapping[str, Any]
) -> dict[str, Any]:
    prefix_length = int(label["history_prefix_length"])
    history = list(trajectory["history"])
    if prefix_length < 0 or prefix_length > len(history):
        raise ValueError("history prefix length is outside the Teacher trajectory")
    return {
        "schema": CONTEXT_SCHEMA,
        "split": "train",
        "unit_id": f"{trajectory['task']}_{trajectory['seed']}",
        "context_id": (
            f"{trajectory['task']}_{trajectory['seed']}_teacher_macro"
            f"{int(label['macro_id']):02d}"
        ),
        "task": str(trajectory["task"]),
        "task_config": require_training_task_config(str(trajectory["task_config"])),
        "seed": int(trajectory["seed"]),
        "prompt": str(trajectory["prompt"]),
        "prompt_token_ids": tuple(
            int(value) for value in trajectory["prompt_token_ids"]
        ),
        "frame_st_id": int(label["frame_st_id"]),
        "macro_id": int(label["macro_id"]),
        "initial_observation": deepcopy(trajectory["initial_observation"]),
        "initial_latent": trajectory["initial_latent"],
        "history": history[:prefix_length],
        "epsilon_v": label["epsilon_v"],
        "epsilon_a": label["epsilon_a"],
        "prepared_z_s": label["teacher_z_t"],
        "_v0_target_initial_latent_mode": "frozen_saved_initial_latent_v1",
    }


def teacher_video_execution_mask(
    reference: torch.Tensor, label: Mapping[str, Any]
) -> torch.Tensor:
    if reference.ndim != 5 or reference.shape[0] != 1:
        raise ValueError("video plan must have shape [1, channels, frames, height, width]")
    target_action = label["teacher_action"]
    executed = terminal_execution_mask(
        label,
        frame_count=int(reference.shape[2]),
        horizon=int(target_action.shape[3]),
    ).to(device=reference.device)
    frame_mask = executed.any(dim=1)
    mask = torch.ones_like(reference, dtype=torch.bool)
    mask &= frame_mask[None, None, :, None, None]
    if int(label["frame_st_id"]) == 0:
        mask[:, :, 0:1] = False
    if not bool(mask.any()):
        raise ValueError("terminal-aware Teacher video mask is empty")
    return mask


def teacher_action_execution_mask(
    valid_mask: torch.Tensor, label: Mapping[str, Any]
) -> torch.Tensor:
    if valid_mask.ndim != 5 or valid_mask.shape[0] != 1 or valid_mask.shape[-1] != 1:
        raise ValueError("action endpoint must have shape [1, channels, frames, horizon, 1]")
    executed = terminal_execution_mask(
        label,
        frame_count=int(valid_mask.shape[2]),
        horizon=int(valid_mask.shape[3]),
    ).to(device=valid_mask.device)
    valid_by_position = valid_mask.any(dim=(0, 1, 4))
    if bool((executed & ~valid_by_position).any()):
        raise ValueError("physical execution selects a model-invalid action position")
    mask = valid_mask & executed[None, None, :, :, None]
    if not bool(mask.any()):
        raise ValueError("terminal-aware Teacher action mask is empty")
    return mask


def trajectory_windows(
    labels: Sequence[dict[str, Any]], *, window_size: int, epoch: int, seed: int
) -> list[list[dict[str, Any]]]:
    """Group every macro once per epoch; window is optimizer grouping, not BPTT."""

    if int(window_size) <= 0:
        raise ValueError("window_size must be positive")
    windows = [
        list(labels[start : start + int(window_size)])
        for start in range(0, len(labels), int(window_size))
    ]
    random.Random(int(seed) + int(epoch)).shuffle(windows)
    return windows


def _student_video_plan(
    runtime: NativeV0VideoRuntime, context: dict[str, Any]
) -> torch.Tensor:
    """Run only the differentiable Student video branch and release its cache."""

    try:
        plan, _video_noise, _initial_latent = runtime._video_plan_student(context)
        return plan.prepared_z_s
    finally:
        runtime.server.transformer.clear_cache(runtime.server.cache_name)


def train_joint_adapter(
    *,
    trajectory_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    student: Path,
    device: str,
    save_root: Path,
    epochs: int,
    window_size: int,
    learning_rate: float,
    video_weight: float,
    action_weight: float,
    max_grad_norm: float,
    enable_offload: bool,
    official_offload_parity: bool,
) -> dict[str, Any]:
    trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=False)
    if not isinstance(trajectory, dict) or trajectory.get("schema") != TRAJECTORY_SCHEMA:
        raise ValueError("unexpected paired Teacher trajectory schema")
    require_training_task_config(str(trajectory["task_config"]))
    labels = list(trajectory["labels"])
    if not labels:
        raise ValueError("paired Teacher trajectory has no labels")
    if int(trajectory.get("success_post_label_count", -1)) != 0:
        raise ValueError("paired Teacher trajectory contains post-success labels")
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    for name, value in {
        "learning_rate": learning_rate,
        "video_weight": video_weight,
        "action_weight": action_weight,
        "max_grad_norm": max_grad_norm,
    }.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    runtime = NativeV0VideoRuntime(
        student_checkpoint=student,
        teacher_transformer=None,
        device=device,
        save_root=save_root,
        enable_offload=enable_offload,
        official_offload_parity=official_offload_parity,
        adapter_rank=8,
        adapter_kind="joint_lora",
        lora_alpha=8.0,
        lora_dropout=0.0,
        lora_block_indices=(26, 27, 28, 29),
    )
    parameters = [parameter for _, parameter in runtime.trainable]
    optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=0.0)
    base_hashes_before = runtime.base_parameter_hashes()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    step = 0
    try:
        with metrics_path.open("w", encoding="utf-8") as handle:
            for epoch in range(int(epochs)):
                windows = trajectory_windows(
                    labels,
                    window_size=int(window_size),
                    epoch=epoch,
                    seed=int(trajectory["seed"]),
                )
                for window in windows:
                    optimizer.zero_grad(set_to_none=True)
                    video_sum = 0.0
                    action_sum = 0.0
                    macro_ids: list[int] = []
                    for label in window:
                        context = materialize_teacher_context(trajectory, label)
                        target_plan = label["teacher_z_t"].to(
                            device=runtime.device, dtype=runtime.dtype
                        )
                        target_action = label["teacher_action"].to(
                            device=runtime.device, dtype=runtime.dtype
                        )
                        prediction_plan = _student_video_plan(runtime, context)
                        prediction_action = runtime.student_action_on_plan(
                            context,
                            target_plan.detach(),
                            require_grad=True,
                        )
                        teacher_valid_mask = label["teacher_action_valid_mask"].to(
                            device=runtime.device, dtype=torch.bool
                        )
                        if not torch.equal(
                            teacher_valid_mask, prediction_action.valid_mask
                        ):
                            raise NativeClosedLoopError(
                                f"paired action mask changed at macro {label['macro_id']}"
                            )
                        teacher_input_noise = label["teacher_action_input_noise"].to(
                            device=runtime.device, dtype=runtime.dtype
                        )
                        if not torch.equal(
                            teacher_input_noise,
                            prediction_action.action_input_noise.detach(),
                        ):
                            raise NativeClosedLoopError(
                                f"paired action noisy state changed at macro {label['macro_id']}"
                            )
                        teacher_timestep = label["teacher_action_timestep"].to(
                            device=runtime.device,
                            dtype=prediction_action.action_timestep.dtype,
                        )
                        if not torch.equal(
                            teacher_timestep,
                            prediction_action.action_timestep.detach(),
                        ):
                            raise NativeClosedLoopError(
                                f"paired action timestep changed at macro {label['macro_id']}"
                            )
                        video_mask = teacher_video_execution_mask(
                            prediction_plan, label
                        )
                        action_mask = teacher_action_execution_mask(
                            prediction_action.valid_mask, label
                        )
                        video_loss = video_huber_loss(
                            prediction_plan, target_plan.detach(), video_mask
                        )
                        action_loss = action_huber_loss(
                            prediction_action.endpoint,
                            target_action.detach(),
                            action_mask,
                        )
                        loss = (
                            float(video_weight) * video_loss
                            + float(action_weight) * action_loss
                        ) / len(window)
                        if not bool(torch.isfinite(loss).item()):
                            raise FloatingPointError("nonfinite paired joint OPD loss")
                        loss.backward()
                        video_sum += float(video_loss.detach().item())
                        action_sum += float(action_loss.detach().item())
                        macro_ids.append(int(label["macro_id"]))
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        parameters, max_norm=float(max_grad_norm)
                    )
                    if not bool(torch.isfinite(gradient_norm).item()):
                        raise FloatingPointError("nonfinite Joint LoRA gradient")
                    optimizer.step()
                    if not all(
                        bool(torch.isfinite(parameter).all().item())
                        for parameter in parameters
                    ):
                        raise FloatingPointError("nonfinite Joint LoRA parameter")
                    step += 1
                    row = {
                        "step": int(step),
                        "epoch": int(epoch + 1),
                        "macro_ids": macro_ids,
                        "actual_window_size": int(len(window)),
                        "video_loss_mean": video_sum / len(window),
                        "teacher_action_loss_mean": action_sum / len(window),
                        "gradient_norm_pre_clip": float(gradient_norm.item()),
                        "learning_rate": float(learning_rate),
                    }
                    rows.append(row)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()

        if runtime.base_parameter_hashes() != base_hashes_before:
            raise NativeClosedLoopError("frozen base parameters changed during Joint LoRA training")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": CHECKPOINT_SCHEMA,
                "training_started": True,
                "adapter_kind": "joint_lora",
                "adapter_state_dict": runtime.adapter_state(),
                "task": str(trajectory["task"]),
                "task_config": str(trajectory["task_config"]),
                "seed": int(trajectory["seed"]),
                "epochs": int(epochs),
                "window_size": int(window_size),
                "optimizer_steps": int(step),
                "video_weight": float(video_weight),
                "action_weight": float(action_weight),
                "action_plan_source": "detached_teacher_plan",
                "retention_weight": 0.0,
            },
            checkpoint_path,
        )
        return {
            "checkpoint": str(checkpoint_path),
            "optimizer_steps": int(step),
            "contexts": int(len(labels)),
            "epochs": int(epochs),
            "window_size": int(window_size),
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in parameters)
            ),
            "first_step": rows[0] if rows else None,
            "last_step": rows[-1] if rows else None,
            "action_plan_source": "detached_teacher_plan",
            "retention_weight": 0.0,
        }
    finally:
        runtime.close()


def _setup_task_with_locked_prompt(
    *,
    project_root: Path,
    task: str,
    seed: int,
    task_config: str,
    prompt: str,
) -> tuple[object, dict[str, Any], object, object, dict[str, Any], np.ndarray, str]:
    """Recreate the gate setup while keeping its exact released prompt."""

    robotwin_root = project_root / "third_party" / "RoboTwin-lingbot-native"
    sys.path[:0] = [
        str(project_root / "src"),
        str(project_root),
        str(robotwin_root),
        str(project_root / "third_party" / "lingbot-va"),
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
    from experiments.robotwin_sim_snapshot import (
        capture_simulator_snapshot,
        simulator_state_sha256,
    )

    install_enhanced_determinism()
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    task_args = build_task_args(robotwin_root, task, task_config)
    task_env = class_decorator(task)
    task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **task_args)
    _initialize_task_local_success_state(task_env, task)
    # The causal gate used ``--prompt`` (prompt_source=locked_cli).  Do not
    # call description generation here: it would consume RNG state before
    # the parent snapshot and make an otherwise identical setup hash differ.
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
    return (
        task_env,
        initial_observation,
        format_obs,
        add_init_pose,
        parent_snapshot,
        initial_eef_pose,
        simulator_state_sha256(parent_snapshot),
    )


def _worker_progress(
    *, worker: object, task_env: object, parent_snapshot: dict[str, Any], task: str
) -> dict[str, Any]:
    from experiments.robotwin_sim_snapshot import restore_simulator_snapshot
    from experiments.stage_h_task_progress import collect_task_progress

    end_snapshot = worker.snapshot()["snapshot"]
    restore_simulator_snapshot(task_env, end_snapshot)
    try:
        return collect_task_progress(task, task_env)
    finally:
        restore_simulator_snapshot(task_env, parent_snapshot)


def _outcome(
    episode: Mapping[str, Any], progress: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "success": bool(episode["success"]),
        "chunks_completed": int(episode["chunks_completed"]),
        "control_steps": int(episode["control_steps"]),
        "progress": dict(progress),
    }


def assert_common_noise(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    left_rows = list(left["chunks"])
    right_rows = list(right["chunks"])
    common = min(len(left_rows), len(right_rows))
    if common <= 0:
        raise NativeClosedLoopError("Teacher/eval episodes share no macro")
    for index in range(common):
        for key in ("video_base_noise_hash", "action_base_noise_hash"):
            if left_rows[index][key] != right_rows[index][key]:
                raise NativeClosedLoopError(
                    f"Teacher/eval raw noise differs at macro {index}: {key}"
                )
    return int(common)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("config must be a JSON object")

    workspace_root = Path(__file__).resolve().parents[1]
    artifact_root = Path(
        os.environ.get("WAM_OPD_ARTIFACT_ROOT", workspace_root / ".artifacts")
    )
    project_root = Path(
        config.get(
            "project_root",
            os.environ.get(
                "WAVE_RL_ROOT",
                os.environ.get("PROJECT_ROOT", workspace_root.parent / "wave-rl"),
            ),
        )
    ).expanduser().resolve()
    sys.path[:0] = [str(workspace_root), str(project_root / "src"), str(project_root)]
    task = str(config["task"])
    task_config = require_training_task_config(
        str(config.get("task_config", "demo_clean"))
    )
    seed = int(config["seed"])
    chunks = resolve_task_chunks(task, config.get("chunks"))
    max_control_steps = int(TASK_SPECS[task].max_control_steps)
    gate_path = Path(config["gate_json"]).expanduser().resolve()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_ss, gate_tt = validate_gate_contract(
        gate,
        task=task,
        task_config=task_config,
        seed=seed,
        chunks=chunks,
        max_control_steps=max_control_steps,
    )
    prompt = str(gate["prompt"])
    student = Path(
        config.get(
            "student",
            os.environ.get(
                "WAM_OPD_STUDENT_ROOT",
                artifact_root / "models" / "FlashWAM-RoboTwin",
            ),
        )
    ).expanduser().resolve()
    teacher = Path(
        config.get(
            "teacher_transformer",
            Path(
                os.environ.get(
                    "WAM_OPD_TEACHER_ROOT",
                    artifact_root / "models" / "lingbot-va-posttrain-robotwin",
                )
            )
            / "transformer",
        )
    ).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = str(config.get("device", "cuda:0"))
    enable_offload = bool(config.get("enable_offload", True))
    official_offload_parity = bool(config.get("official_offload_parity", True))
    trajectory_path = output_dir / "paired_teacher_trajectory.pt"
    checkpoint_path = output_dir / "joint_lora.pt"
    metrics_path = output_dir / "train_metrics.jsonl"
    summary_path = output_dir / "summary.json"

    (
        task_env,
        initial_observation,
        format_obs,
        add_init_pose,
        parent_snapshot,
        initial_eef_pose,
        parent_snapshot_hash,
    ) = _setup_task_with_locked_prompt(
        project_root=project_root,
        task=task,
        seed=seed,
        task_config=task_config,
        prompt=prompt,
    )
    if parent_snapshot_hash != str(gate["parent_snapshot_sha256"]):
        raise NativeClosedLoopError("fresh setup differs from gate parent snapshot")

    from experiments.robotwin_persistent_physics_worker import (
        PersistentNativePhysicsWorker,
    )

    workers: dict[str, Any] = {}
    collection_runtime: NativeV0VideoRuntime | None = None
    eval_runtime: NativeV0VideoRuntime | None = None
    try:
        # Fork physics workers before any model/CUDA initialization.
        for name in ("teacher", "eval"):
            workers[name] = PersistentNativePhysicsWorker(
                task_env=task_env,
                prompt=prompt,
                initial_eef_pose=initial_eef_pose,
                format_obs=None,
                materialize_renderer=None,
                worker_mode="parent_render_bridge",
            )

        collection_runtime = NativeV0VideoRuntime(
            student_checkpoint=student,
            teacher_transformer=teacher,
            device=device,
            save_root=output_dir / "native_save_collection",
            enable_offload=enable_offload,
            official_offload_parity=official_offload_parity,
            adapter_kind="none",
        )
        noise_bank = LockedNoiseBank(
            task=task,
            seed=seed,
            device=collection_runtime.device,
            dtype=collection_runtime.dtype,
        )
        targets: list[dict[str, Any]] = []
        teacher_episode = run_live_episode(
            runtime=collection_runtime,
            task_env=task_env,
            worker=workers["teacher"],
            parent_snapshot=parent_snapshot,
            initial_observation=initial_observation,
            initial_eef_pose=initial_eef_pose,
            format_obs=format_obs,
            add_init_pose=add_init_pose,
            task=task,
            task_config=task_config,
            seed=seed,
            prompt=prompt,
            arm="TT",
            chunks=chunks,
            noise_bank=noise_bank,
            macro_callback=lambda event: targets.append(capture_teacher_target(event)),
            stop_on_success=True,
            max_control_steps=max_control_steps,
            shared_noise_across_arms=True,
        )
        teacher_progress = _worker_progress(
            worker=workers["teacher"],
            task_env=task_env,
            parent_snapshot=parent_snapshot,
            task=task,
        )
        workers.pop("teacher").close()
        trajectory = build_teacher_trajectory(
            task=task,
            task_config=task_config,
            seed=seed,
            prompt=prompt,
            initial_observation=initial_observation,
            episode=teacher_episode,
            targets=targets,
        )
        torch.save(trajectory, trajectory_path)
        collection_runtime.close()
        collection_runtime = None
        del noise_bank
        torch.cuda.empty_cache()

        training = train_joint_adapter(
            trajectory_path=trajectory_path,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
            student=student,
            device=device,
            save_root=output_dir / "native_save_train",
            epochs=int(config.get("epochs", 8)),
            window_size=int(config.get("window_size", 4)),
            learning_rate=float(config.get("learning_rate", 1e-4)),
            video_weight=float(config.get("video_weight", 1.0)),
            action_weight=float(config.get("action_weight", 1.0)),
            max_grad_norm=float(config.get("max_grad_norm", 1.0)),
            enable_offload=enable_offload,
            official_offload_parity=official_offload_parity,
        )

        eval_runtime = NativeV0VideoRuntime(
            student_checkpoint=student,
            teacher_transformer=None,
            device=device,
            save_root=output_dir / "native_save_eval",
            enable_offload=enable_offload,
            official_offload_parity=official_offload_parity,
            adapter_rank=8,
            adapter_state=checkpoint_path,
            adapter_kind="joint_lora",
            lora_alpha=8.0,
            lora_dropout=0.0,
            lora_block_indices=(26, 27, 28, 29),
        )
        eval_noise_bank = LockedNoiseBank(
            task=task,
            seed=seed,
            device=eval_runtime.device,
            dtype=eval_runtime.dtype,
        )
        eval_episode = run_live_episode(
            runtime=eval_runtime,
            task_env=task_env,
            worker=workers["eval"],
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
            noise_bank=eval_noise_bank,
            stop_on_success=True,
            max_control_steps=max_control_steps,
            shared_noise_across_arms=True,
        )
        common_macros = assert_common_noise(teacher_episode, eval_episode)
        eval_progress = _worker_progress(
            worker=workers["eval"],
            task_env=task_env,
            parent_snapshot=parent_snapshot,
            task=task,
        )
        workers.pop("eval").close()
        eval_runtime.close()
        eval_runtime = None

        teacher_outcome = _outcome(teacher_episode, teacher_progress)
        eval_outcome = _outcome(eval_episode, eval_progress)
        baseline_progress = dict(gate_ss.get("progress", {}))
        summary = {
            "schema": "waopd_paired_teacher_joint_vertical_slice_v1",
            "task": task,
            "task_config": task_config,
            "seed": seed,
            "prompt": prompt,
            "chunks": chunks,
            "max_control_steps": max_control_steps,
            "gate_json": str(gate_path),
            "gate_ss_baseline": gate_ss,
            "gate_tt": gate_tt,
            "fresh_teacher_collection": teacher_outcome,
            "trained_teacher_free": eval_outcome,
            "same_noise_common_macros": int(common_macros),
            "training": training,
            "trajectory_path": str(trajectory_path),
            "checkpoint_path": str(checkpoint_path),
            "primary_outcome": {
                "native_success_rescue": bool(
                    eval_outcome["success"] and not bool(gate_ss["success"])
                ),
                "normalized_progress_delta": float(
                    eval_outcome["progress"].get("normalized_progress", 0.0)
                    - baseline_progress.get("normalized_progress", 0.0)
                ),
            },
            "paired_teacher_video_action": True,
            "action_plan_source": "detached_teacher_plan",
            "success_trigger_label_included": True,
            "success_post_label_count": 0,
            "retention_weight": 0.0,
            "randomized_eval_run": False,
            "teacher_loaded_during_eval": False,
            "stopped_after_first_checkpoint": True,
        }
        _write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    finally:
        if collection_runtime is not None:
            collection_runtime.close()
        if eval_runtime is not None:
            eval_runtime.close()
        for worker in workers.values():
            worker.close(force=True)
        task_env.close_env()


if __name__ == "__main__":
    raise SystemExit(main())
