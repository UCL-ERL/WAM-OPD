"""One-task video-first OPD on a full Easy RoboTwin trajectory.

The environment always executes Student actions.  At every visited Student
macro state, the frozen LingBot-VA Teacher produces only a video-plan target
from the exact same public history and video noise.  A video-mode-only LoRA is
then trained on contiguous macro windows; the Student action branch remains
frozen and supplies an action-aware consequence loss.

This is intentionally one reusable vertical slice rather than another round-
specific trainer.  A single invocation collects the baseline trajectory,
runs an exact-common-noise TS oracle, trains one checkpoint, and performs one
teacher-free Easy evaluation before stopping.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch

from experiments.opd_task_specs import (
    TASK_SPECS,
    require_training_task_config,
    resolve_task_chunks,
)
from experiments.train_stage_a_action_opd import terminal_execution_mask
from experiments.waopd_native_closed_loop_runner import (
    HistoryInput,
    NativeClosedLoopError,
    run_live_episode,
)
from experiments.waopd_v0_video_opd import (
    NativeV0VideoRuntime,
    action_huber_loss,
    video_huber_loss,
)


SCHEMA = "waopd_video_trajectory_v1"
CONTEXT_SCHEMA = "waopd_video_trajectory_context_v1"
CHECKPOINT_SCHEMA = "waopd_video_trajectory_lora_v1"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone().cpu()


class NativeStudentVideoLabelRuntime(NativeV0VideoRuntime):
    """Execute SS while labeling its exact occupancy with Teacher video only."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.teacher is None:
            raise ValueError("video-label collection requires a Teacher transformer")
        self._label_active = False
        self._teacher_history_cache = "opd_teacher_video_history"
        self.teacher_video_labels: list[dict[str, Any]] = []

    def begin_label_episode(self) -> None:
        if self._label_active:
            raise RuntimeError("a video-label episode is already active")
        self.teacher_video_labels = []
        self._label_active = True

    def end_label_episode(self) -> list[dict[str, Any]]:
        if not self._label_active:
            raise RuntimeError("no video-label episode is active")
        self._label_active = False
        try:
            self.teacher.clear_cache(self._teacher_history_cache)
        except (KeyError, TypeError, AttributeError):
            pass
        return list(self.teacher_video_labels)

    def reset(self, prompt: str, initial_observation: dict[str, Any]) -> torch.Tensor:
        initial_latent = super().reset(prompt, initial_observation)
        if self._label_active:
            self._create_cache(self.teacher, self._teacher_history_cache)
        return initial_latent

    def _on_teacher_video_plan_label(
        self,
        *,
        frame_st_id: int,
        teacher_plan: Any,
        student_solve: Any,
        action_noise: torch.Tensor,
    ) -> None:
        """Extension hook while the exact Teacher plan is still cached.

        The video-only collector intentionally does nothing.  Joint labelers
        can query the Teacher action branch here without clearing or replacing
        the Teacher video prediction, which preserves the coherent
        ``a_T(h_S, z_T)`` condition.
        """

        del frame_st_id, teacher_plan, student_solve, action_noise

    def _student_plan(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        solve = super()._student_plan(**kwargs)
        if not self._label_active:
            return solve
        teacher_plan = self._teacher_video_plan(
            frame_st_id=int(kwargs["frame_st_id"]),
            initial_latent=kwargs["initial_latent"],
            video_noise=kwargs["video_noise"],
            cache_name=self._teacher_history_cache,
        )
        if tuple(teacher_plan.prepared_z_s.shape) != tuple(solve.plan.prepared_z_s.shape):
            raise NativeClosedLoopError("Teacher and Student video-plan shapes differ")
        frame_st_id = int(kwargs["frame_st_id"])
        self.teacher_video_labels.append(
            {
                "frame_st_id": frame_st_id,
                "epsilon_v": _cpu_tensor(kwargs["video_noise"]),
                "epsilon_a": _cpu_tensor(kwargs["action_noise"]),
                "student_z_s": _cpu_tensor(solve.plan.prepared_z_s),
                "teacher_z_t": _cpu_tensor(teacher_plan.prepared_z_s),
                "teacher_z_t_timestep": _cpu_tensor(
                    teacher_plan.prepared_z_s_timestep
                ),
            }
        )
        self._on_teacher_video_plan_label(
            frame_st_id=frame_st_id,
            teacher_plan=teacher_plan,
            student_solve=solve,
            action_noise=kwargs["action_noise"],
        )
        return solve

    def append_student_history(
        self,
        *,
        record: HistoryInput,
        cache_name: str = "pos",
    ) -> int:
        next_frame = super().append_student_history(record=record, cache_name=cache_name)
        if not self._label_active:
            return next_frame
        self.teacher.clear_pred_cache(self._teacher_history_cache)
        model_input = self.server._prepare_latent_input(
            record.latent.to(device=self.device, dtype=self.dtype).clone(),
            record.action.to(device=self.device, dtype=self.dtype).clone(),
            frame_st_id=int(record.frame_st_id),
        )
        with torch.inference_mode():
            self.teacher(
                self.server._repeat_input_for_cfg(model_input["latent_res_lst"]),
                update_cache=2,
                cache_name=self._teacher_history_cache,
                action_mode=False,
            )
            self.teacher(
                self.server._repeat_input_for_cfg(model_input["action_res_lst"]),
                update_cache=2,
                cache_name=self._teacher_history_cache,
                action_mode=True,
            )
        return next_frame


def capture_student_context(event: dict[str, Any]) -> dict[str, Any]:
    """Keep only per-macro tensors; semantic history is stored once per episode."""

    solve = event["solve"]
    return {
        "macro_id": int(event["chunk_id"]),
        "frame_st_id": int(event["frame_st_id"]),
        "prompt_token_ids": tuple(int(value) for value in event["prompt_token_ids"]),
        "initial_latent": _cpu_tensor(event["initial_latent"]),
        "epsilon_v": _cpu_tensor(event["video_base_noise"]),
        "epsilon_a": _cpu_tensor(event["action_base_noise"]),
        "student_z_s": _cpu_tensor(event["plan"].prepared_z_s),
        "student_model_action": _cpu_tensor(solve.model_action),
        "student_valid_action_mask": _cpu_tensor(solve.mask),
    }


def build_trajectory_artifact(
    *,
    task: str,
    task_config: str,
    seed: int,
    prompt: str,
    initial_observation: dict[str, Any],
    episode: dict[str, Any],
    events: list[dict[str, Any]],
    teacher_labels: list[dict[str, Any]],
) -> dict[str, Any]:
    require_training_task_config(task_config)
    physical_rows = list(episode["chunks"])
    if not (len(events) == len(teacher_labels) == len(physical_rows)):
        raise NativeClosedLoopError(
            "trajectory event/Teacher-label/physical-row counts differ: "
            f"{len(events)}/{len(teacher_labels)}/{len(physical_rows)}"
        )
    labels: list[dict[str, Any]] = []
    cumulative_steps = 0
    for index, (event, teacher, physical) in enumerate(
        zip(events, teacher_labels, physical_rows, strict=True)
    ):
        macro_ids = {
            index,
            int(event["macro_id"]),
            int(physical["chunk_id"]),
        }
        frame_ids = {
            int(event["frame_st_id"]),
            int(teacher["frame_st_id"]),
            int(physical["frame_st_id"]),
        }
        if len(macro_ids) != 1 or len(frame_ids) != 1:
            raise NativeClosedLoopError("trajectory macro/frame chronology mismatch")
        if not torch.equal(event["epsilon_v"], teacher["epsilon_v"]):
            raise NativeClosedLoopError("Teacher video label used different raw video noise")
        if not torch.equal(event["epsilon_a"], teacher["epsilon_a"]):
            raise NativeClosedLoopError("Teacher label used different raw action noise")
        cumulative_steps += int(physical["action_steps"])
        labels.append(
            {
                **teacher,
                **event,
                "history_prefix_length": int(index),
                "start_frame": int(physical["start_frame"]),
                "action_steps": int(physical["action_steps"]),
                "executed_action_mask": physical["executed_action_mask"],
                "terminal_reached": bool(physical["terminal_reached"]),
                "terminal_action_position": physical["terminal_action_position"],
                "horizon_reached": bool(physical.get("horizon_reached", False)),
                "cumulative_control_steps": int(cumulative_steps),
            }
        )
    if not labels:
        raise NativeClosedLoopError("Student trajectory produced no trainable macro")
    return {
        "schema": SCHEMA,
        "task": str(task),
        "task_config": str(task_config),
        "seed": int(seed),
        "prompt": str(prompt),
        "initial_observation": deepcopy(initial_observation),
        "initial_latent": labels[0]["initial_latent"],
        "prompt_token_ids": labels[0]["prompt_token_ids"],
        "history": episode["history"],
        "labels": labels,
        "baseline_episode": episode,
        "success_post_label_count": 0,
    }


def materialize_context(
    trajectory: Mapping[str, Any], label: Mapping[str, Any]
) -> dict[str, Any]:
    prefix_length = int(label["history_prefix_length"])
    history = list(trajectory["history"])
    if prefix_length < 0 or prefix_length > len(history):
        raise ValueError("history prefix length is outside the trajectory")
    task_config = require_training_task_config(str(trajectory["task_config"]))
    return {
        "schema": CONTEXT_SCHEMA,
        "split": "train",
        "unit_id": f"{trajectory['task']}_{trajectory['seed']}",
        "context_id": (
            f"{trajectory['task']}_{trajectory['seed']}_macro"
            f"{int(label['macro_id']):02d}"
        ),
        "task": str(trajectory["task"]),
        "task_config": task_config,
        "seed": int(trajectory["seed"]),
        "prompt": str(trajectory["prompt"]),
        "prompt_token_ids": tuple(int(value) for value in trajectory["prompt_token_ids"]),
        "frame_st_id": int(label["frame_st_id"]),
        "macro_id": int(label["macro_id"]),
        "initial_observation": deepcopy(trajectory["initial_observation"]),
        "initial_latent": trajectory["initial_latent"],
        "history": history[:prefix_length],
        "epsilon_v": label["epsilon_v"],
        "epsilon_a": label["epsilon_a"],
        "prepared_z_s": label["student_z_s"],
        "student_model_action": label["student_model_action"],
        "_v0_target_initial_latent_mode": "frozen_saved_initial_latent_v1",
    }


def video_execution_mask(
    reference: torch.Tensor,
    label: Mapping[str, Any],
) -> torch.Tensor:
    if reference.ndim != 5 or reference.shape[0] != 1:
        raise ValueError("video plan must have shape [1, channels, frames, height, width]")
    executed = terminal_execution_mask(
        label,
        frame_count=int(reference.shape[2]),
        horizon=int(label["student_model_action"].shape[3]),
    ).to(device=reference.device)
    frame_mask = executed.any(dim=1)
    mask = torch.ones_like(reference, dtype=torch.bool)
    mask &= frame_mask[None, None, :, None, None]
    if int(label["frame_st_id"]) == 0:
        mask[:, :, 0:1] = False
    if not bool(mask.any()):
        raise ValueError("terminal-aware video loss mask is empty")
    return mask


def action_execution_mask(
    valid_mask: torch.Tensor,
    label: Mapping[str, Any],
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
        raise ValueError("terminal-aware action loss mask is empty")
    return mask


def _trajectory_windows(
    labels: list[dict[str, Any]], *, window_size: int, epoch: int, seed: int
) -> list[list[dict[str, Any]]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    windows = [
        labels[start : start + window_size]
        for start in range(0, len(labels), window_size)
    ]
    random.Random(int(seed) + int(epoch)).shuffle(windows)
    return windows


def train_video_adapter(
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
    if not isinstance(trajectory, dict) or trajectory.get("schema") != SCHEMA:
        raise ValueError("unexpected video trajectory schema")
    require_training_task_config(str(trajectory["task_config"]))
    labels = list(trajectory["labels"])
    if not labels:
        raise ValueError("trajectory has no labels")
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")

    runtime = NativeV0VideoRuntime(
        student_checkpoint=student,
        teacher_transformer=None,
        device=device,
        save_root=save_root,
        enable_offload=enable_offload,
        official_offload_parity=official_offload_parity,
        adapter_rank=8,
        adapter_kind="video_lora",
        lora_alpha=8.0,
        lora_dropout=0.0,
        lora_block_indices=(26, 27, 28, 29),
    )
    parameters = [parameter for _, parameter in runtime.trainable]
    optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=0.0)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    rows: list[dict[str, Any]] = []
    try:
        with metrics_path.open("w", encoding="utf-8") as handle:
            for epoch in range(int(epochs)):
                windows = _trajectory_windows(
                    labels,
                    window_size=int(window_size),
                    epoch=epoch,
                    seed=int(trajectory["seed"]),
                )
                for window in windows:
                    optimizer.zero_grad(set_to_none=True)
                    window_video = 0.0
                    window_action = 0.0
                    macro_ids: list[int] = []
                    for label in window:
                        context = materialize_context(trajectory, label)
                        teacher_plan = label["teacher_z_t"].to(
                            device=runtime.device, dtype=runtime.dtype
                        )
                        # The video-mode LoRA also changes the Student history
                        # and plan cache consumed by the frozen action branch.
                        # Recompute this stop-gradient target at current theta;
                        # reusing a zero-step target would make the condition stale.
                        target_action = runtime.student_action_on_plan(
                            context,
                            teacher_plan,
                            require_grad=False,
                        )
                        forward = runtime.student_video_forward(context)
                        if not torch.equal(
                            forward.action.valid_mask,
                            target_action.valid_mask,
                        ):
                            raise NativeClosedLoopError(
                                "Student/Teacher-plan action masks differ"
                            )
                        video_mask = video_execution_mask(
                            forward.plan.prepared_z_s,
                            label,
                        )
                        action_mask = action_execution_mask(
                            forward.action.valid_mask,
                            label,
                        )
                        video_loss = video_huber_loss(
                            forward.plan.prepared_z_s,
                            teacher_plan,
                            video_mask,
                        )
                        action_loss = action_huber_loss(
                            forward.action.endpoint,
                            target_action.endpoint.detach(),
                            action_mask,
                        )
                        loss = (
                            float(video_weight) * video_loss
                            + float(action_weight) * action_loss
                        ) / len(window)
                        if not bool(torch.isfinite(loss).item()):
                            raise FloatingPointError("nonfinite video OPD loss")
                        loss.backward()
                        window_video += float(video_loss.detach().item())
                        window_action += float(action_loss.detach().item())
                        macro_ids.append(int(label["macro_id"]))
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        parameters,
                        max_norm=float(max_grad_norm),
                    )
                    if not bool(torch.isfinite(gradient_norm).item()):
                        raise FloatingPointError("nonfinite video LoRA gradient")
                    optimizer.step()
                    if not all(bool(torch.isfinite(parameter).all().item()) for parameter in parameters):
                        raise FloatingPointError("nonfinite video LoRA parameter")
                    step += 1
                    row = {
                        "step": int(step),
                        "epoch": int(epoch + 1),
                        "macro_ids": macro_ids,
                        "actual_window_size": int(len(window)),
                        "video_loss_mean": window_video / len(window),
                        "action_aware_loss_mean": window_action / len(window),
                        "gradient_norm_pre_clip": float(gradient_norm.item()),
                        "learning_rate": float(learning_rate),
                    }
                    rows.append(row)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": CHECKPOINT_SCHEMA,
                "adapter_kind": "video_lora",
                "adapter_state_dict": runtime.adapter_state(),
                "task": str(trajectory["task"]),
                "task_config": str(trajectory["task_config"]),
                "seed": int(trajectory["seed"]),
                "epochs": int(epochs),
                "window_size": int(window_size),
                "optimizer_steps": int(step),
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
            "trainable_parameters": int(sum(parameter.numel() for parameter in parameters)),
            "first_step": rows[0] if rows else None,
            "last_step": rows[-1] if rows else None,
            "retention_weight": 0.0,
        }
    finally:
        runtime.close()


def _worker_progress(
    *,
    worker: object,
    task_env: object,
    parent_snapshot: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    from experiments.robotwin_sim_snapshot import restore_simulator_snapshot
    from experiments.stage_h_task_progress import collect_task_progress

    response = worker.snapshot()
    end_snapshot = response["snapshot"]
    restore_simulator_snapshot(task_env, end_snapshot)
    try:
        return collect_task_progress(task, task_env)
    finally:
        restore_simulator_snapshot(task_env, parent_snapshot)


def _outcome(episode: Mapping[str, Any], progress: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "success": bool(episode["success"]),
        "chunks_completed": int(episode["chunks_completed"]),
        "control_steps": int(
            sum(int(row["action_steps"]) for row in episode["chunks"])
        ),
        "progress": dict(progress),
    }


def assert_common_noise(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> int:
    left_rows = list(left["chunks"])
    right_rows = list(right["chunks"])
    common = min(len(left_rows), len(right_rows))
    if common == 0:
        raise NativeClosedLoopError("paired episodes share no macro")
    for index in range(common):
        for key in ("video_base_noise_hash", "action_base_noise_hash"):
            if left_rows[index][key] != right_rows[index][key]:
                raise NativeClosedLoopError(
                    f"paired raw noise differs at macro {index}: {key}"
                )
    return int(common)


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("config must be a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)

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
    robotwin_root = project_root / "third_party" / "RoboTwin-lingbot-native"
    sys.path[:0] = [
        str(workspace_root),
        str(project_root / "src"),
        str(project_root),
        str(robotwin_root),
        str(project_root / "third_party" / "lingbot-va"),
    ]
    os.environ["ROBOTWIN_ROOT"] = str(robotwin_root)

    task = str(config["task"])
    task_config = require_training_task_config(str(config.get("task_config", "demo_clean")))
    seed = int(config["seed"])
    chunks = resolve_task_chunks(task)
    try:
        max_control_steps = int(TASK_SPECS[task].max_control_steps)
    except KeyError as exc:
        raise ValueError(f"missing native control cap for {task!r}") from exc
    device = str(config.get("device", "cuda:0"))
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
    trajectory_path = output_dir / "student_video_trajectory.pt"
    checkpoint_path = output_dir / "video_lora.pt"
    metrics_path = output_dir / "train_metrics.jsonl"
    summary_path = output_dir / "summary.json"
    enable_offload = bool(config.get("enable_offload", True))
    official_offload_parity = bool(config.get("official_offload_parity", True))
    run_oracle = bool(config.get("run_ts_oracle", True))

    from experiments.waopd_v0_video_collection import (
        V0CommonNoiseBank,
        _setup_task,
    )

    (
        task_env,
        initial_observation,
        format_obs,
        add_init_pose,
        parent_snapshot,
        initial_eef_pose,
        setup_meta,
        _task,
    ) = _setup_task(
        project_root=project_root,
        task=task,
        seed=seed,
        task_config=task_config,
    )
    prompt = str(setup_meta["prompt"])

    from experiments.robotwin_persistent_physics_worker import (
        PersistentNativePhysicsWorker,
    )

    # Every child is forked before the first model/CUDA initialization.
    worker_names = ["baseline", "eval"] + (["oracle"] if run_oracle else [])
    workers: dict[str, Any] = {}
    collection_runtime: NativeStudentVideoLabelRuntime | None = None
    eval_runtime: NativeV0VideoRuntime | None = None
    try:
        for name in worker_names:
            workers[name] = PersistentNativePhysicsWorker(
                task_env=task_env,
                prompt=prompt,
                initial_eef_pose=initial_eef_pose,
                format_obs=None,
                materialize_renderer=None,
                worker_mode="parent_render_bridge",
            )

        collection_runtime = NativeStudentVideoLabelRuntime(
            student_checkpoint=student,
            teacher_transformer=teacher,
            device=device,
            save_root=output_dir / "native_save_collection",
            enable_offload=enable_offload,
            official_offload_parity=official_offload_parity,
            adapter_rank=8,
            adapter_kind="video_lora",
            lora_alpha=8.0,
            lora_dropout=0.0,
            lora_block_indices=(26, 27, 28, 29),
        )
        protocol_id = str(config.get("noise_protocol_id", "video_trajectory_v1"))
        unit_id = f"{task}_{task_config}_{seed}"
        noise_bank = V0CommonNoiseBank(
            protocol_id=protocol_id,
            unit_id=unit_id,
            task=task,
            seed=seed,
            device=collection_runtime.device,
            dtype=collection_runtime.dtype,
            schema="waopd_video_trajectory_common_noise_v1",
        )
        context_events: list[dict[str, Any]] = []
        collection_runtime.begin_label_episode()
        baseline_episode = run_live_episode(
            runtime=collection_runtime,
            task_env=task_env,
            worker=workers["baseline"],
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
            noise_bank=noise_bank,
            macro_callback=lambda event: context_events.append(
                capture_student_context(event)
            ),
            stop_on_success=True,
            max_control_steps=max_control_steps,
        )
        teacher_labels = collection_runtime.end_label_episode()
        baseline_progress = _worker_progress(
            worker=workers["baseline"],
            task_env=task_env,
            parent_snapshot=parent_snapshot,
            task=task,
        )
        workers.pop("baseline").close()

        oracle_episode = None
        oracle_progress = None
        oracle_common_macros = None
        if run_oracle:
            oracle_episode = run_live_episode(
                runtime=collection_runtime,
                task_env=task_env,
                worker=workers["oracle"],
                parent_snapshot=parent_snapshot,
                initial_observation=initial_observation,
                initial_eef_pose=initial_eef_pose,
                format_obs=format_obs,
                add_init_pose=add_init_pose,
                task=task,
                task_config=task_config,
                seed=seed,
                prompt=prompt,
                arm="TS",
                chunks=chunks,
                noise_bank=noise_bank,
                stop_on_success=True,
                max_control_steps=max_control_steps,
            )
            oracle_common_macros = assert_common_noise(
                baseline_episode,
                oracle_episode,
            )
            oracle_progress = _worker_progress(
                worker=workers["oracle"],
                task_env=task_env,
                parent_snapshot=parent_snapshot,
                task=task,
            )
            workers.pop("oracle").close()

        trajectory = build_trajectory_artifact(
            task=task,
            task_config=task_config,
            seed=seed,
            prompt=prompt,
            initial_observation=initial_observation,
            episode=baseline_episode,
            events=context_events,
            teacher_labels=teacher_labels,
        )
        torch.save(trajectory, trajectory_path)
        collection_runtime.close()
        collection_runtime = None
        del noise_bank
        torch.cuda.empty_cache()

        train_summary = train_video_adapter(
            trajectory_path=trajectory_path,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
            student=student,
            device=device,
            save_root=output_dir / "native_save_train",
            epochs=int(config.get("epochs", 3)),
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
            adapter_kind="video_lora",
            lora_alpha=8.0,
            lora_dropout=0.0,
            lora_block_indices=(26, 27, 28, 29),
        )
        eval_noise_bank = V0CommonNoiseBank(
            protocol_id=protocol_id,
            unit_id=unit_id,
            task=task,
            seed=seed,
            device=eval_runtime.device,
            dtype=eval_runtime.dtype,
            schema="waopd_video_trajectory_common_noise_v1",
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
        )
        eval_common_macros = assert_common_noise(baseline_episode, eval_episode)
        eval_progress = _worker_progress(
            worker=workers["eval"],
            task_env=task_env,
            parent_snapshot=parent_snapshot,
            task=task,
        )
        workers.pop("eval").close()
        eval_runtime.close()
        eval_runtime = None

        baseline_outcome = _outcome(baseline_episode, baseline_progress)
        eval_outcome = _outcome(eval_episode, eval_progress)
        summary = {
            "schema": "waopd_video_trajectory_vertical_slice_v1",
            "task": task,
            "task_config": task_config,
            "seed": seed,
            "chunks": chunks,
            "max_control_steps": max_control_steps,
            "baseline": baseline_outcome,
            "ts_oracle": (
                _outcome(oracle_episode, oracle_progress)
                if oracle_episode is not None and oracle_progress is not None
                else None
            ),
            "trained_teacher_free": eval_outcome,
            "same_noise_common_macros": {
                "baseline_vs_ts": oracle_common_macros,
                "baseline_vs_trained": eval_common_macros,
            },
            "training": train_summary,
            "trajectory_path": str(trajectory_path),
            "checkpoint_path": str(checkpoint_path),
            "primary_outcome": {
                "native_success_rescue": bool(
                    eval_outcome["success"] and not baseline_outcome["success"]
                ),
                "normalized_progress_delta": float(
                    eval_outcome["progress"].get("normalized_progress", 0.0)
                    - baseline_outcome["progress"].get("normalized_progress", 0.0)
                ),
            },
            "retention_weight": 0.0,
            "randomized_eval_run": False,
            "stopped_after_first_round": True,
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
