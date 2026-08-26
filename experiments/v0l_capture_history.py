"""Capture exact Student public histories for Goal V0L.

This is a read-only diagnostic hook around the already released native SS
runner.  It does not construct a Teacher, execute Teacher actions, or update
parameters.  Video/telemetry recording is available as an explicit opt-in
through the existing V0K ``RecordingCollector``.  The runner's macro callback
fires immediately
before each Student action; at that point it exposes the exact formatted
public history, canonical Student plan, and actual noise tensors consumed by
the action branch.  The callback stores those values in an append-only
episode envelope for later official full-Teacher deferred replay.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

# Direct execution (`python /abs/path/experiments/v0l_capture_history.py`)
# otherwise puts only the experiments directory on sys.path.  Insert the
# workspace before importing the existing experiment modules; this changes no
# runtime/model behavior and keeps the batch launcher self-contained.
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from experiments.goal1_exact_condition import sequence_hash, stable_hash, tensor_hash
from experiments.opd_task_specs import (
    TRAIN_TASK_CONFIG,
    require_training_task_config,
    resolve_task_chunks,
)
from experiments.waopd_native_closed_loop_runner import (
    STUDENT_FAMILY,
    _derived_seed,
    _serialize_fingerprint,
)
from experiments.waopd_v0j_teacher_free_behavior import _sha256, run_one


def _observation_hash(observation: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        digest.update(str(key).encode("utf-8"))
        value = observation[key]
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
        elif isinstance(value, torch.Tensor):
            digest.update(str(tensor_hash(value)).encode("ascii"))
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _numpy_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _cpu_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().contiguous().cpu().clone()


def _copy_observation(value: dict[str, Any]) -> dict[str, Any]:
    # ``format_obs`` contains numpy camera/state arrays.  Keep the exact
    # formatted public representation, including dtype and camera ordering.
    return deepcopy(value)


@dataclass
class HistoryCapture:
    output_dir: Path
    task: str
    seed: int
    condition: str
    unit_id: str
    task_config: str = TRAIN_TASK_CONFIG
    contexts: list[dict[str, Any]] = field(default_factory=list)
    canonical_rows: list[dict[str, Any]] = field(default_factory=list)
    initial_observation: dict[str, Any] | None = None
    initial_evaluator_state: dict[str, Any] | None = None
    after_macro_evaluator_state: dict[int, dict[str, Any]] = field(default_factory=dict)
    observation_events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.task_config = require_training_task_config(self.task_config)

    def observation_callback(self, event: dict[str, Any]) -> None:
        observation = _copy_observation(event["observation"])
        if event.get("event") == "initial_observation":
            self.initial_observation = observation
            self.initial_evaluator_state = deepcopy(event.get("evaluator_state"))
        else:
            self.after_macro_evaluator_state[int(event["macro_index"])] = deepcopy(
                event.get("evaluator_state")
            )
        self.observation_events.append(
            {
                "event": str(event.get("event")),
                "macro_index": int(event.get("macro_index", 0)),
                "frame_st_id": int(event.get("frame_st_id", 0)),
                "control_step": int(event.get("control_step", 0)),
                "frame_index": event.get("frame_index"),
                "horizon_index": event.get("horizon_index"),
                "observation_hash": _observation_hash(observation),
                "observation_keys": sorted(observation),
                "evaluator_state": deepcopy(event.get("evaluator_state")),
                "task_success": bool(event.get("task_success", False)),
                "eval_success": bool(event.get("eval_success", False)),
            }
        )

    def _materialize_history(self, history: list[Any]) -> None:
        if len(history) < len(self.canonical_rows):
            raise RuntimeError(
                f"history shortened: captured={len(self.canonical_rows)} current={len(history)}"
            )
        for record in history[len(self.canonical_rows) :]:
            observations = [_copy_observation(item) for item in record.observations]
            row = {
                "frame_st_id": int(record.frame_st_id),
                "latent": _cpu_tensor(record.latent),
                "action": _cpu_tensor(record.action),
                "observations": observations,
                "observation_hashes": [_observation_hash(item) for item in observations],
                "latent_hash": tensor_hash(record.latent),
                "action_hash": tensor_hash(record.action),
                "cache_owner": f"student:{self.unit_id}:epoch0",
            }
            if row["latent"] is None or row["action"] is None:
                raise RuntimeError("native HistoryInput contained a missing tensor")
            self.canonical_rows.append(row)

    def macro_callback(self, event: dict[str, Any]) -> None:
        if str(event.get("arm")) != "SS":
            raise RuntimeError(f"V0L history capture expected SS, got {event.get('arm')!r}")
        if self.initial_observation is None:
            raise RuntimeError("initial public observation was not captured before macro")
        event_task_config = require_training_task_config(str(event["task_config"]))
        if event_task_config != self.task_config:
            raise RuntimeError(
                "capture task_config changed within an episode: "
                f"expected {self.task_config!r}, got {event_task_config!r}"
            )

        self._materialize_history(list(event.get("history", [])))
        macro_id = int(event["chunk_id"])
        frame_st_id = int(event["frame_st_id"])
        if frame_st_id < 0 or frame_st_id % 2:
            raise RuntimeError(f"native frame_st_id is not a non-negative even position: {frame_st_id}")
        plan = event["plan"]
        solve = event["solve"]
        epsilon_v = _cpu_tensor(event["video_base_noise"])
        epsilon_a = _cpu_tensor(event["action_base_noise"])
        prepared_z_s = _cpu_tensor(plan.prepared_z_s)
        raw_z_s = _cpu_tensor(plan.raw_z_s)
        prepared_timestep = _cpu_tensor(plan.prepared_z_s_timestep)
        action = _cpu_tensor(solve.model_action)
        action_noise = _cpu_tensor(solve.action_noise)
        action_timestep = _cpu_tensor(solve.action_timestep)
        mask = _cpu_tensor(solve.mask)
        if any(item is None for item in (epsilon_v, epsilon_a, prepared_z_s, raw_z_s, prepared_timestep, action, action_noise, action_timestep, mask)):
            raise RuntimeError("V0L macro event contained a missing canonical tensor")

        if macro_id == 0:
            evaluator_state = deepcopy(self.initial_evaluator_state)
        else:
            evaluator_state = deepcopy(self.after_macro_evaluator_state.get(macro_id - 1))
        context_id = f"v0l_{self.task}_{self.seed}_{self.condition}_macro{macro_id:03d}"
        history = self.canonical_rows[:macro_id]
        if len(history) != macro_id:
            raise RuntimeError(
                f"macro/history chronology mismatch: macro={macro_id}, rows={len(history)}"
            )
        history_hash = sequence_hash(
            {
                "frame_st_id": int(row["frame_st_id"]),
                "latent": row["latent"],
                "action": row["action"],
            }
            for row in history
        )
        observation_hash = sequence_hash(
            {
                "frame_st_id": int(row["frame_st_id"]),
                "observations": row["observation_hashes"],
            }
            for row in history
        )
        action_history_hash = sequence_hash(
            {"frame_st_id": int(row["frame_st_id"]), "action": row["action"]}
            for row in history
        )
        student_base_seed = 2026080401
        derived_video_seed = _derived_seed(
            task=self.task,
            seed=self.seed,
            family=STUDENT_FAMILY,
            chunk_id=macro_id,
            base_seed=student_base_seed,
        )
        # LockedNoiseBank draws video then action from one generator per macro;
        # this root seed is the authoritative derivation metadata, while the
        # saved tensors remain authoritative inputs.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derived_video_seed)
        context = {
            "schema": "waopd_goal_v0l_student_public_history_context_v1",
            "context_id": context_id,
            "unit_id": self.unit_id,
            "task": self.task,
            "task_config": event_task_config,
            "seed": int(self.seed),
            "condition": self.condition,
            "split": "v0l_occupancy",
            "macro_id": macro_id,
            "frame_st_id": frame_st_id,
            "macro_boundary": [1.0, 0.0],
            "prompt": str(event["prompt"]),
            "prompt_token_ids": tuple(int(x) for x in event["prompt_token_ids"]),
            "initial_observation": self.initial_observation,
            "initial_observation_hash": _observation_hash(self.initial_observation),
            "initial_latent": _cpu_tensor(event["initial_latent"]),
            "history": history,
            "history_hash": history_hash,
            "observation_history_hash": observation_hash,
            "model_action_history_hash": action_history_hash,
            "prepared_z_s": prepared_z_s,
            "prepared_z_s_hash": tensor_hash(prepared_z_s),
            "raw_z_s": raw_z_s,
            "raw_z_s_hash": tensor_hash(raw_z_s),
            "prepared_z_s_timestep": prepared_timestep,
            "prepared_z_s_timestep_hash": tensor_hash(prepared_timestep),
            "latent_cond_applied": bool(plan.latent_cond_applied),
            "latent_cond": _cpu_tensor(plan.latent_cond),
            "epsilon_v": epsilon_v,
            "epsilon_v_hash": tensor_hash(epsilon_v),
            "epsilon_a": epsilon_a,
            "epsilon_a_hash": tensor_hash(epsilon_a),
            "noise_contract": {
                "source": "LockedNoiseBank",
                "family": STUDENT_FAMILY,
                "base_seed": student_base_seed,
                "derived_root_seed": int(derived_video_seed),
                "tensor_authority": "saved_actual_tensor",
                "video_shape": list(epsilon_v.shape),
                "action_shape": list(epsilon_a.shape),
            },
            "student_action": action,
            "student_action_hash": tensor_hash(action),
            "student_action_shape": list(action.shape),
            "student_action_dtype": str(action.dtype),
            "student_action_finite": bool(torch.isfinite(action).all().item()),
            "student_env_action": np.asarray(solve.env_action).copy(),
            "student_env_action_hash": _numpy_hash(solve.env_action),
            "student_action_input_noise": action_noise,
            "student_action_input_noise_hash": tensor_hash(action_noise),
            "student_action_timestep": action_timestep,
            "student_action_timestep_hash": tensor_hash(action_timestep),
            "student_action_valid_mask": mask,
            "student_action_valid_mask_hash": tensor_hash(mask),
            "student_action_token_positions": [int(x) for x in solve.action_token_positions],
            "student_cache_valid_length": int(solve.cache_valid_length),
            "student_cache_owner": f"student:{self.unit_id}:epoch0",
            "student_fingerprint": _serialize_fingerprint(event.get("student_fingerprint")),
            "evaluator_state": evaluator_state,
            "evaluator_state_policy_input": False,
            "source_runtime": "native_lingbot_student_only_not_rlinf_adaptation",
            "goal_v0l_extended_frame_st_id": True,
            "teacher_forbidden_inputs_asserted": {
                "student_hidden_states": False,
                "student_kv_cache": False,
                "lora_activations": False,
                "physx_private_state": False,
                "future_observation": False,
                "expert_trajectory": False,
                "success_or_reward": False,
                "future_action": False,
            },
        }
        self.contexts.append(context)

    def finalize(self, result: dict[str, Any], *, output_path: Path) -> dict[str, Any]:
        snapshot_hash = result.get("initial_snapshot_sha256")
        checkpoint_hash = result.get("student_checkpoint_sha256")
        chunk_rows = result.get("chunks")
        if not isinstance(chunk_rows, list):
            raise RuntimeError("native runner result lacks per-chunk execution metadata")
        chunk_by_id = {
            int(row["chunk_id"]): row
            for row in chunk_rows
            if isinstance(row, dict) and "chunk_id" in row
        }
        terminal_macro_ids = [
            chunk_id
            for chunk_id, row in chunk_by_id.items()
            if bool(row.get("terminal_reached", False))
        ]
        if terminal_macro_ids:
            first_terminal_macro = min(terminal_macro_ids)
            post_success_contexts = [
                int(context["macro_id"])
                for context in self.contexts
                if int(context["macro_id"]) > first_terminal_macro
            ]
            if post_success_contexts:
                raise RuntimeError(
                    "training capture contains post-success context(s): "
                    f"terminal_macro={first_terminal_macro} "
                    f"post_success={post_success_contexts}"
                )
        for context in self.contexts:
            macro_id = int(context["macro_id"])
            if macro_id not in chunk_by_id:
                raise RuntimeError(f"native runner result lacks chunk {macro_id}")
            chunk = chunk_by_id[macro_id]
            required_execution_fields = (
                "start_frame",
                "action_steps",
                "executed_action_mask",
                "terminal_reached",
                "terminal_action_position",
            )
            missing = [key for key in required_execution_fields if key not in chunk]
            if missing:
                raise RuntimeError(
                    f"native runner chunk {macro_id} lacks execution fields: {missing}"
                )
            action_shape = np.asarray(context["student_env_action"]).shape
            expected_mask_shape = action_shape[1:]
            executed_action_mask = np.asarray(chunk["executed_action_mask"])
            if (
                executed_action_mask.dtype != np.bool_
                or executed_action_mask.shape != expected_mask_shape
            ):
                raise RuntimeError(
                    "executed_action_mask must be boolean and match model action temporal shape: "
                    f"expected={expected_mask_shape} actual={executed_action_mask.shape} "
                    f"dtype={executed_action_mask.dtype}"
                )
            context["start_frame"] = int(chunk["start_frame"])
            context["action_steps"] = int(chunk["action_steps"])
            context["executed_action_mask"] = executed_action_mask.tolist()
            context["terminal_reached"] = bool(chunk["terminal_reached"])
            context["terminal_action_position"] = deepcopy(
                chunk["terminal_action_position"]
            )
            context["initial_snapshot_sha256"] = snapshot_hash
            context["student_checkpoint"] = result.get("student_checkpoint")
            context["student_checkpoint_sha256"] = checkpoint_hash
            context["adapter_state"] = result.get("adapter_state")
            context["adapter_state_sha256"] = result.get("adapter_state_sha256")
            context["runtime_provenance"] = {
                "backend": result.get("backend"),
                "student_checkpoint_hash_source": result.get("student_checkpoint_hash_source"),
                "adapter_kind": result.get("adapter_kind"),
                "teacher_loaded": False,
                "teacher_called": False,
                "training_started": False,
                "episode_success": result.get("success"),
                "chunks_completed": result.get("chunks"),
            }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "waopd_goal_v0l_student_public_history_episode_v1",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task": self.task,
            "seed": int(self.seed),
            "condition": self.condition,
            "unit_id": self.unit_id,
            "task_config": self.task_config,
            "history_capture": {
                "source": "native_runner_macro_callback",
                "all_pre_action_macro_boundaries_saved": True,
                "video_recording": False,
                "teacher_loaded": False,
                "teacher_called": False,
                "teacher_action_executed": False,
                "optimizer_steps": 0,
                "observation_event_count": len(self.observation_events),
                "macro_count": len(self.contexts),
            },
            "observation_events": self.observation_events,
            "contexts": self.contexts,
        }
        torch.save(payload, output_path)
        return {
            "path": str(output_path.resolve()),
            "sha256": _sha256(output_path),
            "context_count": len(self.contexts),
            "macro_count": len(self.contexts),
            "observation_event_count": len(self.observation_events),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default=TRAIN_TASK_CONFIG)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument(
        "--chunks",
        type=int,
        help="macro horizon override; defaults to the registered task horizon",
    )
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--adapter-state", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--enable-offload", action="store_true")
    parser.add_argument("--official-offload-parity", action="store_true")
    parser.add_argument("--prompt")
    parser.add_argument("--continue-after-success", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    if args.continue_after_success:
        parser.error(
            "--continue-after-success is incompatible with training history capture"
        )
    task_config = require_training_task_config(args.task_config)
    chunks = resolve_task_chunks(str(args.task), args.chunks)

    if args.condition == "CLEAN_SS" and args.adapter_state is not None:
        raise ValueError("CLEAN_SS must not load an adapter")
    if args.condition != "CLEAN_SS" and args.adapter_state is None:
        raise ValueError(f"{args.condition} requires its frozen adapter checkpoint")

    output_dir = args.output_dir.expanduser().resolve()
    unit_id = f"{args.task}:{int(args.seed)}"
    stem = f"{args.task}_{int(args.seed)}_{args.condition}"
    result_path = output_dir / "episodes" / f"{stem}.json"
    context_path = output_dir / "history" / f"{stem}.pt"
    collector = HistoryCapture(
        output_dir=output_dir,
        task=str(args.task),
        seed=int(args.seed),
        condition=str(args.condition),
        unit_id=unit_id,
        task_config=task_config,
    )
    recording_collector = None
    if args.record_video:
        from experiments.v0k_native_video_diagnostic import RecordingCollector

        recording_collector = RecordingCollector(
            root=output_dir,
            task=str(args.task),
            seed=int(args.seed),
            condition=str(args.condition),
            enabled=True,
        )

    def macro_callback(event: dict[str, Any]) -> None:
        collector.macro_callback(event)
        if recording_collector is not None:
            recording_collector.macro_callback(event)

    def observation_callback(event: dict[str, Any]) -> None:
        collector.observation_callback(event)
        if recording_collector is not None:
            recording_collector.observation_callback(event)

    try:
        result = run_one(
            task=str(args.task),
            task_config=task_config,
            seed=int(args.seed),
            chunks=chunks,
            student=args.student,
            output=result_path,
            project_root=args.project_root,
            device=str(args.device),
            enable_offload=bool(args.enable_offload),
            official_offload_parity=bool(args.official_offload_parity),
            adapter_state=args.adapter_state,
            prompt_override=args.prompt,
            macro_callback=macro_callback,
            observation_callback=observation_callback,
            stop_on_success=not bool(args.continue_after_success),
        )
        capture = collector.finalize(result, output_path=context_path)
        recording = recording_collector.finalize(result) if recording_collector is not None else None
        summary = {
            "schema": "waopd_goal_v0l_student_public_history_capture_summary_v1",
            "status": "PASS",
            "task": str(args.task),
            "seed": int(args.seed),
            "condition": str(args.condition),
            "task_config": task_config,
            "chunks_requested": chunks,
            "stop_on_success": not bool(args.continue_after_success),
            "unit_id": unit_id,
            "result_path": str(result_path.resolve()),
            "result_sha256": _sha256(result_path),
            "context_capture": capture,
            "recording": recording,
            "student_checkpoint": result.get("student_checkpoint"),
            "student_checkpoint_sha256": result.get("student_checkpoint_sha256"),
            "adapter_state": result.get("adapter_state"),
            "adapter_state_sha256": result.get("adapter_state_sha256"),
            "initial_snapshot_sha256": result.get("initial_snapshot_sha256"),
            "success": result.get("success"),
            "teacher_called": False,
            "training_started": False,
        }
        summary_path = output_dir / "summaries" / f"{stem}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "schema": "waopd_goal_v0l_student_public_history_capture_summary_v1",
            "status": "BLOCKED",
            "task": str(args.task),
            "seed": int(args.seed),
            "condition": str(args.condition),
            "reason": f"{type(exc).__name__}: {exc}",
            "teacher_called": False,
            "training_started": False,
            "partial_context_count": len(collector.contexts),
        }
        if recording_collector is not None:
            try:
                failure["recording"] = recording_collector.finalize(
                    failure,
                    error=failure["reason"],
                )
            except Exception as recording_exc:
                failure["recording_error"] = (
                    f"{type(recording_exc).__name__}: {recording_exc}"
                )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(failure, sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
