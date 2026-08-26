"""Native action-endpoint OPD core used by the bounded D3 pilot.

Only the action output adapter is trainable.  The Student plan, history cache,
actual action noise, masks and normalization are reconstructed from the
schema-v4 native collection record; a record that cannot reproduce its
Student fingerprint is rejected before the action forward.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from experiments.action_output_adapter import (
    action_output_adapter_state_dict,
    attach_action_output_adapter,
)
from experiments.goal1_exact_condition import (
    ConditionContractError,
    ConditionFingerprint,
    build_condition_fingerprint,
    cache_valid_length,
    fingerprint_mismatches,
    grid_token_positions,
    prepare_plan_input,
    sequence_hash,
    stable_hash,
    tensor_hash,
)
from experiments.waopd_native_closed_loop_runner import (
    HistoryInput,
    _normalization_metadata,
)
from experiments.waopd_native_student_only import NativeStudentOnlyRuntime


def load_label(path: Path) -> dict[str, Any]:
    payload = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ConditionContractError(f"label is not a mapping: {path}")
    if payload.get("schema_version") != 4:
        raise ConditionContractError(f"unsupported D3 label schema: {path}")
    if payload.get("artifact_kind") != "waopd_d3_student_occupancy_teacher_bridge_label":
        raise ConditionContractError(f"D3 label kind mismatch: {path}")
    if payload.get("source_runtime") != "native_lingbot_not_rlinf_adaptation":
        raise ConditionContractError(f"D3 label is not native: {path}")
    if payload.get("training_started") is not False:
        raise ConditionContractError(f"D3 label has training contamination: {path}")
    required = (
        "canonical_action_context",
        "condition_fingerprint",
        "student_model_action",
        "teacher_bridge_model_action",
        "valid_action_mask",
        "replay_context_path",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ConditionContractError(f"D3 label missing {missing}: {path}")
    payload["_artifact_path"] = str(path.expanduser().resolve())
    return payload


def load_labels(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = [load_label(path) for path in paths]
    if not rows:
        raise ValueError("D3 training/evaluation partition is empty")
    checkpoints = {str(row["student_checkpoint"]) for row in rows}
    if len(checkpoints) != 1:
        raise ValueError(f"D3 labels use multiple Student checkpoints: {checkpoints}")
    return rows


def _as_history_hashes(history: list[HistoryInput]) -> tuple[str, str, str]:
    history_hash = sequence_hash(
        {
            "frame_st_id": row.frame_st_id,
            "latent": row.latent,
            "action": row.action,
        }
        for row in history
    )
    observation_hash = sequence_hash(
        {"frame_st_id": row.frame_st_id, "latent": row.latent}
        for row in history
    )
    model_action_history_hash = sequence_hash(
        {"frame_st_id": row.frame_st_id, "action": row.action}
        for row in history
    )
    return history_hash, observation_hash, model_action_history_hash


def _finite_stats(delta: torch.Tensor) -> dict[str, float]:
    flat = delta.detach().float()
    return {
        "max_abs": float(flat.abs().max().item()) if flat.numel() else 0.0,
        "mean_abs": float(flat.abs().mean().item()) if flat.numel() else 0.0,
        "rmse": float(flat.square().mean().sqrt().item()) if flat.numel() else 0.0,
    }


class NativeActionEndpointTrainer:
    """Rebuild one native Student condition and expose a differentiable endpoint."""

    def __init__(
        self,
        *,
        student: Path,
        device: str,
        save_root: Path,
        enable_offload: bool = True,
        adapter_state: Path | None = None,
        adapter_rank: int = 8,
    ) -> None:
        self.runtime = NativeStudentOnlyRuntime(
            student_checkpoint=student,
            device=device,
            save_root=save_root,
            enable_offload=enable_offload,
        )
        if adapter_state is None:
            attach_action_output_adapter(
                self.runtime.server.transformer,
                rank=int(adapter_rank),
                initialization="zero_up",
            )
        else:
            self.runtime.load_action_adapter(adapter_state, rank=int(adapter_rank))
        self.server = self.runtime.server
        self.trainable = [
            (str(name), parameter)
            for name, parameter in self.server.transformer.named_parameters()
            if parameter.requires_grad
        ]
        if not self.trainable:
            raise RuntimeError("native action trainer has no trainable adapter parameters")

    def close(self) -> None:
        del self.runtime
        torch.cuda.empty_cache()

    def parameter_hashes(self) -> dict[str, str]:
        return {
            name: str(tensor_hash(parameter))
            for name, parameter in self.trainable
        }

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return action_output_adapter_state_dict(self.server.transformer)

    def _rebuild_condition(self, artifact: dict[str, Any]) -> dict[str, Any]:
        context = torch.load(
            Path(str(artifact["replay_context_path"])).expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(context, dict) or context.get("schema") != "waopd_d3_student_occupancy_context_v1":
            raise ConditionContractError("D3 replay context schema mismatch")
        expected_context_metadata = {
            "task": str(artifact["task_id"]),
            "task_config": str(artifact["task_config"]),
            "seed": int(artifact["env_seed"]),
            "prompt": str(artifact["prompt"]),
            "collection_id": str(artifact["collection_id"]),
            "round_index": int(artifact["round_index"]),
            "context_chunks_used": int(artifact["context_chunks_used"]),
            "frame_st_id": int(artifact["frame_st_id"]),
        }
        actual_context_metadata = {
            "task": str(context.get("task")),
            "task_config": str(context.get("task_config")),
            "seed": int(context.get("seed", -1)),
            "prompt": str(context.get("prompt")),
            "collection_id": str(context.get("collection_id")),
            "round_index": int(context.get("round_index", -1)),
            "context_chunks_used": int(context.get("context_chunks_used", -1)),
            "frame_st_id": int(context.get("frame_st_id", -1)),
        }
        context_mismatches = {
            key: {
                "expected": expected_context_metadata[key],
                "actual": actual_context_metadata[key],
            }
            for key in expected_context_metadata
            if actual_context_metadata[key] != expected_context_metadata[key]
        }
        if context_mismatches:
            raise ConditionContractError(
                f"D3 replay context metadata mismatch: {context_mismatches}"
            )
        initial_observation = context["initial_observation"]
        initial_latent = self.runtime.reset(str(artifact["prompt"]), initial_observation)
        history: list[HistoryInput] = []
        frame_st_id = 0
        context_count = int(artifact["context_chunks_used"])
        chunks = list(context.get("chunks", []))[:context_count]
        with torch.inference_mode():
            for index, chunk in enumerate(chunks):
                if int(chunk["frame_st_id"]) != frame_st_id:
                    raise ConditionContractError(
                        f"D3 history is non-contiguous at {index}: expected {frame_st_id}, got {chunk['frame_st_id']}"
                    )
                with self.runtime._auxiliary_compute_scope(vae=True):
                    latent = self.server._encode_obs({"obs": chunk["observations"]})
                if frame_st_id == 0:
                    latent = torch.cat([initial_latent, latent], dim=2)
                action = self.server.preprocess_action(
                    np.asarray(chunk["env_action"])
                ).to(latent)
                model_input = self.server._prepare_latent_input(
                    latent,
                    action,
                    frame_st_id=frame_st_id,
                )
                self.server.transformer(
                    self.server._repeat_input_for_cfg(model_input["latent_res_lst"]),
                    update_cache=2,
                    cache_name=self.server.cache_name,
                    action_mode=False,
                )
                self.server.transformer(
                    self.server._repeat_input_for_cfg(model_input["action_res_lst"]),
                    update_cache=2,
                    cache_name=self.server.cache_name,
                    action_mode=True,
                )
                history.append(
                    HistoryInput(
                        frame_st_id=int(frame_st_id),
                        latent=latent.detach().clone(),
                        action=action.detach().clone(),
                        observations=list(chunk["observations"]),
                    )
                )
                frame_st_id += int(latent.shape[2])
        if frame_st_id != int(artifact["frame_st_id"]):
            raise ConditionContractError(
                f"D3 rebuilt frame_st_id={frame_st_id}, expected {artifact['frame_st_id']}"
            )
        self.server.frame_st_id = int(frame_st_id)

        canonical = artifact["canonical_action_context"]
        if not isinstance(canonical, dict):
            raise ConditionContractError("D3 canonical action context is not a mapping")
        prepared_plan = canonical["prepared_z_s"].to(
            device=self.server.device,
            dtype=self.server.dtype,
        )
        plan_input, plan_capture = prepare_plan_input(
            self.server,
            prepared_plan,
            frame_st_id=frame_st_id,
            already_prepared=True,
            latent_t=0,
        )
        if not torch.equal(plan_capture.prepared_z_s, prepared_plan):
            raise ConditionContractError("D3 native replay modified canonical prepared_z_s")
        expected_plan_hash = str(canonical["tensor_hashes"]["prepared_z_s"])
        if tensor_hash(prepared_plan) != expected_plan_hash:
            raise ConditionContractError("D3 canonical prepared_z_s tensor hash changed")
        with torch.inference_mode():
            self.server.transformer(
                self.server._repeat_input_for_cfg(plan_input["latent_res_lst"]),
                update_cache=1,
                cache_name=self.server.cache_name,
                action_mode=False,
            )

        action_noise = canonical["action_base_noise"].to(
            device=self.server.device,
            dtype=self.server.dtype,
        )
        self.server.action_scheduler.set_timesteps(1)
        action_timesteps = F.pad(
            self.server.action_scheduler.timesteps,
            (0, 1),
            mode="constant",
            value=0,
        )
        timestep = action_timesteps[0]
        action_cond = None
        if frame_st_id == 0:
            action_cond = torch.zeros(
                1,
                self.server.job_config.action_dim,
                1,
                self.server.action_per_frame,
                1,
                device=self.server.device,
                dtype=self.server.dtype,
            )
        action_input = self.server._prepare_latent_input(
            None,
            action_noise.clone(),
            timestep,
            timestep,
            None,
            action_cond,
            frame_st_id=frame_st_id,
        )
        actual_noise = action_input["action_res_lst"]["noisy_latents"]
        if not torch.equal(actual_noise, action_noise):
            raise ConditionContractError("D3 native replay modified actual action_base_noise")

        student_fp_payload = artifact["condition_fingerprint"].get("student")
        if not isinstance(student_fp_payload, dict):
            raise ConditionContractError("D3 label lacks Student fingerprint")
        expected_fp = ConditionFingerprint(
            **{
                **student_fp_payload,
                "token_positions": tuple(student_fp_payload.get("token_positions", ())),
            }
        )
        history_hash, observation_hash, model_action_history_hash = _as_history_hashes(history)
        actual_fp = build_condition_fingerprint(
            checkpoint_owner="student",
            history_hash=history_hash,
            prompt_hash=stable_hash(canonical.get("prompt_token_ids", [])),
            observation_hash=observation_hash,
            model_action_history_hash=model_action_history_hash,
            prepared_plan=prepared_plan,
            prepared_plan_timestep=plan_capture.prepared_z_s_timestep,
            action_base_noise=action_noise,
            action_timestep=action_input["action_res_lst"]["timesteps"],
            mask=canonical["valid_action_mask"].to(device=self.server.device),
            normalization_metadata=_normalization_metadata(self.server),
            frame_st_id=frame_st_id,
            token_positions=grid_token_positions(action_input["action_res_lst"], "action_res_lst"),
            cache_valid_length=cache_valid_length(self.server.transformer, self.server.cache_name),
            sigma_start=expected_fp.sigma_start,
            sigma_end=expected_fp.sigma_end,
        )
        mismatches = fingerprint_mismatches(expected_fp, actual_fp)
        if mismatches:
            raise ConditionContractError(f"D3 Student fingerprint mismatch: {mismatches}")
        return {
            "canonical": canonical,
            "frame_st_id": frame_st_id,
            "action_noise": action_noise,
            "action_input": action_input,
            "timestep": timestep,
            "action_cond": action_cond,
            "mask": canonical["valid_action_mask"].to(device=self.server.device).bool(),
            "history": history,
            "fingerprint": actual_fp,
        }

    def endpoint(
        self,
        artifact: dict[str, Any],
        *,
        require_grad: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        state = self._rebuild_condition(artifact)
        action_input = state["action_input"]
        if require_grad:
            output = self.server.transformer(
                self.server._repeat_input_for_cfg(action_input["action_res_lst"]),
                update_cache=0,
                cache_name=self.server.cache_name,
                action_mode=True,
            )
        else:
            with torch.inference_mode():
                output = self.server.transformer(
                    self.server._repeat_input_for_cfg(action_input["action_res_lst"]),
                    update_cache=0,
                    cache_name=self.server.cache_name,
                    action_mode=True,
                )
        velocity = rearrange(
            output,
            "b (f n) c -> b c f n 1",
            f=self.server.job_config.frame_chunk_size,
        )[:1]
        endpoint = self.server.action_scheduler.step(
            velocity,
            state["timestep"],
            state["action_noise"],
        )
        endpoint = endpoint.clone()
        if state["action_cond"] is not None:
            endpoint[:, :, 0:1] = state["action_cond"][:, :, 0:1]
        endpoint[:, ~self.server.action_mask] *= 0
        return endpoint, state

    @staticmethod
    def loss(
        endpoint: torch.Tensor,
        artifact: dict[str, Any],
        *,
        target_key: str = "teacher_bridge_model_action",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = artifact[target_key].to(device=endpoint.device, dtype=endpoint.dtype)
        mask = artifact["valid_action_mask"].to(device=endpoint.device).bool()
        if int(mask.sum()) == 0:
            raise ConditionContractError("D3 action mask has no valid elements")
        prediction = endpoint.float()[mask]
        target_float = target.float()[mask]
        loss = F.smooth_l1_loss(prediction, target_float, beta=1e-3)
        rmse = (prediction - target_float).square().mean().sqrt()
        return loss, rmse

    def evaluate(self, artifacts: Iterable[dict[str, Any]]) -> dict[str, Any]:
        rows = []
        for artifact in artifacts:
            with torch.no_grad():
                endpoint, _state = self.endpoint(artifact, require_grad=False)
                loss, rmse = self.loss(endpoint, artifact)
            rows.append(
                {
                    "artifact": artifact["_artifact_path"],
                    "task": artifact["task_id"],
                    "seed": int(artifact["env_seed"]),
                    "context_chunks_used": int(artifact["context_chunks_used"]),
                    "loss": float(loss.item()),
                    "rmse": float(rmse.item()),
                    "endpoint": endpoint.detach().cpu(),
                }
            )
        mean_loss = sum(row["loss"] for row in rows) / max(len(rows), 1)
        mean_rmse = sum(row["rmse"] for row in rows) / max(len(rows), 1)
        return {"rows": rows, "mean_loss": mean_loss, "mean_rmse": mean_rmse}

    def zero_init_parity(
        self,
        artifacts: Iterable[dict[str, Any]],
        *,
        max_rows: int = 4,
        tolerance: float = 1e-3,
    ) -> dict[str, Any]:
        rows = []
        for artifact in list(artifacts)[:max_rows]:
            with torch.no_grad():
                endpoint, _state = self.endpoint(artifact, require_grad=False)
            expected = artifact["student_model_action"].to(
                device=endpoint.device,
                dtype=endpoint.dtype,
            )
            stats = _finite_stats(endpoint - expected)
            stats.update(
                {
                    "artifact": artifact["_artifact_path"],
                    "bitwise_equal": bool(torch.equal(endpoint, expected)),
                }
            )
            rows.append(stats)
        max_abs = max((row["max_abs"] for row in rows), default=float("inf"))
        return {
            "schema": "waopd_d3_native_zero_init_parity_v1",
            "checked": len(rows),
            "tolerance_max_abs": float(tolerance),
            "rows": rows,
            "pass": bool(rows) and max_abs <= float(tolerance),
            "max_abs": max_abs,
        }

    def train(
        self,
        artifacts: list[dict[str, Any]],
        *,
        steps: int,
        learning_rate: float,
        gradient_clip: float,
        metrics_path: Path,
        phase: str,
        start_step: int = 0,
        optimizer_state: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        optimizer = torch.optim.AdamW(
            [parameter for _name, parameter in self.trainable],
            lr=float(learning_rate),
            weight_decay=0.0,
        )
        if optimizer_state:
            optimizer.load_state_dict(dict(optimizer_state))
        self.last_optimizer_state: dict[str, Any] | None = None
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for local_step in range(int(steps)):
            global_step = int(start_step) + local_step + 1
            artifact = artifacts[local_step % len(artifacts)]
            optimizer.zero_grad(set_to_none=True)
            endpoint, _state = self.endpoint(artifact, require_grad=True)
            loss, rmse = self.loss(endpoint, artifact)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite D3 loss at {phase}/{global_step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _name, parameter in self.trainable],
                max_norm=float(gradient_clip),
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite D3 gradient at {phase}/{global_step}")
            optimizer.step()
            row = {
                "schema": "waopd_d3_train_metric_v1",
                "phase": phase,
                "step": global_step,
                "task": artifact["task_id"],
                "artifact": artifact["_artifact_path"],
                "loss": float(loss.detach().item()),
                "rmse": float(rmse.detach().item()),
                "gradient_norm": float(grad_norm.detach().item()),
                "optimizer_step": True,
                "teacher_called": False,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json_dumps(row) + "\n")
                handle.flush()
            rows.append(row)
        self.last_optimizer_state = optimizer.state_dict()
        return rows


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def teacher_correction_scale(artifacts: Iterable[dict[str, Any]]) -> float:
    values = []
    for artifact in artifacts:
        student = artifact["student_model_action"].float()
        teacher = artifact["teacher_bridge_model_action"].float()
        mask = artifact["valid_action_mask"].bool()
        values.append(float((teacher[mask] - student[mask]).abs().mean().item()))
    return sum(values) / max(len(values), 1)


def retention_anchor_drift(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    by_key = {(row["task"], row["seed"], row["context_chunks_used"]): row for row in baseline["rows"]}
    drifts = []
    for row in current["rows"]:
        key = (row["task"], row["seed"], row["context_chunks_used"])
        if key not in by_key:
            raise ValueError(f"retention baseline missing {key}")
        base_endpoint = by_key[key]["endpoint"].float()
        current_endpoint = row["endpoint"].float()
        mask = torch.ones_like(base_endpoint, dtype=torch.bool)
        drifts.append(float((base_endpoint[mask] - current_endpoint[mask]).square().mean().sqrt().item()))
    return {
        "sample_count": len(drifts),
        "mean_rmse": sum(drifts) / max(len(drifts), 1),
        "max_rmse": max(drifts, default=0.0),
        "rows": drifts,
    }


__all__ = [
    "NativeActionEndpointTrainer",
    "load_label",
    "load_labels",
    "teacher_correction_scale",
    "retention_anchor_drift",
]
