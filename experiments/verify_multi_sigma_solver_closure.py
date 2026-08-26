"""Verify sigma=1 x0 algebra against the active FlashWAM 1v/1a solver."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

from experiments.joint_lora import JointLoRALinear
from experiments.train_iterative_on_policy_flow_opd import (
    OBJECTIVE_MULTI_SIGMA_X0,
    _normalize_config,
    _validate_label,
)
from experiments.train_video_trajectory_opd import (
    action_execution_mask,
    materialize_context,
    video_execution_mask,
)
from experiments.waopd_v0_video_opd import (
    NativeV0VideoRuntime,
    V0VideoSigmaForward,
    video_consistency_map,
)


def _comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        return {
            "equal": False,
            "shape_equal": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    delta = left.detach().float() - right.detach().float()
    finite = bool(torch.isfinite(delta).all().item())
    mismatch_count = int(torch.count_nonzero(left != right).item())
    return {
        "equal": bool(torch.equal(left, right)),
        "shape_equal": True,
        "finite": finite,
        "elements": int(left.numel()),
        "mismatch_count": mismatch_count,
        "max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "mean_abs": float(delta.abs().mean().item()) if delta.numel() else 0.0,
        "rmse": (
            float(delta.square().mean().sqrt().item()) if delta.numel() else 0.0
        ),
    }


def _bool_comparison(left: object, right: object) -> dict[str, Any]:
    equal = bool(left == right)
    return {"equal": equal, "left": left, "right": right}


def _module_call_counts(runtime: NativeV0VideoRuntime) -> dict[str, int]:
    return {
        str(name): int(module.forward_call_count)
        for name, module in runtime.server.transformer.named_modules()
        if isinstance(module, JointLoRALinear)
    }


def _all_modules_advanced(
    before: Mapping[str, int], after: Mapping[str, int]
) -> dict[str, Any]:
    missing = sorted(set(before) ^ set(after))
    not_advanced = sorted(
        name
        for name in set(before) & set(after)
        if int(after[name]) <= int(before[name])
    )
    return {
        "equal": not missing and not not_advanced,
        "module_count": int(len(after)),
        "missing_modules": missing,
        "not_advanced_modules": not_advanced,
        "total_call_delta": int(sum(after.values()) - sum(before.values())),
    }


def _video_prediction_closure_rows(
    sigma_video: V0VideoSigmaForward,
    deployed_plan: torch.Tensor,
    video_mask: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    """Check raw solver closure separately from the Karras training map."""

    consistency_oracle = video_consistency_map(
        sigma_video.noisy_state,
        deployed_plan,
        sigma=float(sigma_video.sigma),
    )
    return {
        "raw_deployment_endpoint": _comparison(
            sigma_video.x0_hat[video_mask],
            deployed_plan[video_mask],
        ),
        "consistency_oracle": _comparison(
            sigma_video.consistency_prediction[video_mask],
            consistency_oracle[video_mask],
        ),
    }


def _anchor_closure(
    runtime: NativeV0VideoRuntime,
    trajectory: Mapping[str, Any],
    label: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_label(label)
    context = materialize_context(trajectory, label)
    target_video = label["teacher_z_t"].to(
        device=runtime.device, dtype=runtime.dtype
    )
    target_action = label["teacher_action"].to(
        device=runtime.device, dtype=runtime.dtype
    )
    saved_plan = label["student_z_s"].to(
        device=runtime.device, dtype=runtime.dtype
    )

    with torch.no_grad():
        deployed_video = runtime.student_video_forward(
            context, detach_plan_for_action=True
        )
        calls_before_video_sigma = _module_call_counts(runtime)
        sigma_video = runtime.student_video_x0_at_sigma(
            context,
            target_video,
            sigma=1.0,
            require_grad=False,
        )
        calls_after_video_sigma = _module_call_counts(runtime)

        deployed_action = runtime.student_action_on_plan(
            context, saved_plan, require_grad=False
        )
        calls_before_action_sigma = _module_call_counts(runtime)
        sigma_action = runtime.student_action_x0_at_sigma(
            context,
            saved_plan,
            target_action,
            sigma=1.0,
            require_grad=False,
        )
        calls_after_action_sigma = _module_call_counts(runtime)

    video_mask = video_execution_mask(
        deployed_video.plan.prepared_z_s, label
    )
    deployed_action_mask = action_execution_mask(
        deployed_action.valid_mask, label
    )
    sigma_action_mask = action_execution_mask(sigma_action.valid_mask, label)
    video_rows = {
        "input_noise": _comparison(
            sigma_video.noisy_state[video_mask],
            deployed_video.raw_video_noise[video_mask],
        ),
        "artifact_student_plan_replay": _comparison(
            deployed_video.plan.prepared_z_s[video_mask],
            saved_plan[video_mask],
        ),
        **_video_prediction_closure_rows(
            sigma_video,
            deployed_video.plan.prepared_z_s,
            video_mask,
        ),
        "all_joint_lora_modules_called": _all_modules_advanced(
            calls_before_video_sigma, calls_after_video_sigma
        ),
    }
    action_rows = {
        "input_noise": _comparison(
            sigma_action.noisy_state, deployed_action.action_input_noise
        ),
        "timestep": _comparison(
            sigma_action.timestep, deployed_action.action_timestep
        ),
        "valid_mask": _comparison(
            sigma_action.valid_mask, deployed_action.valid_mask
        ),
        "token_positions": _bool_comparison(
            tuple(sigma_action.token_positions),
            tuple(deployed_action.token_positions),
        ),
        "cache_valid_length": _bool_comparison(
            int(sigma_action.cache_valid_length),
            int(deployed_action.cache_valid_length),
        ),
        "execution_mask": _comparison(
            sigma_action_mask, deployed_action_mask
        ),
        "endpoint": _comparison(
            sigma_action.x0_prediction[deployed_action_mask],
            deployed_action.endpoint[deployed_action_mask],
        ),
        "all_joint_lora_modules_called": _all_modules_advanced(
            calls_before_action_sigma, calls_after_action_sigma
        ),
    }
    resolved_sigmas = {
        "video": float(sigma_video.sigma),
        "action": float(sigma_action.sigma),
    }
    sigma_pass = all(
        math.isfinite(value) and value == 1.0
        for value in resolved_sigmas.values()
    )
    checks = [*video_rows.values(), *action_rows.values()]
    passed = sigma_pass and all(bool(row["equal"]) for row in checks)
    return {
        "macro_id": int(label["macro_id"]),
        "frame_st_id": int(label["frame_st_id"]),
        "terminal_reached": bool(label["terminal_reached"]),
        "resolved_sigmas": resolved_sigmas,
        "video_mask_elements": int(video_mask.sum().item()),
        "action_mask_elements": int(deployed_action_mask.sum().item()),
        "video": video_rows,
        "action": action_rows,
        "status": "PASS" if passed else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = _normalize_config(
        json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    )
    if config["objective"] != OBJECTIVE_MULTI_SIGMA_X0:
        raise ValueError("solver closure requires objective='multi_sigma_x0'")
    project_root = Path(config["project_root"]).expanduser().resolve()
    robotwin_root = project_root / "third_party" / "RoboTwin-lingbot-native"
    lingbot_root = project_root / "third_party" / "lingbot-va"
    workspace_root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [
        str(workspace_root),
        str(project_root / "src"),
        str(project_root),
        str(robotwin_root),
        str(lingbot_root),
    ]
    os.environ["ROBOTWIN_ROOT"] = str(robotwin_root)
    os.chdir(robotwin_root)

    trajectory = torch.load(
        args.trajectory.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(trajectory, Mapping):
        raise TypeError("trajectory artifact is not a mapping")
    labels = list(trajectory.get("labels", []))
    if not labels:
        raise ValueError("trajectory artifact has no labels")
    selected = [labels[0]]
    if len(labels) > 1:
        selected.append(labels[-1])

    np.random.seed(int(config["adapter_seed"]))
    torch.manual_seed(int(config["adapter_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["adapter_seed"]))
    runtime: NativeV0VideoRuntime | None = None
    try:
        runtime = NativeV0VideoRuntime(
            student_checkpoint=Path(config["student"]).expanduser().resolve(),
            teacher_transformer=None,
            device=str(config["device"]),
            save_root=args.output.expanduser().resolve().parent / "native_save",
            enable_offload=bool(config["enable_offload"]),
            official_offload_parity=bool(config["official_offload_parity"]),
            adapter_rank=int(config["adapter_rank"]),
            adapter_state=None,
            adapter_kind="joint_lora",
            lora_alpha=float(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            lora_block_indices=tuple(int(value) for value in config["lora_block_indices"]),
        )
        contract = runtime.adapter_contract()
        contract_checks = {
            "block_indices": _bool_comparison(
                contract.get("block_indices"), list(range(30))
            ),
            "rank": _bool_comparison(int(contract.get("rank", -1)), 8),
            "parameter_dtype": _bool_comparison(
                contract.get("parameter_dtype"), "torch.float32"
            ),
            "wrapped_modules": _bool_comparison(
                len(list(contract.get("modules", []))), 300
            ),
            "trainable_parameter_dtype": {
                "equal": all(
                    parameter.dtype == torch.float32
                    for _, parameter in runtime.trainable
                )
            },
            "zero_up_initialization": {
                "equal": all(
                    bool(torch.count_nonzero(parameter).item() == 0)
                    for name, parameter in runtime.trainable
                    if name.endswith(".lora_B")
                )
            },
        }
        base_hashes_before = runtime.base_parameter_hashes()
        anchors = [
            _anchor_closure(runtime, trajectory, label) for label in selected
        ]
        base_hashes_after = runtime.base_parameter_hashes()
        contract_checks["frozen_base_unchanged"] = _bool_comparison(
            base_hashes_after, base_hashes_before
        )
        passed = all(
            bool(row["equal"]) for row in contract_checks.values()
        ) and all(row["status"] == "PASS" for row in anchors)
        report = {
            "schema": "waopd_multi_sigma_solver_closure_v1",
            "status": "PASS" if passed else "FAIL",
            "trajectory": str(args.trajectory.expanduser().resolve()),
            "adapter_contract": contract,
            "contract_checks": contract_checks,
            "anchors": anchors,
        }
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0 if passed else 1
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
