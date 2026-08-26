"""Checkpoint screening and final exact-paired eval for success-path v1.

This evaluator is intentionally separate from training.  It reuses the
released teacher-free behavior runner, Stage-H task progress, the frozen
exact-pair validator, and the existing V0K RecordingCollector.  The helper
only adds orchestration plus a process-local, task-specific telemetry bridge;
it never changes policy inputs or training state.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROTOCOL_SCHEMA = "waopd_handover_mic_success_path_eval_protocol_v1"
UNIT_SCHEMA = "waopd_handover_mic_success_path_eval_unit_v1"
PAIR_SCHEMA = "waopd_handover_mic_success_path_progress_pair_v1"
SCREEN_SUMMARY_SCHEMA = "waopd_handover_mic_success_path_screen_summary_v1"
SELECTION_SCHEMA = "waopd_handover_mic_success_path_selection_v1"
HELDOUT_SUMMARY_SCHEMA = "waopd_handover_mic_success_path_heldout_summary_v1"
SCREEN_EPOCH_RECEIPT_SCHEMA = (
    "waopd_handover_mic_success_path_screen_epoch_receipt_v1"
)
OBSERVER_RECORDING_KEY = "recording.images.observer"
OBSERVER_STATE_KEY = "_recording_observer_rgb"
EXPECTED_SELECTION_RULE = (
    "adapted_success_count_desc",
    "sum_paired_max_ordinal_delta_desc",
    "stage_improvement_minus_regression_desc",
    "fixed_calibration_total_loss_asc",
    "epoch_asc",
)

TASK_EVAL_SPECS: dict[str, dict[str, Any]] = {
    "handover_mic": {
        "chunks": 20,
        "max_control_steps": 600,
        "protocol_schema": PROTOCOL_SCHEMA,
        "unit_schema": UNIT_SCHEMA,
        "pair_schema": PAIR_SCHEMA,
        "screen_summary_schema": SCREEN_SUMMARY_SCHEMA,
        "selection_schema": SELECTION_SCHEMA,
        "heldout_summary_schema": HELDOUT_SUMMARY_SCHEMA,
        "screen_epoch_receipt_schema": SCREEN_EPOCH_RECEIPT_SCHEMA,
    },
    "open_microwave": {
        # 1500 native controls = first 16-control chunk + 47 subsequent
        # 32-control chunks in the native runner.  Keeping the full horizon
        # here is important: a shorter legacy 16-chunk pilot cannot reach the
        # microwave terminal predicate and is not a success-path evaluation.
        "chunks": 48,
        "max_control_steps": 1500,
        "protocol_schema": "waopd_open_microwave_success_path_eval_protocol_v1",
        "unit_schema": "waopd_open_microwave_success_path_eval_unit_v1",
        "pair_schema": "waopd_open_microwave_success_path_progress_pair_v1",
        "screen_summary_schema": (
            "waopd_open_microwave_success_path_screen_summary_v1"
        ),
        "selection_schema": "waopd_open_microwave_success_path_selection_v1",
        "heldout_summary_schema": (
            "waopd_open_microwave_success_path_heldout_summary_v1"
        ),
        "screen_epoch_receipt_schema": (
            "waopd_open_microwave_success_path_screen_epoch_receipt_v1"
        ),
    },
    "put_object_cabinet": {
        "chunks": 23,
        "max_control_steps": 700,
        "protocol_schema": (
            "waopd_put_object_cabinet_success_path_eval_protocol_v1"
        ),
        "unit_schema": "waopd_put_object_cabinet_success_path_eval_unit_v1",
        "pair_schema": (
            "waopd_put_object_cabinet_success_path_progress_pair_v1"
        ),
        "screen_summary_schema": (
            "waopd_put_object_cabinet_success_path_screen_summary_v1"
        ),
        "selection_schema": (
            "waopd_put_object_cabinet_success_path_selection_v1"
        ),
        "heldout_summary_schema": (
            "waopd_put_object_cabinet_success_path_heldout_summary_v1"
        ),
        "screen_epoch_receipt_schema": (
            "waopd_put_object_cabinet_success_path_screen_epoch_receipt_v1"
        ),
    },
    "put_bottles_dustbin": {
        # 1700 native controls = the initial 16-control chunk followed by
        # 53 32-control chunks.  The complete 54-chunk horizon is required
        # because all three bottles must reach the official dustbin region.
        "chunks": 54,
        "max_control_steps": 1700,
        "protocol_schema": (
            "waopd_put_bottles_dustbin_success_path_eval_protocol_v1"
        ),
        "unit_schema": (
            "waopd_put_bottles_dustbin_success_path_eval_unit_v1"
        ),
        "pair_schema": (
            "waopd_put_bottles_dustbin_success_path_progress_pair_v1"
        ),
        "screen_summary_schema": (
            "waopd_put_bottles_dustbin_success_path_screen_summary_v1"
        ),
        "selection_schema": (
            "waopd_put_bottles_dustbin_success_path_selection_v1"
        ),
        "heldout_summary_schema": (
            "waopd_put_bottles_dustbin_success_path_heldout_summary_v1"
        ),
        "screen_epoch_receipt_schema": (
            "waopd_put_bottles_dustbin_success_path_screen_epoch_receipt_v1"
        ),
    },
    "place_fan": {
        # The native 400-control horizon consists of the initial 16-control
        # chunk and twelve subsequent 32-control chunks.
        "chunks": 13,
        "max_control_steps": 400,
        "protocol_schema": "waopd_place_fan_success_path_eval_protocol_v1",
        "unit_schema": "waopd_place_fan_success_path_eval_unit_v1",
        "pair_schema": "waopd_place_fan_success_path_progress_pair_v1",
        "screen_summary_schema": (
            "waopd_place_fan_success_path_screen_summary_v1"
        ),
        "selection_schema": "waopd_place_fan_success_path_selection_v1",
        "heldout_summary_schema": (
            "waopd_place_fan_success_path_heldout_summary_v1"
        ),
        "screen_epoch_receipt_schema": (
            "waopd_place_fan_success_path_screen_epoch_receipt_v1"
        ),
    },
    "place_shoe": {
        "chunks": 17,
        "max_control_steps": 500,
        "protocol_schema": "waopd_place_shoe_success_path_eval_protocol_v1",
        "unit_schema": "waopd_place_shoe_success_path_eval_unit_v1",
        "pair_schema": "waopd_place_shoe_success_path_progress_pair_v1",
        "screen_summary_schema": (
            "waopd_place_shoe_success_path_screen_summary_v1"
        ),
        "selection_schema": "waopd_place_shoe_success_path_selection_v1",
        "heldout_summary_schema": (
            "waopd_place_shoe_success_path_heldout_summary_v1"
        ),
        "screen_epoch_receipt_schema": (
            "waopd_place_shoe_success_path_screen_epoch_receipt_v1"
        ),
    },
    "scan_object": {
        "chunks": 17,
        "max_control_steps": 500,
        "protocol_schema": "waopd_scan_object_success_path_eval_protocol_v1",
        "unit_schema": "waopd_scan_object_success_path_eval_unit_v1",
        "pair_schema": "waopd_scan_object_success_path_progress_pair_v1",
        "screen_summary_schema": (
            "waopd_scan_object_success_path_screen_summary_v1"
        ),
        "selection_schema": "waopd_scan_object_success_path_selection_v1",
        "heldout_summary_schema": (
            "waopd_scan_object_success_path_heldout_summary_v1"
        ),
        "screen_epoch_receipt_schema": (
            "waopd_scan_object_success_path_screen_epoch_receipt_v1"
        ),
    },
    "blocks_ranking_size": {
        # Official RoboTwin _eval_step_limit.yml fixes this H3 task at 1200
        # native controls: one 16-control chunk plus 37 32-control chunks.
        "chunks": 38,
        "max_control_steps": 1200,
        "protocol_schema": (
            "waopd_blocks_ranking_size_success_path_eval_protocol_v1"
        ),
        "unit_schema": (
            "waopd_blocks_ranking_size_success_path_eval_unit_v1"
        ),
        "pair_schema": (
            "waopd_blocks_ranking_size_success_path_progress_pair_v1"
        ),
        "screen_summary_schema": (
            "waopd_blocks_ranking_size_success_path_screen_summary_v1"
        ),
        "selection_schema": (
            "waopd_blocks_ranking_size_success_path_selection_v1"
        ),
        "heldout_summary_schema": (
            "waopd_blocks_ranking_size_success_path_heldout_summary_v1"
        ),
        "screen_epoch_receipt_schema": (
            "waopd_blocks_ranking_size_success_path_screen_epoch_receipt_v1"
        ),
    },
    "blocks_ranking_rgb": {
        # The RGB and size variants share the exact official three-block
        # adjacent-ordering/release predicate and 1200-control horizon.
        "chunks": 38,
        "max_control_steps": 1200,
        "protocol_schema": (
            "waopd_blocks_ranking_rgb_success_path_eval_protocol_v1"
        ),
        "unit_schema": (
            "waopd_blocks_ranking_rgb_success_path_eval_unit_v1"
        ),
        "pair_schema": (
            "waopd_blocks_ranking_rgb_success_path_progress_pair_v1"
        ),
        "screen_summary_schema": (
            "waopd_blocks_ranking_rgb_success_path_screen_summary_v1"
        ),
        "selection_schema": (
            "waopd_blocks_ranking_rgb_success_path_selection_v1"
        ),
        "heldout_summary_schema": (
            "waopd_blocks_ranking_rgb_success_path_heldout_summary_v1"
        ),
        "screen_epoch_receipt_schema": (
            "waopd_blocks_ranking_rgb_success_path_screen_epoch_receipt_v1"
        ),
    },
    "place_a2b_left": {
        # Official RoboTwin _eval_step_limit.yml fixes this task at 400
        # native controls: one 16-control chunk plus twelve 32-control chunks.
        "chunks": 13,
        "max_control_steps": 400,
        "protocol_schema": "waopd_place_a2b_left_success_path_eval_protocol_v1",
        "unit_schema": "waopd_place_a2b_left_success_path_eval_unit_v1",
        "pair_schema": "waopd_place_a2b_left_success_path_progress_pair_v1",
        "screen_summary_schema": (
            "waopd_place_a2b_left_success_path_screen_summary_v1"
        ),
        "selection_schema": "waopd_place_a2b_left_success_path_selection_v1",
        "heldout_summary_schema": (
            "waopd_place_a2b_left_success_path_heldout_summary_v1"
        ),
        "screen_epoch_receipt_schema": (
            "waopd_place_a2b_left_success_path_screen_epoch_receipt_v1"
        ),
    },
    "stack_blocks_three": {
        # Official RoboTwin _eval_step_limit.yml fixes this task at 1200
        # native controls: one 16-control chunk plus 37 32-control chunks.
        "chunks": 38,
        "max_control_steps": 1200,
        "protocol_schema": (
            "waopd_stack_blocks_three_success_path_eval_protocol_v1"
        ),
        "unit_schema": "waopd_stack_blocks_three_success_path_eval_unit_v1",
        "pair_schema": (
            "waopd_stack_blocks_three_success_path_progress_pair_v1"
        ),
        "screen_summary_schema": (
            "waopd_stack_blocks_three_success_path_screen_summary_v1"
        ),
        "selection_schema": (
            "waopd_stack_blocks_three_success_path_selection_v1"
        ),
        "heldout_summary_schema": (
            "waopd_stack_blocks_three_success_path_heldout_summary_v1"
        ),
        "screen_epoch_receipt_schema": (
            "waopd_stack_blocks_three_success_path_screen_epoch_receipt_v1"
        ),
    },
}


GENERIC_STAGE_DIAGNOSTIC_TASKS = frozenset(
    {
        "place_a2b_right",
        "place_bread_basket",
        "place_cans_plasticbox",
        "rotate_qrcode",
        "stamp_seal",
    }
)

for _task, _chunks, _max_control_steps in (
    ("place_a2b_right", 13, 400),
    ("place_bread_basket", 23, 700),
    ("place_cans_plasticbox", 26, 800),
    ("rotate_qrcode", 13, 400),
    ("stamp_seal", 13, 400),
):
    TASK_EVAL_SPECS[_task] = {
        "chunks": _chunks,
        "max_control_steps": _max_control_steps,
        "protocol_schema": f"waopd_{_task}_success_path_eval_protocol_v1",
        "unit_schema": f"waopd_{_task}_success_path_eval_unit_v1",
        "pair_schema": f"waopd_{_task}_success_path_progress_pair_v1",
        "screen_summary_schema": (
            f"waopd_{_task}_success_path_screen_summary_v1"
        ),
        "selection_schema": f"waopd_{_task}_success_path_selection_v1",
        "heldout_summary_schema": (
            f"waopd_{_task}_success_path_heldout_summary_v1"
        ),
        "screen_epoch_receipt_schema": (
            f"waopd_{_task}_success_path_screen_epoch_receipt_v1"
        ),
    }


def _task_eval_spec(task: str) -> dict[str, Any]:
    try:
        return TASK_EVAL_SPECS[str(task)]
    except KeyError as exc:
        raise ValueError(f"unsupported success-path eval task: {task!r}") from exc


def _task_schema(task: str, name: str) -> str:
    return str(_task_eval_spec(task)[name])


def _diagnostic_selection_metadata(task: str) -> dict[str, bool]:
    """Declare task diagnostics that never enter checkpoint selection."""

    _task_eval_spec(task)
    if task == "handover_mic":
        return {"contact_is_selection_input": False}
    if task == "open_microwave":
        return {"door_progress_is_selection_input": False}
    if task == "put_object_cabinet":
        return {"placement_diagnostics_are_selection_input": False}
    if task == "put_bottles_dustbin":
        return {"stage_reward_diagnostics_are_selection_input": False}
    if task == "place_fan":
        return {"pose_diagnostics_are_selection_input": False}
    if task == "place_shoe":
        return {
            "pose_diagnostics_are_selection_input": False,
            "shoe_z_is_official": False,
        }
    if task == "scan_object":
        return {"continuous_scan_diagnostics_are_selection_input": False}
    if task in ("blocks_ranking_rgb", "blocks_ranking_size"):
        return {"ordering_diagnostics_are_selection_input": False}
    if task == "place_a2b_left":
        return {"placement_diagnostics_are_selection_input": False}
    if task == "stack_blocks_three":
        return {"stacking_diagnostics_are_selection_input": False}
    if task in GENERIC_STAGE_DIAGNOSTIC_TASKS:
        return {"official_stage_diagnostics_are_selection_input": False}
    raise ValueError(f"unsupported diagnostic metadata task: {task!r}")


def _validate_task_protocol_contract(protocol: Mapping[str, Any]) -> str:
    """Validate the frozen task/domain/horizon portion without filesystem I/O."""

    task = str(protocol.get("task", ""))
    spec = _task_eval_spec(task)
    if protocol.get("schema") != spec["protocol_schema"]:
        raise ValueError(f"unexpected {task} success-path eval protocol schema")
    if protocol.get("task_config") != "demo_clean":
        raise ValueError("protocol task_config must be demo_clean")
    for field in ("chunks", "max_control_steps"):
        value = protocol.get(field)
        expected = int(spec[field])
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(
                f"protocol {field} must be {expected} for {task}, got {value!r}"
            )
    return task


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _under_ssd(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    return resolved == Path("/ssd/data") or Path("/ssd/data") in resolved.parents


def _require_ssd(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not _under_ssd(resolved):
        raise ValueError(f"{label} must be under /ssd/data: {resolved}")
    return resolved


def _parse_gpus(value: str) -> list[int]:
    gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)) or any(gpu < 0 for gpu in gpus):
        raise ValueError(f"expected unique non-negative GPU ids, got {value!r}")
    return gpus


@dataclass(frozen=True)
class EvalUnit:
    phase: str
    noise_base_seed: int
    seed: int
    prompt: str

    @property
    def key(self) -> tuple[int, int]:
        return (self.noise_base_seed, self.seed)


@dataclass(frozen=True)
class Candidate:
    epoch: int
    checkpoint: Path
    checkpoint_sha256: str
    fixed_calibration_total_loss: float

    @property
    def condition(self) -> str:
        return f"epoch_{self.epoch:02d}"


@dataclass
class ProgressTrace:
    rows: list[dict[str, Any]] = field(default_factory=list)
    recording_collector: Any | None = None
    observer_recording_key: str | None = None
    task: str = "handover_mic"

    def observation_callback(self, event: dict[str, Any]) -> None:
        state = event.get("evaluator_state")
        expected_metric = (
            "official_stage_reward"
            if self.task == "put_bottles_dustbin"
            else "official_predicate_milestone"
        )
        if not isinstance(state, Mapping) or state.get("metric") != expected_metric:
            raise RuntimeError(
                f"{self.task} observation lacks Stage-H progress telemetry"
            )
        row = {
            "event": str(event["event"]),
            "control_step": int(event["control_step"]),
            "macro_index": int(event["macro_index"]),
            "frame_st_id": int(event["frame_st_id"]),
            "ordinal_stage": int(state["ordinal_stage"]),
            "task_success": bool(event.get("task_success", False)),
            "eval_success": bool(event.get("eval_success", False)),
        }
        if self.task == "handover_mic":
            contact_present = bool(state.get("contact_present", False))
            receiver_closed = bool(state.get("receiver_closed", False))
            row.update(
                {
                    "official_success": bool(
                        state.get("official_success", False)
                    ),
                    "contact_present": contact_present,
                    "receiver_closed": receiver_closed,
                    "receiver_has_contact": contact_present and receiver_closed,
                    "giver_open": bool(state.get("giver_open", False)),
                    "height_valid": bool(state.get("height_valid", False)),
                    "side_valid": bool(state.get("side_valid", False)),
                    "microphone_x": float(state["microphone_x"]),
                    "microphone_z": float(state["microphone_z"]),
                }
            )
        elif self.task == "open_microwave":
            door_open = bool(state.get("door_open", False))
            # open_microwave has a single official terminal predicate.  The
            # continuous joint position is retained only as a diagnostic;
            # it must never affect checkpoint selection.
            row.update(
                {
                    "official_success": door_open,
                    "door_open": door_open,
                    "microwave_qpos": float(state["microwave_qpos"]),
                    "success_threshold": float(state["success_threshold"]),
                    "normalized_progress": float(
                        state["normalized_progress"]
                    ),
                }
            )
        elif self.task == "put_object_cabinet":
            placement_valid = bool(state.get("placement_valid", False))
            gripper_open = bool(state.get("gripper_open", False))
            row.update(
                {
                    "official_success": placement_valid and gripper_open,
                    "xy_inside": bool(state.get("xy_inside", False)),
                    "z_valid": bool(state.get("z_valid", False)),
                    "gripper_open": gripper_open,
                    "placement_valid": placement_valid,
                    "z_delta": float(state["z_delta"]),
                    "xy_linf_error": float(state["xy_linf_error"]),
                }
            )
        elif self.task == "put_bottles_dustbin":
            placed_bottles = int(state["placed_bottles"])
            stage_reward = float(state["stage_reward"])
            if placed_bottles < 0 or placed_bottles > 3:
                raise RuntimeError(
                    "put_bottles_dustbin placed_bottles is outside [0, 3]"
                )
            if abs(stage_reward * 3.0 - placed_bottles) > 1e-6:
                raise RuntimeError(
                    "put_bottles_dustbin stage-reward telemetry is internally "
                    "inconsistent"
                )
            if int(state["ordinal_stage"]) != placed_bottles:
                raise RuntimeError(
                    "put_bottles_dustbin ordinal stage must equal placed bottles"
                )
            row.update(
                {
                    # This is identical to native check_success(): all three
                    # bottles must be inside the official dustbin predicate.
                    "official_success": placed_bottles == 3,
                    "placed_bottles": placed_bottles,
                    "stage_reward": stage_reward,
                }
            )
        elif self.task == "place_fan":
            position_valid = bool(state.get("position_valid", False))
            quaternion_valid = bool(state.get("quaternion_valid", False))
            both_grippers_open = bool(
                state.get("both_grippers_open", False)
            )
            pose_valid = position_valid and quaternion_valid
            official_success = pose_valid and both_grippers_open
            if bool(state.get("pose_valid", False)) != pose_valid:
                raise RuntimeError(
                    "place_fan pose telemetry is internally inconsistent"
                )
            if bool(state.get("official_success", False)) != official_success:
                raise RuntimeError(
                    "place_fan official-success telemetry is internally "
                    "inconsistent"
                )
            row.update(
                {
                    "official_success": official_success,
                    "position_valid": position_valid,
                    "quaternion_valid": quaternion_valid,
                    "both_grippers_open": both_grippers_open,
                    "pose_valid": pose_valid,
                    "position_linf_normalized_error": float(
                        state["position_linf_normalized_error"]
                    ),
                    "quaternion_linf_normalized_error": float(
                        state["quaternion_linf_normalized_error"]
                    ),
                }
            )
        elif self.task == "place_shoe":
            if state.get("shoe_z_is_official") is not False:
                raise RuntimeError(
                    "place_shoe z diagnostic must remain non-official"
                )
            pose_valid = bool(state.get("pose_valid", False))
            both_grippers_open = bool(
                state.get("both_grippers_open", False)
            )
            row.update(
                {
                    "official_success": pose_valid and both_grippers_open,
                    "xy_valid": bool(state.get("xy_valid", False)),
                    "quaternion_valid": bool(
                        state.get("quaternion_valid", False)
                    ),
                    "both_grippers_open": both_grippers_open,
                    "pose_valid": pose_valid,
                    "xy_linf_normalized_error": float(
                        state["xy_linf_normalized_error"]
                    ),
                    "quaternion_linf_normalized_error": float(
                        state["quaternion_linf_normalized_error"]
                    ),
                    "shoe_z": float(state["shoe_z"]),
                    "shoe_z_is_official": False,
                }
            )
        elif self.task == "scan_object":
            projected_xyz_valid = bool(
                state.get("projected_xyz_valid", False)
            )
            depth_valid = bool(state.get("depth_valid", False))
            both_grippers_closed = bool(
                state.get("both_grippers_closed", False)
            )
            scan_geometry_valid = bool(
                state.get("scan_geometry_valid", False)
            )
            expected_geometry_valid = projected_xyz_valid and depth_valid
            if scan_geometry_valid != expected_geometry_valid:
                raise RuntimeError(
                    "scan_object geometry telemetry is internally inconsistent"
                )
            official_success = scan_geometry_valid and both_grippers_closed
            if bool(state.get("official_success", False)) != official_success:
                raise RuntimeError(
                    "scan_object official-success telemetry is internally "
                    "inconsistent"
                )
            row.update(
                {
                    "official_success": official_success,
                    "projected_xyz_valid": projected_xyz_valid,
                    "depth_valid": depth_valid,
                    "both_grippers_closed": both_grippers_closed,
                    "scan_geometry_valid": scan_geometry_valid,
                    "projected_xyz_linf_error": float(
                        state["projected_xyz_linf_error"]
                    ),
                    "scanner_axis_depth": float(
                        state["scanner_axis_depth"]
                    ),
                }
            )
        elif self.task == "place_a2b_left":
            distance_valid = bool(state.get("distance_valid", False))
            object_left_of_target = bool(
                state.get("object_left_of_target", False)
            )
            y_aligned = bool(state.get("y_aligned", False))
            placement_valid = bool(state.get("placement_valid", False))
            both_grippers_open = bool(
                state.get("both_grippers_open", False)
            )
            expected_placement = (
                distance_valid and object_left_of_target and y_aligned
            )
            if placement_valid != expected_placement:
                raise RuntimeError(
                    "place_a2b_left placement telemetry is internally "
                    "inconsistent"
                )
            official_success = placement_valid and both_grippers_open
            expected_stage = (
                3
                if official_success
                else 2
                if placement_valid
                else 1
                if object_left_of_target and y_aligned
                else 0
            )
            if int(state["ordinal_stage"]) != expected_stage:
                raise RuntimeError(
                    "place_a2b_left ordinal stage is inconsistent with "
                    "the official placement/release predicate"
                )
            row.update(
                {
                    "official_success": official_success,
                    "distance_valid": distance_valid,
                    "object_left_of_target": object_left_of_target,
                    "y_aligned": y_aligned,
                    "placement_valid": placement_valid,
                    "both_grippers_open": both_grippers_open,
                    "xy_distance": float(state["xy_distance"]),
                    "y_abs_error": float(state["y_abs_error"]),
                }
            )
        elif self.task == "stack_blocks_three":
            stacked_pair_12 = bool(state.get("stacked_pair_12", False))
            stacked_pair_23 = bool(state.get("stacked_pair_23", False))
            stacked_pair_count = int(state["stacked_pair_count"])
            both_grippers_open = bool(
                state.get("both_grippers_open", False)
            )
            expected_pair_count = int(stacked_pair_12) + int(stacked_pair_23)
            if stacked_pair_count != expected_pair_count:
                raise RuntimeError(
                    "stack_blocks_three stacking telemetry is internally "
                    "inconsistent"
                )
            official_success = stacked_pair_count == 2 and both_grippers_open
            expected_stage = 3 if official_success else stacked_pair_count
            if int(state["ordinal_stage"]) != expected_stage:
                raise RuntimeError(
                    "stack_blocks_three ordinal stage is inconsistent with "
                    "the official stacking/release predicate"
                )
            row.update(
                {
                    "official_success": official_success,
                    "stacked_pair_12": stacked_pair_12,
                    "stacked_pair_23": stacked_pair_23,
                    "stacked_pair_count": stacked_pair_count,
                    "both_grippers_open": both_grippers_open,
                    "min_pair_linf_normalized_error": float(
                        state["min_pair_linf_normalized_error"]
                    ),
                    "max_pair_linf_normalized_error": float(
                        state["max_pair_linf_normalized_error"]
                    ),
                }
            )
        elif self.task in ("blocks_ranking_rgb", "blocks_ranking_size"):
            ordered_pair_12 = bool(state.get("ordered_pair_12", False))
            ordered_pair_23 = bool(state.get("ordered_pair_23", False))
            ordered_pair_count = int(state["ordered_pair_count"])
            both_grippers_open = bool(
                state.get("both_grippers_open", False)
            )
            expected_pair_count = int(ordered_pair_12) + int(ordered_pair_23)
            if ordered_pair_count != expected_pair_count:
                raise RuntimeError(
                    f"{self.task} ordering telemetry is internally "
                    "inconsistent"
                )
            official_success = ordered_pair_count == 2 and both_grippers_open
            expected_stage = 3 if official_success else ordered_pair_count
            if int(state["ordinal_stage"]) != expected_stage:
                raise RuntimeError(
                    f"{self.task} ordinal stage is inconsistent with "
                    "the official ordering/release predicate"
                )
            row.update(
                {
                    "official_success": official_success,
                    "ordered_pair_12": ordered_pair_12,
                    "ordered_pair_23": ordered_pair_23,
                    "ordered_pair_count": ordered_pair_count,
                    "both_grippers_open": both_grippers_open,
                }
            )
        elif self.task in GENERIC_STAGE_DIAGNOSTIC_TASKS:
            row.update(
                {
                    "official_success": bool(
                        state.get("official_success", False)
                    ),
                }
            )
        else:
            raise ValueError(f"unsupported progress trace task: {self.task!r}")
        self.rows.append(row)
        if self.recording_collector is not None:
            recording_event = dict(event)
            recording_state = dict(state)
            observer_frame = recording_state.pop(OBSERVER_STATE_KEY, None)
            if self.observer_recording_key is not None:
                if observer_frame is None:
                    raise RuntimeError(
                        f"recorded {self.task} observation lacks observer_camera frame"
                    )
                recording_observation = dict(event["observation"])
                recording_observation[self.observer_recording_key] = observer_frame
                recording_event["observation"] = recording_observation
            recording_event["evaluator_state"] = recording_state
            self.recording_collector.observation_callback(recording_event)

    def macro_callback(self, event: dict[str, Any]) -> None:
        if self.recording_collector is not None:
            self.recording_collector.macro_callback(event)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                for row in self.rows
            ),
            encoding="utf-8",
        )

    def summarize(self, result: Mapping[str, Any]) -> dict[str, Any]:
        if not self.rows:
            raise RuntimeError("progress trace is empty")
        stage_first_steps: dict[str, int | None] = {}
        for stage in (1, 2, 3):
            reached = [
                int(row["control_step"])
                for row in self.rows
                if int(row["ordinal_stage"]) >= stage
            ]
            stage_first_steps[str(stage)] = min(reached) if reached else None
        summary = {
            "success": bool(result.get("success", False)),
            "observations": len(self.rows),
            "max_ordinal_stage": max(int(row["ordinal_stage"]) for row in self.rows),
            "stage_first_steps": stage_first_steps,
            "terminal_progress": result.get("progress"),
        }
        if self.task == "handover_mic":
            contact_rows = [row for row in self.rows if row["contact_present"]]
            receiver_contact_rows = [
                row for row in self.rows if row["receiver_has_contact"]
            ]
            max_streak = 0
            streak = 0
            for row in self.rows:
                streak = streak + 1 if row["contact_present"] else 0
                max_streak = max(max_streak, streak)
            summary.update(
                {
                    "contact_step_semantics": (
                        "first/last native runner key-snapshot observation with "
                        "contact_present"
                    ),
                    "max_contact": bool(contact_rows),
                    "first_contact_step": (
                        min(int(row["control_step"]) for row in contact_rows)
                        if contact_rows
                        else None
                    ),
                    "last_contact_step": (
                        max(int(row["control_step"]) for row in contact_rows)
                        if contact_rows
                        else None
                    ),
                    "contact_observations": len(contact_rows),
                    "max_contact_streak_observations": int(max_streak),
                    "max_receiver_contact": bool(receiver_contact_rows),
                    "first_receiver_contact_step": (
                        min(
                            int(row["control_step"])
                            for row in receiver_contact_rows
                        )
                        if receiver_contact_rows
                        else None
                    ),
                    "max_microphone_z": max(
                        float(row["microphone_z"]) for row in self.rows
                    ),
                }
            )
            return summary

        if self.task == "open_microwave":
            open_rows = [row for row in self.rows if row["door_open"]]
            best_row = max(
                self.rows, key=lambda row: float(row["normalized_progress"])
            )
            summary.update(
                {
                    "door_step_semantics": (
                        "native runner key-snapshot observations evaluated with "
                        "open_microwave.check_success(target=0.6)"
                    ),
                    "max_door_open": bool(open_rows),
                    "first_door_open_step": (
                        min(int(row["control_step"]) for row in open_rows)
                        if open_rows
                        else None
                    ),
                    "door_open_observations": len(open_rows),
                    "official_success_observations": sum(
                        int(bool(row["official_success"])) for row in self.rows
                    ),
                    "max_microwave_qpos": max(
                        float(row["microwave_qpos"]) for row in self.rows
                    ),
                    "max_normalized_progress": float(
                        best_row["normalized_progress"]
                    ),
                    "door_progress_is_selection_input": False,
                }
            )
            return summary

        if self.task == "put_object_cabinet":
            placement_rows = [
                row for row in self.rows if row["placement_valid"]
            ]
            best_xy_row = min(
                self.rows, key=lambda row: float(row["xy_linf_error"])
            )
            summary.update(
                {
                    "placement_step_semantics": (
                        "native runner key-snapshot observations evaluated with "
                        "the official Cabinet pose/release predicate"
                    ),
                    "max_placement_valid": bool(placement_rows),
                    "first_placement_valid_step": (
                        min(int(row["control_step"]) for row in placement_rows)
                        if placement_rows
                        else None
                    ),
                    "placement_observations": len(placement_rows),
                    "official_success_observations": sum(
                        int(bool(row["official_success"])) for row in self.rows
                    ),
                    "min_xy_linf_error": float(best_xy_row["xy_linf_error"]),
                    "min_xy_linf_error_step": int(best_xy_row["control_step"]),
                    "min_z_delta": min(
                        float(row["z_delta"]) for row in self.rows
                    ),
                    "max_z_delta": max(
                        float(row["z_delta"]) for row in self.rows
                    ),
                }
            )
            return summary

        if self.task == "put_bottles_dustbin":
            complete_rows = [
                row for row in self.rows if bool(row["official_success"])
            ]
            best_row = max(
                self.rows, key=lambda row: int(row["placed_bottles"])
            )
            summary.update(
                {
                    "bottle_step_semantics": (
                        "native runner key-snapshot observations evaluated with "
                        "put_bottles_dustbin.stage_reward/check_success"
                    ),
                    "max_placed_bottles": int(best_row["placed_bottles"]),
                    "max_stage_reward": float(best_row["stage_reward"]),
                    "first_all_bottles_placed_step": (
                        min(int(row["control_step"]) for row in complete_rows)
                        if complete_rows
                        else None
                    ),
                    "all_bottles_placed_observations": len(complete_rows),
                    "official_success_observations": sum(
                        int(bool(row["official_success"])) for row in self.rows
                    ),
                    "stage_reward_diagnostics_are_selection_input": False,
                }
            )
            return summary

        if self.task == "place_fan":
            pose_rows = [row for row in self.rows if row["pose_valid"]]
            best_position_row = min(
                self.rows,
                key=lambda row: float(
                    row["position_linf_normalized_error"]
                ),
            )
            best_quaternion_row = min(
                self.rows,
                key=lambda row: float(
                    row["quaternion_linf_normalized_error"]
                ),
            )
            summary.update(
                {
                    "pose_step_semantics": (
                        "native runner key-snapshot observations evaluated "
                        "with the official place_fan xyz/quaternion/release "
                        "predicate"
                    ),
                    "max_pose_valid": bool(pose_rows),
                    "first_pose_valid_step": (
                        min(int(row["control_step"]) for row in pose_rows)
                        if pose_rows
                        else None
                    ),
                    "pose_valid_observations": len(pose_rows),
                    "official_success_observations": sum(
                        int(bool(row["official_success"]))
                        for row in self.rows
                    ),
                    "min_position_linf_normalized_error": float(
                        best_position_row[
                            "position_linf_normalized_error"
                        ]
                    ),
                    "min_position_linf_normalized_error_step": int(
                        best_position_row["control_step"]
                    ),
                    "min_quaternion_linf_normalized_error": float(
                        best_quaternion_row[
                            "quaternion_linf_normalized_error"
                        ]
                    ),
                    "min_quaternion_linf_normalized_error_step": int(
                        best_quaternion_row["control_step"]
                    ),
                    "pose_diagnostics_are_selection_input": False,
                }
            )
            return summary

        if self.task == "place_shoe":
            pose_rows = [row for row in self.rows if row["pose_valid"]]
            best_xy_row = min(
                self.rows,
                key=lambda row: float(row["xy_linf_normalized_error"]),
            )
            best_quaternion_row = min(
                self.rows,
                key=lambda row: float(
                    row["quaternion_linf_normalized_error"]
                ),
            )
            summary.update(
                {
                    "pose_step_semantics": (
                        "native runner key-snapshot observations evaluated "
                        "with the official place_shoe xy/quaternion/release "
                        "predicate"
                    ),
                    "max_pose_valid": bool(pose_rows),
                    "first_pose_valid_step": (
                        min(int(row["control_step"]) for row in pose_rows)
                        if pose_rows
                        else None
                    ),
                    "pose_valid_observations": len(pose_rows),
                    "official_success_observations": sum(
                        int(bool(row["official_success"]))
                        for row in self.rows
                    ),
                    "min_xy_linf_normalized_error": float(
                        best_xy_row["xy_linf_normalized_error"]
                    ),
                    "min_xy_linf_normalized_error_step": int(
                        best_xy_row["control_step"]
                    ),
                    "min_quaternion_linf_normalized_error": float(
                        best_quaternion_row[
                            "quaternion_linf_normalized_error"
                        ]
                    ),
                    "min_quaternion_linf_normalized_error_step": int(
                        best_quaternion_row["control_step"]
                    ),
                    "min_shoe_z": min(
                        float(row["shoe_z"]) for row in self.rows
                    ),
                    "max_shoe_z": max(
                        float(row["shoe_z"]) for row in self.rows
                    ),
                    "shoe_z_is_official": False,
                    "shoe_z_is_selection_input": False,
                }
            )
            return summary

        if self.task == "scan_object":
            projection_rows = [
                row for row in self.rows if row["projected_xyz_valid"]
            ]
            depth_rows = [row for row in self.rows if row["depth_valid"]]
            geometry_rows = [
                row for row in self.rows if row["scan_geometry_valid"]
            ]
            closed_rows = [
                row for row in self.rows if row["both_grippers_closed"]
            ]
            best_projection_row = min(
                self.rows,
                key=lambda row: float(row["projected_xyz_linf_error"]),
            )
            summary.update(
                {
                    "scan_step_semantics": (
                        "native runner key-snapshot observations evaluated "
                        "with the official scan_object projected-xyz/depth/"
                        "both-grippers-closed predicate"
                    ),
                    "max_projected_xyz_valid": bool(projection_rows),
                    "max_depth_valid": bool(depth_rows),
                    "max_scan_geometry_valid": bool(geometry_rows),
                    "max_both_grippers_closed": bool(closed_rows),
                    "first_scan_geometry_valid_step": (
                        min(
                            int(row["control_step"])
                            for row in geometry_rows
                        )
                        if geometry_rows
                        else None
                    ),
                    "scan_geometry_valid_observations": len(geometry_rows),
                    "official_success_observations": sum(
                        int(bool(row["official_success"]))
                        for row in self.rows
                    ),
                    "min_projected_xyz_linf_error": float(
                        best_projection_row["projected_xyz_linf_error"]
                    ),
                    "min_projected_xyz_linf_error_step": int(
                        best_projection_row["control_step"]
                    ),
                    "min_scanner_axis_depth": min(
                        float(row["scanner_axis_depth"])
                        for row in self.rows
                    ),
                    "max_scanner_axis_depth": max(
                        float(row["scanner_axis_depth"])
                        for row in self.rows
                    ),
                    "continuous_scan_diagnostics_are_selection_input": False,
                }
            )
            return summary

        if self.task == "place_a2b_left":
            placement_rows = [
                row for row in self.rows if row["placement_valid"]
            ]
            best_distance_row = min(
                self.rows,
                key=lambda row: min(
                    abs(float(row["xy_distance"]) - 0.08),
                    abs(float(row["xy_distance"]) - 0.2),
                ),
            )
            summary.update(
                {
                    "placement_step_semantics": (
                        "native runner key-snapshot observations evaluated "
                        "with place_a2b_left.check_success: xy annulus, "
                        "left-of-target, y alignment, and both grippers open"
                    ),
                    "max_placement_valid": bool(placement_rows),
                    "first_placement_valid_step": (
                        min(
                            int(row["control_step"])
                            for row in placement_rows
                        )
                        if placement_rows
                        else None
                    ),
                    "placement_valid_observations": len(placement_rows),
                    "official_success_observations": sum(
                        int(bool(row["official_success"]))
                        for row in self.rows
                    ),
                    "best_xy_distance": float(
                        best_distance_row["xy_distance"]
                    ),
                    "min_y_abs_error": min(
                        float(row["y_abs_error"]) for row in self.rows
                    ),
                    "placement_diagnostics_are_selection_input": False,
                }
            )
            return summary

        if self.task == "stack_blocks_three":
            fully_stacked_rows = [
                row for row in self.rows if row["stacked_pair_count"] == 2
            ]
            best_row = max(
                self.rows, key=lambda row: int(row["stacked_pair_count"])
            )
            summary.update(
                {
                    "stacking_step_semantics": (
                        "native runner key-snapshot observations evaluated "
                        "with stack_blocks_three.check_success: both vertical "
                        "adjacency conjuncts plus both grippers open"
                    ),
                    "max_stacked_pair_count": int(
                        best_row["stacked_pair_count"]
                    ),
                    "first_fully_stacked_step": (
                        min(
                            int(row["control_step"])
                            for row in fully_stacked_rows
                        )
                        if fully_stacked_rows
                        else None
                    ),
                    "fully_stacked_observations": len(fully_stacked_rows),
                    "official_success_observations": sum(
                        int(bool(row["official_success"]))
                        for row in self.rows
                    ),
                    "max_stacked_pair_12": any(
                        bool(row["stacked_pair_12"]) for row in self.rows
                    ),
                    "max_stacked_pair_23": any(
                        bool(row["stacked_pair_23"]) for row in self.rows
                    ),
                    "min_pair_linf_normalized_error": min(
                        float(row["min_pair_linf_normalized_error"])
                        for row in self.rows
                    ),
                    "min_max_pair_linf_normalized_error": min(
                        float(row["max_pair_linf_normalized_error"])
                        for row in self.rows
                    ),
                    "stacking_diagnostics_are_selection_input": False,
                }
            )
            return summary

        if self.task in ("blocks_ranking_rgb", "blocks_ranking_size"):
            fully_ordered_rows = [
                row for row in self.rows if row["ordered_pair_count"] == 2
            ]
            best_row = max(
                self.rows, key=lambda row: int(row["ordered_pair_count"])
            )
            summary.update(
                {
                    "ordering_step_semantics": (
                        "native runner key-snapshot observations evaluated "
                        f"with {self.task}.check_success: both adjacent "
                        "xy ordering conjuncts plus both grippers open"
                    ),
                    "max_ordered_pair_count": int(
                        best_row["ordered_pair_count"]
                    ),
                    "first_fully_ordered_step": (
                        min(
                            int(row["control_step"])
                            for row in fully_ordered_rows
                        )
                        if fully_ordered_rows
                        else None
                    ),
                    "fully_ordered_observations": len(fully_ordered_rows),
                    "official_success_observations": sum(
                        int(bool(row["official_success"]))
                        for row in self.rows
                    ),
                    "max_ordered_pair_12": any(
                        bool(row["ordered_pair_12"]) for row in self.rows
                    ),
                    "max_ordered_pair_23": any(
                        bool(row["ordered_pair_23"]) for row in self.rows
                    ),
                    "ordering_diagnostics_are_selection_input": False,
                }
            )
            return summary

        if self.task in GENERIC_STAGE_DIAGNOSTIC_TASKS:
            summary.update(
                {
                    "official_success_observations": sum(
                        int(bool(row["official_success"]))
                        for row in self.rows
                    ),
                    "official_stage_diagnostics_are_selection_input": False,
                }
            )
            return summary

        raise ValueError(f"unsupported progress trace task: {self.task!r}")


def _install_task_progress_bridge(
    *, task: str, capture_observer: bool = False
) -> None:
    """Expose frozen Stage-H progress in the existing observation callback.

    The native runner currently records generic evaluator state for only two
    older tasks.  Patch that diagnostic function in this fresh eval process;
    policy inputs, solver state, simulator state, and the source files remain
    untouched.
    """

    from experiments import waopd_native_closed_loop_runner as runner
    from experiments.stage_h_task_progress import collect_task_progress

    original = runner._evaluator_stage_state

    def bridged(task_env: object, observed_task: str) -> dict[str, Any]:
        if observed_task != task:
            return original(task_env, observed_task)
        state = {
            "task": observed_task,
            "source": "experiments.stage_h_task_progress.collect_task_progress",
            "policy_input": False,
            **dict(collect_task_progress(observed_task, task_env)),
        }
        if capture_observer:
            state[OBSERVER_STATE_KEY] = task_env.cameras.get_observer_rgb()
        return state

    runner._evaluator_stage_state = bridged


def _install_handover_progress_bridge(*, capture_observer: bool = False) -> None:
    """Backward-compatible alias for the completed Handover protocol."""

    _install_task_progress_bridge(
        task="handover_mic", capture_observer=capture_observer
    )


def _run_unit(args: argparse.Namespace) -> int:
    output = _require_ssd(args.output, "unit output")
    task_spec = _task_eval_spec(args.task)
    if args.task_config != "demo_clean":
        raise ValueError("unit task_config must be demo_clean")
    if int(args.chunks) != int(task_spec["chunks"]):
        raise ValueError(
            f"unit chunks must be {task_spec['chunks']} for {args.task}"
        )
    if int(args.max_control_steps) != int(task_spec["max_control_steps"]):
        raise ValueError(
            "unit max_control_steps must be "
            f"{task_spec['max_control_steps']} for {args.task}"
        )
    _install_task_progress_bridge(
        task=args.task, capture_observer=bool(args.record)
    )

    recording_collector = None
    if args.record:
        from experiments import v0k_native_video_diagnostic as recording_module

        # Process-local recorder extension only.  The policy-facing formatted
        # observation remains unchanged; observer_camera is rendered solely
        # for the diagnostic video stream.
        recording_module.CAMERAS = {
            **recording_module.CAMERAS,
            OBSERVER_RECORDING_KEY: "observer_camera",
        }

        recording_collector = recording_module.RecordingCollector(
            root=output.parent / "recording",
            task=args.task,
            seed=int(args.seed),
            condition=str(args.condition),
        )
    trace = ProgressTrace(
        task=args.task,
        recording_collector=recording_collector,
        observer_recording_key=(OBSERVER_RECORDING_KEY if args.record else None),
    )
    unit_summary_path = output.parent / "unit_summary.json"
    recording_summary: dict[str, Any] | None = None
    try:
        from experiments.waopd_v0j_teacher_free_behavior import run_one

        result = run_one(
            task=args.task,
            task_config=args.task_config,
            seed=int(args.seed),
            chunks=int(args.chunks),
            max_control_steps=int(args.max_control_steps),
            noise_base_seed=int(args.noise_base_seed),
            student=args.student,
            output=output,
            project_root=args.project_root,
            device=args.device,
            enable_offload=True,
            official_offload_parity=True,
            adapter_state=args.adapter_state,
            adapter_kind_override=(
                "joint_lora" if args.adapter_state is not None else None
            ),
            prompt_override=args.prompt,
            macro_callback=trace.macro_callback,
            observation_callback=trace.observation_callback,
            stop_on_success=True,
        )
        trace.write(output.parent / "progress_trace.jsonl")
        progress = trace.summarize(result)
        if recording_collector is not None:
            recording_summary = recording_collector.finalize(result)
        summary = {
            "schema": _task_schema(args.task, "unit_schema"),
            "status": "PASS",
            "phase": args.phase,
            "condition": args.condition,
            "task": args.task,
            "task_config": args.task_config,
            "seed": int(args.seed),
            "prompt": args.prompt,
            "noise_base_seed": int(args.noise_base_seed),
            "chunks": int(args.chunks),
            "max_control_steps": int(args.max_control_steps),
            "student": str(args.student.expanduser().resolve()),
            "adapter_state": (
                str(args.adapter_state.expanduser().resolve())
                if args.adapter_state is not None
                else None
            ),
            "protocol_sha256": args.protocol_sha256,
            "episode": str(output),
            "progress_trace": str((output.parent / "progress_trace.jsonl").resolve()),
            "progress": progress,
            "record_requested": bool(args.record),
            "recording": recording_summary,
        }
        _write_json(unit_summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        recording_error = None
        if recording_collector is not None:
            try:
                recording_summary = recording_collector.finalize(
                    {"status": "BLOCKED", "success": False}, error=str(exc)
                )
            except Exception as recording_exc:  # retain the primary failure
                recording_error = f"{type(recording_exc).__name__}: {recording_exc}"
        trace.write(output.parent / "progress_trace.jsonl")
        blocked = {
            "schema": _task_schema(args.task, "unit_schema"),
            "status": "BLOCKED",
            "phase": args.phase,
            "condition": args.condition,
            "seed": int(args.seed),
            "noise_base_seed": int(args.noise_base_seed),
            "reason": f"{type(exc).__name__}: {exc}",
            "record_requested": bool(args.record),
            "recording": recording_summary,
            "recording_error": recording_error,
        }
        _write_json(unit_summary_path, blocked)
        print(json.dumps(blocked, indent=2, sort_keys=True), file=sys.stderr)
        return 2


def _phase_units(protocol: Mapping[str, Any], phase: str) -> list[EvalUnit]:
    section = protocol[phase]
    rows = section.get("episode_records")
    banks = section.get("noise_base_seeds")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{phase}.episode_records must be non-empty")
    if not isinstance(banks, list) or not banks:
        raise ValueError(f"{phase}.noise_base_seeds must be non-empty")
    units: list[EvalUnit] = []
    for bank in banks:
        if isinstance(bank, bool) or not isinstance(bank, int):
            raise ValueError(f"{phase} noise bank must be an integer")
        for row in rows:
            seed = row.get("seed")
            prompt = row.get("instruction")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(f"{phase} seed must be an integer")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{phase} instruction must be non-empty")
            units.append(
                EvalUnit(
                    phase=phase,
                    noise_base_seed=int(bank),
                    seed=int(seed),
                    prompt=prompt.strip(),
                )
            )
    if len({unit.key for unit in units}) != len(units):
        raise ValueError(f"{phase} contains duplicate bank/seed units")
    return units


def _record_keys(protocol: Mapping[str, Any], phase: str) -> set[tuple[int, int]]:
    rows = protocol[phase].get("record_pairs", [])
    if not isinstance(rows, list):
        raise ValueError(f"{phase}.record_pairs must be a list")
    result: set[tuple[int, int]] = set()
    for row in rows:
        key = (int(row["noise_base_seed"]), int(row["seed"]))
        if key in result:
            raise ValueError(f"duplicate {phase} recording pair: {key}")
        result.add(key)
    return result


def _validate_protocol(path: Path) -> tuple[dict[str, Any], str]:
    path = path.expanduser().resolve()
    protocol = _read_json(path)
    task = _validate_task_protocol_contract(protocol)
    if tuple(protocol.get("selection_rule", ())) != EXPECTED_SELECTION_RULE:
        raise ValueError("protocol checkpoint selection rule differs from frozen rule")
    _require_ssd(Path(protocol["output_root"]), "protocol output_root")
    _require_ssd(Path(protocol["student"]), "protocol student")

    screen_units = _phase_units(protocol, "screening")
    heldout_units = _phase_units(protocol, "heldout")
    screen_seeds = {unit.seed for unit in screen_units}
    heldout_seeds = {unit.seed for unit in heldout_units}
    if screen_seeds & heldout_seeds:
        raise ValueError("screening and heldout seeds overlap")
    screen_banks = {unit.noise_base_seed for unit in screen_units}
    heldout_banks = {unit.noise_base_seed for unit in heldout_units}
    if screen_banks & heldout_banks:
        raise ValueError("screening and heldout noise banks overlap")

    training_config = _read_json(Path(protocol["training_config"]).expanduser().resolve())
    for field in ("task", "task_config", "chunks"):
        if training_config.get(field) != protocol.get(field):
            raise ValueError(
                f"training config {field} differs from eval protocol for {task}"
            )
    train_calib_seeds = {
        int(row["seed"])
        for row in training_config.get("rollouts", [])
        if row.get("role") in {"train", "calibration"}
    }
    overlap = train_calib_seeds & (screen_seeds | heldout_seeds)
    if overlap:
        raise ValueError(f"eval seeds overlap train/calibration seeds: {sorted(overlap)}")

    for phase, units in (("screening", screen_units), ("heldout", heldout_units)):
        available = {unit.key for unit in units}
        extra = _record_keys(protocol, phase) - available
        if extra:
            raise ValueError(f"{phase} recording pairs are not eval units: {sorted(extra)}")
    epoch_ids = protocol.get("epoch_ids")
    if epoch_ids != [1, 2, 3]:
        raise ValueError("protocol epoch_ids must be exactly [1, 2, 3]")
    return protocol, _sha256(path)


def _resolve_epoch_candidate_checkpoint(
    *,
    epoch: int,
    training_summary: Mapping[str, Any],
    summary_path: Path,
    protocol: Mapping[str, Any],
) -> tuple[Path, str]:
    """Resolve one epoch checkpoint, strictly binding the final-root fallback."""

    summary_path = summary_path.expanduser().resolve()
    legacy = _require_ssd(
        summary_path.parent
        / "epoch_checkpoints"
        / f"checkpoint_epoch_{int(epoch):02d}.pt",
        f"epoch {epoch} checkpoint",
    )
    if legacy.is_file():
        return legacy, _sha256(legacy)

    update = training_summary.get("update")
    if not isinstance(update, Mapping):
        raise RuntimeError("training summary has no update mapping")
    inner_epochs = update.get("inner_epochs")
    if (
        isinstance(inner_epochs, bool)
        or not isinstance(inner_epochs, int)
        or int(inner_epochs) <= 0
    ):
        raise RuntimeError("training summary inner_epochs is invalid")
    total_inner_epochs = int(inner_epochs)

    epoch_ids = protocol.get("epoch_ids")
    if (
        not isinstance(epoch_ids, list)
        or not epoch_ids
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in epoch_ids
        )
        or max(epoch_ids) != total_inner_epochs
    ):
        raise RuntimeError(
            "protocol epoch_ids do not match training summary inner_epochs"
        )
    if int(epoch) != total_inner_epochs:
        raise FileNotFoundError(f"epoch checkpoint is missing: {legacy}")

    checkpoint = _require_ssd(
        summary_path.parent / "checkpoint_trajectory_update.pt",
        f"epoch {epoch} final checkpoint",
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"final epoch checkpoint is missing: {checkpoint}")

    if training_summary.get("status") != "PASS":
        raise RuntimeError("final checkpoint summary is not PASS")
    if training_summary.get("coherent_tt_variant") != "success_path_v1":
        raise RuntimeError("final checkpoint summary is not success_path_v1")
    if update.get("objective") != "coherent_tt_consistency":
        raise RuntimeError("final checkpoint summary objective mismatch")
    if update.get("coherent_tt_variant") != "success_path_v1":
        raise RuntimeError("final checkpoint summary update variant mismatch")

    try:
        bound_checkpoint = Path(
            str(training_summary["checkpoint"])
        ).expanduser().resolve()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("training summary checkpoint path is malformed") from exc
    if bound_checkpoint != checkpoint:
        raise RuntimeError("training summary points to a different checkpoint")
    observed_sha256 = _sha256(checkpoint)
    summary_sha256 = training_summary.get("checkpoint_sha256")
    if summary_sha256 != observed_sha256:
        raise RuntimeError("training summary checkpoint SHA-256 mismatch")

    try:
        import torch

        loaded = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError("final checkpoint failed safe CPU load") from exc
    post_load_sha256 = _sha256(checkpoint)
    if (
        post_load_sha256 != observed_sha256
        or post_load_sha256 != summary_sha256
    ):
        raise RuntimeError("final checkpoint changed during safe CPU load")
    if not isinstance(loaded, Mapping):
        raise RuntimeError("final checkpoint is not a mapping")
    if loaded.get("checkpoint_role") != "success_path_final":
        raise RuntimeError("final checkpoint checkpoint_role mismatch")
    completed = loaded.get("completed_inner_epochs")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or int(completed) != total_inner_epochs
    ):
        raise RuntimeError("final checkpoint completed_inner_epochs mismatch")
    if loaded.get("task") != protocol.get("task") or loaded.get(
        "task_config"
    ) != protocol.get("task_config"):
        raise RuntimeError("final checkpoint task contract mismatch")
    if loaded.get("objective") != "coherent_tt_consistency":
        raise RuntimeError("final checkpoint objective mismatch")
    if loaded.get("coherent_tt_variant") != "success_path_v1":
        raise RuntimeError("final checkpoint coherent_tt_variant mismatch")

    task_contract_hash = training_summary.get("task_contract_hash")
    if not isinstance(task_contract_hash, str) or not task_contract_hash:
        raise RuntimeError("training summary task_contract_hash is missing")
    if loaded.get("task_contract_hash") != task_contract_hash:
        raise RuntimeError("final checkpoint task_contract_hash mismatch")

    global_step = training_summary.get("global_optimizer_step")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or int(global_step) <= 0
        or loaded.get("global_optimizer_step") != global_step
    ):
        raise RuntimeError("final checkpoint global_optimizer_step mismatch")
    policy_version = training_summary.get("policy_version_after")
    if (
        not isinstance(policy_version, str)
        or not policy_version
        or loaded.get("policy_version_after") != policy_version
    ):
        raise RuntimeError("final checkpoint policy_version_after mismatch")
    return checkpoint, observed_sha256


def _load_candidates(protocol: Mapping[str, Any]) -> tuple[list[Candidate], str]:
    summary_path = Path(protocol["training_summary"]).expanduser().resolve()
    summary = _read_json(summary_path)
    if summary.get("status") != "PASS":
        raise RuntimeError("training summary is not PASS")
    if summary.get("coherent_tt_variant") != "success_path_v1":
        raise RuntimeError("training summary is not success_path_v1")
    update = summary.get("update")
    if not isinstance(update, Mapping):
        raise RuntimeError("training summary has no update mapping")
    rows = update.get("epoch_metrics")
    if not isinstance(rows, list):
        raise RuntimeError("training summary has no epoch_metrics")
    expected_epochs = list(protocol["epoch_ids"])
    by_epoch: dict[int, Candidate] = {}
    for row in rows:
        epoch = int(row["epoch"])
        if epoch not in expected_epochs:
            continue
        calibration = row.get("calibration")
        if not isinstance(calibration, Mapping):
            raise RuntimeError(f"epoch {epoch} has no fixed calibration mapping")
        loss = float(calibration["loss"])
        if not math.isfinite(loss):
            raise RuntimeError(f"epoch {epoch} fixed calibration loss is nonfinite")
        checkpoint, checkpoint_sha256 = _resolve_epoch_candidate_checkpoint(
            epoch=epoch,
            training_summary=summary,
            summary_path=summary_path,
            protocol=protocol,
        )
        by_epoch[epoch] = Candidate(
            epoch=epoch,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            fixed_calibration_total_loss=loss,
        )
    if sorted(by_epoch) != expected_epochs:
        raise RuntimeError(
            f"training summary epochs differ: {sorted(by_epoch)} != {expected_epochs}"
        )
    return [by_epoch[epoch] for epoch in expected_epochs], _sha256(summary_path)


def _unit_complete(
    *,
    output: Path,
    phase: str,
    condition: str,
    unit: EvalUnit,
    protocol_sha256: str,
    student: Path,
    adapter: Path | None,
    expected_adapter_state_sha256: str | None,
    record: bool,
) -> dict[str, Any] | None:
    summary_path = output.parent / "unit_summary.json"
    if not output.is_file() or not summary_path.is_file():
        return None
    try:
        summary = _read_json(summary_path)
        episode = _read_json(output)
    except (OSError, ValueError, TypeError):
        return None
    expected = {
        "status": "PASS",
        "phase": phase,
        "condition": condition,
        "seed": unit.seed,
        "prompt": unit.prompt,
        "noise_base_seed": unit.noise_base_seed,
        "student": str(student.expanduser().resolve()),
        "adapter_state": (
            str(adapter.expanduser().resolve()) if adapter is not None else None
        ),
        "protocol_sha256": protocol_sha256,
        "record_requested": bool(record),
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        return None
    if episode.get("status") != "PASS":
        return None
    if (
        "adapter_state_sha256" not in episode
        or episode["adapter_state_sha256"] != expected_adapter_state_sha256
    ):
        return None
    if record:
        recording = summary.get("recording")
        if not isinstance(recording, Mapping):
            return None
        video_root = recording.get("video_root")
        if not isinstance(video_root, str) or not (Path(video_root) / "composite.mp4").is_file():
            return None
    return summary


def _run_policy(
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    gpu: int,
    phase: str,
    unit: EvalUnit,
    condition: str,
    adapter: Path | None,
    expected_adapter_state_sha256: str | None,
    output: Path,
    record: bool,
) -> dict[str, Any]:
    if (adapter is None) != (expected_adapter_state_sha256 is None):
        raise ValueError("adapter path and expected adapter SHA-256 must agree")
    student = Path(protocol["student"]).expanduser().resolve()
    existing = _unit_complete(
        output=output,
        phase=phase,
        condition=condition,
        unit=unit,
        protocol_sha256=protocol_sha256,
        student=student,
        adapter=adapter,
        expected_adapter_state_sha256=expected_adapter_state_sha256,
        record=record,
    )
    if existing is not None:
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        # Do not resolve this path: venv executables may be symlinks to the
        # base interpreter, and dereferencing one bypasses the venv's
        # site-packages in the child process.
        str(Path(protocol["python_bin"]).expanduser()),
        "-u",
        "-m",
        "experiments.run_handover_mic_success_path_eval",
        "unit",
        "--phase",
        phase,
        "--condition",
        condition,
        "--task",
        str(protocol["task"]),
        "--task-config",
        str(protocol["task_config"]),
        "--seed",
        str(unit.seed),
        "--prompt",
        unit.prompt,
        "--noise-base-seed",
        str(unit.noise_base_seed),
        "--chunks",
        str(protocol["chunks"]),
        "--max-control-steps",
        str(protocol["max_control_steps"]),
        "--student",
        str(student),
        "--project-root",
        str(Path(protocol["project_root"]).expanduser().resolve()),
        "--device",
        "cuda:0",
        "--output",
        str(output),
        "--protocol-sha256",
        protocol_sha256,
    ]
    if adapter is not None:
        command.extend(["--adapter-state", str(adapter.expanduser().resolve())])
    if record:
        command.append("--record")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    log_path = output.parent / "run.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n=== success-path eval unit start ===\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=Path(protocol["workspace"]).expanduser().resolve(),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{condition} failed for seed={unit.seed} bank={unit.noise_base_seed} "
            f"gpu={gpu}; see {log_path}"
        )
    result = _unit_complete(
        output=output,
        phase=phase,
        condition=condition,
        unit=unit,
        protocol_sha256=protocol_sha256,
        student=student,
        adapter=adapter,
        expected_adapter_state_sha256=expected_adapter_state_sha256,
        record=record,
    )
    if result is None:
        raise RuntimeError(f"unit completed without a valid receipt: {output.parent}")
    return result


def _build_pair(
    *,
    protocol: Mapping[str, Any],
    unit: EvalUnit,
    condition: str,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    released_path: Path,
    adapted_path: Path,
    released_unit: Mapping[str, Any],
    adapted_unit: Mapping[str, Any],
    gpu: int,
) -> dict[str, Any]:
    from experiments.run_open_microwave_paired_multinoise import _validate_pair

    base = _validate_pair(
        released_path=released_path,
        opd_path=adapted_path,
        seed=unit.seed,
        prompt=unit.prompt,
        noise_base_seed=unit.noise_base_seed,
        task=str(protocol["task"]),
        task_config=str(protocol["task_config"]),
        chunks=int(protocol["chunks"]),
        max_control_steps=int(protocol["max_control_steps"]),
        student=Path(protocol["student"]),
        adapter=checkpoint,
    )
    released_progress = released_unit["progress"]
    adapted_progress = adapted_unit["progress"]
    released_stage = int(released_progress["max_ordinal_stage"])
    adapted_stage = int(adapted_progress["max_ordinal_stage"])
    released_episode = _read_json(released_path)
    adapted_episode = _read_json(adapted_path)
    if (
        "adapter_state_sha256" not in released_episode
        or released_episode["adapter_state_sha256"] is not None
    ):
        raise RuntimeError("Released episode has adapted checkpoint identity")
    if (
        "adapter_state_sha256" not in adapted_episode
        or adapted_episode["adapter_state_sha256"]
        != expected_checkpoint_sha256
    ):
        raise RuntimeError("adapted episode checkpoint SHA-256 mismatch")

    def composite_mp4(unit_summary: Mapping[str, Any]) -> str | None:
        recording = unit_summary.get("recording")
        if not isinstance(recording, Mapping):
            return None
        video_root = recording.get("video_root")
        if not isinstance(video_root, str):
            return None
        return str((Path(video_root) / "composite.mp4").resolve())

    pair = {
        **base,
        "schema": _task_schema(str(protocol["task"]), "pair_schema"),
        "condition": condition,
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "checkpoint_sha256": adapted_episode["adapter_state_sha256"],
        "released_checkpoint_sha256": released_episode[
            "adapter_state_sha256"
        ],
        "gpu": int(gpu),
        "released_max_ordinal_stage": released_stage,
        "adapted_max_ordinal_stage": adapted_stage,
        "paired_max_ordinal_delta": adapted_stage - released_stage,
        "stage_improvement": adapted_stage > released_stage,
        "stage_regression": adapted_stage < released_stage,
        "released_progress_receipt": str(
            (released_path.parent / "unit_summary.json").resolve()
        ),
        "adapted_progress_receipt": str(
            (adapted_path.parent / "unit_summary.json").resolve()
        ),
        "released_composite_mp4": composite_mp4(released_unit),
        "adapted_composite_mp4": composite_mp4(adapted_unit),
    }
    task = str(protocol["task"])
    if task == "handover_mic":
        released_first_contact = released_progress.get("first_contact_step")
        adapted_first_contact = adapted_progress.get("first_contact_step")
        contact_cap = int(protocol["max_control_steps"]) + 1
        contact_onset_delta = (
            int(
                released_first_contact
                if released_first_contact is not None
                else contact_cap
            )
            - int(
                adapted_first_contact
                if adapted_first_contact is not None
                else contact_cap
            )
        )
        pair.update(
            {
                "released_contact": bool(released_progress["max_contact"]),
                "adapted_contact": bool(adapted_progress["max_contact"]),
                "released_first_contact_step": released_first_contact,
                "adapted_first_contact_step": adapted_first_contact,
                "contact_onset_delta": contact_onset_delta,
                "released_max_contact_streak_observations": int(
                    released_progress["max_contact_streak_observations"]
                ),
                "adapted_max_contact_streak_observations": int(
                    adapted_progress["max_contact_streak_observations"]
                ),
                "released_max_receiver_contact": bool(
                    released_progress["max_receiver_contact"]
                ),
                "adapted_max_receiver_contact": bool(
                    adapted_progress["max_receiver_contact"]
                ),
            }
        )
    elif task == "open_microwave":
        released_progress_value = float(
            released_progress["max_normalized_progress"]
        )
        adapted_progress_value = float(
            adapted_progress["max_normalized_progress"]
        )
        pair.update(
            {
                "released_door_open": bool(
                    released_progress["max_door_open"]
                ),
                "adapted_door_open": bool(
                    adapted_progress["max_door_open"]
                ),
                "released_first_door_open_step": released_progress.get(
                    "first_door_open_step"
                ),
                "adapted_first_door_open_step": adapted_progress.get(
                    "first_door_open_step"
                ),
                "released_max_normalized_progress": released_progress_value,
                "adapted_max_normalized_progress": adapted_progress_value,
                "paired_max_normalized_progress_delta": (
                    adapted_progress_value - released_progress_value
                ),
                "door_progress_is_selection_input": False,
            }
        )
    elif task == "put_object_cabinet":
        released_xy_error = float(released_progress["min_xy_linf_error"])
        adapted_xy_error = float(adapted_progress["min_xy_linf_error"])
        pair.update(
            {
                "released_placement_valid": bool(
                    released_progress["max_placement_valid"]
                ),
                "adapted_placement_valid": bool(
                    adapted_progress["max_placement_valid"]
                ),
                "released_first_placement_valid_step": released_progress.get(
                    "first_placement_valid_step"
                ),
                "adapted_first_placement_valid_step": adapted_progress.get(
                    "first_placement_valid_step"
                ),
                "released_min_xy_linf_error": released_xy_error,
                "adapted_min_xy_linf_error": adapted_xy_error,
                "paired_min_xy_linf_error_delta": (
                    released_xy_error - adapted_xy_error
                ),
            }
        )
    elif task == "put_bottles_dustbin":
        released_placed = int(released_progress["max_placed_bottles"])
        adapted_placed = int(adapted_progress["max_placed_bottles"])
        pair.update(
            {
                "released_max_placed_bottles": released_placed,
                "adapted_max_placed_bottles": adapted_placed,
                "released_first_all_bottles_placed_step": (
                    released_progress.get("first_all_bottles_placed_step")
                ),
                "adapted_first_all_bottles_placed_step": (
                    adapted_progress.get("first_all_bottles_placed_step")
                ),
                "paired_max_placed_bottles_delta": adapted_placed - released_placed,
                "stage_reward_diagnostics_are_selection_input": False,
            }
        )
    elif task == "place_fan":
        released_position_error = float(
            released_progress["min_position_linf_normalized_error"]
        )
        adapted_position_error = float(
            adapted_progress["min_position_linf_normalized_error"]
        )
        released_quaternion_error = float(
            released_progress["min_quaternion_linf_normalized_error"]
        )
        adapted_quaternion_error = float(
            adapted_progress["min_quaternion_linf_normalized_error"]
        )
        pair.update(
            {
                "released_pose_valid": bool(
                    released_progress["max_pose_valid"]
                ),
                "adapted_pose_valid": bool(
                    adapted_progress["max_pose_valid"]
                ),
                "released_first_pose_valid_step": released_progress.get(
                    "first_pose_valid_step"
                ),
                "adapted_first_pose_valid_step": adapted_progress.get(
                    "first_pose_valid_step"
                ),
                "released_min_position_linf_normalized_error": (
                    released_position_error
                ),
                "adapted_min_position_linf_normalized_error": (
                    adapted_position_error
                ),
                "paired_min_position_linf_normalized_error_delta": (
                    released_position_error - adapted_position_error
                ),
                "released_min_quaternion_linf_normalized_error": (
                    released_quaternion_error
                ),
                "adapted_min_quaternion_linf_normalized_error": (
                    adapted_quaternion_error
                ),
                "paired_min_quaternion_linf_normalized_error_delta": (
                    released_quaternion_error - adapted_quaternion_error
                ),
                "pose_diagnostics_are_selection_input": False,
            }
        )
    elif task == "place_shoe":
        released_xy_error = float(
            released_progress["min_xy_linf_normalized_error"]
        )
        adapted_xy_error = float(
            adapted_progress["min_xy_linf_normalized_error"]
        )
        released_quaternion_error = float(
            released_progress["min_quaternion_linf_normalized_error"]
        )
        adapted_quaternion_error = float(
            adapted_progress["min_quaternion_linf_normalized_error"]
        )
        released_max_shoe_z = float(released_progress["max_shoe_z"])
        adapted_max_shoe_z = float(adapted_progress["max_shoe_z"])
        pair.update(
            {
                "released_pose_valid": bool(
                    released_progress["max_pose_valid"]
                ),
                "adapted_pose_valid": bool(
                    adapted_progress["max_pose_valid"]
                ),
                "released_first_pose_valid_step": released_progress.get(
                    "first_pose_valid_step"
                ),
                "adapted_first_pose_valid_step": adapted_progress.get(
                    "first_pose_valid_step"
                ),
                "released_min_xy_linf_normalized_error": released_xy_error,
                "adapted_min_xy_linf_normalized_error": adapted_xy_error,
                "paired_min_xy_linf_normalized_error_delta": (
                    released_xy_error - adapted_xy_error
                ),
                "released_min_quaternion_linf_normalized_error": (
                    released_quaternion_error
                ),
                "adapted_min_quaternion_linf_normalized_error": (
                    adapted_quaternion_error
                ),
                "paired_min_quaternion_linf_normalized_error_delta": (
                    released_quaternion_error - adapted_quaternion_error
                ),
                "released_max_shoe_z": released_max_shoe_z,
                "adapted_max_shoe_z": adapted_max_shoe_z,
                "paired_max_shoe_z_delta": (
                    adapted_max_shoe_z - released_max_shoe_z
                ),
                "shoe_z_is_official": False,
                "shoe_z_is_selection_input": False,
            }
        )
    elif task == "scan_object":
        released_projection_error = float(
            released_progress["min_projected_xyz_linf_error"]
        )
        adapted_projection_error = float(
            adapted_progress["min_projected_xyz_linf_error"]
        )
        pair.update(
            {
                "released_scan_geometry_valid": bool(
                    released_progress["max_scan_geometry_valid"]
                ),
                "adapted_scan_geometry_valid": bool(
                    adapted_progress["max_scan_geometry_valid"]
                ),
                "released_first_scan_geometry_valid_step": (
                    released_progress.get("first_scan_geometry_valid_step")
                ),
                "adapted_first_scan_geometry_valid_step": (
                    adapted_progress.get("first_scan_geometry_valid_step")
                ),
                "released_depth_valid": bool(
                    released_progress["max_depth_valid"]
                ),
                "adapted_depth_valid": bool(
                    adapted_progress["max_depth_valid"]
                ),
                "released_both_grippers_closed": bool(
                    released_progress["max_both_grippers_closed"]
                ),
                "adapted_both_grippers_closed": bool(
                    adapted_progress["max_both_grippers_closed"]
                ),
                "released_min_projected_xyz_linf_error": (
                    released_projection_error
                ),
                "adapted_min_projected_xyz_linf_error": (
                    adapted_projection_error
                ),
                "paired_min_projected_xyz_linf_error_delta": (
                    released_projection_error - adapted_projection_error
                ),
                "continuous_scan_diagnostics_are_selection_input": False,
            }
        )
    elif task == "place_a2b_left":
        released_placement_valid = bool(
            released_progress["max_placement_valid"]
        )
        adapted_placement_valid = bool(
            adapted_progress["max_placement_valid"]
        )
        released_y_error = float(released_progress["min_y_abs_error"])
        adapted_y_error = float(adapted_progress["min_y_abs_error"])
        pair.update(
            {
                "released_placement_valid": released_placement_valid,
                "adapted_placement_valid": adapted_placement_valid,
                "released_first_placement_valid_step": released_progress.get(
                    "first_placement_valid_step"
                ),
                "adapted_first_placement_valid_step": adapted_progress.get(
                    "first_placement_valid_step"
                ),
                "released_min_y_abs_error": released_y_error,
                "adapted_min_y_abs_error": adapted_y_error,
                "paired_min_y_abs_error_delta": (
                    released_y_error - adapted_y_error
                ),
                "placement_diagnostics_are_selection_input": False,
            }
        )
    elif task == "stack_blocks_three":
        released_stacked_pairs = int(
            released_progress["max_stacked_pair_count"]
        )
        adapted_stacked_pairs = int(
            adapted_progress["max_stacked_pair_count"]
        )
        released_stack_error = float(
            released_progress["min_max_pair_linf_normalized_error"]
        )
        adapted_stack_error = float(
            adapted_progress["min_max_pair_linf_normalized_error"]
        )
        pair.update(
            {
                "released_fully_stacked": released_stacked_pairs == 2,
                "adapted_fully_stacked": adapted_stacked_pairs == 2,
                "released_first_fully_stacked_step": released_progress.get(
                    "first_fully_stacked_step"
                ),
                "adapted_first_fully_stacked_step": adapted_progress.get(
                    "first_fully_stacked_step"
                ),
                "released_max_stacked_pair_count": released_stacked_pairs,
                "adapted_max_stacked_pair_count": adapted_stacked_pairs,
                "paired_max_stacked_pair_count_delta": (
                    adapted_stacked_pairs - released_stacked_pairs
                ),
                "released_min_max_pair_linf_normalized_error": (
                    released_stack_error
                ),
                "adapted_min_max_pair_linf_normalized_error": (
                    adapted_stack_error
                ),
                "paired_min_max_pair_linf_normalized_error_delta": (
                    released_stack_error - adapted_stack_error
                ),
                "stacking_diagnostics_are_selection_input": False,
            }
        )
    elif task in ("blocks_ranking_rgb", "blocks_ranking_size"):
        released_ordered_pairs = int(
            released_progress["max_ordered_pair_count"]
        )
        adapted_ordered_pairs = int(
            adapted_progress["max_ordered_pair_count"]
        )
        pair.update(
            {
                "released_fully_ordered": released_ordered_pairs == 2,
                "adapted_fully_ordered": adapted_ordered_pairs == 2,
                "released_first_fully_ordered_step": released_progress.get(
                    "first_fully_ordered_step"
                ),
                "adapted_first_fully_ordered_step": adapted_progress.get(
                    "first_fully_ordered_step"
                ),
                "released_max_ordered_pair_count": released_ordered_pairs,
                "adapted_max_ordered_pair_count": adapted_ordered_pairs,
                "paired_max_ordered_pair_count_delta": (
                    adapted_ordered_pairs - released_ordered_pairs
                ),
                "ordering_diagnostics_are_selection_input": False,
            }
        )
    else:
        raise ValueError(f"unsupported progress pair task: {task!r}")
    return pair


def _run_screen_queue(
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    output_root: Path,
    gpu: int,
    units: Sequence[EvalUnit],
    candidates: Sequence[Candidate],
    record_keys: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in units:
        unit_root = (
            output_root
            / "screen"
            / f"bank_{unit.noise_base_seed}"
            / f"seed_{unit.seed}"
        )
        record = unit.key in record_keys
        released_path = unit_root / "released" / "episode.json"
        released_unit = _run_policy(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            gpu=gpu,
            phase="screening",
            unit=unit,
            condition="released",
            adapter=None,
            expected_adapter_state_sha256=None,
            output=released_path,
            record=record,
        )
        for candidate in candidates:
            adapted_path = unit_root / candidate.condition / "episode.json"
            adapted_unit = _run_policy(
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                gpu=gpu,
                phase="screening",
                unit=unit,
                condition=candidate.condition,
                adapter=candidate.checkpoint,
                expected_adapter_state_sha256=candidate.checkpoint_sha256,
                output=adapted_path,
                record=record,
            )
            pair = _build_pair(
                protocol=protocol,
                unit=unit,
                condition=candidate.condition,
                checkpoint=candidate.checkpoint,
                expected_checkpoint_sha256=candidate.checkpoint_sha256,
                released_path=released_path,
                adapted_path=adapted_path,
                released_unit=released_unit,
                adapted_unit=adapted_unit,
                gpu=gpu,
            )
            _write_json(unit_root / "pairs" / f"{candidate.condition}.json", pair)
            rows.append(pair)
    return rows


def _candidate_aggregate(
    candidate: Candidate,
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
) -> dict[str, Any]:
    hashes = {row.get("checkpoint_sha256") for row in rows}
    if hashes != {candidate.checkpoint_sha256}:
        raise RuntimeError(
            f"{candidate.condition} episode checkpoint hashes differ from candidate"
        )
    released_hashes = {
        row.get("released_checkpoint_sha256") for row in rows
    }
    if released_hashes != {None}:
        raise RuntimeError(
            f"{candidate.condition} Released episodes have adapter hashes"
        )
    adapted_success = sum(int(bool(row["opd_success"])) for row in rows)
    released_success = sum(int(bool(row["released_success"])) for row in rows)
    stage_improvements = sum(int(bool(row["stage_improvement"])) for row in rows)
    stage_regressions = sum(int(bool(row["stage_regression"])) for row in rows)
    stage_delta = sum(int(row["paired_max_ordinal_delta"]) for row in rows)
    selection_key = (
        adapted_success,
        stage_delta,
        stage_improvements - stage_regressions,
        -float(candidate.fixed_calibration_total_loss),
        -int(candidate.epoch),
    )
    aggregate = {
        "epoch": candidate.epoch,
        "condition": candidate.condition,
        "checkpoint": str(candidate.checkpoint),
        "checkpoint_sha256": candidate.checkpoint_sha256,
        "pairs": len(rows),
        "released_success": released_success,
        "adapted_success": adapted_success,
        "rescues": sum(int(bool(row["rescue"])) for row in rows),
        "regressions": sum(int(bool(row["regression"])) for row in rows),
        "paired_net_gain": sum(
            int(bool(row["rescue"])) - int(bool(row["regression"]))
            for row in rows
        ),
        "sum_paired_max_ordinal_delta": stage_delta,
        "stage_improvements": stage_improvements,
        "stage_regressions": stage_regressions,
        "stage_improvement_minus_regression": stage_improvements - stage_regressions,
        "fixed_calibration_total_loss": candidate.fixed_calibration_total_loss,
        "selection_key": list(selection_key),
    }
    if task == "handover_mic":
        aggregate["contact_diagnostics"] = {
            "released_contact_units": sum(
                int(bool(row["released_contact"])) for row in rows
            ),
            "adapted_contact_units": sum(
                int(bool(row["adapted_contact"])) for row in rows
            ),
            "sum_contact_onset_delta": sum(
                int(row["contact_onset_delta"]) for row in rows
            ),
            "released_receiver_contact_units": sum(
                int(bool(row["released_max_receiver_contact"])) for row in rows
            ),
            "adapted_receiver_contact_units": sum(
                int(bool(row["adapted_max_receiver_contact"])) for row in rows
            ),
            "selection_input": False,
        }
    elif task == "open_microwave":
        aggregate["door_diagnostics"] = {
            "released_door_open_units": sum(
                int(bool(row["released_door_open"])) for row in rows
            ),
            "adapted_door_open_units": sum(
                int(bool(row["adapted_door_open"])) for row in rows
            ),
            "sum_paired_max_normalized_progress_delta": sum(
                float(row["paired_max_normalized_progress_delta"])
                for row in rows
            ),
            "selection_input": False,
        }
    elif task == "put_object_cabinet":
        aggregate["placement_diagnostics"] = {
            "released_placement_valid_units": sum(
                int(bool(row["released_placement_valid"])) for row in rows
            ),
            "adapted_placement_valid_units": sum(
                int(bool(row["adapted_placement_valid"])) for row in rows
            ),
            "sum_paired_min_xy_linf_error_delta": sum(
                float(row["paired_min_xy_linf_error_delta"]) for row in rows
            ),
            "selection_input": False,
        }
    elif task == "put_bottles_dustbin":
        aggregate["bottle_progress_diagnostics"] = {
            "released_max_placed_bottles_sum": sum(
                int(row["released_max_placed_bottles"]) for row in rows
            ),
            "adapted_max_placed_bottles_sum": sum(
                int(row["adapted_max_placed_bottles"]) for row in rows
            ),
            "sum_paired_max_placed_bottles_delta": sum(
                int(row["paired_max_placed_bottles_delta"]) for row in rows
            ),
            "selection_input": False,
        }
    elif task == "place_fan":
        aggregate["pose_diagnostics"] = {
            "released_pose_valid_units": sum(
                int(bool(row["released_pose_valid"])) for row in rows
            ),
            "adapted_pose_valid_units": sum(
                int(bool(row["adapted_pose_valid"])) for row in rows
            ),
            "sum_paired_min_position_linf_normalized_error_delta": sum(
                float(row["paired_min_position_linf_normalized_error_delta"])
                for row in rows
            ),
            "sum_paired_min_quaternion_linf_normalized_error_delta": sum(
                float(row["paired_min_quaternion_linf_normalized_error_delta"])
                for row in rows
            ),
            "selection_input": False,
        }
    elif task == "place_shoe":
        aggregate["pose_diagnostics"] = {
            "released_pose_valid_units": sum(
                int(bool(row["released_pose_valid"])) for row in rows
            ),
            "adapted_pose_valid_units": sum(
                int(bool(row["adapted_pose_valid"])) for row in rows
            ),
            "sum_paired_min_xy_linf_normalized_error_delta": sum(
                float(row["paired_min_xy_linf_normalized_error_delta"])
                for row in rows
            ),
            "sum_paired_min_quaternion_linf_normalized_error_delta": sum(
                float(
                    row[
                        "paired_min_quaternion_linf_normalized_error_delta"
                    ]
                )
                for row in rows
            ),
            "sum_paired_max_shoe_z_delta": sum(
                float(row["paired_max_shoe_z_delta"]) for row in rows
            ),
            "shoe_z_is_official": False,
            "selection_input": False,
        }
    elif task == "scan_object":
        aggregate["scan_diagnostics"] = {
            "released_scan_geometry_valid_units": sum(
                int(bool(row["released_scan_geometry_valid"]))
                for row in rows
            ),
            "adapted_scan_geometry_valid_units": sum(
                int(bool(row["adapted_scan_geometry_valid"]))
                for row in rows
            ),
            "released_depth_valid_units": sum(
                int(bool(row["released_depth_valid"])) for row in rows
            ),
            "adapted_depth_valid_units": sum(
                int(bool(row["adapted_depth_valid"])) for row in rows
            ),
            "released_both_grippers_closed_units": sum(
                int(bool(row["released_both_grippers_closed"]))
                for row in rows
            ),
            "adapted_both_grippers_closed_units": sum(
                int(bool(row["adapted_both_grippers_closed"]))
                for row in rows
            ),
            "sum_paired_min_projected_xyz_linf_error_delta": sum(
                float(
                    row["paired_min_projected_xyz_linf_error_delta"]
                )
                for row in rows
            ),
            "selection_input": False,
        }
    elif task == "place_a2b_left":
        aggregate["placement_diagnostics"] = {
            "released_placement_valid_units": sum(
                int(bool(row["released_placement_valid"])) for row in rows
            ),
            "adapted_placement_valid_units": sum(
                int(bool(row["adapted_placement_valid"])) for row in rows
            ),
            "sum_paired_min_y_abs_error_delta": sum(
                float(row["paired_min_y_abs_error_delta"]) for row in rows
            ),
            "selection_input": False,
        }
    elif task == "stack_blocks_three":
        aggregate["stacking_diagnostics"] = {
            "released_fully_stacked_units": sum(
                int(bool(row["released_fully_stacked"])) for row in rows
            ),
            "adapted_fully_stacked_units": sum(
                int(bool(row["adapted_fully_stacked"])) for row in rows
            ),
            "released_max_stacked_pair_count_sum": sum(
                int(row["released_max_stacked_pair_count"]) for row in rows
            ),
            "adapted_max_stacked_pair_count_sum": sum(
                int(row["adapted_max_stacked_pair_count"]) for row in rows
            ),
            "sum_paired_max_stacked_pair_count_delta": sum(
                int(row["paired_max_stacked_pair_count_delta"])
                for row in rows
            ),
            "sum_paired_min_max_pair_linf_normalized_error_delta": sum(
                float(
                    row[
                        "paired_min_max_pair_linf_normalized_error_delta"
                    ]
                )
                for row in rows
            ),
            "selection_input": False,
        }
    elif task in ("blocks_ranking_rgb", "blocks_ranking_size"):
        aggregate["ordering_diagnostics"] = {
            "released_fully_ordered_units": sum(
                int(bool(row["released_fully_ordered"])) for row in rows
            ),
            "adapted_fully_ordered_units": sum(
                int(bool(row["adapted_fully_ordered"])) for row in rows
            ),
            "released_max_ordered_pair_count_sum": sum(
                int(row["released_max_ordered_pair_count"])
                for row in rows
            ),
            "adapted_max_ordered_pair_count_sum": sum(
                int(row["adapted_max_ordered_pair_count"])
                for row in rows
            ),
            "sum_paired_max_ordered_pair_count_delta": sum(
                int(row["paired_max_ordered_pair_count_delta"])
                for row in rows
            ),
            "selection_input": False,
        }
    elif task in GENERIC_STAGE_DIAGNOSTIC_TASKS:
        aggregate["official_stage_diagnostics"] = {
            "released_max_ordinal_stage_sum": sum(
                int(row["released_max_ordinal_stage"]) for row in rows
            ),
            "adapted_max_ordinal_stage_sum": sum(
                int(row["adapted_max_ordinal_stage"]) for row in rows
            ),
            "selection_input": False,
        }
    else:
        raise ValueError(f"unsupported candidate aggregate task: {task!r}")
    return aggregate


def _parallel_queues(
    *, units: Sequence[EvalUnit], gpus: Sequence[int]
) -> list[tuple[int, list[EvalUnit]]]:
    queues: list[list[EvalUnit]] = [[] for _ in gpus]
    for index, unit in enumerate(units):
        queues[index % len(gpus)].append(unit)
    return [
        (gpu, queue)
        for gpu, queue in zip(gpus, queues, strict=True)
        if queue
    ]


def _run_screen(args: argparse.Namespace) -> int:
    protocol, protocol_sha256 = _validate_protocol(args.protocol)
    candidates, training_summary_sha256 = _load_candidates(protocol)
    units = _phase_units(protocol, "screening")
    gpus = _parse_gpus(args.gpus)
    output_root = _require_ssd(Path(protocol["output_root"]), "screen output_root")
    run_config = {
        "mode": "screen",
        "protocol": str(args.protocol.expanduser().resolve()),
        "protocol_sha256": protocol_sha256,
        "training_summary": str(Path(protocol["training_summary"]).expanduser().resolve()),
        "training_summary_sha256": training_summary_sha256,
        "gpus": gpus,
        "units": [unit.__dict__ for unit in units],
        "candidates": [
            {
                "epoch": candidate.epoch,
                "checkpoint": str(candidate.checkpoint),
                "checkpoint_sha256": candidate.checkpoint_sha256,
                "fixed_calibration_total_loss": candidate.fixed_calibration_total_loss,
            }
            for candidate in candidates
        ],
    }
    if args.dry_run:
        print(json.dumps(run_config, indent=2, sort_keys=True))
        return 0
    _write_json(output_root / "screen" / "run_config.json", run_config)
    record_keys = _record_keys(protocol, "screening")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(
                _run_screen_queue,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                output_root=output_root,
                gpu=gpu,
                units=queue,
                candidates=candidates,
                record_keys=record_keys,
            )
            for gpu, queue in _parallel_queues(units=units, gpus=gpus)
        ]
        for future in futures:
            rows.extend(future.result())
    rows.sort(key=lambda row: (str(row["condition"]), int(row["noise_base_seed"]), int(row["seed"])))
    aggregates = []
    for candidate in candidates:
        candidate_rows = [row for row in rows if row["condition"] == candidate.condition]
        if len(candidate_rows) != len(units):
            raise RuntimeError(
                f"{candidate.condition} has {len(candidate_rows)} pairs, expected {len(units)}"
            )
        aggregates.append(
            _candidate_aggregate(
                candidate, candidate_rows, task=str(protocol["task"])
            )
        )
    for candidate in candidates:
        if _sha256(candidate.checkpoint) != candidate.checkpoint_sha256:
            raise RuntimeError(
                f"{candidate.condition} checkpoint changed during screening"
            )
    selected = max(aggregates, key=lambda row: tuple(row["selection_key"]))
    summary = {
        "schema": _task_schema(str(protocol["task"]), "screen_summary_schema"),
        "status": "PASS",
        "task": protocol["task"],
        "task_config": protocol["task_config"],
        "screen_units": len(units),
        "released_episodes": len(units),
        "released_baseline_reused_across_checkpoints": True,
        "adapted_episodes": len(units) * len(candidates),
        "recorded_pair_keys": [list(key) for key in sorted(record_keys)],
        "recorded_pair_videos": [
            {
                "condition": row["condition"],
                "noise_base_seed": row["noise_base_seed"],
                "seed": row["seed"],
                "released_composite_mp4": row["released_composite_mp4"],
                "adapted_composite_mp4": row["adapted_composite_mp4"],
            }
            for row in rows
            if row["released_composite_mp4"] is not None
            or row["adapted_composite_mp4"] is not None
        ],
        "selection_rule": list(EXPECTED_SELECTION_RULE),
        **_diagnostic_selection_metadata(str(protocol["task"])),
        "candidates": aggregates,
        "selected_epoch": int(selected["epoch"]),
        "selected_checkpoint": selected["checkpoint"],
    }
    _write_json(output_root / "screen" / "summary.json", summary)
    selection = {
        "schema": _task_schema(str(protocol["task"]), "selection_schema"),
        "status": "PASS",
        "protocol": str(args.protocol.expanduser().resolve()),
        "protocol_sha256": protocol_sha256,
        "training_summary_sha256": training_summary_sha256,
        "screen_summary": str((output_root / "screen" / "summary.json").resolve()),
        "selection_rule": list(EXPECTED_SELECTION_RULE),
        **_diagnostic_selection_metadata(str(protocol["task"])),
        "selected_epoch": int(selected["epoch"]),
        "selected_condition": selected["condition"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_fixed_calibration_total_loss": selected[
            "fixed_calibration_total_loss"
        ],
        "selected_selection_key": selected["selection_key"],
    }
    _write_json(output_root / "screen" / "selection.json", selection)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


def _run_screen_epoch(args: argparse.Namespace) -> int:
    """Pre-compute one epoch's screen episodes before final training summary."""

    protocol, protocol_sha256 = _validate_protocol(args.protocol)
    epoch = int(args.epoch)
    if epoch not in protocol["epoch_ids"]:
        raise ValueError(f"screen epoch is not frozen in protocol: {epoch}")
    summary_path = Path(protocol["training_summary"]).expanduser().resolve()
    training_summary = (
        _read_json(summary_path) if summary_path.is_file() else {}
    )
    checkpoint, checkpoint_sha256_before_eval = (
        _resolve_epoch_candidate_checkpoint(
            epoch=epoch,
            training_summary=training_summary,
            summary_path=summary_path,
            protocol=protocol,
        )
    )
    candidate = Candidate(
        epoch=epoch,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256_before_eval,
        # The fixed calibration value is consumed only by final `screen`,
        # after training has atomically written summary.json.
        fixed_calibration_total_loss=float("nan"),
    )
    units = _phase_units(protocol, "screening")
    gpus = _parse_gpus(args.gpus)
    output_root = _require_ssd(Path(protocol["output_root"]), "screen output_root")
    run_config = {
        "mode": "screen_epoch",
        "epoch": epoch,
        "protocol": str(args.protocol.expanduser().resolve()),
        "protocol_sha256": protocol_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before_eval": checkpoint_sha256_before_eval,
        "gpus": gpus,
        "units": [unit.__dict__ for unit in units],
    }
    if args.dry_run:
        print(json.dumps(run_config, indent=2, sort_keys=True))
        return 0
    _write_json(
        output_root / "screen" / f"epoch_{epoch:02d}_run_config.json",
        run_config,
    )
    record_keys = _record_keys(protocol, "screening")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(
                _run_screen_queue,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                output_root=output_root,
                gpu=gpu,
                units=queue,
                candidates=[candidate],
                record_keys=record_keys,
            )
            for gpu, queue in _parallel_queues(units=units, gpus=gpus)
        ]
        for future in futures:
            rows.extend(future.result())
    rows.sort(key=lambda row: (int(row["noise_base_seed"]), int(row["seed"])))
    if len(rows) != len(units):
        raise RuntimeError(
            f"epoch {epoch} screen has {len(rows)} pairs, expected {len(units)}"
        )
    hashes = {row.get("checkpoint_sha256") for row in rows}
    if hashes != {checkpoint_sha256_before_eval}:
        raise RuntimeError(
            f"epoch {epoch} episode hashes differ from checkpoint_sha256_before_eval"
        )
    if {
        row.get("released_checkpoint_sha256") for row in rows
    } != {None}:
        raise RuntimeError(f"epoch {epoch} Released episodes have adapter hashes")
    if _sha256(checkpoint) != checkpoint_sha256_before_eval:
        raise RuntimeError(f"epoch {epoch} checkpoint changed during screening")
    receipt = {
        "schema": _task_schema(
            str(protocol["task"]), "screen_epoch_receipt_schema"
        ),
        "status": "PASS",
        **run_config,
        "checkpoint_sha256_from_episodes": checkpoint_sha256_before_eval,
        "released_baseline_reused_if_present": True,
        "pairs": len(rows),
        "adapted_success": sum(int(bool(row["opd_success"])) for row in rows),
        "sum_paired_max_ordinal_delta": sum(
            int(row["paired_max_ordinal_delta"]) for row in rows
        ),
        "pair_receipts": [
            str(
                (
                    output_root
                    / "screen"
                    / f"bank_{row['noise_base_seed']}"
                    / f"seed_{row['seed']}"
                    / "pairs"
                    / f"epoch_{epoch:02d}.json"
                ).resolve()
            )
            for row in rows
        ],
        "selection_performed": False,
    }
    _write_json(
        output_root / "screen" / f"epoch_{epoch:02d}_receipt.json", receipt
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _run_heldout_queue(
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    output_root: Path,
    gpu: int,
    units: Sequence[EvalUnit],
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    record_keys: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in units:
        unit_root = (
            output_root
            / "heldout"
            / f"bank_{unit.noise_base_seed}"
            / f"seed_{unit.seed}"
        )
        record = unit.key in record_keys
        released_path = unit_root / "released" / "episode.json"
        adapted_path = unit_root / "adapted" / "episode.json"
        released_unit = _run_policy(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            gpu=gpu,
            phase="heldout",
            unit=unit,
            condition="released",
            adapter=None,
            expected_adapter_state_sha256=None,
            output=released_path,
            record=record,
        )
        adapted_unit = _run_policy(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            gpu=gpu,
            phase="heldout",
            unit=unit,
            condition="adapted",
            adapter=checkpoint,
            expected_adapter_state_sha256=expected_checkpoint_sha256,
            output=adapted_path,
            record=record,
        )
        pair = _build_pair(
            protocol=protocol,
            unit=unit,
            condition="adapted",
            checkpoint=checkpoint,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            released_path=released_path,
            adapted_path=adapted_path,
            released_unit=released_unit,
            adapted_unit=adapted_unit,
            gpu=gpu,
        )
        _write_json(unit_root / "pair.json", pair)
        rows.append(pair)
    return rows


def _heldout_bucket(
    rows: Sequence[Mapping[str, Any]], *, task: str
) -> dict[str, Any]:
    bucket = {
        "pairs": len(rows),
        "released_success": sum(int(bool(row["released_success"])) for row in rows),
        "adapted_success": sum(int(bool(row["opd_success"])) for row in rows),
        "rescues": sum(int(bool(row["rescue"])) for row in rows),
        "regressions": sum(int(bool(row["regression"])) for row in rows),
        "paired_net_gain": sum(
            int(bool(row["rescue"])) - int(bool(row["regression"]))
            for row in rows
        ),
        "sum_paired_max_ordinal_delta": sum(
            int(row["paired_max_ordinal_delta"]) for row in rows
        ),
        "stage_improvements": sum(int(bool(row["stage_improvement"])) for row in rows),
        "stage_regressions": sum(int(bool(row["stage_regression"])) for row in rows),
    }
    if task == "handover_mic":
        bucket["contact_diagnostics"] = {
            "released_contact_units": sum(
                int(bool(row["released_contact"])) for row in rows
            ),
            "adapted_contact_units": sum(
                int(bool(row["adapted_contact"])) for row in rows
            ),
            "sum_contact_onset_delta": sum(
                int(row["contact_onset_delta"]) for row in rows
            ),
            "released_receiver_contact_units": sum(
                int(bool(row["released_max_receiver_contact"])) for row in rows
            ),
            "adapted_receiver_contact_units": sum(
                int(bool(row["adapted_max_receiver_contact"])) for row in rows
            ),
            "selection_input": False,
        }
    elif task == "open_microwave":
        bucket["door_diagnostics"] = {
            "released_door_open_units": sum(
                int(bool(row["released_door_open"])) for row in rows
            ),
            "adapted_door_open_units": sum(
                int(bool(row["adapted_door_open"])) for row in rows
            ),
            "sum_paired_max_normalized_progress_delta": sum(
                float(row["paired_max_normalized_progress_delta"])
                for row in rows
            ),
            "selection_input": False,
        }
    elif task == "put_object_cabinet":
        bucket["placement_diagnostics"] = {
            "released_placement_valid_units": sum(
                int(bool(row["released_placement_valid"])) for row in rows
            ),
            "adapted_placement_valid_units": sum(
                int(bool(row["adapted_placement_valid"])) for row in rows
            ),
            "sum_paired_min_xy_linf_error_delta": sum(
                float(row["paired_min_xy_linf_error_delta"]) for row in rows
            ),
            "selection_input": False,
        }
    elif task == "put_bottles_dustbin":
        bucket["bottle_progress_diagnostics"] = {
            "released_max_placed_bottles_sum": sum(
                int(row["released_max_placed_bottles"]) for row in rows
            ),
            "adapted_max_placed_bottles_sum": sum(
                int(row["adapted_max_placed_bottles"]) for row in rows
            ),
            "sum_paired_max_placed_bottles_delta": sum(
                int(row["paired_max_placed_bottles_delta"]) for row in rows
            ),
            "selection_input": False,
        }
    elif task == "place_fan":
        bucket["pose_diagnostics"] = {
            "released_pose_valid_units": sum(
                int(bool(row["released_pose_valid"])) for row in rows
            ),
            "adapted_pose_valid_units": sum(
                int(bool(row["adapted_pose_valid"])) for row in rows
            ),
            "sum_paired_min_position_linf_normalized_error_delta": sum(
                float(row["paired_min_position_linf_normalized_error_delta"])
                for row in rows
            ),
            "sum_paired_min_quaternion_linf_normalized_error_delta": sum(
                float(row["paired_min_quaternion_linf_normalized_error_delta"])
                for row in rows
            ),
            "selection_input": False,
        }
    elif task == "place_shoe":
        bucket["pose_diagnostics"] = {
            "released_pose_valid_units": sum(
                int(bool(row["released_pose_valid"])) for row in rows
            ),
            "adapted_pose_valid_units": sum(
                int(bool(row["adapted_pose_valid"])) for row in rows
            ),
            "sum_paired_min_xy_linf_normalized_error_delta": sum(
                float(row["paired_min_xy_linf_normalized_error_delta"])
                for row in rows
            ),
            "sum_paired_min_quaternion_linf_normalized_error_delta": sum(
                float(
                    row[
                        "paired_min_quaternion_linf_normalized_error_delta"
                    ]
                )
                for row in rows
            ),
            "sum_paired_max_shoe_z_delta": sum(
                float(row["paired_max_shoe_z_delta"]) for row in rows
            ),
            "shoe_z_is_official": False,
            "selection_input": False,
        }
    elif task == "scan_object":
        bucket["scan_diagnostics"] = {
            "released_scan_geometry_valid_units": sum(
                int(bool(row["released_scan_geometry_valid"]))
                for row in rows
            ),
            "adapted_scan_geometry_valid_units": sum(
                int(bool(row["adapted_scan_geometry_valid"]))
                for row in rows
            ),
            "released_depth_valid_units": sum(
                int(bool(row["released_depth_valid"])) for row in rows
            ),
            "adapted_depth_valid_units": sum(
                int(bool(row["adapted_depth_valid"])) for row in rows
            ),
            "released_both_grippers_closed_units": sum(
                int(bool(row["released_both_grippers_closed"]))
                for row in rows
            ),
            "adapted_both_grippers_closed_units": sum(
                int(bool(row["adapted_both_grippers_closed"]))
                for row in rows
            ),
            "sum_paired_min_projected_xyz_linf_error_delta": sum(
                float(
                    row["paired_min_projected_xyz_linf_error_delta"]
                )
                for row in rows
            ),
            "selection_input": False,
        }
    elif task == "place_a2b_left":
        bucket["placement_diagnostics"] = {
            "released_placement_valid_units": sum(
                int(bool(row["released_placement_valid"])) for row in rows
            ),
            "adapted_placement_valid_units": sum(
                int(bool(row["adapted_placement_valid"])) for row in rows
            ),
            "sum_paired_min_y_abs_error_delta": sum(
                float(row["paired_min_y_abs_error_delta"]) for row in rows
            ),
            "selection_input": False,
        }
    elif task == "stack_blocks_three":
        bucket["stacking_diagnostics"] = {
            "released_fully_stacked_units": sum(
                int(bool(row["released_fully_stacked"])) for row in rows
            ),
            "adapted_fully_stacked_units": sum(
                int(bool(row["adapted_fully_stacked"])) for row in rows
            ),
            "released_max_stacked_pair_count_sum": sum(
                int(row["released_max_stacked_pair_count"]) for row in rows
            ),
            "adapted_max_stacked_pair_count_sum": sum(
                int(row["adapted_max_stacked_pair_count"]) for row in rows
            ),
            "sum_paired_max_stacked_pair_count_delta": sum(
                int(row["paired_max_stacked_pair_count_delta"])
                for row in rows
            ),
            "sum_paired_min_max_pair_linf_normalized_error_delta": sum(
                float(
                    row[
                        "paired_min_max_pair_linf_normalized_error_delta"
                    ]
                )
                for row in rows
            ),
            "selection_input": False,
        }
    elif task in ("blocks_ranking_rgb", "blocks_ranking_size"):
        bucket["ordering_diagnostics"] = {
            "released_fully_ordered_units": sum(
                int(bool(row["released_fully_ordered"])) for row in rows
            ),
            "adapted_fully_ordered_units": sum(
                int(bool(row["adapted_fully_ordered"])) for row in rows
            ),
            "released_max_ordered_pair_count_sum": sum(
                int(row["released_max_ordered_pair_count"])
                for row in rows
            ),
            "adapted_max_ordered_pair_count_sum": sum(
                int(row["adapted_max_ordered_pair_count"])
                for row in rows
            ),
            "sum_paired_max_ordered_pair_count_delta": sum(
                int(row["paired_max_ordered_pair_count_delta"])
                for row in rows
            ),
            "selection_input": False,
        }
    elif task in GENERIC_STAGE_DIAGNOSTIC_TASKS:
        bucket["official_stage_diagnostics"] = {
            "released_max_ordinal_stage_sum": sum(
                int(row["released_max_ordinal_stage"]) for row in rows
            ),
            "adapted_max_ordinal_stage_sum": sum(
                int(row["adapted_max_ordinal_stage"]) for row in rows
            ),
            "selection_input": False,
        }
    else:
        raise ValueError(f"unsupported heldout aggregate task: {task!r}")
    return bucket


def _run_heldout(args: argparse.Namespace) -> int:
    protocol, protocol_sha256 = _validate_protocol(args.protocol)
    output_root = _require_ssd(Path(protocol["output_root"]), "heldout output_root")
    selection_path = output_root / "screen" / "selection.json"
    selection = _read_json(selection_path)
    if (
        selection.get("schema")
        != _task_schema(str(protocol["task"]), "selection_schema")
        or selection.get("status") != "PASS"
    ):
        raise RuntimeError("selection receipt is missing or not PASS")
    if selection.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError("selection receipt belongs to a different protocol")
    if tuple(selection.get("selection_rule", ())) != EXPECTED_SELECTION_RULE:
        raise RuntimeError("selection receipt rule differs from frozen rule")
    checkpoint = _require_ssd(
        Path(selection["selected_checkpoint"]), "selected checkpoint"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"selected checkpoint is missing: {checkpoint}")
    selected_checkpoint_sha256 = selection.get("selected_checkpoint_sha256")
    if (
        not isinstance(selected_checkpoint_sha256, str)
        or _sha256(checkpoint) != selected_checkpoint_sha256
    ):
        raise RuntimeError("selected checkpoint hash differs from selection receipt")

    units = _phase_units(protocol, "heldout")
    gpus = _parse_gpus(args.gpus)
    run_config = {
        "mode": "heldout",
        "protocol": str(args.protocol.expanduser().resolve()),
        "protocol_sha256": protocol_sha256,
        "selection_receipt": str(selection_path.resolve()),
        "selected_epoch": int(selection["selected_epoch"]),
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": selected_checkpoint_sha256,
        "gpus": gpus,
        "units": [unit.__dict__ for unit in units],
    }
    if args.dry_run:
        print(json.dumps(run_config, indent=2, sort_keys=True))
        return 0
    _write_json(output_root / "heldout" / "run_config.json", run_config)
    record_keys = _record_keys(protocol, "heldout")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(
                _run_heldout_queue,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                output_root=output_root,
                gpu=gpu,
                units=queue,
                checkpoint=checkpoint,
                expected_checkpoint_sha256=selected_checkpoint_sha256,
                record_keys=record_keys,
            )
            for gpu, queue in _parallel_queues(units=units, gpus=gpus)
        ]
        for future in futures:
            rows.extend(future.result())
    rows.sort(key=lambda row: (int(row["noise_base_seed"]), int(row["seed"])))
    if len(rows) != len(units):
        raise RuntimeError(f"heldout has {len(rows)} pairs, expected {len(units)}")
    if {
        row.get("checkpoint_sha256") for row in rows
    } != {selected_checkpoint_sha256}:
        raise RuntimeError(
            "heldout adapted episode hashes differ from selected checkpoint"
        )
    if {
        row.get("released_checkpoint_sha256") for row in rows
    } != {None}:
        raise RuntimeError("heldout Released episodes have adapter hashes")
    banks = sorted({int(row["noise_base_seed"]) for row in rows})
    summary = {
        "schema": _task_schema(str(protocol["task"]), "heldout_summary_schema"),
        "status": "PASS",
        "task": protocol["task"],
        "task_config": protocol["task_config"],
        "selection_receipt": str(selection_path.resolve()),
        "selected_epoch": int(selection["selected_epoch"]),
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": selected_checkpoint_sha256,
        "recorded_pair_keys": [list(key) for key in sorted(record_keys)],
        "recorded_pair_videos": [
            {
                "noise_base_seed": row["noise_base_seed"],
                "seed": row["seed"],
                "released_composite_mp4": row["released_composite_mp4"],
                "adapted_composite_mp4": row["adapted_composite_mp4"],
            }
            for row in rows
            if row["released_composite_mp4"] is not None
            or row["adapted_composite_mp4"] is not None
        ],
        **_heldout_bucket(rows, task=str(protocol["task"])),
        "by_bank": {
            str(bank): _heldout_bucket(
                [row for row in rows if int(row["noise_base_seed"]) == bank],
                task=str(protocol["task"]),
            )
            for bank in banks
        },
        "pair_receipts": [
            str(
                (
                    output_root
                    / "heldout"
                    / f"bank_{row['noise_base_seed']}"
                    / f"seed_{row['seed']}"
                    / "pair.json"
                ).resolve()
            )
            for row in rows
        ],
    }
    if _sha256(checkpoint) != selected_checkpoint_sha256:
        raise RuntimeError("selected checkpoint changed before heldout commit")
    _write_json(output_root / "heldout" / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("screen", "heldout"):
        phase = subparsers.add_parser(command)
        phase.add_argument("--protocol", type=Path, required=True)
        phase.add_argument("--gpus", default="3,5,6,7")
        phase.add_argument("--dry-run", action="store_true")

    screen_epoch = subparsers.add_parser("screen-epoch")
    screen_epoch.add_argument("--protocol", type=Path, required=True)
    screen_epoch.add_argument("--epoch", type=int, choices=(1, 2, 3), required=True)
    screen_epoch.add_argument("--gpus", default="3,5,6,7")
    screen_epoch.add_argument("--dry-run", action="store_true")

    unit = subparsers.add_parser("unit")
    unit.add_argument("--phase", choices=("screening", "heldout"), required=True)
    unit.add_argument("--condition", required=True)
    unit.add_argument("--task", required=True)
    unit.add_argument("--task-config", required=True)
    unit.add_argument("--seed", type=int, required=True)
    unit.add_argument("--prompt", required=True)
    unit.add_argument("--noise-base-seed", type=int, required=True)
    unit.add_argument("--chunks", type=int, required=True)
    unit.add_argument("--max-control-steps", type=int, required=True)
    unit.add_argument("--student", type=Path, required=True)
    unit.add_argument("--project-root", type=Path, required=True)
    unit.add_argument("--device", default="cuda:0")
    unit.add_argument("--output", type=Path, required=True)
    unit.add_argument("--adapter-state", type=Path)
    unit.add_argument("--protocol-sha256", required=True)
    unit.add_argument("--record", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "unit":
        return _run_unit(args)
    if args.command == "screen":
        return _run_screen(args)
    if args.command == "screen-epoch":
        return _run_screen_epoch(args)
    if args.command == "heldout":
        return _run_heldout(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
