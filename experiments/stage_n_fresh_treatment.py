"""Fail-closed construction of one fresh Stage-N live treatment."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

try:
    from .robotwin_branch_oracle import array_sha256
    from .stage_m_live_bridge_contract import (
        LIVE_CONTEXT_SCHEMA,
        semantic_sha256 as _stage_m_semantic_sha256,
    )
except ImportError:  # Direct execution from the experiments/ script directory.
    from robotwin_branch_oracle import array_sha256
    from stage_m_live_bridge_contract import (
        LIVE_CONTEXT_SCHEMA,
        semantic_sha256 as _stage_m_semantic_sha256,
    )


TREATMENT_SHAPE = (16, 2, 16)
REGISTERED_ALPHAS = (0.25, 0.5, 1.0)
SIMULATOR_SNAPSHOT_SCHEMA = "robotwin_simulator_snapshot_v2"
MAX_INTERVENTIONS = 4


def parse_intervention_frames(
    value: str | None,
    *,
    default_frame: int,
) -> tuple[int, ...]:
    """Parse a bounded, aligned sequence of live intervention frames."""

    default = int(default_frame)
    raw = str(value).strip() if value is not None else ""
    frames = tuple(int(item.strip()) for item in raw.split(",")) if raw else (default,)
    if not frames or frames[0] != default:
        raise ValueError("Stage N intervention schedule must start at the oracle frame")
    if len(frames) > MAX_INTERVENTIONS:
        raise ValueError(f"Stage N allows at most {MAX_INTERVENTIONS} interventions")
    if any(frame <= 0 or frame % 2 for frame in frames):
        raise ValueError("Stage N intervention frames must be positive 2-frame boundaries")
    if tuple(sorted(set(frames))) != frames:
        raise ValueError("Stage N intervention frames must be unique and increasing")
    return frames


def validate_completed_schedule(
    *,
    applied_count: int,
    recorded_count: int,
    expected_count: int,
    success: bool,
    pending: bool,
) -> dict[str, object]:
    """Validate a bounded schedule, allowing official success to stop it early."""

    applied = int(applied_count)
    recorded = int(recorded_count)
    expected = int(expected_count)
    if pending:
        raise ValueError("Stage N reached terminal with a pending treatment")
    if applied != recorded:
        raise ValueError("Stage N applied and recorded treatment counts differ")
    if expected < 1 or applied < 1 or applied > expected:
        raise ValueError("Stage N treatment count is outside its schedule")
    if applied < expected and not bool(success):
        raise ValueError("Stage N unsuccessful episode missed scheduled treatments")
    return {
        "application_count": applied,
        "application_expected": expected,
        "early_terminal": bool(applied < expected),
        "skipped_treatments": expected - applied,
    }


def fresh_history_context_sha256(context: object) -> str:
    """Hash causal live-context content, excluding capture-time provenance."""

    if not isinstance(context, dict) or context.get("schema") != LIVE_CONTEXT_SCHEMA:
        raise ValueError("Stage N requires a Stage-M live context")
    causal = dict(context)
    causal.pop("semantic_sha256", None)
    causal.pop("capture_started_unix_ns", None)
    return _stage_m_semantic_sha256(causal)


class FreshPlannerEventTrace:
    """Count continuation planner outcomes without changing planner behavior."""

    def __init__(self, robot: object) -> None:
        self.robot = robot
        self._original = {
            "left": robot.left_plan_path,
            "right": robot.right_plan_path,
        }
        self._counts = {"left": 0, "right": 0}
        self._status_counts: dict[str, dict[str, int]] = {
            "left": {},
            "right": {},
        }
        self._non_success = 0
        self._exceptions = 0
        self._active = True
        robot.left_plan_path = self._wrapper("left")
        robot.right_plan_path = self._wrapper("right")

    def _wrapper(self, arm: str):
        def traced(target_pose, *args, **kwargs):
            try:
                result = self._original[arm](target_pose, *args, **kwargs)
            except Exception:
                self._exceptions += 1
                raise
            status = str(result.get("status", ""))
            self._counts[arm] += 1
            self._status_counts[arm][status] = (
                self._status_counts[arm].get(status, 0) + 1
            )
            if status != "Success":
                self._non_success += 1
            return result

        return traced

    def snapshot(self) -> dict[str, Any]:
        if not self._active:
            raise AssertionError("planner event trace is not active")
        return {
            "schema": "flashwam_stage_n_planner_events_v1",
            "call_counts": dict(self._counts),
            "status_counts": {
                arm: dict(counts) for arm, counts in self._status_counts.items()
            },
            "non_success_calls": int(self._non_success),
            "exception_calls": int(self._exceptions),
        }

    def finish(self) -> dict[str, Any]:
        payload = self.snapshot()
        self.robot.left_plan_path = self._original["left"]
        self.robot.right_plan_path = self._original["right"]
        self._active = False
        return payload


class FreshPlannerPrefixTrace:
    """Passive, deployment-free fingerprint of planners before treatment."""

    def __init__(self, robot: object) -> None:
        self.robot = robot
        self._original = {
            "left": robot.left_plan_path,
            "right": robot.right_plan_path,
        }
        self._calls: list[dict[str, Any]] = []
        self._counts = {"left": 0, "right": 0}
        self._active = True
        robot.left_plan_path = self._wrapper("left")
        robot.right_plan_path = self._wrapper("right")

    @staticmethod
    def _array_payload(value: object) -> dict[str, Any]:
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(
            array
        ).all():
            raise ValueError("planner fingerprint arrays must be finite")
        return {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "values": array.tolist(),
            "sha256": array_sha256(array),
        }

    def _wrapper(self, arm: str):
        def traced(target_pose, *args, **kwargs):
            entity = (
                self.robot.left_entity
                if arm == "left"
                else self.robot.right_entity
            )
            start = self._array_payload(entity.get_qpos())
            target = self._array_payload(target_pose)
            result = self._original[arm](target_pose, *args, **kwargs)
            raw_position = result.get("position")
            raw_velocity = result.get("velocity")
            position = self._array_payload(
                [] if raw_position is None else raw_position
            )
            velocity = self._array_payload(
                [] if raw_velocity is None else raw_velocity
            )
            self._calls.append(
                {
                    "index": len(self._calls),
                    "arm_index": self._counts[arm],
                    "arm": arm,
                    "start_qpos": start["values"],
                    "start_qpos_dtype": start["dtype"],
                    "start_qpos_shape": start["shape"],
                    "start_qpos_sha256": start["sha256"],
                    "target_pose": target["values"],
                    "target_pose_dtype": target["dtype"],
                    "target_pose_shape": target["shape"],
                    "target_pose_sha256": target["sha256"],
                    "status": str(result.get("status", "")),
                    "position_shape": position["shape"],
                    "position_sha256": position["sha256"],
                    "velocity_shape": velocity["shape"],
                    "velocity_sha256": velocity["sha256"],
                }
            )
            self._counts[arm] += 1
            return result

        return traced

    def event_snapshot(self) -> dict[str, Any]:
        """Summarize planner outcomes without ending the prefix fingerprint."""

        if not self._active:
            raise AssertionError("planner prefix trace is not active")
        status_counts: dict[str, dict[str, int]] = {
            "left": {},
            "right": {},
        }
        non_success = 0
        for item in self._calls:
            arm = str(item["arm"])
            status = str(item["status"])
            status_counts[arm][status] = status_counts[arm].get(status, 0) + 1
            if status != "Success":
                non_success += 1
        return {
            "schema": "flashwam_stage_n_planner_events_v1",
            "call_counts": dict(self._counts),
            "status_counts": status_counts,
            "non_success_calls": non_success,
            "exception_calls": 0,
        }

    def finish(self) -> dict[str, Any]:
        if not self._active:
            raise AssertionError("planner prefix trace was already finished")
        self.robot.left_plan_path = self._original["left"]
        self.robot.right_plan_path = self._original["right"]
        self._active = False
        payload = {
            "schema": "flashwam_stage_n_planner_prefix_v1",
            "call_counts": dict(self._counts),
            "calls": list(self._calls),
        }
        payload["sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        return payload


def build_intervention_provenance(
    *,
    prefix_runtime_sha256: object,
    snapshot: object,
) -> dict[str, Any]:
    """Freeze the intervention-time runtime and exposed qpos/qvel state."""

    runtimes = list(prefix_runtime_sha256)  # Copy: continuation mutates its list.
    if not runtimes or not all(
        isinstance(value, str) and value for value in runtimes
    ):
        raise ValueError("Stage N requires non-empty prefix runtime hashes")
    if not isinstance(snapshot, dict) or snapshot.get("schema") != (
        SIMULATOR_SNAPSHOT_SCHEMA
    ):
        raise ValueError("Stage N requires a RoboTwin simulator snapshot")
    raw_articulations = snapshot.get("articulations")
    if not isinstance(raw_articulations, list) or not raw_articulations:
        raise ValueError("Stage N snapshot lacks articulations")

    articulations = []
    for item in raw_articulations:
        if not isinstance(item, dict):
            raise ValueError("Stage N articulation state must be a dictionary")
        qpos = np.asarray(item.get("qpos"))
        qvel = np.asarray(item.get("qvel"))
        if not (
            np.issubdtype(qpos.dtype, np.number)
            and np.issubdtype(qvel.dtype, np.number)
            and np.isfinite(qpos).all()
            and np.isfinite(qvel).all()
        ):
            raise ValueError("Stage N articulation qpos/qvel must be finite")
        articulations.append(
            {
                "index": int(item.get("index", len(articulations))),
                "name": str(item.get("name", "")),
                "qpos_dtype": str(qpos.dtype),
                "qpos_shape": list(qpos.shape),
                "qpos": qpos.tolist(),
                "qpos_sha256": array_sha256(qpos),
                "qvel_dtype": str(qvel.dtype),
                "qvel_shape": list(qvel.shape),
                "qvel": qvel.tolist(),
                "qvel_sha256": array_sha256(qvel),
            }
        )
    exposed = {
        "schema": "flashwam_stage_n_exposed_qpos_qvel_v1",
        "articulations": articulations,
    }
    exposed["sha256"] = hashlib.sha256(
        json.dumps(exposed, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "prefix_runtime_sha256": runtimes,
        "exposed_qpos_qvel": exposed,
    }


def _action(label: dict[str, object], key: str) -> np.ndarray:
    if key not in label:
        raise ValueError(f"live label lacks {key}")
    value = np.asarray(label[key])
    if value.shape != TREATMENT_SHAPE:
        raise ValueError(
            f"live label {key} shape must be {TREATMENT_SHAPE}, got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"live label {key} contains a non-finite value")
    return value


def select_live_treatment(
    *,
    actual_student_action: object,
    label: dict[str, object],
    arm: str,
    alpha: float | None,
) -> dict[str, Any]:
    """Verify the live Student source and select exactly one bounded action."""

    if label.get("artifact_kind") != "stage_m_live_teacher_bridge_label":
        raise ValueError("Stage N requires a Stage-M live Teacher-Bridge label")
    student = _action(label, "student_env_action")
    teacher = _action(label, "teacher_bridge_env_action")
    actual = np.asarray(actual_student_action)
    if actual.shape != TREATMENT_SHAPE or not np.array_equal(actual, student):
        raise ValueError("live source Student action differs from the GPU7 label")

    if arm == "base":
        if alpha is not None:
            raise ValueError("base treatment must not specify alpha")
        coefficient = 0.0
    elif arm == "correction":
        coefficient = float(alpha) if alpha is not None else float("nan")
        if coefficient not in REGISTERED_ALPHAS:
            raise ValueError(
                "correction alpha must be pre-registered in "
                f"{REGISTERED_ALPHAS}, got {alpha!r}"
            )
    else:
        raise ValueError("Stage N treatment arm must be 'base' or 'correction'")

    full_residual = teacher.astype(np.float64) - student.astype(np.float64)
    treatment = (
        student.astype(np.float64) + coefficient * full_residual
    ).astype(student.dtype)
    lower = np.minimum(student, teacher)
    upper = np.maximum(student, teacher)
    convex_bounded = bool(
        np.all(treatment >= lower) and np.all(treatment <= upper)
    )
    if not convex_bounded or not np.isfinite(treatment).all():
        raise AssertionError("proximal treatment left the Student-Teacher envelope")

    treatment_delta = treatment.astype(np.float64) - student.astype(np.float64)
    return {
        "action": treatment,
        "record": {
            "schema": "flashwam_stage_n_single_treatment_v1",
            "live_context_id": str(label.get("live_context_id", "")),
            "arm": arm,
            "alpha": coefficient,
            "application_count": 1,
            "source_student_action_exact": True,
            "coordinatewise_convex_bounded": convex_bounded,
            "student_action_sha256": array_sha256(student),
            "teacher_bridge_action_sha256": array_sha256(teacher),
            "treatment_action_sha256": array_sha256(treatment),
            "full_bridge_rmse": float(np.sqrt(np.mean(full_residual ** 2))),
            "full_bridge_max_abs": float(np.max(np.abs(full_residual))),
            "treatment_rmse": float(np.sqrt(np.mean(treatment_delta ** 2))),
            "treatment_max_abs": float(np.max(np.abs(treatment_delta))),
        },
    }


def validate_live_pause_contract(payload: object) -> dict[str, object]:
    """Require that Teacher labeling did not alter any live causal input."""

    if not isinstance(payload, dict):
        raise ValueError("Stage N pause contract must be a dictionary")
    if payload.get("schema") != "flashwam_stage_m_pause_contract_v1":
        raise ValueError("Stage N requires the Stage-M pause contract schema")
    if int(payload.get("simulator_step_before", -1)) != int(
        payload.get("simulator_step_after", -2)
    ):
        raise ValueError("simulator steps differ across the Teacher pause")
    required = {
        "simulator_full_state_exact": "simulator state",
        "observation_exact": "observation",
        "gpu6_gpu7_runtime_exact": "runtime",
        "gpu6_gpu7_components_exact": "components",
    }
    for key, description in required.items():
        if payload.get(key) is not True:
            raise ValueError(f"Stage N live {description} contract failed")
    return payload
