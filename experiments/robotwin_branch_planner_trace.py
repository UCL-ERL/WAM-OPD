"""Fail-closed same-process record/replay for RoboTwin EE planner paths."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import numpy as np


class BranchPlannerTrace:
    """Record one planner trajectory family and replay it for paired branches."""

    def __init__(self, robot: object) -> None:
        self.robot = robot
        self._original = {
            "left": robot.left_plan_path,
            "right": robot.right_plan_path,
        }
        self._trace_sets: dict[
            str, dict[str, list[dict[str, Any]]]
        ] = {}
        self._records: dict[str, list[dict[str, Any]]] = {
            "left": [],
            "right": [],
        }
        self._indices = {"left": 0, "right": 0}
        self._mode: str | None = None
        self._restored = False
        robot.left_plan_path = self._wrapper("left")
        robot.right_plan_path = self._wrapper("right")

    def _wrapper(self, arm: str) -> Callable[..., dict[str, object]]:
        def traced(target_pose, *args, **kwargs):
            if self._mode not in ("record", "replay"):
                raise AssertionError("planner trace branch mode was not initialized")
            entity = (
                self.robot.left_entity
                if arm == "left"
                else self.robot.right_entity
            )
            start_qpos = np.asarray(entity.get_qpos()).copy()
            target = np.asarray(target_pose).copy()
            index = self._indices[arm]
            self._indices[arm] += 1
            if self._mode == "record":
                result = self._original[arm](target_pose, *args, **kwargs)
                self._records[arm].append(
                    {
                        "start_qpos": start_qpos,
                        "target_pose": target,
                        "result": deepcopy(result),
                    }
                )
                return result

            if index >= len(self._records[arm]):
                raise AssertionError(f"{arm} planner replay exceeded trace")
            saved = self._records[arm][index]
            if not np.array_equal(start_qpos, saved["start_qpos"]):
                delta = np.abs(start_qpos - saved["start_qpos"])
                raise AssertionError(
                    f"{arm} planner call {index} start qpos differs; "
                    f"max_abs={float(delta.max()) if delta.size else 0.0}"
                )
            if not np.array_equal(target, saved["target_pose"]):
                delta = np.abs(target - saved["target_pose"])
                raise AssertionError(
                    f"{arm} planner call {index} target differs; "
                    f"max_abs={float(delta.max()) if delta.size else 0.0}"
                )
            return deepcopy(saved["result"])

        return traced

    def begin_record(self, name: str = "default") -> None:
        self._records = {"left": [], "right": []}
        self._trace_sets[name] = self._records
        self._indices = {"left": 0, "right": 0}
        self._mode = "record"

    def begin_replay(self, name: str = "default") -> None:
        if name not in self._trace_sets:
            raise AssertionError(f"unknown planner trace: {name}")
        self._records = self._trace_sets[name]
        if not any(self._records.values()):
            raise AssertionError("cannot replay an empty planner trace")
        self._indices = {"left": 0, "right": 0}
        self._mode = "replay"

    def finish_branch(self) -> None:
        if self._mode is None:
            raise AssertionError("planner trace branch mode was not initialized")
        if self._mode == "replay":
            unconsumed = {
                arm: len(self._records[arm]) - self._indices[arm]
                for arm in ("left", "right")
                if len(self._records[arm]) != self._indices[arm]
            }
            if unconsumed:
                raise AssertionError(f"unconsumed planner trace calls: {unconsumed}")
        self._mode = None

    def restore(self) -> None:
        """Restore the robot planners when no paired branch will be run."""
        if self._mode is not None:
            raise AssertionError("cannot restore planner trace during a branch")
        if self._restored:
            return
        self.robot.left_plan_path = self._original["left"]
        self.robot.right_plan_path = self._original["right"]
        self._restored = True

    @property
    def call_counts(self) -> dict[str, int]:
        return {arm: len(records) for arm, records in self._records.items()}
