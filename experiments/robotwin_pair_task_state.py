"""Reconstruct task metadata omitted by expert-free paired evaluation.

Strict paired replay deliberately skips ``play_once`` so that it can evaluate
the exact Student seed manifest without rerunning seed selection.  A small
number of RoboTwin tasks initialize success-predicate metadata in ``play_once``
rather than ``setup_demo``.  This module reconstructs only that metadata from
the freshly reset simulator state; it does not move the simulator or alter
model inputs.
"""

from __future__ import annotations

from typing import Any


def initialize_pair_task_state(task_name: str, task_env: Any) -> dict[str, Any]:
    """Initialize task-local success metadata needed by paired replay."""

    if task_name != "put_object_cabinet":
        return {"initialized": False, "task": task_name}

    object_position = task_env.object.get_pose().p
    arm_tag = "right" if float(object_position[0]) > 0.0 else "left"
    origin_z = float(object_position[2])
    task_env.arm_tag = arm_tag
    task_env.origin_z = origin_z
    return {
        "initialized": True,
        "task": task_name,
        "arm_tag": arm_tag,
        "origin_z": origin_z,
    }
