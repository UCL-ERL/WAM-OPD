"""Task horizons shared by OPD capture and evaluation entry points."""

from __future__ import annotations

from dataclasses import dataclass
import math


TRAIN_TASK_CONFIG = "demo_clean"


@dataclass(frozen=True)
class TaskSpec:
    max_control_steps: int
    first_macro_steps: int = 16
    later_macro_steps: int = 32

    @property
    def chunks(self) -> int:
        remaining = max(0, int(self.max_control_steps) - int(self.first_macro_steps))
        return 1 + math.ceil(remaining / int(self.later_macro_steps))


TASK_SPECS = {
    "move_stapler_pad": TaskSpec(max_control_steps=400),
    "open_microwave": TaskSpec(max_control_steps=1500),
    "place_fan": TaskSpec(max_control_steps=400),
    "put_object_cabinet": TaskSpec(max_control_steps=700),
    "put_bottles_dustbin": TaskSpec(max_control_steps=1700),
    "handover_mic": TaskSpec(max_control_steps=600),
    "place_shoe": TaskSpec(max_control_steps=500),
    "scan_object": TaskSpec(max_control_steps=500),
    "place_a2b_left": TaskSpec(max_control_steps=400),
    "place_a2b_right": TaskSpec(max_control_steps=400),
    "place_bread_basket": TaskSpec(max_control_steps=700),
    "place_cans_plasticbox": TaskSpec(max_control_steps=800),
    "rotate_qrcode": TaskSpec(max_control_steps=400),
    "stamp_seal": TaskSpec(max_control_steps=400),
    "stack_blocks_three": TaskSpec(max_control_steps=1200),
    "blocks_ranking_rgb": TaskSpec(max_control_steps=1200),
    "blocks_ranking_size": TaskSpec(max_control_steps=1200),
}


def require_training_task_config(task_config: str) -> str:
    """Reject non-Easy domains at OPD collection/training entry points."""

    value = str(task_config)
    if value != TRAIN_TASK_CONFIG:
        raise ValueError(
            "OPD collection/training requires RoboTwin demo_clean (Easy); "
            f"got {value!r}. Use demo_randomized only for frozen-checkpoint "
            "robustness evaluation."
        )
    return value


def resolve_task_chunks(task: str, override: int | None = None) -> int:
    """Return a positive explicit override or the benchmark task default."""

    if override is not None:
        if isinstance(override, bool) or int(override) <= 0:
            raise ValueError(f"--chunks must be a positive integer, got {override!r}")
        return int(override)
    try:
        return TASK_SPECS[str(task)].chunks
    except KeyError as exc:
        raise ValueError(
            f"no default OPD horizon for task {task!r}; pass --chunks explicitly"
        ) from exc


__all__ = [
    "TASK_SPECS",
    "TRAIN_TASK_CONFIG",
    "TaskSpec",
    "require_training_task_config",
    "resolve_task_chunks",
]
