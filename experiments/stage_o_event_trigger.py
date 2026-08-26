"""Causal, bounded event triggers for Stage-O Teacher interventions.

The detector consumes only state available at the current action boundary.  It
does not inspect terminal success or any future observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class EventConfig:
    """Limits and thresholds for one episode's event detector."""

    min_frame: int = 10
    window_samples: int = 3
    continuous_epsilon: float = 0.01
    cooldown_frames: int = 12
    max_events: int = 2

    def __post_init__(self) -> None:
        if self.min_frame < 0:
            raise ValueError("min_frame must be non-negative")
        if self.window_samples < 2:
            raise ValueError("window_samples must be at least two")
        if self.continuous_epsilon < 0 or not math.isfinite(
            self.continuous_epsilon
        ):
            raise ValueError("continuous_epsilon must be finite and non-negative")
        if self.cooldown_frames < 0:
            raise ValueError("cooldown_frames must be non-negative")
        if self.max_events < 1:
            raise ValueError("max_events must be positive")


def _ordinal(progress: Mapping[str, object]) -> int:
    if "ordinal_stage" not in progress:
        raise ValueError("progress must contain ordinal_stage")
    value = progress["ordinal_stage"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("ordinal_stage must be a non-negative integer")
    return value


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def progress_scalar(task_name: str, progress: Mapping[str, object]) -> float | None:
    """Return a phase-local scalar where larger always means more progress.

    Tasks without a validated continuous statistic deliberately return None;
    their stall trigger therefore depends on unchanged ordinal stage plus fresh
    planner failures.
    """

    _ordinal(progress)
    if task_name == "turn_switch":
        if "normalized_progress" not in progress:
            raise ValueError("turn_switch progress lacks normalized_progress")
        return _finite_float(progress["normalized_progress"], "normalized_progress")
    if task_name == "put_object_cabinet":
        if "xy_linf_error" not in progress:
            raise ValueError("put_object_cabinet progress lacks xy_linf_error")
        return -_finite_float(progress["xy_linf_error"], "xy_linf_error")
    if task_name in {"click_bell", "blocks_ranking_size"}:
        return None
    raise ValueError(f"unsupported task for Stage-O progress: {task_name}")


def observable_change_metrics(
    previous: Mapping[str, object],
    current: Mapping[str, object],
) -> dict[str, Any]:
    """Measure causal change using only deployable cameras and robot state."""

    image_prefix = "observation.images."
    previous_cameras = sorted(key for key in previous if key.startswith(image_prefix))
    current_cameras = sorted(key for key in current if key.startswith(image_prefix))
    if not previous_cameras or previous_cameras != current_cameras:
        raise ValueError("observable camera keys must be non-empty and identical")

    camera_changes: dict[str, float] = {}
    camera_maxima: list[float] = []
    for key in previous_cameras:
        before = np.asarray(previous[key])
        after = np.asarray(current[key])
        if before.shape != after.shape:
            raise ValueError(f"observable camera shape differs for {key}")
        if before.size == 0:
            raise ValueError(f"observable camera is empty for {key}")
        before_float = before.astype(np.float64) / 255.0
        after_float = after.astype(np.float64) / 255.0
        delta = np.abs(after_float - before_float)
        if not np.isfinite(delta).all():
            raise ValueError(f"observable camera change is non-finite for {key}")
        short_name = key.removeprefix(image_prefix)
        camera_changes[short_name] = float(delta.mean())
        camera_maxima.append(float(delta.max()))

    state_key = "observation.state"
    if state_key not in previous or state_key not in current:
        raise ValueError("observable trace requires observation.state")
    before_state = np.asarray(previous[state_key], dtype=np.float64)
    after_state = np.asarray(current[state_key], dtype=np.float64)
    if before_state.shape != after_state.shape or before_state.size == 0:
        raise ValueError("observable state shape must be non-empty and identical")
    state_delta = after_state - before_state
    if not np.isfinite(state_delta).all():
        raise ValueError("observable state change is non-finite")

    return {
        "schema": "flashwam_observable_change_v1",
        "camera_mean_abs_change": camera_changes,
        "image_mean_abs_change": float(np.mean(list(camera_changes.values()))),
        "image_max_abs_change": float(max(camera_maxima)),
        "state_rmse": float(np.sqrt(np.mean(state_delta**2))),
        "state_max_abs": float(np.max(np.abs(state_delta))),
    }


def _infer_progress_scalar(progress: Mapping[str, object]) -> float | None:
    if "normalized_progress" in progress:
        return _finite_float(progress["normalized_progress"], "normalized_progress")
    if "xy_linf_error" in progress:
        return -_finite_float(progress["xy_linf_error"], "xy_linf_error")
    return None


@dataclass(frozen=True)
class _Sample:
    frame: int
    ordinal: int
    scalar: float | None
    planner_non_success: int


class EventDetector:
    """Detect phase transitions or stalled progress from past/current samples."""

    def __init__(self, config: EventConfig | None = None) -> None:
        self.config = config or EventConfig()
        self._history: list[_Sample] = []
        self._last_event_frame: int | None = None
        self._events_emitted = 0

    def observe(
        self,
        *,
        frame: int,
        progress: Mapping[str, object],
        planner_non_success: int,
    ) -> dict[str, object] | None:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("frame must be a non-negative integer")
        if self._history and frame <= self._history[-1].frame:
            raise ValueError("frame must increase strictly")
        if (
            isinstance(planner_non_success, bool)
            or not isinstance(planner_non_success, int)
            or planner_non_success < 0
        ):
            raise ValueError("planner_non_success must be a non-negative integer")
        if (
            self._history
            and planner_non_success < self._history[-1].planner_non_success
        ):
            raise ValueError("planner_non_success must be cumulative")

        sample = _Sample(
            frame=frame,
            ordinal=_ordinal(progress),
            scalar=_infer_progress_scalar(progress),
            planner_non_success=planner_non_success,
        )
        previous = self._history[-1] if self._history else None
        self._history.append(sample)

        if self._events_emitted >= self.config.max_events:
            return None
        if frame < self.config.min_frame or not self._cooldown_elapsed(frame):
            return None

        if previous is not None and sample.ordinal > previous.ordinal:
            return self._emit(
                frame,
                "phase_transition",
                ordinal_before=previous.ordinal,
                ordinal_after=sample.ordinal,
            )

        width = self.config.window_samples
        if len(self._history) < width:
            return None
        window = self._history[-width:]
        if any(item.ordinal != window[0].ordinal for item in window):
            return None
        planner_delta = (
            window[-1].planner_non_success - window[0].planner_non_success
        )

        scalars = [item.scalar for item in window]
        if all(value is None for value in scalars):
            scalar_delta = None
            if planner_delta <= 0:
                return None
        elif any(value is None for value in scalars):
            return None
        else:
            scalar_delta = float(scalars[-1]) - float(scalars[0])
            if scalar_delta > self.config.continuous_epsilon:
                return None

        return self._emit(
            frame,
            "stalled_progress",
            ordinal_before=window[0].ordinal,
            ordinal_after=window[-1].ordinal,
            planner_non_success_delta=planner_delta,
            continuous_progress_delta=scalar_delta,
        )

    def _cooldown_elapsed(self, frame: int) -> bool:
        return (
            self._last_event_frame is None
            or frame - self._last_event_frame >= self.config.cooldown_frames
        )

    def _emit(
        self,
        frame: int,
        trigger_type: str,
        **details: object,
    ) -> dict[str, object]:
        event: dict[str, object] = {
            "schema": "flashwam_stage_o_event_v1",
            "event_index": self._events_emitted,
            "trigger_type": trigger_type,
            "frame": frame,
            **details,
        }
        self._events_emitted += 1
        self._last_event_frame = frame
        return event


@dataclass(frozen=True)
class ObservableEventConfig:
    """Task-agnostic thresholds for a real-robot-compatible event gate."""

    min_frame: int = 10
    window_samples: int = 3
    image_mean_threshold: float = 0.05
    state_rmse_threshold: float = 0.05
    cooldown_frames: int = 12
    max_events: int = 2

    def __post_init__(self) -> None:
        if self.min_frame < 0:
            raise ValueError("min_frame must be non-negative")
        if self.window_samples < 2:
            raise ValueError("window_samples must be at least two")
        for name, value in (
            ("image_mean_threshold", self.image_mean_threshold),
            ("state_rmse_threshold", self.state_rmse_threshold),
        ):
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.cooldown_frames < 0:
            raise ValueError("cooldown_frames must be non-negative")
        if self.max_events < 1:
            raise ValueError("max_events must be positive")


@dataclass(frozen=True)
class _ObservableSample:
    frame: int
    image_mean_abs_change: float
    state_rmse: float
    planner_non_success: int


class ObservableEventDetector:
    """Trigger only from cameras, robot state, and motion-planner outcomes."""

    def __init__(self, config: ObservableEventConfig | None = None) -> None:
        self.config = config or ObservableEventConfig()
        self._history: list[_ObservableSample] = []
        self._last_event_frame: int | None = None
        self._last_event_planner_non_success = 0
        self._events_emitted = 0

    def observe(
        self,
        *,
        frame: int,
        image_mean_abs_change: float,
        state_rmse: float,
        planner_non_success: int,
        terminal_success: bool = False,
    ) -> dict[str, object] | None:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("frame must be a non-negative integer")
        if self._history and frame <= self._history[-1].frame:
            raise ValueError("frame must increase strictly")
        image_change = _finite_float(
            image_mean_abs_change, "image_mean_abs_change"
        )
        state_change = _finite_float(state_rmse, "state_rmse")
        if image_change < 0 or state_change < 0:
            raise ValueError("observable changes must be non-negative")
        if (
            isinstance(planner_non_success, bool)
            or not isinstance(planner_non_success, int)
            or planner_non_success < 0
        ):
            raise ValueError("planner_non_success must be non-negative")
        if (
            self._history
            and planner_non_success < self._history[-1].planner_non_success
        ):
            raise ValueError("planner_non_success must be cumulative")
        sample = _ObservableSample(
            frame=frame,
            image_mean_abs_change=image_change,
            state_rmse=state_change,
            planner_non_success=planner_non_success,
        )
        self._history.append(sample)
        if terminal_success or self._events_emitted >= self.config.max_events:
            return None
        if frame < self.config.min_frame or not self._cooldown_elapsed(frame):
            return None

        width = self.config.window_samples
        if len(self._history) < width:
            return None
        window = self._history[-width:]
        planner_delta = (
            window[-1].planner_non_success - window[0].planner_non_success
        )
        planner_since_event = (
            sample.planner_non_success - self._last_event_planner_non_success
        )
        if planner_since_event > 0:
            self._last_event_planner_non_success = sample.planner_non_success
            return self._emit(
                frame,
                "planner_non_success",
                planner_non_success_delta=planner_delta,
                planner_non_success_since_event=planner_since_event,
            )

        if all(
            item.image_mean_abs_change <= self.config.image_mean_threshold
            and item.state_rmse <= self.config.state_rmse_threshold
            for item in window
        ):
            self._last_event_planner_non_success = sample.planner_non_success
            return self._emit(
                frame,
                "observable_stall",
                max_image_mean_abs_change=max(
                    item.image_mean_abs_change for item in window
                ),
                max_state_rmse=max(item.state_rmse for item in window),
            )
        return None

    def _cooldown_elapsed(self, frame: int) -> bool:
        return (
            self._last_event_frame is None
            or frame - self._last_event_frame >= self.config.cooldown_frames
        )

    def _emit(
        self,
        frame: int,
        trigger_type: str,
        **details: object,
    ) -> dict[str, object]:
        event: dict[str, object] = {
            "schema": "flashwam_stage_o_observable_event_v1",
            "event_index": self._events_emitted,
            "trigger_type": trigger_type,
            "frame": frame,
            **details,
        }
        self._events_emitted += 1
        self._last_event_frame = frame
        return event


class StageOTraceRecorder:
    """Collect a compact per-chunk trace and its causal trigger decisions."""

    def __init__(
        self,
        *,
        task_name: str,
        seed: int,
        config: EventConfig | None = None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.task_name = task_name
        self.seed = seed
        self.config = config or EventConfig()
        self.detector = EventDetector(self.config)
        self.observable_config = ObservableEventConfig(
            min_frame=self.config.min_frame,
            window_samples=self.config.window_samples,
            cooldown_frames=self.config.cooldown_frames,
            max_events=self.config.max_events,
        )
        self.observable_detector = ObservableEventDetector(self.observable_config)
        self.samples: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.observable_events: list[dict[str, Any]] = []
        self._finalized = False

    def observe(
        self,
        *,
        frame: int,
        progress: Mapping[str, object],
        planner_events: Mapping[str, object],
        terminal_success: bool = False,
        observable_change: Mapping[str, object] | None = None,
    ) -> dict[str, object] | None:
        if self._finalized:
            raise AssertionError("Stage-O trace was already finalized")
        if planner_events.get("schema") != "flashwam_stage_n_planner_events_v1":
            raise ValueError("Stage-O requires the planner event schema")
        raw_non_success = planner_events.get("non_success_calls")
        if (
            isinstance(raw_non_success, bool)
            or not isinstance(raw_non_success, int)
            or raw_non_success < 0
        ):
            raise ValueError("planner non_success_calls must be non-negative")

        scalar = progress_scalar(self.task_name, progress)
        progress_copy = dict(progress)
        observable_copy = None
        if observable_change is not None:
            if observable_change.get("schema") != "flashwam_observable_change_v1":
                raise ValueError("Stage-O observable change schema differs")
            for field in (
                "image_mean_abs_change",
                "image_max_abs_change",
                "state_rmse",
                "state_max_abs",
            ):
                _finite_float(observable_change.get(field), field)
            observable_copy = dict(observable_change)
        sample = {
            "frame": int(frame),
            "progress": progress_copy,
            "continuous_progress_scalar": scalar,
            "planner_non_success_calls": raw_non_success,
            "terminal_success": bool(terminal_success),
            "observable_change": observable_copy,
        }
        self.samples.append(sample)
        if observable_copy is not None:
            observable_event = self.observable_detector.observe(
                frame=frame,
                image_mean_abs_change=float(
                    observable_copy["image_mean_abs_change"]
                ),
                state_rmse=float(observable_copy["state_rmse"]),
                planner_non_success=raw_non_success,
                terminal_success=terminal_success,
            )
            if observable_event is not None:
                self.observable_events.append(
                    {
                        **observable_event,
                        "task": self.task_name,
                        "seed": self.seed,
                    }
                )
        if terminal_success:
            return None
        event = self.detector.observe(
            frame=frame,
            progress=progress_copy,
            planner_non_success=raw_non_success,
        )
        if event is not None:
            event = {**event, "task": self.task_name, "seed": self.seed}
            self.events.append(event)
        return event

    def finalize(
        self,
        *,
        success: bool,
        completion_step: int,
        final_progress: Mapping[str, object],
    ) -> dict[str, Any]:
        if self._finalized:
            raise AssertionError("Stage-O trace was already finalized")
        if not self.samples:
            raise ValueError("Stage-O trace has no causal samples")
        progress_scalar(self.task_name, final_progress)
        self._finalized = True
        return {
            "schema": "flashwam_stage_o_event_trace_v1",
            "task": self.task_name,
            "seed": self.seed,
            "config": {
                "min_frame": self.config.min_frame,
                "window_samples": self.config.window_samples,
                "continuous_epsilon": self.config.continuous_epsilon,
                "cooldown_frames": self.config.cooldown_frames,
                "max_events": self.config.max_events,
            },
            "causal_past_only": True,
            "samples": list(self.samples),
            "events": list(self.events),
            "observable_event_config": {
                "min_frame": self.observable_config.min_frame,
                "window_samples": self.observable_config.window_samples,
                "image_mean_threshold": (
                    self.observable_config.image_mean_threshold
                ),
                "state_rmse_threshold": self.observable_config.state_rmse_threshold,
                "cooldown_frames": self.observable_config.cooldown_frames,
                "max_events": self.observable_config.max_events,
            },
            "observable_events": list(self.observable_events),
            "outcome": {
                "success": bool(success),
                "completion_step": int(completion_step),
                "progress": dict(final_progress),
            },
        }
