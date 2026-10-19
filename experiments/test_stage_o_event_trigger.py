"""Unit tests for the Stage-O causal event detector.

These tests run without a model runtime.  They pin the progress-scalar
contract (unsupported tasks fall back to ordinal-only detection instead of
raising) and the stall semantics (a regression must not be mislabelled as a
stall).
"""

from __future__ import annotations

import unittest

from experiments.stage_o_event_trigger import (
    EventConfig,
    EventDetector,
    progress_scalar,
)


class ProgressScalarTest(unittest.TestCase):
    def test_unsupported_task_returns_none_instead_of_raising(self) -> None:
        # The recorder is wired for arbitrary tasks; a missing continuous
        # statistic must fall back to ordinal-only detection, not crash.
        self.assertIsNone(progress_scalar("place_fan", {"ordinal_stage": 1}))

    def test_known_tasks_still_return_scalars(self) -> None:
        self.assertAlmostEqual(
            progress_scalar(
                "turn_switch",
                {"ordinal_stage": 1, "normalized_progress": 0.5},
            ),
            0.5,
        )
        self.assertIsNone(
            progress_scalar("click_bell", {"ordinal_stage": 1})
        )

    def test_missing_statistic_for_known_task_still_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "normalized_progress"):
            progress_scalar("turn_switch", {"ordinal_stage": 1})


class StallDetectionTest(unittest.TestCase):
    def _config(self) -> EventConfig:
        return EventConfig(
            min_frame=0,
            window_samples=3,
            continuous_epsilon=0.01,
            cooldown_frames=0,
            max_events=5,
        )

    def _run(self, samples: list[tuple[int, float]]) -> list[str | None]:
        detector = EventDetector(self._config())
        events = []
        for frame, (ordinal, scalar) in enumerate(samples):
            event = detector.observe(
                frame=frame,
                progress={
                    "ordinal_stage": ordinal,
                    "normalized_progress": scalar,
                },
                planner_non_success=0,
            )
            events.append(event["trigger_type"] if event else None)
        return events

    def test_flat_progress_triggers_stall(self) -> None:
        self.assertEqual(
            self._run([(1, 0.5), (1, 0.5), (1, 0.5)]),
            [None, None, "stalled_progress"],
        )

    def test_positive_growth_does_not_trigger_stall(self) -> None:
        self.assertEqual(
            self._run([(1, 0.1), (1, 0.5), (1, 0.9)]),
            [None, None, None],
        )

    def test_regression_does_not_trigger_stall(self) -> None:
        # A strongly negative delta is a regression: the episode is still
        # moving (in the wrong direction) and must not fire a Teacher
        # intervention for a stall.
        self.assertEqual(
            self._run([(1, 0.9), (1, 0.5), (1, 0.1)]),
            [None, None, None],
        )

    def test_ordinal_transition_is_still_detected(self) -> None:
        self.assertEqual(
            self._run([(1, 0.5), (2, 0.6), (2, 0.6)]),
            [None, "phase_transition", None],
        )

    def test_all_none_scalars_fall_back_to_planner_stall(self) -> None:
        detector = EventDetector(self._config())
        events = []
        for frame in range(3):
            event = detector.observe(
                frame=frame,
                progress={"ordinal_stage": 1},
                planner_non_success=frame + 1,  # planner failures keep rising
            )
            events.append(event["trigger_type"] if event else None)
        self.assertEqual(events[2], "stalled_progress")


if __name__ == "__main__":
    unittest.main()
