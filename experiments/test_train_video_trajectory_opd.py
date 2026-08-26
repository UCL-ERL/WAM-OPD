"""Focused CPU tests for trajectory/window and physical-tail masking."""

from __future__ import annotations

import unittest

import torch

from experiments.train_video_trajectory_opd import (
    _trajectory_windows,
    action_execution_mask,
    video_execution_mask,
)


def _label(
    *,
    frame_st_id: int,
    executed: list[list[bool]],
    terminal: bool = False,
    terminal_position: list[int] | None = None,
    horizon_reached: bool = False,
) -> dict:
    return {
        "macro_id": frame_st_id // 2,
        "frame_st_id": frame_st_id,
        "student_model_action": torch.zeros((1, 30, 2, 16, 1)),
        "start_frame": 1 if frame_st_id == 0 else 0,
        "action_steps": sum(sum(row) for row in executed),
        "executed_action_mask": executed,
        "terminal_reached": terminal,
        "terminal_action_position": terminal_position,
        "horizon_reached": horizon_reached,
    }


class VideoTrajectoryMaskTest(unittest.TestCase):
    def test_initial_macro_excludes_observation_condition(self) -> None:
        label = _label(
            frame_st_id=0,
            executed=[[False] * 16, [True] * 16],
        )
        plan = torch.zeros((1, 48, 2, 4, 4))

        mask = video_execution_mask(plan, label)

        self.assertFalse(bool(mask[:, :, 0].any()))
        self.assertTrue(bool(mask[:, :, 1].all()))

    def test_global_horizon_masks_unexecuted_video_frame_and_action_tail(self) -> None:
        label = _label(
            frame_st_id=94,
            executed=[[True] * 12 + [False] * 4, [False] * 16],
            horizon_reached=True,
        )
        plan = torch.zeros((1, 48, 2, 4, 4))
        valid = torch.ones((1, 30, 2, 16, 1), dtype=torch.bool)

        video_mask = video_execution_mask(plan, label)
        action_mask = action_execution_mask(valid, label)

        self.assertTrue(bool(video_mask[:, :, 0].all()))
        self.assertFalse(bool(video_mask[:, :, 1].any()))
        self.assertEqual(int(action_mask.sum()), 30 * 12)
        self.assertFalse(bool(action_mask[:, :, 0, 12:].any()))
        self.assertFalse(bool(action_mask[:, :, 1].any()))


class VideoTrajectoryWindowTest(unittest.TestCase):
    def test_windows_are_contiguous_and_cover_each_macro_once(self) -> None:
        labels = [{"macro_id": index} for index in range(10)]

        windows = _trajectory_windows(labels, window_size=4, epoch=1, seed=17)

        flattened = [row["macro_id"] for window in windows for row in window]
        self.assertEqual(sorted(flattened), list(range(10)))
        for window in windows:
            ids = [row["macro_id"] for row in window]
            self.assertEqual(ids, list(range(ids[0], ids[0] + len(ids))))


if __name__ == "__main__":
    unittest.main()
