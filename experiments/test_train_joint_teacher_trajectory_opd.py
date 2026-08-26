"""Focused CPU tests for paired Teacher trajectory OPD contracts."""

from __future__ import annotations

import unittest

import torch

from experiments.train_joint_teacher_trajectory_opd import (
    build_teacher_trajectory,
    teacher_action_execution_mask,
    teacher_video_execution_mask,
    trajectory_windows,
    validate_gate_contract,
)


def _gate() -> dict:
    return {
        "status": "PASS",
        "training_started": False,
        "shared_noise_across_arms": True,
        "arms": ["SS", "ST", "TS", "TT"],
        "task": "open_microwave",
        "task_config": "demo_clean",
        "seed": 10008,
        "chunks_requested": 48,
        "max_control_steps": 1500,
        "episodes": [
            {"arm": "SS", "success": False, "chunks_completed": 48},
            {"arm": "ST", "success": False, "chunks_completed": 48},
            {"arm": "TS", "success": False, "chunks_completed": 48},
            {"arm": "TT", "success": True, "chunks_completed": 2},
        ],
    }


def _target(macro_id: int) -> dict:
    return {
        "macro_id": macro_id,
        "frame_st_id": macro_id * 2,
        "prompt_token_ids": (1, 2, 3),
        "initial_latent": torch.zeros((1, 48, 1, 2, 2)),
        "epsilon_v": torch.zeros((1, 48, 2, 2, 2)),
        "epsilon_a": torch.zeros((1, 30, 2, 16, 1)),
        "teacher_z_t": torch.zeros((1, 48, 2, 2, 2)),
        "teacher_z_t_timestep": torch.zeros((1, 2)),
        "teacher_action": torch.zeros((1, 30, 2, 16, 1)),
        "teacher_action_input_noise": torch.zeros((1, 30, 2, 16, 1)),
        "teacher_action_timestep": torch.zeros((1, 2)),
        "teacher_action_valid_mask": torch.ones(
            (1, 30, 2, 16, 1), dtype=torch.bool
        ),
        "teacher_action_token_positions": tuple(range(32)),
        "teacher_cache_valid_length": macro_id * 64,
    }


def _physical(
    macro_id: int,
    *,
    executed: list[list[bool]],
    terminal: bool,
    terminal_position: list[int] | None,
) -> dict:
    return {
        "chunk_id": macro_id,
        "frame_st_id": macro_id * 2,
        "start_frame": 1 if macro_id == 0 else 0,
        "action_steps": sum(sum(row) for row in executed),
        "executed_action_mask": executed,
        "terminal_reached": terminal,
        "terminal_action_position": terminal_position,
        "horizon_reached": False,
        "task_success": terminal,
        "eval_success": terminal,
    }


class GateContractTest(unittest.TestCase):
    def test_requires_failed_ss_and_successful_tt(self) -> None:
        ss, tt = validate_gate_contract(
            _gate(),
            task="open_microwave",
            task_config="demo_clean",
            seed=10008,
            chunks=48,
            max_control_steps=1500,
        )
        self.assertFalse(ss["success"])
        self.assertTrue(tt["success"])

    def test_rejects_non_shared_gate(self) -> None:
        gate = _gate()
        gate["shared_noise_across_arms"] = False
        with self.assertRaisesRegex(ValueError, "exact shared noise"):
            validate_gate_contract(
                gate,
                task="open_microwave",
                task_config="demo_clean",
                seed=10008,
                chunks=48,
                max_control_steps=1500,
            )


class TeacherTrajectoryTest(unittest.TestCase):
    def test_full_prefix_keeps_success_trigger_and_has_no_tail(self) -> None:
        first = [[False] * 16, [True] * 16]
        final = [[True] * 5 + [False] * 11, [False] * 16]
        episode = {
            "success": True,
            "chunks_completed": 2,
            "control_steps": 21,
            "history": [
                {"frame_st_id": 0, "latent": torch.zeros(1), "action": torch.zeros(1)},
                {"frame_st_id": 2, "latent": torch.zeros(1), "action": torch.zeros(1)},
            ],
            "chunks": [
                _physical(0, executed=first, terminal=False, terminal_position=None),
                _physical(1, executed=final, terminal=True, terminal_position=[0, 4]),
            ],
        }
        trajectory = build_teacher_trajectory(
            task="open_microwave",
            task_config="demo_clean",
            seed=10008,
            prompt="Pull the the microwave's handle using the left arm.",
            initial_observation={"state": torch.zeros(1)},
            episode=episode,
            targets=[_target(0), _target(1)],
        )

        self.assertEqual(len(trajectory["labels"]), 2)
        self.assertTrue(trajectory["success_trigger_label_included"])
        self.assertEqual(trajectory["success_post_label_count"], 0)
        self.assertEqual(trajectory["labels"][-1]["action_steps"], 5)

        video_mask = teacher_video_execution_mask(
            trajectory["labels"][-1]["teacher_z_t"], trajectory["labels"][-1]
        )
        action_mask = teacher_action_execution_mask(
            trajectory["labels"][-1]["teacher_action_valid_mask"],
            trajectory["labels"][-1],
        )
        self.assertTrue(bool(video_mask[:, :, 0].all()))
        self.assertFalse(bool(video_mask[:, :, 1].any()))
        self.assertEqual(int(action_mask.sum()), 30 * 5)
        self.assertTrue(bool(action_mask[:, :, 0, 4].all()))
        self.assertFalse(bool(action_mask[:, :, 0, 5:].any()))

    def test_windows_reuse_every_macro_once_per_epoch(self) -> None:
        labels = [{"macro_id": index} for index in range(11)]
        windows = trajectory_windows(labels, window_size=4, epoch=3, seed=10008)
        ids = [row["macro_id"] for window in windows for row in window]
        self.assertEqual(sorted(ids), list(range(11)))
        self.assertEqual(sorted(len(window) for window in windows), [3, 4, 4])


if __name__ == "__main__":
    unittest.main()
