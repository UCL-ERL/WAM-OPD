from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.bind_scaled_qualified_success_path_task import (
    _indices,
    _load_metadata,
)
from experiments.run_scaled_qualified_success_path_pipeline import (
    _eligible_gpus,
    _expected_roles,
    _split_bounds,
)


class ScaledQualifiedSuccessPathTest(unittest.TestCase):
    def test_24_12_4_6_split_is_contiguous_and_role_exact(self) -> None:
        counts = {
            "train": 24,
            "calibration": 12,
            "screening": 4,
            "heldout": 6,
        }
        expected = {
            "train": [0, 24],
            "calibration": [24, 36],
            "screening": [36, 40],
            "heldout": [40, 46],
        }
        self.assertEqual(_indices(**counts), expected)
        self.assertEqual(_split_bounds(counts), expected)
        roles = _expected_roles(counts)
        self.assertEqual(len(roles), 46)
        self.assertEqual(roles[:24], ["train"] * 24)
        self.assertEqual(roles[24:36], ["calibration"] * 12)
        self.assertEqual(roles[36:40], ["screening"] * 4)
        self.assertEqual(roles[40:], ["heldout"] * 6)

    def test_metadata_binding_projects_only_seed_instruction_and_role(self) -> None:
        roles = ["train", "calibration", "screening", "heldout"]
        payload = {
            "status": "PASS",
            "task": "place_a2b_left",
            "task_config": "demo_clean",
            "instruction_type": "seen",
            "accepted_seed_list": [101, 102, 103, 104],
            "episode_records": [
                {
                    "seed": seed,
                    "instruction": f"instruction {seed}",
                    "expert_admission_provenance": "not projected",
                }
                for seed in (101, 102, 103, 104)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "metadata.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            records = _load_metadata(
                source,
                task="place_a2b_left",
                roles=roles,
            )
        self.assertEqual(
            [set(record) for record in records],
            [{"episode", "seed", "instruction", "role"}] * 4,
        )
        self.assertEqual([record["role"] for record in records], roles)

    def test_metadata_binding_rejects_duplicate_or_reordered_seeds(self) -> None:
        payload = {
            "status": "PASS",
            "task": "place_a2b_left",
            "task_config": "demo_clean",
            "instruction_type": "seen",
            "accepted_seed_list": [101, 101],
            "episode_records": [
                {"seed": 101, "instruction": "one"},
                {"seed": 101, "instruction": "two"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "metadata.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique and ordered"):
                _load_metadata(
                    source,
                    task="place_a2b_left",
                    roles=["train", "calibration"],
                )

    def test_two_gpu_policy_selects_only_the_bound_pair(self) -> None:
        manifest = {
            "gpu_policy": {
                "stable_gpu_count": 2,
                "allowed_gpu_ids": [4, 7],
                "min_free_mib": {
                    "collection": 70000,
                    "update": 70000,
                    "evaluation": 45000,
                },
            }
        }
        rows = [
            {"index": 4, "free_mib": 80000},
            {"index": 7, "free_mib": 80000},
            {"index": 0, "free_mib": 80000},
        ]
        with mock.patch(
            "experiments.run_scaled_qualified_success_path_pipeline.base._gpu_rows",
            return_value=rows,
        ):
            self.assertEqual(_eligible_gpus(manifest, "collection"), [4, 7])
            self.assertEqual(_eligible_gpus(manifest, "screening"), [4, 7])
            self.assertEqual(_eligible_gpus(manifest, "update"), [4])


if __name__ == "__main__":
    unittest.main()
