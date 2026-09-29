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


class ScaledCanaryGateTest(unittest.TestCase):
    """The formal collection stage must refuse a FAILED concurrency canary."""

    def _manifest(self, canary_root: Path) -> dict[str, object]:
        return {
            "canary_root": str(canary_root),
            "formal_root": "/ssd/data/formal",
            "collection_configs": ["/ssd/data/cfg.json"],
            "gpu_policy": {
                "stable_gpu_count": 2,
                "allowed_gpu_ids": [4, 7],
                "min_free_mib": {
                    "collection": 70000,
                    "update": 70000,
                    "evaluation": 45000,
                },
            },
        }

    def _write_decision(self, root: Path, *, status: str, workers: int) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        decision = root / "decision.json"
        decision.write_text(
            json.dumps(
                {
                    "schema": "waopd_collection_concurrency_canary_v1",
                    "status": status,
                    "selected_workers_per_gpu": workers,
                    "failures": [],
                }
            )
        )
        return decision

    def test_failed_canary_blocks_formal_collection(self) -> None:
        from experiments.run_scaled_qualified_success_path_pipeline import _run_stage

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_decision(root, status="FAIL", workers=1)
            with self.assertRaisesRegex(RuntimeError, "canary FAILED"):
                with mock.patch(
                    "experiments.run_scaled_qualified_success_path_pipeline._launch_configs",
                    return_value=[],
                ):
                    _run_stage(self._manifest(root), "collection")

    def test_fallback_canary_still_allows_formal_collection(self) -> None:
        from experiments.run_scaled_qualified_success_path_pipeline import _run_stage

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_decision(root, status="FALLBACK", workers=1)
            with mock.patch(
                "experiments.run_scaled_qualified_success_path_pipeline._launch_configs",
                return_value=[],
            ):
                result = _run_stage(self._manifest(root), "collection")
            self.assertEqual(result["workers_per_gpu"], 1)

    def test_pass_canary_uses_selected_worker_count(self) -> None:
        from experiments.run_scaled_qualified_success_path_pipeline import _run_stage

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_decision(root, status="PASS", workers=2)
            with mock.patch(
                "experiments.run_scaled_qualified_success_path_pipeline._launch_configs",
                return_value=[],
            ):
                result = _run_stage(self._manifest(root), "collection")
            self.assertEqual(result["workers_per_gpu"], 2)

    def test_missing_canary_decision_is_rejected(self) -> None:
        from experiments.run_scaled_qualified_success_path_pipeline import _run_stage

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                _run_stage(self._manifest(Path(tmp)), "collection")

    def test_unknown_canary_status_is_rejected(self) -> None:
        from experiments.run_scaled_qualified_success_path_pipeline import _run_stage

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_decision(root, status="BOGUS", workers=1)
            with self.assertRaisesRegex(RuntimeError, "unexpected status"):
                with mock.patch(
                    "experiments.run_scaled_qualified_success_path_pipeline._launch_configs",
                    return_value=[],
                ):
                    _run_stage(self._manifest(root), "collection")


if __name__ == "__main__":
    unittest.main()


class ScaledBindingValidationTest(unittest.TestCase):
    """The scaled validator must bind configs to the shipped split hashes."""

    def test_training_config_missing_split_binding_is_rejected(self) -> None:
        from experiments import run_qualified_success_path_pipeline as base

        with tempfile.TemporaryDirectory() as tmp:
            split_path = Path(tmp).resolve() / "split.json"
            split_path.write_text(
                json.dumps(
                    {
                        "task": "place_a2b_left",
                        "binding_status": "BOUND",
                        "split_counts": {
                            "train": 24,
                            "calibration": 12,
                            "screening": 4,
                            "heldout": 6,
                        },
                        "split_indices": {
                            "train": [0, 24],
                            "calibration": [24, 36],
                            "screening": [36, 40],
                            "heldout": [40, 46],
                        },
                        "accepted_seed_list": list(range(46)),
                        "ordered_metadata_projection_sha256": "abc",
                        "episode_records": [
                            {
                                "episode": index,
                                "seed": index,
                                "instruction": f"inst {index}",
                                "role": "train",
                            }
                            for index in range(46)
                        ],
                    }
                )
            )
            training = {
                # No source_binding_status / split_manifest_sha256 at all.
                "rollouts": [],
            }
            with self.assertRaises(ValueError):
                base._validate_binding(
                    training,
                    split_path=split_path,
                    split_sha=base._sha256(split_path),
                    metadata_sha="abc",
                    label="training",
                )

    def test_training_config_wrong_split_sha_is_rejected(self) -> None:
        from experiments import run_qualified_success_path_pipeline as base

        with tempfile.TemporaryDirectory() as tmp:
            split_path = Path(tmp).resolve() / "split.json"
            split_path.write_text(json.dumps({"task": "x"}), encoding="utf-8")
            training = {
                "source_binding_status": "BOUND",
                "split_manifest": str(split_path),
                "split_manifest_sha256": "deadbeef",
                "split_indices": {"train": [0, 1]},
                "ordered_metadata_projection_sha256": "abc",
            }
            with self.assertRaisesRegex(ValueError, "split_manifest_sha256"):
                base._validate_binding(
                    training,
                    split_path=split_path,
                    split_sha=base._sha256(split_path),
                    metadata_sha="abc",
                    label="training",
                )


if __name__ == "__main__":
    unittest.main()
