"""Offline contract tests for the task-agnostic qualified OPD pipeline."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).with_name("run_qualified_success_path_pipeline.py")
SPEC = importlib.util.spec_from_file_location("qualified_pipeline", MODULE_PATH)
PIPELINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PIPELINE)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _json_sha(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class QualifiedPipelineContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.workspace = self.root / "workspace"
        self.project = self.root / "project"
        self.workspace.mkdir()
        self.project.mkdir()
        self.formal = self.root / "formal"
        self.collection = self.root / "collection"
        self.task = "unit_task"

        records = [
            {
                "episode": index,
                "seed": 10_000 + index,
                "instruction": f"instruction {index}",
                "role": PIPELINE.ROLES[index],
            }
            for index in range(20)
        ]
        projection = [
            {key: row[key] for key in ("episode", "seed", "instruction")}
            for row in records
        ]
        self.split_path = self.workspace / "configs" / "split.json"
        split = {
            "task": self.task,
            "task_config": "demo_clean",
            "binding_status": "BOUND",
            "split_indices": PIPELINE.SPLITS,
            "accepted_seed_list": [row["seed"] for row in records],
            "ordered_metadata_projection_sha256": _json_sha(projection),
            "episode_records": records,
        }
        _write(self.split_path, split)
        split_sha = hashlib.sha256(self.split_path.read_bytes()).hexdigest()

        base = {
            **PIPELINE.RECIPE,
            "project_root": str(self.project),
            "task": self.task,
            "task_config": "demo_clean",
            "chunks": 38,
            "source_binding_status": "BOUND",
            "split_manifest": str(self.split_path),
            "split_manifest_sha256": split_sha,
            "split_indices": PIPELINE.SPLITS,
            "ordered_metadata_projection_sha256": split[
                "ordered_metadata_projection_sha256"
            ],
            "student": str(self.root / "models" / "student"),
            "teacher_transformer": str(self.root / "models" / "teacher"),
            "lora_block_indices": list(range(30)),
        }
        self.training_path = self.workspace / "configs" / "training.json"
        artifacts = [
            str(
                self.collection
                / f"collect_shard_{index % 4:02d}"
                / f"{self.task}_round_00_rollout_{index // 4:02d}.pt"
            )
            for index in range(12)
        ]
        training = {
            **base,
            "run_mode": "trajectory_update",
            "inner_epochs": 3,
            "initial_checkpoint": None,
            "resume_optimizer_state": False,
            "rollouts": [PIPELINE._rollout(row) for row in records[:12]],
            "trajectory_artifacts": artifacts,
            "output_dir": str(self.formal / "update"),
        }
        _write(self.training_path, training)

        self.collection_paths: list[Path] = []
        for shard in range(4):
            path = self.workspace / "configs" / f"collect_{shard:02d}.json"
            config = {
                **base,
                "run_mode": "collect",
                "inner_epochs": 1,
                "rollouts": [
                    PIPELINE._rollout(records[index])
                    for index in range(shard, 12, 4)
                ],
                "output_dir": str(self.collection / f"collect_shard_{shard:02d}"),
            }
            _write(path, config)
            self.collection_paths.append(path)

        self.protocol_path = self.workspace / "configs" / "protocol.json"
        screening = [
            {"seed": row["seed"], "instruction": row["instruction"]}
            for row in records[12:14]
        ]
        heldout = [
            {"seed": row["seed"], "instruction": row["instruction"]}
            for row in records[14:20]
        ]
        heldout_banks = [2002, 2003]
        protocol = {
            "source_binding_status": "BOUND",
            "split_manifest": str(self.split_path),
            "split_manifest_sha256": split_sha,
            "split_indices": PIPELINE.SPLITS,
            "ordered_metadata_projection_sha256": split[
                "ordered_metadata_projection_sha256"
            ],
            "task": self.task,
            "task_config": "demo_clean",
            "chunks": 38,
            "epoch_ids": [1, 2, 3],
            "selection_rule": PIPELINE.SELECTION_RULE,
            "training_config": str(self.training_path),
            "training_summary": str(self.formal / "update" / "summary.json"),
            "output_root": str(self.formal / "eval"),
            "screening": {
                "noise_base_seeds": [2000, 2001],
                "episode_records": screening,
            },
            "heldout": {
                "noise_base_seeds": heldout_banks,
                "episode_records": heldout,
                "record_pairs": [
                    {"noise_base_seed": bank, "seed": row["seed"]}
                    for bank in heldout_banks
                    for row in heldout
                ],
            },
        }
        _write(self.protocol_path, protocol)

        self.decision_path = self.root / "qualification" / "decision.json"
        decision = {
            "schema": PIPELINE.GATE_SCHEMA,
            "status": "PASS",
            "task_id": self.task,
            "episodes": 8,
            "thresholds": PIPELINE.STRICT_GATE_THRESHOLDS,
        }
        _write(self.decision_path, decision)
        self.manifest_path = self.workspace / "configs" / "pipeline.json"
        self.manifest = {
            "schema": PIPELINE.PIPELINE_SCHEMA,
            "task": self.task,
            "run_id": "unit-run-v1",
            "workspace": str(self.workspace),
            "project_root": str(self.project),
            "python_bin": sys.executable,
            "formal_root": str(self.formal),
            "collection_root": str(self.collection),
            "qualification_decision": str(self.decision_path),
            "split_manifest": str(self.split_path),
            "training_config": str(self.training_path),
            "collection_configs": [str(path) for path in self.collection_paths],
            "eval_protocol": str(self.protocol_path),
            "gpu_policy": {
                "stable_gpu_count": 4,
                "burst_max_gpu_count": 8,
                "train_world_size": 4,
                "world_size_change_boundary": "epoch",
                "allow_mid_epoch_elastic_resize": False,
                "allowed_gpu_ids": list(range(8)),
                "require_graphics_exclusive_for_rollout": True,
                "min_free_mib": {
                    "collection": 70_000,
                    "update": 70_000,
                    "evaluation": 45_000,
                },
            },
            "execution": {
                "training_backend": "joint_lora_ddp_v1",
                "update_argv": [sys.executable, "-c", "raise SystemExit(0)"],
            },
            "optional_eval": {
                "enabled": False,
                "after_stage": "heldout",
                "selection_input": False,
                "argv": [],
            },
        }
        self._save_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _save_manifest(self) -> None:
        _write(self.manifest_path, self.manifest)

    def _validate(self) -> dict[str, object]:
        def local_data_path(value, label, *, must_exist=False):
            return PIPELINE._path(value, label, must_exist=must_exist)

        with mock.patch.object(PIPELINE, "_ssd_path", side_effect=local_data_path):
            return PIPELINE.validate_contract(self.manifest_path)

    def test_valid_fixed_contract_passes(self) -> None:
        receipt = self._validate()
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["split_counts"], {
            "train": 8,
            "calibration": 4,
            "screening": 2,
            "heldout": 6,
        })
        self.assertEqual(receipt["epochs"], 3)
        self.assertEqual(receipt["effective_batch_size"], 4)
        self.assertEqual(receipt["heldout_exact_pairs"], 12)
        self.assertEqual(receipt["training_backend"], "joint_lora_ddp_v1")

    def test_certified_single_process_backend_is_usable_without_contract_drift(self) -> None:
        self.manifest["execution"]["training_backend"] = (
            "joint_lora_single_process_v1"
        )
        self.manifest["gpu_policy"]["train_world_size"] = 1
        self._save_manifest()
        receipt = self._validate()
        self.assertEqual(receipt["training_backend"], "joint_lora_single_process_v1")
        self.assertEqual(receipt["gpu_policy"]["train_world_size"], 1)

    def test_disjoint_four_gpu_allocation_and_explicit_overlay_are_valid(self) -> None:
        self.manifest["execution"]["training_backend"] = (
            "joint_lora_single_process_v1"
        )
        self.manifest["gpu_policy"].update(
            train_world_size=1,
            burst_max_gpu_count=4,
            allowed_gpu_ids=[4, 5, 6, 7],
            allow_existing_compute_processes=True,
        )
        self._save_manifest()
        receipt = self._validate()
        self.assertEqual(receipt["gpu_policy"]["allowed_gpu_ids"], [4, 5, 6, 7])
        self.assertTrue(
            receipt["gpu_policy"]["allow_existing_compute_processes"]
        )
        self.assertTrue(
            receipt["gpu_policy"]["require_graphics_exclusive_for_rollout"]
        )

    def test_pmon_parser_detects_only_compute_graphics_processes(self) -> None:
        output = """# gpu pid type sm mem enc dec jpg ofa command
    3 - - - - - - - - -
    3 751537 C 9 1 - - - - python
    3 753920 G 0 0 - - - - secondary_renderer.py
    3 753921 C+G 16 0 - - - - rt_dual_source.py
"""
        self.assertEqual(
            PIPELINE._parse_pmon_graphics_processes(output, expected_gpu=3),
            [
                {
                    "gpu": 3,
                    "pid": 753921,
                    "type": "C+G",
                    "command": "rt_dual_source.py",
                }
            ],
        )

    def test_rollout_rejects_graphics_overlay_but_update_allows_it(self) -> None:
        manifest = {
            "gpu_policy": {
                "allowed_gpu_ids": [0, 1, 2, 3],
                "stable_gpu_count": 4,
                "burst_max_gpu_count": 4,
                "train_world_size": 1,
                "allow_existing_compute_processes": True,
                "require_graphics_exclusive_for_rollout": True,
                "min_free_mib": {
                    "collection": 65_000,
                    "update": 70_000,
                    "evaluation": 45_000,
                },
            }
        }
        gpu_rows = [
            {"index": index, "free_mib": 80_000} for index in range(4)
        ]
        graphics = lambda index: ([{"pid": 99}] if index == 3 else [])
        with mock.patch.object(PIPELINE, "_gpu_rows", return_value=gpu_rows), \
             mock.patch.object(
                 PIPELINE, "_gpu_graphics_processes", side_effect=graphics
             ):
            with self.assertRaisesRegex(
                RuntimeError, "rollout graphics exclusivity=True"
            ):
                PIPELINE._allocate_gpus(manifest, "collection")
            self.assertEqual(PIPELINE._allocate_gpus(manifest, "update"), [0])

    def test_gpu_subset_must_still_cover_stable_allocation(self) -> None:
        self.manifest["gpu_policy"].update(
            burst_max_gpu_count=3,
            allowed_gpu_ids=[0, 1, 2],
        )
        self._save_manifest()
        with self.assertRaisesRegex(ValueError, "at least four unique ids"):
            self._validate()

    def test_training_backend_and_world_size_cannot_disagree(self) -> None:
        self.manifest["execution"]["training_backend"] = (
            "joint_lora_single_process_v1"
        )
        self._save_manifest()
        with self.assertRaisesRegex(ValueError, "train_world_size"):
            self._validate()

    def test_prompt_or_noise_leakage_fails(self) -> None:
        original = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        changed = copy.deepcopy(original)
        changed["heldout"]["episode_records"][0]["instruction"] += " changed"
        _write(self.protocol_path, changed)
        with self.assertRaisesRegex(ValueError, "heldout.episode_records"):
            self._validate()

        changed = copy.deepcopy(original)
        changed["heldout"]["noise_base_seeds"][0] = 2000
        _write(self.protocol_path, changed)
        with self.assertRaisesRegex(ValueError, "noise banks overlap"):
            self._validate()

    def test_failed_gate_is_rejected(self) -> None:
        decision = json.loads(self.decision_path.read_text(encoding="utf-8"))
        decision["status"] = "FAIL"
        _write(self.decision_path, decision)
        with self.assertRaisesRegex(ValueError, "qualification.status"):
            self._validate()

    def test_recipe_drift_is_rejected(self) -> None:
        mutations = {
            "inner_epochs": 4,
            "adapter_rank": 16,
            "lora_block_indices": list(range(29)),
            "effective_batch_size": 8,
        }
        original = json.loads(self.training_path.read_text(encoding="utf-8"))
        for field, value in mutations.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed[field] = value
                _write(self.training_path, changed)
                with self.assertRaisesRegex(ValueError, field):
                    self._validate()
        _write(self.training_path, original)

    def test_optional_eval_can_never_select_a_checkpoint(self) -> None:
        self.manifest["optional_eval"]["selection_input"] = True
        self._save_manifest()
        with self.assertRaisesRegex(ValueError, "selection_input"):
            self._validate()

    def test_partial_stage_is_not_automatically_retried(self) -> None:
        for path in self.collection_paths:
            output = Path(json.loads(path.read_text(encoding="utf-8"))["output_dir"])
            _write(output / "summary.json", {"status": "PASS"})
        update_log = self.formal / "logs" / "update.log"
        update_log.parent.mkdir(parents=True, exist_ok=True)
        update_log.write_text("started\n", encoding="utf-8")
        status = PIPELINE._phase_status(self.manifest)
        with self.assertRaisesRegex(RuntimeError, "refusing automatic rerun"):
            PIPELINE._next_stage(status)
        contract = {"manifest_sha256": "unit"}
        payload = PIPELINE._status_payload(self.manifest_path, contract)
        self.assertEqual(payload["status"], "BLOCKED")
        self.assertIsNone(payload["next_stage"])
        self.assertIn("refusing automatic rerun", payload["blocker"])

    def test_epoch_three_uses_the_final_checkpoint_artifact(self) -> None:
        for path in self.collection_paths:
            output = Path(json.loads(path.read_text(encoding="utf-8"))["output_dir"])
            _write(output / "summary.json", {"status": "PASS"})
        update = self.formal / "update"
        _write(update / "summary.json", {"status": "PASS"})
        checkpoints = update / "epoch_checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        (checkpoints / "checkpoint_epoch_01.pt").write_bytes(b"epoch1")
        (checkpoints / "checkpoint_epoch_02.pt").write_bytes(b"epoch2")
        (update / "checkpoint_trajectory_update.pt").write_bytes(b"epoch3")

        status = PIPELINE._phase_status(self.manifest)
        self.assertTrue(status["update"]["done"])
        self.assertEqual(
            Path(status["update"]["checkpoints"][-1]).name,
            "checkpoint_trajectory_update.pt",
        )

    def test_large_artifact_paths_are_restricted_to_ssd(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be under /ssd/data"):
            PIPELINE._ssd_path(str(self.root / "large.pt"), "artifact")
        accepted = PIPELINE._ssd_path(
            "/ssd/data/example/large.pt", "artifact", must_exist=False
        )
        self.assertEqual(str(accepted), "/ssd/data/example/large.pt")


if __name__ == "__main__":
    unittest.main()
