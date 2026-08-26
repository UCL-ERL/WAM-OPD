from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import ModuleType
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import torch

try:
    import experiments.run_handover_mic_success_path_eval as success_path_eval
except ModuleNotFoundError as exc:
    if exc.name != "experiments":
        raise
    import remote_patch.experiments.run_handover_mic_success_path_eval as success_path_eval


ProgressTrace = success_path_eval.ProgressTrace
EvalUnit = success_path_eval.EvalUnit
Candidate = success_path_eval.Candidate
_candidate_aggregate = success_path_eval._candidate_aggregate
_build_pair = success_path_eval._build_pair
_heldout_bucket = success_path_eval._heldout_bucket
_load_candidates = success_path_eval._load_candidates
_resolve_epoch_candidate_checkpoint = (
    success_path_eval._resolve_epoch_candidate_checkpoint
)
_run_heldout = success_path_eval._run_heldout
_run_policy = success_path_eval._run_policy
_unit_complete = success_path_eval._unit_complete
_validate_task_protocol_contract = (
    success_path_eval._validate_task_protocol_contract
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_final_checkpoint_fixture(
    output_dir: Path,
    *,
    checkpoint_role: str = "success_path_final",
    completed_inner_epochs: int = 3,
) -> tuple[Path, dict[str, object]]:
    checkpoint = output_dir / "checkpoint_trajectory_update.pt"
    payload = {
        "checkpoint_role": checkpoint_role,
        "completed_inner_epochs": completed_inner_epochs,
        "task": "handover_mic",
        "task_config": "demo_clean",
        "objective": "coherent_tt_consistency",
        "coherent_tt_variant": "success_path_v1",
        "global_optimizer_step": 12,
        "policy_version_after": "policy-after",
        "task_contract_hash": "task-contract",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    summary: dict[str, object] = {
        "status": "PASS",
        "coherent_tt_variant": "success_path_v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "global_optimizer_step": 12,
        "policy_version_after": "policy-after",
        "task_contract_hash": "task-contract",
        "update": {
            "inner_epochs": 3,
            "objective": "coherent_tt_consistency",
            "coherent_tt_variant": "success_path_v1",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return checkpoint, summary


def _protocol() -> dict[str, object]:
    return {
        "epoch_ids": [1, 2, 3],
        "task": "handover_mic",
        "task_config": "demo_clean",
    }


def _write_unit_receipt(
    output: Path,
    *,
    unit: object,
    student: Path,
    adapter: Path | None,
    adapter_state_sha256: str | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "status": "PASS",
                "adapter_state_sha256": adapter_state_sha256,
            }
        ),
        encoding="utf-8",
    )
    assert isinstance(unit, EvalUnit)
    (output.parent / "unit_summary.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "phase": "screening",
                "condition": "adapted" if adapter is not None else "released",
                "seed": unit.seed,
                "prompt": unit.prompt,
                "noise_base_seed": unit.noise_base_seed,
                "student": str(student.resolve()),
                "adapter_state": (
                    str(adapter.resolve()) if adapter is not None else None
                ),
                "protocol_sha256": "protocol-sha",
                "record_requested": False,
            }
        ),
        encoding="utf-8",
    )


class NewCandidateTaskPortabilityTest(unittest.TestCase):
    def test_protocol_progress_and_aggregates_support_all_pass_candidates(self):
        horizons = {
            "place_a2b_right": (13, 400),
            "place_bread_basket": (23, 700),
            "place_cans_plasticbox": (26, 800),
            "rotate_qrcode": (13, 400),
            "stamp_seal": (13, 400),
        }
        row = {
            "checkpoint_sha256": "adapter-sha",
            "released_checkpoint_sha256": None,
            "released_success": False,
            "opd_success": True,
            "rescue": True,
            "regression": False,
            "stage_improvement": True,
            "stage_regression": False,
            "released_max_ordinal_stage": 1,
            "adapted_max_ordinal_stage": 3,
            "paired_max_ordinal_delta": 2,
        }
        candidate = Candidate(
            epoch=1,
            checkpoint=Path("/ssd/data/adapter.pt"),
            checkpoint_sha256="adapter-sha",
            fixed_calibration_total_loss=0.5,
        )
        for task, (chunks, max_control_steps) in horizons.items():
            with self.subTest(task=task):
                protocol = {
                    "schema": f"waopd_{task}_success_path_eval_protocol_v1",
                    "task": task,
                    "task_config": "demo_clean",
                    "chunks": chunks,
                    "max_control_steps": max_control_steps,
                }
                self.assertEqual(
                    _validate_task_protocol_contract(protocol), task
                )

                trace = ProgressTrace(task=task)
                for index, stage in enumerate((1, 3)):
                    trace.observation_callback(
                        {
                            "event": "post_action_observation",
                            "control_step": index * 32,
                            "macro_index": index,
                            "frame_st_id": index * 2,
                            "task_success": stage == 3,
                            "eval_success": stage == 3,
                            "evaluator_state": {
                                "metric": "official_predicate_milestone",
                                "ordinal_stage": stage,
                                "official_success": stage == 3,
                            },
                        }
                    )
                summary = trace.summarize(
                    {"success": True, "progress": {"ordinal_stage": 3}}
                )
                self.assertEqual(summary["max_ordinal_stage"], 3)
                self.assertEqual(summary["official_success_observations"], 1)
                self.assertFalse(
                    summary[
                        "official_stage_diagnostics_are_selection_input"
                    ]
                )

                candidate_summary = _candidate_aggregate(
                    candidate, [row], task=task
                )
                heldout_summary = _heldout_bucket([row], task=task)
                self.assertFalse(
                    candidate_summary["official_stage_diagnostics"][
                        "selection_input"
                    ]
                )
                self.assertFalse(
                    heldout_summary["official_stage_diagnostics"][
                        "selection_input"
                    ]
                )


class CabinetSuccessPathEvalTest(unittest.TestCase):
    def test_cabinet_progress_trace_summarizes_official_pose_release_stages(
        self,
    ) -> None:
        trace = ProgressTrace(task="put_object_cabinet")
        states = [
            {
                "ordinal_stage": 0,
                "xy_inside": False,
                "z_valid": False,
                "gripper_open": True,
                "placement_valid": False,
                "z_delta": 0.0,
                "xy_linf_error": 0.30,
            },
            {
                "ordinal_stage": 2,
                "xy_inside": True,
                "z_valid": True,
                "gripper_open": False,
                "placement_valid": True,
                "z_delta": 0.05,
                "xy_linf_error": 0.01,
            },
            {
                "ordinal_stage": 3,
                "xy_inside": True,
                "z_valid": True,
                "gripper_open": True,
                "placement_valid": True,
                "z_delta": 0.04,
                "xy_linf_error": 0.02,
            },
        ]
        for index, state in enumerate(states):
            trace.observation_callback(
                {
                    "event": (
                        "initial_observation"
                        if index == 0
                        else "post_action_observation"
                    ),
                    "control_step": index * 32,
                    "macro_index": index,
                    "frame_st_id": index * 2,
                    "task_success": index == 2,
                    "eval_success": index == 2,
                    "evaluator_state": {
                        "metric": "official_predicate_milestone",
                        **state,
                    },
                }
            )

        summary = trace.summarize(
            {
                "success": True,
                "progress": {"ordinal_stage": 3, "placement_valid": True},
            }
        )

        self.assertTrue(summary["success"])
        self.assertEqual(summary["max_ordinal_stage"], 3)
        self.assertEqual(
            summary["stage_first_steps"], {"1": 32, "2": 32, "3": 64}
        )
        self.assertTrue(summary["max_placement_valid"])
        self.assertEqual(summary["first_placement_valid_step"], 32)
        self.assertEqual(summary["official_success_observations"], 1)
        self.assertAlmostEqual(summary["min_xy_linf_error"], 0.01)
        self.assertEqual(summary["min_xy_linf_error_step"], 32)
        self.assertNotIn("max_contact", summary)

    def test_cabinet_protocol_requires_native_23_chunk_700_control_horizon(
        self,
    ) -> None:
        protocol = {
            "schema": "waopd_put_object_cabinet_success_path_eval_protocol_v1",
            "task": "put_object_cabinet",
            "task_config": "demo_clean",
            "chunks": 23,
            "max_control_steps": 700,
        }
        self.assertEqual(
            _validate_task_protocol_contract(protocol), "put_object_cabinet"
        )

        for field, invalid in (("chunks", 20), ("max_control_steps", 600)):
            malformed = {**protocol, field: invalid}
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, rf"protocol {field} must be"
            ):
                _validate_task_protocol_contract(malformed)


class PlaceShoeSuccessPathEvalTest(unittest.TestCase):
    def test_place_shoe_progress_trace_uses_official_pose_release_stages(self):
        trace = ProgressTrace(task="place_shoe")
        states = [
            {
                "ordinal_stage": 0,
                "xy_valid": False,
                "quaternion_valid": False,
                "both_grippers_open": True,
                "pose_valid": False,
                "xy_linf_normalized_error": 2.0,
                "quaternion_linf_normalized_error": 3.0,
                "shoe_z": 0.74,
                "shoe_z_is_official": False,
            },
            {
                "ordinal_stage": 2,
                "xy_valid": True,
                "quaternion_valid": True,
                "both_grippers_open": False,
                "pose_valid": True,
                "xy_linf_normalized_error": 0.1,
                "quaternion_linf_normalized_error": 0.2,
                "shoe_z": -4.0,
                "shoe_z_is_official": False,
            },
            {
                "ordinal_stage": 3,
                "xy_valid": True,
                "quaternion_valid": True,
                "both_grippers_open": True,
                "pose_valid": True,
                "xy_linf_normalized_error": 0.2,
                "quaternion_linf_normalized_error": 0.1,
                "shoe_z": 12.0,
                "shoe_z_is_official": False,
            },
        ]
        for index, state in enumerate(states):
            trace.observation_callback(
                {
                    "event": (
                        "initial_observation"
                        if index == 0
                        else "post_action_observation"
                    ),
                    "control_step": index * 32,
                    "macro_index": index,
                    "frame_st_id": index * 2,
                    "task_success": index == 2,
                    "eval_success": index == 2,
                    "evaluator_state": {
                        "metric": "official_predicate_milestone",
                        **state,
                    },
                }
            )

        summary = trace.summarize(
            {
                "success": True,
                "progress": {"ordinal_stage": 3, "pose_valid": True},
            }
        )

        self.assertEqual(summary["max_ordinal_stage"], 3)
        self.assertEqual(
            summary["stage_first_steps"], {"1": 32, "2": 32, "3": 64}
        )
        self.assertTrue(summary["max_pose_valid"])
        self.assertEqual(summary["first_pose_valid_step"], 32)
        self.assertEqual(summary["official_success_observations"], 1)
        self.assertAlmostEqual(
            summary["min_xy_linf_normalized_error"], 0.1
        )
        self.assertAlmostEqual(
            summary["min_quaternion_linf_normalized_error"], 0.1
        )
        self.assertEqual(summary["min_shoe_z"], -4.0)
        self.assertEqual(summary["max_shoe_z"], 12.0)
        self.assertFalse(summary["shoe_z_is_official"])
        self.assertFalse(summary["shoe_z_is_selection_input"])

    def test_place_shoe_protocol_requires_native_17_chunk_500_control_horizon(
        self,
    ) -> None:
        protocol = {
            "schema": "waopd_place_shoe_success_path_eval_protocol_v1",
            "task": "place_shoe",
            "task_config": "demo_clean",
            "chunks": 17,
            "max_control_steps": 500,
        }
        self.assertEqual(
            _validate_task_protocol_contract(protocol), "place_shoe"
        )
        expected_schemas = {
            "unit_schema": "waopd_place_shoe_success_path_eval_unit_v1",
            "pair_schema": "waopd_place_shoe_success_path_progress_pair_v1",
            "screen_summary_schema": (
                "waopd_place_shoe_success_path_screen_summary_v1"
            ),
            "selection_schema": "waopd_place_shoe_success_path_selection_v1",
            "heldout_summary_schema": (
                "waopd_place_shoe_success_path_heldout_summary_v1"
            ),
            "screen_epoch_receipt_schema": (
                "waopd_place_shoe_success_path_screen_epoch_receipt_v1"
            ),
        }
        for name, expected in expected_schemas.items():
            with self.subTest(schema=name):
                self.assertEqual(
                    success_path_eval._task_schema("place_shoe", name),
                    expected,
                )

        for field, invalid in (("chunks", 20), ("max_control_steps", 600)):
            malformed = {**protocol, field: invalid}
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, rf"protocol {field} must be"
            ):
                _validate_task_protocol_contract(malformed)


class BlocksRankingSizeSuccessPathEvalTest(unittest.TestCase):
    def test_pair_reports_ordering_diagnostics_without_selection_input(self):
        released_progress = {
            "max_ordinal_stage": 1,
            "max_ordered_pair_count": 1,
            "first_fully_ordered_step": None,
        }
        adapted_progress = {
            "max_ordinal_stage": 3,
            "max_ordered_pair_count": 2,
            "first_fully_ordered_step": 256,
        }
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            released_path = root / "released.json"
            adapted_path = root / "adapted.json"
            released_path.write_text(
                json.dumps({"adapter_state_sha256": None}), encoding="utf-8"
            )
            adapted_path.write_text(
                json.dumps({"adapter_state_sha256": "blocks-size-sha"}),
                encoding="utf-8",
            )
            fake_pair_module = ModuleType(
                "experiments.run_open_microwave_paired_multinoise"
            )
            fake_pair_module._validate_pair = lambda **_kwargs: {
                "released_success": False,
                "opd_success": True,
                "rescue": True,
                "regression": False,
                "noise_base_seed": 41001,
                "seed": 12,
            }
            with mock.patch.dict(
                sys.modules,
                {
                    "experiments.run_open_microwave_paired_multinoise": (
                        fake_pair_module
                    )
                },
            ):
                row = _build_pair(
                    protocol={
                        "task": "blocks_ranking_size",
                        "task_config": "demo_clean",
                        "chunks": 38,
                        "max_control_steps": 1200,
                        "student": "/ssd/data/student",
                    },
                    unit=EvalUnit(
                        phase="screening",
                        noise_base_seed=41001,
                        seed=12,
                        prompt="rank the blocks by size",
                    ),
                    condition="epoch_01",
                    checkpoint=Path("/ssd/data/blocks_ranking_size_epoch_01.pt"),
                    expected_checkpoint_sha256="blocks-size-sha",
                    released_path=released_path,
                    adapted_path=adapted_path,
                    released_unit={"progress": released_progress},
                    adapted_unit={"progress": adapted_progress},
                    gpu=0,
                )

        self.assertEqual(
            row["schema"],
            "waopd_blocks_ranking_size_success_path_progress_pair_v1",
        )
        self.assertFalse(row["released_fully_ordered"])
        self.assertTrue(row["adapted_fully_ordered"])
        self.assertIsNone(row["released_first_fully_ordered_step"])
        self.assertEqual(row["adapted_first_fully_ordered_step"], 256)
        self.assertEqual(row["paired_max_ordered_pair_count_delta"], 1)
        self.assertFalse(row["ordering_diagnostics_are_selection_input"])

    def test_progress_trace_uses_official_ordering_and_release_stages(self):
        trace = ProgressTrace(task="blocks_ranking_size")
        states = [
            {
                "ordinal_stage": 0,
                "ordered_pair_12": False,
                "ordered_pair_23": False,
                "ordered_pair_count": 0,
                "both_grippers_open": False,
            },
            {
                "ordinal_stage": 1,
                "ordered_pair_12": True,
                "ordered_pair_23": False,
                "ordered_pair_count": 1,
                "both_grippers_open": False,
            },
            {
                "ordinal_stage": 2,
                "ordered_pair_12": True,
                "ordered_pair_23": True,
                "ordered_pair_count": 2,
                "both_grippers_open": False,
            },
            {
                "ordinal_stage": 3,
                "ordered_pair_12": True,
                "ordered_pair_23": True,
                "ordered_pair_count": 2,
                "both_grippers_open": True,
            },
        ]
        for index, state in enumerate(states):
            trace.observation_callback(
                {
                    "event": (
                        "initial_observation"
                        if index == 0
                        else "post_action_observation"
                    ),
                    "control_step": index * 32,
                    "macro_index": index,
                    "frame_st_id": index * 2,
                    "task_success": index == 3,
                    "eval_success": index == 3,
                    "evaluator_state": {
                        "metric": "official_predicate_milestone",
                        **state,
                    },
                }
            )

        summary = trace.summarize(
            {
                "success": True,
                "progress": {
                    "ordinal_stage": 3,
                    "ordered_pair_count": 2,
                    "both_grippers_open": True,
                },
            }
        )

        self.assertEqual(summary["max_ordinal_stage"], 3)
        self.assertEqual(
            summary["stage_first_steps"], {"1": 32, "2": 64, "3": 96}
        )
        self.assertEqual(summary["max_ordered_pair_count"], 2)
        self.assertEqual(summary["first_fully_ordered_step"], 64)
        self.assertEqual(summary["official_success_observations"], 1)
        self.assertFalse(summary["ordering_diagnostics_are_selection_input"])

    def test_protocol_requires_native_38_chunk_1200_control_horizon(self):
        try:
            from experiments.opd_task_specs import resolve_task_chunks
        except ModuleNotFoundError:
            from remote_patch.experiments.opd_task_specs import (
                resolve_task_chunks,
            )

        self.assertEqual(resolve_task_chunks("blocks_ranking_size"), 38)
        protocol = {
            "schema": (
                "waopd_blocks_ranking_size_success_path_eval_protocol_v1"
            ),
            "task": "blocks_ranking_size",
            "task_config": "demo_clean",
            "chunks": 38,
            "max_control_steps": 1200,
        }
        self.assertEqual(
            _validate_task_protocol_contract(protocol), "blocks_ranking_size"
        )

        for field, invalid in (("chunks", 37), ("max_control_steps", 1199)):
            malformed = {**protocol, field: invalid}
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, rf"protocol {field} must be"
            ):
                _validate_task_protocol_contract(malformed)

    def test_blocks_ranking_rgb_reuses_official_ordering_contract_with_distinct_schemas(
        self,
    ) -> None:
        try:
            from experiments.opd_task_specs import resolve_task_chunks
        except ModuleNotFoundError:
            from remote_patch.experiments.opd_task_specs import (
                resolve_task_chunks,
            )

        self.assertEqual(resolve_task_chunks("blocks_ranking_rgb"), 38)
        protocol = {
            "schema": "waopd_blocks_ranking_rgb_success_path_eval_protocol_v1",
            "task": "blocks_ranking_rgb",
            "task_config": "demo_clean",
            "chunks": 38,
            "max_control_steps": 1200,
        }
        self.assertEqual(
            _validate_task_protocol_contract(protocol), "blocks_ranking_rgb"
        )
        self.assertEqual(
            success_path_eval._task_schema("blocks_ranking_rgb", "pair_schema"),
            "waopd_blocks_ranking_rgb_success_path_progress_pair_v1",
        )

        trace = ProgressTrace(task="blocks_ranking_rgb")
        trace.observation_callback(
            {
                "event": "initial_observation",
                "control_step": 0,
                "macro_index": 0,
                "frame_st_id": 0,
                "task_success": False,
                "eval_success": False,
                "evaluator_state": {
                    "metric": "official_predicate_milestone",
                    "ordinal_stage": 1,
                    "ordered_pair_12": True,
                    "ordered_pair_23": False,
                    "ordered_pair_count": 1,
                    "both_grippers_open": False,
                },
            }
        )
        summary = trace.summarize(
            {
                "success": False,
                "progress": {
                    "ordinal_stage": 1,
                    "ordered_pair_count": 1,
                    "both_grippers_open": False,
                },
            }
        )
        self.assertEqual(summary["max_ordered_pair_count"], 1)
        self.assertFalse(summary["ordering_diagnostics_are_selection_input"])

    def test_ordering_diagnostics_are_reported_but_never_select_checkpoints(
        self,
    ) -> None:
        candidate = Candidate(
            epoch=1,
            checkpoint=Path("/ssd/data/blocks_ranking_size_epoch_01.pt"),
            checkpoint_sha256="blocks-size-sha",
            fixed_calibration_total_loss=0.25,
        )
        row = {
            "checkpoint_sha256": "blocks-size-sha",
            "released_checkpoint_sha256": None,
            "opd_success": True,
            "released_success": False,
            "stage_improvement": True,
            "stage_regression": False,
            "paired_max_ordinal_delta": 2,
            "rescue": True,
            "regression": False,
            "released_fully_ordered": False,
            "adapted_fully_ordered": True,
            "released_max_ordered_pair_count": 0,
            "adapted_max_ordered_pair_count": 2,
            "paired_max_ordered_pair_count_delta": 2,
        }

        aggregate = _candidate_aggregate(
            candidate, [row], task="blocks_ranking_size"
        )
        self.assertEqual(aggregate["selection_key"], [1, 2, 1, -0.25, -1])
        diagnostics = aggregate["ordering_diagnostics"]
        self.assertEqual(diagnostics["adapted_fully_ordered_units"], 1)
        self.assertEqual(
            diagnostics["sum_paired_max_ordered_pair_count_delta"], 2
        )
        self.assertFalse(diagnostics["selection_input"])

        heldout = _heldout_bucket([row], task="blocks_ranking_size")
        self.assertEqual(heldout["adapted_success"], 1)
        self.assertEqual(
            heldout["ordering_diagnostics"]
            ["sum_paired_max_ordered_pair_count_delta"],
            2,
        )
        self.assertFalse(
            heldout["ordering_diagnostics"]["selection_input"]
        )

    def test_place_shoe_candidate_aggregate_keeps_z_diagnostic_nonselection(
        self,
    ) -> None:
        candidate = Candidate(
            epoch=1,
            checkpoint=Path("/ssd/data/place_shoe_epoch_01.pt"),
            checkpoint_sha256="place-shoe-sha",
            fixed_calibration_total_loss=0.25,
        )
        row = {
            "checkpoint_sha256": "place-shoe-sha",
            "released_checkpoint_sha256": None,
            "opd_success": True,
            "released_success": False,
            "stage_improvement": True,
            "stage_regression": False,
            "paired_max_ordinal_delta": 1,
            "rescue": True,
            "regression": False,
            "released_pose_valid": False,
            "adapted_pose_valid": True,
            "paired_min_xy_linf_normalized_error_delta": 0.3,
            "paired_min_quaternion_linf_normalized_error_delta": 0.2,
            "paired_max_shoe_z_delta": 1000.0,
        }

        aggregate = _candidate_aggregate(
            candidate, [row], task="place_shoe"
        )

        self.assertEqual(aggregate["selection_key"], [1, 1, 1, -0.25, -1])
        diagnostics = aggregate["pose_diagnostics"]
        self.assertEqual(diagnostics["adapted_pose_valid_units"], 1)
        self.assertEqual(diagnostics["sum_paired_max_shoe_z_delta"], 1000.0)
        self.assertFalse(diagnostics["shoe_z_is_official"])
        self.assertFalse(diagnostics["selection_input"])

        heldout = _heldout_bucket([row], task="place_shoe")
        self.assertEqual(heldout["adapted_success"], 1)
        self.assertEqual(
            heldout["pose_diagnostics"]["sum_paired_max_shoe_z_delta"],
            1000.0,
        )
        self.assertFalse(
            heldout["pose_diagnostics"]["shoe_z_is_official"]
        )


class RankedScoutTaskSuccessPathEvalTest(unittest.TestCase):
    def test_ranked_task_protocols_use_native_horizons_and_unique_schemas(self):
        cases = (
            (
                "place_a2b_left",
                13,
                400,
                "waopd_place_a2b_left_success_path_eval_protocol_v1",
            ),
            (
                "stack_blocks_three",
                38,
                1200,
                "waopd_stack_blocks_three_success_path_eval_protocol_v1",
            ),
        )
        for task, chunks, max_control_steps, schema in cases:
            with self.subTest(task=task):
                protocol = {
                    "schema": schema,
                    "task": task,
                    "task_config": "demo_clean",
                    "chunks": chunks,
                    "max_control_steps": max_control_steps,
                }
                self.assertEqual(
                    _validate_task_protocol_contract(protocol), task
                )
                self.assertEqual(
                    success_path_eval._task_schema(task, "protocol_schema"),
                    schema,
                )
                for field, invalid in (
                    ("chunks", chunks - 1),
                    ("max_control_steps", max_control_steps - 1),
                ):
                    malformed = {**protocol, field: invalid}
                    with self.assertRaisesRegex(
                        ValueError, rf"protocol {field} must be"
                    ):
                        _validate_task_protocol_contract(malformed)

    def test_place_a2b_left_trace_and_aggregates_keep_diagnostics_nonselection(
        self,
    ) -> None:
        trace = ProgressTrace(task="place_a2b_left")
        for index, state in enumerate(
            (
                {
                    "ordinal_stage": 1,
                    "distance_valid": False,
                    "object_left_of_target": True,
                    "y_aligned": True,
                    "placement_valid": False,
                    "both_grippers_open": False,
                    "xy_distance": 0.25,
                    "y_abs_error": 0.01,
                },
                {
                    "ordinal_stage": 3,
                    "distance_valid": True,
                    "object_left_of_target": True,
                    "y_aligned": True,
                    "placement_valid": True,
                    "both_grippers_open": True,
                    "xy_distance": 0.12,
                    "y_abs_error": 0.005,
                },
            )
        ):
            trace.observation_callback(
                {
                    "event": (
                        "initial_observation"
                        if index == 0
                        else "post_action_observation"
                    ),
                    "control_step": index * 32,
                    "macro_index": index,
                    "frame_st_id": index * 2,
                    "task_success": index == 1,
                    "eval_success": index == 1,
                    "evaluator_state": {
                        "metric": "official_predicate_milestone",
                        **state,
                    },
                }
            )
        summary = trace.summarize(
            {
                "success": True,
                "progress": {"ordinal_stage": 3, "placement_valid": True},
            }
        )
        self.assertEqual(summary["max_ordinal_stage"], 3)
        self.assertTrue(summary["max_placement_valid"])
        self.assertEqual(summary["first_placement_valid_step"], 32)
        self.assertFalse(
            summary["placement_diagnostics_are_selection_input"]
        )

        candidate = Candidate(
            epoch=2,
            checkpoint=Path("/ssd/data/place_a2b_left_epoch_02.pt"),
            checkpoint_sha256="a2b-sha",
            fixed_calibration_total_loss=0.2,
        )
        row = {
            "checkpoint_sha256": "a2b-sha",
            "released_checkpoint_sha256": None,
            "opd_success": True,
            "released_success": False,
            "stage_improvement": True,
            "stage_regression": False,
            "paired_max_ordinal_delta": 2,
            "rescue": True,
            "regression": False,
            "released_placement_valid": False,
            "adapted_placement_valid": True,
            "paired_min_y_abs_error_delta": 0.02,
        }
        aggregate = _candidate_aggregate(
            candidate, [row], task="place_a2b_left"
        )
        self.assertEqual(aggregate["selection_key"], [1, 2, 1, -0.2, -2])
        self.assertFalse(
            aggregate["placement_diagnostics"]["selection_input"]
        )
        heldout = _heldout_bucket([row], task="place_a2b_left")
        self.assertEqual(
            heldout["placement_diagnostics"]
            ["sum_paired_min_y_abs_error_delta"],
            0.02,
        )

    def test_stack_blocks_three_trace_and_aggregates_keep_diagnostics_nonselection(
        self,
    ) -> None:
        trace = ProgressTrace(task="stack_blocks_three")
        for index, state in enumerate(
            (
                {
                    "ordinal_stage": 1,
                    "stacked_pair_12": True,
                    "stacked_pair_23": False,
                    "stacked_pair_count": 1,
                    "both_grippers_open": False,
                    "min_pair_linf_normalized_error": 0.2,
                    "max_pair_linf_normalized_error": 4.0,
                },
                {
                    "ordinal_stage": 3,
                    "stacked_pair_12": True,
                    "stacked_pair_23": True,
                    "stacked_pair_count": 2,
                    "both_grippers_open": True,
                    "min_pair_linf_normalized_error": 0.1,
                    "max_pair_linf_normalized_error": 0.3,
                },
            )
        ):
            trace.observation_callback(
                {
                    "event": (
                        "initial_observation"
                        if index == 0
                        else "post_action_observation"
                    ),
                    "control_step": index * 32,
                    "macro_index": index,
                    "frame_st_id": index * 2,
                    "task_success": index == 1,
                    "eval_success": index == 1,
                    "evaluator_state": {
                        "metric": "official_predicate_milestone",
                        **state,
                    },
                }
            )
        summary = trace.summarize(
            {
                "success": True,
                "progress": {
                    "ordinal_stage": 3,
                    "stacked_pair_count": 2,
                },
            }
        )
        self.assertEqual(summary["max_stacked_pair_count"], 2)
        self.assertEqual(summary["first_fully_stacked_step"], 32)
        self.assertFalse(
            summary["stacking_diagnostics_are_selection_input"]
        )

        candidate = Candidate(
            epoch=3,
            checkpoint=Path("/ssd/data/stack_blocks_three_epoch_03.pt"),
            checkpoint_sha256="stack-sha",
            fixed_calibration_total_loss=0.15,
        )
        row = {
            "checkpoint_sha256": "stack-sha",
            "released_checkpoint_sha256": None,
            "opd_success": True,
            "released_success": False,
            "stage_improvement": True,
            "stage_regression": False,
            "paired_max_ordinal_delta": 2,
            "rescue": True,
            "regression": False,
            "released_fully_stacked": False,
            "adapted_fully_stacked": True,
            "released_max_stacked_pair_count": 0,
            "adapted_max_stacked_pair_count": 2,
            "paired_max_stacked_pair_count_delta": 2,
            "paired_min_max_pair_linf_normalized_error_delta": 1.5,
        }
        aggregate = _candidate_aggregate(
            candidate, [row], task="stack_blocks_three"
        )
        self.assertEqual(
            aggregate["selection_key"], [1, 2, 1, -0.15, -3]
        )
        diagnostics = aggregate["stacking_diagnostics"]
        self.assertEqual(
            diagnostics["adapted_max_stacked_pair_count_sum"], 2
        )
        self.assertFalse(diagnostics["selection_input"])
        heldout = _heldout_bucket([row], task="stack_blocks_three")
        self.assertEqual(
            heldout["stacking_diagnostics"]
            ["sum_paired_max_stacked_pair_count_delta"],
            2,
        )


class ScanObjectSuccessPathEvalTest(unittest.TestCase):
    def test_scan_object_progress_trace_uses_official_conjunct_stages(self):
        trace = ProgressTrace(task="scan_object")
        states = [
            {
                "ordinal_stage": 0,
                "official_success": False,
                "projected_xyz_valid": False,
                "depth_valid": True,
                "both_grippers_closed": True,
                "scan_geometry_valid": False,
                "projected_xyz_linf_error": 0.30,
                "scanner_axis_depth": 0.04,
            },
            {
                "ordinal_stage": 1,
                "official_success": False,
                "projected_xyz_valid": True,
                "depth_valid": False,
                "both_grippers_closed": True,
                "scan_geometry_valid": False,
                "projected_xyz_linf_error": 0.01,
                "scanner_axis_depth": 0.09,
            },
            {
                "ordinal_stage": 2,
                "official_success": False,
                "projected_xyz_valid": True,
                "depth_valid": True,
                "both_grippers_closed": False,
                "scan_geometry_valid": True,
                "projected_xyz_linf_error": 0.02,
                "scanner_axis_depth": 0.05,
            },
            {
                "ordinal_stage": 3,
                "official_success": True,
                "projected_xyz_valid": True,
                "depth_valid": True,
                "both_grippers_closed": True,
                "scan_geometry_valid": True,
                "projected_xyz_linf_error": 0.015,
                "scanner_axis_depth": 0.06,
            },
        ]
        for index, state in enumerate(states):
            trace.observation_callback(
                {
                    "event": (
                        "initial_observation"
                        if index == 0
                        else "post_action_observation"
                    ),
                    "control_step": index * 32,
                    "macro_index": index,
                    "frame_st_id": index * 2,
                    "task_success": index == 3,
                    "eval_success": index == 3,
                    "evaluator_state": {
                        "metric": "official_predicate_milestone",
                        **state,
                    },
                }
            )

        summary = trace.summarize(
            {
                "success": True,
                "progress": {
                    "ordinal_stage": 3,
                    "scan_geometry_valid": True,
                },
            }
        )

        self.assertEqual(summary["max_ordinal_stage"], 3)
        self.assertEqual(
            summary["stage_first_steps"],
            {"1": 32, "2": 64, "3": 96},
        )
        self.assertTrue(summary["max_scan_geometry_valid"])
        self.assertEqual(summary["first_scan_geometry_valid_step"], 64)
        self.assertEqual(summary["official_success_observations"], 1)
        self.assertAlmostEqual(
            summary["min_projected_xyz_linf_error"], 0.01
        )
        self.assertEqual(
            summary["min_projected_xyz_linf_error_step"], 32
        )
        self.assertAlmostEqual(summary["min_scanner_axis_depth"], 0.04)
        self.assertAlmostEqual(summary["max_scanner_axis_depth"], 0.09)
        self.assertFalse(
            summary["continuous_scan_diagnostics_are_selection_input"]
        )

    def test_scan_object_protocol_uses_native_17_chunk_500_control_horizon(
        self,
    ) -> None:
        protocol = {
            "schema": "waopd_scan_object_success_path_eval_protocol_v1",
            "task": "scan_object",
            "task_config": "demo_clean",
            "chunks": 17,
            "max_control_steps": 500,
        }
        self.assertEqual(
            _validate_task_protocol_contract(protocol), "scan_object"
        )
        expected_schemas = {
            "unit_schema": "waopd_scan_object_success_path_eval_unit_v1",
            "pair_schema": "waopd_scan_object_success_path_progress_pair_v1",
            "screen_summary_schema": (
                "waopd_scan_object_success_path_screen_summary_v1"
            ),
            "selection_schema": (
                "waopd_scan_object_success_path_selection_v1"
            ),
            "heldout_summary_schema": (
                "waopd_scan_object_success_path_heldout_summary_v1"
            ),
            "screen_epoch_receipt_schema": (
                "waopd_scan_object_success_path_screen_epoch_receipt_v1"
            ),
        }
        for name, expected in expected_schemas.items():
            with self.subTest(schema=name):
                self.assertEqual(
                    success_path_eval._task_schema("scan_object", name),
                    expected,
                )

        for field, invalid in (("chunks", 20), ("max_control_steps", 600)):
            malformed = {**protocol, field: invalid}
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, rf"protocol {field} must be"
            ):
                _validate_task_protocol_contract(malformed)

    def test_scan_object_pair_and_aggregates_keep_continuous_metrics_diagnostic(
        self,
    ) -> None:
        candidate = Candidate(
            epoch=2,
            checkpoint=Path("/ssd/data/scan_object_epoch_02.pt"),
            checkpoint_sha256="scan-sha",
            fixed_calibration_total_loss=0.2,
        )
        released_progress = {
            "max_ordinal_stage": 1,
            "max_scan_geometry_valid": False,
            "first_scan_geometry_valid_step": None,
            "max_depth_valid": False,
            "max_both_grippers_closed": True,
            "min_projected_xyz_linf_error": 0.03,
        }
        adapted_progress = {
            "max_ordinal_stage": 3,
            "max_scan_geometry_valid": True,
            "first_scan_geometry_valid_step": 96,
            "max_depth_valid": True,
            "max_both_grippers_closed": True,
            "min_projected_xyz_linf_error": 0.01,
        }
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            released_path = root / "released.json"
            adapted_path = root / "adapted.json"
            released_path.write_text(
                json.dumps({"adapter_state_sha256": None}),
                encoding="utf-8",
            )
            adapted_path.write_text(
                json.dumps({"adapter_state_sha256": "scan-sha"}),
                encoding="utf-8",
            )
            fake_pair_module = ModuleType(
                "experiments.run_open_microwave_paired_multinoise"
            )
            fake_pair_module._validate_pair = lambda **_kwargs: {
                "released_success": False,
                "opd_success": True,
                "rescue": True,
                "regression": False,
                "noise_base_seed": 41001,
                "seed": 12,
            }
            with mock.patch.dict(
                sys.modules,
                {
                    "experiments.run_open_microwave_paired_multinoise": (
                        fake_pair_module
                    )
                },
            ):
                row = _build_pair(
                    protocol={
                        "task": "scan_object",
                        "task_config": "demo_clean",
                        "chunks": 17,
                        "max_control_steps": 500,
                        "student": "/ssd/data/student",
                    },
                    unit=EvalUnit(
                        phase="screening",
                        noise_base_seed=41001,
                        seed=12,
                        prompt="scan the tea box",
                    ),
                    condition="epoch_02",
                    checkpoint=candidate.checkpoint,
                    expected_checkpoint_sha256="scan-sha",
                    released_path=released_path,
                    adapted_path=adapted_path,
                    released_unit={"progress": released_progress},
                    adapted_unit={"progress": adapted_progress},
                    gpu=0,
                )

        self.assertEqual(
            row["schema"],
            "waopd_scan_object_success_path_progress_pair_v1",
        )
        self.assertEqual(row["paired_max_ordinal_delta"], 2)
        self.assertAlmostEqual(
            row["paired_min_projected_xyz_linf_error_delta"], 0.02
        )
        self.assertFalse(
            row["continuous_scan_diagnostics_are_selection_input"]
        )

        aggregate = _candidate_aggregate(
            candidate, [row], task="scan_object"
        )
        self.assertEqual(aggregate["selection_key"], [1, 2, 1, -0.2, -2])
        self.assertEqual(
            aggregate["scan_diagnostics"][
                "adapted_scan_geometry_valid_units"
            ],
            1,
        )
        self.assertFalse(
            aggregate["scan_diagnostics"]["selection_input"]
        )

        heldout = _heldout_bucket([row], task="scan_object")
        self.assertEqual(heldout["adapted_success"], 1)
        self.assertAlmostEqual(
            heldout["scan_diagnostics"][
                "sum_paired_min_projected_xyz_linf_error_delta"
            ],
            0.02,
        )
        self.assertFalse(heldout["scan_diagnostics"]["selection_input"])


class SuccessPathFinalCheckpointResolutionTest(unittest.TestCase):
    def test_existing_legacy_epoch_checkpoint_has_priority(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            output_dir = Path(raw_dir)
            legacy = (
                output_dir
                / "epoch_checkpoints"
                / "checkpoint_epoch_03.pt"
            )
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(b"legacy")
            _write_final_checkpoint_fixture(output_dir)

            with mock.patch.object(
                success_path_eval, "_under_ssd", return_value=True
            ):
                resolved, resolved_sha256 = _resolve_epoch_candidate_checkpoint(
                    epoch=3,
                    training_summary={},
                    summary_path=output_dir / "summary.json",
                    protocol=_protocol(),
                )

            self.assertEqual(resolved, legacy.resolve())
            self.assertEqual(resolved_sha256, _sha256(legacy))

    def test_final_epoch_falls_back_to_bound_root_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            output_dir = Path(raw_dir)
            checkpoint, summary = _write_final_checkpoint_fixture(output_dir)

            with mock.patch.object(
                success_path_eval, "_under_ssd", return_value=True
            ):
                resolved, resolved_sha256 = _resolve_epoch_candidate_checkpoint(
                    epoch=3,
                    training_summary=summary,
                    summary_path=output_dir / "summary.json",
                    protocol=_protocol(),
                )

            self.assertEqual(resolved, checkpoint.resolve())
            self.assertEqual(resolved_sha256, summary["checkpoint_sha256"])

    def test_root_fallback_rejects_wrong_checkpoint_role(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            output_dir = Path(raw_dir)
            _checkpoint, summary = _write_final_checkpoint_fixture(
                output_dir, checkpoint_role="success_path_epoch"
            )

            with mock.patch.object(
                success_path_eval, "_under_ssd", return_value=True
            ), self.assertRaisesRegex(RuntimeError, "checkpoint_role"):
                _resolve_epoch_candidate_checkpoint(
                    epoch=3,
                    training_summary=summary,
                    summary_path=output_dir / "summary.json",
                    protocol=_protocol(),
                )

    def test_root_fallback_rejects_incomplete_final_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            output_dir = Path(raw_dir)
            _checkpoint, summary = _write_final_checkpoint_fixture(
                output_dir, completed_inner_epochs=2
            )

            with mock.patch.object(
                success_path_eval, "_under_ssd", return_value=True
            ), self.assertRaisesRegex(RuntimeError, "completed_inner_epochs"):
                _resolve_epoch_candidate_checkpoint(
                    epoch=3,
                    training_summary=summary,
                    summary_path=output_dir / "summary.json",
                    protocol=_protocol(),
                )

    def test_nonfinal_epoch_never_uses_root_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            output_dir = Path(raw_dir)
            _checkpoint, summary = _write_final_checkpoint_fixture(output_dir)

            with mock.patch.object(
                success_path_eval, "_under_ssd", return_value=True
            ), self.assertRaisesRegex(FileNotFoundError, "epoch checkpoint"):
                _resolve_epoch_candidate_checkpoint(
                    epoch=2,
                    training_summary=summary,
                    summary_path=output_dir / "summary.json",
                    protocol=_protocol(),
                )

    def test_root_fallback_rejects_summary_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            output_dir = Path(raw_dir)
            _checkpoint, summary = _write_final_checkpoint_fixture(output_dir)
            summary["checkpoint_sha256"] = "0" * 64

            with mock.patch.object(
                success_path_eval, "_under_ssd", return_value=True
            ), self.assertRaisesRegex(RuntimeError, "SHA-256"):
                _resolve_epoch_candidate_checkpoint(
                    epoch=3,
                    training_summary=summary,
                    summary_path=output_dir / "summary.json",
                    protocol=_protocol(),
                )

    def test_root_fallback_rejects_mutation_between_hash_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            output_dir = Path(raw_dir)
            checkpoint, summary = _write_final_checkpoint_fixture(output_dir)
            real_torch_load = torch.load
            mutated_payload = real_torch_load(
                checkpoint, map_location="cpu", weights_only=True
            )
            mutated_payload["mutation_marker"] = "changed-during-load"

            def mutate_then_load(*args: object, **kwargs: object) -> object:
                torch.save(mutated_payload, checkpoint)
                return real_torch_load(*args, **kwargs)

            with (
                mock.patch.object(
                    success_path_eval, "_under_ssd", return_value=True
                ),
                mock.patch.object(
                    torch, "load", side_effect=mutate_then_load
                ),
                self.assertRaisesRegex(
                    RuntimeError, "changed during safe CPU load"
                ),
            ):
                _resolve_epoch_candidate_checkpoint(
                    epoch=3,
                    training_summary=summary,
                    summary_path=output_dir / "summary.json",
                    protocol=_protocol(),
                )

    def test_candidate_loader_accepts_none_for_final_epoch_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            output_dir = Path(raw_dir)
            checkpoint, summary = _write_final_checkpoint_fixture(output_dir)
            legacy_paths = []
            for epoch in (1, 2):
                legacy = (
                    output_dir
                    / "epoch_checkpoints"
                    / f"checkpoint_epoch_{epoch:02d}.pt"
                )
                legacy.parent.mkdir(parents=True, exist_ok=True)
                legacy.write_bytes(f"epoch-{epoch}".encode())
                legacy_paths.append(legacy.resolve())
            update = summary["update"]
            assert isinstance(update, dict)
            update["epoch_metrics"] = [
                {
                    "epoch": epoch,
                    "calibration": {"loss": float(4 - epoch)},
                    "checkpoint": (
                        str(legacy_paths[epoch - 1]) if epoch < 3 else None
                    ),
                }
                for epoch in (1, 2, 3)
            ]
            summary_path = output_dir / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            protocol = {
                **_protocol(),
                "training_summary": str(summary_path),
            }

            with mock.patch.object(
                success_path_eval, "_under_ssd", return_value=True
            ):
                candidates, summary_sha256 = _load_candidates(protocol)

            self.assertEqual(
                [candidate.checkpoint for candidate in candidates],
                [*legacy_paths, checkpoint.resolve()],
            )
            self.assertEqual(summary_sha256, _sha256(summary_path))


class SuccessPathEpisodeCheckpointIdentityTest(unittest.TestCase):
    def test_adapted_episode_hash_chain_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            adapter = root / "adapter.pt"
            adapter.write_bytes(b"adapter-v1")
            expected_sha256 = _sha256(adapter)
            output = root / "adapted" / "episode.json"
            student = root / "student"
            unit = EvalUnit(
                phase="screening",
                noise_base_seed=31001,
                seed=7,
                prompt="handover the microphone",
            )
            _write_unit_receipt(
                output,
                unit=unit,
                student=student,
                adapter=adapter,
                adapter_state_sha256=expected_sha256,
            )

            receipt = _run_policy(
                protocol={"student": str(student)},
                protocol_sha256="protocol-sha",
                gpu=0,
                phase="screening",
                unit=unit,
                condition="adapted",
                adapter=adapter,
                expected_adapter_state_sha256=expected_sha256,
                output=output,
                record=False,
            )

            self.assertIsNotNone(receipt)

    def test_same_path_mutated_checkpoint_rejects_old_episode(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            adapter = root / "adapter.pt"
            adapter.write_bytes(b"adapter-v1")
            old_sha256 = _sha256(adapter)
            output = root / "adapted" / "episode.json"
            student = root / "student"
            unit = EvalUnit(
                phase="screening",
                noise_base_seed=31001,
                seed=7,
                prompt="handover the microphone",
            )
            _write_unit_receipt(
                output,
                unit=unit,
                student=student,
                adapter=adapter,
                adapter_state_sha256=old_sha256,
            )
            adapter.write_bytes(b"adapter-v2")

            receipt = _unit_complete(
                output=output,
                phase="screening",
                condition="adapted",
                unit=unit,
                protocol_sha256="protocol-sha",
                student=student,
                adapter=adapter,
                expected_adapter_state_sha256=_sha256(adapter),
                record=False,
            )

            self.assertIsNone(receipt)

    def test_released_episode_requires_explicit_null_adapter_hash(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            output = root / "released" / "episode.json"
            student = root / "student"
            unit = EvalUnit(
                phase="screening",
                noise_base_seed=31001,
                seed=7,
                prompt="handover the microphone",
            )
            _write_unit_receipt(
                output,
                unit=unit,
                student=student,
                adapter=None,
                adapter_state_sha256=None,
            )
            kwargs = {
                "output": output,
                "phase": "screening",
                "condition": "released",
                "unit": unit,
                "protocol_sha256": "protocol-sha",
                "student": student,
                "adapter": None,
                "expected_adapter_state_sha256": None,
                "record": False,
            }

            self.assertIsNotNone(_unit_complete(**kwargs))
            output.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "adapter_state_sha256": "stale-adapted-hash",
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNone(_unit_complete(**kwargs))

    def test_selection_hash_chain_accepts_exact_candidate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            checkpoint = Path(raw_dir) / "adapter.pt"
            checkpoint.write_bytes(b"adapter-v1")
            checkpoint_sha256 = _sha256(checkpoint)
            candidate = Candidate(
                epoch=3,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                fixed_calibration_total_loss=0.25,
            )
            row = {
                "checkpoint_sha256": checkpoint_sha256,
                "released_checkpoint_sha256": None,
                "opd_success": True,
                "released_success": False,
                "stage_improvement": True,
                "stage_regression": False,
                "paired_max_ordinal_delta": 1,
                "rescue": True,
                "regression": False,
                "released_placement_valid": False,
                "adapted_placement_valid": True,
                "paired_min_xy_linf_error_delta": 0.1,
            }

            aggregate = _candidate_aggregate(
                candidate, [row], task="put_object_cabinet"
            )

            self.assertEqual(
                aggregate["checkpoint_sha256"], checkpoint_sha256
            )

    def test_heldout_commit_records_the_frozen_checkpoint_sha(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            checkpoint = root / "adapter.pt"
            checkpoint.write_bytes(b"adapter-v1")
            checkpoint_sha256 = _sha256(checkpoint)
            protocol_sha256 = "protocol-sha"
            protocol = {
                "output_root": str(root),
                "task": "put_object_cabinet",
                "task_config": "demo_clean",
            }
            selection = {
                "schema": "waopd_put_object_cabinet_success_path_selection_v1",
                "status": "PASS",
                "protocol_sha256": protocol_sha256,
                "selection_rule": list(success_path_eval.EXPECTED_SELECTION_RULE),
                "selected_epoch": 3,
                "selected_checkpoint": str(checkpoint),
                "selected_checkpoint_sha256": checkpoint_sha256,
            }
            selection_path = root / "screen" / "selection.json"
            selection_path.parent.mkdir(parents=True)
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            unit = EvalUnit(
                phase="heldout",
                noise_base_seed=41001,
                seed=17,
                prompt="put the object in the cabinet",
            )
            row = {
                "noise_base_seed": unit.noise_base_seed,
                "seed": unit.seed,
                "checkpoint_sha256": checkpoint_sha256,
                "released_checkpoint_sha256": None,
                "released_success": False,
                "opd_success": True,
                "rescue": True,
                "regression": False,
                "paired_max_ordinal_delta": 1,
                "stage_improvement": True,
                "stage_regression": False,
                "released_placement_valid": False,
                "adapted_placement_valid": True,
                "paired_min_xy_linf_error_delta": 0.1,
                "released_composite_mp4": None,
                "adapted_composite_mp4": None,
            }

            def heldout_queue(**kwargs: object) -> list[dict[str, object]]:
                self.assertEqual(
                    kwargs["expected_checkpoint_sha256"], checkpoint_sha256
                )
                return [row]

            with (
                mock.patch.object(
                    success_path_eval,
                    "_validate_protocol",
                    return_value=(protocol, protocol_sha256),
                ),
                mock.patch.object(
                    success_path_eval,
                    "_require_ssd",
                    side_effect=lambda path, _label: Path(path).resolve(),
                ),
                mock.patch.object(
                    success_path_eval, "_phase_units", return_value=[unit]
                ),
                mock.patch.object(
                    success_path_eval, "_parse_gpus", return_value=[0]
                ),
                mock.patch.object(
                    success_path_eval, "_record_keys", return_value=set()
                ),
                mock.patch.object(
                    success_path_eval,
                    "_run_heldout_queue",
                    side_effect=heldout_queue,
                ),
            ):
                result = _run_heldout(
                    SimpleNamespace(
                        protocol=root / "protocol.json",
                        gpus="0",
                        dry_run=False,
                    )
                )

            self.assertEqual(result, 0)
            summary = json.loads(
                (root / "heldout" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                summary["selected_checkpoint_sha256"], checkpoint_sha256
            )

    def test_heldout_refuses_commit_if_checkpoint_changes_after_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            checkpoint = root / "adapter.pt"
            checkpoint.write_bytes(b"adapter-v1")
            checkpoint_sha256 = _sha256(checkpoint)
            protocol_sha256 = "protocol-sha"
            protocol = {
                "output_root": str(root),
                "task": "put_object_cabinet",
                "task_config": "demo_clean",
            }
            selection_path = root / "screen" / "selection.json"
            selection_path.parent.mkdir(parents=True)
            selection_path.write_text(
                json.dumps(
                    {
                        "schema": (
                            "waopd_put_object_cabinet_success_path_selection_v1"
                        ),
                        "status": "PASS",
                        "protocol_sha256": protocol_sha256,
                        "selection_rule": list(
                            success_path_eval.EXPECTED_SELECTION_RULE
                        ),
                        "selected_epoch": 3,
                        "selected_checkpoint": str(checkpoint),
                        "selected_checkpoint_sha256": checkpoint_sha256,
                    }
                ),
                encoding="utf-8",
            )
            unit = EvalUnit(
                phase="heldout",
                noise_base_seed=41001,
                seed=17,
                prompt="put the object in the cabinet",
            )
            row = {
                "noise_base_seed": unit.noise_base_seed,
                "seed": unit.seed,
                "checkpoint_sha256": checkpoint_sha256,
                "released_checkpoint_sha256": None,
                "released_success": False,
                "opd_success": True,
                "rescue": True,
                "regression": False,
                "paired_max_ordinal_delta": 1,
                "stage_improvement": True,
                "stage_regression": False,
                "released_placement_valid": False,
                "adapted_placement_valid": True,
                "paired_min_xy_linf_error_delta": 0.1,
                "released_composite_mp4": None,
                "adapted_composite_mp4": None,
            }

            def mutate_after_episodes(**_kwargs: object) -> list[dict[str, object]]:
                checkpoint.write_bytes(b"adapter-v2")
                return [row]

            with (
                mock.patch.object(
                    success_path_eval,
                    "_validate_protocol",
                    return_value=(protocol, protocol_sha256),
                ),
                mock.patch.object(
                    success_path_eval,
                    "_require_ssd",
                    side_effect=lambda path, _label: Path(path).resolve(),
                ),
                mock.patch.object(
                    success_path_eval, "_phase_units", return_value=[unit]
                ),
                mock.patch.object(
                    success_path_eval, "_parse_gpus", return_value=[0]
                ),
                mock.patch.object(
                    success_path_eval, "_record_keys", return_value=set()
                ),
                mock.patch.object(
                    success_path_eval,
                    "_run_heldout_queue",
                    side_effect=mutate_after_episodes,
                ),
                self.assertRaisesRegex(
                    RuntimeError, "changed before heldout commit"
                ),
            ):
                _run_heldout(
                    SimpleNamespace(
                        protocol=root / "protocol.json",
                        gpus="0",
                        dry_run=False,
                    )
                )

            self.assertFalse((root / "heldout" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
