from __future__ import annotations

from pathlib import Path

import pytest
import torch

import experiments.train_iterative_on_policy_flow_opd as iterative_flow
from experiments.train_iterative_on_policy_flow_opd import (
    CHECKPOINT_SCHEMA,
    DUAL_CHECKPOINT_SCHEMA,
    LOSS_REDUCTION_MEAN_ALL,
    LOSS_REDUCTION_MEAN_TASKS,
    LOSS_REDUCTION_MEAN_TRAJECTORIES,
    OBJECTIVE_ENDPOINT,
    OBJECTIVE_MULTI_SIGMA_X0,
    ROLLOUT_BUNDLE_SCHEMA,
    _adapter_state_policy_version,
    _checkpoint_objective,
    _functional_candidate_gate,
    _functional_closure_stats,
    _normalize_config,
    _run_branch_update,
    _sample_optimization_weights,
    _stratified_calibration_indices,
    _trust_region_sgd_learning_rate,
    _validate_label,
    _validate_checkpoint_task_contract,
    _validate_dual_optimizer_checkpoint_metadata,
    _validate_rollout_bundle,
)
from experiments.waopd_native_closed_loop_runner import NativeClosedLoopError
from experiments.waopd_v0_video_opd import scheduler_sigma_timestep


def _coherent_tt_label() -> dict[str, object]:
    student_plan = torch.tensor([1.0, 2.0])
    teacher_plan = torch.tensor([3.0, 4.0])
    action_noise = torch.tensor([5.0, 6.0])
    action_timestep = torch.tensor([1000.0])
    return {
        "student_z_s": student_plan,
        "teacher_z_t": teacher_plan,
        "teacher_action_consumed_plan_hash": iterative_flow.tensor_hash(
            teacher_plan
        ),
        "teacher_controls_environment": False,
        "environment_execution": "SS",
        "action_target_condition": "teacher_on_teacher_z_t",
        "student_action_input_noise": action_noise,
        "teacher_action_input_noise": action_noise.clone(),
        "student_action_timestep": action_timestep,
        "teacher_action_timestep": action_timestep.clone(),
    }


def test_coherent_tt_label_binds_teacher_action_to_teacher_plan() -> None:
    _validate_label(_coherent_tt_label())


def test_coherent_tt_label_rejects_teacher_action_bound_to_student_plan() -> None:
    label = _coherent_tt_label()
    label["teacher_action_consumed_plan_hash"] = iterative_flow.tensor_hash(
        label["student_z_s"]
    )

    with pytest.raises(
        NativeClosedLoopError,
        match="saved Teacher action plan hash is not Teacher z_T",
    ):
        _validate_label(label)


def _trajectory(
    collection_id: str, count: int, *, task: str = "open_microwave"
) -> dict[str, object]:
    return {
        "task": task,
        "labels": [
            {"collection_id": collection_id, "macro_id": macro_id}
            for macro_id in range(count)
        ]
    }


def test_per_trajectory_reduction_equalizes_terminal_lengths() -> None:
    trajectories = [_trajectory("short", 2), _trajectory("long", 4)]
    weights = _sample_optimization_weights(
        trajectories, loss_reduction=LOSS_REDUCTION_MEAN_TRAJECTORIES
    )

    assert weights == pytest.approx([0.25, 0.25, 0.125, 0.125, 0.125, 0.125])
    assert sum(weights[:2]) == pytest.approx(0.5)
    assert sum(weights[2:]) == pytest.approx(0.5)


def test_mean_all_reduction_preserves_legacy_weighting() -> None:
    trajectories = [_trajectory("short", 2), _trajectory("long", 4)]
    weights = _sample_optimization_weights(
        trajectories, loss_reduction=LOSS_REDUCTION_MEAN_ALL
    )

    assert weights == pytest.approx([1.0 / 6.0] * 6)


def test_per_trajectory_reduction_rejects_empty_rollout() -> None:
    with pytest.raises(NativeClosedLoopError, match="trajectory has no labels"):
        _sample_optimization_weights(
            [_trajectory("empty", 0)],
            loss_reduction=LOSS_REDUCTION_MEAN_TRAJECTORIES,
        )


def test_per_task_reduction_equalizes_tasks_trajectories_and_terminal_lengths() -> None:
    trajectories = [
        _trajectory("microwave-short", 2, task="open_microwave"),
        _trajectory("microwave-long", 4, task="open_microwave"),
        _trajectory("cabinet", 3, task="put_object_cabinet"),
    ]
    weights = _sample_optimization_weights(
        trajectories, loss_reduction=LOSS_REDUCTION_MEAN_TASKS
    )

    assert weights == pytest.approx(
        [
            0.125,
            0.125,
            0.0625,
            0.0625,
            0.0625,
            0.0625,
            1.0 / 6.0,
            1.0 / 6.0,
            1.0 / 6.0,
        ]
    )
    assert sum(weights[:2]) == pytest.approx(0.25)
    assert sum(weights[2:6]) == pytest.approx(0.25)
    assert sum(weights[6:]) == pytest.approx(0.5)


def test_multi_task_config_preserves_per_seed_prompts_and_native_horizons() -> None:
    config = _normalize_config(
        {
            "tasks": [
                {
                    "task": "open_microwave",
                    "rollouts": [
                        {"seed": 10010, "prompt": "open seed 10010"},
                        {"seed": 10011, "prompt": "open seed 10011"},
                    ],
                },
                {
                    "task": "put_object_cabinet",
                    "rollouts": [
                        {"seed": 40023, "prompt": "cabinet seed 40023"}
                    ],
                },
            ],
            "rounds": 1,
            "adapter_seed": 20260817,
        }
    )

    assert config["task"] == "multi_task"
    assert config["rollouts_per_round"] == 3
    assert config["loss_reduction"] == LOSS_REDUCTION_MEAN_TASKS
    assert config["adapter_seed"] == 20260817
    assert [entry["chunks"] for entry in config["task_entries"]] == [48, 23]
    assert config["task_entries"][0]["rollouts"] == [
        {"seed": 10010, "prompt": "open seed 10010", "rollout_id": 0},
        {"seed": 10011, "prompt": "open seed 10011", "rollout_id": 1},
    ]


def test_dual_action_config_is_mode_isolated_fp32_sgd_contract() -> None:
    config = _normalize_config(
        {
            "task": "handover_mic",
            "rollouts": [{"seed": 10000, "prompt": "handover"}],
            "rounds": 1,
            "adapter_kind": "dual_lora",
            "trainable_bank": "action",
        }
    )

    assert config["optimizer_kind"] == "trust_region_sgd"
    assert config["learning_rate"] == pytest.approx(1.0)
    assert config["max_update_norm"] == pytest.approx(0.003)
    assert config["video_weight"] == 0.0
    assert config["action_weight"] == 1.0
    assert config["action_velocity_weight"] == 0.0


def test_dual_video_config_rejects_cross_mode_objective() -> None:
    with pytest.raises(ValueError, match="zero action weights"):
        _normalize_config(
            {
                "task": "handover_mic",
                "rollouts": [{"seed": 10000, "prompt": "handover"}],
                "rounds": 1,
                "adapter_kind": "dual_lora",
                "trainable_bank": "video",
                "action_weight": 1.0,
            }
        )


def test_move_stapler_multi_sigma_config_freezes_formal_scope() -> None:
    config = _normalize_config(
        {
            "task": "move_stapler_pad",
            "rollouts": [
                {"seed": 10000, "prompt": "move the stapler to the pad"},
                {"seed": 10001, "prompt": "place the stapler on the pad"},
            ],
            "rounds": 1,
            "objective": OBJECTIVE_MULTI_SIGMA_X0,
            "adapter_kind": "joint_lora",
            "trainable_bank": "both",
            "optimizer_kind": "functional_sgd",
            "adapter_rank": 8,
            "lora_alpha": 8.0,
            "lora_dropout": 0.0,
            "lora_block_indices": list(range(30)),
            "sigma_values": [1.0, 0.5, 0.25],
            "line_search_sigma_values": [1.0],
            "loss_reduction": LOSS_REDUCTION_MEAN_TRAJECTORIES,
            "retention_weight": 0.0,
        }
    )

    assert config["task_config"] == "demo_clean"
    assert config["chunks"] == 13
    assert config["objective"] == OBJECTIVE_MULTI_SIGMA_X0
    assert config["optimizer_kind"] == "functional_sgd"
    assert config["adapter_rank"] == 8
    assert config["lora_alpha"] == pytest.approx(8.0)
    assert config["lora_dropout"] == 0.0
    assert config["lora_block_indices"] == list(range(30))
    assert config["sigma_values"] == [1.0, 0.5, 0.25]
    assert config["action_velocity_weight"] == 0.0
    assert config["retention_weight"] == 0.0


def test_collect_accepts_joint_lora_for_exact_trajectory_replay() -> None:
    config = _normalize_config(
        {
            "run_mode": "collect",
            "task": "handover_mic",
            "rollouts": [{"seed": 10000, "prompt": "handover"}],
            "rounds": 1,
            "objective": OBJECTIVE_MULTI_SIGMA_X0,
            "adapter_kind": "joint_lora",
            "trainable_bank": "both",
            "optimizer_kind": "functional_sgd",
            "lora_block_indices": list(range(30)),
            "rollout_bundle": None,
        }
    )

    assert config["run_mode"] == "collect"
    assert config["adapter_kind"] == "joint_lora"
    assert config["trainable_bank"] == "both"
    assert config["rollout_bundle"] is None


def test_joint_lora_collect_rejects_dual_branch_bundle_output() -> None:
    with pytest.raises(ValueError, match="not a dual_lora branch rollout_bundle"):
        _normalize_config(
            {
                "run_mode": "collect",
                "task": "handover_mic",
                "rollouts": [{"seed": 10000, "prompt": "handover"}],
                "rounds": 1,
                "objective": OBJECTIVE_MULTI_SIGMA_X0,
                "adapter_kind": "joint_lora",
                "trainable_bank": "both",
                "optimizer_kind": "functional_sgd",
                "lora_block_indices": list(range(30)),
                "rollout_bundle": "wrong.pt",
            }
        )


def test_trajectory_update_normalizes_joint_artifact_replay_contract() -> None:
    config = _normalize_config(
        {
            "run_mode": "trajectory_update",
            "task": "move_stapler_pad",
            "rollouts": [
                {"seed": 10000, "prompt": "move the stapler to the pad"},
                {"seed": 10001, "prompt": "place the stapler on the pad"},
            ],
            "trajectory_artifacts": ["rollout_00.pt", "rollout_01.pt"],
            "objective": OBJECTIVE_MULTI_SIGMA_X0,
            "adapter_kind": "joint_lora",
            "trainable_bank": "both",
            "optimizer_kind": "functional_sgd",
            "lora_block_indices": list(range(30)),
        }
    )

    assert config["run_mode"] == "trajectory_update"
    assert config["rounds"] == 1
    assert config["adapter_kind"] == "joint_lora"
    assert config["trajectory_artifacts"] == [
        "rollout_00.pt",
        "rollout_01.pt",
    ]


def test_trajectory_loader_rejects_label_mask_that_differs_from_physical_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    student = tmp_path / "student"
    artifact_path = tmp_path / "rollout.pt"
    task_entries = [
        {
            "task": "move_stapler_pad",
            "task_config": "demo_clean",
            "chunks": 13,
            "rollouts": [{"seed": 10000, "prompt": "move stapler"}],
            "gate_json": None,
        }
    ]
    label_mask = torch.tensor([True, True], dtype=torch.bool)
    physical_mask = torch.tensor([True, False], dtype=torch.bool)
    torch.save(
        {
            "schema": iterative_flow.TRAJECTORY_SCHEMA,
            "task": "move_stapler_pad",
            "task_config": "demo_clean",
            "seed": 10000,
            "prompt": "move stapler",
            "collection_id": "move_stapler_pad_round00_rollout00",
            "round_id": 0,
            "rollout_id": 0,
            "behavior_policy_version": "policy",
            "history_owner": "current_student_on_policy",
            "environment_execution": "SS",
            "teacher_controls_environment": False,
            "fresh_passes_allowed": 1,
            "success_post_label_count": 0,
            "history": [{}],
            "labels": [
                {
                    "collection_id": "move_stapler_pad_round00_rollout00",
                    "round_id": 0,
                    "rollout_id": 0,
                    "behavior_policy_version": "policy",
                    "history_prefix_length": 0,
                    "macro_id": 0,
                    "frame_st_id": 0,
                    "start_frame": 0,
                    "action_steps": 2,
                    "executed_action_mask": label_mask,
                    "terminal_reached": False,
                    "terminal_action_position": None,
                    "horizon_reached": False,
                }
            ],
            "baseline_episode": {
                "task": "move_stapler_pad",
                "task_config": "demo_clean",
                "seed": 10000,
                "prompt": "move stapler",
                "arm": "SS",
                "student_checkpoint": str(student),
                "chunks_requested": 13,
                "max_control_steps": 400,
                "shared_noise_across_arms": True,
                "chunks": [
                    {
                        "chunk_id": 0,
                        "frame_st_id": 0,
                        "start_frame": 0,
                        "action_steps": 2,
                        "executed_action_mask": physical_mask,
                        "terminal_reached": False,
                        "terminal_action_position": None,
                        "horizon_reached": False,
                    }
                ],
            },
        },
        artifact_path,
    )
    monkeypatch.setattr(iterative_flow, "_validate_label", lambda _label: None)

    with pytest.raises(NativeClosedLoopError, match="physical chunk"):
        iterative_flow._load_trajectory_artifacts(
            [artifact_path],
            expected_task_entries=task_entries,
            expected_student=student,
        )


def test_functional_sgd_cannot_silently_run_legacy_endpoint_objective() -> None:
    with pytest.raises(ValueError, match="only valid for multi_sigma_x0"):
        _normalize_config(
            {
                "task": "move_stapler_pad",
                "rollouts": [{"seed": 10000, "prompt": "move stapler"}],
                "rounds": 1,
                "objective": OBJECTIVE_ENDPOINT,
                "optimizer_kind": "functional_sgd",
            }
        )


def test_checkpoint_objective_is_explicit_with_legacy_endpoint_fallback() -> None:
    assert _checkpoint_objective({}) == OBJECTIVE_ENDPOINT
    assert (
        _checkpoint_objective({"loss": {"objective": OBJECTIVE_MULTI_SIGMA_X0}})
        == OBJECTIVE_MULTI_SIGMA_X0
    )
    assert (
        _checkpoint_objective({"objective": OBJECTIVE_MULTI_SIGMA_X0})
        == OBJECTIVE_MULTI_SIGMA_X0
    )
    with pytest.raises(ValueError, match="unknown objective"):
        _checkpoint_objective({"objective": "not_a_real_objective"})


def test_stratified_calibration_keeps_early_middle_late_train_coverage() -> None:
    assert _stratified_calibration_indices(13, 3) == (1, 6, 11)
    assert _stratified_calibration_indices(2, 3) == (1,)
    with pytest.raises(NativeClosedLoopError, match="at least two labels"):
        _stratified_calibration_indices(1, 1)


def test_functional_closure_ratio_measures_output_motion_toward_target() -> None:
    before = {
        "outputs": [
            {
                "requested_sigma": 0.5,
                "video_sigma": 0.5006,
                "action_sigma": 0.5,
                "video_prediction": torch.zeros(4),
                "video_target": torch.ones(4),
            }
        ]
    }
    after = {
        "outputs": [
            {
                "requested_sigma": 0.5,
                "video_sigma": 0.5006,
                "action_sigma": 0.5,
                "video_prediction": torch.full((4,), 0.1),
                "video_target": torch.ones(4),
            }
        ]
    }

    stats = _functional_closure_stats(before, after, modality="video")

    assert stats["median"] == pytest.approx(0.1)
    assert stats["p95"] == pytest.approx(0.1)


def _closure_fixture(
    *,
    video_median: float = 0.10,
    video_p95: float = 0.20,
    action_median: float = 0.28,
    action_p95: float = 0.40,
) -> dict[str, dict[str, float]]:
    return {
        "video": {
            "median": video_median,
            "p95": video_p95,
            "min": video_median,
            "max": video_p95,
        },
        "action": {
            "median": action_median,
            "p95": action_p95,
            "min": action_median,
            "max": action_p95,
        },
    }


def test_functional_candidate_gate_accepts_resolved_bf16_step() -> None:
    gate = _functional_candidate_gate(
        before={"loss": 0.10, "video_loss": 0.07, "action_loss": 0.03},
        after={"loss": 0.09, "video_loss": 0.065, "action_loss": 0.025},
        closure=_closure_fixture(),
        p95_max=0.50,
    )

    assert gate == {"accepted": True, "rejection_reasons": []}


@pytest.mark.parametrize(
    ("after", "closure", "reason"),
    [
        (
            {"loss": 0.09, "video_loss": 0.059, "action_loss": 0.031},
            _closure_fixture(),
            "action_loss_regressed",
        ),
        (
            {"loss": 0.09, "video_loss": 0.065, "action_loss": 0.025},
            _closure_fixture(action_median=0.0, action_p95=0.0),
            "action_functionally_zero",
        ),
        (
            {"loss": 0.09, "video_loss": 0.065, "action_loss": 0.025},
            _closure_fixture(action_p95=0.60),
            "action_p95_overshoot",
        ),
    ],
)
def test_functional_candidate_gate_rejects_unsafe_step(
    after: dict[str, float],
    closure: dict[str, dict[str, float]],
    reason: str,
) -> None:
    gate = _functional_candidate_gate(
        before={"loss": 0.10, "video_loss": 0.06, "action_loss": 0.03},
        after=after,
        closure=closure,
        p95_max=0.50,
    )

    assert gate["accepted"] is False
    assert reason in gate["rejection_reasons"]


def test_multi_sigma_requires_the_validated_all_shared_block_scope() -> None:
    with pytest.raises(ValueError, match="all shared blocks 0-29"):
        _normalize_config(
            {
                "task": "move_stapler_pad",
                "rollouts": [{"seed": 10000, "prompt": "move stapler"}],
                "rounds": 1,
                "objective": OBJECTIVE_MULTI_SIGMA_X0,
                "lora_block_indices": [26, 27, 28, 29],
            }
        )


def test_scheduler_sigma_timestep_uses_native_grid() -> None:
    class FakeScheduler:
        config = type("Config", (), {"num_train_timesteps": 4})()

        def set_timesteps(self, steps: int) -> None:
            assert steps == 4
            self.timesteps = torch.tensor([4.0, 3.0, 2.0, 1.0])
            self.sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])

    timestep, sigma = scheduler_sigma_timestep(FakeScheduler(), 0.5)

    assert float(timestep.item()) == pytest.approx(2.0)
    assert sigma == pytest.approx(0.5)


def test_scheduler_sigma_timestep_snaps_to_shifted_native_grid() -> None:
    class FakeShiftedScheduler:
        config = type("Config", (), {"num_train_timesteps": 1000})()

        def set_timesteps(self, steps: int) -> None:
            assert steps == 1000
            self.timesteps = torch.tensor([1000.0, 250.0, 1.0])
            self.sigmas = torch.tensor(
                [1.0, 0.25159743428230286, 0.004980079829692841, 0.0]
            )

    timestep, sigma = scheduler_sigma_timestep(FakeShiftedScheduler(), 0.25)

    assert float(timestep.item()) == pytest.approx(250.0)
    assert sigma == pytest.approx(0.25159743428230286)
    with pytest.raises(ValueError, match="outside the native scheduler range"):
        scheduler_sigma_timestep(FakeShiftedScheduler(), 0.001)


def test_trust_region_sgd_scales_only_when_proposed_step_exceeds_bound() -> None:
    assert _trust_region_sgd_learning_rate(
        configured_learning_rate=1.0,
        gradient_norm=0.001,
        max_update_norm=0.003,
    ) == pytest.approx(1.0)
    assert _trust_region_sgd_learning_rate(
        configured_learning_rate=1.0,
        gradient_norm=0.03,
        max_update_norm=0.003,
    ) == pytest.approx(0.1)


def test_dual_checkpoint_always_requires_exact_task_contract() -> None:
    with pytest.raises(ValueError, match="no task contract"):
        _validate_checkpoint_task_contract(
            {},
            expected_schema=DUAL_CHECKPOINT_SCHEMA,
            expected_hash="expected",
            multi_task=False,
        )
    with pytest.raises(ValueError, match="contract mismatch"):
        _validate_checkpoint_task_contract(
            {"task_contract_hash": "other"},
            expected_schema=DUAL_CHECKPOINT_SCHEMA,
            expected_hash="expected",
            multi_task=False,
        )


def test_legacy_single_task_checkpoint_may_precede_task_contract_field() -> None:
    _validate_checkpoint_task_contract(
        {},
        expected_schema=CHECKPOINT_SCHEMA,
        expected_hash="expected",
        multi_task=False,
    )


def test_dual_optimizer_metadata_checks_raw_state_before_load_cast() -> None:
    checkpoint = {
        "optimizer_parameter_dtypes": ["torch.float32"],
        "optimizer_state_dtypes": ["torch.float32"],
        "optimizer_state_dict": {
            "state": {0: {"momentum_buffer": torch.ones(2, dtype=torch.float32)}},
            "param_groups": [],
        },
    }
    _validate_dual_optimizer_checkpoint_metadata(checkpoint)

    bad = dict(checkpoint)
    bad["optimizer_state_dict"] = {
        "state": {0: {"momentum_buffer": torch.ones(2, dtype=torch.bfloat16)}},
        "param_groups": [],
    }
    bad["optimizer_state_dtypes"] = ["torch.bfloat16"]
    with pytest.raises(ValueError, match="state is not FP32"):
        _validate_dual_optimizer_checkpoint_metadata(bad)


def test_dual_optimizer_metadata_rejects_false_dtype_manifest() -> None:
    with pytest.raises(ValueError, match="metadata mismatch"):
        _validate_dual_optimizer_checkpoint_metadata(
            {
                "optimizer_parameter_dtypes": ["torch.float32"],
                "optimizer_state_dtypes": [],
                "optimizer_state_dict": {
                    "state": {
                        0: {"momentum_buffer": torch.ones(2, dtype=torch.float32)}
                    },
                    "param_groups": [],
                },
            }
        )


def _minimal_rollout_bundle(
    *, student: Path, task_entries: list[dict[str, object]]
) -> dict[str, object]:
    state = {
        "block.video_lora_A": torch.tensor([1.0], dtype=torch.float32),
        "block.action_lora_A": torch.tensor([2.0], dtype=torch.float32),
    }
    policy_version = _adapter_state_policy_version(state)
    task_hash = iterative_flow._stable_hash(task_entries)
    collection_contract = iterative_flow._rollout_collection_contract(task_entries)
    return {
        "schema": ROLLOUT_BUNDLE_SCHEMA,
        "adapter_kind": "dual_lora",
        "adapter_contract": {
            "rank": 1,
            "alpha": 1.0,
            "dropout": 0.0,
            "block_indices": [0],
        },
        "adapter_state_dict": state,
        "behavior_policy_version": policy_version,
        "base_student_checkpoint": str(student.resolve()),
        "adapter_seed": 20260817,
        "task_contract": {
            "task": "handover_mic",
            "task_config": "demo_clean",
            "task_entries": task_entries,
            "task_contract_hash": task_hash,
            "collection_contract": collection_contract,
        },
        "terminal_label_contract": dict(iterative_flow.TERMINAL_LABEL_CONTRACT),
        "round_id": 0,
        "global_optimizer_step": 0,
        "trajectories": [
            {
                "task": "handover_mic",
                "round_id": 0,
                "behavior_policy_version": policy_version,
                "success_post_label_count": 0,
                "labels": [
                    {
                        "round_id": 0,
                        "behavior_policy_version": policy_version,
                        "executed_action_mask": torch.ones(1, dtype=torch.bool),
                        "terminal_reached": False,
                        "terminal_action_position": None,
                        "horizon_reached": False,
                    }
                ],
            }
        ],
        "outcomes": [
            {
                "round_id": 0,
                "policy_version": policy_version,
            }
        ],
    }


def test_rollout_bundle_policy_hash_is_bound_to_exact_dual_state(
    tmp_path: Path,
) -> None:
    student = tmp_path / "student"
    task_entries = [
        {
            "task": "handover_mic",
            "task_config": "demo_clean",
            "chunks": 20,
            "rollouts": [{"seed": 10000, "prompt": "handover"}],
            "gate_json": None,
        }
    ]
    bundle = _minimal_rollout_bundle(student=student, task_entries=task_entries)
    task_hash = iterative_flow._stable_hash(task_entries)
    collection = bundle["task_contract"]["collection_contract"][0]
    assert collection["chunks"] == 20
    assert collection["max_control_steps"] == 600

    _validate_rollout_bundle(
        bundle,
        expected_task_entries=task_entries,
        expected_task_contract_hash=task_hash,
        expected_student=student,
    )
    changed = dict(bundle)
    changed["adapter_state_dict"] = {
        "block.video_lora_A": torch.tensor([9.0], dtype=torch.float32),
        "block.action_lora_A": torch.tensor([2.0], dtype=torch.float32),
    }
    with pytest.raises(NativeClosedLoopError, match="policy hash differs"):
        _validate_rollout_bundle(
            changed,
            expected_task_entries=task_entries,
            expected_task_contract_hash=task_hash,
            expected_student=student,
        )


def test_branch_update_restores_bundle_teacher_free_with_fresh_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    student = tmp_path / "student"
    config = _normalize_config(
        {
            "run_mode": "branch_update",
            "task": "handover_mic",
            "rollouts": [{"seed": 10000, "prompt": "handover"}],
            "rollout_bundle": str(tmp_path / "rollout_bundle.pt"),
            "trainable_bank": "action",
        }
    )
    task_entries = config["task_entries"]
    bundle = _minimal_rollout_bundle(
        student=student, task_entries=task_entries
    )
    bundle_path = Path(config["rollout_bundle"])
    torch.save(bundle, bundle_path)
    constructor: dict[str, object] = {}
    loaded_payload: dict[str, object] = {}

    class FakeTeacherFreeRuntime:
        adapter_kind = "dual_lora"

        def __init__(self, **kwargs: object) -> None:
            constructor.update(kwargs)
            self.server = type("FakeServer", (), {})()
            self.server.transformer = self
            self.adapter_info = object()
            self._contract: dict[str, object] = {}
            self._parameters: dict[str, torch.nn.Parameter] = {}
            self.trainable: list[tuple[str, torch.nn.Parameter]] = []
            self.adapter_parameter_names: list[str] = []
            self.closed = False

        def select_adapter_trainable_bank(
            self, bank: str
        ) -> list[tuple[str, torch.nn.Parameter]]:
            suffix = f".{bank}_lora_A"
            self.trainable = [
                (name, parameter)
                for name, parameter in self._parameters.items()
                if name.endswith(suffix)
            ]
            self.adapter_parameter_names = [name for name, _ in self.trainable]
            return list(self.trainable)

        def adapter_contract(self) -> dict[str, object]:
            return dict(self._contract)

        def adapter_state(self) -> dict[str, torch.Tensor]:
            return {
                name: parameter.detach().clone()
                for name, parameter in self._parameters.items()
            }

        def close(self) -> None:
            self.closed = True

    def fake_load_dual_checkpoint(
        transformer: FakeTeacherFreeRuntime,
        _info: object,
        payload: dict[str, object],
    ) -> None:
        loaded_payload.update(payload)
        transformer._contract = dict(payload["adapter_contract"])
        transformer._parameters = {
            name: torch.nn.Parameter(value.detach().clone())
            for name, value in payload["adapter_state_dict"].items()
        }

    def fake_update_round(**kwargs: object) -> dict[str, object]:
        optimizer = kwargs["optimizer"]
        runtime = kwargs["runtime"]
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert optimizer.state == {}
        with torch.no_grad():
            runtime.trainable[0][1].add_(0.25)
        return {"optimizer_steps_this_round": 1}

    monkeypatch.setattr(
        iterative_flow, "NativeV0VideoRuntime", FakeTeacherFreeRuntime
    )
    monkeypatch.setattr(
        iterative_flow,
        "load_dual_mode_lora_checkpoint",
        fake_load_dual_checkpoint,
    )
    monkeypatch.setattr(iterative_flow, "_update_round", fake_update_round)
    summary = _run_branch_update(
        config=config,
        student=student,
        output_dir=tmp_path / "branch",
        task_contract_hash=iterative_flow._stable_hash(task_entries),
    )

    assert constructor["teacher_transformer"] is None
    assert constructor["adapter_state"] is None
    assert loaded_payload["behavior_policy_version"] == bundle[
        "behavior_policy_version"
    ]
    assert summary["teacher_loaded"] is False
    assert summary["fresh_optimizer"] is True
    assert summary["adapter_seed"] == bundle["adapter_seed"]
    assert Path(str(summary["checkpoint"])).is_file()


def test_trajectory_update_reconstructs_teacher_free_joint_policy_and_steps_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    student = tmp_path / "student"
    artifact_paths = [tmp_path / "rollout_00.pt", tmp_path / "rollout_01.pt"]
    config = _normalize_config(
        {
            "run_mode": "trajectory_update",
            "task": "move_stapler_pad",
            "rollouts": [
                {"seed": 10000, "prompt": "move stapler"},
                {"seed": 10001, "prompt": "place stapler"},
            ],
            "trajectory_artifacts": [str(path) for path in artifact_paths],
            "objective": OBJECTIVE_MULTI_SIGMA_X0,
            "adapter_kind": "joint_lora",
            "trainable_bank": "both",
            "optimizer_kind": "functional_sgd",
            "lora_block_indices": list(range(30)),
            "pre_update_solver_closure": False,
        }
    )
    trajectories = [
        {
            "task": "move_stapler_pad",
            "seed": seed,
            "round_id": 0,
            "behavior_policy_version": "policy-before",
            "labels": [{"round_id": 0}],
        }
        for seed in (10000, 10001)
    ]
    constructor: dict[str, object] = {}
    runtime_holder: dict[str, object] = {}
    update_calls = 0

    class FakeJointRuntime:
        adapter_kind = "joint_lora"

        def __init__(self, **kwargs: object) -> None:
            constructor.update(kwargs)
            self.teacher = None
            self.parameter = torch.nn.Parameter(torch.tensor([1.0]))
            self.trainable = [("joint_lora_A", self.parameter)]
            self.adapter_parameter_names = ["joint_lora_A"]
            self.closed = False
            runtime_holder["runtime"] = self

        def adapter_contract(self) -> dict[str, object]:
            return {
                "kind": "joint_lora",
                "rank": 8,
                "alpha": 8.0,
                "dropout": 0.0,
                "block_indices": list(range(30)),
            }

        def adapter_state(self) -> dict[str, torch.Tensor]:
            return {"joint_lora_A": self.parameter.detach().clone()}

        def close(self) -> None:
            self.closed = True

    def fake_policy_version(runtime: FakeJointRuntime) -> str:
        return (
            "policy-before"
            if float(runtime.parameter.item()) == pytest.approx(1.0)
            else "policy-after"
        )

    def fake_update_round(**kwargs: object) -> dict[str, object]:
        nonlocal update_calls
        update_calls += 1
        runtime = kwargs["runtime"]
        optimizer = kwargs["optimizer"]
        assert isinstance(runtime, FakeJointRuntime)
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.state == {}
        with torch.no_grad():
            runtime.parameter.add_(0.25)
        return {"optimizer_steps_this_round": 1}

    monkeypatch.setattr(
        iterative_flow,
        "_load_trajectory_artifacts",
        lambda *args, **kwargs: trajectories,
    )
    monkeypatch.setattr(iterative_flow, "NativeV0VideoRuntime", FakeJointRuntime)
    monkeypatch.setattr(iterative_flow, "_policy_version", fake_policy_version)
    monkeypatch.setattr(iterative_flow, "_update_round", fake_update_round)
    monkeypatch.setattr(
        iterative_flow,
        "_configure_cuda_memory_limit",
        lambda _config: None,
    )

    summary = iterative_flow._run_trajectory_update(
        config=config,
        student=student,
        teacher=None,
        output_dir=tmp_path / "update",
        task_contract_hash=iterative_flow._stable_hash(config["task_entries"]),
    )

    assert constructor["teacher_transformer"] is None
    assert constructor["adapter_state"] is None
    assert update_calls == 1
    assert summary["teacher_loaded"] is False
    assert summary["fresh_optimizer"] is True
    assert summary["behavior_policy_version"] == "policy-before"
    assert summary["policy_version_after"] == "policy-after"
    assert Path(str(summary["checkpoint"])).is_file()
    assert runtime_holder["runtime"].closed is True
