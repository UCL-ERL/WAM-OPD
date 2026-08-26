from __future__ import annotations

import importlib
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def iterative_flow() -> object:
    """Import the pure contract seams without remote-only runtime dependencies."""

    target_name = "experiments.train_iterative_on_policy_flow_opd"
    if target_name in sys.modules:
        yield sys.modules[target_name]
        return

    dual_lora_stub = ModuleType("experiments.dual_mode_lora")
    dual_lora_stub.load_dual_mode_lora_checkpoint = lambda *args, **kwargs: None

    joint_teacher_stub = ModuleType("experiments.train_joint_teacher_trajectory_opd")
    joint_teacher_stub._outcome = lambda *args, **kwargs: {}
    joint_teacher_stub._setup_task_with_locked_prompt = lambda *args, **kwargs: None
    joint_teacher_stub._worker_progress = lambda *args, **kwargs: None

    video_train_stub = ModuleType("experiments.train_video_trajectory_opd")
    video_train_stub.SCHEMA = "waopd_video_trajectory_v1"
    video_train_stub.NativeStudentVideoLabelRuntime = object
    video_train_stub.action_execution_mask = lambda mask, _label: mask
    video_train_stub.build_trajectory_artifact = lambda *args, **kwargs: {}
    video_train_stub.capture_student_context = lambda *args, **kwargs: {}
    video_train_stub.materialize_context = lambda *args, **kwargs: {}
    video_train_stub.video_execution_mask = (
        lambda target, _label: torch.ones_like(target, dtype=torch.bool)
    )

    runner_stub = ModuleType("experiments.waopd_native_closed_loop_runner")
    runner_stub.ActionSolve = object
    runner_stub.LockedNoiseBank = object
    runner_stub.NativeClosedLoopError = RuntimeError
    runner_stub.run_live_episode = lambda *args, **kwargs: {}
    runner_stub.tensor_hash = lambda value: str(value.detach().cpu().tolist())

    video_runtime_stub = ModuleType("experiments.waopd_v0_video_opd")
    video_runtime_stub.NativeV0VideoRuntime = object
    video_runtime_stub.action_velocity_mse_loss = lambda *args, **kwargs: None
    video_runtime_stub.video_consistency_map = lambda *args, **kwargs: None

    stubs = {
        "experiments.dual_mode_lora": dual_lora_stub,
        "experiments.train_joint_teacher_trajectory_opd": joint_teacher_stub,
        "experiments.train_video_trajectory_opd": video_train_stub,
        "experiments.waopd_native_closed_loop_runner": runner_stub,
        "experiments.waopd_v0_video_opd": video_runtime_stub,
    }
    missing = object()
    managed_names = (*stubs, target_name)
    previous = {name: sys.modules.get(name, missing) for name in managed_names}
    sys.modules.update(stubs)
    try:
        yield importlib.import_module(target_name)
    finally:
        for name, prior in previous.items():
            if prior is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def test_native_consistency_indices_are_seeded_and_clip_k500_at_last_index(
    iterative_flow: object,
) -> None:
    frame_count = 4096
    first_generator = torch.Generator().manual_seed(20260820)
    second_generator = torch.Generator().manual_seed(20260820)

    start, end = iterative_flow.sample_native_consistency_indices(
        frame_count,
        generator=first_generator,
        num_train_timesteps=1000,
        stride=500,
    )
    repeated_start, repeated_end = iterative_flow.sample_native_consistency_indices(
        frame_count,
        generator=second_generator,
        num_train_timesteps=1000,
        stride=500,
    )

    assert start.dtype == torch.long
    assert end.dtype == torch.long
    assert start.shape == (frame_count,)
    assert end.shape == (frame_count,)
    torch.testing.assert_close(start, repeated_start)
    torch.testing.assert_close(end, repeated_end)
    assert int(start.min()) >= 0
    assert int(start.max()) <= 999
    assert bool((start < 500).any())
    assert bool((start >= 500).any())

    expected_end = (start + 500).clamp(max=999)
    torch.testing.assert_close(end, expected_end)
    assert bool((end[start >= 500] == 999).all())


def test_teacher_euler_bridge_supports_per_frame_sigma_and_zero_length_boundary(
    iterative_flow: object,
) -> None:
    start_state = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2, 1)
    teacher_velocity = torch.full_like(start_state, 2.0)
    sigma_start = torch.tensor([1.0, 0.75, 0.25])
    sigma_end = torch.tensor([0.5, 0.25, 0.0])
    original_start = start_state.clone()
    original_velocity = teacher_velocity.clone()

    bridged = iterative_flow.teacher_euler_bridge(
        start_state,
        teacher_velocity,
        sigma_start=sigma_start,
        sigma_end=sigma_end,
    )

    expected_delta = torch.tensor([-0.5, -0.5, -0.25]).reshape(1, 1, 3, 1, 1)
    torch.testing.assert_close(
        bridged,
        start_state + teacher_velocity * expected_delta,
    )
    torch.testing.assert_close(start_state, original_start)
    torch.testing.assert_close(teacher_velocity, original_velocity)

    zero_length = iterative_flow.teacher_euler_bridge(
        start_state,
        teacher_velocity,
        sigma_start=torch.tensor([0.4, 0.2, 0.0]),
        sigma_end=torch.tensor([0.4, 0.2, 0.0]),
    )
    torch.testing.assert_close(zero_length, start_state)


def _online_lora_state(a: float, b: float) -> dict[str, torch.nn.Parameter]:
    return {
        "blocks.0.attn1.to_q.lora_A": torch.nn.Parameter(
            torch.tensor([a], dtype=torch.float32)
        ),
        "blocks.0.attn1.to_q.lora_B": torch.nn.Parameter(
            torch.tensor([b], dtype=torch.float32)
        ),
    }


def test_lora_ema_updates_only_after_a_committed_optimizer_step(
    iterative_flow: object,
) -> None:
    initial_online = _online_lora_state(2.0, 4.0)
    ema = iterative_flow.LoRAEMAState.from_online(initial_online, decay=0.5)
    updated_online = _online_lora_state(6.0, 10.0)

    before = ema.target_state()
    ema.after_committed_step(updated_online, committed=False)
    after_rejected_step = ema.target_state()
    for name in before:
        torch.testing.assert_close(after_rejected_step[name], before[name])

    ema.after_committed_step(updated_online, committed=True)
    committed = ema.target_state()
    torch.testing.assert_close(
        committed["blocks.0.attn1.to_q.lora_A"], torch.tensor([4.0])
    )
    torch.testing.assert_close(
        committed["blocks.0.attn1.to_q.lora_B"], torch.tensor([7.0])
    )


def test_lora_ema_target_is_stop_gradient_and_target_swap_restores_online_state(
    iterative_flow: object,
) -> None:
    ema = iterative_flow.LoRAEMAState.from_online(
        _online_lora_state(3.0, 5.0), decay=0.995
    )
    target = ema.target_state()

    assert set(target) == {
        "blocks.0.attn1.to_q.lora_A",
        "blocks.0.attn1.to_q.lora_B",
    }
    assert all(not tensor.requires_grad for tensor in target.values())
    assert all(tensor.grad_fn is None for tensor in target.values())

    # target_state() must not expose mutable aliases of the EMA's owned state.
    target["blocks.0.attn1.to_q.lora_A"].add_(100.0)
    torch.testing.assert_close(
        ema.target_state()["blocks.0.attn1.to_q.lora_A"], torch.tensor([3.0])
    )

    live_state = _online_lora_state(11.0, 13.0)
    live_before = {name: value.detach().clone() for name, value in live_state.items()}
    live_object_ids = {name: id(value) for name, value in live_state.items()}

    class ExpectedFailure(RuntimeError):
        pass

    with pytest.raises(ExpectedFailure, match="force transactional restore"):
        with ema.use_target(live_state):
            torch.testing.assert_close(
                live_state["blocks.0.attn1.to_q.lora_A"], torch.tensor([3.0])
            )
            torch.testing.assert_close(
                live_state["blocks.0.attn1.to_q.lora_B"], torch.tensor([5.0])
            )
            assert {
                name: id(value) for name, value in live_state.items()
            } == live_object_ids
            raise ExpectedFailure("force transactional restore")

    assert {name: id(value) for name, value in live_state.items()} == live_object_ids
    for name, expected in live_before.items():
        torch.testing.assert_close(live_state[name], expected)


def test_action_forward_receives_detached_teacher_plan_and_preserves_other_arguments(
    iterative_flow: object,
) -> None:
    teacher_plan = torch.tensor([1.0, 3.0], requires_grad=True)
    teacher_action = torch.tensor([5.0, 7.0])
    context = {"context_id": "student-history-macro-0"}
    action_scale = torch.nn.Parameter(torch.tensor(2.0))
    captured: dict[str, object] = {}

    def action_forward(
        received_context: dict[str, object],
        received_plan: torch.Tensor,
        received_teacher_action: torch.Tensor,
        *,
        sigma: float,
        require_grad: bool,
    ) -> torch.Tensor:
        captured.update(
            {
                "context": received_context,
                "plan": received_plan,
                "teacher_action": received_teacher_action,
                "sigma": sigma,
                "require_grad": require_grad,
            }
        )
        return action_scale * received_plan.sum()

    output = iterative_flow.student_action_on_detached_teacher_plan(
        action_forward,
        context=context,
        teacher_plan=teacher_plan,
        teacher_action=teacher_action,
        sigma=0.5,
        require_grad=True,
    )
    output.backward()

    received_plan = captured["plan"]
    assert isinstance(received_plan, torch.Tensor)
    assert captured["context"] is context
    assert captured["teacher_action"] is teacher_action
    assert captured["sigma"] == pytest.approx(0.5)
    assert captured["require_grad"] is True
    torch.testing.assert_close(received_plan, teacher_plan.detach())
    assert received_plan.requires_grad is False
    assert teacher_plan.grad is None
    assert action_scale.grad is not None
    torch.testing.assert_close(action_scale.grad, torch.tensor(4.0))


def _coherent_config() -> dict[str, object]:
    return {
        "objective": "coherent_tt_consistency",
        "task": "handover_mic",
        "rounds": 1,
        "adapter_kind": "joint_lora",
        "adapter_rank": 8,
        "lora_block_indices": list(range(30)),
        "rollouts": [
            {"seed": 10000, "prompt": "handover", "role": "train"},
            {
                "seed": 10001,
                "prompt": "handover calibration",
                "role": "calibration",
            },
        ],
    }


def test_coherent_tt_config_freezes_first_baseline_contract(
    iterative_flow: object,
) -> None:
    config = iterative_flow._normalize_config(_coherent_config())

    assert config["coherent_tt_variant"] == "baseline"
    assert config["run_mode"] == "iterative"
    assert config["optimizer_kind"] == "adamw"
    assert config["learning_rate"] == pytest.approx(5e-6)
    assert config["max_grad_norm"] == pytest.approx(2.0)
    assert config["action_fm_weight"] == pytest.approx(0.2)
    assert config["action_velocity_weight"] == 0.0
    assert config["ema_decay"] == pytest.approx(0.995)
    assert config["effective_batch_size"] == 4
    assert config["inner_epochs"] == 1
    assert config["consistency_video_stride"] == 500
    assert config["consistency_action_stride"] == 500
    assert config["consistency_noise_source"] == "artifact_epsilon"
    assert config["calibration_anchors_per_trajectory"] == 5
    assert config["loss_reduction"] == "mean_trajectories_mean_labels"


def test_success_path_v1_config_allows_three_inner_epochs_without_changing_baseline(
    iterative_flow: object,
) -> None:
    success_config = _coherent_config()
    success_config.update(
        {
            "coherent_tt_variant": "success_path_v1",
            "inner_epochs": 3,
        }
    )

    normalized = iterative_flow._normalize_config(success_config)

    assert normalized["coherent_tt_variant"] == "success_path_v1"
    assert normalized["inner_epochs"] == 3
    baseline = iterative_flow._normalize_config(_coherent_config())
    assert baseline["coherent_tt_variant"] == "baseline"
    assert baseline["inner_epochs"] == 1


def test_success_path_trajectory_update_is_the_only_exact_resume_config(
    iterative_flow: object,
) -> None:
    config = {
        "run_mode": "trajectory_update",
        "objective": "coherent_tt_consistency",
        "coherent_tt_variant": "success_path_v1",
        "task": "put_object_cabinet",
        "chunks": 23,
        "rounds": 1,
        "rollouts": [
            {
                "rollout_id": 0,
                "seed": 10003,
                "prompt": "cabinet train",
                "role": "train",
            },
            {
                "rollout_id": 1,
                "seed": 10055,
                "prompt": "cabinet calibration",
                "role": "calibration",
            },
        ],
        "collection_group_id": "cabinet-formal-r0",
        "adapter_seed": 20260820,
        "trajectory_artifacts": ["train.pt", "calibration.pt"],
        "adapter_kind": "joint_lora",
        "adapter_rank": 8,
        "lora_block_indices": list(range(30)),
        "inner_epochs": 3,
        "initial_checkpoint": "/ssd/data/cabinet/checkpoint_epoch_01.pt",
        "resume_optimizer_state": True,
    }

    normalized = iterative_flow._normalize_config(config)

    assert normalized["initial_checkpoint"].endswith("checkpoint_epoch_01.pt")
    assert normalized["resume_optimizer_state"] is True
    assert normalized["inner_epochs"] == 3

    without_optimizer = deepcopy(config)
    without_optimizer["resume_optimizer_state"] = False
    with pytest.raises(ValueError, match="requires resume_optimizer_state=true"):
        iterative_flow._normalize_config(without_optimizer)

    baseline = deepcopy(config)
    baseline["coherent_tt_variant"] = "baseline"
    baseline["inner_epochs"] = 1
    with pytest.raises(ValueError, match="only supports exact resume"):
        iterative_flow._normalize_config(baseline)


@pytest.mark.parametrize("publisher", ["json", "torch"])
def test_success_path_atomic_publish_never_exposes_partial_final_path(
    iterative_flow: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    publisher: str,
) -> None:
    target = tmp_path / ("summary.json" if publisher == "json" else "epoch.pt")

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(iterative_flow.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated atomic replace failure"):
        if publisher == "json":
            iterative_flow._write_json(target, {"status": "PASS"})
        else:
            iterative_flow._atomic_torch_save(
                target,
                {"tensor": torch.tensor([1.0])},
            )

    assert not target.exists()
    assert not list(tmp_path.glob(f"{iterative_flow._atomic_temp_prefix(target)}*"))


def test_success_path_temp_cleanup_preserves_a_live_atomic_writer(
    iterative_flow: object,
    tmp_path: Path,
) -> None:
    target = tmp_path / "checkpoint.pt"
    prefix = iterative_flow._atomic_temp_prefix(target)
    dead_temp = tmp_path / f"{prefix}999999999-dead.tmp"
    live_temp = tmp_path / f"{prefix}{iterative_flow.os.getpid()}-live.tmp"
    dead_temp.write_bytes(b"dead partial")
    live_temp.write_bytes(b"active writer")

    removed = iterative_flow._cleanup_atomic_temps(target)

    assert removed == [dead_temp]
    assert not dead_temp.exists()
    assert live_temp.exists()
    live_temp.unlink()


def test_success_path_writer_lock_excludes_live_writer_and_releases(
    iterative_flow: object,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "cabinet-update"
    contract = {"task_contract_hash": "cabinet-contract-v1"}
    lock_path = output_dir / ".success_path_writer.lock"

    with iterative_flow._success_path_output_lock(
        output_dir=output_dir,
        contract=contract,
    ):
        assert lock_path.is_file()
        with pytest.raises(RuntimeError, match="live writer"):
            iterative_flow._acquire_success_path_writer_lock(
                output_dir=output_dir,
                contract=contract,
            )

    assert not lock_path.exists()


def test_success_path_writer_lock_reclaims_only_dead_matching_contract(
    iterative_flow: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "cabinet-update"
    output_dir.mkdir()
    lock_path = output_dir / ".success_path_writer.lock"
    contract = {"task_contract_hash": "cabinet-contract-v1"}
    stale_owner = {
        "schema": iterative_flow.SUCCESS_PATH_WRITER_LOCK_SCHEMA,
        "pid": 999999999,
        "host": iterative_flow.socket.gethostname(),
        "process_started_at": "2026-08-20T00:00:00+00:00",
        "owner_token": "dead-owner",
        "contract": contract,
        "contract_hash": iterative_flow._stable_hash(contract),
    }
    iterative_flow._write_json(lock_path, stale_owner)
    monkeypatch.setattr(
        iterative_flow,
        "_writer_pid_is_alive",
        lambda _pid: False,
    )

    with iterative_flow._success_path_output_lock(
        output_dir=output_dir,
        contract=contract,
    ) as replacement_owner:
        assert replacement_owner["owner_token"] != "dead-owner"
        assert replacement_owner["contract"] == contract

    assert not lock_path.exists()


def test_success_path_writer_lock_preserves_dead_foreign_contract(
    iterative_flow: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "cabinet-update"
    output_dir.mkdir()
    lock_path = output_dir / ".success_path_writer.lock"
    foreign_contract = {"task_contract_hash": "handover-contract-v1"}
    requested_contract = {"task_contract_hash": "cabinet-contract-v1"}
    stale_owner = {
        "schema": iterative_flow.SUCCESS_PATH_WRITER_LOCK_SCHEMA,
        "pid": 999999999,
        "host": iterative_flow.socket.gethostname(),
        "process_started_at": "2026-08-20T00:00:00+00:00",
        "owner_token": "foreign-dead-owner",
        "contract": foreign_contract,
        "contract_hash": iterative_flow._stable_hash(foreign_contract),
    }
    iterative_flow._write_json(lock_path, stale_owner)
    original_lock_bytes = lock_path.read_bytes()
    monkeypatch.setattr(
        iterative_flow,
        "_writer_pid_is_alive",
        lambda _pid: False,
    )

    with pytest.raises(RuntimeError, match="different contract"):
        iterative_flow._acquire_success_path_writer_lock(
            output_dir=output_dir,
            contract=requested_contract,
        )

    assert lock_path.read_bytes() == original_lock_bytes


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task", "handover_mic", "task condition mismatch"),
        ("task_contract_hash", "handover-contract", "task contract mismatch"),
    ],
)
def test_success_path_resume_checkpoint_rejects_handover_identity(
    iterative_flow: object,
    field: str,
    value: str,
    message: str,
) -> None:
    config = iterative_flow._normalize_config(
        {
            "run_mode": "trajectory_update",
            "objective": "coherent_tt_consistency",
            "coherent_tt_variant": "success_path_v1",
            "task": "put_object_cabinet",
            "chunks": 23,
            "rounds": 1,
            "rollouts": [
                {
                    "rollout_id": 0,
                    "seed": 10003,
                    "prompt": "cabinet train",
                    "role": "train",
                },
                {
                    "rollout_id": 1,
                    "seed": 10055,
                    "prompt": "cabinet calibration",
                    "role": "calibration",
                },
            ],
            "collection_group_id": "cabinet-formal-r0",
            "adapter_seed": 20260820,
            "trajectory_artifacts": ["train.pt", "calibration.pt"],
            "adapter_kind": "joint_lora",
            "adapter_rank": 8,
            "lora_block_indices": list(range(30)),
            "inner_epochs": 3,
            "initial_checkpoint": "/ssd/data/cabinet/checkpoint_epoch_01.pt",
            "resume_optimizer_state": True,
        }
    )
    task_contract_hash = iterative_flow._stable_hash(config["task_entries"])
    checkpoint = {
        "schema": iterative_flow.CHECKPOINT_SCHEMA,
        "adapter_kind": "joint_lora",
        "objective": "coherent_tt_consistency",
        "coherent_tt_variant": "success_path_v1",
        "task": "put_object_cabinet",
        "task_config": "demo_clean",
        "task_contract_hash": task_contract_hash,
    }
    checkpoint[field] = value

    with pytest.raises(ValueError, match=message):
        iterative_flow._validate_success_path_resume_checkpoint(
            checkpoint,
            config=config,
            expected_task_contract_hash=task_contract_hash,
            expected_behavior_policy_version="released-cabinet-policy",
            expected_round_id=0,
        )


def test_coherent_tt_variant_dispatch_keeps_baseline_as_the_default(
    iterative_flow: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def baseline_update(**_kwargs: object) -> dict[str, str]:
        calls.append("baseline")
        return {"variant": "baseline"}

    def success_path_update(**_kwargs: object) -> dict[str, str]:
        calls.append("success_path_v1")
        return {"variant": "success_path_v1"}

    monkeypatch.setattr(
        iterative_flow,
        "_update_round_coherent_tt_consistency",
        baseline_update,
    )
    monkeypatch.setattr(
        iterative_flow,
        "_update_round_success_path_tt",
        success_path_update,
    )
    common = {
        "runtime": object(),
        "optimizer": object(),
        "trajectories": [],
        "round_id": 0,
        "video_weight": 1.0,
        "action_weight": 1.0,
        "action_velocity_weight": 0.0,
        "pseudo_huber_c": 0.1,
        "max_grad_norm": 2.0,
        "objective": "coherent_tt_consistency",
        "action_fm_weight": 0.2,
        "calibration_anchors_per_trajectory": 2,
    }

    baseline = iterative_flow._update_round(**common)
    success_path = iterative_flow._update_round(
        **common,
        coherent_tt_variant="success_path_v1",
    )

    assert baseline == {"variant": "baseline"}
    assert success_path == {"variant": "success_path_v1"}
    assert calls == ["baseline", "success_path_v1"]


def test_coherent_tt_calibration_anchor_selection_is_deterministic_and_keeps_endpoints(
    iterative_flow: object,
) -> None:
    assert iterative_flow._coherent_tt_calibration_indices(13, 5) == (
        0,
        3,
        6,
        9,
        12,
    )
    assert iterative_flow._coherent_tt_calibration_indices(3, 5) == (0, 1, 2)
    assert iterative_flow._coherent_tt_calibration_indices(1, 5) == (0,)


def test_coherent_tt_config_rejects_one_calibration_anchor(
    iterative_flow: object,
) -> None:
    config = _coherent_config()
    config["calibration_anchors_per_trajectory"] = 1

    with pytest.raises(ValueError, match="at least two"):
        iterative_flow._normalize_config(config)


def test_coherent_tt_anchor_samples_preserve_whole_trajectory_roles_and_weights(
    iterative_flow: object,
) -> None:
    trajectories = [
        {
            "seed": 10,
            "dataset_role": "calibration",
            "labels": [{"macro_id": index} for index in range(7)],
        },
        {
            "seed": 11,
            "dataset_role": "calibration",
            "labels": [{"macro_id": index} for index in range(2)],
        },
    ]

    samples, weights, indices = iterative_flow._coherent_tt_calibration_samples(
        trajectories, anchors_per_trajectory=3
    )

    assert indices == {10: [0, 3, 6], 11: [0, 1]}
    assert [int(label["macro_id"]) for _trajectory, label in samples] == [
        0,
        3,
        6,
        0,
        1,
    ]
    assert all(
        trajectory is trajectories[0] for trajectory, _label in samples[:3]
    )
    assert all(
        trajectory is trajectories[1] for trajectory, _label in samples[3:]
    )
    assert sum(weights[:3]) == pytest.approx(0.5)
    assert sum(weights[3:]) == pytest.approx(0.5)
    assert [len(trajectory["labels"]) for trajectory in trajectories] == [7, 2]
    assert all(
        trajectory["dataset_role"] == "calibration" for trajectory in trajectories
    )


def test_coherent_tt_config_rejects_missing_whole_trajectory_role(
    iterative_flow: object,
) -> None:
    config = _coherent_config()
    config["rollouts"] = [
        {"seed": 10000, "prompt": "handover"},
        {"seed": 10001, "prompt": "handover calibration"},
    ]

    with pytest.raises(ValueError, match="explicit train/calibration role"):
        iterative_flow._normalize_config(config)


def test_coherent_tt_collect_allows_one_role_shard_with_global_identity(
    iterative_flow: object,
) -> None:
    config = _coherent_config()
    config.update(
        {
            "run_mode": "collect",
            "collection_group_id": "handover-formal-r0",
            "adapter_seed": 20260820,
            "rollouts": [
                {
                    "rollout_id": 17,
                    "seed": 10000,
                    "prompt": "handover",
                    "role": "train",
                }
            ],
        }
    )

    normalized = iterative_flow._normalize_config(config)

    assert normalized["run_mode"] == "collect"
    assert normalized["collection_group_id"] == "handover-formal-r0"
    assert normalized["adapter_seed"] == 20260820
    assert normalized["task_entries"][0]["rollouts"][0]["rollout_id"] == 17


def test_sharded_coherent_modes_require_explicit_shared_adapter_seed(
    iterative_flow: object,
) -> None:
    config = _coherent_config()
    config.update(
        {
            "run_mode": "collect",
            "collection_group_id": "handover-formal-r0",
            "rollouts": [
                {
                    "rollout_id": 17,
                    "seed": 10000,
                    "prompt": "handover",
                    "role": "train",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="explicit adapter_seed"):
        iterative_flow._normalize_config(config)


def _sharded_trajectory_artifact(
    iterative_flow: object,
    *,
    student: Path,
    teacher: Path,
    rollout_id: int,
    seed: int,
    prompt: str,
    role: str,
) -> dict[str, object]:
    collection_id = f"handover_mic_round00_rollout{rollout_id:02d}"
    label = {
        "collection_id": collection_id,
        "collection_group_id": "handover-formal-r0",
        "adapter_seed": 20260820,
        "round_id": 0,
        "rollout_id": rollout_id,
        "behavior_policy_version": "shared-policy",
        "history_prefix_length": 0,
        "macro_id": 0,
        "frame_st_id": 0,
        "start_frame": 0,
        "action_steps": 1,
        "executed_action_mask": torch.tensor([True]),
        "terminal_reached": False,
        "terminal_action_position": None,
        "horizon_reached": False,
        "cumulative_control_steps": 1,
    }
    physical = {
        "chunk_id": 0,
        "frame_st_id": 0,
        "start_frame": 0,
        "action_steps": 1,
        "executed_action_mask": torch.tensor([True]),
        "terminal_reached": False,
        "terminal_action_position": None,
        "horizon_reached": False,
    }
    return {
        "schema": iterative_flow.TRAJECTORY_SCHEMA,
        "task": "handover_mic",
        "task_config": "demo_clean",
        "seed": seed,
        "prompt": prompt,
        "dataset_role": role,
        "collection_id": collection_id,
        "collection_group_id": "handover-formal-r0",
        "adapter_seed": 20260820,
        "round_id": 0,
        "rollout_id": rollout_id,
        "behavior_policy_version": "shared-policy",
        "base_student_checkpoint": str(student.resolve()),
        "teacher_transformer": str(teacher.resolve()),
        "objective": "coherent_tt_consistency",
        "history_owner": "current_student_on_policy",
        "environment_execution": "SS",
        "teacher_controls_environment": False,
        "fresh_passes_allowed": 1,
        "success_post_label_count": 0,
        "history": [{}],
        "labels": [label],
        "baseline_episode": {
            "task": "handover_mic",
            "task_config": "demo_clean",
            "seed": seed,
            "prompt": prompt,
            "arm": "SS",
            "student_checkpoint": str(student.resolve()),
            "chunks_requested": 20,
            "max_control_steps": 600,
            "shared_noise_across_arms": True,
            "chunks": [physical],
        },
    }


def test_coherent_trajectory_merge_validates_and_orders_independent_shards(
    iterative_flow: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    student = tmp_path / "student"
    teacher = tmp_path / "teacher"
    task_entries = [
        {
            "task": "handover_mic",
            "task_config": "demo_clean",
            "chunks": 20,
            "gate_json": None,
            "rollouts": [
                {
                    "rollout_id": 10,
                    "seed": 10000,
                    "prompt": "train prompt",
                    "role": "train",
                },
                {
                    "rollout_id": 11,
                    "seed": 10001,
                    "prompt": "calibration prompt",
                    "role": "calibration",
                },
            ],
        }
    ]
    artifacts = [
        _sharded_trajectory_artifact(
            iterative_flow,
            student=student,
            teacher=teacher,
            rollout_id=10,
            seed=10000,
            prompt="train prompt",
            role="train",
        ),
        _sharded_trajectory_artifact(
            iterative_flow,
            student=student,
            teacher=teacher,
            rollout_id=11,
            seed=10001,
            prompt="calibration prompt",
            role="calibration",
        ),
    ]
    paths = [tmp_path / "shard_train.pt", tmp_path / "shard_calibration.pt"]
    for path, artifact in zip(paths, artifacts, strict=True):
        torch.save(artifact, path)
    monkeypatch.setattr(iterative_flow, "_validate_label", lambda _label: None)

    merged = iterative_flow._load_trajectory_artifacts(
        list(reversed(paths)),
        expected_task_entries=task_entries,
        expected_student=student,
        expected_teacher=teacher,
        expected_adapter_seed=20260820,
        expected_collection_group_id="handover-formal-r0",
        require_coherent_collection_contract=True,
    )

    assert [int(artifact["rollout_id"]) for artifact in merged] == [10, 11]
    assert [str(artifact["dataset_role"]) for artifact in merged] == [
        "train",
        "calibration",
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("collection_group_id", "other-group", "collection group"),
        ("round_id", 1, "different rounds"),
        ("adapter_seed", 7, "adapter seed"),
        ("base_student_checkpoint", "/other/student", "base Student"),
        ("teacher_transformer", "/other/teacher", "Teacher checkpoint"),
        ("behavior_policy_version", "other-policy", "different Student policies"),
        ("rollout_id", 10, "global rollout_id is duplicated"),
        ("dataset_role", "train", "dataset role"),
        ("prompt", "other prompt", "task/seed/prompt"),
    ],
)
def test_coherent_trajectory_merge_rejects_cross_shard_contract_drift(
    iterative_flow: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    student = tmp_path / "student"
    teacher = tmp_path / "teacher"
    task_entries = [
        {
            "task": "handover_mic",
            "task_config": "demo_clean",
            "chunks": 20,
            "gate_json": None,
            "rollouts": [
                {
                    "rollout_id": 10,
                    "seed": 10000,
                    "prompt": "train prompt",
                    "role": "train",
                },
                {
                    "rollout_id": 11,
                    "seed": 10001,
                    "prompt": "calibration prompt",
                    "role": "calibration",
                },
            ],
        }
    ]
    artifacts = [
        _sharded_trajectory_artifact(
            iterative_flow,
            student=student,
            teacher=teacher,
            rollout_id=10,
            seed=10000,
            prompt="train prompt",
            role="train",
        ),
        _sharded_trajectory_artifact(
            iterative_flow,
            student=student,
            teacher=teacher,
            rollout_id=11,
            seed=10001,
            prompt="calibration prompt",
            role="calibration",
        ),
    ]
    artifacts[1][field] = value
    if field in {
        "collection_group_id",
        "round_id",
        "adapter_seed",
        "behavior_policy_version",
        "rollout_id",
    }:
        artifacts[1]["labels"][0][field] = value
    path_a = tmp_path / "shard_a.pt"
    path_b = tmp_path / "shard_b.pt"
    torch.save(artifacts[0], path_a)
    torch.save(artifacts[1], path_b)
    monkeypatch.setattr(iterative_flow, "_validate_label", lambda _label: None)

    with pytest.raises((ValueError, RuntimeError), match=message):
        iterative_flow._load_trajectory_artifacts(
            [path_a, path_b],
            expected_task_entries=task_entries,
            expected_student=student,
            expected_teacher=teacher,
            expected_adapter_seed=20260820,
            expected_collection_group_id="handover-formal-r0",
            require_coherent_collection_contract=True,
        )


def test_coherent_trajectory_update_loads_teacher_forwards_settings_and_counts_steps(
    iterative_flow: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    student = tmp_path / "student"
    teacher = tmp_path / "teacher"
    artifact_paths = [tmp_path / "train.pt", tmp_path / "calibration.pt"]
    config = iterative_flow._normalize_config(
        {
            "run_mode": "trajectory_update",
            "objective": "coherent_tt_consistency",
            "task": "handover_mic",
            "rollouts": [
                {
                    "rollout_id": 10,
                    "seed": 10000,
                    "prompt": "train prompt",
                    "role": "train",
                },
                {
                    "rollout_id": 11,
                    "seed": 10001,
                    "prompt": "calibration prompt",
                    "role": "calibration",
                },
            ],
            "collection_group_id": "handover-formal-r0",
            "adapter_seed": 20260820,
            "trajectory_artifacts": [str(path) for path in artifact_paths],
            "adapter_kind": "joint_lora",
            "adapter_rank": 8,
            "lora_block_indices": list(range(30)),
            "teacher_video_steps": 17,
            "teacher_video_exec_steps": 4,
            "teacher_action_steps": 23,
            "action_fm_weight": 0.3,
            "effective_batch_size": 2,
            "ema_decay": 0.91,
            "consistency_video_stride": 200,
            "consistency_action_stride": 300,
            "consistency_seed": 123,
        }
    )
    trajectories = [
        {
            "task": "handover_mic",
            "seed": seed,
            "round_id": 0,
            "behavior_policy_version": "policy-before",
            "labels": [{"round_id": 0}],
        }
        for seed in (10000, 10001)
    ]
    constructor: dict[str, object] = {}
    loader: dict[str, object] = {}
    solver: dict[str, object] = {}

    class FakeRuntime:
        adapter_kind = "joint_lora"

        def __init__(self, **kwargs: object) -> None:
            constructor.update(kwargs)
            self.parameter = torch.nn.Parameter(torch.tensor([1.0]))
            self.trainable = [("joint_lora_A", self.parameter)]
            self.adapter_parameter_names = ["joint_lora_A"]
            self.teacher = object()
            self.closed = False

        def configure_teacher_solver(self, **kwargs: object) -> None:
            solver.update(kwargs)

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

    runtime_holder: dict[str, FakeRuntime] = {}

    def fake_runtime(**kwargs: object) -> FakeRuntime:
        runtime = FakeRuntime(**kwargs)
        runtime_holder["runtime"] = runtime
        return runtime

    def fake_loader(*_args: object, **kwargs: object) -> list[dict[str, object]]:
        loader.update(kwargs)
        return trajectories

    def fake_policy_version(runtime: FakeRuntime) -> str:
        return "policy-before" if float(runtime.parameter.item()) == 1.0 else "policy-after"

    def fake_update(**kwargs: object) -> dict[str, object]:
        assert kwargs["objective"] == "coherent_tt_consistency"
        assert kwargs["action_fm_weight"] == pytest.approx(0.3)
        assert kwargs["effective_batch_size"] == 2
        assert kwargs["inner_epochs"] == 1
        assert kwargs["ema_decay"] == pytest.approx(0.91)
        assert kwargs["consistency_video_stride"] == 200
        assert kwargs["consistency_action_stride"] == 300
        assert kwargs["consistency_seed"] == 123
        runtime = kwargs["runtime"]
        assert isinstance(runtime, FakeRuntime)
        with torch.no_grad():
            runtime.parameter.add_(0.25)
        return {"optimizer_steps_this_round": 3}

    def fake_save_checkpoint(*, path: Path, **_kwargs: object) -> None:
        path.write_bytes(b"checkpoint")

    monkeypatch.setattr(iterative_flow, "NativeV0VideoRuntime", fake_runtime)
    monkeypatch.setattr(iterative_flow, "_load_trajectory_artifacts", fake_loader)
    monkeypatch.setattr(iterative_flow, "_policy_version", fake_policy_version)
    monkeypatch.setattr(iterative_flow, "_update_round", fake_update)
    monkeypatch.setattr(iterative_flow, "_save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(iterative_flow, "_configure_cuda_memory_limit", lambda _config: None)

    summary = iterative_flow._run_trajectory_update(
        config=config,
        student=student,
        teacher=teacher,
        output_dir=tmp_path / "update",
        task_contract_hash=iterative_flow._stable_hash(config["task_entries"]),
    )

    assert constructor["teacher_transformer"] == teacher
    assert solver == {"video_steps": 17, "video_exec_steps": 4, "action_steps": 23}
    assert loader["expected_teacher"] == teacher
    assert loader["expected_adapter_seed"] == 20260820
    assert loader["expected_collection_group_id"] == "handover-formal-r0"
    assert loader["require_coherent_collection_contract"] is True
    assert summary["teacher_loaded"] is True
    assert summary["teacher_transformer"] == str(teacher)
    assert summary["global_optimizer_step"] == 3
    assert summary["optimizer_steps_this_run"] == 3
    assert runtime_holder["runtime"].closed is True


def test_trajectory_epoch_batches_are_distinct_and_consume_every_label_once(
    iterative_flow: object,
) -> None:
    trajectories = [
        {
            "seed": seed,
            "labels": [
                {"collection_id": f"seed-{seed}", "macro_id": macro_id}
                for macro_id in range(count)
            ],
        }
        for seed, count in ((1, 3), (2, 2), (3, 4), (4, 1), (5, 2))
    ]
    batches = iterative_flow._trajectory_distinct_epoch_batches(
        trajectories,
        batch_size=4,
        generator=torch.Generator().manual_seed(7),
    )

    observed: list[tuple[str, int]] = []
    for batch in batches:
        seeds = [int(trajectory["seed"]) for trajectory, _label in batch]
        assert len(seeds) == len(set(seeds))
        assert len(batch) <= 4
        observed.extend(
            (str(label["collection_id"]), int(label["macro_id"]))
            for _trajectory, label in batch
        )
    expected = [
        (str(label["collection_id"]), int(label["macro_id"]))
        for trajectory in trajectories
        for label in trajectory["labels"]
    ]
    assert sorted(observed) == sorted(expected)


def test_trajectory_equal_epoch_scales_give_short_and_long_trajectories_equal_total_weight(
    iterative_flow: object,
) -> None:
    trajectories = [
        {
            "seed": seed,
            "labels": [
                {"collection_id": f"seed-{seed}", "macro_id": macro_id}
                for macro_id in range(count)
            ],
        }
        for seed, count in ((1, 1), (2, 3))
    ]
    batches = iterative_flow._trajectory_distinct_epoch_batches(
        trajectories,
        batch_size=2,
        generator=torch.Generator().manual_seed(7),
    )

    batch_scales = iterative_flow._trajectory_equal_epoch_batch_scales(
        trajectories,
        batches,
    )

    totals = {id(trajectory): 0.0 for trajectory in trajectories}
    for batch, scales in zip(batches, batch_scales, strict=True):
        assert len(batch) == len(scales)
        for (trajectory, _label), scale in zip(batch, scales, strict=True):
            totals[id(trajectory)] += float(scale)
    assert totals[id(trajectories[0])] == pytest.approx(1.5)
    assert totals[id(trajectories[1])] == pytest.approx(1.5)
    assert sum(sum(scales) for scales in batch_scales) / len(batches) == pytest.approx(
        1.0
    )


def test_coherent_calibration_uses_frozen_bridge_ema_and_action_fm_without_state_drift(
    iterative_flow: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_parameter = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float32))

    class FakeRuntime:
        device = torch.device("cpu")
        dtype = torch.float32

        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[float, ...], bool | None]] = []

        @staticmethod
        def _state_forward(state: torch.Tensor, prediction: torch.Tensor) -> object:
            return SimpleNamespace(
                noisy_state=state.detach().clone(),
                velocity=prediction.expand_as(state),
                x0_prediction=prediction.expand_as(state),
                consistency_prediction=prediction.expand_as(state),
                valid_mask=torch.ones_like(state, dtype=torch.bool),
                token_positions=(0, 1),
                cache_valid_length=2,
            )

        def teacher_video_velocity_at_state(
            self, _context: object, state: torch.Tensor, *, timestep: torch.Tensor, sigma: torch.Tensor
        ) -> object:
            self.calls.append(("teacher_video", tuple(timestep.tolist()), None))
            return self._state_forward(state, torch.zeros(1))

        def teacher_action_velocity_at_state(
            self,
            _context: object,
            _plan: torch.Tensor,
            _plan_timestep: torch.Tensor,
            state: torch.Tensor,
            *,
            timestep: torch.Tensor,
            sigma: torch.Tensor,
        ) -> object:
            self.calls.append(("teacher_action", tuple(timestep.tolist()), None))
            return self._state_forward(state, torch.zeros(1))

        def student_video_consistency_at_state(
            self,
            _context: object,
            state: torch.Tensor,
            *,
            timestep: torch.Tensor,
            sigma: torch.Tensor,
            require_grad: bool,
        ) -> object:
            self.calls.append(("student_video", tuple(timestep.tolist()), require_grad))
            return self._state_forward(state, live_parameter)

        def student_action_consistency_at_state(
            self,
            _context: object,
            _plan: torch.Tensor,
            _plan_timestep: torch.Tensor,
            state: torch.Tensor,
            *,
            timestep: torch.Tensor,
            sigma: torch.Tensor,
            require_grad: bool,
        ) -> object:
            self.calls.append(("student_action", tuple(timestep.tolist()), require_grad))
            return self._state_forward(state, live_parameter)

    monkeypatch.setattr(
        iterative_flow,
        "materialize_context",
        lambda *_args: {
            "epsilon_v": torch.ones((1, 1, 2, 1, 1)),
            "epsilon_a": torch.ones((1, 1, 2, 1, 1)),
        },
    )
    monkeypatch.setattr(
        iterative_flow,
        "video_execution_mask",
        lambda reference, _label: torch.ones_like(reference, dtype=torch.bool),
    )
    monkeypatch.setattr(
        iterative_flow,
        "action_execution_mask",
        lambda valid_mask, _label: valid_mask,
    )
    monkeypatch.setattr(
        iterative_flow,
        "action_velocity_mse_loss",
        lambda prediction, target, mask: torch.nn.functional.mse_loss(
            prediction[mask], target[mask]
        ),
    )

    shape = (1, 1, 2, 1, 1)
    label = {
        "teacher_z_t": torch.zeros(shape),
        "teacher_z_t_timestep": torch.zeros(2),
        "teacher_action": torch.zeros(shape),
    }
    schedule = {
        "start_timestep": torch.tensor([900.0, 800.0]),
        "end_timestep": torch.tensor([400.0, 300.0]),
        "start_sigma": torch.tensor([0.9, 0.8]),
        "end_sigma": torch.tensor([0.4, 0.3]),
    }
    cases = [
        {
            "trajectory": {"labels": [label]},
            "label": label,
            "video_schedule": schedule,
            "action_schedule": schedule,
            "weight": 1.0,
        }
    ]
    runtime = FakeRuntime()
    target = iterative_flow.LoRAEMAState.from_online(
        {"adapter.lora_A": live_parameter}, decay=0.995
    )
    # Freeze the target at 2.0, then move only the online parameter.
    with torch.no_grad():
        live_parameter.fill_(3.0)
    live_before = live_parameter.detach().clone()
    target_before = target.target_state()

    prepared_cases = iterative_flow._prepare_coherent_tt_calibration_cases(
        runtime=runtime,
        cases=cases,
        live_state={"adapter.lora_A": live_parameter},
        target_ema=target,
    )
    first = iterative_flow._evaluate_coherent_tt_calibration(
        runtime=runtime,
        cases=prepared_cases,
        pseudo_huber_c=0.1,
        video_weight=1.0,
        action_weight=1.0,
        action_fm_weight=0.2,
    )
    second = iterative_flow._evaluate_coherent_tt_calibration(
        runtime=runtime,
        cases=prepared_cases,
        pseudo_huber_c=0.1,
        video_weight=1.0,
        action_weight=1.0,
        action_fm_weight=0.2,
    )

    assert first == pytest.approx(second)
    assert first["action_fm_loss"] > 0.0
    assert first["loss"] == pytest.approx(
        first["video_loss"]
        + first["action_loss"]
        + 0.2 * first["action_fm_loss"]
    )
    assert runtime.calls == [
        ("teacher_video", (900.0, 800.0), None),
        ("teacher_action", (900.0, 800.0), None),
        ("student_video", (400.0, 300.0), False),
        ("student_action", (400.0, 300.0), False),
        ("student_video", (900.0, 800.0), False),
        ("student_action", (900.0, 800.0), False),
        ("student_video", (900.0, 800.0), False),
        ("student_action", (900.0, 800.0), False),
    ]
    torch.testing.assert_close(live_parameter, live_before)
    for name, value in target_before.items():
        torch.testing.assert_close(target.target_state()[name], value)


def _success_path_label(
    *,
    collection_id: str,
    macro_id: int,
    teacher_plan: torch.Tensor | None = None,
    teacher_action: torch.Tensor | None = None,
    epsilon_a: torch.Tensor | None = None,
) -> dict[str, object]:
    plan = (
        torch.zeros(2, dtype=torch.float32)
        if teacher_plan is None
        else teacher_plan.detach().clone()
    )
    action = (
        torch.zeros(2, dtype=torch.float32)
        if teacher_action is None
        else teacher_action.detach().clone()
    )
    noise = (
        torch.zeros_like(action)
        if epsilon_a is None
        else epsilon_a.detach().clone()
    )
    return {
        "round_id": 0,
        "collection_id": collection_id,
        "macro_id": macro_id,
        "teacher_z_t": plan,
        "teacher_action": action,
        "epsilon_a": noise,
        "teacher_action_input_noise": torch.tensor([0.25, 0.75]),
        "teacher_action_timestep": torch.tensor([900.0, 800.0]),
        "teacher_action_valid_mask": torch.ones_like(action, dtype=torch.bool),
        "teacher_action_token_positions": (0, 1),
        "teacher_action_cache_valid_length": 2,
    }


def _success_path_forward(
    *,
    plan: torch.Tensor,
    endpoint: torch.Tensor,
    initial_velocity: torch.Tensor,
    label: dict[str, object],
) -> SimpleNamespace:
    return SimpleNamespace(
        plan=SimpleNamespace(prepared_z_s=plan),
        action=SimpleNamespace(
            endpoint=endpoint,
            initial_velocity=initial_velocity,
            action_input_noise=label["teacher_action_input_noise"],
            action_timestep=label["teacher_action_timestep"],
            valid_mask=label["teacher_action_valid_mask"],
            token_positions=label["teacher_action_token_positions"],
            cache_valid_length=label["teacher_action_cache_valid_length"],
        ),
    )


def _patch_success_path_tensor_contracts(
    iterative_flow: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(iterative_flow, "_validate_label", lambda _label: None)
    monkeypatch.setattr(
        iterative_flow,
        "materialize_context",
        lambda _trajectory, label: {
            "epsilon_a": label["epsilon_a"],
            "label": label,
        },
    )
    monkeypatch.setattr(
        iterative_flow,
        "video_execution_mask",
        lambda reference, _label: torch.ones_like(reference, dtype=torch.bool),
    )
    monkeypatch.setattr(
        iterative_flow,
        "action_execution_mask",
        lambda valid_mask, _label: valid_mask,
    )
    monkeypatch.setattr(
        iterative_flow,
        "action_velocity_mse_loss",
        lambda prediction, target, mask: torch.nn.functional.mse_loss(
            prediction[mask], target[mask]
        ),
    )


def test_success_path_calibration_uses_deployment_joint_forward_and_artifact_targets(
    iterative_flow: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_success_path_tensor_contracts(iterative_flow, monkeypatch)
    teacher_plan = torch.tensor([10.0, 12.0])
    teacher_action = torch.tensor([20.0, 24.0])
    epsilon_a = torch.tensor([6.0, 8.0])
    label = _success_path_label(
        collection_id="calibration-0",
        macro_id=0,
        teacher_plan=teacher_plan,
        teacher_action=teacher_action,
        epsilon_a=epsilon_a,
    )
    student_plan = torch.tensor([2.0, 4.0])
    action_offset = torch.tensor([1.0, 1.0])
    initial_velocity = torch.tensor([0.5, 1.5])

    class FakeRuntime:
        device = torch.device("cpu")
        dtype = torch.float32

        def __init__(self) -> None:
            self.calls: list[bool] = []

        def student_video_forward(
            self,
            context: dict[str, object],
            *,
            detach_plan_for_action: bool,
        ) -> SimpleNamespace:
            self.calls.append(detach_plan_for_action)
            # Model the deployment dependency explicitly: action consumes z_S,
            # while the Teacher plan is available only as a loss target.
            z_s = student_plan
            endpoint = z_s.detach() + action_offset
            return _success_path_forward(
                plan=z_s,
                endpoint=endpoint,
                initial_velocity=initial_velocity,
                label=context["label"],
            )

    runtime = FakeRuntime()
    prepared = iterative_flow._prepare_success_path_calibration_cases(
        runtime=runtime,
        cases=[
            {
                "trajectory": {"labels": [label]},
                "label": label,
                "weight": 1.0,
            }
        ],
    )

    torch.testing.assert_close(prepared[0]["target_plan"], teacher_plan)
    torch.testing.assert_close(prepared[0]["target_action"], teacher_action)
    torch.testing.assert_close(
        prepared[0]["action_fm_target"], epsilon_a - teacher_action
    )
    metrics = iterative_flow._evaluate_success_path_calibration(
        runtime=runtime,
        cases=prepared,
        pseudo_huber_c=0.1,
        video_weight=1.0,
        action_weight=1.0,
        action_fm_weight=0.2,
    )

    expected_video = iterative_flow._masked_pseudo_huber_loss(
        student_plan,
        teacher_plan,
        torch.ones_like(student_plan, dtype=torch.bool),
        c=0.1,
    )
    expected_action = iterative_flow._masked_pseudo_huber_loss(
        student_plan + action_offset,
        teacher_action,
        torch.ones_like(teacher_action, dtype=torch.bool),
        c=0.1,
    )
    expected_action_fm = torch.nn.functional.mse_loss(
        initial_velocity,
        epsilon_a - teacher_action,
    )
    assert metrics["video_loss"] == pytest.approx(float(expected_video.item()))
    assert metrics["action_loss"] == pytest.approx(float(expected_action.item()))
    assert metrics["action_fm_loss"] == pytest.approx(
        float(expected_action_fm.item())
    )
    assert metrics["loss"] == pytest.approx(
        float(expected_video.item())
        + float(expected_action.item())
        + 0.2 * float(expected_action_fm.item())
    )
    assert runtime.calls == [True]
    assert not torch.equal(student_plan, teacher_plan)


def test_success_path_three_epochs_cover_every_label_and_reuse_fixed_calibration_cases(
    iterative_flow: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_success_path_tensor_contracts(iterative_flow, monkeypatch)
    monkeypatch.setattr(iterative_flow, "_policy_version", lambda _runtime: "policy-0")

    train_trajectories = [
        {
            "task": "handover_mic",
            "seed": seed,
            "dataset_role": "train",
            "behavior_policy_version": "policy-0",
            "labels": [
                _success_path_label(
                    collection_id=collection_id,
                    macro_id=macro_id,
                )
                for macro_id in range(2)
            ],
        }
        for seed, collection_id in ((101, "train-a"), (102, "train-b"))
    ]
    calibration_labels = [
        _success_path_label(
            collection_id="calibration",
            macro_id=macro_id,
        )
        for macro_id in range(2)
    ]
    calibration_trajectory = {
        "task": "handover_mic",
        "seed": 201,
        "dataset_role": "calibration",
        "behavior_policy_version": "policy-0",
        "labels": calibration_labels,
    }

    class FakeRuntime:
        adapter_kind = "joint_lora"
        device = torch.device("cpu")
        dtype = torch.float32

        def __init__(self) -> None:
            self.lora_a = torch.nn.Parameter(torch.tensor(1.0))
            self.lora_b = torch.nn.Parameter(torch.tensor(1.0))
            self.trainable = [
                ("blocks.0.attn.lora_A", self.lora_a),
                ("blocks.0.attn.lora_B", self.lora_b),
            ]
            self.calls: list[bool] = []

        def student_video_forward(
            self,
            context: dict[str, object],
            *,
            detach_plan_for_action: bool,
        ) -> SimpleNamespace:
            self.calls.append(detach_plan_for_action)
            label = context["label"]
            reference = label["teacher_z_t"]
            plan = self.lora_a.expand_as(reference)
            endpoint = self.lora_b.expand_as(reference) + 0.25 * plan.detach()
            velocity = self.lora_b.expand_as(reference)
            return _success_path_forward(
                plan=plan,
                endpoint=endpoint,
                initial_velocity=velocity,
                label=label,
            )

    runtime = FakeRuntime()
    optimizer = torch.optim.AdamW(
        [runtime.lora_a, runtime.lora_b],
        lr=1e-2,
        weight_decay=0.0,
    )
    original_evaluate = iterative_flow._evaluate_success_path_calibration
    evaluated_cases: list[object] = []

    def evaluate_spy(**kwargs: object) -> dict[str, float]:
        evaluated_cases.append(kwargs["cases"])
        return original_evaluate(**kwargs)

    monkeypatch.setattr(
        iterative_flow,
        "_evaluate_success_path_calibration",
        evaluate_spy,
    )
    checkpoints: list[tuple[int, int]] = []

    summary = iterative_flow._update_round_success_path_tt(
        runtime=runtime,
        optimizer=optimizer,
        trajectories=[*train_trajectories, calibration_trajectory],
        round_id=0,
        video_weight=1.0,
        action_weight=1.0,
        action_fm_weight=0.2,
        pseudo_huber_c=0.1,
        max_grad_norm=10.0,
        effective_batch_size=2,
        inner_epochs=3,
        consistency_seed=7,
        calibration_anchors_per_trajectory=2,
        epoch_checkpoint_callback=lambda epoch, steps, _progress: (
            checkpoints.append((epoch, steps)) or f"epoch-{epoch}.pt"
        ),
    )

    assert summary["coherent_tt_variant"] == "success_path_v1"
    assert summary["inner_epochs"] == 3
    assert summary["optimizer_steps_this_round"] == 6
    assert summary["train_samples_per_epoch"] == 4
    assert summary["train_samples"] == 12
    assert len(summary["epoch_metrics"]) == 3
    assert [row["optimizer_steps"] for row in summary["epoch_metrics"]] == [2, 2, 2]
    assert checkpoints == [(1, 2), (2, 4)]
    assert [row["checkpoint"] for row in summary["epoch_metrics"]] == [
        "epoch-1.pt",
        "epoch-2.pt",
        None,
    ]

    expected_samples = {
        ("train-a", 0),
        ("train-a", 1),
        ("train-b", 0),
        ("train-b", 1),
    }
    for epoch in (1, 2, 3):
        observed = [
            (row["collection_id"], row["macro_id"])
            for row in summary["samples"]
            if row["epoch"] == epoch
        ]
        assert len(observed) == 4
        assert set(observed) == expected_samples

    # One immutable prepared list is evaluated before training and after each
    # epoch, so pre/post scores cannot silently change anchors or targets.
    assert len(evaluated_cases) == 4
    assert all(cases is evaluated_cases[0] for cases in evaluated_cases[1:])
    assert len(summary["calibration_history"]) == 4
    assert runtime.calls
    assert all(runtime.calls)


def test_success_path_exact_resume_after_epoch_one_runs_only_epochs_two_and_three(
    iterative_flow: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_success_path_tensor_contracts(iterative_flow, monkeypatch)
    monkeypatch.setattr(iterative_flow, "_policy_version", lambda _runtime: "policy-0")

    trajectories = [
        {
            "task": "put_object_cabinet",
            "seed": seed,
            "dataset_role": "train",
            "behavior_policy_version": "policy-0",
            "labels": [
                _success_path_label(collection_id=collection_id, macro_id=macro_id)
                for macro_id in range(2)
            ],
        }
        for seed, collection_id in ((101, "train-a"), (102, "train-b"))
    ]
    trajectories.append(
        {
            "task": "put_object_cabinet",
            "seed": 201,
            "dataset_role": "calibration",
            "behavior_policy_version": "policy-0",
            "labels": [
                _success_path_label(collection_id="calibration", macro_id=macro_id)
                for macro_id in range(2)
            ],
        }
    )

    class FakeRuntime:
        adapter_kind = "joint_lora"
        device = torch.device("cpu")
        dtype = torch.float32

        def __init__(self) -> None:
            self.lora_a = torch.nn.Parameter(torch.tensor(1.0))
            self.lora_b = torch.nn.Parameter(torch.tensor(1.0))
            self.trainable = [
                ("blocks.0.attn.lora_A", self.lora_a),
                ("blocks.0.attn.lora_B", self.lora_b),
            ]

        def student_video_forward(
            self,
            context: dict[str, object],
            *,
            detach_plan_for_action: bool,
        ) -> SimpleNamespace:
            assert detach_plan_for_action is True
            label = context["label"]
            reference = label["teacher_z_t"]
            plan = self.lora_a.expand_as(reference)
            endpoint = self.lora_b.expand_as(reference) + 0.25 * plan.detach()
            return _success_path_forward(
                plan=plan,
                endpoint=endpoint,
                initial_velocity=self.lora_b.expand_as(reference),
                label=label,
            )

    def run(
        runtime: FakeRuntime,
        optimizer: torch.optim.Optimizer,
        *,
        callback: object,
        resume_state: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return iterative_flow._update_round_success_path_tt(
            runtime=runtime,
            optimizer=optimizer,
            trajectories=trajectories,
            round_id=0,
            video_weight=1.0,
            action_weight=1.0,
            action_fm_weight=0.2,
            pseudo_huber_c=0.1,
            max_grad_norm=10.0,
            effective_batch_size=2,
            inner_epochs=3,
            consistency_seed=7,
            calibration_anchors_per_trajectory=2,
            epoch_checkpoint_callback=callback,
            success_path_resume_state=resume_state,
        )

    reference_runtime = FakeRuntime()
    reference_optimizer = torch.optim.AdamW(
        [reference_runtime.lora_a, reference_runtime.lora_b],
        lr=1e-2,
        weight_decay=0.0,
    )
    reference = run(
        reference_runtime,
        reference_optimizer,
        callback=lambda epoch, _steps, _progress: f"reference-{epoch}.pt",
    )

    interrupted_runtime = FakeRuntime()
    interrupted_optimizer = torch.optim.AdamW(
        [interrupted_runtime.lora_a, interrupted_runtime.lora_b],
        lr=1e-2,
        weight_decay=0.0,
    )
    captured: dict[str, object] = {}

    class StopAfterEpochOne(RuntimeError):
        pass

    def stop_after_epoch_one(
        epoch: int,
        steps: int,
        progress: dict[str, object],
    ) -> str:
        assert epoch == 1
        assert steps == 2
        saved = deepcopy(progress)
        saved["epoch_metrics"][-1]["checkpoint"] = "epoch-1.pt"
        captured["progress"] = saved
        captured["generator"] = (
            interrupted_runtime._coherent_tt_generator_state.detach().clone()
        )
        raise StopAfterEpochOne

    with pytest.raises(StopAfterEpochOne):
        run(
            interrupted_runtime,
            interrupted_optimizer,
            callback=stop_after_epoch_one,
        )

    monkeypatch.setattr(
        iterative_flow,
        "_policy_version",
        lambda _runtime: "policy-after-epoch-1",
    )
    resumed_checkpoints: list[tuple[int, int]] = []
    resumed = run(
        interrupted_runtime,
        interrupted_optimizer,
        callback=lambda epoch, steps, _progress: (
            resumed_checkpoints.append((epoch, steps)) or f"epoch-{epoch}.pt"
        ),
        resume_state={
            "completed_inner_epochs": 1,
            "behavior_policy_version": "policy-0",
            "consistency_generator_state": captured["generator"],
            "success_path_progress": captured["progress"],
        },
    )

    assert resumed_checkpoints == [(2, 4)]
    assert [row["epoch"] for row in resumed["epoch_metrics"]] == [1, 2, 3]
    assert resumed["optimizer_steps_this_round"] == 6
    assert resumed["optimizer_steps_this_invocation"] == 4
    assert resumed["train_samples"] == 12
    assert [row["trajectory_seeds"] for row in resumed["steps"]] == [
        row["trajectory_seeds"] for row in reference["steps"]
    ]
    torch.testing.assert_close(
        interrupted_runtime.lora_a,
        reference_runtime.lora_a,
    )
    torch.testing.assert_close(
        interrupted_runtime.lora_b,
        reference_runtime.lora_b,
    )


def test_success_path_runner_restores_adamw_and_reports_only_remaining_steps(
    iterative_flow: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "cabinet-update"
    resume_path = (
        output_dir / "epoch_checkpoints" / "checkpoint_epoch_01.pt"
    )
    student = tmp_path / "released-student"
    student.mkdir()
    (student / "transformer.bin").write_bytes(b"released-cabinet-student")
    teacher = tmp_path / "teacher-transformer.pt"
    teacher.write_bytes(b"frozen-cabinet-teacher")
    train_artifact = tmp_path / "cabinet-train.pt"
    calibration_artifact = tmp_path / "cabinet-cal.pt"
    train_artifact.write_bytes(b"cabinet-train-artifact-v1")
    calibration_artifact.write_bytes(b"cabinet-calibration-artifact-v1")
    config = iterative_flow._normalize_config(
        {
            "run_mode": "trajectory_update",
            "objective": "coherent_tt_consistency",
            "coherent_tt_variant": "success_path_v1",
            "task": "put_object_cabinet",
            "chunks": 23,
            "rounds": 1,
            "rollouts": [
                {
                    "rollout_id": 0,
                    "seed": 10003,
                    "prompt": "cabinet train",
                    "role": "train",
                },
                {
                    "rollout_id": 1,
                    "seed": 10055,
                    "prompt": "cabinet calibration",
                    "role": "calibration",
                },
            ],
            "collection_group_id": "cabinet-formal-r0",
            "adapter_seed": 20260820,
            "trajectory_artifacts": [
                str(train_artifact),
                str(calibration_artifact),
            ],
            "adapter_kind": "joint_lora",
            "adapter_rank": 8,
            "lora_block_indices": list(range(30)),
            "inner_epochs": 3,
            "initial_checkpoint": str(resume_path),
            "resume_optimizer_state": True,
        }
    )
    task_contract_hash = iterative_flow._stable_hash(config["task_entries"])
    exact_identity = iterative_flow._success_path_exact_identity(
        artifact_paths=[train_artifact, calibration_artifact],
        student=student,
        teacher=teacher,
    )
    base_parameter_hashes = {"transformer.weight": "cabinet-base-v1"}

    class FakeRuntime:
        adapter_kind = "joint_lora"
        teacher = None

        def __init__(
            self,
            *,
            adapter_state: Path | None = None,
            **_kwargs: object,
        ) -> None:
            adapter = None
            if adapter_state is not None:
                loaded = torch.load(
                    adapter_state, map_location="cpu", weights_only=True
                )
                adapter = loaded["adapter_state_dict"]
            self.lora_a = torch.nn.Parameter(
                torch.tensor(1.0)
                if adapter is None
                else adapter["blocks.0.attn.lora_A"].detach().clone()
            )
            self.lora_b = torch.nn.Parameter(
                torch.tensor(1.0)
                if adapter is None
                else adapter["blocks.0.attn.lora_B"].detach().clone()
            )
            self.trainable = [
                ("blocks.0.attn.lora_A", self.lora_a),
                ("blocks.0.attn.lora_B", self.lora_b),
            ]
            self.adapter_parameter_names = [name for name, _ in self.trainable]
            self.closed = False

        def adapter_contract(self) -> dict[str, object]:
            return {
                "kind": "joint_lora",
                "rank": 8,
                "alpha": 8.0,
                "dropout": 0.0,
                "block_indices": list(range(30)),
                "base_parameter_hashes": dict(base_parameter_hashes),
            }

        def base_parameter_hashes(self) -> dict[str, str]:
            return dict(base_parameter_hashes)

        def adapter_state(self) -> dict[str, torch.Tensor]:
            return {
                name: parameter.detach().clone()
                for name, parameter in self.trainable
            }

        def close(self) -> None:
            self.closed = True

    checkpoint_runtime = FakeRuntime()
    checkpoint_optimizer = torch.optim.AdamW(
        [parameter for _, parameter in checkpoint_runtime.trainable],
        lr=float(config["learning_rate"]),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    for _ in range(40):
        checkpoint_optimizer.zero_grad(set_to_none=True)
        loss = (
            (checkpoint_runtime.lora_a - 0.25).square()
            + (checkpoint_runtime.lora_b + 0.50).square()
        )
        loss.backward()
        checkpoint_optimizer.step()
    checkpoint_parameters = {
        name: parameter.detach().clone()
        for name, parameter in checkpoint_runtime.trainable
    }
    checkpoint_runtime._coherent_tt_generator_state = (
        torch.Generator().manual_seed(7).get_state()
    )
    epoch_one_progress = {
        "schema": iterative_flow.SUCCESS_PATH_PROGRESS_SCHEMA,
        "completed_inner_epochs": 1,
        "behavior_policy_version": "cabinet-behavior-policy",
        "calibration_history": [
            {"epoch": 0, "loss": 2.0},
            {"epoch": 1, "loss": 1.5},
        ],
        "epoch_metrics": [
            {"epoch": 1, "checkpoint": str(resume_path)},
        ],
        "steps": [{"step_id": step} for step in range(40)],
        "samples": [],
        "loss_totals": {"video": 1.0, "action": 1.0, "action_fm": 1.0},
    }
    iterative_flow._save_checkpoint(
        path=resume_path,
        runtime=checkpoint_runtime,
        optimizer=checkpoint_optimizer,
        config=config,
        round_id=0,
        global_optimizer_step=40,
        policy_version_before="cabinet-behavior-policy",
        policy_version_after="cabinet-policy-epoch-1",
        checkpoint_role="success_path_epoch",
        completed_inner_epochs=1,
        success_path_progress=epoch_one_progress,
        success_path_exact_identity=exact_identity,
    )
    saved_resume = torch.load(
        resume_path, map_location="cpu", weights_only=True
    )
    legacy_resume = deepcopy(saved_resume)
    legacy_resume.pop("success_path_exact_identity")
    legacy_resume.pop("success_path_exact_identity_hash")
    with pytest.raises(ValueError, match="legacy.*no exact content identity"):
        iterative_flow._validate_success_path_resume_checkpoint(
            legacy_resume,
            config=config,
            expected_task_contract_hash=task_contract_hash,
            expected_behavior_policy_version="cabinet-behavior-policy",
            expected_round_id=0,
            expected_exact_identity=exact_identity,
        )
    wrong_live_contract = deepcopy(saved_resume["adapter_contract"])
    wrong_live_contract["base_parameter_hashes"] = {
        "transformer.weight": "different-live-base"
    }
    with pytest.raises(ValueError, match="live adapter contract mismatch"):
        iterative_flow._validate_success_path_resume_checkpoint(
            saved_resume,
            config=config,
            expected_task_contract_hash=task_contract_hash,
            expected_behavior_policy_version="cabinet-behavior-policy",
            expected_round_id=0,
            expected_exact_identity=exact_identity,
            expected_adapter_contract=wrong_live_contract,
            expected_base_parameter_hashes=(
                wrong_live_contract["base_parameter_hashes"]
            ),
        )
    wrong_direction = deepcopy(saved_resume)
    wrong_direction["optimizer_state_dict"]["param_groups"][0][
        "maximize"
    ] = True
    with pytest.raises(ValueError, match="parameter-group contract mismatch"):
        iterative_flow._validate_success_path_resume_checkpoint(
            wrong_direction,
            config=config,
            expected_task_contract_hash=task_contract_hash,
            expected_behavior_policy_version="cabinet-behavior-policy",
            expected_round_id=0,
            expected_exact_identity=exact_identity,
        )
    incomplete_optimizer = deepcopy(saved_resume)
    incomplete_optimizer["optimizer_state_dict"]["state"].pop(1)
    with pytest.raises(ValueError, match="does not cover its manifest"):
        iterative_flow._validate_success_path_resume_checkpoint(
            incomplete_optimizer,
            config=config,
            expected_task_contract_hash=task_contract_hash,
            expected_behavior_policy_version="cabinet-behavior-policy",
            expected_round_id=0,
            expected_exact_identity=exact_identity,
        )

    trajectories = [
        {
            "task": "put_object_cabinet",
            "seed": seed,
            "round_id": 0,
            "behavior_policy_version": "cabinet-behavior-policy",
            "labels": [{"round_id": 0}],
        }
        for seed in (10003, 10055)
    ]
    observed: dict[str, object] = {}
    runtime_instances: list[FakeRuntime] = []

    def fake_runtime(**kwargs: object) -> FakeRuntime:
        runtime = FakeRuntime(**kwargs)
        runtime_instances.append(runtime)
        return runtime

    def fake_policy_version(runtime: FakeRuntime) -> str:
        if all(
            torch.equal(parameter.detach(), checkpoint_parameters[name])
            for name, parameter in runtime.trainable
        ):
            return "cabinet-policy-epoch-1"
        return "cabinet-policy-final"

    def fake_update(**kwargs: object) -> dict[str, object]:
        observed["update_calls"] = int(observed.get("update_calls", 0)) + 1
        runtime = kwargs["runtime"]
        optimizer = kwargs["optimizer"]
        resume_state = kwargs["success_path_resume_state"]
        callback = kwargs["epoch_checkpoint_callback"]
        assert isinstance(runtime, FakeRuntime)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert isinstance(resume_state, dict)
        assert callable(callback)
        observed["adapter_restored"] = all(
            torch.equal(parameter.detach(), checkpoint_parameters[name])
            for name, parameter in runtime.trainable
        )
        observed["optimizer_steps_restored"] = sorted(
            int(state["step"].item()) for state in optimizer.state.values()
        )
        observed["optimizer_lr"] = optimizer.param_groups[0]["lr"]

        progress = deepcopy(resume_state["success_path_progress"])
        for epoch_id in (2, 3):
            for _ in range(40):
                optimizer.zero_grad(set_to_none=True)
                loss = (
                    (runtime.lora_a - 0.25).square()
                    + (runtime.lora_b + 0.50).square()
                )
                loss.backward()
                optimizer.step()
                progress["steps"].append(
                    {"step_id": len(progress["steps"]), "epoch": epoch_id}
                )
            progress["completed_inner_epochs"] = epoch_id
            progress["calibration_history"].append(
                {"epoch": epoch_id, "loss": 1.5 - 0.1 * epoch_id}
            )
            progress["epoch_metrics"].append(
                {"epoch": epoch_id, "checkpoint": None}
            )
            runtime._coherent_tt_generator_state = (
                torch.Generator().manual_seed(7 + epoch_id).get_state()
            )
            if epoch_id < 3:
                progress["epoch_metrics"][-1]["checkpoint"] = callback(
                    epoch_id,
                    len(progress["steps"]),
                    progress,
                )
        return {
            "round_id": 0,
            "objective": "coherent_tt_consistency",
            "coherent_tt_variant": "success_path_v1",
            "optimizer_steps_this_round": 120,
            "optimizer_steps_this_invocation": 80,
            "_success_path_progress": progress,
        }

    monkeypatch.setattr(iterative_flow, "NativeV0VideoRuntime", fake_runtime)
    monkeypatch.setattr(
        iterative_flow,
        "_load_trajectory_artifacts",
        lambda *_args, **_kwargs: trajectories,
    )
    monkeypatch.setattr(iterative_flow, "_policy_version", fake_policy_version)
    monkeypatch.setattr(iterative_flow, "_update_round", fake_update)
    monkeypatch.setattr(
        iterative_flow,
        "_configure_cuda_memory_limit",
        lambda _config: None,
    )

    for identity_path, original_content in (
        (train_artifact, b"cabinet-train-artifact-v1"),
        (student / "transformer.bin", b"released-cabinet-student"),
        (teacher, b"frozen-cabinet-teacher"),
    ):
        identity_path.write_bytes(original_content + b"-tampered")
        with pytest.raises(ValueError, match="exact input identity mismatch"):
            iterative_flow._run_trajectory_update(
                config=config,
                student=student,
                teacher=teacher,
                output_dir=output_dir,
                task_contract_hash=task_contract_hash,
            )
        identity_path.write_bytes(original_content)
    partial_temp = (
        output_dir
        / "epoch_checkpoints"
        / ".checkpoint_epoch_02.pt.atomic-999999999-interrupted.tmp"
    )
    partial_temp.write_bytes(b"partial checkpoint")

    summary = iterative_flow._run_trajectory_update(
        config=config,
        student=student,
        teacher=teacher,
        output_dir=output_dir,
        task_contract_hash=task_contract_hash,
    )

    assert observed["adapter_restored"] is True
    assert observed["optimizer_steps_restored"] == [40, 40]
    assert observed["optimizer_lr"] == pytest.approx(config["learning_rate"])
    assert summary["fresh_optimizer"] is False
    assert summary["resumed_optimizer_state"] is True
    assert summary["initial_checkpoint"] == str(resume_path)
    assert summary["starting_global_optimizer_step"] == 40
    assert summary["optimizer_steps_this_run"] == 80
    assert summary["global_optimizer_step"] == 120
    assert "_success_path_progress" not in summary["update"]
    assert runtime_instances[-1].closed is True
    assert not partial_temp.exists()
    assert summary["success_path_commit_schema"] == (
        iterative_flow.SUCCESS_PATH_COMMIT_SCHEMA
    )
    assert summary["success_path_exact_identity"] == exact_identity
    assert summary["success_path_finalization_schema"] == (
        iterative_flow.SUCCESS_PATH_FINALIZATION_SCHEMA
    )
    assert summary["commit_recovered_from_final_checkpoint"] is False
    assert observed["update_calls"] == 1

    final_checkpoint = torch.load(
        summary["checkpoint"], map_location="cpu", weights_only=True
    )
    assert final_checkpoint["checkpoint_role"] == "success_path_final"
    assert final_checkpoint["completed_inner_epochs"] == 3
    assert final_checkpoint["global_optimizer_step"] == 120
    assert len(final_checkpoint["success_path_progress"]["steps"]) == 120
    assert not (
        output_dir / "epoch_checkpoints" / "checkpoint_epoch_03.pt"
    ).exists()
    checkpoint_sha256, _ = iterative_flow._sha256_file(
        Path(summary["checkpoint"])
    )
    assert summary["checkpoint_sha256"] == checkpoint_sha256
    finalization = final_checkpoint["success_path_finalization"]
    assert finalization["success_path_finalization_schema"] == (
        iterative_flow.SUCCESS_PATH_FINALIZATION_SCHEMA
    )
    assert "checkpoint_sha256" not in finalization
    assert final_checkpoint["success_path_finalization_hash"] == (
        iterative_flow._stable_hash(finalization)
    )
    legacy_final = deepcopy(final_checkpoint)
    legacy_final.pop("success_path_finalization")
    legacy_final.pop("success_path_finalization_hash")
    with pytest.raises(ValueError, match="legacy.*no atomic finalization"):
        iterative_flow._validate_success_path_resume_checkpoint(
            legacy_final,
            config=config,
            expected_task_contract_hash=task_contract_hash,
            expected_behavior_policy_version="cabinet-behavior-policy",
            expected_round_id=0,
            expected_exact_identity=exact_identity,
            expected_checkpoint_role="success_path_final",
        )
    wrong_finalization = deepcopy(final_checkpoint)
    wrong_finalization["success_path_finalization"][
        "global_optimizer_step"
    ] = 121
    wrong_finalization["success_path_finalization_hash"] = (
        iterative_flow._stable_hash(
            wrong_finalization["success_path_finalization"]
        )
    )
    with pytest.raises(RuntimeError, match="summary/checkpoint contract mismatch"):
        iterative_flow._validate_success_path_finalization_payload(
            checkpoint=wrong_finalization,
            checkpoint_path=Path(summary["checkpoint"]),
            config=config,
            exact_identity=exact_identity,
            task_contract_hash=task_contract_hash,
            behavior_policy_version="cabinet-behavior-policy",
            round_id=0,
        )
    final_checkpoint_bytes = Path(summary["checkpoint"]).read_bytes()

    with pytest.raises(FileExistsError, match="already complete"):
        iterative_flow._run_trajectory_update(
            config=config,
            student=student,
            teacher=teacher,
            output_dir=output_dir,
            task_contract_hash=task_contract_hash,
        )
    assert observed["update_calls"] == 1

    (output_dir / "summary.json").unlink()
    resume_path.unlink()
    recovered_summary = iterative_flow._run_trajectory_update(
        config=config,
        student=student,
        teacher=teacher,
        output_dir=output_dir,
        task_contract_hash=task_contract_hash,
    )

    assert recovered_summary["commit_recovered_from_final_checkpoint"] is True
    assert recovered_summary["checkpoint_sha256"] == checkpoint_sha256
    assert observed["update_calls"] == 1
    assert Path(summary["checkpoint"]).read_bytes() == final_checkpoint_bytes
    assert json.loads((output_dir / "summary.json").read_text()) == (
        recovered_summary
    )
    assert not (output_dir / ".success_path_writer.lock").exists()
