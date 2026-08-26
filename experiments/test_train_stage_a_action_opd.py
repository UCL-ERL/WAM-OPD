"""Focused CPU/static tests for the config-driven Stage-A trainer."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from experiments.action_output_adapter import (
    action_output_adapter_state_dict,
    attach_action_output_adapter,
)
from experiments.goal1_exact_condition import ConditionContractError
from experiments.train_stage_a_action_opd import (
    CONFIG_SCHEMA,
    TEACHER_ARTIFACT_KIND,
    TEACHER_TARGET,
    StageAConfig,
    epoch_batches,
    evaluate_stage_a,
    load_stage_a_config,
    stage_a_endpoint_loss,
    terminal_execution_mask,
    validate_stage_a_replays,
)
from experiments.waopd_native_action_opd import NativeActionEndpointTrainer, load_labels
from experiments.waopd_native_student_only import NativeStudentOnlyRuntime


def _config_payload(artifact: Path, output: Path) -> dict:
    return {
        "schema": CONFIG_SCHEMA,
        "task_id": "open_microwave",
        "task_config": "demo_clean",
        "artifacts": [str(artifact)],
        "output": str(output),
        "adapter": {
            "kind": "action_output_residual",
            "target": "action_proj_out",
            "rank": 32,
            "initialization": "zero_up",
        },
        "loss": {
            "target_key": "teacher_bridge_model_action",
            "target_kind": TEACHER_TARGET,
            "retention_weight": 0.0,
        },
        "training": {
            "epochs": 3,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "gradient_clip": 2.0,
            "seed": 17,
            "device": "cpu",
        },
    }


def _stage_config(artifact_count: int = 1) -> StageAConfig:
    return StageAConfig(
        config_path=Path("/config.json"),
        task_id="open_microwave",
        task_config="demo_clean",
        artifacts=tuple(Path(f"/{index}.pt") for index in range(artifact_count)),
        output=Path("/delta.pt"),
        batch_size=2,
        learning_rate=1e-4,
        weight_decay=0.0,
        gradient_clip=2.0,
        seed=17,
        device="cpu",
    )


def _replay(
    *,
    frame_st_id: int = 0,
    terminal: bool = True,
    terminal_position: list[int] | None = None,
    executed: list[list[bool]] | None = None,
) -> dict:
    if terminal_position is None and terminal:
        terminal_position = [1, 4]
    if executed is None:
        if terminal:
            executed = [[False] * 16, [True] * 5 + [False] * 11]
        else:
            executed = [[True] * 16, [True] * 16]
    valid = torch.zeros((1, 30, 2, 16, 1), dtype=torch.bool)
    valid[:, :16, :, :, :] = True
    if frame_st_id == 0:
        valid[:, :, 0:1] = False
    return {
        "_artifact_path": f"/label-{frame_st_id}.pt",
        "task_id": "open_microwave",
        "task_config": "demo_clean",
        "schema_version": 4,
        "artifact_kind": TEACHER_ARTIFACT_KIND,
        "teacher_target_kind": TEACHER_TARGET,
        "collection_id": "collection-1",
        "frame_st_id": frame_st_id,
        "teacher_bridge_model_action": torch.ones((1, 30, 2, 16, 1)),
        "valid_action_mask": valid,
        "start_frame": 1 if frame_st_id == 0 else 0,
        "action_steps": sum(sum(row) for row in executed),
        "executed_action_mask": executed,
        "terminal_reached": terminal,
        "terminal_action_position": terminal_position,
    }


class StageAConfigTest(unittest.TestCase):
    def test_accepts_only_fixed_stage_a_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = root / "label.pt"
            artifact.touch()
            config_path = root / "config.json"
            payload = _config_payload(artifact, root / "delta.pt")
            config_path.write_text(json.dumps(payload))

            config = load_stage_a_config(config_path)

            self.assertEqual(config.task_id, "open_microwave")
            self.assertEqual(config.task_config, "demo_clean")
            self.assertEqual(config.batch_size, 2)

            for mutation, message in (
                (("task_config", "demo_randomized"), "task_config"),
                (("adapter.rank", 8), "rank-32 zero-up"),
                (("loss.retention_weight", 1.0), "forbids retention"),
                (("training.epochs", 2), "exactly 3 epochs|fixed at 3"),
            ):
                changed = json.loads(json.dumps(payload))
                dotted, value = mutation
                target = changed
                parts = dotted.split(".")
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
                config_path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, message):
                    load_stage_a_config(config_path)


class StageATerminalMaskTest(unittest.TestCase):
    def test_keeps_trigger_action_and_masks_every_later_position(self) -> None:
        replay = _replay()
        endpoint = torch.zeros(
            (1, 30, 2, 16, 1),
            dtype=torch.float32,
            requires_grad=True,
        )

        loss, _rmse, count = stage_a_endpoint_loss(endpoint, replay)
        loss.backward()

        self.assertEqual(count, 16 * 5)
        self.assertGreater(float(endpoint.grad[:, :16, 1, 4].abs().sum()), 0.0)
        self.assertEqual(float(endpoint.grad[:, :16, 1, 5:].abs().sum()), 0.0)
        self.assertEqual(float(endpoint.grad[:, 16:].abs().sum()), 0.0)

    def test_rejects_mask_terminal_and_action_step_disagreement(self) -> None:
        replay = _replay()
        replay["executed_action_mask"][1][4] = False
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            terminal_execution_mask(replay, frame_count=2, horizon=16)

        replay = _replay()
        replay["action_steps"] = 4
        with self.assertRaisesRegex(ValueError, "action_steps"):
            terminal_execution_mask(replay, frame_count=2, horizon=16)

        already_successful = _replay(
            terminal=True,
            terminal_position=None,
            executed=[[False] * 16, [False] * 16],
        )
        already_successful["terminal_action_position"] = None
        already_successful["action_steps"] = 0
        with self.assertRaisesRegex(ValueError, "terminal_action_position|no executed"):
            terminal_execution_mask(already_successful, frame_count=2, horizon=16)

    def test_accepts_only_the_real_prefix_at_global_control_horizon(self) -> None:
        replay = _replay(
            terminal=False,
            terminal_position=None,
            executed=[[True] * 12 + [False] * 4, [False] * 16],
        )
        replay["start_frame"] = 0
        replay["action_steps"] = 12
        replay["horizon_reached"] = True

        mask = terminal_execution_mask(replay, frame_count=2, horizon=16)

        self.assertEqual(int(mask.sum()), 12)
        self.assertTrue(bool(mask[0, :12].all()))
        self.assertFalse(bool(mask[0, 12:].any()))
        self.assertFalse(bool(mask[1].any()))


class StageAEvalModeTest(unittest.TestCase):
    def test_offline_eval_leaves_frozen_transformer_in_eval_mode(self) -> None:
        transformer = torch.nn.Linear(1, 1)
        transformer.train()

        class FakeTrainer:
            server = SimpleNamespace(transformer=transformer)

            @staticmethod
            def endpoint(_replay: dict, *, require_grad: bool):
                self.assertFalse(require_grad)
                return torch.zeros((1, 30, 2, 16, 1)), {}

        evaluate_stage_a(FakeTrainer(), [_replay()])

        self.assertFalse(transformer.training)


class StageAReplaySetTest(unittest.TestCase):
    def test_accepts_native_schema_v4_and_rejects_legacy_label_kind(self) -> None:
        validate_stage_a_replays([_replay()], _stage_config())

        legacy = _replay()
        legacy["artifact_kind"] = "stage_g_teacher_bridge_label"
        with self.assertRaisesRegex(ValueError, "artifact_kind"):
            validate_stage_a_replays([legacy], _stage_config())

    def test_fails_closed_on_mixed_split_task_and_post_success_context(self) -> None:
        config = _stage_config(2)
        first = _replay(frame_st_id=0)
        later = _replay(
            frame_st_id=2,
            terminal=False,
            terminal_position=None,
            executed=[[True] * 16, [True] * 16],
        )
        with self.assertRaisesRegex(ValueError, "post-success"):
            validate_stage_a_replays([first, later], config)

        mixed_task = _replay()
        mixed_task["task_id"] = "move_stapler_pad"
        with self.assertRaisesRegex(ValueError, "task_id"):
            validate_stage_a_replays([mixed_task], _stage_config())

        randomized = _replay()
        randomized["task_config"] = "demo_randomized"
        with self.assertRaisesRegex(ValueError, "task_config"):
            validate_stage_a_replays([randomized], _stage_config())

    def test_three_epochs_cover_every_replay_once_per_epoch(self) -> None:
        replays = [{"id": index} for index in range(5)]
        batches = list(epoch_batches(replays, batch_size=2, seed=3))

        self.assertEqual(len(batches), 9)
        for epoch in range(3):
            ids = [
                row["id"]
                for batch_epoch, _batch_index, batch in batches
                if batch_epoch == epoch
                for row in batch
            ]
            self.assertEqual(sorted(ids), list(range(5)))


class NativeLabelLoaderTest(unittest.TestCase):
    def test_loader_intentionally_accepts_only_persistent_schema_v4(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "label.pt"
            payload = {
                "schema_version": 4,
                "artifact_kind": TEACHER_ARTIFACT_KIND,
                "source_runtime": "native_lingbot_not_rlinf_adaptation",
                "training_started": False,
                "canonical_action_context": {},
                "condition_fingerprint": {},
                "student_model_action": torch.zeros(1),
                "teacher_bridge_model_action": torch.zeros(1),
                "valid_action_mask": torch.ones(1, dtype=torch.bool),
                "replay_context_path": "/unused-in-loader.pt",
                "student_checkpoint": "/student.pt",
            }
            torch.save(payload, path)

            rows = load_labels([path])

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["_artifact_path"], str(path.resolve()))

            payload["schema_version"] = 3
            torch.save(payload, path)
            with self.assertRaisesRegex(ConditionContractError, "schema"):
                load_labels([path])


class NativeReplayContextMetadataTest(unittest.TestCase):
    def test_rejects_clean_label_pointing_to_randomized_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            context_path = Path(temp_dir) / "context.pt"
            torch.save(
                {
                    "schema": "waopd_d3_student_occupancy_context_v1",
                    "task": "open_microwave",
                    "task_config": "demo_randomized",
                    "seed": 10000,
                    "prompt": "open",
                    "collection_id": "collection-1",
                    "round_index": 0,
                    "context_chunks_used": 0,
                    "frame_st_id": 0,
                    "initial_observation": {},
                    "chunks": [],
                },
                context_path,
            )
            artifact = {
                "replay_context_path": str(context_path),
                "task_id": "open_microwave",
                "task_config": "demo_clean",
                "env_seed": 10000,
                "prompt": "open",
                "collection_id": "collection-1",
                "round_index": 0,
                "context_chunks_used": 0,
                "frame_st_id": 0,
            }
            trainer = NativeActionEndpointTrainer.__new__(
                NativeActionEndpointTrainer
            )

            with self.assertRaisesRegex(
                ConditionContractError, "task_config"
            ):
                trainer._rebuild_condition(artifact)


class NativeCheckpointCompatibilityTest(unittest.TestCase):
    def test_native_loader_consumes_dual_compatible_checkpoint(self) -> None:
        class TinyTransformer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.action_proj_out = torch.nn.Linear(4, 3)

        trained = TinyTransformer()
        attach_action_output_adapter(trained, rank=2, initialization="zero_up")
        trained.action_proj_out.up.weight.data.fill_(0.25)
        state = action_output_adapter_state_dict(trained)
        checkpoint = {
            "format": "flashwam_action_output_adapter_v1",
            "adapter": {
                "kind": "action_output_residual",
                "rank": 2,
                "initialization": "zero_up",
            },
            "state_dict": state,
            "adapter_state_dict": state,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "delta.pt"
            torch.save(checkpoint, path)
            runtime = NativeStudentOnlyRuntime.__new__(NativeStudentOnlyRuntime)
            runtime.server = SimpleNamespace(transformer=TinyTransformer())

            runtime.load_action_adapter(path, rank=2)

        loaded = action_output_adapter_state_dict(runtime.server.transformer)
        self.assertEqual(set(loaded), set(state))
        for name, tensor in state.items():
            self.assertTrue(torch.equal(loaded[name], tensor))


if __name__ == "__main__":
    unittest.main()
