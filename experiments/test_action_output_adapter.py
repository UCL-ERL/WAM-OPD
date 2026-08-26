"""Unit tests for the Stage G shared action-output residual adapter."""

from __future__ import annotations

import unittest

import torch

from experiments.action_output_adapter import (
    ActionOutputResidualAdapter,
    action_output_adapter_state_dict,
    attach_action_output_adapter,
)
from experiments.prototype_flashwam_robotwin_server import (
    apply_transformer_delta,
)


class ActionOutputResidualAdapterTest(unittest.TestCase):
    def test_zero_gate_preserves_released_output_exactly(self) -> None:
        torch.manual_seed(7)
        base = torch.nn.Linear(6, 4)
        adapter = ActionOutputResidualAdapter(base, rank=2)
        inputs = torch.randn(3, 6)

        expected = base(inputs)
        actual = adapter(inputs)

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(adapter.gate, torch.zeros_like(adapter.gate)))

    def test_only_adapter_parameters_train_and_gate_has_gradient(self) -> None:
        torch.manual_seed(11)
        base = torch.nn.Linear(5, 3)
        adapter = ActionOutputResidualAdapter(base, rank=2)
        inputs = torch.randn(4, 5)

        adapter(inputs).square().mean().backward()

        self.assertFalse(adapter.base.weight.requires_grad)
        self.assertFalse(adapter.base.bias.requires_grad)
        self.assertIsNone(adapter.base.weight.grad)
        self.assertIsNone(adapter.base.bias.grad)
        self.assertIsNotNone(adapter.gate.grad)
        self.assertTrue(torch.isfinite(adapter.gate.grad).all())
        self.assertGreater(float(adapter.gate.grad.abs().sum()), 0.0)

    def test_zero_up_initialization_is_exact_and_trains_full_up_projection(self) -> None:
        torch.manual_seed(12)
        base = torch.nn.Linear(5, 3)
        adapter = ActionOutputResidualAdapter(
            base,
            rank=2,
            initialization="zero_up",
        )
        inputs = torch.randn(4, 5)

        expected = base(inputs)
        actual = adapter(inputs)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(adapter.up.weight, torch.zeros_like(adapter.up.weight)))
        self.assertFalse(adapter.gate.requires_grad)

        actual.square().mean().backward()
        self.assertIsNotNone(adapter.up.weight.grad)
        self.assertGreater(float(adapter.up.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(adapter.down.weight.grad)
        self.assertEqual(float(adapter.down.weight.grad.abs().sum()), 0.0)

    def test_attach_and_reload_round_trip(self) -> None:
        class TinyTransformer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj_out = torch.nn.Linear(4, 2)
                self.action_proj_out = torch.nn.Linear(4, 3)

        torch.manual_seed(13)
        first = TinyTransformer()
        original_video_weight = first.proj_out.weight.detach().clone()
        info = attach_action_output_adapter(first, rank=2)
        first.action_proj_out.gate.data.fill_(0.25)
        state = action_output_adapter_state_dict(first)

        torch.manual_seed(13)
        second = TinyTransformer()
        attach_action_output_adapter(second, rank=2)
        second.load_state_dict(state, strict=False)

        inputs = torch.randn(2, 4)
        torch.testing.assert_close(
            first.action_proj_out(inputs),
            second.action_proj_out(inputs),
        )
        self.assertTrue(torch.equal(first.proj_out.weight, original_video_weight))
        self.assertEqual(info.trainable_parameter_count, 17)

    def test_rejects_double_attachment_and_invalid_rank(self) -> None:
        transformer = torch.nn.Module()
        transformer.action_proj_out = torch.nn.Linear(4, 3)
        attach_action_output_adapter(transformer, rank=2)

        with self.assertRaisesRegex(ValueError, "already attached"):
            attach_action_output_adapter(transformer, rank=2)
        with self.assertRaisesRegex(ValueError, "rank must be positive"):
            ActionOutputResidualAdapter(torch.nn.Linear(4, 3), rank=0)

    def test_deployment_loader_attaches_adapter_from_checkpoint(self) -> None:
        class TinyTransformer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.action_proj_out = torch.nn.Linear(4, 3)

        torch.manual_seed(17)
        trained = TinyTransformer()
        attach_action_output_adapter(trained, rank=2)
        trained.action_proj_out.gate.data.fill_(0.5)
        checkpoint = {
            "format": "flashwam_action_output_adapter_v1",
            "adapter": {"kind": "action_output_residual", "rank": 2},
            "state_dict": action_output_adapter_state_dict(trained),
        }

        torch.manual_seed(17)
        deployed = TinyTransformer()
        result = apply_transformer_delta(deployed, checkpoint)

        inputs = torch.randn(2, 4)
        torch.testing.assert_close(
            trained.action_proj_out(inputs),
            deployed.action_proj_out(inputs),
        )
        self.assertEqual(result["format"], checkpoint["format"])
        self.assertEqual(result["tensor_count"], 3)

    def test_deployment_loader_supports_zero_up_initialization(self) -> None:
        class TinyTransformer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.action_proj_out = torch.nn.Linear(4, 3)

        torch.manual_seed(19)
        trained = TinyTransformer()
        attach_action_output_adapter(
            trained,
            rank=2,
            initialization="zero_up",
        )
        trained.action_proj_out.up.weight.data.fill_(0.25)
        checkpoint = {
            "format": "flashwam_action_output_adapter_v1",
            "adapter": {
                "kind": "action_output_residual",
                "rank": 2,
                "initialization": "zero_up",
            },
            "state_dict": action_output_adapter_state_dict(trained),
        }

        torch.manual_seed(19)
        deployed = TinyTransformer()
        result = apply_transformer_delta(deployed, checkpoint)

        inputs = torch.randn(2, 4)
        torch.testing.assert_close(
            trained.action_proj_out(inputs),
            deployed.action_proj_out(inputs),
        )
        self.assertEqual(result["tensor_count"], 2)


if __name__ == "__main__":
    unittest.main()
