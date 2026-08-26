"""Focused correctness tests for independently gated video/action LoRA banks."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from experiments import dual_mode_lora as dual_lora
from experiments.dual_mode_lora import (
    DualModeLoRALinear,
    attach_dual_mode_lora,
    dual_mode_lora_active_mode,
    dual_mode_lora_contract,
    dual_mode_lora_gate_counts,
    dual_mode_lora_named_parameters,
    dual_mode_lora_scope,
    dual_mode_lora_state_dict,
    load_dual_mode_lora_checkpoint,
    load_dual_mode_lora_state_dict,
    select_dual_mode_lora_trainable_bank,
    validate_dual_mode_lora_contract,
)


class _Attention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(width, width)
        self.to_k = nn.Linear(width, width)
        self.to_v = nn.Linear(width, width)
        self.to_out = nn.Sequential(nn.Linear(width, width))


class _Projection(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width)


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn1 = _Attention(width)
        self.attn2 = _Attention(width)
        self.ffn = nn.Module()
        self.ffn.net = nn.ModuleList(
            [_Projection(width), nn.Identity(), nn.Linear(width, width)]
        )


class _Transformer(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(width)])


class DualModeLoRALinearTest(unittest.TestCase):
    def test_zero_up_initialization_is_exact_base_in_every_mode(self) -> None:
        torch.manual_seed(7)
        base = nn.Linear(4, 3)
        wrapper = DualModeLoRALinear(base, rank=2, alpha=2.0, dropout=0.0)
        inputs = torch.randn(5, 4)
        expected = base(inputs)

        self.assertTrue(torch.equal(wrapper(inputs), expected))
        with dual_mode_lora_scope("video"):
            self.assertTrue(torch.equal(wrapper(inputs), expected))
        with dual_mode_lora_scope("action"):
            self.assertTrue(torch.equal(wrapper(inputs), expected))

    def test_forward_and_gradient_activation_are_strictly_isolated(self) -> None:
        base = nn.Linear(4, 3)
        wrapper = DualModeLoRALinear(base, rank=2, alpha=2.0, dropout=0.0)
        inputs = torch.ones(5, 4)
        with torch.no_grad():
            wrapper.video_lora_A.fill_(1.0)
            wrapper.action_lora_A.fill_(2.0)

        with dual_mode_lora_scope("video"):
            wrapper(inputs).sum().backward()
        self.assertIsNotNone(wrapper.video_lora_A.grad)
        self.assertIsNotNone(wrapper.video_lora_B.grad)
        self.assertGreater(
            int(torch.count_nonzero(wrapper.video_lora_B.grad).item()), 0
        )
        self.assertIsNone(wrapper.action_lora_A.grad)
        self.assertIsNone(wrapper.action_lora_B.grad)
        self.assertIsNone(wrapper.base.weight.grad)

        wrapper.zero_grad(set_to_none=True)
        with dual_mode_lora_scope("action"):
            wrapper(inputs).sum().backward()
        self.assertIsNone(wrapper.video_lora_A.grad)
        self.assertIsNone(wrapper.video_lora_B.grad)
        self.assertIsNotNone(wrapper.action_lora_A.grad)
        self.assertIsNotNone(wrapper.action_lora_B.grad)
        self.assertGreater(
            int(torch.count_nonzero(wrapper.action_lora_B.grad).item()), 0
        )
        self.assertIsNone(wrapper.base.weight.grad)

    def test_scope_is_nested_exception_safe_and_rejects_unknown_modes(self) -> None:
        self.assertEqual(dual_mode_lora_active_mode(), "none")
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with dual_mode_lora_scope("video"):
                self.assertEqual(dual_mode_lora_active_mode(), "video")
                with dual_mode_lora_scope("action"):
                    self.assertEqual(dual_mode_lora_active_mode(), "action")
                    raise RuntimeError("sentinel")
        self.assertEqual(dual_mode_lora_active_mode(), "none")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            with dual_mode_lora_scope("joint"):
                pass


class DualModeLoRAAttachmentTest(unittest.TestCase):
    def test_each_bank_matches_single_bank_capacity_and_state_is_strict(self) -> None:
        transformer = _Transformer(width=4)
        info = attach_dual_mode_lora(
            transformer,
            rank=2,
            alpha=2.0,
            dropout=0.0,
            block_indices=(0,),
        )
        video = dual_mode_lora_named_parameters(transformer, bank="video")
        action = dual_mode_lora_named_parameters(transformer, bank="action")
        both = dual_mode_lora_named_parameters(transformer, bank="both")

        # One block has 10 target Linear modules; rank * (in + out) = 16.
        self.assertEqual(sum(parameter.numel() for _, parameter in video), 160)
        self.assertEqual(sum(parameter.numel() for _, parameter in action), 160)
        self.assertEqual(sum(parameter.numel() for _, parameter in both), 320)
        self.assertEqual(info.trainable_parameter_count_per_bank, 160)
        self.assertEqual(info.trainable_parameter_count, 320)
        self.assertEqual(info.parameter_dtype, "torch.float32")
        self.assertTrue(
            all(parameter.dtype == torch.float32 for _, parameter in both)
        )
        self.assertEqual(len(video), 20)
        self.assertEqual(len(action), 20)
        self.assertTrue(all("video_lora" in name for name, _ in video))
        self.assertTrue(all("action_lora" in name for name, _ in action))
        self.assertFalse(any(parameter.requires_grad for parameter in (
            module.base.weight
            for module in transformer.modules()
            if isinstance(module, DualModeLoRALinear)
        )))

        state = dual_mode_lora_state_dict(transformer)
        with torch.no_grad():
            for _, parameter in both:
                parameter.add_(1.0)
        load_dual_mode_lora_state_dict(transformer, state)
        restored = dual_mode_lora_state_dict(transformer)
        self.assertEqual(set(restored), set(state))
        self.assertTrue(
            all(torch.equal(restored[name], state[name]) for name in state)
        )

        missing = dict(state)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, "key mismatch"):
            load_dual_mode_lora_state_dict(transformer, missing)
        unexpected = dict(state)
        unexpected["unexpected.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "key mismatch"):
            load_dual_mode_lora_state_dict(transformer, unexpected)
        with self.assertRaisesRegex(ValueError, "already attached"):
            attach_dual_mode_lora(transformer, block_indices=(0,))

        contract = dual_mode_lora_contract(info)
        validate_dual_mode_lora_contract(info, contract)
        wrong_contract = dict(contract)
        wrong_contract["alpha"] = 999.0
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            validate_dual_mode_lora_contract(info, wrong_contract)
        payload = {
            "adapter_kind": "dual_lora",
            "adapter_contract": contract,
            "adapter_state_dict": state,
        }
        load_dual_mode_lora_checkpoint(transformer, info, payload)
        wrong_payload = dict(payload)
        wrong_payload["adapter_contract"] = wrong_contract
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            load_dual_mode_lora_checkpoint(transformer, info, wrong_payload)

    def test_single_bank_state_round_trips_without_touching_other_bank(self) -> None:
        transformer = _Transformer(width=4)
        attach_dual_mode_lora(
            transformer,
            rank=2,
            alpha=2.0,
            dropout=0.0,
            block_indices=(0,),
        )
        video_state = dual_mode_lora_state_dict(transformer, bank="video")
        action_before = dual_mode_lora_state_dict(transformer, bank="action")
        with torch.no_grad():
            for _, parameter in dual_mode_lora_named_parameters(
                transformer, bank="video"
            ):
                parameter.add_(1.0)
        load_dual_mode_lora_state_dict(
            transformer, video_state, bank="video"
        )
        self.assertTrue(
            all(
                torch.equal(value, video_state[name])
                for name, value in dual_mode_lora_state_dict(
                    transformer, bank="video"
                ).items()
            )
        )

        selected_video = select_dual_mode_lora_trainable_bank(
            transformer, bank="video"
        )
        self.assertEqual(
            {name for name, _ in selected_video},
            {
                name
                for name, _ in dual_mode_lora_named_parameters(
                    transformer, bank="video"
                )
            },
        )
        self.assertTrue(all(parameter.requires_grad for _, parameter in selected_video))
        self.assertFalse(
            any(
                parameter.requires_grad
                for _, parameter in dual_mode_lora_named_parameters(
                    transformer, bank="action"
                )
            )
        )
        selected_action = select_dual_mode_lora_trainable_bank(
            transformer, bank="action"
        )
        self.assertTrue(all(parameter.requires_grad for _, parameter in selected_action))
        self.assertFalse(
            any(
                parameter.requires_grad
                for _, parameter in dual_mode_lora_named_parameters(
                    transformer, bank="video"
                )
            )
        )
        self.assertTrue(
            all(
                torch.equal(value, action_before[name])
                for name, value in dual_mode_lora_state_dict(
                    transformer, bank="action"
                ).items()
            )
        )

    def test_attach_validation_and_mid_attach_failure_leave_model_unchanged(self) -> None:
        empty_target = _Transformer(width=4)
        flags_before = [
            parameter.requires_grad for parameter in empty_target.parameters()
        ]
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            attach_dual_mode_lora(empty_target, block_indices=())
        self.assertEqual(
            [parameter.requires_grad for parameter in empty_target.parameters()],
            flags_before,
        )
        self.assertFalse(
            any(
                isinstance(module, DualModeLoRALinear)
                for module in empty_target.modules()
            )
        )

        transformer = _Transformer(width=4)
        original_targets = {
            name: module
            for name, module in transformer.named_modules()
            if isinstance(module, nn.Linear)
        }
        original_flags = {
            name: parameter.requires_grad
            for name, parameter in transformer.named_parameters()
        }
        original_set_child = dual_lora._set_child
        calls = 0

        def fail_before_second_set(
            parent: nn.Module, leaf: str, value: nn.Module
        ) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic attach failure")
            original_set_child(parent, leaf, value)

        with patch.object(dual_lora, "_set_child", fail_before_second_set):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                attach_dual_mode_lora(
                    transformer,
                    rank=2,
                    alpha=2.0,
                    dropout=0.0,
                    block_indices=(0,),
                )
        restored_targets = {
            name: module
            for name, module in transformer.named_modules()
            if isinstance(module, nn.Linear)
        }
        self.assertEqual(set(restored_targets), set(original_targets))
        self.assertTrue(
            all(restored_targets[name] is original_targets[name] for name in original_targets)
        )
        self.assertEqual(
            {
                name: parameter.requires_grad
                for name, parameter in transformer.named_parameters()
            },
            original_flags,
        )

    def test_load_validation_is_atomic_and_checks_converted_values(self) -> None:
        transformer = _Transformer(width=4).half()
        attach_dual_mode_lora(
            transformer,
            rank=2,
            alpha=2.0,
            dropout=0.0,
            block_indices=(0,),
        )
        before = dual_mode_lora_state_dict(transformer)
        invalid = {name: value.double() for name, value in before.items()}
        invalid[next(reversed(invalid))].fill_(1e100)
        with self.assertRaisesRegex(ValueError, "after dtype conversion"):
            load_dual_mode_lora_state_dict(transformer, invalid)
        after = dual_mode_lora_state_dict(transformer)
        self.assertTrue(
            all(torch.equal(after[name], before[name]) for name in before)
        )

    def test_gate_counters_distinguish_all_three_paths(self) -> None:
        transformer = _Transformer(width=4)
        attach_dual_mode_lora(
            transformer,
            rank=2,
            alpha=2.0,
            dropout=0.0,
            block_indices=(0,),
        )
        target = transformer.blocks[0].attn1.to_q
        self.assertIsInstance(target, DualModeLoRALinear)
        inputs = torch.ones(1, 4)
        target(inputs)
        with dual_mode_lora_scope("video"):
            target(inputs)
        with dual_mode_lora_scope("action"):
            target(inputs)
        self.assertEqual(
            dual_mode_lora_gate_counts(transformer),
            {
                "total_forward_calls": 3,
                "video_forward_calls": 1,
                "action_forward_calls": 1,
                "bypass_forward_calls": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
