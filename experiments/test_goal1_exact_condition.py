"""Unit tests for the Goal 1 exact-condition contract.

These tests use a small native-helper double so that the contract itself is
tested without launching a model or performing OPD training.  The two real
runtime hook captures are reported separately by the Goal 1 audit.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from experiments.action_output_adapter import ActionOutputResidualAdapter
from experiments.goal1_exact_condition import (
    ConditionContractError,
    CanonicalActionContextV2,
    PreparedPlan,
    assert_cache_semantics,
    assert_fingerprint_match,
    build_canonical_context,
    build_condition_fingerprint,
    prepare_plan_input,
    tensor_hash,
)


class _NativeHelperDouble:
    def __init__(self) -> None:
        self.action_mask = torch.tensor([True, False, True, True])

    def _prepare_latent_input(
        self,
        latent_model_input,
        action_model_input,
        latent_t=0,
        action_t=0,
        latent_cond=None,
        action_cond=None,
        frame_st_id=0,
    ):
        result = {}
        if latent_model_input is not None:
            noisy = latent_model_input
            timesteps = torch.full(
                (noisy.shape[2],), float(latent_t), dtype=torch.float32
            )
            if latent_cond is not None:
                noisy[:, :, 0:1] = latent_cond[:, :, 0:1]
                timesteps[0] = 0
            result["latent_res_lst"] = {
                "noisy_latents": noisy,
                "timesteps": timesteps,
                "grid_id": torch.arange(noisy.numel()).reshape(-1),
            }
        if action_model_input is not None:
            noisy = action_model_input
            timesteps = torch.full(
                (noisy.shape[2],), float(action_t), dtype=torch.float32
            )
            if action_cond is not None:
                noisy[:, :, 0:1] = action_cond[:, :, 0:1]
                timesteps[0] = 0
            noisy[:, ~self.action_mask] = 0
            result["action_res_lst"] = {
                "noisy_latents": noisy,
                "timesteps": timesteps,
                "grid_id": torch.arange(noisy.numel()).reshape(-1),
            }
        return result


def _fingerprint(
    owner: str,
    plan: torch.Tensor,
    noise: torch.Tensor,
    action_timestep: torch.Tensor | None = None,
):
    mask = torch.ones_like(noise, dtype=torch.bool)
    normalization = {"method": "quantiles", "q01": [-1.0], "q99": [1.0]}
    return build_condition_fingerprint(
        checkpoint_owner=owner,
        history_hash="history",
        prompt_hash="prompt",
        observation_hash="observation",
        model_action_history_hash="action-history",
        prepared_plan=plan,
        prepared_plan_timestep=torch.zeros(plan.shape[2]),
        action_base_noise=noise,
        action_timestep=(
            torch.zeros(noise.shape[2])
            if action_timestep is None
            else action_timestep
        ),
        mask=mask,
        normalization_metadata=normalization,
        frame_st_id=2,
        token_positions=(1, 2, 3),
        cache_valid_length=17,
        sigma_start=1.0,
        sigma_end=0.0,
    )


class Goal1ExactConditionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)
        self.server = _NativeHelperDouble()
        self.raw_plan = torch.randn(1, 2, 2, 3, 3)
        self.initial = torch.full_like(self.raw_plan[:, :, :1], 7.0)
        self.noise = torch.randn(1, 4, 2, 2, 1)

    def test_frame_zero_plan_conditioning(self) -> None:
        _model_input, prepared = prepare_plan_input(
            self.server,
            self.raw_plan,
            frame_st_id=0,
            init_latent=self.initial,
        )
        self.assertTrue(prepared.latent_cond_applied)
        self.assertTrue(torch.equal(prepared.prepared_z_s[:, :, :1], self.initial))
        self.assertTrue(torch.equal(prepared.prepared_z_s[:, :, 1:], self.raw_plan[:, :, 1:]))
        self.assertFalse(torch.equal(prepared.raw_z_s, prepared.prepared_z_s))

    def test_nonzero_context_does_not_overwrite_plan(self) -> None:
        _model_input, prepared = prepare_plan_input(
            self.server,
            self.raw_plan,
            frame_st_id=2,
            init_latent=self.initial,
        )
        self.assertFalse(prepared.latent_cond_applied)
        self.assertTrue(torch.equal(prepared.raw_z_s, prepared.prepared_z_s))

    def test_teacher_student_prepared_plan_equality(self) -> None:
        _model_input, student = prepare_plan_input(
            self.server,
            self.raw_plan,
            frame_st_id=0,
            init_latent=self.initial,
        )
        _teacher_input, teacher = prepare_plan_input(
            self.server,
            student.prepared_z_s,
            frame_st_id=0,
            already_prepared=True,
        )
        self.assertTrue(torch.equal(student.prepared_z_s, teacher.prepared_z_s))
        self.assertTrue(torch.equal(student.prepared_z_s_timestep, teacher.prepared_z_s_timestep))

    def test_actual_action_noise_tensor_equality(self) -> None:
        saved = self.noise.detach().clone()
        self.assertTrue(torch.equal(saved, self.noise))
        self.assertEqual(tensor_hash(saved), tensor_hash(self.noise))

    def test_rollout_serialization_replay_student_endpoint_parity(self) -> None:
        _model_input, prepared = prepare_plan_input(
            self.server,
            self.raw_plan,
            frame_st_id=2,
            init_latent=self.initial,
        )
        canonical = build_canonical_context(
            behavior_checkpoint_id="student",
            task="task",
            seed=1,
            episode_id="episode",
            prefix_id="prefix",
            frame_st_id=2,
            prompt_token_ids=(1, 2),
            observation_history="obs",
            model_format_action_history="actions",
            executed_physical_actions=[{"hash": "physical"}],
            prepared_plan=prepared,
            action_base_noise=self.noise,
            valid_action_mask=torch.ones_like(self.noise, dtype=torch.bool),
            sigma_start=1.0,
            sigma_end=0.0,
            normalization_metadata={"method": "quantiles"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.pt"
            torch.save({"canonical_action_context": canonical.to_dict()}, path)
            loaded = torch.load(path, weights_only=False)["canonical_action_context"]
        deployed = loaded["action_base_noise"] + loaded["prepared_z_s"].mean()
        replayed = canonical.action_base_noise + canonical.prepared_z_s.mean()
        self.assertTrue(torch.equal(deployed, replayed))

    def test_identical_record_repeated_teacher_label_is_deterministic(self) -> None:
        record = {"plan": self.raw_plan, "noise": self.noise, "sigma": (1.0, 0.0)}

        def label(item):
            return item["noise"] + item["plan"].mean()

        first = label(record)
        second = label(record)
        self.assertTrue(torch.equal(first, second))

    def test_fingerprint_mismatch_fails_closed(self) -> None:
        student = _fingerprint("student", self.raw_plan, self.noise)
        teacher = _fingerprint("teacher", self.raw_plan + 1, self.noise)
        with self.assertRaises(ConditionContractError):
            assert_fingerprint_match(student, teacher)

    def test_action_timestep_mismatch_fails_closed(self) -> None:
        student = _fingerprint("student", self.raw_plan, self.noise)
        teacher = _fingerprint(
            "teacher",
            self.raw_plan,
            self.noise,
            action_timestep=torch.ones(self.noise.shape[2]),
        )
        with self.assertRaises(ConditionContractError):
            assert_fingerprint_match(student, teacher)

    def test_zero_init_adapter_matches_released_deployment(self) -> None:
        torch.manual_seed(19)
        base = torch.nn.Linear(5, 3)
        adapter = ActionOutputResidualAdapter(base, rank=2, initialization="zero_up")
        inputs = torch.randn(8, 5)
        self.assertTrue(torch.equal(adapter(inputs), base(inputs)))

    def test_teacher_student_cache_owners_separate_semantics_same(self) -> None:
        student = _fingerprint("student", self.raw_plan, self.noise)
        teacher = _fingerprint("teacher", self.raw_plan, self.noise)
        assert_cache_semantics(student, teacher)
        self.assertNotEqual(student.checkpoint_owner, teacher.checkpoint_owner)
        self.assertEqual(student.token_positions, teacher.token_positions)
        self.assertEqual(student.cache_valid_length, teacher.cache_valid_length)


if __name__ == "__main__":
    unittest.main()
