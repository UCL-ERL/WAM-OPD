"""Pure protocol tests for the native closed-loop runner."""

from __future__ import annotations

import unittest
from unittest import mock
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from experiments.robotwin_branch_oracle import execute_env_action_chunk_physics_only
from experiments.goal1_exact_condition import PreparedPlan
from experiments.v0l_capture_history import HistoryCapture
from experiments.waopd_native_closed_loop_runner import (
    LockedNoiseBank,
    NativeModelRuntime,
    _initialize_task_local_success_state,
    _derived_seed,
)


class _FakeTask:
    def __init__(self) -> None:
        self.actions = []

    def take_action(self, action, action_type="ee") -> None:
        self.actions.append((np.asarray(action), action_type))


class _FakePose:
    def __init__(self, xyz):
        self.p = np.asarray(xyz, dtype=np.float64)


class _FakeActor:
    def __init__(self, xyz):
        self._pose = _FakePose(xyz)

    def get_pose(self):
        return self._pose


class _FakeInstructionTask:
    def __init__(self, *, x=0.1):
        self.object = _FakeActor([x, 0.0, 0.42])
        self.stapler = _FakeActor([x, 0.0, 0.42])
        self.shoe = _FakeActor([x, 0.0, 0.42])
        self.scanner = _FakeActor([x, 0.0, 0.42])
        self.selected_modelname = "075_bread"
        self.selected_model_id = 2
        self.stapler_id = 5
        self.shoe_id = 6
        self.object_id = 4
        self.scanner_id = 3
        self.color_name = "Black"
        self.bottle_id = [1, 2, 3]
        self.info = {}


class _FakeStreamingCache:
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear_cache(self) -> None:
        self.clear_calls += 1


class _FakeServer:
    def __init__(self) -> None:
        self.streaming_vae = _FakeStreamingCache()
        self.streaming_vae_half = _FakeStreamingCache()
        self.init_latent = None
        self.reset_payloads = []

    def infer(self, payload):
        self.reset_payloads.append(payload)
        return {}

    def _encode_obs(self, payload):
        return torch.ones((1, 48, 1, 24, 20), dtype=torch.float32)


class _FakeCacheTransformer:
    """Small cache-aware transformer double for action-boundary tests."""

    def __init__(self, *, cache_name: str, initial_length: int) -> None:
        self.update_cache_calls = []
        total_length = initial_length + 8
        mask = torch.zeros(total_length, dtype=torch.bool)
        mask[:initial_length] = True
        attention = SimpleNamespace(
            attn_caches={cache_name: {"mask": mask}}
        )
        self.blocks = [SimpleNamespace(attn1=attention)]

    def __call__(self, _model_input, *, update_cache, cache_name, action_mode):
        del action_mode
        self.update_cache_calls.append(int(update_cache))
        if update_cache == 1:
            cache = self.blocks[0].attn1.attn_caches[cache_name]
            slot = cache["mask"].nonzero(as_tuple=False).shape[0]
            cache["mask"][slot] = True
        return torch.zeros((1, 1, 1), dtype=torch.float32)


class _FakeActionServer:
    def __init__(self, *, cache_name: str, initial_length: int) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.action_mask = torch.tensor([True])
        self.job_config = SimpleNamespace(
            action_dim=1,
            action_per_frame=1,
            frame_chunk_size=1,
        )
        self.teacher = None
        self._transformer = _FakeCacheTransformer(
            cache_name=cache_name,
            initial_length=initial_length,
        )

    def _prepare_latent_input(
        self,
        _latent_model_input,
        action_model_input,
        _latent_t=0,
        action_t=0,
        _latent_cond=None,
        _action_cond=None,
        frame_st_id=0,
    ):
        del frame_st_id
        action = action_model_input.clone()
        return {
            "action_res_lst": {
                "noisy_latents": action,
                "timesteps": torch.full(
                    (action.shape[2],), float(action_t), dtype=torch.float32
                ),
                "grid_id": torch.arange(action.numel()),
            }
        }

    def _repeat_input_for_cfg(self, model_input):
        return model_input

    def postprocess_action(self, _action):
        return np.zeros((16, 2, 16), dtype=np.float32)


class _FakeVideoScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.empty(0, dtype=torch.float32)
        self.requested_steps = None
        self.step_calls = 0

    def set_timesteps(self, steps: int) -> None:
        self.requested_steps = int(steps)
        self.timesteps = torch.linspace(1000.0, 200.0, int(steps))

    def step(self, model_output, _timestep, sample):
        self.step_calls += 1
        return sample + model_output


class _FakeVideoTeacher:
    def __init__(self) -> None:
        self.update_cache_calls = []

    def __call__(self, model_input, *, update_cache, cache_name, action_mode):
        del cache_name, action_mode
        self.update_cache_calls.append(int(update_cache))
        latent = model_input["noisy_latents"]
        return torch.ones((2, *latent.shape[1:]), dtype=latent.dtype)


class _FakeVideoServer:
    def __init__(self) -> None:
        self.scheduler = _FakeVideoScheduler()
        self.job_config = SimpleNamespace(
            patch_size=(1, 1, 1),
            frame_chunk_size=1,
            guidance_scale=1.0,
        )
        self.latent_height = 1
        self.latent_width = 1
        self.use_cfg = True

    def _prepare_latent_input(
        self,
        latent_model_input,
        _action_model_input,
        latent_t=0,
        _action_t=0,
        latent_cond=None,
        _action_cond=None,
        frame_st_id=0,
    ):
        del latent_cond, frame_st_id
        return {
            "latent_res_lst": {
                "noisy_latents": latent_model_input.clone(),
                "timesteps": torch.as_tensor([latent_t], dtype=torch.float32),
            }
        }

    def _repeat_input_for_cfg(self, model_input):
        return model_input


def _identity_pose(action: np.ndarray, _initial_pose: np.ndarray) -> np.ndarray:
    return np.asarray(action).copy()


class NativeClosedLoopProtocolTest(unittest.TestCase):
    def test_teacher_solver_configuration_is_explicit_and_validated(self) -> None:
        runtime = object.__new__(NativeModelRuntime)
        runtime.configure_teacher_solver(
            video_steps=5,
            video_exec_steps=3,
            action_steps=10,
        )
        self.assertEqual(runtime.teacher_video_steps, 5)
        self.assertEqual(runtime.teacher_video_exec_steps, 3)
        self.assertEqual(runtime.teacher_action_steps, 10)
        with self.assertRaisesRegex(ValueError, "video exec steps"):
            runtime.configure_teacher_solver(
                video_steps=5,
                video_exec_steps=6,
                action_steps=10,
            )

    def test_teacher_video_uses_five_grid_steps_and_three_executed_steps(self) -> None:
        try:
            import wan_va  # noqa: F401
        except ImportError:
            self.skipTest("requires the external LingBot-VA runtime")
        runtime = object.__new__(NativeModelRuntime)
        runtime.server = _FakeVideoServer()
        runtime.teacher = _FakeVideoTeacher()
        runtime.dtype = torch.float32
        runtime.configure_teacher_solver(
            video_steps=5,
            video_exec_steps=3,
            action_steps=10,
        )
        with mock.patch(
            "wan_va.utils.data_seq_to_patch",
            side_effect=lambda _patch_size, value, *_args, **_kwargs: value,
        ):
            plan = runtime._teacher_video_plan(
                frame_st_id=1,
                initial_latent=torch.zeros((1, 1, 1, 1, 1)),
                video_noise=torch.zeros((1, 1, 1, 1, 1)),
                cache_name="teacher_video",
            )
        self.assertEqual(runtime.server.scheduler.requested_steps, 5)
        self.assertEqual(runtime.server.scheduler.step_calls, 3)
        self.assertEqual(runtime.teacher.update_cache_calls, [0, 0, 1])
        self.assertTrue(
            torch.equal(plan.prepared_z_s, torch.full_like(plan.prepared_z_s, 3.0))
        )
        self.assertEqual(float(plan.prepared_z_s_timestep.item()), 400.0)

    def test_teacher_action_uses_configured_ten_steps(self) -> None:
        cache_name = "teacher_action_ten"
        server = _FakeActionServer(cache_name=cache_name, initial_length=10)
        runtime = object.__new__(NativeModelRuntime)
        runtime.server = server
        runtime.teacher = server._transformer
        runtime.dtype = torch.float32
        runtime.configure_teacher_solver(
            video_steps=5,
            video_exec_steps=3,
            action_steps=10,
        )
        plan = PreparedPlan(
            raw_z_s=torch.zeros((1, 1, 1, 1, 1)),
            prepared_z_s=torch.zeros((1, 1, 1, 1, 1)),
            prepared_z_s_timestep=torch.zeros((1,)),
            latent_cond_applied=False,
        )
        runtime._teacher_action(
            frame_st_id=1,
            action_noise=torch.zeros((1, 1, 1, 1, 1)),
            cache_name=cache_name,
            plan=plan,
            arm="ST",
        )
        self.assertEqual(len(server._transformer.update_cache_calls), 10)
        self.assertEqual(server._transformer.update_cache_calls, [0] * 9 + [1])

    def test_history_capture_rejects_any_context_after_terminal_macro(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            capture = HistoryCapture(
                output_dir=Path(temp_dir),
                task="put_object_cabinet",
                seed=60000,
                condition="CLEAN_SS",
                unit_id="put_object_cabinet:60000",
            )
            capture.contexts = [
                {
                    "macro_id": macro_id,
                    "student_env_action": np.zeros((16, 2, 16), dtype=np.float32),
                }
                for macro_id in (0, 1)
            ]
            chunks = [
                {
                    "chunk_id": macro_id,
                    "start_frame": 1 if macro_id == 0 else 0,
                    "action_steps": 1,
                    "executed_action_mask": [[False] * 16, [True] + [False] * 15],
                    "terminal_reached": macro_id == 0,
                    "terminal_action_position": [1, 0] if macro_id == 0 else None,
                }
                for macro_id in (0, 1)
            ]

            with self.assertRaisesRegex(RuntimeError, "post-success context"):
                capture.finalize(
                    {"chunks": chunks},
                    output_path=Path(temp_dir) / "history.pt",
                )

    def test_history_capture_attaches_physical_execution_to_matching_macro(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            capture = HistoryCapture(
                output_dir=Path(temp_dir),
                task="put_object_cabinet",
                seed=60000,
                condition="CLEAN_SS",
                unit_id="put_object_cabinet:60000",
            )
            capture.contexts = [
                {
                    "macro_id": 0,
                    "student_env_action": np.zeros((16, 2, 16), dtype=np.float32),
                }
            ]
            output_path = Path(temp_dir) / "history.pt"
            capture.finalize(
                {
                    "chunks": [
                        {
                            "chunk_id": 0,
                            "start_frame": 1,
                            "action_steps": 5,
                            "executed_action_mask": [
                                [False] * 16,
                                [True] * 5 + [False] * 11,
                            ],
                            "terminal_reached": True,
                            "terminal_action_position": [1, 4],
                        }
                    ]
                },
                output_path=output_path,
            )
            context = torch.load(output_path, map_location="cpu", weights_only=False)[
                "contexts"
            ][0]

        self.assertEqual(context["start_frame"], 1)
        self.assertEqual(context["action_steps"], 5)
        self.assertEqual(
            context["executed_action_mask"],
            [[False] * 16, [True] * 5 + [False] * 11],
        )
        self.assertTrue(context["terminal_reached"])
        self.assertEqual(context["terminal_action_position"], [1, 4])

    def test_instruction_metadata_is_reconstructed_without_expert_actions(self) -> None:
        blocks = _FakeInstructionTask()
        blocks.block1 = _FakeActor([-0.12, 0.0, 0.77])
        blocks.block2 = _FakeActor([0.14, 0.0, 0.77])
        blocks.block3 = _FakeActor([-0.18, 0.0, 0.77])
        _initialize_task_local_success_state(blocks, "blocks_ranking_size")
        self.assertEqual(blocks.info["info"], {
            "{A}": "large block",
            "{B}": "medium block",
            "{C}": "small block",
            "{a}": "left",
            "{b}": "right",
            "{c}": "left",
        })

        rgb_blocks = _FakeInstructionTask()
        rgb_blocks.block1 = _FakeActor([0.12, 0.0, 0.77])
        rgb_blocks.block2 = _FakeActor([-0.14, 0.0, 0.77])
        rgb_blocks.block3 = _FakeActor([0.18, 0.0, 0.77])
        _initialize_task_local_success_state(rgb_blocks, "blocks_ranking_rgb")
        self.assertEqual(rgb_blocks.info["info"], {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": "right",
            "{b}": "left",
            "{c}": "right",
        })

        cabinet = _FakeInstructionTask(x=0.1)
        _initialize_task_local_success_state(cabinet, "put_object_cabinet")
        self.assertEqual(cabinet.info["info"], {
            "{A}": "075_bread/base2",
            "{B}": "036_cabinet/base0",
            "{a}": "right",
            "{b}": "left",
        })
        self.assertEqual(cabinet.origin_z, 0.42)
        self.assertEqual(cabinet.arm_tag, "right")

        dustbin = _FakeInstructionTask()
        _initialize_task_local_success_state(dustbin, "put_bottles_dustbin")
        self.assertEqual(dustbin.info["info"], {
            "{A}": "114_bottle/base1",
            "{B}": "114_bottle/base2",
            "{C}": "114_bottle/base3",
            "{D}": "011_dustbin/base0",
        })

        stapler = _FakeInstructionTask(x=-0.1)
        _initialize_task_local_success_state(stapler, "move_stapler_pad")
        self.assertEqual(stapler.info["info"], {
            "{A}": "048_stapler/base5",
            "{B}": "Black",
            "{a}": "left",
        })

        microphone = _FakeInstructionTask()
        microphone.microphone_id = 4
        microphone.grasp_arm_tag = "left"
        microphone.handover_arm_tag = "right"
        _initialize_task_local_success_state(microphone, "handover_mic")
        self.assertEqual(microphone.info["info"], {
            "{A}": "018_microphone/base4",
            "{a}": "left",
            "{b}": "right",
        })

        shoe = _FakeInstructionTask(x=-0.1)
        _initialize_task_local_success_state(shoe, "place_shoe")
        self.assertEqual(shoe.info["info"], {
            "{A}": "041_shoe/base6",
            "{a}": "left",
        })

        centered_shoe = _FakeInstructionTask(x=0.0)
        _initialize_task_local_success_state(centered_shoe, "place_shoe")
        self.assertEqual(centered_shoe.info["info"]["{a}"], "right")

        scanner_left = _FakeInstructionTask(x=-0.1)
        _initialize_task_local_success_state(scanner_left, "scan_object")
        self.assertEqual(scanner_left.info["info"], {
            "{A}": "112_tea-box/base4",
            "{B}": "024_scanner/base3",
            "{a}": "right",
            "{b}": "left",
        })

        scanner_centered = _FakeInstructionTask(x=0.0)
        _initialize_task_local_success_state(scanner_centered, "scan_object")
        self.assertEqual(scanner_centered.info["info"]["{a}"], "left")
        self.assertEqual(scanner_centered.info["info"]["{b}"], "right")

        place_a2b = _FakeInstructionTask(x=0.12)
        place_a2b.selected_modelname_A = "050_bell"
        place_a2b.selected_model_id_A = 2
        place_a2b.selected_modelname_B = "086_woodenblock"
        place_a2b.selected_model_id_B = 1
        _initialize_task_local_success_state(place_a2b, "place_a2b_left")
        self.assertEqual(place_a2b.info["info"], {
            "{A}": "050_bell/base2",
            "{B}": "086_woodenblock/base1",
            "{a}": "right",
        })

        centered_place_a2b = _FakeInstructionTask(x=0.0)
        centered_place_a2b.selected_modelname_A = "047_mouse"
        centered_place_a2b.selected_model_id_A = 3
        centered_place_a2b.selected_modelname_B = "075_bread"
        centered_place_a2b.selected_model_id_B = 4
        _initialize_task_local_success_state(
            centered_place_a2b, "place_a2b_left"
        )
        self.assertEqual(centered_place_a2b.info["info"]["{a}"], "left")

        place_a2b_right = _FakeInstructionTask(x=-0.12)
        place_a2b_right.selected_modelname_A = "047_mouse"
        place_a2b_right.selected_model_id_A = 1
        place_a2b_right.selected_modelname_B = "075_bread"
        place_a2b_right.selected_model_id_B = 2
        _initialize_task_local_success_state(
            place_a2b_right, "place_a2b_right"
        )
        self.assertEqual(place_a2b_right.info["info"]["{a}"], "left")

        bread_basket = _FakeInstructionTask()
        bread_basket.breadbasket = _FakeActor([0.0, -0.2, 0.74])
        bread_basket.basket_id = 3
        bread_basket.bread = [
            _FakeActor([-0.2, -0.1, 0.74]),
            _FakeActor([0.2, -0.1, 0.74]),
        ]
        bread_basket.bread_id = [1, 5]
        _initialize_task_local_success_state(
            bread_basket, "place_bread_basket"
        )
        self.assertEqual(bread_basket.info["info"], {
            "{A}": "076_breadbasket/base3",
            "{B}": "075_bread/base1",
            "{C}": "075_bread/base5",
            "{a}": "dual",
        })

        qrcode = _FakeInstructionTask()
        qrcode.qrcode = _FakeActor([-0.2, -0.1, 0.74])
        qrcode.model_id = 2
        _initialize_task_local_success_state(qrcode, "rotate_qrcode")
        self.assertEqual(qrcode.info["info"], {
            "{A}": "070_paymentsign/base2",
            "{a}": "left",
        })

        cans = _FakeInstructionTask()
        cans.object1_id = 1
        cans.object2_id = 6
        cans.plasticbox_id = 5
        _initialize_task_local_success_state(cans, "place_cans_plasticbox")
        self.assertEqual(cans.info["info"], {
            "{A}": "071_can/base1",
            "{B}": "062_plasticbox/base5",
            "{C}": "071_can/base6",
        })

        stamp = _FakeInstructionTask(x=0.12)
        stamp.seal = _FakeActor([0.12, 0.0, 0.77])
        stamp.seal_id = 4
        stamp.color_name = "Coral"
        _initialize_task_local_success_state(stamp, "stamp_seal")
        self.assertEqual(stamp.info["info"], {
            "{A}": "100_seal/base4",
            "{B}": "Coral",
            "{a}": "right",
        })

        centered_stamp = _FakeInstructionTask(x=0.0)
        centered_stamp.seal = _FakeActor([0.0, 0.0, 0.77])
        centered_stamp.seal_id = 0
        centered_stamp.color_name = "Beige"
        _initialize_task_local_success_state(centered_stamp, "stamp_seal")
        self.assertEqual(centered_stamp.info["info"]["{a}"], "left")

    def test_first_chunk_start_frame_matches_released_deployment(self) -> None:
        action = np.zeros((16, 2, 16), dtype=np.float32)
        action[3, :, :] = 1.0
        action[11, :, :] = 1.0
        task = _FakeTask()
        result = execute_env_action_chunk_physics_only(
            task_env=task,
            action=action,
            initial_eef_pose=np.zeros(16, dtype=np.float64),
            add_init_pose=_identity_pose,
            start_frame=1,
        )
        self.assertEqual(result["action_steps"], 16)
        self.assertEqual(result["start_frame"], 1)
        self.assertEqual(len(task.actions), 16)

    def test_first_and_later_chunks_emit_four_and_eight_key_snapshots(self) -> None:
        action = np.zeros((16, 2, 16), dtype=np.float32)
        action[3, :, :] = 1.0
        action[11, :, :] = 1.0
        captured = []

        def capture(_task, capture_cuda_rng=False):
            value = {"id": len(captured)}
            captured.append(value)
            return value

        from unittest.mock import patch

        with patch(
            "experiments.robotwin_sim_snapshot.capture_simulator_snapshot",
            side_effect=capture,
        ):
            first = execute_env_action_chunk_physics_only(
                task_env=_FakeTask(),
                action=action,
                initial_eef_pose=np.zeros(16, dtype=np.float64),
                add_init_pose=_identity_pose,
                start_frame=1,
                capture_intermediate_snapshots=True,
            )
            later = execute_env_action_chunk_physics_only(
                task_env=_FakeTask(),
                action=action,
                initial_eef_pose=np.zeros(16, dtype=np.float64),
                add_init_pose=_identity_pose,
                start_frame=0,
                capture_intermediate_snapshots=True,
            )

        self.assertEqual(len(first["frame_snapshots"]), 4)
        self.assertEqual(len(later["frame_snapshots"]), 8)

    def test_reset_retains_released_streaming_vae_chronology(self) -> None:
        runtime = object.__new__(NativeModelRuntime)
        runtime.server = _FakeServer()
        result = runtime.reset("prompt", {"obs": "initial"})
        self.assertEqual(tuple(result.shape), (1, 48, 1, 24, 20))
        self.assertEqual(runtime.server.streaming_vae.clear_calls, 0)
        self.assertEqual(runtime.server.streaming_vae_half.clear_calls, 0)
        self.assertEqual(runtime.server.reset_payloads, [{"reset": True, "prompt": "prompt"}])

    def test_noise_pair_is_exactly_shared_by_same_family(self) -> None:
        shape_v = (1, 48, 2, 48, 60)
        shape_a = (1, 30, 2, 16, 1)
        left = LockedNoiseBank(
            task="put_object_cabinet",
            seed=60000,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).pair(family="student", chunk_id=0, video_shape=shape_v, action_shape=shape_a)
        right = LockedNoiseBank(
            task="put_object_cabinet",
            seed=60000,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).pair(family="student", chunk_id=0, video_shape=shape_v, action_shape=shape_a)
        self.assertTrue(torch.equal(left["video"], right["video"]))
        self.assertTrue(torch.equal(left["action"], right["action"]))

    def test_student_and_teacher_noise_families_are_separate(self) -> None:
        student = _derived_seed(
            task="put_object_cabinet",
            seed=60000,
            family="student",
            chunk_id=0,
            base_seed=2026080401,
        )
        teacher = _derived_seed(
            task="put_object_cabinet",
            seed=60000,
            family="teacher",
            chunk_id=0,
            base_seed=2026080402,
        )
        self.assertNotEqual(student, teacher)

    def test_action_fingerprint_cache_length_is_captured_before_action_forward(self) -> None:
        cache_name = "teacher_action"
        server = _FakeActionServer(cache_name=cache_name, initial_length=10)
        teacher = server._transformer
        runtime = object.__new__(NativeModelRuntime)
        runtime.server = server
        runtime.teacher = teacher
        runtime.dtype = torch.float32
        plan = PreparedPlan(
            raw_z_s=torch.zeros((1, 1, 1, 1, 1)),
            prepared_z_s=torch.zeros((1, 1, 1, 1, 1)),
            prepared_z_s_timestep=torch.zeros((1,)),
            latent_cond_applied=False,
        )

        solve = runtime._teacher_action(
            frame_st_id=1,
            action_noise=torch.zeros((1, 1, 1, 1, 1)),
            cache_name=cache_name,
            plan=plan,
            arm="TT",
        )

        self.assertEqual(solve.cache_valid_length, 10)


if __name__ == "__main__":
    unittest.main()
