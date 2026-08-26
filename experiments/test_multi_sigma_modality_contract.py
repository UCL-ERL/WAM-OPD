from __future__ import annotations

import hashlib
import importlib
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch


@contextmanager
def _noop_scope(*args: object, **kwargs: object):
    yield


@pytest.fixture(scope="module")
def contract_modules() -> object:
    """Import the patch slice without leaking local-only dependency stubs."""

    production_names = (
        "experiments.waopd_native_closed_loop_runner",
        "experiments.waopd_v0_video_opd",
        "experiments.train_iterative_on_policy_flow_opd",
        "experiments.verify_multi_sigma_solver_closure",
    )
    if all(name in sys.modules for name in production_names[1:]):
        yield SimpleNamespace(
            video=sys.modules[production_names[1]],
            iterative=sys.modules[production_names[2]],
            closure=sys.modules[production_names[3]],
        )
        return

    einops_stub = ModuleType("einops")
    einops_stub.rearrange = lambda value, *args, **kwargs: value

    goal1_stub = ModuleType("experiments.goal1_exact_condition")
    goal1_stub.ConditionContractError = RuntimeError
    goal1_stub.ConditionFingerprint = object
    goal1_stub.PreparedPlan = object
    goal1_stub.assert_cache_semantics = lambda *args, **kwargs: None
    goal1_stub.assert_fingerprint_match = lambda *args, **kwargs: None
    goal1_stub.build_condition_fingerprint = lambda *args, **kwargs: None
    goal1_stub.cache_valid_length = lambda *args, **kwargs: 0
    goal1_stub.capture_prepared_plan = lambda *args, **kwargs: None
    goal1_stub.fingerprint_mismatches = lambda *args, **kwargs: []
    goal1_stub.grid_token_positions = lambda *args, **kwargs: ()
    goal1_stub.prepare_plan_input = lambda *args, **kwargs: None
    goal1_stub.sequence_hash = lambda *args, **kwargs: "hash"
    goal1_stub.stable_hash = lambda *args, **kwargs: "hash"
    goal1_stub.tensor_diff = lambda *args, **kwargs: {}
    goal1_stub.tensor_hash = lambda value: hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()

    video_output_stub = ModuleType("experiments.video_output_adapter")
    video_output_stub.VideoOutputResidualAdapter = object
    video_output_stub.attach_video_output_adapter = lambda *args, **kwargs: None
    video_output_stub.video_output_adapter_state_dict = lambda *args, **kwargs: {}

    video_lora_stub = ModuleType("experiments.video_mode_lora")
    video_lora_stub.attach_video_mode_lora = lambda *args, **kwargs: None
    video_lora_stub.video_mode_lora_base_parameter_hashes = lambda *args, **kwargs: {}
    video_lora_stub.video_mode_lora_scope = _noop_scope
    video_lora_stub.video_mode_lora_state_dict = lambda *args, **kwargs: {}

    dual_lora_stub = ModuleType("experiments.dual_mode_lora")
    dual_lora_stub.attach_dual_mode_lora = lambda *args, **kwargs: None
    dual_lora_stub.dual_mode_lora_base_parameter_hashes = lambda *args, **kwargs: {}
    dual_lora_stub.dual_mode_lora_contract = lambda *args, **kwargs: {}
    dual_lora_stub.dual_mode_lora_named_parameters = lambda *args, **kwargs: []
    dual_lora_stub.dual_mode_lora_scope = _noop_scope
    dual_lora_stub.dual_mode_lora_state_dict = lambda *args, **kwargs: {}
    dual_lora_stub.load_dual_mode_lora_checkpoint = lambda *args, **kwargs: None
    dual_lora_stub.select_dual_mode_lora_trainable_bank = lambda *args, **kwargs: []

    joint_lora_stub = ModuleType("experiments.joint_lora")
    joint_lora_stub.JointLoRALinear = torch.nn.Linear
    joint_lora_stub.attach_joint_lora = lambda *args, **kwargs: None
    joint_lora_stub.joint_lora_base_parameter_hashes = lambda *args, **kwargs: {}
    joint_lora_stub.joint_lora_state_dict = lambda *args, **kwargs: {}

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

    stubs = {
        "einops": einops_stub,
        "experiments.goal1_exact_condition": goal1_stub,
        "experiments.video_output_adapter": video_output_stub,
        "experiments.video_mode_lora": video_lora_stub,
        "experiments.dual_mode_lora": dual_lora_stub,
        "experiments.joint_lora": joint_lora_stub,
        "experiments.train_joint_teacher_trajectory_opd": joint_teacher_stub,
        "experiments.train_video_trajectory_opd": video_train_stub,
    }
    missing = object()
    managed_names = (*stubs, *production_names)
    previous = {name: sys.modules.get(name, missing) for name in managed_names}
    sys.modules.update(stubs)
    try:
        video = importlib.import_module(production_names[1])
        iterative = importlib.import_module(production_names[2])
        closure = importlib.import_module(production_names[3])
        yield SimpleNamespace(video=video, iterative=iterative, closure=closure)
    finally:
        for name, prior in previous.items():
            if prior is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def _karras_oracle(
    noisy_state: torch.Tensor,
    x0: torch.Tensor,
    *,
    sigma: float,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    denominator = sigma**2 + sigma_data**2
    c_skip = sigma_data**2 / denominator
    c_out = sigma * sigma_data / denominator**0.5
    return c_skip * noisy_state + c_out * x0


def test_video_consistency_map_matches_independent_karras_oracle(
    contract_modules: object,
) -> None:
    noisy_state = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    velocity = torch.tensor([[0.25, -0.5, 2.0]], dtype=torch.float64)
    sigma = 0.5
    x0_hat = noisy_state - sigma * velocity

    prediction = contract_modules.video.video_consistency_map(
        noisy_state,
        x0_hat,
        sigma=sigma,
    )
    expected = _karras_oracle(noisy_state, x0_hat, sigma=sigma)

    torch.testing.assert_close(prediction, expected, rtol=0.0, atol=0.0)
    assert not torch.equal(prediction, x0_hat)


@pytest.mark.parametrize("sigma", [1.0, 0.5, 0.25])
def test_action_flow_x0_map_remains_linear_in_sigma(
    sigma: float,
    contract_modules: object,
) -> None:
    noisy_state = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    velocity = torch.tensor([0.25, -0.5, 2.0], dtype=torch.float64)

    prediction = contract_modules.video.flow_x0_prediction(
        noisy_state,
        velocity,
        sigma=sigma,
    )

    torch.testing.assert_close(
        prediction,
        noisy_state - sigma * velocity,
        rtol=0.0,
        atol=0.0,
    )


def test_video_runtime_composes_raw_x0_with_consistency_map(
    monkeypatch: pytest.MonkeyPatch,
    contract_modules: object,
) -> None:
    video_module = contract_modules.video
    sigma = 0.5
    teacher_endpoint = torch.tensor([[[[[3.0]]]]])
    epsilon = torch.tensor([[[[[1.0]]]]])
    canonical_noisy = (1.0 - sigma) * teacher_endpoint + sigma * epsilon
    velocity = torch.tensor([[[[[0.25]]]]], requires_grad=True)
    calls: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    class FakeScheduler:
        config = SimpleNamespace(num_train_timesteps=2)

        def set_timesteps(self, steps: int) -> None:
            assert steps == 2
            self.timesteps = torch.tensor([2.0, 1.0])
            self.sigmas = torch.tensor([1.0, 0.5, 0.0])

    class FakeTransformer:
        def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
            return velocity

        def clear_cache(self, cache_name: str) -> None:
            assert cache_name == "student"

    server = SimpleNamespace(
        scheduler=FakeScheduler(),
        transformer=FakeTransformer(),
        cache_name="student",
        job_config=SimpleNamespace(
            patch_size=1,
            frame_chunk_size=1,
            guidance_scale=1.0,
        ),
        latent_height=1,
        latent_width=1,
        use_cfg=False,
        _repeat_input_for_cfg=lambda value: value,
    )
    runtime = object.__new__(video_module.NativeV0VideoRuntime)
    runtime.server = server
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float32
    runtime._prepare_context = lambda _context: (torch.zeros(1), [])
    runtime._student_video_call_scope = _noop_scope

    monkeypatch.setattr(
        video_module,
        "prepare_plan_input",
        lambda *args, **kwargs: (
            {
                "latent_res_lst": {
                    "noisy_latents": canonical_noisy.clone(),
                    "timesteps": torch.tensor([1.0]),
                }
            },
            None,
        ),
    )
    sentinel = torch.full_like(canonical_noisy, 17.0)

    def fake_consistency_map(
        noisy_state: torch.Tensor,
        x0_hat: torch.Tensor,
        *,
        sigma: float,
    ) -> torch.Tensor:
        calls.append((noisy_state, x0_hat, sigma))
        return sentinel

    monkeypatch.setattr(video_module, "video_consistency_map", fake_consistency_map)
    wan_va = ModuleType("wan_va")
    wan_va_utils = ModuleType("wan_va.utils")
    wan_va_utils.data_seq_to_patch = lambda *args, **kwargs: args[1]
    monkeypatch.setitem(sys.modules, "wan_va", wan_va)
    monkeypatch.setitem(sys.modules, "wan_va.utils", wan_va_utils)

    result = runtime.student_video_x0_at_sigma(
        {"frame_st_id": 1, "epsilon_v": epsilon},
        teacher_endpoint,
        sigma=sigma,
        require_grad=True,
    )

    expected_x0 = canonical_noisy - sigma * velocity
    torch.testing.assert_close(result.x0_hat, expected_x0)
    torch.testing.assert_close(result.consistency_prediction, sentinel)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], canonical_noisy)
    torch.testing.assert_close(calls[0][1], expected_x0)
    assert calls[0][2] == pytest.approx(sigma)


def test_multi_sigma_trainer_uses_video_consistency_prediction_and_target(
    monkeypatch: pytest.MonkeyPatch,
    contract_modules: object,
) -> None:
    iterative_flow = contract_modules.iterative
    sigma = 0.5
    noisy_state = torch.tensor([1.0, 2.0], requires_grad=True)
    teacher_video = torch.tensor([3.0, 4.0], requires_grad=True)
    teacher_action = torch.tensor([5.0, 6.0])
    video_target = _karras_oracle(
        noisy_state,
        teacher_video,
        sigma=sigma,
    )

    class FakeRuntime:
        device = "cpu"
        dtype = torch.float32

        def __init__(self) -> None:
            self.video_prediction = video_target.detach().clone().requires_grad_()

        def student_video_x0_at_sigma(self, *args: object, **kwargs: object) -> object:
            return SimpleNamespace(
                sigma=sigma,
                noisy_state=noisy_state,
                x0_hat=torch.full_like(video_target, -100.0),
                consistency_prediction=self.video_prediction,
            )

        def student_action_x0_at_sigma(self, *args: object, **kwargs: object) -> object:
            return SimpleNamespace(
                sigma=sigma,
                noisy_state=torch.zeros_like(teacher_action),
                timestep=torch.tensor([1.0]),
                valid_mask=torch.ones_like(teacher_action, dtype=torch.bool),
                token_positions=(),
                cache_valid_length=0,
                x0_prediction=teacher_action.clone(),
            )

    monkeypatch.setattr(iterative_flow, "_validate_label", lambda _label: None)
    monkeypatch.setattr(
        iterative_flow,
        "materialize_context",
        lambda _trajectory, _label: {},
    )
    monkeypatch.setattr(
        iterative_flow,
        "video_execution_mask",
        lambda target, _label: torch.ones_like(target, dtype=torch.bool),
    )
    monkeypatch.setattr(
        iterative_flow,
        "action_execution_mask",
        lambda mask, _label: mask,
    )
    label = {
        "student_z_s": torch.zeros_like(teacher_video),
        "teacher_z_t": teacher_video,
        "teacher_action": teacher_action,
    }
    runtime = FakeRuntime()

    matching = iterative_flow._multi_sigma_anchor_forward(
        runtime=runtime,
        trajectory={},
        label=label,
        sigma_values=[sigma],
        pseudo_huber_c=1e-3,
        require_grad=True,
        capture_outputs=True,
    )

    assert float(matching["video_loss"].item()) == pytest.approx(0.0)
    torch.testing.assert_close(
        matching["outputs"][0]["video_target"],
        video_target,
    )

    runtime.video_prediction = (video_target.detach() + 1.0).requires_grad_()
    mismatching = iterative_flow._multi_sigma_anchor_forward(
        runtime=runtime,
        trajectory={},
        label=label,
        sigma_values=[sigma],
        pseudo_huber_c=1e-3,
        require_grad=True,
        capture_outputs=False,
    )

    assert float(mismatching["video_loss"].item()) > 0.0
    mismatching["video_loss"].backward()
    assert runtime.video_prediction.grad is not None
    assert float(runtime.video_prediction.grad.abs().sum().item()) > 0.0
    assert teacher_video.grad is None
    assert noisy_state.grad is None


def test_streamed_multi_sigma_backward_matches_summed_gradient_and_order(
    monkeypatch: pytest.MonkeyPatch,
    contract_modules: object,
) -> None:
    iterative_flow = contract_modules.iterative
    sigmas = (0.5, 0.25)
    teacher_video = torch.tensor([3.0, 4.0])
    teacher_action = torch.tensor([5.0, 6.0])
    noisy_states = {
        0.5: torch.tensor([1.0, 2.0]),
        0.25: torch.tensor([2.0, 1.0]),
    }
    video_features = {
        0.5: torch.tensor([1.0, -0.5]),
        0.25: torch.tensor([-0.25, 2.0]),
    }
    action_features = {
        0.5: torch.tensor([0.75, -1.5]),
        0.25: torch.tensor([1.25, 0.5]),
    }

    class FakeRuntime:
        device = "cpu"
        dtype = torch.float32

        def __init__(self) -> None:
            self.weight = torch.nn.Parameter(torch.tensor(0.2))
            self.events: list[tuple[str, float]] = []

        def _record_backward(self, modality: str, sigma: float):
            def hook(gradient: torch.Tensor) -> torch.Tensor:
                self.events.append((f"{modality}_backward", sigma))
                return gradient

            return hook

        def student_video_x0_at_sigma(
            self,
            *args: object,
            sigma: float,
            **kwargs: object,
        ) -> object:
            resolved_sigma = float(sigma)
            self.events.append(("video_forward", resolved_sigma))
            target = iterative_flow.video_consistency_map(
                noisy_states[resolved_sigma],
                teacher_video,
                sigma=resolved_sigma,
            )
            prediction = target + self.weight * video_features[resolved_sigma]
            prediction.register_hook(
                self._record_backward("video", resolved_sigma)
            )
            return SimpleNamespace(
                sigma=resolved_sigma,
                noisy_state=noisy_states[resolved_sigma],
                x0_hat=teacher_video,
                consistency_prediction=prediction,
            )

        def student_action_x0_at_sigma(
            self,
            *args: object,
            sigma: float,
            **kwargs: object,
        ) -> object:
            resolved_sigma = float(sigma)
            self.events.append(("action_forward", resolved_sigma))
            prediction = (
                teacher_action + self.weight * action_features[resolved_sigma]
            )
            prediction.register_hook(
                self._record_backward("action", resolved_sigma)
            )
            return SimpleNamespace(
                sigma=resolved_sigma,
                noisy_state=torch.zeros_like(teacher_action),
                timestep=torch.tensor([resolved_sigma]),
                valid_mask=torch.ones_like(teacher_action, dtype=torch.bool),
                token_positions=(),
                cache_valid_length=0,
                x0_prediction=prediction,
            )

    monkeypatch.setattr(iterative_flow, "_validate_label", lambda _label: None)
    monkeypatch.setattr(
        iterative_flow,
        "materialize_context",
        lambda _trajectory, _label: {},
    )
    monkeypatch.setattr(
        iterative_flow,
        "video_execution_mask",
        lambda target, _label: torch.ones_like(target, dtype=torch.bool),
    )
    monkeypatch.setattr(
        iterative_flow,
        "action_execution_mask",
        lambda mask, _label: mask,
    )
    label = {
        "student_z_s": torch.zeros_like(teacher_video),
        "teacher_z_t": teacher_video,
        "teacher_action": teacher_action,
    }
    runtime = FakeRuntime()
    video_weight = 1.75
    action_weight = 0.6

    summed = iterative_flow._multi_sigma_anchor_forward(
        runtime=runtime,
        trajectory={},
        label=label,
        sigma_values=sigmas,
        pseudo_huber_c=1e-3,
        require_grad=True,
        capture_outputs=False,
    )
    summed_loss = (
        video_weight * summed["video_loss"]
        + action_weight * summed["action_loss"]
    )
    summed_loss.backward()
    expected_gradient = runtime.weight.grad.detach().clone()
    expected_video_loss = summed["video_loss"].detach().clone()
    expected_action_loss = summed["action_loss"].detach().clone()

    runtime.weight.grad = None
    runtime.events = []
    streamed = iterative_flow._multi_sigma_anchor_forward(
        runtime=runtime,
        trajectory={},
        label=label,
        sigma_values=sigmas,
        pseudo_huber_c=1e-3,
        require_grad=True,
        capture_outputs=False,
        backward_scales=(video_weight, action_weight),
    )

    assert runtime.events == [
        ("video_forward", 0.5),
        ("video_backward", 0.5),
        ("action_forward", 0.5),
        ("action_backward", 0.5),
        ("video_forward", 0.25),
        ("video_backward", 0.25),
        ("action_forward", 0.25),
        ("action_backward", 0.25),
    ]
    torch.testing.assert_close(runtime.weight.grad, expected_gradient)
    torch.testing.assert_close(streamed["video_loss"], expected_video_loss)
    torch.testing.assert_close(streamed["action_loss"], expected_action_loss)
    assert not streamed["video_loss"].requires_grad
    assert not streamed["action_loss"].requires_grad


@pytest.mark.parametrize("nonfinite_modality", ["video", "action"])
def test_streamed_multi_sigma_rejects_nonfinite_before_backward(
    monkeypatch: pytest.MonkeyPatch,
    contract_modules: object,
    nonfinite_modality: str,
) -> None:
    iterative_flow = contract_modules.iterative
    sigma = 0.5
    teacher_video = torch.tensor([3.0, 4.0])
    teacher_action = torch.tensor([5.0, 6.0])

    class FakeRuntime:
        device = "cpu"
        dtype = torch.float32

        def __init__(self) -> None:
            self.weight = torch.nn.Parameter(torch.tensor(0.2))
            self.action_called = False
            self.action_backward_called = False

        def student_video_x0_at_sigma(self, *args: object, **kwargs: object) -> object:
            fill_value = float("nan") if nonfinite_modality == "video" else 1.0
            return SimpleNamespace(
                sigma=sigma,
                noisy_state=torch.zeros_like(teacher_video),
                consistency_prediction=self.weight
                * torch.full_like(teacher_video, fill_value),
            )

        def student_action_x0_at_sigma(self, *args: object, **kwargs: object) -> object:
            self.action_called = True
            fill_value = float("nan") if nonfinite_modality == "action" else 1.0
            prediction = teacher_action + self.weight * torch.full_like(
                teacher_action,
                fill_value,
            )

            def record_backward(gradient: torch.Tensor) -> torch.Tensor:
                self.action_backward_called = True
                return gradient

            prediction.register_hook(record_backward)
            return SimpleNamespace(
                sigma=sigma,
                noisy_state=torch.zeros_like(teacher_action),
                timestep=torch.tensor([sigma]),
                valid_mask=torch.ones_like(teacher_action, dtype=torch.bool),
                token_positions=(),
                cache_valid_length=0,
                x0_prediction=prediction,
            )

    monkeypatch.setattr(iterative_flow, "_validate_label", lambda _label: None)
    monkeypatch.setattr(
        iterative_flow,
        "materialize_context",
        lambda _trajectory, _label: {},
    )
    monkeypatch.setattr(
        iterative_flow,
        "video_execution_mask",
        lambda target, _label: torch.ones_like(target, dtype=torch.bool),
    )
    monkeypatch.setattr(
        iterative_flow,
        "action_execution_mask",
        lambda mask, _label: mask,
    )
    runtime = FakeRuntime()
    label = {
        "student_z_s": torch.zeros_like(teacher_video),
        "teacher_z_t": teacher_video,
        "teacher_action": teacher_action,
    }

    with pytest.raises(FloatingPointError, match=f"{nonfinite_modality} loss"):
        iterative_flow._multi_sigma_anchor_forward(
            runtime=runtime,
            trajectory={},
            label=label,
            sigma_values=[sigma],
            pseudo_huber_c=1e-3,
            require_grad=True,
            capture_outputs=False,
            backward_scales=(1.0, 1.0),
        )

    if nonfinite_modality == "video":
        assert runtime.weight.grad is None
        assert not runtime.action_called
    else:
        assert runtime.weight.grad is not None
        assert runtime.action_called
        assert not runtime.action_backward_called


def _minimal_config(**overrides: object) -> dict[str, object]:
    return {
        "task": "open_microwave",
        "rollouts": [{"seed": 10000, "prompt": "open the microwave"}],
        "rounds": 1,
        **overrides,
    }


def test_teacher_solver_config_defaults_and_minus_one_normalization(
    contract_modules: object,
) -> None:
    iterative = contract_modules.iterative

    defaults = iterative._normalize_config(_minimal_config())
    shortened = iterative._normalize_config(
        _minimal_config(
            teacher_video_steps=5,
            teacher_video_exec_steps=-1,
            teacher_action_steps=10,
        )
    )

    assert defaults["teacher_video_steps"] == 25
    assert defaults["teacher_video_exec_steps"] is None
    assert defaults["teacher_action_steps"] == 50
    assert shortened["teacher_video_steps"] == 5
    assert shortened["teacher_video_exec_steps"] is None
    assert shortened["teacher_action_steps"] == 10


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"teacher_video_steps": 0}, "teacher_video_steps"),
        ({"teacher_action_steps": 0}, "teacher_action_steps"),
        (
            {"teacher_video_steps": 5, "teacher_video_exec_steps": 0},
            "teacher_video_exec_steps",
        ),
        (
            {"teacher_video_steps": 5, "teacher_video_exec_steps": 6},
            "teacher_video_exec_steps",
        ),
        ({"teacher_video_exec_steps": -2}, "teacher_video_exec_steps"),
    ],
)
def test_teacher_solver_config_rejects_invalid_geometry(
    overrides: dict[str, object],
    message: str,
    contract_modules: object,
) -> None:
    with pytest.raises(ValueError, match=message):
        contract_modules.iterative._normalize_config(_minimal_config(**overrides))


def test_teacher_solver_config_is_propagated_to_runtime(
    contract_modules: object,
) -> None:
    calls: list[dict[str, object]] = []
    runtime = SimpleNamespace(
        configure_teacher_solver=lambda **kwargs: calls.append(kwargs)
    )
    config = contract_modules.iterative._normalize_config(
        _minimal_config(
            teacher_video_steps=5,
            teacher_video_exec_steps=3,
            teacher_action_steps=10,
        )
    )

    contract_modules.iterative._configure_teacher_solver_from_config(
        runtime,
        config,
    )

    assert calls == [
        {
            "video_steps": 5,
            "video_exec_steps": 3,
            "action_steps": 10,
        }
    ]


def test_cuda_memory_fraction_is_validated_and_applied(
    monkeypatch: pytest.MonkeyPatch,
    contract_modules: object,
) -> None:
    iterative = contract_modules.iterative
    config = iterative._normalize_config(
        _minimal_config(cuda_memory_fraction=0.65)
    )
    calls: list[tuple[float, torch.device]] = []
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, *, device: calls.append((float(fraction), device)),
    )

    iterative._configure_cuda_memory_limit(config)

    assert calls == [(0.65, torch.device("cuda:0"))]
    with pytest.raises(ValueError, match="cuda_memory_fraction"):
        iterative._normalize_config(
            _minimal_config(cuda_memory_fraction=1.01)
        )


def test_pre_update_solver_closure_uses_first_and_last_real_anchors(
    monkeypatch: pytest.MonkeyPatch,
    contract_modules: object,
) -> None:
    calls: list[int] = []
    verifier = ModuleType("experiments.verify_multi_sigma_solver_closure")

    def fake_anchor_closure(
        runtime: object,
        trajectory: object,
        label: dict[str, int],
    ) -> dict[str, object]:
        del runtime, trajectory
        calls.append(int(label["macro_id"]))
        return {"macro_id": int(label["macro_id"]), "status": "PASS"}

    verifier._anchor_closure = fake_anchor_closure
    monkeypatch.setitem(
        sys.modules,
        "experiments.verify_multi_sigma_solver_closure",
        verifier,
    )
    report = contract_modules.iterative._pre_update_solver_closure(
        object(),
        [
            {
                "collection_id": "stage0-seed10000",
                "labels": [
                    {"macro_id": 0},
                    {"macro_id": 1},
                    {"macro_id": 12},
                ],
            }
        ],
    )

    assert calls == [0, 12]
    assert report["status"] == "PASS"
    assert report["trajectory_collection_id"] == "stage0-seed10000"


def test_video_closure_separates_raw_endpoint_from_consistency_oracle(
    contract_modules: object,
) -> None:
    sigma = 1.0
    noisy_state = torch.tensor([1.0, 2.0])
    deployed_plan = torch.tensor([3.0, 4.0])
    mask = torch.ones_like(deployed_plan, dtype=torch.bool)
    consistency = _karras_oracle(
        noisy_state,
        deployed_plan,
        sigma=sigma,
    )
    forward = SimpleNamespace(
        sigma=sigma,
        noisy_state=noisy_state,
        x0_hat=deployed_plan.clone(),
        consistency_prediction=consistency,
    )

    rows = contract_modules.closure._video_prediction_closure_rows(
        forward,
        deployed_plan,
        mask,
    )

    assert rows["raw_deployment_endpoint"]["equal"] is True
    assert rows["consistency_oracle"]["equal"] is True

    forward.consistency_prediction = forward.x0_hat
    bad_rows = contract_modules.closure._video_prediction_closure_rows(
        forward,
        deployed_plan,
        mask,
    )
    assert bad_rows["raw_deployment_endpoint"]["equal"] is True
    assert bad_rows["consistency_oracle"]["equal"] is False
