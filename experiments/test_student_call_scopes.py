"""Focused mocked tests for Student video/action call-scope ownership."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import sys
from types import ModuleType, SimpleNamespace
import unittest


try:
    import torch
except ModuleNotFoundError:
    class _FakeTensor:
        def __init__(self, shape: tuple[int, ...] = (1,)) -> None:
            self.shape = shape
            self.device = "cpu"

        def clone(self):  # type: ignore[no-untyped-def]
            return _FakeTensor(self.shape)

        def detach(self):  # type: ignore[no-untyped-def]
            return self

        def to(self, *_args: object, **_kwargs: object):  # type: ignore[no-untyped-def]
            return self

        def expand_as(self, other: "_FakeTensor") -> "_FakeTensor":
            return _FakeTensor(other.shape)

        def reshape(self, *_shape: object):  # type: ignore[no-untyped-def]
            return self

        def tolist(self) -> list[int]:
            return [0]

        def __getitem__(self, _key: object):  # type: ignore[no-untyped-def]
            return self

        def __setitem__(self, _key: object, _value: object) -> None:
            pass

        def __invert__(self):  # type: ignore[no-untyped-def]
            return self

        def __imul__(self, _value: object):  # type: ignore[no-untyped-def]
            return self

    torch = ModuleType("torch")
    torch.Tensor = _FakeTensor
    torch.dtype = object
    torch.bool = bool
    torch.float32 = object()
    torch.device = lambda value: value
    torch.zeros = lambda *shape, **_kwargs: _FakeTensor(
        tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else tuple(shape)
    )
    torch.tensor = lambda value, **_kwargs: _FakeTensor(
        (len(value),) if isinstance(value, (list, tuple)) else (1,)
    )
    torch.inference_mode = contextmanager(lambda: (yield))
    torch.no_grad = contextmanager(lambda: (yield))
    torch.cuda = SimpleNamespace(empty_cache=lambda: None, synchronize=lambda *_args: None)
    torch_nn = ModuleType("torch.nn")
    torch_nn_functional = ModuleType("torch.nn.functional")
    torch_nn_functional.pad = lambda value, *_args, **_kwargs: value
    torch.nn = torch_nn
    torch_nn.functional = torch_nn_functional
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch_nn
    sys.modules["torch.nn.functional"] = torch_nn_functional

try:
    import einops  # noqa: F401
except ModuleNotFoundError:
    einops = ModuleType("einops")
    einops.rearrange = lambda value, *_args, **_kwargs: value
    sys.modules["einops"] = einops


def _install_stub(name: str, **attributes: object) -> None:
    try:
        importlib.import_module(name)
        return
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module


class _ConditionContractError(RuntimeError):
    pass


@contextmanager
def _noop_scope(_enabled: bool):
    yield


_install_stub(
    "experiments.goal1_exact_condition",
    ConditionContractError=_ConditionContractError,
    ConditionFingerprint=object,
    PreparedPlan=object,
    assert_cache_semantics=lambda *args, **kwargs: None,
    assert_fingerprint_match=lambda *args, **kwargs: None,
    build_condition_fingerprint=lambda *args, **kwargs: None,
    cache_valid_length=lambda *args, **kwargs: 0,
    capture_prepared_plan=lambda *args, **kwargs: None,
    fingerprint_mismatches=lambda *args, **kwargs: [],
    grid_token_positions=lambda *args, **kwargs: (),
    prepare_plan_input=lambda *args, **kwargs: None,
    sequence_hash=lambda *args, **kwargs: "hash",
    stable_hash=lambda *args, **kwargs: "hash",
    tensor_diff=lambda *args, **kwargs: {},
    tensor_hash=lambda *args, **kwargs: "hash",
)
_install_stub(
    "experiments.video_output_adapter",
    VideoOutputResidualAdapter=object,
    attach_video_output_adapter=lambda *args, **kwargs: None,
    video_output_adapter_state_dict=lambda *args, **kwargs: {},
)
_install_stub(
    "experiments.video_mode_lora",
    attach_video_mode_lora=lambda *args, **kwargs: None,
    video_mode_lora_base_parameter_hashes=lambda *args, **kwargs: {},
    video_mode_lora_scope=_noop_scope,
    video_mode_lora_state_dict=lambda *args, **kwargs: {},
)
_install_stub(
    "experiments.dual_mode_lora",
    attach_dual_mode_lora=lambda *args, **kwargs: None,
    dual_mode_lora_base_parameter_hashes=lambda *args, **kwargs: {},
    dual_mode_lora_contract=lambda *args, **kwargs: {},
    dual_mode_lora_named_parameters=lambda *args, **kwargs: [],
    dual_mode_lora_scope=_noop_scope,
    dual_mode_lora_state_dict=lambda *args, **kwargs: {},
    load_dual_mode_lora_checkpoint=lambda *args, **kwargs: None,
    load_dual_mode_lora_state_dict=lambda *args, **kwargs: None,
    select_dual_mode_lora_trainable_bank=lambda *args, **kwargs: [],
    validate_dual_mode_lora_contract=lambda *args, **kwargs: None,
)
_install_stub(
    "experiments.joint_lora",
    attach_joint_lora=lambda *args, **kwargs: None,
    joint_lora_base_parameter_hashes=lambda *args, **kwargs: {},
    joint_lora_state_dict=lambda *args, **kwargs: {},
)

from experiments.waopd_native_closed_loop_runner import (  # noqa: E402
    HistoryInput,
    NativeModelRuntime,
)
from experiments import waopd_v0_video_opd as video_opd  # noqa: E402


class _Scheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([1.0])

    def set_timesteps(self, _steps: int) -> None:
        self.timesteps = torch.tensor([1.0])


class _ScopeAwareTransformer:
    def __init__(self, scope_state: dict[str, object]) -> None:
        self.scope_state = scope_state
        self.calls: list[tuple[str, object]] = []

    def clear_pred_cache(self, _cache_name: str) -> None:
        pass

    def __call__(self, _model_input: object, *, action_mode: bool, **_kwargs: object):
        branch = "action" if action_mode else "video"
        self.calls.append((branch, self.scope_state["active"]))
        return torch.zeros(1)


class _FakeServer:
    def __init__(self, scope_state: dict[str, object]) -> None:
        self.transformer = _ScopeAwareTransformer(scope_state)
        self.scope_state = scope_state
        self.frame_st_id = 0
        self.device = "cpu"
        self.action_mask = torch.tensor([True, True])
        self.action_scheduler = _Scheduler()
        self.job_config = SimpleNamespace(action_dim=2, action_per_frame=1)
        self.action_per_frame = 1
        self.action_prepare_scopes: list[object] = []

    def _repeat_input_for_cfg(self, model_input: object) -> object:
        return model_input

    def _prepare_latent_input(
        self,
        latent: torch.Tensor | None,
        action: torch.Tensor,
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, dict[str, torch.Tensor]]:
        if latent is None:
            self.action_prepare_scopes.append(self.scope_state["active"])
        return {
            "latent_res_lst": {"stream": torch.tensor(0)},
            "action_res_lst": {
                "stream": torch.tensor(1),
                "noisy_latents": action,
                "timesteps": torch.tensor([1.0]),
                "grid_id": torch.tensor([0]),
            },
        }


def _record() -> HistoryInput:
    return HistoryInput(
        frame_st_id=0,
        latent=torch.zeros((1, 1, 2, 1, 1)),
        action=torch.zeros((1, 2, 1, 1, 1)),
    )


def _runtime(runtime_type: type[NativeModelRuntime], adapter_kind: str | None = None):
    scope_state: dict[str, object] = {"active": None}
    runtime = runtime_type.__new__(runtime_type)
    runtime.server = _FakeServer(scope_state)
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float32
    if adapter_kind is not None:
        runtime.adapter_kind = adapter_kind
    return runtime, scope_state


class StudentCallScopeTest(unittest.TestCase):
    def test_base_runtime_scopes_are_noop(self) -> None:
        runtime, scope_state = _runtime(NativeModelRuntime)

        runtime.append_student_history(record=_record())

        self.assertEqual(
            runtime.server.transformer.calls,
            [("video", None), ("action", None)],
        )
        self.assertIsNone(scope_state["active"])

    def test_video_lora_uses_true_for_video_false_for_action_only_on_student(self) -> None:
        runtime, scope_state = _runtime(video_opd.NativeV0VideoRuntime, "video_lora")
        events: list[tuple[str, bool]] = []

        @contextmanager
        def recorded_scope(enabled: bool):
            previous = scope_state["active"]
            events.append(("enter", enabled))
            scope_state["active"] = enabled
            try:
                yield
            finally:
                events.append(("exit", enabled))
                scope_state["active"] = previous

        original_scope = video_opd.video_mode_lora_scope
        video_opd.video_mode_lora_scope = recorded_scope
        try:
            runtime.append_student_history(record=_record())
            self.assertEqual(
                runtime.server.transformer.calls,
                [("video", True), ("action", False)],
            )

            runtime.server.action_prepare_scopes.clear()
            runtime.action_probe(
                owner="student",
                frame_st_id=0,
                action_noise=torch.zeros((1, 2, 1, 1, 1)),
            )
            event_count = len(events)
            runtime.action_probe(
                owner="teacher",
                frame_st_id=0,
                action_noise=torch.zeros((1, 2, 1, 1, 1)),
            )
            self.assertEqual(runtime.server.action_prepare_scopes, [False, None])
            self.assertEqual(len(events), event_count)

            runtime._create_cache = lambda *_args, **_kwargs: None
            runtime.server.transformer.calls.clear()
            runtime._replay_history(
                model=runtime.server.transformer,
                cache_name="student",
                history=[_record()],
            )
            self.assertEqual(
                runtime.server.transformer.calls,
                [("video", True), ("action", False)],
            )

            teacher = _ScopeAwareTransformer(scope_state)
            runtime._replay_history(
                model=teacher,
                cache_name="teacher",
                history=[_record()],
            )
            self.assertEqual(teacher.calls, [("video", None), ("action", None)])
        finally:
            video_opd.video_mode_lora_scope = original_scope

        self.assertIsNone(scope_state["active"])
        self.assertEqual(
            events,
            [
                ("enter", True),
                ("exit", True),
                ("enter", False),
                ("exit", False),
                ("enter", False),
                ("exit", False),
                ("enter", True),
                ("exit", True),
                ("enter", False),
                ("exit", False),
            ],
        )

    def test_non_video_lora_adapter_preserves_noop_semantics(self) -> None:
        runtime, scope_state = _runtime(video_opd.NativeV0VideoRuntime, "output")

        runtime.append_student_history(record=_record())

        self.assertEqual(
            runtime.server.transformer.calls,
            [("video", None), ("action", None)],
        )
        self.assertIsNone(scope_state["active"])

    def test_dual_lora_selects_video_and_action_banks_on_student_only(self) -> None:
        runtime, scope_state = _runtime(video_opd.NativeV0VideoRuntime, "dual_lora")
        events: list[tuple[str, str]] = []

        @contextmanager
        def recorded_scope(mode: str):
            previous = scope_state["active"]
            events.append(("enter", mode))
            scope_state["active"] = mode
            try:
                yield
            finally:
                events.append(("exit", mode))
                scope_state["active"] = previous

        original_scope = video_opd.dual_mode_lora_scope
        video_opd.dual_mode_lora_scope = recorded_scope
        try:
            runtime.append_student_history(record=_record())
            self.assertEqual(
                runtime.server.transformer.calls,
                [("video", "video"), ("action", "action")],
            )

            runtime.server.action_prepare_scopes.clear()
            runtime.action_probe(
                owner="student",
                frame_st_id=0,
                action_noise=torch.zeros((1, 2, 1, 1, 1)),
            )
            event_count = len(events)
            runtime.action_probe(
                owner="teacher",
                frame_st_id=0,
                action_noise=torch.zeros((1, 2, 1, 1, 1)),
            )
            self.assertEqual(
                runtime.server.action_prepare_scopes, ["action", None]
            )
            self.assertEqual(len(events), event_count)

            runtime._create_cache = lambda *_args, **_kwargs: None
            runtime.server.transformer.calls.clear()
            runtime._replay_history(
                model=runtime.server.transformer,
                cache_name="student",
                history=[_record()],
            )
            self.assertEqual(
                runtime.server.transformer.calls,
                [("video", "video"), ("action", "action")],
            )

            teacher = _ScopeAwareTransformer(scope_state)
            runtime._replay_history(
                model=teacher,
                cache_name="teacher",
                history=[_record()],
            )
            self.assertEqual(teacher.calls, [("video", None), ("action", None)])
        finally:
            video_opd.dual_mode_lora_scope = original_scope

        self.assertIsNone(scope_state["active"])
        self.assertEqual(
            events,
            [
                ("enter", "video"),
                ("exit", "video"),
                ("enter", "action"),
                ("exit", "action"),
                ("enter", "action"),
                ("exit", "action"),
                ("enter", "video"),
                ("exit", "video"),
                ("enter", "action"),
                ("exit", "action"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
