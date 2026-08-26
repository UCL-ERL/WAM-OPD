"""PROTOTYPE — Run the native RoboTwin client with paired instructions.

The upstream evaluator samples a seen instruction from process-local NumPy RNG
state. Parallel 1a/2a clients can therefore receive different text for the same
environment seed. This throwaway launcher changes only that single selection:
the instruction index is derived from ``now_seed``.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import os
import json
import sys
import types
from copy import deepcopy as _stage_l_deepcopy
from pathlib import Path

import torch

from robotwin_pair_task_state import initialize_pair_task_state
from robotwin_sim_snapshot import (
    capture_simulator_snapshot,
    compare_simulator_states,
    perturb_simulator_state_for_audit,
    restore_simulator_snapshot,
    simulator_state_sha256,
)
from robotwin_branch_planner_trace import BranchPlannerTrace
from robotwin_branch_oracle import (
    array_sha256 as _stage_l_array_sha256,
    configure_native_physics_only_child as _stage_l_configure_native_physics_only_child,
    execute_env_action_chunk as _stage_l_execute_env_action_chunk,
    execute_env_action_chunk_physics_only as _stage_l_execute_physics_action_chunk,
    load_branch_intervention as _stage_l_load_branch_intervention,
    rebuild_flashwam_prefix as _stage_l_rebuild_flashwam_prefix,
)
from robotwin_fork_clone_oracle import (
    render_end_observation_in_parent as _stage_l_render_end_observation_in_parent,
    run_forked_physical_branches as _stage_l_run_forked_physical_branches,
)
from robotwin_planner_service import RoboTwinPlannerService as _stage_l_planner_service_type
from stage_h_context_contract import compare_formatted_observations
from stage_h_task_progress import collect_task_progress
from stage_g_data_manifest import build_policy_version as _stage_m_policy_version
from stage_m_live_bridge_contract import (
    build_live_context as _stage_m_build_live_context,
    build_teacher_bridge_command as _stage_m_build_teacher_command,
    validate_live_bridge_label as _stage_m_validate_live_label,
)
from stage_n_fresh_treatment import (
    FreshPlannerEventTrace as _stage_n_planner_event_trace_type,
    FreshPlannerPrefixTrace as _stage_n_planner_prefix_trace_type,
    build_intervention_provenance as _stage_n_build_provenance,
    fresh_history_context_sha256 as _stage_n_fresh_context_sha256,
    parse_intervention_frames as _stage_n_parse_frames,
    select_live_treatment as _stage_n_select_live_treatment,
    validate_completed_schedule as _stage_n_validate_schedule,
    validate_live_pause_contract as _stage_n_validate_pause,
)
from stage_o_event_trigger import (
    EventConfig as _stage_o_event_config_type,
    ObservableEventConfig as _stage_o_observable_config_type,
    ObservableEventDetector as _stage_o_observable_detector_type,
    StageOTraceRecorder as _stage_o_trace_recorder_type,
    observable_change_metrics as _stage_o_observable_change,
)


def _stage_l_tensor_contract_sha256(value):
    """Hash a saved tensor with the same contract as the preflight validator."""

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _stage_l_capture_noise_provenance(frame_st_id):
    """Locate the actual server-captured noise inherited by fork branches.

    A physical-only fork branch does not call the model, so it cannot create a
    second independent noise file.  The one parent inference is the canonical
    source tensor for every child branch; this function records that fact and
    the tensor hashes explicitly instead of reducing provenance to a seed.
    """

    prefix_value = os.environ.get("DIFFUSION_NOISE_CAPTURE_ARTIFACT")
    if not prefix_value:
        return {
            "status": "UNAVAILABLE",
            "reason": "DIFFUSION_NOISE_CAPTURE_ARTIFACT is not set",
            "frame_st_id": int(frame_st_id),
        }
    prefix = Path(prefix_value)
    records = []
    for path in sorted(prefix.parent.glob(f"{prefix.stem}_*.pt")):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            records.append(
                {
                    "path": str(path),
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if not isinstance(payload, dict) or int(payload.get("frame_st_id", -1)) != int(
            frame_st_id
        ):
            continue
        action_noise = payload.get("action_base_noise")
        video_noise = payload.get("video_base_noise")
        if not isinstance(action_noise, torch.Tensor) or not isinstance(
            video_noise, torch.Tensor
        ):
            records.append(
                {
                    "path": str(path),
                    "status": "ERROR",
                    "error": "noise artifact lacks tensor base-noise fields",
                }
            )
            continue
        records.append(
            {
                "path": str(path),
                "capture_index": int(payload.get("capture_index", -1)),
                "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "action_noise_sha256": _stage_l_tensor_contract_sha256(action_noise),
                "video_noise_sha256": _stage_l_tensor_contract_sha256(video_noise),
                "action_shape": list(action_noise.shape),
                "video_shape": list(video_noise.shape),
                "action_timestep_start": payload.get("action_timestep_start"),
                "action_timestep_end": payload.get("action_timestep_end"),
                "status": "PASS",
            }
        )
    valid = [row for row in records if row.get("status") == "PASS"]
    action_hashes = {row["action_noise_sha256"] for row in valid}
    video_hashes = {row["video_noise_sha256"] for row in valid}
    status = "PASS" if valid and len(action_hashes) == 1 and len(video_hashes) == 1 else "BLOCKED"
    return {
        "status": status,
        "frame_st_id": int(frame_st_id),
        "propagation": "parent_canonical_noise_inherited_by_fork_children",
        "records": records,
        "action_noise_sha256": next(iter(action_hashes), None),
        "video_noise_sha256": next(iter(video_hashes), None),
        "record_count": len(valid),
    }


def _stage_l_install_physics_only_task_env(task_env):
    """Defer SAPIEN renderer/camera construction until after a fork.

    RoboTwin's normal ``_init_task_env_`` creates a Vulkan renderer in
    ``setup_scene`` and adds cameras in ``load_camera`` before it finishes
    constructing the physical episode.  A child forked after that point
    cannot safely create camera buffers (and a child forked later can hang in
    OIDN/Vulkan).  This hook preserves the original physics and RNG ordering,
    but postpones only renderer objects and non-physical camera entities.
    """

    from types import MethodType

    import numpy as np
    import sapien.core as sapien

    def _setup_scene_without_renderer(self, **kwargs):
        self.engine = sapien.Engine()
        self.renderer = None
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(kwargs.get("timestep", 1 / 250))
        self.scene.add_ground(kwargs.get("ground_height", 0))
        self.scene.default_physical_material = self.scene.create_physical_material(
            kwargs.get("static_friction", 0.5),
            kwargs.get("dynamic_friction", 0.5),
            kwargs.get("restitution", 0),
        )
        self.scene.set_ambient_light(kwargs.get("ambient_light", [0.5, 0.5, 0.5]))
        shadow = kwargs.get("shadow", True)
        direction_lights = kwargs.get(
            "direction_lights",
            [[[0, 0.5, -1], [0.5, 0.5, 0.5]]],
        )
        self._stage_l_direction_light_specs = []
        for direction_light in direction_lights:
            direction = list(direction_light[0])
            color = list(direction_light[1])
            if self.random_light:
                # Keep the exact NumPy draw order from Base_Task.setup_scene;
                # the lights themselves are materialized in the child.
                color = np.random.rand(3).tolist()
            self._stage_l_direction_light_specs.append(
                (direction, color, shadow)
            )
        point_lights = kwargs.get(
            "point_lights",
            [[[1, 0, 1.8], [1, 1, 1]], [[-1, 0, 1.8], [1, 1, 1]]],
        )
        self._stage_l_point_light_specs = []
        for point_light in point_lights:
            position = list(point_light[0])
            color = list(point_light[1])
            if self.random_light:
                color = np.random.rand(3).tolist()
            self._stage_l_point_light_specs.append((position, color, shadow))
        self.direction_light_lst = []
        self.point_light_lst = []
        self.viewer = None

    def _defer_camera_load(self, **kwargs):
        from envs.camera import Camera

        # Camera.__init__ has no renderer side effects, and doing it here
        # preserves its config mutation before actors are randomized.
        self.cameras = Camera(
            bias=self.table_z_bias,
            random_head_camera_dis=self.random_head_camera_dis,
            **kwargs,
        )
        self._stage_l_camera_rng_before = np.random.get_state()
        # Camera.load_camera consumes these draws while placing every static
        # camera, including the zero-distance case.
        for _camera_info in self.cameras.static_camera_info_list:
            np.random.randn(3)
            np.random.uniform(low=0, high=self.random_head_camera_dis)
        self._stage_l_camera_rng_after = np.random.get_state()
        # This is the physical step performed by the native load_camera.
        self.scene.step()
        self._stage_l_camera_kwargs = dict(kwargs)
        self._stage_l_camera_deferred = True

    def _load_robot_without_planner(self, **kwargs):
        # Curobo's native planner construction calls torch.Tensor.cuda().
        # Keep that CUDA initialization out of the parent so the child can
        # safely initialize both its renderer and planner after fork.
        Robot = sys.modules["envs._base_task"].__dict__["Robot"]
        if not hasattr(self, "robot"):
            self.robot = Robot(self.scene, self.need_topp, **kwargs)
        else:
            self.robot._init_robot_(self.scene, self.need_topp, **kwargs)
        self.robot.init_joints()
        class _StageLDeferredGripperPlanner:
            @staticmethod
            def plan_grippers(now_val, target_val):
                num_step = 200
                dis_val = target_val - now_val
                return {
                    "num_step": num_step,
                    "per_step": dis_val / num_step,
                    "result": np.linspace(now_val, target_val, num_step),
                }

        # Base_Task._init_task_env_ opens the grippers before the child can
        # materialize curobo.  Preserve the native gripper schedule locally;
        # child set_planner() replaces these placeholders before any policy
        # action is planned.
        self.robot.communication_flag = False
        self.robot.left_planner = _StageLDeferredGripperPlanner()
        self.robot.right_planner = _StageLDeferredGripperPlanner()
        for link in self.robot.left_entity.get_links():
            link.set_mass(1)
        for link in self.robot.right_entity.get_links():
            link.set_mass(1)

    task_env.setup_scene = MethodType(_setup_scene_without_renderer, task_env)
    task_env.load_robot = MethodType(_load_robot_without_planner, task_env)
    task_env.load_camera = MethodType(_defer_camera_load, task_env)
    task_env._stage_l_physics_only_setup = True


def _stage_l_install_deferred_planner_stub():
    """Prevent import-time curobo CUDA initialization in the fork parent."""

    module_name = "envs.robot.planner"
    existing = sys.modules.get(module_name)
    if existing is not None:
        raise RuntimeError(
            "deferred planner stub must be installed before envs.robot.planner "
            "is imported; import-time CUDA state would already be unsafe"
        )
    stub = types.ModuleType(module_name)
    stub.__stage_l_deferred_stub__ = True

    class _DeferredPlannerImportStub:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "planner construction must occur in the post-fork child"
            )

    stub.CuroboPlanner = _DeferredPlannerImportStub
    stub.MplibPlanner = _DeferredPlannerImportStub
    sys.modules[module_name] = stub


def _stage_l_materialize_task_planner(task_env):
    """Replace the parent-only planner stub with real curobo in a child."""

    import importlib

    module_name = "envs.robot.planner"
    stub = sys.modules.get(module_name)
    if stub is not None and getattr(stub, "__stage_l_deferred_stub__", False):
        del sys.modules[module_name]
        robot_package = sys.modules.get("envs.robot")
        if robot_package is not None and hasattr(robot_package, "planner"):
            delattr(robot_package, "planner")
        real_planner = importlib.import_module(module_name)
        robot_cls = type(task_env.robot)
        robot_globals = robot_cls.set_planner.__globals__
        robot_globals["CuroboPlanner"] = real_planner.CuroboPlanner
        robot_globals["MplibPlanner"] = real_planner.MplibPlanner
    task_env.robot.set_planner(task_env.scene)


def _stage_l_materialize_task_renderer(task_env):
    """Create the native renderer/cameras in a post-fork child."""

    if not getattr(task_env, "_stage_l_camera_deferred", False):
        return
    import numpy as np
    import sapien.render

    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(32)
    sapien.render.set_ray_tracing_path_depth(8)
    sapien.render.set_ray_tracing_denoiser("oidn")
    task_env.renderer = sapien.SapienRenderer()
    task_env.engine.set_renderer(task_env.renderer)
    task_env.direction_light_lst = [
        task_env.scene.add_directional_light(direction, color, shadow=shadow)
        for direction, color, shadow in task_env._stage_l_direction_light_specs
    ]
    task_env.point_light_lst = [
        task_env.scene.add_point_light(position, color, shadow=shadow)
        for position, color, shadow in task_env._stage_l_point_light_specs
    ]
    camera_rng_before = getattr(task_env, "_stage_l_camera_rng_before", None)
    camera_rng_after = getattr(task_env, "_stage_l_camera_rng_after", None)
    if camera_rng_before is not None:
        np.random.set_state(camera_rng_before)
    task_env.cameras.load_camera(task_env.scene)
    if camera_rng_after is not None:
        np.random.set_state(camera_rng_after)
    # load_camera's physical step already ran before fork; this is only the
    # renderer synchronization that was previously paired with that step.
    task_env.scene.update_render()
    task_env._stage_l_camera_deferred = False


@contextmanager
def _stage_m_teacher_query_lock(path: str | None):
    if not path:
        yield
        return
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(
    os.environ.get(
        "WAVE_RL_ROOT",
        os.environ.get("PROJECT_ROOT", str(WORKSPACE_ROOT.parent / "wave-rl")),
    )
).expanduser().resolve()
UPSTREAM = (
    PROJECT_ROOT
    / "third_party"
    / "lingbot-va"
    / "evaluation"
    / "robotwin"
    / "eval_polict_client_openpi.py"
)
NEEDLE = "instruction = np.random.choice(results[0][instruction_type])"
REPLACEMENT = (
    "instruction_candidates = results[0][instruction_type]\n"
    "        instruction = instruction_candidates[now_seed % len(instruction_candidates)]"
)
RESET_NEEDLE = (
    "model.infer(dict(reset = True, prompt=prompt, "
    "save_visualization=save_visualization))"
)
SEEDED_RESET_REPLACEMENT = (
    "model.infer(dict(reset=True, prompt=prompt, "
    "save_visualization=save_visualization, seed=now_seed))"
)


source = UPSTREAM.read_text(encoding="utf-8")
if os.environ.get("ROBOTWIN_TRACE_INITIAL_OBS", "0") == "1":
    _trace_replacements = {
        "        initial_obs = TASK_ENV.get_obs() \n": (
            "        log_stage(f\"initial get_obs start seed={now_seed}\")\n"
            "        initial_obs = TASK_ENV.get_obs()\n"
            "        log_stage(f\"initial get_obs done seed={now_seed}\")\n"
        ),
        "        initial_formatted_obs = format_obs(initial_obs, prompt)\n": (
            "        log_stage(f\"initial format_obs start seed={now_seed}\")\n"
            "        initial_formatted_obs = format_obs(initial_obs, prompt)\n"
            "        log_stage(f\"initial format_obs done seed={now_seed}\")\n"
        ),
        "            if first:\n"
        "                observation = TASK_ENV.get_obs()\n"
        "                first_obs = format_obs(observation, prompt)\n": (
            "            if first:\n"
            "                log_stage(f\"first-loop get_obs start seed={now_seed}\")\n"
            "                observation = TASK_ENV.get_obs()\n"
            "                log_stage(f\"first-loop get_obs done seed={now_seed}\")\n"
            "                log_stage(f\"first-loop format_obs start seed={now_seed}\")\n"
            "                first_obs = format_obs(observation, prompt)\n"
            "                log_stage(f\"first-loop format_obs done seed={now_seed}\")\n"
            "                if os.environ.get(\"ROBOTWIN_DIAG_EXIT_BEFORE_INFER\", \"0\") == \"1\":\n"
            "                    TASK_ENV.close_env()\n"
            "                    print(\"ROBOTWIN_DIAG_INITIAL_OBS_PASS\", flush=True)\n"
            "                    raise SystemExit(0)\n"
        ),
    }
    for _trace_needle, _trace_replacement in _trace_replacements.items():
        if source.count(_trace_needle) != 1:
            raise RuntimeError(
                "diagnostic initial-observation insertion changed: "
                + _trace_needle.strip()
            )
        source = source.replace(_trace_needle, _trace_replacement)
pair_manifest = os.environ.get("PAIR_MANIFEST")
pair_recheck_expert = (
    os.environ.get("ROBOTWIN_PAIR_RECHECK_EXPERT", "0") == "1"
)
planner_prefix_trace_dir = os.environ.get("ROBOTWIN_PREFIX_TRACE_DIR")
context_capture_root = os.environ.get("ROBOTWIN_CONTEXT_CAPTURE_ROOT")
expected_context_path = os.environ.get("ROBOTWIN_EXPECTED_CONTEXT")
setup_audit_only = os.environ.get("ROBOTWIN_SETUP_AUDIT_ONLY", "0") == "1"
snapshot_roundtrip_output = os.environ.get(
    "ROBOTWIN_SNAPSHOT_ROUNDTRIP_OUTPUT"
)
stage_l_oracle_output = os.environ.get("ROBOTWIN_STAGE_L_ORACLE_OUTPUT")
stage_l_oracle_intervention_frame = int(
    os.environ.get("ROBOTWIN_STAGE_L_INTERVENTION_FRAME", "-1")
)
stage_l_oracle_repeats = int(
    os.environ.get("ROBOTWIN_STAGE_L_ORACLE_REPEATS", "3")
)
stage_l_fork_clone_output = os.environ.get(
    "ROBOTWIN_STAGE_L_FORK_CLONE_OUTPUT"
)
stage_l_fork_early_output = os.environ.get(
    "ROBOTWIN_STAGE_L_FORK_EARLY_OUTPUT"
)
stage_l_fork_early_enabled = bool(stage_l_fork_early_output)
stage_l_fork_early_child = False
stage_l_fork_early_child_output = None
stage_l_fork_early_child_branch = "A"
stage_l_fork_early_child_repeat = -1
stage_l_fork_clone_enabled = bool(
    stage_l_fork_clone_output and not stage_l_fork_early_enabled
)
stage_l_branch_b_artifact = os.environ.get(
    "ROBOTWIN_STAGE_L_BRANCH_B_ARTIFACT"
)
stage_l_branch_b_key = os.environ.get(
    "ROBOTWIN_STAGE_L_BRANCH_B_KEY", "teacher_bridge_env_action"
)
stage_m_context_output = os.environ.get("ROBOTWIN_STAGE_M_CONTEXT_OUTPUT")
stage_m_label_output = os.environ.get("ROBOTWIN_STAGE_M_LABEL_OUTPUT")
stage_m_runtime_audit_output = os.environ.get(
    "ROBOTWIN_STAGE_M_RUNTIME_AUDIT_OUTPUT"
)
stage_m_student_checkpoint = os.environ.get(
    "ROBOTWIN_STAGE_M_STUDENT_CHECKPOINT"
)
stage_m_teacher_transformer = os.environ.get(
    "ROBOTWIN_STAGE_M_TEACHER_TRANSFORMER"
)
stage_m_teacher_python = os.environ.get("ROBOTWIN_STAGE_M_TEACHER_PYTHON")
stage_m_teacher_lock_path = os.environ.get("ROBOTWIN_STAGE_M_TEACHER_LOCK")
stage_m_diffusion_seed = int(
    os.environ.get("ROBOTWIN_STAGE_M_DIFFUSION_SEED", "-1")
)
stage_m_teacher_gpu = int(os.environ.get("ROBOTWIN_STAGE_M_TEACHER_GPU", "7"))
stage_m_enabled = bool(stage_m_context_output)
stage_n_treatment_output = os.environ.get("ROBOTWIN_STAGE_N_TREATMENT_OUTPUT")
stage_n_treatment_arm = os.environ.get("ROBOTWIN_STAGE_N_TREATMENT_ARM")
stage_n_alpha_text = os.environ.get("ROBOTWIN_STAGE_N_ALPHA")
stage_n_alpha = float(stage_n_alpha_text) if stage_n_alpha_text else None
stage_n_enabled = bool(stage_n_treatment_output)
stage_o_trace_output = os.environ.get("ROBOTWIN_STAGE_O_TRACE_OUTPUT")
stage_o_enabled = bool(stage_o_trace_output)
stage_o_dynamic_treatment = (
    os.environ.get("ROBOTWIN_STAGE_O_DYNAMIC_TREATMENT", "0") == "1"
)
stage_o_event_output = os.environ.get("ROBOTWIN_STAGE_O_EVENT_OUTPUT")
stage_o_dynamic_trace_output = os.environ.get(
    "ROBOTWIN_STAGE_O_DYNAMIC_TRACE_OUTPUT"
)
stage_o_event_config = _stage_o_event_config_type(
    min_frame=int(os.environ.get("ROBOTWIN_STAGE_O_MIN_FRAME", "10")),
    window_samples=int(os.environ.get("ROBOTWIN_STAGE_O_WINDOW_SAMPLES", "3")),
    continuous_epsilon=float(
        os.environ.get("ROBOTWIN_STAGE_O_CONTINUOUS_EPSILON", "0.01")
    ),
    cooldown_frames=int(
        os.environ.get("ROBOTWIN_STAGE_O_COOLDOWN_FRAMES", "12")
    ),
    max_events=int(os.environ.get("ROBOTWIN_STAGE_O_MAX_EVENTS", "2")),
)
stage_o_observable_config = _stage_o_observable_config_type(
    min_frame=stage_o_event_config.min_frame,
    window_samples=stage_o_event_config.window_samples,
    image_mean_threshold=float(
        os.environ.get("ROBOTWIN_STAGE_O_IMAGE_MEAN_THRESHOLD", "0.05")
    ),
    state_rmse_threshold=float(
        os.environ.get("ROBOTWIN_STAGE_O_STATE_RMSE_THRESHOLD", "0.05")
    ),
    cooldown_frames=stage_o_event_config.cooldown_frames,
    max_events=stage_o_event_config.max_events,
)
stage_n_intervention_frames = (
    _stage_n_parse_frames(
        os.environ.get("ROBOTWIN_STAGE_N_INTERVENTION_FRAMES"),
        default_frame=stage_l_oracle_intervention_frame,
    )
    if stage_n_enabled
    else ()
)
stage_n_expected_treatments = len(stage_n_intervention_frames)
if stage_m_enabled:
    required_stage_m = {
        "ROBOTWIN_STAGE_M_LABEL_OUTPUT": stage_m_label_output,
        "ROBOTWIN_STAGE_M_RUNTIME_AUDIT_OUTPUT": stage_m_runtime_audit_output,
        "ROBOTWIN_STAGE_M_STUDENT_CHECKPOINT": stage_m_student_checkpoint,
        "ROBOTWIN_STAGE_M_TEACHER_TRANSFORMER": stage_m_teacher_transformer,
        "ROBOTWIN_STAGE_M_TEACHER_PYTHON": stage_m_teacher_python,
    }
    missing_stage_m = [key for key, value in required_stage_m.items() if not value]
    if missing_stage_m:
        raise ValueError(f"Stage-M live oracle missing {missing_stage_m}")
    if not stage_l_oracle_output or not stage_l_branch_b_artifact:
        raise ValueError("Stage-M live oracle requires the Stage-L branch harness")
    if Path(stage_l_branch_b_artifact).resolve() != Path(stage_m_label_output).resolve():
        raise ValueError("Stage-M label output must be the branch-B artifact")
    if stage_m_diffusion_seed < 0:
        raise ValueError("Stage-M diffusion seed must be non-negative")
    if stage_m_teacher_gpu != 7:
        raise ValueError("Stage-M synchronous Teacher is restricted to GPU7")
if stage_n_enabled:
    if not stage_m_enabled:
        raise ValueError("Stage-N single treatment requires the Stage-M live label")
    if stage_n_treatment_arm not in ("base", "correction"):
        raise ValueError("Stage-N treatment arm must be base or correction")
    if stage_n_treatment_arm == "base" and stage_n_alpha is not None:
        raise ValueError("Stage-N base treatment must not set alpha")
    if stage_n_treatment_arm == "correction" and stage_n_alpha is None:
        raise ValueError("Stage-N correction treatment requires alpha")
    if not stage_m_teacher_lock_path:
        raise ValueError("Stage-N requires a shared GPU7 Teacher lock")
if stage_o_enabled:
    if stage_n_enabled or stage_l_oracle_output:
        raise ValueError(
            "Stage-O passive trace currently requires a released-Student rollout"
        )
    if os.environ.get("ROBOTWIN_LOG_STAGE_H_PROGRESS", "0") != "1":
        raise ValueError("Stage-O passive trace requires Stage-H progress logging")
if stage_o_dynamic_treatment:
    if not stage_n_enabled or not stage_l_oracle_output:
        raise ValueError(
            "Stage-O dynamic treatment requires the Stage-N live-label path"
        )
    if stage_o_enabled:
        raise ValueError("Stage-O passive trace and dynamic treatment cannot overlap")
    if not stage_o_event_output:
        raise ValueError("Stage-O dynamic treatment requires an event output")
    if not stage_o_dynamic_trace_output:
        raise ValueError("Stage-O dynamic treatment requires a trace output")
    if stage_o_observable_config.max_events != 1:
        raise ValueError("Stage-O first dynamic screen permits exactly one event")
if snapshot_roundtrip_output and not setup_audit_only:
    raise ValueError("snapshot round-trip currently requires setup-audit-only mode")
if stage_l_oracle_output:
    if setup_audit_only or snapshot_roundtrip_output:
        raise ValueError("Stage-L branch oracle requires normal evaluation mode")
    if stage_l_oracle_intervention_frame <= 0:
        raise ValueError("Stage-L intervention frame must be positive")
    if stage_l_oracle_intervention_frame % 2:
        raise ValueError("Stage-L intervention frame must align to 2-frame chunks")
    if stage_l_oracle_repeats < 1:
        raise ValueError("Stage-L oracle repeats must be positive")
if stage_l_fork_clone_enabled:
    if not stage_l_oracle_output:
        raise ValueError("fork clone requires Stage-L oracle output")
    if not hasattr(os, "fork"):
        raise ValueError("fork clone requires POSIX os.fork")
if stage_l_fork_early_enabled:
    if not stage_l_oracle_output:
        raise ValueError("early fork clone requires Stage-L oracle output")
    if not hasattr(os, "fork"):
        raise ValueError("early fork clone requires POSIX os.fork")
if stage_l_branch_b_artifact and not stage_l_oracle_output:
    raise ValueError("Stage-L branch-B artifact requires the branch oracle")
pair_warmup_seeds = [
    int(value)
    for value in os.environ.get("ROBOTWIN_PAIR_WARMUP_SEEDS", "").split(",")
    if value
]
pair_warmup_expected_unstable_seeds = {
    int(value)
    for value in os.environ.get(
        "ROBOTWIN_PAIR_WARMUP_EXPECT_UNSTABLE_SEEDS", ""
    ).split(",")
    if value
}
if not pair_warmup_expected_unstable_seeds.issubset(set(pair_warmup_seeds)):
    raise ValueError("expected-unstable seeds must be present in pair warm-ups")
if (expected_context_path or setup_audit_only or pair_warmup_seeds) and not pair_manifest:
    raise ValueError("Stage H context diagnostics require PAIR_MANIFEST")
if stage_l_oracle_output and not pair_manifest:
    raise ValueError("Stage-L branch oracle requires PAIR_MANIFEST")
expected_pair_observation = None
if expected_context_path:
    expected_pair_context = torch.load(
        expected_context_path,
        map_location="cpu",
        weights_only=False,
    )
    expected_pair_observation = expected_pair_context["initial_observation"]
if planner_prefix_trace_dir:
    if not pair_manifest:
        raise ValueError("strict planner replay requires PAIR_MANIFEST")
    with Path(pair_manifest).open("r", encoding="utf-8") as handle:
        trace_episode_records = json.load(handle)["episode_records"]
    if len(trace_episode_records) != 1:
        raise ValueError("strict planner replay prototype requires one episode")

    trace_needle = "from envs.utils.create_actor import UnStableError\n"
    trace_replacement = """from envs.utils.create_actor import UnStableError
import envs.robot.robot as _robotwin_robot
import numpy as _planner_np
from pathlib import Path as _PlannerPath

_prefix_trace_dir = _PlannerPath(os.environ["ROBOTWIN_PREFIX_TRACE_DIR"])
_prefix_calls_per_arm = int(os.environ["ROBOTWIN_PREFIX_CALLS"])
_branch_trace_mode = os.environ["ROBOTWIN_BRANCH_TRACE_MODE"]
_branch_trace_dir = _PlannerPath(os.environ["ROBOTWIN_BRANCH_TRACE_DIR"])
if _branch_trace_mode not in ("record", "replay"):
    raise ValueError("ROBOTWIN_BRANCH_TRACE_MODE must be record or replay")
if _branch_trace_mode == "record":
    _branch_trace_dir.mkdir(parents=True, exist_ok=True)
_planner_trace_counts = {"left": 0, "right": 0}

def _strict_traced_plan(arm, original):
    def traced(self, target_pose, *args, **kwargs):
        absolute_index = _planner_trace_counts[arm]
        _planner_trace_counts[arm] += 1
        if absolute_index < _prefix_calls_per_arm:
            mode = "replay"
            trace_dir = _prefix_trace_dir
            trace_index = absolute_index
        else:
            mode = _branch_trace_mode
            trace_dir = _branch_trace_dir
            trace_index = absolute_index - _prefix_calls_per_arm

        entity = self.left_entity if arm == "left" else self.right_entity
        start_qpos = _planner_np.asarray(entity.get_qpos()).copy()
        path = trace_dir / f"{arm}_{trace_index:04d}.npz"
        if mode == "replay":
            saved = _planner_np.load(path)
            target_array = _planner_np.asarray(target_pose)
            if not _planner_np.array_equal(start_qpos, saved["start_qpos"]):
                raise AssertionError(
                    f"{arm} call {trace_index} start qpos differs; "
                    f"max_abs={_planner_np.max(_planner_np.abs(start_qpos - saved['start_qpos']))}"
                )
            if not _planner_np.array_equal(target_array, saved["target_pose"]):
                raise AssertionError(
                    f"{arm} call {trace_index} target differs; "
                    f"max_abs={_planner_np.max(_planner_np.abs(target_array - saved['target_pose']))}"
                )
            return {
                "status": str(saved["status"].item()),
                "position": saved["position"],
                "velocity": saved["velocity"],
            }

        result = original(self, target_pose, *args, **kwargs)
        _planner_np.savez(
            path,
            start_qpos=start_qpos,
            target_pose=_planner_np.asarray(target_pose).copy(),
            status=_planner_np.asarray(str(result["status"])),
            position=_planner_np.asarray(result.get("position", [])),
            velocity=_planner_np.asarray(result.get("velocity", [])),
        )
        return result
    return traced

_robotwin_robot.Robot.left_plan_path = _strict_traced_plan(
    "left", _robotwin_robot.Robot.left_plan_path
)
_robotwin_robot.Robot.right_plan_path = _strict_traced_plan(
    "right", _robotwin_robot.Robot.right_plan_path
)
print(
    "PROTOTYPE_STRICT_PLANNER_REPLAY "
    f"prefix_calls={_prefix_calls_per_arm} branch_mode={_branch_trace_mode}",
    flush=True,
)
"""
    if source.count(trace_needle) != 1:
        raise RuntimeError("Upstream planner import changed")
    source = source.replace(trace_needle, trace_replacement)
if os.environ.get("ROBOTWIN_ENHANCED_DETERMINISM", "0") == "1":
    import_needle = "from envs import CONFIGS_PATH\n"
    import_replacement = """from envs import CONFIGS_PATH
import envs._base_task as _robotwin_base_task

_original_scene_config = _robotwin_base_task.sapien.SceneConfig

def _enhanced_scene_config(*args, **kwargs):
    config = _original_scene_config(*args, **kwargs)
    config.enable_enhanced_determinism = True
    return config

_robotwin_base_task.sapien.SceneConfig = _enhanced_scene_config
_original_init_task_env = _robotwin_base_task.Base_Task._init_task_env_

def _seeded_init_task_env(self, *args, **kwargs):
    _robotwin_base_task.random.seed(kwargs.get("seed", 0))
    return _original_init_task_env(self, *args, **kwargs)

_robotwin_base_task.Base_Task._init_task_env_ = _seeded_init_task_env
print("PROTOTYPE_ROBOTWIN_ENHANCED_DETERMINISM=1", flush=True)
"""
    if source.count(import_needle) != 1:
        raise RuntimeError("Upstream CONFIGS_PATH import changed")
    source = source.replace(import_needle, import_replacement)
if context_capture_root:
    context_init_needle = """        initial_formatted_obs = format_obs(initial_obs, prompt)
        if save_video:
"""
    context_init_replacement = """        initial_formatted_obs = format_obs(initial_obs, prompt)
        _stage_g_context_chunks = []
        _stage_g_frame_st_id = 0
        _stage_i_context_stop_frame = int(
            os.environ.get("ROBOTWIN_CONTEXT_CAPTURE_STOP_FRAME", "-1")
        )
        if save_video:
"""
    if source.count(context_init_needle) != 1:
        raise RuntimeError("Upstream Stage G context initialization changed")
    source = source.replace(context_init_needle, context_init_replacement)

    context_chunk_needle = """            model.infer(dict(obs = key_frame_list, compute_kv_cache=True, imagine=False, save_visualization=save_visualization, state=action))

            if TASK_ENV.eval_success:
"""
    context_chunk_replacement = """            model.infer(dict(obs = key_frame_list, compute_kv_cache=True, imagine=False, save_visualization=save_visualization, state=action))
            _stage_g_context_chunks.append({
                "frame_st_id": int(_stage_g_frame_st_id),
                "observations": key_frame_list,
                "env_action": np.asarray(action).copy(),
            })
            _stage_g_frame_st_id += int(action.shape[1])
            if (
                _stage_i_context_stop_frame >= 0
                and _stage_g_frame_st_id >= _stage_i_context_stop_frame
            ):
                print(
                    "STAGE_I_CONTEXT_CAPTURE_EARLY_STOP "
                    f"frame_st_id={_stage_g_frame_st_id}",
                    flush=True,
                )
                break

            if TASK_ENV.eval_success:
"""
    if source.count(context_chunk_needle) != 1:
        raise RuntimeError("Upstream Stage G context chunk hook changed")
    source = source.replace(context_chunk_needle, context_chunk_replacement)

    context_save_needle = """        if save_video:
            vis_dir = Path(args['save_root']) / f'stseed-{st_seed}' / 'visualization' / task_name
"""
    context_save_replacement = """        _stage_g_context_root = Path(os.environ["ROBOTWIN_CONTEXT_CAPTURE_ROOT"])
        _stage_g_context_root.mkdir(parents=True, exist_ok=True)
        _stage_g_context_path = _stage_g_context_root / (
            f"{task_name}_{args['ckpt_setting']}_seed{int(now_seed)}.pt"
        )
        torch.save({
            "schema": "flashwam_stage_g_rollout_context_v1",
            "task": task_name,
            "task_config": args["task_config"],
            "seed": int(now_seed),
            "prompt": prompt,
            "model_name": str(args["ckpt_setting"]),
            "policy_delta_path": (
                os.environ.get("ROBOTWIN_POLICY_DELTA_PATH") or None
            ),
            "initial_observation": initial_formatted_obs,
            "chunks": _stage_g_context_chunks,
        }, _stage_g_context_path)
        print(f"STAGE_G_CONTEXT_CAPTURE path={_stage_g_context_path}", flush=True)

        if save_video:
            vis_dir = Path(args['save_root']) / f'stseed-{st_seed}' / 'visualization' / task_name
"""
    if source.count(context_save_needle) != 1:
        raise RuntimeError("Upstream Stage G context save hook changed")
    source = source.replace(context_save_needle, context_save_replacement)
if (
    os.environ.get("ROBOTWIN_LOG_SWITCH_PROGRESS", "0") == "1"
    and os.environ.get("ROBOTWIN_LOG_STAGE_H_PROGRESS", "0") == "1"
):
    raise ValueError("switch and Stage H progress hooks are mutually exclusive")
if os.environ.get("ROBOTWIN_LOG_SWITCH_PROGRESS", "0") == "1":
    record_needle = """        episode_records.append({
            "episode": int(TASK_ENV.test_num),
            "seed": int(now_seed),
            "success": bool(succ),
            "take_action_cnt": int(getattr(TASK_ENV, "take_action_cnt", -1)),
            "instruction": prompt,
        })
"""
    record_replacement = """        _switch_qpos = float(TASK_ENV.switch.get_qpos()[0])
        _switch_limits = np.asarray(TASK_ENV.switch.get_qlimits()[0], dtype=np.float64)
        episode_records.append({
            "episode": int(TASK_ENV.test_num),
            "seed": int(now_seed),
            "success": bool(succ),
            "take_action_cnt": int(getattr(TASK_ENV, "take_action_cnt", -1)),
            "instruction": prompt,
            "switch_qpos": _switch_qpos,
            "switch_normalized_progress": float(
                (_switch_qpos - _switch_limits[0])
                / max(_switch_limits[1] - _switch_limits[0], 1e-12)
            ),
        })
"""
    if source.count(record_needle) != 1:
        raise RuntimeError("Upstream episode record block changed")
    source = source.replace(record_needle, record_replacement)
elif os.environ.get("ROBOTWIN_LOG_STAGE_H_PROGRESS", "0") == "1":
    record_needle = """        episode_records.append({
            "episode": int(TASK_ENV.test_num),
            "seed": int(now_seed),
            "success": bool(succ),
            "take_action_cnt": int(getattr(TASK_ENV, "take_action_cnt", -1)),
            "instruction": prompt,
        })
"""
    record_replacement = """        _stage_h_terminal_progress = collect_task_progress(
            args["task_name"], TASK_ENV
        )
        if stage_n_enabled:
            _stage_o_no_event_terminal = bool(
                stage_o_dynamic_treatment
                and _stage_n_applied_count == 0
                and len(_stage_n_treatment_records) == 0
                and _stage_n_pending_treatment is None
            )
            if _stage_o_no_event_terminal:
                _stage_n_schedule_completion = {
                    "application_count": 0,
                    "application_expected": stage_n_expected_treatments,
                    "early_terminal": bool(succ),
                    "skipped_treatments": stage_n_expected_treatments,
                    "no_event_terminal": True,
                    "valid_treatment_block": False,
                }
                _stage_n_continuation_events = (
                    _stage_n_planner_prefix_trace.event_snapshot()
                )
                _stage_n_planner_prefix_trace.finish()
            else:
                _stage_n_schedule_completion = _stage_n_validate_schedule(
                    applied_count=_stage_n_applied_count,
                    recorded_count=len(_stage_n_treatment_records),
                    expected_count=stage_n_expected_treatments,
                    success=bool(succ),
                    pending=(_stage_n_pending_treatment is not None),
                )
                if _stage_n_continuation_planner_trace is None:
                    raise AssertionError(
                        "Stage-N planner event trace was not installed"
                    )
                _stage_n_continuation_events = (
                    _stage_n_continuation_planner_trace.finish()
                )
            _stage_n_safety_events = {
                "task_plan_success": bool(
                    getattr(TASK_ENV, "plan_success", False)
                ),
                "collision_counter_available": False,
                "collision_events": None,
            }
            _stage_n_outcome = {
                "success": bool(succ),
                "completion_step": int(
                    getattr(TASK_ENV, "take_action_cnt", -1)
                ),
                "progress": _stage_h_terminal_progress,
            }
            if _stage_o_no_event_terminal:
                _stage_n_treatment_output_record = {
                    "schema": "flashwam_stage_o_no_event_terminal_v1",
                    "task": args["task_name"],
                    "task_config": args["task_config"],
                    "seed": int(now_seed),
                    "prompt": prompt,
                    "requested_arm": stage_n_treatment_arm,
                    "requested_alpha": (
                        0.0
                        if stage_n_treatment_arm == "base"
                        else float(stage_n_alpha)
                    ),
                    "application_count": 0,
                    "valid_treatment_block": False,
                    "event": None,
                    "continuation_policy": "frozen_current_student",
                    "deployment_use": "training_or_evaluation_only",
                }
            elif stage_n_expected_treatments == 1:
                _stage_n_treatment_output_record = _stage_n_treatment_record
            else:
                _stage_n_treatment_output_record = {
                    "schema": "flashwam_stage_n_multi_treatment_v1",
                    "task": args["task_name"],
                    "task_config": args["task_config"],
                    "seed": int(now_seed),
                    "prompt": prompt,
                    "arm": stage_n_treatment_arm,
                    "alpha": (
                        0.0
                        if stage_n_treatment_arm == "base"
                        else float(stage_n_alpha)
                    ),
                    "intervention_frames": list(stage_n_intervention_frames),
                    "application_count": _stage_n_applied_count,
                    "schedule_completion": _stage_n_schedule_completion,
                    "treatments": _stage_n_treatment_records,
                    "continuation_policy": "frozen_current_student",
                    "deployment_use": "training_or_evaluation_only",
                }
            _stage_n_treatment_output_record["schedule_completion"] = (
                _stage_n_schedule_completion
            )
            _stage_n_treatment_output_record["continuation_planner_events"] = (
                _stage_n_continuation_events
            )
            _stage_n_treatment_output_record["safety_events"] = (
                _stage_n_safety_events
            )
            _stage_n_treatment_output_record["outcome"] = _stage_n_outcome
            if stage_o_dynamic_treatment:
                _stage_o_dynamic_trace_payload["status"] = (
                    "no_event_terminal"
                    if _stage_o_no_event_terminal
                    else "completed_event_treatment"
                )
                _stage_o_dynamic_trace_payload["outcome"] = _stage_n_outcome
                Path(stage_o_dynamic_trace_output).write_text(
                    json.dumps(
                        _stage_o_dynamic_trace_payload, indent=2
                    )
                    + "\\n",
                    encoding="utf-8",
                )
                if _stage_o_no_event_terminal:
                    _stage_o_no_event_record = {
                        "schema": "flashwam_stage_o_no_event_terminal_v1",
                        "task": args["task_name"],
                        "task_config": args["task_config"],
                        "seed": int(now_seed),
                        "requested_arm": stage_n_treatment_arm,
                        "requested_alpha": (
                            0.0
                            if stage_n_treatment_arm == "base"
                            else float(stage_n_alpha)
                        ),
                        "valid_treatment_block": False,
                        "outcome": _stage_n_outcome,
                    }
                    Path(stage_o_event_output).write_text(
                        json.dumps(
                            _stage_o_no_event_record, indent=2
                        )
                        + "\\n",
                        encoding="utf-8",
                    )
            Path(stage_n_treatment_output).write_text(
                json.dumps(_stage_n_treatment_output_record, indent=2) + "\\n",
                encoding="utf-8",
            )
        _stage_h_episode_record = {
            "episode": int(TASK_ENV.test_num),
            "seed": int(now_seed),
            "success": bool(succ),
            "take_action_cnt": int(getattr(TASK_ENV, "take_action_cnt", -1)),
            "instruction": prompt,
            "stage_h_progress": _stage_h_terminal_progress,
        }
        if stage_n_enabled:
            _stage_h_episode_record["stage_n_treatment"] = (
                _stage_n_treatment_output_record
            )
        if stage_o_enabled:
            _stage_o_final_planner_events = _stage_o_planner_trace.finish()
            _stage_o_trace_payload = _stage_o_trace.finalize(
                success=bool(succ),
                completion_step=int(getattr(TASK_ENV, "take_action_cnt", -1)),
                final_progress=_stage_h_terminal_progress,
            )
            _stage_o_trace_payload["terminal_planner_events"] = (
                _stage_o_final_planner_events
            )
            _stage_o_trace_path = Path(stage_o_trace_output)
            _stage_o_trace_path.parent.mkdir(parents=True, exist_ok=True)
            _stage_o_trace_path.write_text(
                json.dumps(_stage_o_trace_payload, indent=2) + "\\n",
                encoding="utf-8",
            )
            _stage_h_episode_record["stage_o_event_trace"] = str(
                _stage_o_trace_path.resolve()
            )
            print(
                "STAGE_O_EVENT_TRACE "
                + json.dumps(
                    {
                        "output": str(_stage_o_trace_path.resolve()),
                        "samples": len(_stage_o_trace_payload["samples"]),
                        "events": len(_stage_o_trace_payload["events"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        episode_records.append(_stage_h_episode_record)
"""
    if source.count(record_needle) != 1:
        raise RuntimeError("Upstream Stage H episode record block changed")
    source = source.replace(record_needle, record_replacement)
if pair_manifest:
    with Path(pair_manifest).open("r", encoding="utf-8") as handle:
        _paired_episode_records = json.load(handle)["episode_records"]
    print(
        "PROTOTYPE_PAIR_SETUP_HISTORY "
        f"recheck_expert={str(pair_recheck_expert).lower()}",
        flush=True,
    )
    replacements = {
        "expert_check = True": (
            "expert_check = True"
            if pair_recheck_expert
            else "expert_check = False"
        ),
        "while succ_seed < test_num:\n"
        "        render_freq = args[\"render_freq\"]": (
            "while succ_seed < test_num:\n"
            "        now_seed = int(_paired_episode_records[now_id][\"seed\"])\n"
            "        render_freq = args[\"render_freq\"]"
        ),
        "episode_info_list = [episode_info[\"info\"]]\n"
        "        results = generate_episode_descriptions(args[\"task_name\"], episode_info_list, test_num)\n"
        "        instruction = np.random.choice(results[0][instruction_type])": (
            (
                "instruction = _paired_episode_records[now_id][\"instruction\"]"
                if pair_recheck_expert
                else (
                    "episode_info = {\"info\": {}}\n"
                    "        episode_info_list = [episode_info[\"info\"]]\n"
                    "        instruction = _paired_episode_records[now_id][\"instruction\"]"
                )
            )
        ),
        RESET_NEEDLE: SEEDED_RESET_REPLACEMENT,
    }
    for needle, replacement in replacements.items():
        if source.count(needle) != 1:
            raise RuntimeError(f"Upstream paired-eval insertion changed: {needle}")
        source = source.replace(needle, replacement)
    if setup_audit_only:
        model_needle = "    model = WebsocketClientPolicy(port=usr_args['port'])\n"
        model_replacement = """    class _StageHSetupAuditPolicy:
        def infer(self, payload):
            if payload.get("reset"):
                return {}
            raise AssertionError(
                "setup-only context audit unexpectedly reached policy inference"
            )
    model = _StageHSetupAuditPolicy()
"""
        if source.count(model_needle) != 1:
            raise RuntimeError("Upstream setup-only policy insertion changed")
        source = source.replace(model_needle, model_replacement)
    if pair_warmup_seeds:
        warmup_needle = '    args["eval_mode"] = True\n'
        warmup_replacement = warmup_needle + """
    for _stage_h_warmup_seed in pair_warmup_seeds:
        log_stage(
            f"Stage H setup warm-up start seed={_stage_h_warmup_seed}"
        )
        try:
            TASK_ENV.setup_demo(
                now_ep_num=-1,
                seed=_stage_h_warmup_seed,
                is_test=True,
                **args,
            )
        except UnStableError:
            if _stage_h_warmup_seed not in pair_warmup_expected_unstable_seeds:
                raise
            log_stage(
                "Stage H setup warm-up expected unstable "
                f"seed={_stage_h_warmup_seed}"
            )
        else:
            if _stage_h_warmup_seed in pair_warmup_expected_unstable_seeds:
                raise AssertionError(
                    "Stage H setup warm-up unexpectedly succeeded for "
                    f"expected-unstable seed={_stage_h_warmup_seed}"
                )
        finally:
            TASK_ENV.close_env()
        log_stage(
            f"Stage H setup warm-up done seed={_stage_h_warmup_seed}"
        )
"""
        if source.count(warmup_needle) != 1:
            raise RuntimeError("Upstream paired warm-up insertion changed")
        source = source.replace(warmup_needle, warmup_replacement)
    pair_setup_needle = (
        '        log_stage(f"eval setup_demo done seed={now_seed} '
        'elapsed={time.perf_counter() - stage_t0:.2f}s")\n'
    )
    pair_setup_replacement = pair_setup_needle + (
        "        _pair_task_state_audit = initialize_pair_task_state(\n"
        "            args[\"task_name\"], TASK_ENV\n"
        "        )\n"
        "        print(\n"
        "            \"PROTOTYPE_PAIR_TASK_STATE_INIT \"\n"
        "            + json.dumps(_pair_task_state_audit, sort_keys=True),\n"
        "            flush=True,\n"
        "        )\n"
    )
    if source.count(pair_setup_needle) != 1:
        raise RuntimeError("Upstream paired task-state insertion changed")
    source = source.replace(pair_setup_needle, pair_setup_replacement)
    if stage_l_fork_early_enabled:
        eval_setup_needle = (
            '        stage_t0 = time.perf_counter()\n'
            '        log_stage(f"eval setup_demo start seed={now_seed} episode={now_id}")\n'
            '        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n'
        )
        eval_setup_replacement = (
            '        stage_t0 = time.perf_counter()\n'
            '        log_stage(f"eval setup_demo start seed={now_seed} episode={now_id}")\n'
            '        if os.environ.get("ROBOTWIN_STAGE_L_FORK_EARLY_CHILD", "0") != "1":\n'
            '            _stage_l_install_physics_only_task_env(TASK_ENV)\n'
            '        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n'
        )
        if source.count(eval_setup_needle) != 1:
            raise RuntimeError("early fork eval setup insertion changed")
        source = source.replace(eval_setup_needle, eval_setup_replacement)
        early_setup_needle = (
            '        instruction = _paired_episode_records[now_id]["instruction"]\n'
        )
        early_setup_replacement = """        if (
            stage_l_fork_early_enabled
            and os.environ.get("ROBOTWIN_STAGE_L_FORK_EARLY_CHILD", "0") != "1"
        ):
            _stage_l_fork_parent_root = Path(stage_l_fork_early_output)
            _stage_l_fork_parent_root.parent.mkdir(parents=True, exist_ok=True)
            _stage_l_fork_parent_setup_path = _stage_l_fork_parent_root.with_name(
                _stage_l_fork_parent_root.stem + ".parent_setup.pt"
            )
            _stage_l_fork_parent_setup = capture_simulator_snapshot(
                TASK_ENV, capture_cuda_rng=False
            )
            print(
                "PROTOTYPE_STAGE_L_FORK_PARENT_CUDA "
                + json.dumps(
                    {
                        "torch_cuda_available": bool(torch.cuda.is_available()),
                        "torch_cuda_initialized": bool(torch.cuda.is_initialized()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            torch.save(_stage_l_fork_parent_setup, _stage_l_fork_parent_setup_path)
            _stage_l_fork_child_specs = []
            for _stage_l_fork_repeat in range(stage_l_oracle_repeats):
                _stage_l_fork_child_specs.append(("A", _stage_l_fork_repeat))
                if stage_l_branch_b_artifact:
                    _stage_l_fork_child_specs.append(("B", _stage_l_fork_repeat))
            _stage_l_fork_child_paths = []
            for _stage_l_fork_child_branch_spec, _stage_l_fork_child_repeat_spec in _stage_l_fork_child_specs:
                _stage_l_fork_child_path = _stage_l_fork_parent_root.with_name(
                    _stage_l_fork_parent_root.stem
                    + f".child_{_stage_l_fork_child_branch_spec}_repeat{_stage_l_fork_child_repeat_spec:02d}.pt"
                )
                _stage_l_fork_child_pid = os.fork()
                if _stage_l_fork_child_pid == 0:
                    os.environ["ROBOTWIN_STAGE_L_FORK_EARLY_CHILD"] = "1"
                    def _stage_l_fork_child_excepthook(
                        _exc_type, _exc_value, _exc_traceback
                    ):
                        traceback.print_exception(
                            _exc_type, _exc_value, _exc_traceback
                        )
                        os._exit(1)

                    sys.excepthook = _stage_l_fork_child_excepthook
                    stage_l_fork_early_child_output = str(_stage_l_fork_child_path)
                    stage_l_fork_early_child_branch = _stage_l_fork_child_branch_spec
                    stage_l_fork_early_child_repeat = _stage_l_fork_child_repeat_spec
                    # ``eval_policy`` receives a WebsocketClientPolicy that
                    # was constructed before fork.  websockets.sync stores
                    # protocol mutexes and a receiver thread in that object;
                    # after fork the thread does not exist in the child while
                    # its lock may remain held, so the first infer can block
                    # forever in futex_wait_queue_me.  Close only the
                    # inherited socket fd (never the close-handshake API,
                    # which would try to acquire the inherited mutex), then
                    # establish an independent client connection in the
                    # child.  The parent keeps its own connection and waits
                    # for this child, so the two model/cache owners remain
                    # process-separated.
                    _stage_l_inherited_ws = getattr(model, "_ws", None)
                    _stage_l_inherited_socket = getattr(
                        _stage_l_inherited_ws, "socket", None
                    )
                    if _stage_l_inherited_socket is not None:
                        try:
                            _stage_l_inherited_socket.close()
                        except OSError:
                            pass
                    _stage_l_inherited_uri = str(
                        getattr(model, "_uri", "ws://127.0.0.1:8000")
                    )
                    _stage_l_child_port = int(
                        _stage_l_inherited_uri.rsplit(":", 1)[1]
                    )
                    model = WebsocketClientPolicy(
                        host="127.0.0.1", port=_stage_l_child_port
                    )
                    print(
                        "PROTOTYPE_STAGE_L_FORK_CHILD_WEBSOCKET_RECONNECTED "
                        + json.dumps(
                            {
                                "branch": stage_l_fork_early_child_branch,
                                "repeat": int(stage_l_fork_early_child_repeat),
                                "port": _stage_l_child_port,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    _stage_l_materialize_task_renderer(TASK_ENV)
                    _stage_l_materialize_task_planner(TASK_ENV)
                    print(
                        "PROTOTYPE_STAGE_L_FORK_CHILD_RENDERER_MATERIALIZED",
                        flush=True,
                    )
                    break
                _stage_l_fork_child_paths.append(str(_stage_l_fork_child_path))
                _stage_l_fork_waited_pid, _stage_l_fork_wait_status = os.waitpid(
                    _stage_l_fork_child_pid, 0
                )
                if (
                    not os.WIFEXITED(_stage_l_fork_wait_status)
                    or os.WEXITSTATUS(_stage_l_fork_wait_status) != 0
                ):
                    raise RuntimeError(
                        "early fork child failed: "
                        f"branch={_stage_l_fork_child_branch_spec} "
                        f"repeat={_stage_l_fork_child_repeat_spec} "
                        f"status={_stage_l_fork_wait_status}"
                    )
            if os.environ.get("ROBOTWIN_STAGE_L_FORK_EARLY_CHILD", "0") != "1":
                _stage_l_fork_manifest = {
                    "schema": "robotwin_fork_clone_manifest_v1",
                    "clone_method": "os.fork_copy_on_write_before_renderer",
                    "task": args["task_name"],
                    "seed": int(now_seed),
                    "prompt": instruction,
                    "intervention_frame": int(stage_l_oracle_intervention_frame),
                    "repeats": int(stage_l_oracle_repeats),
                    "parent_setup_snapshot": str(_stage_l_fork_parent_setup_path.resolve()),
                    "child_records": _stage_l_fork_child_paths,
                    "parent_cuda_rng_inherited": bool(
                        torch.cuda.is_available() and torch.cuda.is_initialized()
                    ),
                }
                _stage_l_fork_parent_root.with_suffix(".manifest.json").write_text(
                    json.dumps(_stage_l_fork_manifest, indent=2) + "\\n",
                    encoding="utf-8",
                )
                TASK_ENV.close_env()
                raise SystemExit(0)
""" + early_setup_needle
        if source.count(early_setup_needle) != 1:
            raise RuntimeError("early fork setup instruction hook changed")
        source = source.replace(early_setup_needle, early_setup_replacement)
    if expected_pair_observation is not None:
        observation_needle = (
            "        initial_formatted_obs = format_obs(initial_obs, prompt)\n"
        )
        observation_replacement = observation_needle + """
        _stage_h_context_contract = compare_formatted_observations(
            expected_pair_observation,
            initial_formatted_obs,
        )
        print(
            "PROTOTYPE_STAGE_H_CONTEXT_CONTRACT "
            + json.dumps(_stage_h_context_contract, sort_keys=True),
            flush=True,
        )
        if not _stage_h_context_contract["nonvisual_exact"]:
            raise AssertionError(
                "formal paired nonvisual context differs from captured Student context: "
                + "; ".join(_stage_h_context_contract["mismatches"])
            )
        if snapshot_roundtrip_output:
            _stage_l_snapshot = capture_simulator_snapshot(TASK_ENV)
            _stage_l_perturbed = perturb_simulator_state_for_audit(TASK_ENV)
            restore_simulator_snapshot(TASK_ENV, _stage_l_snapshot)
            _stage_l_restored_obs = format_obs(TASK_ENV.get_obs(), prompt)
            _stage_l_roundtrip_contract = compare_formatted_observations(
                initial_formatted_obs,
                _stage_l_restored_obs,
            )
            _stage_l_output = Path(snapshot_roundtrip_output)
            _stage_l_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema": "robotwin_snapshot_roundtrip_audit_v1",
                    "simulation_only": True,
                    "snapshot": _stage_l_snapshot,
                    "perturbed_state_families": _stage_l_perturbed,
                    "observation_contract": _stage_l_roundtrip_contract,
                },
                _stage_l_output.with_suffix(".pt"),
            )
            _stage_l_output.write_text(
                json.dumps(
                    {
                        "schema": "robotwin_snapshot_roundtrip_audit_v1",
                        "simulation_only": True,
                        "perturbed_state_families": _stage_l_perturbed,
                        "observation_contract": _stage_l_roundtrip_contract,
                        "snapshot_artifact": str(
                            _stage_l_output.with_suffix(".pt").resolve()
                        ),
                    },
                    indent=2,
                )
                + "\\n",
                encoding="utf-8",
            )
            print(
                "STAGE_L_SNAPSHOT_ROUNDTRIP "
                + json.dumps(
                    {
                        "output": str(_stage_l_output.resolve()),
                        "exact": _stage_l_roundtrip_contract["exact"],
                        "perturbed": _stage_l_perturbed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if not _stage_l_roundtrip_contract["exact"]:
                raise AssertionError(
                    "snapshot round-trip observation differs: "
                    + "; ".join(_stage_l_roundtrip_contract["mismatches"])
                )
        if setup_audit_only:
            TASK_ENV.close_env()
            raise SystemExit(0)
"""
        if source.count(observation_needle) != 1:
            raise RuntimeError("Upstream Stage H context contract insertion changed")
        source = source.replace(observation_needle, observation_replacement)
else:
    if source.count(NEEDLE) != 1:
        raise RuntimeError("Upstream instruction-selection line changed")
    source = source.replace(NEEDLE, REPLACEMENT)
    if source.count(RESET_NEEDLE) != 1:
        raise RuntimeError("Upstream reset call changed")
    source = source.replace(RESET_NEEDLE, SEEDED_RESET_REPLACEMENT)
if stage_o_enabled:
    stage_o_init_needle = """        first_obs = None
        while TASK_ENV.take_action_cnt<TASK_ENV.step_lim:
"""
    stage_o_init_replacement = """        first_obs = None
        _stage_o_frame_st_id = 0
        _stage_o_planner_trace = _stage_n_planner_event_trace_type(
            TASK_ENV.robot
        )
        _stage_o_trace = _stage_o_trace_recorder_type(
            task_name=args["task_name"],
            seed=int(now_seed),
            config=stage_o_event_config,
        )
        _stage_o_previous_observation = _stage_l_deepcopy(
            initial_formatted_obs
        )
        while TASK_ENV.take_action_cnt<TASK_ENV.step_lim:
"""
    if source.count(stage_o_init_needle) != 1:
        raise RuntimeError("Upstream Stage-O trace initialization changed")
    source = source.replace(stage_o_init_needle, stage_o_init_replacement)

    stage_o_chunk_needle = """            model.infer(dict(obs = key_frame_list, compute_kv_cache=True, imagine=False, save_visualization=save_visualization, state=action))

            if TASK_ENV.eval_success:
"""
    stage_o_chunk_replacement = """            model.infer(dict(obs = key_frame_list, compute_kv_cache=True, imagine=False, save_visualization=save_visualization, state=action))
            _stage_o_frame_st_id += int(action.shape[1])
            _stage_o_progress = collect_task_progress(
                args["task_name"], TASK_ENV
            )
            _stage_o_current_observation = key_frame_list[-1]
            _stage_o_observable_metrics = _stage_o_observable_change(
                _stage_o_previous_observation,
                _stage_o_current_observation,
            )
            _stage_o_previous_observation = _stage_l_deepcopy(
                _stage_o_current_observation
            )
            _stage_o_event = _stage_o_trace.observe(
                frame=_stage_o_frame_st_id,
                progress=_stage_o_progress,
                planner_events=_stage_o_planner_trace.snapshot(),
                terminal_success=bool(TASK_ENV.eval_success),
                observable_change=_stage_o_observable_metrics,
            )
            if _stage_o_event is not None:
                print(
                    "STAGE_O_CAUSAL_EVENT "
                    + json.dumps(_stage_o_event, sort_keys=True),
                    flush=True,
                )

            if TASK_ENV.eval_success:
"""
    if source.count(stage_o_chunk_needle) != 1:
        raise RuntimeError("Upstream Stage-O chunk trace hook changed")
    source = source.replace(stage_o_chunk_needle, stage_o_chunk_replacement)
if stage_l_oracle_output:
    if context_capture_root:
        raise ValueError("Stage-L oracle and context capture hooks are mutually exclusive")
    oracle_init_needle = """        first_obs = None
        while TASK_ENV.take_action_cnt<TASK_ENV.step_lim:
"""
    oracle_init_replacement = """        first_obs = None
        _stage_l_first_observation = None
        _stage_l_prefix_records = []
        _stage_l_canonical_kv_hashes = []
        _stage_l_canonical_kv_components = []
        _stage_l_frame_st_id = 0
        _stage_l_target_frame = (
            1_000_000_000
            if stage_o_dynamic_treatment
            else stage_n_intervention_frames[0]
            if stage_n_enabled
            else stage_l_oracle_intervention_frame
        )
        _stage_n_pending_treatment = None
        _stage_n_treatment_record = None
        _stage_n_treatment_records = []
        _stage_n_applied_count = 0
        _stage_n_continuation_planner_trace = None
        _stage_n_initial_planner_prefix = None
        _stage_n_planner_prefix_trace = (
            _stage_n_planner_prefix_trace_type(TASK_ENV.robot)
            if stage_n_enabled
            else None
        )
        _stage_o_dynamic_detector = (
            _stage_o_observable_detector_type(stage_o_observable_config)
            if stage_o_dynamic_treatment
            else None
        )
        _stage_o_dynamic_previous_observation = (
            _stage_l_deepcopy(initial_formatted_obs)
            if stage_o_dynamic_treatment
            else None
        )
        _stage_o_dynamic_event_record = None
        _stage_o_dynamic_samples = []
        while TASK_ENV.take_action_cnt<TASK_ENV.step_lim:
"""
    if source.count(oracle_init_needle) != 1:
        raise RuntimeError("Upstream Stage-L oracle initialization changed")
    source = source.replace(oracle_init_needle, oracle_init_replacement)

    oracle_first_obs_needle = """                first_obs = format_obs(observation, prompt)
"""
    oracle_first_obs_replacement = oracle_first_obs_needle + """                if _stage_l_first_observation is None:
                    _stage_l_first_observation = _stage_l_deepcopy(first_obs)
"""
    if source.count(oracle_first_obs_needle) != 1:
        raise RuntimeError("Upstream Stage-L first observation hook changed")
    source = source.replace(
        oracle_first_obs_needle,
        oracle_first_obs_replacement,
    )

    oracle_action_needle = """            action = ret['action']
"""
    oracle_action_replacement = oracle_action_needle + """            if stage_n_enabled and _stage_n_pending_treatment is not None:
                _stage_n_selected = _stage_n_select_live_treatment(
                    actual_student_action=action,
                    label=_stage_n_pending_treatment["label"],
                    arm=stage_n_treatment_arm,
                    alpha=stage_n_alpha,
                )
                action = _stage_n_selected["action"]
                _stage_n_applied_count += 1
                if _stage_n_applied_count > stage_n_expected_treatments:
                    raise AssertionError("Stage-N exceeded its intervention schedule")
                _stage_n_treatment_record.update(
                    _stage_n_selected["record"]
                )
                _stage_n_treatment_record["sequence_index"] = (
                    _stage_n_applied_count - 1
                )
                _stage_n_pending_treatment = None
                _stage_n_partial_output = (
                    _stage_n_treatment_record
                    if stage_n_expected_treatments == 1
                    else {
                        "schema": "flashwam_stage_n_multi_treatment_v1",
                        "task": args["task_name"],
                        "task_config": args["task_config"],
                        "seed": int(now_seed),
                        "prompt": prompt,
                        "arm": stage_n_treatment_arm,
                        "alpha": (
                            0.0
                            if stage_n_treatment_arm == "base"
                            else float(stage_n_alpha)
                        ),
                        "intervention_frames": list(
                            stage_n_intervention_frames
                        ),
                        "application_count": _stage_n_applied_count,
                        "treatments": _stage_n_treatment_records,
                        "continuation_policy": "frozen_current_student",
                        "deployment_use": "training_or_evaluation_only",
                    }
                )
                Path(stage_n_treatment_output).write_text(
                    json.dumps(_stage_n_partial_output, indent=2) + "\\n",
                    encoding="utf-8",
                )
                print(
                    (
                        "STAGE_N_SINGLE_TREATMENT "
                        if stage_n_expected_treatments == 1
                        else "STAGE_N_MULTI_TREATMENT "
                    )
                    + json.dumps(
                        {
                            "output": str(
                                Path(stage_n_treatment_output).resolve()
                            ),
                            "arm": stage_n_treatment_arm,
                            "alpha": _stage_n_treatment_record["alpha"],
                            "source_exact": True,
                            "application_count": _stage_n_applied_count,
                            "application_expected": (
                                stage_n_expected_treatments
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
"""
    if source.count(oracle_action_needle) != 1:
        raise RuntimeError("Upstream Stage-N action treatment hook changed")
    source = source.replace(oracle_action_needle, oracle_action_replacement)

    oracle_kv_needle = """            model.infer(dict(obs = key_frame_list, compute_kv_cache=True, imagine=False, save_visualization=save_visualization, state=action))
"""
    oracle_kv_replacement = """            _stage_l_canonical_kv_result = model.infer(dict(obs = key_frame_list, compute_kv_cache=True, imagine=False, save_visualization=save_visualization, state=action))
            _stage_l_canonical_kv_hash = _stage_l_canonical_kv_result.get(
                "post_kv_runtime_sha256"
            )
            if _stage_l_canonical_kv_hash is None:
                raise AssertionError(
                    "Stage-L oracle requires server runtime-state audit"
                )
            _stage_l_canonical_kv_component = _stage_l_canonical_kv_result.get(
                "runtime_state_component_sha256"
            )
            if _stage_l_canonical_kv_component is None:
                raise AssertionError(
                    "Stage-L/M oracle requires component runtime-state audit"
                )
            _stage_l_prefix_records.append({
                "frame_st_id": int(_stage_l_frame_st_id),
                "observations": _stage_l_deepcopy(key_frame_list),
                "env_action": np.asarray(action).copy(),
            })
            _stage_l_canonical_kv_hashes.append(_stage_l_canonical_kv_hash)
            _stage_l_canonical_kv_components.append(
                _stage_l_canonical_kv_component
            )
            _stage_l_frame_st_id += int(action.shape[1])
            if (
                stage_o_dynamic_treatment
                and _stage_n_applied_count == 0
                and _stage_n_pending_treatment is None
            ):
                _stage_o_dynamic_current_observation = key_frame_list[-1]
                _stage_o_dynamic_observable = _stage_o_observable_change(
                    _stage_o_dynamic_previous_observation,
                    _stage_o_dynamic_current_observation,
                )
                _stage_o_dynamic_previous_observation = _stage_l_deepcopy(
                    _stage_o_dynamic_current_observation
                )
                _stage_o_dynamic_planner_events = (
                    _stage_n_planner_prefix_trace.event_snapshot()
                )
                _stage_o_dynamic_progress = collect_task_progress(
                    args["task_name"], TASK_ENV
                )
                _stage_o_dynamic_event = _stage_o_dynamic_detector.observe(
                    frame=_stage_l_frame_st_id,
                    image_mean_abs_change=_stage_o_dynamic_observable[
                        "image_mean_abs_change"
                    ],
                    state_rmse=_stage_o_dynamic_observable["state_rmse"],
                    planner_non_success=_stage_o_dynamic_planner_events[
                        "non_success_calls"
                    ],
                    terminal_success=bool(TASK_ENV.eval_success),
                )
                _stage_o_dynamic_samples.append({
                    "frame": int(_stage_l_frame_st_id),
                    "terminal_success": bool(TASK_ENV.eval_success),
                    "observable_change": _stage_o_dynamic_observable,
                    "planner_events": _stage_o_dynamic_planner_events,
                    "diagnostic_sim_progress": _stage_o_dynamic_progress,
                })
                _stage_o_dynamic_trace_payload = {
                    "schema": "flashwam_stage_o_dynamic_trace_v1",
                    "task": args["task_name"],
                    "task_config": args["task_config"],
                    "seed": int(now_seed),
                    "prompt": prompt,
                    "trigger_inputs_simulator_privilege_free": True,
                    "teacher_free_trigger": True,
                    "future_free_trigger": True,
                    "observable_config": {
                        "min_frame": stage_o_observable_config.min_frame,
                        "window_samples": (
                            stage_o_observable_config.window_samples
                        ),
                        "image_mean_threshold": (
                            stage_o_observable_config.image_mean_threshold
                        ),
                        "state_rmse_threshold": (
                            stage_o_observable_config.state_rmse_threshold
                        ),
                        "cooldown_frames": (
                            stage_o_observable_config.cooldown_frames
                        ),
                        "max_events": stage_o_observable_config.max_events,
                    },
                    "status": (
                        "event_triggered"
                        if _stage_o_dynamic_event is not None
                        else "pending"
                    ),
                    "event": _stage_o_dynamic_event,
                    "samples": _stage_o_dynamic_samples,
                }
                _stage_o_dynamic_trace_path = Path(
                    stage_o_dynamic_trace_output
                )
                _stage_o_dynamic_trace_path.parent.mkdir(
                    parents=True, exist_ok=True
                )
                _stage_o_dynamic_trace_path.write_text(
                    json.dumps(_stage_o_dynamic_trace_payload, indent=2)
                    + "\\n",
                    encoding="utf-8",
                )
                if _stage_o_dynamic_event is not None:
                    _stage_l_target_frame = _stage_l_frame_st_id
                    _stage_o_dynamic_event_record = {
                        **_stage_o_dynamic_event,
                        "task": args["task_name"],
                        "task_config": args["task_config"],
                        "seed": int(now_seed),
                        "prompt": prompt,
                        "observable_change": _stage_o_dynamic_observable,
                        "planner_events": _stage_o_dynamic_planner_events,
                        "diagnostic_sim_progress": _stage_o_dynamic_progress,
                        "trigger_inputs_simulator_privilege_free": True,
                        "teacher_free_trigger": True,
                        "future_free_trigger": True,
                    }
                    _stage_o_dynamic_event_path = Path(stage_o_event_output)
                    _stage_o_dynamic_event_path.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    _stage_o_dynamic_event_path.write_text(
                        json.dumps(
                            _stage_o_dynamic_event_record, indent=2
                        )
                        + "\\n",
                        encoding="utf-8",
                    )
                    print(
                        "STAGE_O_DYNAMIC_EVENT "
                        + json.dumps(
                            _stage_o_dynamic_event_record, sort_keys=True
                        ),
                        flush=True,
                    )
            if _stage_l_frame_st_id > _stage_l_target_frame:
                raise AssertionError("Stage-L oracle skipped intervention boundary")
            if _stage_l_frame_st_id == _stage_l_target_frame:
                _stage_l_snapshot = capture_simulator_snapshot(
                    TASK_ENV,
                    capture_cuda_rng=(
                        os.environ.get("ROBOTWIN_STAGE_L_FORK_EARLY_CHILD", "0")
                        != "1"
                    ),
                )
                _stage_l_snapshot_hash = simulator_state_sha256(_stage_l_snapshot)
                if (
                    stage_l_fork_early_enabled
                    and os.environ.get("ROBOTWIN_STAGE_L_FORK_EARLY_CHILD", "0")
                    == "1"
                ):
                    _stage_l_child_is_ab = bool(stage_l_branch_b_artifact)
                    _stage_l_child_intervention = None
                    if _stage_l_child_is_ab:
                        _stage_l_child_intervention = _stage_l_load_branch_intervention(
                            path=stage_l_branch_b_artifact,
                            action_key=stage_l_branch_b_key,
                            task_name=args["task_name"],
                            environment_seed=now_seed,
                            intervention_frame=_stage_l_frame_st_id,
                            prompt=prompt,
                        )
                    _stage_l_child_next = _stage_l_rebuild_flashwam_prefix(
                        model=model,
                        prompt=prompt,
                        environment_seed=now_seed,
                        first_observation=_stage_l_first_observation,
                        prefix_records=_stage_l_prefix_records,
                        inference_kwargs={
                            "video_guidance_scale": video_guidance_scale,
                            "action_guidance_scale": action_guidance_scale,
                        },
                        offline_context_replay=_stage_l_child_is_ab,
                    )
                    _stage_l_child_student_action = np.asarray(
                        _stage_l_child_next["action"]
                    ).copy()
                    _stage_l_child_action = (
                        _stage_l_child_student_action
                        if stage_l_fork_early_child_branch == "A"
                        else _stage_l_child_intervention["action"]
                    )
                    _stage_l_child_family = (
                        "student"
                        if stage_l_fork_early_child_branch == "A"
                        else "bridge"
                    )
                    _stage_l_child_start_observation = format_obs(
                        TASK_ENV.get_obs(), prompt
                    )
                    _stage_l_child_end_observations = (
                        _stage_l_execute_env_action_chunk(
                            task_env=TASK_ENV,
                            action=_stage_l_child_action,
                            initial_eef_pose=inint_eef_pose,
                            add_init_pose=add_init_pose,
                            format_obs=format_obs,
                            prompt=prompt,
                        )
                    )
                    _stage_l_child_end_snapshot = capture_simulator_snapshot(
                        TASK_ENV, capture_cuda_rng=False
                    )
                    _stage_l_child_start_hash = compare_formatted_observations(
                        _stage_l_child_start_observation,
                        _stage_l_child_start_observation,
                    )["sha256"]
                    _stage_l_child_end_hash = compare_formatted_observations(
                        _stage_l_child_end_observations[-1],
                        _stage_l_child_end_observations[-1],
                    )["sha256"]
                    _stage_l_child_payload = {
                        "schema": "robotwin_fork_clone_physical_record_v2",
                        "status": "PASS",
                        "branch": stage_l_fork_early_child_branch,
                        "repeat": int(stage_l_fork_early_child_repeat),
                        "task": args["task_name"],
                        "seed": int(now_seed),
                        "intervention_frame": int(_stage_l_frame_st_id),
                        "start_snapshot": _stage_l_snapshot,
                        "start_observation": _stage_l_child_start_observation,
                        "start_observation_sha256": _stage_l_child_start_hash,
                        "end_observations": _stage_l_child_end_observations,
                        "end_observation_sha256": _stage_l_child_end_hash,
                        "end_snapshot": _stage_l_child_end_snapshot,
                        "prefix_runtime_sha256": [
                            item["post_kv_runtime_sha256"]
                            for item in _stage_l_child_next["prefix_replay_audit"]
                        ],
                        "canonical_prefix_runtime_sha256": (
                            _stage_l_canonical_kv_hashes
                        ),
                        "pre_action_runtime_sha256": _stage_l_child_next.get(
                            "pre_action_runtime_sha256"
                        ),
                        "student_action": _stage_l_child_student_action,
                        "executed_action": _stage_l_child_action,
                        "student_action_sha256": _stage_l_array_sha256(
                            _stage_l_child_student_action
                        ),
                        "executed_action_sha256": _stage_l_array_sha256(
                            _stage_l_child_action
                        ),
                        "action_family": _stage_l_child_family,
                        "source_student_exact": True,
                        "cuda_rng_provenance": (
                            "inherited_at_fork; CUDA APIs not called in child"
                        ),
                    }
                    Path(stage_l_fork_early_child_output).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    torch.save(
                        _stage_l_child_payload,
                        stage_l_fork_early_child_output,
                    )
                    TASK_ENV.close_env()
                    raise SystemExit(0)
                if stage_l_fork_clone_enabled:
                    if stage_n_enabled or stage_m_enabled:
                        raise AssertionError(
                            "fork clone preflight cannot overlap Stage-M/Stage-N"
                        )
                    _stage_l_fork_output = Path(stage_l_fork_clone_output)
                    _stage_l_fork_root = _stage_l_fork_output.parent / (
                        _stage_l_fork_output.stem + ".physical_children"
                    )
                    _stage_l_fork_root.mkdir(parents=True, exist_ok=True)
                    _stage_l_is_ab = bool(stage_l_branch_b_artifact)
                    _stage_l_intervention = None
                    if _stage_l_is_ab:
                        _stage_l_intervention = _stage_l_load_branch_intervention(
                            path=stage_l_branch_b_artifact,
                            action_key=stage_l_branch_b_key,
                            task_name=args["task_name"],
                            environment_seed=now_seed,
                            intervention_frame=_stage_l_frame_st_id,
                            prompt=prompt,
                        )

                    # All model inference is completed in the parent.  Fork
                    # children only execute physical actions, avoiding CUDA
                    # calls after fork while preserving the hidden PhysX state.
                    _stage_l_fork_next = _stage_l_rebuild_flashwam_prefix(
                        model=model,
                        prompt=prompt,
                        environment_seed=now_seed,
                        first_observation=_stage_l_first_observation,
                        prefix_records=_stage_l_prefix_records,
                        inference_kwargs={
                            "video_guidance_scale": video_guidance_scale,
                            "action_guidance_scale": action_guidance_scale,
                        },
                        offline_context_replay=_stage_l_is_ab,
                    )
                    _stage_l_fork_student_action = np.asarray(
                        _stage_l_fork_next["action"]
                    ).copy()
                    _stage_l_fork_actions = []
                    for _stage_l_repeat in range(stage_l_oracle_repeats):
                        _stage_l_fork_actions.append(
                            {
                                "branch": "A",
                                "repeat": _stage_l_repeat,
                                "action": _stage_l_fork_student_action,
                            }
                        )
                        if _stage_l_is_ab:
                            _stage_l_fork_actions.append(
                                {
                                    "branch": "B",
                                    "repeat": _stage_l_repeat,
                                "action": _stage_l_intervention["action"],
                            }
                        )
                    _stage_l_fork_start_observation = format_obs(
                        TASK_ENV.get_obs(), prompt
                    )
                    _stage_l_fork_start_observation_sha256 = (
                        compare_formatted_observations(
                            _stage_l_fork_start_observation,
                            _stage_l_fork_start_observation,
                        )["sha256"]
                    )
                    _stage_l_planner_service = _stage_l_planner_service_type(
                        TASK_ENV.robot,
                        start_method=os.environ.get(
                            "ROBOTWIN_STAGE_L_PLANNER_START_METHOD",
                            "forkserver",
                        ),
                    )
                    try:
                        _stage_l_fork_records = _stage_l_run_forked_physical_branches(
                            task_env=TASK_ENV,
                            branch_actions=_stage_l_fork_actions,
                            initial_eef_pose=inint_eef_pose,
                            add_init_pose=add_init_pose,
                            execute_physics_action_chunk=(
                                _stage_l_execute_physics_action_chunk
                            ),
                            configure_physics_child=lambda **_stage_l_kwargs: (
                                _stage_l_configure_native_physics_only_child(
                                    task_env=_stage_l_kwargs["task_env"],
                                    planner_service=_stage_l_planner_service,
                                )
                            ),
                            parent_start_observation=_stage_l_fork_start_observation,
                            parent_start_observation_sha256=(
                                _stage_l_fork_start_observation_sha256
                            ),
                            prompt=prompt,
                            output_dir=_stage_l_fork_root,
                        )
                    finally:
                        _stage_l_planner_service.close()
                    _stage_l_action_noise_provenance = (
                        _stage_l_capture_noise_provenance(_stage_l_frame_st_id)
                    )
                    for _stage_l_record in _stage_l_fork_records:
                        _stage_l_record["action_noise_provenance"] = (
                            _stage_l_action_noise_provenance
                        )
                    # Physical-only children intentionally do not call the
                    # renderer.  The parent remained at the intervention
                    # boundary while they ran, so render each child end state
                    # here through the normal observation path, then restore
                    # the parent's visible/RNG state.  This is observation
                    # capture only; the parent must not step physics after
                    # restoring a child snapshot because the public snapshot
                    # schema does not contain PhysX solver warm-start state.
                    _stage_l_parent_render_snapshot = capture_simulator_snapshot(
                        TASK_ENV, capture_cuda_rng=True
                    )
                    for _stage_l_record in _stage_l_fork_records:
                        if _stage_l_record["end_observations"]:
                            continue
                        _stage_l_parent_end_observation = (
                            _stage_l_render_end_observation_in_parent(
                                task_env=TASK_ENV,
                                end_snapshot=_stage_l_record["end_snapshot"],
                                parent_snapshot=_stage_l_parent_render_snapshot,
                                format_obs=format_obs,
                                prompt=prompt,
                            )
                        )
                        _stage_l_record["end_observations"] = [
                            _stage_l_parent_end_observation
                        ]
                        _stage_l_record["end_observation_sha256"] = (
                            compare_formatted_observations(
                                _stage_l_parent_end_observation,
                                _stage_l_parent_end_observation,
                            )["sha256"]
                        )
                        _stage_l_record["post_observation_capture"] = (
                            "CAPTURED_PARENT_RENDER"
                        )
                    _stage_l_fork_prefix_runtime = [
                        item["post_kv_runtime_sha256"]
                        for item in _stage_l_fork_next["prefix_replay_audit"]
                    ]
                    _stage_l_fork_prefix_exact = bool(
                        _stage_l_fork_prefix_runtime
                        == _stage_l_canonical_kv_hashes
                    )
                    _stage_l_fork_pre_action_runtime = _stage_l_fork_next.get(
                        "pre_action_runtime_sha256"
                    )
                    _stage_l_fork_summaries = []
                    for _stage_l_record in _stage_l_fork_records:
                        _stage_l_child_start = compare_formatted_observations(
                            _stage_l_record["start_observation"],
                            _stage_l_record["start_observation"],
                        )
                        _stage_l_child_end = None
                        if _stage_l_record["end_observations"]:
                            _stage_l_child_end = compare_formatted_observations(
                                _stage_l_record["end_observations"][-1],
                                _stage_l_record["end_observations"][-1],
                            )
                        _stage_l_child_action = (
                            _stage_l_fork_student_action
                            if _stage_l_record["branch"] == "A"
                            else _stage_l_intervention["action"]
                        )
                        _stage_l_child_family = (
                            "student"
                            if _stage_l_record["branch"] == "A"
                            else "bridge"
                        )
                        _stage_l_fork_summaries.append(
                            {
                                "repeat": int(_stage_l_record["repeat"]),
                                "branch": _stage_l_record["branch"],
                                "start_simulator_sha256": simulator_state_sha256(
                                    _stage_l_record["start_snapshot"]
                                ),
                                "start_observation_sha256": _stage_l_child_start[
                                    "sha256"
                                ],
                                "prefix_runtime_sha256": (
                                    _stage_l_fork_prefix_runtime
                                ),
                                "pre_action_runtime_sha256": (
                                    _stage_l_fork_pre_action_runtime
                                ),
                                "student_action_sha256": _stage_l_array_sha256(
                                    _stage_l_fork_student_action
                                ),
                                "action_noise_sha256": (
                                    _stage_l_action_noise_provenance.get(
                                        "action_noise_sha256"
                                    )
                                ),
                                "video_noise_sha256": (
                                    _stage_l_action_noise_provenance.get(
                                        "video_noise_sha256"
                                    )
                                ),
                                "action_family": _stage_l_child_family,
                                "executed_action_sha256": _stage_l_array_sha256(
                                    _stage_l_child_action
                                ),
                                "source_student_exact": True,
                                "planner_mode": "fork_clone_physical_only",
                                "end_simulator_sha256": simulator_state_sha256(
                                    _stage_l_record["end_snapshot"]
                                ),
                                "end_observation_sha256": (
                                    _stage_l_child_end["sha256"]
                                    if _stage_l_child_end is not None
                                    else None
                                ),
                                "post_observation_capture": (
                                    _stage_l_record.get(
                                        "post_observation_capture", "CAPTURED_CHILD_RENDER"
                                    )
                                    if _stage_l_child_end is not None
                                    else "NOT_CAPTURED_PHYSICAL_ONLY"
                                ),
                                "progress": None,
                            }
                        )

                    _stage_l_fork_reference = _stage_l_fork_summaries[0]
                    _stage_l_fork_starts = [
                        compare_simulator_states(
                            _stage_l_fork_records[0]["start_snapshot"],
                            _stage_l_item["start_snapshot"],
                        )
                        for _stage_l_item in _stage_l_fork_records
                    ]
                    _stage_l_fork_end_family_refs = {}
                    for _stage_l_item in _stage_l_fork_records:
                        _stage_l_fork_end_family_refs.setdefault(
                            "student"
                            if _stage_l_item["branch"] == "A"
                            else "bridge",
                            _stage_l_item,
                        )
                    _stage_l_fork_end_comparisons = []
                    for _stage_l_item in _stage_l_fork_records:
                        _stage_l_family = (
                            "student"
                            if _stage_l_item["branch"] == "A"
                            else "bridge"
                        )
                        _stage_l_fork_end_comparisons.append(
                            compare_simulator_states(
                                _stage_l_fork_end_family_refs[_stage_l_family][
                                    "end_snapshot"
                                ],
                                _stage_l_item["end_snapshot"],
                            )
                        )
                    _stage_l_fork_common_exact = all(
                        item["prefix_runtime_sha256"]
                        == _stage_l_fork_reference["prefix_runtime_sha256"]
                        and item["pre_action_runtime_sha256"]
                        == _stage_l_fork_reference["pre_action_runtime_sha256"]
                        and item["student_action_sha256"]
                        == _stage_l_fork_reference["student_action_sha256"]
                        for item in _stage_l_fork_summaries
                    )
                    _stage_l_fork_family_observable_exact = all(
                        _stage_l_summary["end_observation_sha256"] is not None
                        for _stage_l_summary in _stage_l_fork_summaries
                    ) and all(
                        _stage_l_fork_summaries[index][
                            "executed_action_sha256"
                        ]
                        == _stage_l_fork_summaries[
                            next_index
                        ]["executed_action_sha256"]
                        and _stage_l_fork_summaries[index][
                            "end_observation_sha256"
                        ]
                        == _stage_l_fork_summaries[next_index][
                            "end_observation_sha256"
                        ]
                        for index in range(len(_stage_l_fork_summaries))
                        for next_index in range(index + 1, len(_stage_l_fork_summaries))
                        if _stage_l_fork_summaries[index]["action_family"]
                        == _stage_l_fork_summaries[next_index]["action_family"]
                    )
                    _stage_l_fork_start_full = all(
                        item["exact"] for item in _stage_l_fork_starts
                    )
                    _stage_l_fork_end_full = all(
                        item["exact"] for item in _stage_l_fork_end_comparisons
                    )
                    _stage_l_fork_result = {
                        "schema": (
                            "flashwam_same_state_ab_oracle_v1"
                            if _stage_l_is_ab
                            else "flashwam_same_state_aa_oracle_v3"
                        ),
                        "clone_method": "os.fork_copy_on_write_at_intervention",
                        "task": args["task_name"],
                        "seed": int(now_seed),
                        "prompt": prompt,
                        "intervention_frame": _stage_l_frame_st_id,
                        "repeats": stage_l_oracle_repeats,
                        "symmetric_restore": False,
                        "canonical_snapshot_sha256": _stage_l_snapshot_hash,
                        "canonical_prefix_runtime_sha256": _stage_l_canonical_kv_hashes,
                        "prefix_reconstruction_exact": _stage_l_fork_prefix_exact,
                        "common_causal_inputs_exact": _stage_l_fork_common_exact,
                        "within_family_observable_exact": (
                            _stage_l_fork_family_observable_exact
                        ),
                        "start_full_state_exact": _stage_l_fork_start_full,
                        "start_core_state_exact": _stage_l_fork_start_full,
                        "within_family_end_full_state_exact": _stage_l_fork_end_full,
                        "within_family_end_core_state_exact": _stage_l_fork_end_full,
                        "strict_causal_contract_go": bool(
                            _stage_l_fork_prefix_exact
                            and _stage_l_fork_common_exact
                            and _stage_l_fork_family_observable_exact
                            and _stage_l_fork_start_full
                            and _stage_l_fork_end_full
                        ),
                        "near_strict_causal_contract_go": bool(
                            _stage_l_fork_prefix_exact
                            and _stage_l_fork_common_exact
                            and _stage_l_fork_family_observable_exact
                            and _stage_l_fork_start_full
                            and _stage_l_fork_end_full
                        ),
                        "fork_parent_cuda_rng_inherited": bool(
                            _stage_l_snapshot["rng"].get("torch_cuda") is not None
                        ),
                        "fork_child_cuda_rng_api": "disabled; inherited at fork",
                        "action_noise_provenance": _stage_l_action_noise_provenance,
                        "branches": _stage_l_fork_summaries,
                    }
                    _stage_l_fork_output = Path(stage_l_fork_clone_output)
                    _stage_l_fork_output.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {**_stage_l_fork_result, "records": _stage_l_fork_records},
                        _stage_l_fork_output.with_suffix(".pt"),
                    )
                    _stage_l_fork_output.write_text(
                        json.dumps(_stage_l_fork_result, indent=2) + "\\n",
                        encoding="utf-8",
                    )
                    print(
                        "STAGE_L_FORK_CLONE "
                        + json.dumps(
                            {
                                "output": str(_stage_l_fork_output.resolve()),
                                "clone_method": "os.fork_copy_on_write_at_intervention",
                                "strict_contract": _stage_l_fork_result[
                                    "strict_causal_contract_go"
                                ],
                                "branches": len(_stage_l_fork_summaries),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    TASK_ENV.close_env()
                    raise SystemExit(0)
                if stage_n_enabled:
                    if _stage_n_applied_count == 0:
                        _stage_n_initial_planner_prefix = (
                            _stage_n_planner_prefix_trace.finish()
                        )
                        _stage_n_continuation_planner_trace = (
                            _stage_n_planner_event_trace_type(TASK_ENV.robot)
                        )
                        _stage_n_planner_prefix_fingerprint = (
                            _stage_n_initial_planner_prefix
                        )
                    else:
                        _stage_n_planner_prefix_fingerprint = {
                            "schema": "flashwam_stage_n_multi_planner_prefix_v1",
                            "initial_prefix": _stage_n_initial_planner_prefix,
                            "continuation_before_intervention": (
                                _stage_n_continuation_planner_trace.snapshot()
                            ),
                        }
                    _stage_l_planner_trace = None
                else:
                    _stage_l_planner_trace = BranchPlannerTrace(TASK_ENV.robot)
                _stage_l_records = []
                _stage_l_summaries = []
                _stage_l_is_ab = bool(stage_l_branch_b_artifact)
                _stage_l_intervention = None
                _stage_m_pause_contract = None
                if stage_m_enabled:
                    _stage_n_query_index = len(_stage_n_treatment_records)
                    _stage_m_context_path = Path(stage_m_context_output)
                    _stage_m_label_path = Path(stage_m_label_output)
                    _stage_m_runtime_path = Path(stage_m_runtime_audit_output)
                    if stage_n_expected_treatments > 1:
                        _stage_n_suffix = (
                            f"_i{_stage_n_query_index:02d}_"
                            f"frame{int(_stage_l_frame_st_id)}"
                        )
                        _stage_m_context_path = _stage_m_context_path.with_name(
                            _stage_m_context_path.stem
                            + _stage_n_suffix
                            + _stage_m_context_path.suffix
                        )
                        _stage_m_label_path = _stage_m_label_path.with_name(
                            _stage_m_label_path.stem
                            + _stage_n_suffix
                            + _stage_m_label_path.suffix
                        )
                        _stage_m_runtime_path = _stage_m_runtime_path.with_name(
                            _stage_m_runtime_path.stem
                            + _stage_n_suffix
                            + _stage_m_runtime_path.suffix
                        )
                    for _stage_m_path in (
                        _stage_m_context_path,
                        _stage_m_label_path,
                        _stage_m_runtime_path,
                    ):
                        _stage_m_path.parent.mkdir(parents=True, exist_ok=True)
                    _stage_m_live_context_id = (
                        f"{args['task_name']}-seed{int(now_seed)}-"
                        f"frame{int(_stage_l_frame_st_id)}"
                    )
                    _stage_m_pause_started_ns = time.time_ns()
                    _stage_m_pause_step_count = int(TASK_ENV.take_action_cnt)
                    _stage_m_pause_observation = format_obs(
                        TASK_ENV.get_obs(), prompt
                    )
                    _stage_m_pause_observation_hash = (
                        compare_formatted_observations(
                            _stage_m_pause_observation,
                            _stage_m_pause_observation,
                        )["sha256"]
                    )
                    _stage_m_policy_delta = (
                        os.environ.get("ROBOTWIN_POLICY_DELTA_PATH") or None
                    )
                    _stage_m_policy = _stage_m_policy_version(
                        stage_m_student_checkpoint,
                        _stage_m_policy_delta,
                    )
                    _stage_m_context = _stage_m_build_live_context(
                        live_context_id=_stage_m_live_context_id,
                        task=args["task_name"],
                        task_config=args["task_config"],
                        seed=now_seed,
                        prompt=prompt,
                        model_name=args["ckpt_setting"],
                        policy_version=_stage_m_policy,
                        policy_delta_path=_stage_m_policy_delta,
                        initial_observation=_stage_l_first_observation,
                        chunks=_stage_l_deepcopy(_stage_l_prefix_records),
                        prefix_runtime_sha256=_stage_l_canonical_kv_hashes,
                        prefix_runtime_components=(
                            _stage_l_canonical_kv_components
                        ),
                        simulator_snapshot_sha256=_stage_l_snapshot_hash,
                        start_observation_sha256=(
                            _stage_m_pause_observation_hash
                        ),
                        simulator_step_count=_stage_m_pause_step_count,
                        capture_started_unix_ns=_stage_m_pause_started_ns,
                    )
                    torch.save(_stage_m_context, _stage_m_context_path)
                    _stage_m_command, _stage_m_env_updates = (
                        _stage_m_build_teacher_command(
                            python=Path(stage_m_teacher_python),
                            script=(
                                WORKSPACE_ROOT
                                / "experiments"
                                / "prototype_real_obs_action_teacher_bridge.py"
                            ),
                            project_root=PROJECT_ROOT,
                            student=Path(stage_m_student_checkpoint),
                            teacher_transformer=Path(
                                stage_m_teacher_transformer
                            ),
                            context=_stage_m_context_path,
                            label=_stage_m_label_path,
                            runtime_audit=_stage_m_runtime_path,
                            live_context_id=_stage_m_live_context_id,
                            diffusion_seed=stage_m_diffusion_seed,
                            context_chunks=len(_stage_l_prefix_records),
                            teacher_gpu=stage_m_teacher_gpu,
                        )
                    )
                    _stage_m_teacher_lock_requested_ns = time.time_ns()
                    with _stage_m_teacher_query_lock(stage_m_teacher_lock_path):
                        _stage_m_teacher_lock_acquired_ns = time.time_ns()
                        _stage_m_completed = subprocess.run(
                            _stage_m_command,
                            env={**os.environ, **_stage_m_env_updates},
                            check=False,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                        )
                        _stage_m_teacher_query_ended_ns = time.time_ns()
                    _stage_m_context_path.with_suffix(".teacher.stdout.log").write_text(
                        _stage_m_completed.stdout,
                        encoding="utf-8",
                    )
                    _stage_m_context_path.with_suffix(".teacher.stderr.log").write_text(
                        _stage_m_completed.stderr,
                        encoding="utf-8",
                    )
                    if _stage_m_completed.returncode != 0:
                        raise RuntimeError(
                            "Stage-M synchronous Teacher failed with code "
                            f"{_stage_m_completed.returncode}: "
                            f"{_stage_m_completed.stderr[-4000:]}"
                        )
                    _stage_m_pause_ended_ns = time.time_ns()
                    _stage_m_after_step_count = int(TASK_ENV.take_action_cnt)
                    _stage_m_after_snapshot = capture_simulator_snapshot(
                        TASK_ENV
                    )
                    _stage_m_sim_comparison = compare_simulator_states(
                        _stage_l_snapshot,
                        _stage_m_after_snapshot,
                    )
                    _stage_m_after_observation = format_obs(
                        TASK_ENV.get_obs(), prompt
                    )
                    _stage_m_observation_comparison = (
                        compare_formatted_observations(
                            _stage_m_pause_observation,
                            _stage_m_after_observation,
                        )
                    )
                    if _stage_m_after_step_count != _stage_m_pause_step_count:
                        raise AssertionError(
                            "Stage-M simulator advanced during Teacher query"
                        )
                    if not _stage_m_observation_comparison["exact"]:
                        raise AssertionError(
                            "Stage-M observation changed during Teacher query"
                        )
                    _stage_m_runtime_audit = json.loads(
                        _stage_m_runtime_path.read_text(encoding="utf-8")
                    )
                    _stage_m_runtime_exact = bool(
                        _stage_m_runtime_audit["runtime_state_sha256"]
                        == _stage_l_canonical_kv_hashes[-1]
                    )
                    _stage_m_components_exact = bool(
                        _stage_m_runtime_audit["component_sha256"]
                        == _stage_l_canonical_kv_components[-1]
                    )
                    if not (
                        _stage_m_runtime_exact and _stage_m_components_exact
                    ):
                        raise AssertionError(
                            "Stage-M GPU7 label context differs from live GPU6 "
                            "Flash recurrent state"
                        )
                    _stage_m_label = torch.load(
                        _stage_m_label_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                    _stage_m_validate_live_label(
                        context=_stage_m_context,
                        context_path=_stage_m_context_path,
                        label=_stage_m_label,
                    )
                    _stage_m_pause_contract = {
                        "schema": "flashwam_stage_m_pause_contract_v1",
                        "live_context_id": _stage_m_live_context_id,
                        "context": str(_stage_m_context_path.resolve()),
                        "label": str(_stage_m_label_path.resolve()),
                        "runtime_audit": str(_stage_m_runtime_path.resolve()),
                        "pause_started_unix_ns": _stage_m_pause_started_ns,
                        "pause_ended_unix_ns": _stage_m_pause_ended_ns,
                        "pause_elapsed_seconds": (
                            (_stage_m_pause_ended_ns - _stage_m_pause_started_ns)
                            / 1e9
                        ),
                        "teacher_lock_wait_seconds": (
                            (
                                _stage_m_teacher_lock_acquired_ns
                                - _stage_m_teacher_lock_requested_ns
                            )
                            / 1e9
                        ),
                        "teacher_query_elapsed_seconds": (
                            (
                                _stage_m_teacher_query_ended_ns
                                - _stage_m_teacher_lock_acquired_ns
                            )
                            / 1e9
                        ),
                        "simulator_step_before": _stage_m_pause_step_count,
                        "simulator_step_after": _stage_m_after_step_count,
                        "simulator_full_state_exact": (
                            _stage_m_sim_comparison["exact"]
                        ),
                        "simulator_core_state_exact": all(
                            key.endswith("/qacc")
                            for key in _stage_m_sim_comparison["differences"]
                        ),
                        "observation_exact": (
                            _stage_m_observation_comparison["exact"]
                        ),
                        "gpu6_gpu7_runtime_exact": _stage_m_runtime_exact,
                        "gpu6_gpu7_components_exact": (
                            _stage_m_components_exact
                        ),
                    }
                    _stage_m_context_path.with_suffix(".pause.json").write_text(
                        json.dumps(_stage_m_pause_contract, indent=2) + "\\n",
                        encoding="utf-8",
                    )
                if stage_n_enabled:
                    _stage_n_validate_pause(_stage_m_pause_contract)
                    _stage_n_provenance = _stage_n_build_provenance(
                        prefix_runtime_sha256=(
                            _stage_l_canonical_kv_hashes
                        ),
                        snapshot=_stage_l_snapshot,
                    )
                    _stage_n_pending_treatment = {"label": _stage_m_label}
                    _stage_n_treatment_record = {
                        "schema": "flashwam_stage_n_single_treatment_v1",
                        "task": args["task_name"],
                        "task_config": args["task_config"],
                        "seed": int(now_seed),
                        "prompt": prompt,
                        "intervention_frame": int(_stage_l_frame_st_id),
                        "context": str(_stage_m_context_path.resolve()),
                        "label": str(_stage_m_label_path.resolve()),
                        "label_sha256": hashlib.sha256(
                            _stage_m_label_path.read_bytes()
                        ).hexdigest(),
                        "context_semantic_sha256": _stage_m_context[
                            "semantic_sha256"
                        ],
                        "fresh_history_context_sha256": (
                            _stage_n_fresh_context_sha256(_stage_m_context)
                        ),
                        "start_observation_sha256": _stage_m_context[
                            "start_observation_sha256"
                        ],
                        "simulator_snapshot_sha256": _stage_l_snapshot_hash,
                        "client_process_id": int(os.getpid()),
                        "prefix_runtime_sha256": _stage_n_provenance[
                            "prefix_runtime_sha256"
                        ],
                        "planner_prefix": (
                            _stage_n_planner_prefix_fingerprint
                        ),
                        "intervention_exposed_state": _stage_n_provenance[
                            "exposed_qpos_qvel"
                        ],
                        "policy_version": _stage_m_context["policy_version"],
                        "student_checkpoint": str(
                            _stage_m_label["student_checkpoint"]
                        ),
                        "teacher_transformer": str(
                            _stage_m_label["teacher_transformer"]
                        ),
                        "diffusion_seed": int(
                            _stage_m_label["diffusion_seed"]
                        ),
                        "pause_contract": _stage_m_pause_contract,
                        "application_count": 0,
                        "continuation_policy": "frozen_current_student",
                        "deployment_use": "training_or_evaluation_only",
                    }
                    _stage_n_treatment_record["sequence_index"] = (
                        _stage_n_query_index
                    )
                    if stage_o_dynamic_treatment:
                        if _stage_o_dynamic_event_record is None:
                            raise AssertionError(
                                "Stage-O dynamic treatment lacks its causal event"
                            )
                        _stage_n_treatment_record["stage_o_dynamic_event"] = (
                            _stage_o_dynamic_event_record
                        )
                    _stage_n_treatment_records.append(
                        _stage_n_treatment_record
                    )
                    Path(stage_n_treatment_output).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    _stage_n_pending_output = (
                        _stage_n_treatment_record
                        if stage_n_expected_treatments == 1
                        else {
                            "schema": "flashwam_stage_n_multi_treatment_v1",
                            "task": args["task_name"],
                            "task_config": args["task_config"],
                            "seed": int(now_seed),
                            "prompt": prompt,
                            "arm": stage_n_treatment_arm,
                            "alpha": (
                                0.0
                                if stage_n_treatment_arm == "base"
                                else float(stage_n_alpha)
                            ),
                            "intervention_frames": list(
                                stage_n_intervention_frames
                            ),
                            "application_count": _stage_n_applied_count,
                            "treatments": _stage_n_treatment_records,
                            "pending_sequence_index": _stage_n_query_index,
                            "continuation_policy": "frozen_current_student",
                            "deployment_use": "training_or_evaluation_only",
                        }
                    )
                    Path(stage_n_treatment_output).write_text(
                        json.dumps(_stage_n_pending_output, indent=2) + "\\n",
                        encoding="utf-8",
                    )
                    _stage_n_next_index = len(_stage_n_treatment_records)
                    _stage_l_target_frame = (
                        stage_n_intervention_frames[_stage_n_next_index]
                        if _stage_n_next_index < stage_n_expected_treatments
                        else 1_000_000_000
                    )
                    continue
                if _stage_l_is_ab:
                    _stage_l_intervention = _stage_l_load_branch_intervention(
                        path=stage_l_branch_b_artifact,
                        action_key=stage_l_branch_b_key,
                        task_name=args["task_name"],
                        environment_seed=now_seed,
                        intervention_frame=_stage_l_frame_st_id,
                        prompt=prompt,
                    )
                for _stage_l_repeat in range(stage_l_oracle_repeats):
                    for _stage_l_branch in ("A", "B"):
                        restore_simulator_snapshot(TASK_ENV, _stage_l_snapshot)
                        _stage_l_start_observation = format_obs(
                            TASK_ENV.get_obs(), prompt
                        )
                        _stage_l_start_snapshot = capture_simulator_snapshot(
                            TASK_ENV
                        )
                        _stage_l_next = _stage_l_rebuild_flashwam_prefix(
                            model=model,
                            prompt=prompt,
                            environment_seed=now_seed,
                            first_observation=_stage_l_first_observation,
                            prefix_records=_stage_l_prefix_records,
                            inference_kwargs={
                                "video_guidance_scale": video_guidance_scale,
                                "action_guidance_scale": action_guidance_scale,
                            },
                            offline_context_replay=_stage_l_is_ab,
                        )
                        _stage_l_student_action = np.asarray(
                            _stage_l_next["action"]
                        ).copy()
                        _stage_l_action_family = (
                            "student"
                            if _stage_l_branch == "A" or not _stage_l_is_ab
                            else "bridge"
                        )
                        _stage_l_executed_action = (
                            _stage_l_student_action
                            if _stage_l_action_family == "student"
                            else _stage_l_intervention["action"]
                        )
                        _stage_l_source_student_exact = bool(
                            not _stage_l_is_ab
                            or np.array_equal(
                                _stage_l_student_action,
                                _stage_l_intervention["student_action"],
                            )
                        )
                        if not _stage_l_source_student_exact:
                            raise AssertionError(
                                "Stage-L generated Student action differs from "
                                "the Bridge artifact's causal source action"
                            )
                        if (
                            _stage_l_repeat == 0
                            and (
                                _stage_l_is_ab
                                or _stage_l_branch == "A"
                            )
                        ):
                            _stage_l_planner_trace.begin_record(
                                _stage_l_action_family
                            )
                            _stage_l_planner_mode = "record"
                        else:
                            _stage_l_planner_trace.begin_replay(
                                _stage_l_action_family
                            )
                            _stage_l_planner_mode = "replay"
                        _stage_l_end_observations = (
                            _stage_l_execute_env_action_chunk(
                                task_env=TASK_ENV,
                                action=_stage_l_executed_action,
                                initial_eef_pose=inint_eef_pose,
                                add_init_pose=add_init_pose,
                                format_obs=format_obs,
                                prompt=prompt,
                            )
                        )
                        _stage_l_planner_trace.finish_branch()
                        _stage_l_end_snapshot = capture_simulator_snapshot(
                            TASK_ENV
                        )
                        _stage_l_start_contract = compare_formatted_observations(
                            _stage_l_start_observation,
                            _stage_l_start_observation,
                        )
                        _stage_l_end_contract = compare_formatted_observations(
                            _stage_l_end_observations[-1],
                            _stage_l_end_observations[-1],
                        )
                        _stage_l_replay_hashes = [
                            item["post_kv_runtime_sha256"]
                            for item in _stage_l_next["prefix_replay_audit"]
                        ]
                        _stage_l_summary = {
                            "repeat": _stage_l_repeat,
                            "branch": _stage_l_branch,
                            "start_simulator_sha256": simulator_state_sha256(
                                _stage_l_start_snapshot
                            ),
                            "start_observation_sha256": _stage_l_start_contract[
                                "sha256"
                            ],
                            "prefix_runtime_sha256": _stage_l_replay_hashes,
                            "pre_action_runtime_sha256": _stage_l_next.get(
                                "pre_action_runtime_sha256"
                            ),
                            "student_action_sha256": _stage_l_array_sha256(
                                _stage_l_student_action
                            ),
                            "action_family": _stage_l_action_family,
                            "executed_action_sha256": _stage_l_array_sha256(
                                _stage_l_executed_action
                            ),
                            "source_student_action_exact": (
                                _stage_l_source_student_exact
                            ),
                            "planner_mode": _stage_l_planner_mode,
                            "planner_call_counts": (
                                _stage_l_planner_trace.call_counts
                            ),
                            "end_simulator_sha256": simulator_state_sha256(
                                _stage_l_end_snapshot
                            ),
                            "end_observation_sha256": _stage_l_end_contract[
                                "sha256"
                            ],
                            "progress": collect_task_progress(
                                args["task_name"], TASK_ENV
                            ),
                        }
                        _stage_l_summaries.append(_stage_l_summary)
                        _stage_l_records.append({
                            "summary": _stage_l_summary,
                            "student_action": _stage_l_student_action,
                            "executed_action": _stage_l_executed_action,
                            "start_observation": _stage_l_start_observation,
                            "end_observations": _stage_l_end_observations,
                            "start_snapshot": _stage_l_start_snapshot,
                            "end_snapshot": _stage_l_end_snapshot,
                        })

                _stage_l_reference = _stage_l_summaries[0]
                _stage_l_start_state_comparisons = [
                    compare_simulator_states(
                        _stage_l_records[0]["start_snapshot"],
                        item["start_snapshot"],
                    )
                    for item in _stage_l_records
                ]
                _stage_l_family_references = {}
                for _stage_l_record in _stage_l_records:
                    _stage_l_family_references.setdefault(
                        _stage_l_record["summary"]["action_family"],
                        _stage_l_record,
                    )
                for _stage_l_index, _stage_l_summary in enumerate(
                    _stage_l_summaries
                ):
                    _stage_l_start_comparison = (
                        _stage_l_start_state_comparisons[_stage_l_index]
                    )
                    _stage_l_family_reference = _stage_l_family_references[
                        _stage_l_summary["action_family"]
                    ]
                    _stage_l_end_comparison = compare_simulator_states(
                        _stage_l_family_reference["end_snapshot"],
                        _stage_l_records[_stage_l_index]["end_snapshot"],
                    )
                    _stage_l_summary["start_full_state_exact"] = (
                        _stage_l_start_comparison["exact"]
                    )
                    _stage_l_summary["start_core_state_exact"] = all(
                        key.endswith("/qacc")
                        for key in _stage_l_start_comparison["differences"]
                    )
                    _stage_l_summary["start_state_max_abs"] = (
                        _stage_l_start_comparison["max_abs"]
                    )
                    _stage_l_summary["end_family_full_state_exact"] = (
                        _stage_l_end_comparison["exact"]
                    )
                    _stage_l_summary["end_family_core_state_exact"] = all(
                        key.endswith("/qacc")
                        for key in _stage_l_end_comparison["differences"]
                    )
                    _stage_l_summary["end_family_state_max_abs"] = (
                        _stage_l_end_comparison["max_abs"]
                    )
                _stage_l_common_fields = (
                    "start_observation_sha256",
                    "prefix_runtime_sha256",
                    "pre_action_runtime_sha256",
                    "student_action_sha256",
                    "source_student_action_exact",
                )
                _stage_l_common_exact = all(
                    all(
                        item[field] == _stage_l_reference[field]
                        for field in _stage_l_common_fields
                    )
                    for item in _stage_l_summaries
                )
                _stage_l_family_observable_exact = all(
                    all(
                        item[field]
                        == _stage_l_family_references[
                            item["action_family"]
                        ]["summary"][field]
                        for field in (
                            "executed_action_sha256",
                            "end_observation_sha256",
                            "progress",
                        )
                    )
                    for item in _stage_l_summaries
                )
                _stage_l_start_full_state_exact = all(
                    item["start_full_state_exact"]
                    for item in _stage_l_summaries
                )
                _stage_l_start_core_state_exact = all(
                    item["start_core_state_exact"]
                    for item in _stage_l_summaries
                )
                _stage_l_end_family_full_state_exact = all(
                    item["end_family_full_state_exact"]
                    for item in _stage_l_summaries
                )
                _stage_l_end_family_core_state_exact = all(
                    item["end_family_core_state_exact"]
                    for item in _stage_l_summaries
                )
                _stage_l_prefix_exact = all(
                    item["prefix_runtime_sha256"]
                    == _stage_l_canonical_kv_hashes
                    for item in _stage_l_summaries
                )
                _stage_l_strict_contract = bool(
                    _stage_l_prefix_exact
                    and _stage_l_common_exact
                    and _stage_l_family_observable_exact
                    and _stage_l_start_full_state_exact
                    and _stage_l_end_family_full_state_exact
                )
                _stage_l_near_strict_contract = bool(
                    _stage_l_prefix_exact
                    and _stage_l_common_exact
                    and _stage_l_family_observable_exact
                    and _stage_l_start_core_state_exact
                    and _stage_l_end_family_core_state_exact
                )
                _stage_l_transition = None
                if _stage_l_is_ab:
                    _stage_l_student_record = _stage_l_family_references[
                        "student"
                    ]
                    _stage_l_bridge_record = _stage_l_family_references[
                        "bridge"
                    ]
                    _stage_l_action_delta = np.abs(
                        _stage_l_bridge_record["executed_action"]
                        - _stage_l_student_record["executed_action"]
                    )
                    _stage_l_transition = {
                        "student_progress": _stage_l_student_record[
                            "summary"
                        ]["progress"],
                        "bridge_progress": _stage_l_bridge_record[
                            "summary"
                        ]["progress"],
                        "ordinal_delta": int(
                            _stage_l_bridge_record["summary"]["progress"][
                                "ordinal_stage"
                            ]
                            - _stage_l_student_record["summary"]["progress"][
                                "ordinal_stage"
                            ]
                        ),
                        "action_rmse": float(
                            np.sqrt(np.mean(_stage_l_action_delta ** 2))
                        ),
                        "action_max_abs": float(_stage_l_action_delta.max()),
                        "end_observation_equal": bool(
                            _stage_l_student_record["summary"][
                                "end_observation_sha256"
                            ]
                            == _stage_l_bridge_record["summary"][
                                "end_observation_sha256"
                            ]
                        ),
                        "end_state_comparison": compare_simulator_states(
                            _stage_l_student_record["end_snapshot"],
                            _stage_l_bridge_record["end_snapshot"],
                        ),
                    }
                _stage_l_output = Path(stage_l_oracle_output)
                _stage_l_output.parent.mkdir(parents=True, exist_ok=True)
                _stage_l_result = {
                    "schema": (
                        "flashwam_same_state_ab_oracle_v1"
                        if _stage_l_is_ab
                        else "flashwam_same_state_aa_oracle_v3"
                    ),
                    "task": args["task_name"],
                    "seed": int(now_seed),
                    "prompt": prompt,
                    "intervention_frame": _stage_l_frame_st_id,
                    "repeats": stage_l_oracle_repeats,
                    "symmetric_restore": True,
                    "canonical_snapshot_sha256": _stage_l_snapshot_hash,
                    "canonical_prefix_runtime_sha256": (
                        _stage_l_canonical_kv_hashes
                    ),
                    "prefix_reconstruction_exact": _stage_l_prefix_exact,
                    "common_causal_inputs_exact": _stage_l_common_exact,
                    "within_family_observable_exact": (
                        _stage_l_family_observable_exact
                    ),
                    "start_full_state_exact": _stage_l_start_full_state_exact,
                    "start_core_state_exact": _stage_l_start_core_state_exact,
                    "within_family_end_full_state_exact": (
                        _stage_l_end_family_full_state_exact
                    ),
                    "within_family_end_core_state_exact": (
                        _stage_l_end_family_core_state_exact
                    ),
                    "strict_causal_contract_go": _stage_l_strict_contract,
                    "near_strict_causal_contract_go": (
                        _stage_l_near_strict_contract
                    ),
                    "branch_b_artifact": (
                        _stage_l_intervention["artifact"]
                        if _stage_l_intervention
                        else None
                    ),
                    "branch_b_key": (
                        _stage_l_intervention["action_key"]
                        if _stage_l_intervention
                        else None
                    ),
                    "stage_m_pause_contract": _stage_m_pause_contract,
                    "transition": _stage_l_transition,
                    "branches": _stage_l_summaries,
                }
                torch.save(
                    {**_stage_l_result, "records": _stage_l_records},
                    _stage_l_output.with_suffix(".pt"),
                )
                _stage_l_output.write_text(
                    json.dumps(_stage_l_result, indent=2) + "\\n",
                    encoding="utf-8",
                )
                print(
                    (
                        "STAGE_L_SAME_STATE_AB "
                        if _stage_l_is_ab
                        else "STAGE_L_SAME_STATE_AA "
                    )
                    + json.dumps(
                        {
                            "output": str(_stage_l_output.resolve()),
                            "prefix_exact": _stage_l_prefix_exact,
                            "common_inputs_exact": _stage_l_common_exact,
                            "family_observable_exact": (
                                _stage_l_family_observable_exact
                            ),
                            "strict_contract": _stage_l_strict_contract,
                            "near_strict_contract": (
                                _stage_l_near_strict_contract
                            ),
                            "branches": len(_stage_l_summaries),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                TASK_ENV.close_env()
                if not _stage_l_near_strict_contract:
                    raise AssertionError(
                        "Stage-L same-state causal contract failed"
                    )
                raise SystemExit(0)
"""
    if source.count(oracle_kv_needle) != 1:
        raise RuntimeError("Upstream Stage-L KV oracle hook changed")
    source = source.replace(oracle_kv_needle, oracle_kv_replacement)
if stage_l_fork_early_enabled:
    _stage_l_install_deferred_planner_stub()
exec(compile(source, str(UPSTREAM), "exec"), globals())
