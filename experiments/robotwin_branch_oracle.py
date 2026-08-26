"""Same-process utilities for branch-consistent RoboTwin interventions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch


def array_sha256(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_branch_intervention(
    *,
    path: str | Path,
    action_key: str,
    task_name: str,
    environment_seed: int,
    intervention_frame: int,
    prompt: str,
) -> dict[str, object]:
    """Load a Bridge action only when it belongs to the exact causal state."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("intervention artifact must contain a dictionary")
    expected = {
        "task_id": task_name,
        "env_seed": int(environment_seed),
        "frame_st_id": int(intervention_frame),
        "prompt": prompt,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"intervention artifact {key} mismatch: "
                f"expected={value!r} actual={payload.get(key)!r}"
            )
    if action_key not in payload:
        raise ValueError(f"intervention artifact lacks {action_key!r}")
    if "student_env_action" not in payload:
        raise ValueError("intervention artifact lacks 'student_env_action'")
    action = np.asarray(payload[action_key]).copy()
    student_action = np.asarray(payload["student_env_action"]).copy()
    expected_shape = (16, 2, 16)
    if action.shape != expected_shape or student_action.shape != expected_shape:
        raise ValueError(
            "intervention actions must both have shape "
            f"{expected_shape}: action={action.shape}, student={student_action.shape}"
        )
    return {
        "action": action,
        "student_action": student_action,
        "artifact": str(Path(path).resolve()),
        "action_key": action_key,
    }


def rebuild_flashwam_prefix(
    *,
    model: object,
    prompt: str,
    environment_seed: int,
    first_observation: dict[str, object],
    prefix_records: Iterable[dict[str, object]],
    inference_kwargs: dict[str, object],
    offline_context_replay: bool = False,
) -> dict[str, object]:
    """Reset, replay the exact canonical KV stream, and infer the next action."""

    model.infer(
        {
            "reset": True,
            "prompt": prompt,
            "save_visualization": False,
            "seed": int(environment_seed),
        }
    )
    if offline_context_replay:
        model.infer(
            {
                "obs": first_observation,
                "initialize_context": True,
                "save_visualization": False,
            }
        )
    replay_audit = []
    for record in prefix_records:
        generated = (
            {}
            if offline_context_replay
            else model.infer(
                {
                    "obs": first_observation,
                    "prompt": prompt,
                    "save_visualization": False,
                    **inference_kwargs,
                }
            )
        )
        cached = model.infer(
            {
                "obs": record["observations"],
                "compute_kv_cache": True,
                "imagine": False,
                "save_visualization": False,
                "state": record["env_action"],
            }
        )
        replay_audit.append(
            {
                "generated_action_sha256": (
                    array_sha256(generated["action"])
                    if "action" in generated
                    else None
                ),
                "pre_action_runtime_sha256": generated.get(
                    "pre_action_runtime_sha256"
                ),
                "post_kv_runtime_sha256": cached.get(
                    "post_kv_runtime_sha256"
                ),
            }
        )
    result = model.infer(
        {
            "obs": first_observation,
            "prompt": prompt,
            "save_visualization": False,
            **inference_kwargs,
        }
    )
    result["prefix_replay_audit"] = replay_audit
    return result


def execute_env_action_chunk(
    *,
    task_env: object,
    action: object,
    initial_eef_pose: object,
    add_init_pose: Callable[[np.ndarray, np.ndarray], np.ndarray],
    format_obs: Callable[[object, str], dict[str, object]],
    prompt: str,
) -> list[dict[str, object]]:
    """Execute one non-initial FlashWAM action chunk exactly as RoboTwin does."""

    action_array = np.asarray(action)
    if action_array.ndim != 3 or action_array.shape[1] < 1:
        raise ValueError(f"unexpected action shape: {action_array.shape}")
    if action_array.shape[2] % 4 != 0:
        raise ValueError("action horizon must be divisible into four observations")
    action_per_frame = action_array.shape[2] // 4
    initial_pose_array = np.asarray(initial_eef_pose, dtype=np.float64)
    observations = []
    for frame_index in range(action_array.shape[1]):
        for horizon_index in range(action_array.shape[2]):
            ee_action = action_array[:, frame_index, horizon_index]
            if action_array.shape[0] == 16:
                ee_action = add_init_pose(ee_action, initial_pose_array)
                ee_action = np.concatenate(
                    [
                        ee_action[:3],
                        ee_action[3:7] / np.linalg.norm(ee_action[3:7]),
                        ee_action[7:11],
                        ee_action[11:15] / np.linalg.norm(ee_action[11:15]),
                        ee_action[15:16],
                    ]
                )
            else:
                raise NotImplementedError(
                    "Stage-L branch oracle currently requires 16-channel actions"
                )
            task_env.take_action(ee_action, action_type="ee")
            if (horizon_index + 1) % action_per_frame == 0:
                observations.append(format_obs(task_env.get_obs(), prompt))
    return observations


def execute_env_action_chunk_physics_only(
    *,
    task_env: object,
    action: object,
    initial_eef_pose: object,
    add_init_pose: Callable[[np.ndarray, np.ndarray], np.ndarray],
    start_frame: int = 0,
    capture_intermediate_snapshots: bool = False,
    max_action_steps: int | None = None,
) -> dict[str, object]:
    """Execute an action chunk without calling the renderer or observation API.

    This seam is intentionally narrower than :func:`execute_env_action_chunk`.
    It is used only by a fork child whose parent has already completed model
    inference and captured the intervention observation.  Calling SAPIEN's
    renderer after a PyTorch/SAPIEN fork is unsafe on the native RoboTwin
    stack; direct ``take_action`` is the remaining CPU PhysX operation that
    can test whether copy-on-write actually preserves the hidden contact state.

    Native RoboTwin's ``take_action`` still calls ``_update_render`` and its
    EE planner.  Call :func:`configure_native_physics_only_child` before this
    function when using a real task.  That helper disables only the render and
    success-observation side effects and requires the planner to be an
    external spawned service; it does not fake a planner path or a post-state.
    """

    action_array = np.asarray(action)
    if action_array.ndim != 3 or action_array.shape[1] < 1:
        raise ValueError(f"unexpected action shape: {action_array.shape}")
    if action_array.shape[2] % 4 != 0:
        raise ValueError("action horizon must be divisible into four observations")
    initial_pose_array = np.asarray(initial_eef_pose, dtype=np.float64)
    if int(start_frame) < 0 or int(start_frame) >= action_array.shape[1]:
        raise ValueError(
            f"start_frame must be in [0, {action_array.shape[1]}), got {start_frame}"
        )
    available_steps = (
        action_array.shape[1] - int(start_frame)
    ) * action_array.shape[2]
    if max_action_steps is not None:
        if (
            isinstance(max_action_steps, bool)
            or int(max_action_steps) <= 0
            or int(max_action_steps) > int(available_steps)
        ):
            raise ValueError(
                "max_action_steps must be in "
                f"[1, {available_steps}], got {max_action_steps!r}"
            )
        max_action_steps = int(max_action_steps)
    action_per_frame = action_array.shape[2] // 4
    action_steps = 0
    executed_action_mask = np.zeros(action_array.shape[1:], dtype=np.bool_)
    terminal_reached = bool(getattr(task_env, "eval_success", False))
    terminal_action_position = None
    horizon_reached = False
    frame_snapshots = []
    for frame_index in range(int(start_frame), action_array.shape[1]):
        for horizon_index in range(action_array.shape[2]):
            if not terminal_reached and not horizon_reached:
                ee_action = action_array[:, frame_index, horizon_index]
                if action_array.shape[0] != 16:
                    raise NotImplementedError(
                        "physics-only branch oracle currently requires 16-channel actions"
                    )
                ee_action = add_init_pose(ee_action, initial_pose_array)
                ee_action = np.concatenate(
                    [
                        ee_action[:3],
                        ee_action[3:7] / np.linalg.norm(ee_action[3:7]),
                        ee_action[7:11],
                        ee_action[11:15] / np.linalg.norm(ee_action[11:15]),
                        ee_action[15:16],
                    ]
                )
                task_env.take_action(ee_action, action_type="ee")
                executed_action_mask[frame_index, horizon_index] = True
                action_steps += 1
                if bool(getattr(task_env, "eval_success", False)):
                    terminal_reached = True
                    terminal_action_position = [int(frame_index), int(horizon_index)]
                if (
                    max_action_steps is not None
                    and action_steps >= max_action_steps
                ):
                    horizon_reached = True
            # RoboTwin emits one key observation after every quarter of the
            # low-level horizon.  A native (16, 2, 16) action therefore
            # produces 4 observations per model frame, not one observation
            # per video latent frame.  The LingBot streaming VAE relies on
            # this exact temporal chunk boundary.
            if capture_intermediate_snapshots and (horizon_index + 1) % action_per_frame == 0:
                from experiments.robotwin_sim_snapshot import capture_simulator_snapshot

                frame_snapshots.append(
                    {
                        "frame_index": int(frame_index),
                        "horizon_index": int(horizon_index),
                        "snapshot": capture_simulator_snapshot(
                            task_env, capture_cuda_rng=False
                        ),
                    }
                )
    return {
        "action_steps": int(action_steps),
        "start_frame": int(start_frame),
        "executed_action_mask": executed_action_mask.tolist(),
        "terminal_reached": bool(terminal_reached),
        "terminal_action_position": terminal_action_position,
        "horizon_reached": bool(horizon_reached),
        "frame_snapshots": frame_snapshots,
    }


def configure_native_physics_only_child(
    *, task_env: object, planner_service: object | None = None
) -> Callable[[], None]:
    """Install the child-only native RoboTwin physical execution guard.

    ``Base_Task.take_action`` refreshes the renderer before/after an EE step
    and captures ``get_obs`` once the real task predicate succeeds.  The fork
    child must keep ``check_success`` intact so native terminal state is
    latched at the producing low-level action, while suppressing only those
    renderer/observation side effects.  Curobo is also forbidden in the child;
    the caller must attach a planner service whose planner calls are handled
    by another spawned process.

    The returned cleanup restores all patched attributes.  The fork runner
    invokes it before writing the child artifact.  Cleanup is deliberately
    local to the child and does not close the shared planner service.
    """

    if planner_service is None:
        raise ValueError(
            "native physics-only child requires an external spawned planner service"
        )
    robot = getattr(task_env, "robot", None)
    if robot is None or getattr(robot, "communication_flag", False) is not True:
        raise ValueError(
            "native physics-only child requires robot.communication_flag=True"
        )

    sentinel = object()
    saved = {
        "_update_render": getattr(task_env, "_update_render", sentinel),
        "get_obs": getattr(task_env, "get_obs", sentinel),
        "render_freq": getattr(task_env, "render_freq", sentinel),
        "eval_video_path": getattr(task_env, "eval_video_path", sentinel),
    }
    task_env._update_render = lambda: None
    task_env.get_obs = lambda: None
    task_env.render_freq = 0
    task_env.eval_video_path = None

    restored = False

    def cleanup() -> None:
        nonlocal restored
        if restored:
            return
        restored = True
        for key, value in saved.items():
            if value is sentinel:
                try:
                    delattr(task_env, key)
                except AttributeError:
                    pass
            else:
                setattr(task_env, key, value)

    return cleanup
