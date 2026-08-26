"""Simulation-only RoboTwin state capture for same-state intervention tests."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random
from typing import Any

import numpy as np
import torch


SCHEMA = "robotwin_simulator_snapshot_v2"
TASK_STATE_FIELDS = (
    "eval_success",
    "stage_success_tag",
    "take_action_cnt",
    "plan_success",
    "left_cnt",
    "right_cnt",
    "FRAME_IDX",
    "suc",
    "test_num",
    "origin_z",
    "arm_tag",
)
ROBOT_STATE_FIELDS = ("left_gripper_val", "right_gripper_val")


def _update_digest(digest: Any, value: object) -> None:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"array\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
        return
    if isinstance(value, np.generic):
        _update_digest(digest, np.asarray(value))
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(f"{type(value).__name__}:{len(value)}\0".encode("ascii"))
        for item in value:
            _update_digest(digest, item)
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        digest.update(b"bytes\0")
        digest.update(bytes(value))
        return
    if hasattr(value, "p") and hasattr(value, "q"):
        digest.update(b"pose\0")
        _update_digest(digest, value.p)
        _update_digest(digest, value.q)
        return
    if value is None or isinstance(value, (str, int, float, bool)):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(value, allow_nan=False).encode("utf-8"))
        digest.update(b"\0")
        return
    raise TypeError(f"unsupported simulator state value: {type(value).__name__}")


def simulator_state_sha256(snapshot: dict[str, Any]) -> str:
    """Hash all captured physical/controller/task state, excluding RNG."""

    if snapshot.get("schema") != SCHEMA:
        raise ValueError("unsupported simulator snapshot schema")
    digest = hashlib.sha256()
    for key in (
        "scene_pose_blob",
        "articulations",
        "dynamic_components",
        "task_state",
        "robot_state",
    ):
        _update_digest(digest, key)
        _update_digest(digest, snapshot[key])
    return digest.hexdigest()


def _flatten_state_arrays(
    value: object,
    prefix: str,
) -> dict[str, np.ndarray]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, (np.ndarray, np.generic)):
        return {prefix: np.asarray(value)}
    if isinstance(value, dict):
        result = {}
        for key in sorted(value):
            result.update(_flatten_state_arrays(value[key], f"{prefix}/{key}"))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten_state_arrays(item, f"{prefix}/{index}"))
        return result
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {prefix: np.frombuffer(bytes(value), dtype=np.uint8)}
    if hasattr(value, "p") and hasattr(value, "q"):
        result = _flatten_state_arrays(value.p, f"{prefix}/p")
        result.update(_flatten_state_arrays(value.q, f"{prefix}/q"))
        return result
    if value is None or isinstance(value, (str, int, float, bool)):
        return {prefix: np.asarray(value)}
    raise TypeError(f"unsupported simulator state value: {type(value).__name__}")


def compare_simulator_states(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    """Report exact and numeric differences between two captured states."""

    keys = (
        "scene_pose_blob",
        "articulations",
        "dynamic_components",
        "task_state",
        "robot_state",
    )
    left_arrays = {}
    right_arrays = {}
    for key in keys:
        left_arrays.update(_flatten_state_arrays(left[key], key))
        right_arrays.update(_flatten_state_arrays(right[key], key))
    differences = {}
    max_abs = 0.0
    for key in sorted(set(left_arrays) | set(right_arrays)):
        if key not in left_arrays or key not in right_arrays:
            differences[key] = {
                "present_left": key in left_arrays,
                "present_right": key in right_arrays,
            }
            continue
        lvalue = left_arrays[key]
        rvalue = right_arrays[key]
        if lvalue.shape != rvalue.shape or lvalue.dtype != rvalue.dtype:
            differences[key] = {
                "left_shape": list(lvalue.shape),
                "right_shape": list(rvalue.shape),
                "left_dtype": str(lvalue.dtype),
                "right_dtype": str(rvalue.dtype),
            }
            continue
        if np.array_equal(lvalue, rvalue):
            continue
        item = {"exact": False}
        if np.issubdtype(lvalue.dtype, np.number):
            delta = np.abs(
                lvalue.astype(np.float64) - rvalue.astype(np.float64)
            )
            item["max_abs"] = float(delta.max()) if delta.size else 0.0
            item["mean_abs"] = float(delta.mean()) if delta.size else 0.0
            max_abs = max(max_abs, item["max_abs"])
        differences[key] = item
    return {
        "exact": not differences,
        "max_abs": max_abs,
        "differences": differences,
        "left_sha256": simulator_state_sha256(left),
        "right_sha256": simulator_state_sha256(right),
    }


def _name(value: object) -> str:
    getter = getattr(value, "get_name", None)
    if callable(getter):
        return str(getter())
    return str(getattr(value, "name"))


def _array(value: object) -> np.ndarray:
    return np.asarray(value).copy()


def _pose_array(pose: object) -> np.ndarray:
    return np.concatenate([_array(pose.p), _array(pose.q)])


def _pose_like(template: object, value: object) -> object:
    data = np.asarray(value)
    if data.shape != (7,):
        raise ValueError(f"pose shape differs: expected (7,), got {data.shape}")
    pose_type = type(template)
    try:
        return pose_type(p=data[:3], q=data[3:])
    except TypeError:
        return pose_type(data[:3], data[3:])


def _dynamic_components(scene: object) -> list[tuple[int, int, object, str]]:
    found = []
    for actor_index, actor in enumerate(scene.get_all_actors()):
        components = actor.get_components()
        for component_index, component in enumerate(components):
            required = (
                "get_linear_velocity",
                "set_linear_velocity",
                "get_angular_velocity",
                "set_angular_velocity",
            )
            if all(callable(getattr(component, name, None)) for name in required):
                found.append(
                    (actor_index, component_index, component, _name(actor))
                )
    return found


def _capture_rng(*, capture_cuda: bool = True) -> dict[str, Any]:
    cuda_states = None
    if capture_cuda and torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda": cuda_states,
    }


def capture_simulator_snapshot(
    task_env: object, *, capture_cuda_rng: bool = True
) -> dict[str, Any]:
    """Capture physical, controller-target, task-latch, and RNG state.

    ``capture_cuda_rng=False`` is reserved for a post-fork physical child.
    Calling CUDA RNG APIs after forking a CUDA-initialized parent is unsafe;
    the parent records the inherited CUDA RNG provenance separately.  The
    default remains unchanged for normal same-process snapshots.
    """

    scene = task_env.scene
    articulations = []
    for index, articulation in enumerate(scene.get_all_articulations()):
        joints = []
        for joint_index, joint in enumerate(articulation.get_active_joints()):
            joints.append(
                {
                    "index": joint_index,
                    "name": _name(joint),
                    "drive_target": _array(joint.get_drive_target()),
                    "drive_velocity_target": _array(
                        joint.get_drive_velocity_target()
                    ),
                }
            )
        articulations.append(
            {
                "index": index,
                "name": _name(articulation),
                "qpos": _array(articulation.get_qpos()),
                "qvel": _array(articulation.get_qvel()),
                "qacc": _array(articulation.get_qacc()),
                "qf": _array(articulation.get_qf()),
                "root_pose": _pose_array(articulation.get_root_pose()),
                "root_linear_velocity": _array(
                    articulation.get_root_linear_velocity()
                ),
                "root_angular_velocity": _array(
                    articulation.get_root_angular_velocity()
                ),
                "joints": joints,
            }
        )

    dynamic_components = []
    for actor_index, component_index, component, actor_name in _dynamic_components(
        scene
    ):
        dynamic_components.append(
            {
                "actor_index": actor_index,
                "actor_name": actor_name,
                "component_index": component_index,
                "component_type": type(component).__name__,
                "linear_velocity": _array(component.get_linear_velocity()),
                "angular_velocity": _array(component.get_angular_velocity()),
                "sleeping": bool(
                    component.is_sleeping()
                    if callable(getattr(component, "is_sleeping", None))
                    else component.is_sleeping
                ),
            }
        )

    task_state = {
        field: deepcopy(getattr(task_env, field))
        for field in TASK_STATE_FIELDS
        if hasattr(task_env, field)
    }
    robot_state = {
        field: deepcopy(getattr(task_env.robot, field))
        for field in ROBOT_STATE_FIELDS
        if hasattr(task_env, "robot") and hasattr(task_env.robot, field)
    }
    return {
        "schema": SCHEMA,
        "simulation_only": True,
        "scene_pose_blob": scene.pack_poses(),
        "articulations": articulations,
        "dynamic_components": dynamic_components,
        "task_state": task_state,
        "robot_state": robot_state,
        "rng": _capture_rng(capture_cuda=capture_cuda_rng),
    }


def perturb_simulator_state_for_audit(task_env: object) -> list[str]:
    """Make small visible mutations so a round-trip cannot pass vacuously."""

    changed = []
    for actor in task_env.scene.get_all_actors():
        getter = getattr(actor, "get_pose", None)
        setter = getattr(actor, "set_pose", None)
        if callable(getter) and callable(setter):
            pose = getter()
            value = _pose_array(pose)
            value[0] += 0.01
            setter(_pose_like(pose, value))
            changed.append("scene_pose")
            break

    articulations = list(task_env.scene.get_all_articulations())
    if articulations:
        articulation = articulations[0]
        qpos = _array(articulation.get_qpos())
        if qpos.size:
            qpos.flat[0] += 0.01
            articulation.set_qpos(qpos)
            changed.append("articulation")

    dynamics = _dynamic_components(task_env.scene)
    if dynamics:
        component = dynamics[0][2]
        velocity = _array(component.get_linear_velocity())
        velocity.flat[0] += 0.01
        component.set_linear_velocity(velocity)
        changed.append("dynamic_velocity")

    for field in ("eval_success", "stage_success_tag", "plan_success"):
        if hasattr(task_env, field):
            setattr(task_env, field, not bool(getattr(task_env, field)))
            changed.append("task_flag")
            break
    if not changed:
        raise ValueError("no simulator state family was available to perturb")
    return changed


def _validate_identity(task_env: object, snapshot: dict[str, Any]) -> None:
    if snapshot.get("schema") != SCHEMA:
        raise ValueError(f"unsupported simulator snapshot schema: {snapshot.get('schema')}")
    if snapshot.get("simulation_only") is not True:
        raise ValueError("simulator snapshot must be marked simulation_only")

    actual_articulations = list(task_env.scene.get_all_articulations())
    expected_articulations = snapshot["articulations"]
    if len(actual_articulations) != len(expected_articulations):
        raise ValueError("articulation count differs")
    for actual, expected in zip(actual_articulations, expected_articulations):
        if _name(actual) != expected["name"]:
            raise ValueError("articulation identity differs")
        if actual.get_qpos().shape != expected["qpos"].shape:
            raise ValueError("articulation qpos shape differs")
        if actual.get_qvel().shape != expected["qvel"].shape:
            raise ValueError("articulation qvel shape differs")
        if actual.get_qacc().shape != expected["qacc"].shape:
            raise ValueError("articulation qacc shape differs")
        if actual.get_qf().shape != expected["qf"].shape:
            raise ValueError("articulation qf shape differs")
        actual_joints = list(actual.get_active_joints())
        if len(actual_joints) != len(expected["joints"]):
            raise ValueError("active-joint count differs")
        for actual_joint, expected_joint in zip(actual_joints, expected["joints"]):
            if _name(actual_joint) != expected_joint["name"]:
                raise ValueError("active-joint identity differs")

    actual_dynamics = _dynamic_components(task_env.scene)
    expected_dynamics = snapshot["dynamic_components"]
    if len(actual_dynamics) != len(expected_dynamics):
        raise ValueError("dynamic actor count differs")
    for actual, expected in zip(actual_dynamics, expected_dynamics):
        actor_index, component_index, component, actor_name = actual
        if (
            actor_index != expected["actor_index"]
            or component_index != expected["component_index"]
            or actor_name != expected["actor_name"]
            or type(component).__name__ != expected["component_type"]
        ):
            raise ValueError("actor identity differs")


def _restore_rng(rng: dict[str, Any]) -> None:
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.random.set_rng_state(rng["torch_cpu"])
    cuda_states = rng["torch_cuda"]
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise ValueError("snapshot has CUDA RNG state but CUDA is unavailable")
        if torch.cuda.device_count() != len(cuda_states):
            raise ValueError("CUDA RNG device count differs")
        torch.cuda.set_rng_state_all(cuda_states)


def restore_simulator_snapshot(task_env: object, snapshot: dict[str, Any]) -> None:
    """Restore a snapshot after validating all indexed simulator identities."""

    _validate_identity(task_env, snapshot)
    scene = task_env.scene
    scene.unpack_poses(snapshot["scene_pose_blob"])

    for articulation, saved in zip(
        scene.get_all_articulations(), snapshot["articulations"]
    ):
        articulation.set_root_pose(
            _pose_like(articulation.get_root_pose(), saved["root_pose"])
        )
        articulation.set_qpos(saved["qpos"])
        articulation.set_qvel(saved["qvel"])
        articulation.set_qf(saved["qf"])
        articulation.set_root_linear_velocity(saved["root_linear_velocity"])
        articulation.set_root_angular_velocity(saved["root_angular_velocity"])
        for joint, saved_joint in zip(
            articulation.get_active_joints(), saved["joints"]
        ):
            joint.set_drive_target(saved_joint["drive_target"])
            joint.set_drive_velocity_target(saved_joint["drive_velocity_target"])
        articulation.set_qacc(saved["qacc"])

    for actual, saved in zip(
        _dynamic_components(scene), snapshot["dynamic_components"]
    ):
        component = actual[2]
        component.set_linear_velocity(saved["linear_velocity"])
        component.set_angular_velocity(saved["angular_velocity"])
        if saved["sleeping"]:
            component.put_to_sleep()
        else:
            component.wake_up()

    for field, value in snapshot["task_state"].items():
        if not hasattr(task_env, field):
            raise ValueError(f"task state field disappeared: {field}")
        setattr(task_env, field, deepcopy(value))
    if snapshot["robot_state"] and not hasattr(task_env, "robot"):
        raise ValueError("snapshot has robot state but task robot disappeared")
    for field, value in snapshot["robot_state"].items():
        if not hasattr(task_env.robot, field):
            raise ValueError(f"robot state field disappeared: {field}")
        setattr(task_env.robot, field, deepcopy(value))
    _restore_rng(snapshot["rng"])
