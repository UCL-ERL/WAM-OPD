"""Spawned RoboTwin planner service for forked physical branches.

The native RoboTwin ``Robot`` normally owns a Curobo planner in the same
Python process.  A branch made with ``os.fork`` cannot call that planner after
the fork: PyTorch marks the child as a bad CUDA fork and Curobo then fails
while constructing tensors.  This module keeps the planner in an independent
clean ``spawn``/``forkserver`` process and exposes the same pipe protocol that RoboTwin's
``Robot.left_plan_path``/``right_plan_path`` already use.

The child branch therefore sends the *live child qpos* to a pre-existing
planner service.  It does not replay a path recorded at a different qpos and
does not silently approximate planner chronology.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import multiprocessing as mp
from pathlib import Path
import pickle
import traceback
from typing import Any

import numpy as np


def _planner_request_sha256(message: dict[str, Any]) -> str:
    """Hash the semantic planner input, including the live child qpos."""

    payload = {key: value for key, value in message.items() if key != "cmd"}
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


def _planner_process_worker(conn: Any, spec: dict[str, Any]) -> None:
    """Run one Curobo worker in a clean spawned interpreter."""

    try:
        import sapien.core as sapien
        from envs.robot.planner import CuroboPlanner

        origin_pose = sapien.Pose(
            np.asarray(spec["origin_p"], dtype=np.float64),
            np.asarray(spec["origin_q"], dtype=np.float64),
        )
        planner = CuroboPlanner(
            origin_pose,
            list(spec["joints_name"]),
            list(spec["all_joints"]),
            yml_path=str(spec["yml_path"]),
        )
        plan_cache: dict[str, Any] = {}
        conn.send({"__waopd_ready__": True})
        while True:
            message = conn.recv()
            command = message.get("cmd")
            if command == "exit":
                conn.send({"__waopd_exit__": True})
                return
            if command in ("plan_path", "plan_batch"):
                request_hash = _planner_request_sha256(message)
                if request_hash in plan_cache:
                    result = copy.deepcopy(plan_cache[request_hash])
                elif command == "plan_path":
                    result = planner.plan_path(
                        message["qpos"],
                        message["target_pose"],
                        constraint_pose=message.get("constraint_pose"),
                        arms_tag=message.get("arms_tag"),
                    )
                    plan_cache[request_hash] = copy.deepcopy(result)
                else:
                    result = planner.plan_batch(
                        message["qpos"],
                        message["target_pose_list"],
                        constraint_pose=message.get("constraint_pose"),
                        arms_tag=message.get("arms_tag"),
                    )
                    plan_cache[request_hash] = copy.deepcopy(result)
            elif command == "plan_grippers":
                result = planner.plan_grippers(
                    message["now_val"], message["target_val"]
                )
            elif command == "update_point_cloud":
                planner.update_point_cloud(
                    message["pcd"], resolution=message.get("resolution", 0.02)
                )
                plan_cache.clear()
                result = "ok"
            elif command == "reset":
                planner.motion_gen.reset(reset_seed=True)
                plan_cache.clear()
                result = "ok"
            else:
                raise ValueError(f"unknown planner command: {command!r}")
            conn.send(result)
    except BaseException as exc:  # pragma: no cover - runs in spawned process
        try:
            conn.send(
                {
                    "__waopd_error__": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


class _FailClosedEndpoint:
    """Pipe facade that converts worker failures into immediate exceptions."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def send(self, message: dict[str, Any]) -> None:
        self._connection.send(message)

    def recv(self) -> Any:
        result = self._connection.recv()
        if isinstance(result, dict) and "__waopd_error__" in result:
            raise RuntimeError(
                "RoboTwin planner service failed: "
                f"{result['__waopd_error__']}: {result.get('message', '')}\n"
                f"{result.get('traceback', '')}"
            )
        return result

    def close(self) -> None:
        self._connection.close()


def _robot_planner_spec(robot: object, arm: str) -> dict[str, Any]:
    entity = getattr(robot, f"{arm}_entity")
    origin = getattr(robot, f"{arm}_entity_origion_pose")
    robot_file = Path(getattr(robot, f"{arm}_curobo_yml_path"))
    # ``Robot.set_planner`` derives these names for a dual-arm embodiment;
    # the shared ``curobo.yml`` path is only an intermediate field and is not
    # present in the native Aloha assets.
    if robot_file.name == "curobo.yml":
        robot_file = robot_file.with_name(f"curobo_{arm}.yml")
    # RoboTwin stores a relative path rooted at CONFIGS.ROOT_PATH.  Resolve it
    # in the parent so the spawned worker does not depend on its cwd.
    try:
        from envs import _GLOBAL_CONFIGS as configs

        yml_path = Path(configs.ROOT_PATH) / robot_file
    except Exception:
        yml_path = robot_file
    return {
        "origin_p": np.asarray(origin.p).copy(),
        "origin_q": np.asarray(origin.q).copy(),
        "joints_name": list(getattr(robot, f"{arm}_arm_joints_name")),
        "all_joints": [joint.get_name() for joint in entity.get_active_joints()],
        "yml_path": str(yml_path.resolve()),
    }


@dataclass
class RoboTwinPlannerService:
    """Two spawned planner workers attached to a native RoboTwin robot."""

    robot: object
    start_method: str = "spawn"

    def __post_init__(self) -> None:
        if self.start_method not in ("spawn", "forkserver"):
            raise ValueError(
                "forked physical branches require a clean spawned/forkserver planner service"
            )
        self._context = mp.get_context(self.start_method)
        self._connections: dict[str, Any] = {}
        self._processes: dict[str, Any] = {}
        for arm in ("left", "right"):
            parent_conn, child_conn = self._context.Pipe()
            process = self._context.Process(
                target=_planner_process_worker,
                args=(child_conn, _robot_planner_spec(self.robot, arm)),
                name=f"waopd-{arm}-curobo-planner",
            )
            process.daemon = True
            process.start()
            child_conn.close()
            self._connections[arm] = _FailClosedEndpoint(parent_conn)
            self._processes[arm] = process

        # Do not let the branch process call a local Curobo object.  The
        # native Robot methods already use these fields when this flag is set.
        self.robot.communication_flag = True
        self.robot.left_conn = self._connections["left"]
        self.robot.right_conn = self._connections["right"]
        self.robot.left_proc = self._processes["left"]
        self.robot.right_proc = self._processes["right"]

        for arm in ("left", "right"):
            ready = self._connections[arm].recv()
            if not isinstance(ready, dict) or ready.get("__waopd_ready__") is not True:
                self.close()
                raise RuntimeError(f"planner service {arm} did not become ready")

    @property
    def worker_pids(self) -> dict[str, int | None]:
        return {
            arm: (process.pid if process.pid is not None else None)
            for arm, process in self._processes.items()
        }

    def close(self) -> None:
        for arm, endpoint in self._connections.items():
            try:
                endpoint.send({"cmd": "exit"})
                endpoint.recv()
            except (BrokenPipeError, EOFError, OSError, RuntimeError):
                pass
            try:
                endpoint.close()
            except OSError:
                pass
        for process in self._processes.values():
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self._connections.clear()
        self._processes.clear()
