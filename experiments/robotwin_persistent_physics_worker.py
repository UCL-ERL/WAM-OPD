"""Persistent native RoboTwin PhysX worker with a parent-render bridge.

The production mode forks a renderer-free child before the next physics chunk
and keeps that child's SAPIEN scene and hidden PhysX solver state alive across
RPC calls.  The normal parent owns the renderer.  It temporarily restores a
worker end snapshot only to capture RGB/model observations, then restores its
untouched intervention snapshot; the worker, never the restored parent,
performs the next physics step.  A previous two-level fork experiment
(renderer grandchild) is intentionally not used: SAPIEN/Vulkan rejects that
second fork with ``cannot create buffer`` and a device-loss abort.

``clean_worker_renderer`` remains a diagnostic mode only.  It is not the
closed-loop production path because the observed native stack cannot safely
materialize a second renderer after the parent has initialized CUDA/Vulkan.

This module is infrastructure only.  It does not run a policy, create a
Teacher target, or perform training.
"""

from __future__ import annotations

import os
import pickle
import select
import struct
import traceback
from pathlib import Path
from typing import Any

import numpy as np


_HEADER = struct.Struct("!Q")


class PersistentWorkerError(RuntimeError):
    """The persistent simulator worker failed or violated its protocol."""


def validate_worker_operation(worker_mode: str, operation: str) -> None:
    """Fail closed when a renderer-free worker is asked to render.

    Keeping this check separate makes the safety boundary unit-testable without
    importing SAPIEN or forking a native simulator in a test process.
    """

    if operation == "render" and worker_mode != "clean_worker_renderer":
        raise PersistentWorkerError(
            "parent_render_bridge worker cannot render in child; "
            "use the parent snapshot bridge"
        )


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise BrokenPipeError("persistent worker pipe write returned zero")
        view = view[written:]


def _read_all(fd: int, size: int, *, timeout: float | None = None) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining:
        if timeout is not None:
            ready, _, _ = select.select([fd], [], [], timeout)
            if not ready:
                raise TimeoutError("persistent worker response timed out")
        chunk = os.read(fd, remaining)
        if not chunk:
            raise EOFError("persistent worker pipe closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_message(fd: int, value: object) -> None:
    payload = pickle.dumps(value, protocol=5)
    _write_all(fd, _HEADER.pack(len(payload)))
    _write_all(fd, payload)


def _receive_message(fd: int, *, timeout: float | None = None) -> object:
    header = _read_all(fd, _HEADER.size, timeout=timeout)
    (size,) = _HEADER.unpack(header)
    if size > 512 * 1024 * 1024:
        raise PersistentWorkerError(f"oversized worker response: {size} bytes")
    return pickle.loads(_read_all(fd, int(size), timeout=timeout))


def _error_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "status": "ERROR",
        "exception": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _render_from_worker_state(
    *,
    task_env: object,
    format_obs: object,
    materialize_renderer: object,
    prompt: str,
) -> dict[str, Any]:
    """Legacy diagnostic: fork a renderer-only grandchild.

    Kept as a narrow diagnostic seam, but production persistent workers use
    :func:`_render_in_worker` because a second Vulkan fork is not safe.
    """

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child executes independently
        os.close(read_fd)
        try:
            materialize_renderer(task_env)
            raw_obs = task_env.get_obs()
            formatted = format_obs(raw_obs, prompt)
            # check_success is intentionally evaluated after rendering.  Its
            # task-specific predicates may inspect the freshly updated camera
            # observation, but no worker physics state is mutated by COW.
            success = bool(task_env.check_success())
            _send_message(
                write_fd,
                {
                    "status": "PASS",
                    "observation": formatted,
                    "task_eval_success": bool(getattr(task_env, "eval_success", False)),
                    "task_check_success": success,
                },
            )
            os.close(write_fd)
            os._exit(0)
        except BaseException as exc:
            try:
                _send_message(write_fd, _error_payload(exc))
            finally:
                os.close(write_fd)
                os._exit(1)

    os.close(write_fd)
    try:
        response = _receive_message(read_fd, timeout=180.0)
    finally:
        os.close(read_fd)
    waited_pid, status = os.waitpid(pid, 0)
    if waited_pid != pid or not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
        raise PersistentWorkerError(
            f"renderer child failed pid={pid} status={status}: {response!r}"
        )
    if not isinstance(response, dict) or response.get("status") != "PASS":
        raise PersistentWorkerError(f"renderer child returned {response!r}")
    return response


def _render_in_worker(
    *,
    task_env: object,
    format_obs: object,
    prompt: str,
) -> dict[str, Any]:
    """Render using the renderer materialized once in the persistent worker."""

    raw_obs = task_env.get_obs()
    formatted = format_obs(raw_obs, prompt)
    return {
        "status": "PASS",
        "observation": formatted,
        "task_eval_success": bool(getattr(task_env, "eval_success", False)),
        "task_check_success": bool(task_env.check_success()),
    }


def _worker_main(
    *,
    task_env: object,
    prompt: str,
    initial_eef_pose: np.ndarray,
    command_read_fd: int,
    response_write_fd: int,
    format_obs: object,
    materialize_renderer: object,
    worker_mode: str,
) -> None:
    """Worker entry point; called only in the pre-CUDA fork child."""

    service = None
    try:
        from experiments.robotwin_branch_oracle import (
            configure_native_physics_only_child,
            execute_env_action_chunk_physics_only,
        )
        from experiments.robotwin_planner_service import RoboTwinPlannerService
        from experiments.robotwin_sim_snapshot import (
            capture_simulator_snapshot,
            restore_simulator_snapshot,
            simulator_state_sha256,
        )
        from evaluation.robotwin.eval_polict_client_openpi import add_init_pose

        if worker_mode == "clean_worker_renderer":
            # Materialize the renderer once in this clean fork child.  It must
            # not be created in a later grandchild: the native Vulkan backend
            # loses its device/buffers on a second fork.
            if materialize_renderer is None:
                raise ValueError("clean_worker_renderer requires a renderer callback")
            materialize_renderer(task_env)
        elif worker_mode != "parent_render_bridge":
            raise ValueError(f"unknown persistent worker mode: {worker_mode!r}")
        # Delay planner-process creation until the first physical step.  The
        # parent creates one persistent worker per arm before loading CUDA;
        # eagerly creating two Curobo processes in every idle arm worker
        # consumed roughly 8--12 GiB before model inference began and caused
        # long-horizon native discovery to OOM.  Lazy startup keeps the same
        # clean forkserver boundary and live-qpos planner semantics while
        # allowing only the currently executing arm to own planner processes.
        service = None
        _send_message(
            response_write_fd,
            {
                "status": "READY",
                "pid": os.getpid(),
                "planner_worker_pids": {},
                "planner_start": "lazy_on_first_step",
            },
        )

        while True:
            message = _receive_message(command_read_fd, timeout=None)
            if not isinstance(message, dict):
                raise PersistentWorkerError("worker command must be a dict")
            operation = message.get("op")
            if operation == "close":
                _send_message(response_write_fd, {"status": "CLOSED"})
                if service is not None:
                    service.close()
                os._exit(0)
            if operation == "render":
                validate_worker_operation(worker_mode, operation)
                result = _render_in_worker(
                    task_env=task_env,
                    format_obs=format_obs,
                    prompt=prompt,
                )
                _send_message(response_write_fd, result)
                continue
            if operation == "snapshot":
                snapshot = capture_simulator_snapshot(task_env, capture_cuda_rng=False)
                _send_message(
                    response_write_fd,
                    {
                        "status": "PASS",
                        "simulator_sha256": simulator_state_sha256(snapshot),
                        "snapshot": snapshot,
                    },
                )
                continue
            if operation == "restore":
                snapshot = message.get("snapshot")
                if not isinstance(snapshot, dict):
                    raise PersistentWorkerError("restore requires a serialized snapshot")
                before = capture_simulator_snapshot(task_env, capture_cuda_rng=False)
                restore_simulator_snapshot(task_env, snapshot)
                after = capture_simulator_snapshot(task_env, capture_cuda_rng=False)
                _send_message(
                    response_write_fd,
                    {
                        "status": "PASS",
                        "before_simulator_sha256": simulator_state_sha256(before),
                        "after_simulator_sha256": simulator_state_sha256(after),
                        "requested_simulator_sha256": simulator_state_sha256(snapshot),
                        "exact": simulator_state_sha256(after)
                        == simulator_state_sha256(snapshot),
                    },
                )
                continue
            if operation != "step":
                raise PersistentWorkerError(f"unknown worker operation: {operation!r}")

            if service is None:
                # The worker itself was forked before parent CUDA
                # initialization.  The planner children use forkserver and
                # therefore remain clean even though this lazy call happens
                # after the parent has loaded the native model.
                service = RoboTwinPlannerService(
                    task_env.robot,
                    start_method="forkserver",
                )

            action = np.asarray(message["action"])
            if action.shape != (16, 2, 16):
                raise ValueError(f"expected native 1v action shape (16, 2, 16), got {action.shape}")
            start_frame = int(message.get("start_frame", 0))
            capture_intermediate_snapshots = bool(
                message.get("capture_intermediate_snapshots", False)
            )
            before = capture_simulator_snapshot(task_env, capture_cuda_rng=False)
            before_hash = simulator_state_sha256(before)
            cleanup = configure_native_physics_only_child(
                task_env=task_env,
                planner_service=service,
            )
            try:
                execution = execute_env_action_chunk_physics_only(
                    task_env=task_env,
                    action=action,
                    initial_eef_pose=initial_eef_pose,
                    add_init_pose=add_init_pose,
                    start_frame=start_frame,
                    capture_intermediate_snapshots=capture_intermediate_snapshots,
                    max_action_steps=message.get("max_action_steps"),
                )
            finally:
                cleanup()
            after = capture_simulator_snapshot(task_env, capture_cuda_rng=False)
            after_hash = simulator_state_sha256(after)
            rendered = (
                _render_in_worker(
                    task_env=task_env,
                    format_obs=format_obs,
                    prompt=prompt,
                )
                if worker_mode == "clean_worker_renderer"
                else {
                    "task_eval_success": None,
                    "task_check_success": None,
                }
            )
            _send_message(
                response_write_fd,
                {
                    "status": "PASS",
                    "before_simulator_sha256": before_hash,
                    "after_simulator_sha256": after_hash,
                    "before_snapshot": before,
                    "after_snapshot": after,
                    "physical_execution": execution,
                    "observation": rendered.get("observation"),
                    "task_eval_success": rendered.get("task_eval_success"),
                    "task_check_success": rendered.get("task_check_success"),
                },
            )
    except BaseException as exc:  # pragma: no cover - worker failure path
        try:
            _send_message(response_write_fd, _error_payload(exc))
        except BaseException:
            pass
        os._exit(1)
    finally:
        if service is not None:
            try:
                service.close()
            except BaseException:
                pass
        try:
            os.close(command_read_fd)
            os.close(response_write_fd)
        except OSError:
            pass


class PersistentNativePhysicsWorker:
    """Parent-side RPC handle for one persistent native simulator state."""

    def __init__(
        self,
        *,
        task_env: object,
        prompt: str,
        initial_eef_pose: np.ndarray,
        format_obs: object,
        materialize_renderer: object,
        worker_mode: str = "clean_worker_renderer",
        response_timeout: float = 240.0,
    ) -> None:
        if not hasattr(os, "fork"):
            raise RuntimeError("persistent native worker requires POSIX os.fork")
        command_read_fd, command_write_fd = os.pipe()
        response_read_fd, response_write_fd = os.pipe()
        self._command_write_fd = command_write_fd
        self._response_read_fd = response_read_fd
        self._response_timeout = float(response_timeout)
        self._closed = False
        pid = os.fork()
        if pid == 0:  # pragma: no cover - child executes independently
            os.close(command_write_fd)
            os.close(response_read_fd)
            _worker_main(
                task_env=task_env,
                prompt=prompt,
                initial_eef_pose=np.asarray(initial_eef_pose, dtype=np.float64),
                command_read_fd=command_read_fd,
                response_write_fd=response_write_fd,
                format_obs=format_obs,
                materialize_renderer=materialize_renderer,
                worker_mode=worker_mode,
            )
            os._exit(1)

        os.close(command_read_fd)
        os.close(response_write_fd)
        self.pid = int(pid)
        ready = self._receive()
        if not isinstance(ready, dict) or ready.get("status") != "READY":
            self.close(force=True)
            raise PersistentWorkerError(f"worker did not become ready: {ready!r}")
        self.ready = ready

    def _send(self, message: dict[str, Any]) -> None:
        if self._closed:
            raise PersistentWorkerError("worker is closed")
        _send_message(self._command_write_fd, message)

    def _receive(self) -> object:
        response = _receive_message(
            self._response_read_fd,
            timeout=self._response_timeout,
        )
        if isinstance(response, dict) and response.get("status") == "ERROR":
            raise PersistentWorkerError(
                f"worker error {response.get('exception')}: {response.get('message')}\n"
                f"{response.get('traceback', '')}"
            )
        return response

    def render(self) -> dict[str, Any]:
        self._send({"op": "render"})
        response = self._receive()
        if not isinstance(response, dict) or response.get("status") != "PASS":
            raise PersistentWorkerError(f"invalid render response: {response!r}")
        return response

    def step(
        self,
        action: np.ndarray,
        *,
        start_frame: int = 0,
        capture_intermediate_snapshots: bool = False,
        max_action_steps: int | None = None,
    ) -> dict[str, Any]:
        self._send(
            {
                "op": "step",
                "action": np.asarray(action).copy(),
                "start_frame": int(start_frame),
                "capture_intermediate_snapshots": bool(
                    capture_intermediate_snapshots
                ),
                "max_action_steps": max_action_steps,
            }
        )
        response = self._receive()
        if not isinstance(response, dict) or response.get("status") != "PASS":
            raise PersistentWorkerError(f"invalid step response: {response!r}")
        return response

    def snapshot(self) -> dict[str, Any]:
        self._send({"op": "snapshot"})
        response = self._receive()
        if not isinstance(response, dict) or response.get("status") != "PASS":
            raise PersistentWorkerError(f"invalid snapshot response: {response!r}")
        return response

    def restore(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Restore a serialized root inside this same persistent worker.

        V0I uses one worker for all arms of a paired unit.  This operation is
        deliberately narrow: it restores only an already serialized root and
        returns an independent post-restore state hash.  It does not create a
        new worker or call a renderer.
        """

        self._send({"op": "restore", "snapshot": snapshot})
        response = self._receive()
        if not isinstance(response, dict) or response.get("status") != "PASS":
            raise PersistentWorkerError(f"invalid restore response: {response!r}")
        return response

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if not force:
            try:
                _send_message(self._command_write_fd, {"op": "close"})
                self._receive()
            except BaseException:
                force = True
        try:
            os.close(self._command_write_fd)
        except OSError:
            pass
        try:
            os.close(self._response_read_fd)
        except OSError:
            pass
        waited_pid, status = os.waitpid(self.pid, 0 if not force else os.WNOHANG)
        if force and waited_pid == 0:
            os.kill(self.pid, 9)
            os.waitpid(self.pid, 0)
        elif waited_pid != self.pid:
            os.waitpid(self.pid, 0)
        if not force and (not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0):
            raise PersistentWorkerError(
                f"worker exited unsuccessfully pid={self.pid} status={status}"
            )

    def __enter__(self) -> "PersistentNativePhysicsWorker":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close(force=exc_type is not None)
