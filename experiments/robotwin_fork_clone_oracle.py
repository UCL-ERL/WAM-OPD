"""Fork-based causal simulator clones for native RoboTwin preflight.

The installed SAPIEN API does not expose PhysX solver/contact warm-start
state.  A same-process restore therefore cannot prove that two intervention
arms start from the same causal simulator state.  This module creates each
physical arm in a child forked directly at the intervention boundary.  The
child inherits the complete in-memory PhysX state through copy-on-write and
does not run CUDA/model inference; it only executes a precomputed action
chunk and captures simulation/observation evidence.

This is not an independent seeded reset.  A missing ``fork`` or any child
failure is a hard blocker; no tolerance is inferred.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
import traceback
from typing import Any, Callable, Iterable

import torch

from robotwin_sim_snapshot import (
    capture_simulator_snapshot,
    compare_simulator_states,
    restore_simulator_snapshot,
    simulator_state_sha256,
)


def _fork_child_cuda_safety_error() -> str | None:
    """Return a fail-closed diagnostic for PyTorch's unsafe fork state."""

    is_in_bad_fork = getattr(torch.cuda, "_is_in_bad_fork", None)
    if callable(is_in_bad_fork) and is_in_bad_fork():
        return (
            "BLOCKED: SAPIEN/PyTorch fork child is in a bad CUDA fork state; "
            "native curobo/renderer initialization is not safe after SAPIEN "
            "has been imported. Use a full simulator clone backend or a "
            "fresh process with an exact causal-state guarantee."
        )
    return None


def _write_child_error(path: Path, exc: BaseException) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "ERROR",
                "exception": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _child_run(
    *,
    task_env: object,
    action: object,
    initial_eef_pose: object,
    add_init_pose: Callable[[Any, Any], Any],
    execute_env_action_chunk: Callable[..., list[dict[str, Any]]] | None,
    execute_physics_action_chunk: Callable[..., dict[str, int]] | None,
    configure_physics_child: Callable[..., Callable[[], None] | None] | None,
    format_obs: Callable[[object, str], dict[str, Any]] | None,
    prompt: str,
    output_path: Path,
    branch: str,
    repeat: int,
    parent_start_observation: dict[str, Any] | None,
    parent_start_observation_sha256: str | None,
) -> None:
    """Run physical simulation in a fork child and persist raw evidence.

    ``execute_physics_action_chunk`` is the renderer-free mode.  It is the
    only mode allowed to bypass the bad-fork guard: the child must not call
    CUDA, curobo, SAPIEN rendering, or ``get_obs`` in that mode.
    """

    try:
        physical_only = execute_physics_action_chunk is not None
        if physical_only and execute_env_action_chunk is not None:
            raise ValueError(
                "fork child cannot receive both normal and physics-only action callbacks"
            )
        if not physical_only and execute_env_action_chunk is None:
            raise ValueError("normal fork child requires an action callback")
        if not physical_only:
            safety_error = _fork_child_cuda_safety_error()
            if safety_error is not None:
                raise RuntimeError(safety_error)
        # Do not call torch.cuda APIs in a child forked after CUDA init.  The
        # parent records the inherited CUDA RNG provenance in the oracle.
        start_snapshot = capture_simulator_snapshot(
            task_env, capture_cuda_rng=False
        )
        if physical_only:
            if parent_start_observation is None:
                raise ValueError(
                    "physics-only fork requires the parent intervention observation"
                )
            start_observation = deepcopy(parent_start_observation)
            cleanup = None
            if configure_physics_child is not None:
                cleanup = configure_physics_child(
                    task_env=task_env,
                    action=action,
                    branch=branch,
                    repeat=repeat,
                )
            try:
                execution = execute_physics_action_chunk(
                    task_env=task_env,
                    action=action,
                    initial_eef_pose=initial_eef_pose,
                    add_init_pose=add_init_pose,
                )
            finally:
                if cleanup is not None:
                    cleanup()
            end_observations: list[dict[str, Any]] = []
            end_observation_sha256 = None
        else:
            if format_obs is None:
                raise ValueError("normal fork child requires format_obs")
            start_observation = format_obs(task_env.get_obs(), prompt)
            end_observations = execute_env_action_chunk(
                task_env=task_env,
                action=action,
                initial_eef_pose=initial_eef_pose,
                add_init_pose=add_init_pose,
                format_obs=format_obs,
                prompt=prompt,
            )
            execution = {"action_steps": len(end_observations)}
            end_observation_sha256 = None
        end_snapshot = capture_simulator_snapshot(
            task_env, capture_cuda_rng=False
        )
        payload = {
            "schema": "robotwin_fork_clone_physical_record_v1",
            "branch": branch,
            "repeat": int(repeat),
            "status": "PASS",
            "physical_only": physical_only,
            "start_snapshot": start_snapshot,
            "start_observation": start_observation,
            "end_observations": end_observations,
            "start_observation_sha256": parent_start_observation_sha256,
            "end_observation_sha256": end_observation_sha256,
            "physical_execution": execution,
            "end_snapshot": end_snapshot,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
        # A physical-only child may share planner-service pipe endpoints and
        # a renderer-owned parent environment.  Calling close_env here can
        # close shared resources or invoke backend cleanup in the child.  The
        # process is exiting; let the OS reclaim its private COW state.
        if not physical_only:
            task_env.close_env()
        os._exit(0)
    except BaseException as exc:  # pragma: no cover - exercised in child
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_child_error(output_path.with_suffix(".error.json"), exc)
        if not physical_only:
            try:
                task_env.close_env()
            finally:
                os._exit(1)
        os._exit(1)


def run_forked_physical_branches(
    *,
    task_env: object,
    branch_actions: Iterable[dict[str, Any]],
    initial_eef_pose: object,
    add_init_pose: Callable[[Any, Any], Any],
    execute_env_action_chunk: Callable[..., list[dict[str, Any]]] | None = None,
    format_obs: Callable[[object, str], dict[str, Any]] | None = None,
    execute_physics_action_chunk: Callable[..., dict[str, int]] | None = None,
    configure_physics_child: Callable[..., Callable[[], None] | None] | None = None,
    parent_start_observation: dict[str, Any] | None = None,
    parent_start_observation_sha256: str | None = None,
    prompt: str,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """Fork each action arm from the current, untouched intervention state."""

    if not hasattr(os, "fork"):
        raise RuntimeError("fork-based causal clone requires POSIX os.fork")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for item in branch_actions:
        branch = str(item["branch"])
        repeat = int(item["repeat"])
        action = item["action"]
        output_path = root / f"physical_{branch}_repeat{repeat:02d}.pt"
        pid = os.fork()
        if pid == 0:  # pragma: no cover - child executes independently
            _child_run(
                task_env=task_env,
                action=action,
                initial_eef_pose=initial_eef_pose,
                add_init_pose=add_init_pose,
                execute_env_action_chunk=execute_env_action_chunk,
                execute_physics_action_chunk=execute_physics_action_chunk,
                configure_physics_child=configure_physics_child,
                format_obs=format_obs,
                prompt=prompt,
                output_path=output_path,
                branch=branch,
                repeat=repeat,
                parent_start_observation=parent_start_observation,
                parent_start_observation_sha256=parent_start_observation_sha256,
            )
            raise AssertionError("child runner returned unexpectedly")

        waited_pid, status = os.waitpid(pid, 0)
        if waited_pid != pid:
            raise RuntimeError(f"waitpid returned {waited_pid}, expected {pid}")
        if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
            error_path = output_path.with_suffix(".error.json")
            detail = error_path.read_text(encoding="utf-8") if error_path.is_file() else ""
            raise RuntimeError(
                f"fork physical child failed branch={branch} repeat={repeat} "
                f"status={status}: {detail[-4000:]}"
            )
        if not output_path.is_file():
            raise RuntimeError(
                f"fork physical child produced no artifact: {output_path}"
            )
        payload = torch.load(output_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("status") != "PASS":
            raise RuntimeError(f"invalid fork child artifact: {output_path}")
        records.append(payload)
    return records


def render_end_observation_in_parent(
    *,
    task_env: object,
    end_snapshot: dict[str, Any],
    parent_snapshot: dict[str, Any],
    format_obs: Callable[[object, str], dict[str, Any]],
    prompt: str,
) -> dict[str, Any]:
    """Render a physical-only child result without rendering in the child.

    The child owns the exact inherited PhysX state while it executes the
    action, but cannot safely initialize or use the inherited CUDA/Vulkan
    renderer.  The parent remains untouched during the child run, so after
    the child exits it can temporarily restore the child's visible state,
    call the normal observation path (which performs no physics step), and
    restore the parent's pre-render state.  The observation is deep-copied
    before restoration because camera arrays may be backed by renderer-owned
    buffers.

    This is a render bridge only.  It does not claim that
    ``restore_simulator_snapshot`` can restore hidden PhysX solver state for a
    subsequent step; no subsequent step is allowed on this parent state.
    """

    restore_simulator_snapshot(task_env, end_snapshot)
    try:
        rendered = format_obs(task_env.get_obs(), prompt)
        return deepcopy(rendered)
    finally:
        restore_simulator_snapshot(task_env, parent_snapshot)


def finalize_fork_manifest(
    manifest_path: str | Path, oracle_path: str | Path
) -> dict[str, Any]:
    """Build the standard Stage-L oracle from early-fork child records."""

    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("fork manifest must be a JSON object")
    if manifest.get("clone_method") != "os.fork_copy_on_write_before_renderer":
        raise ValueError("unexpected fork clone method")
    records = []
    for raw_path in manifest.get("child_records", []):
        path = Path(str(raw_path))
        if not path.is_file():
            raise FileNotFoundError(f"fork child artifact missing: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("status") != "PASS":
            raise ValueError(f"invalid fork child artifact: {path}")
        records.append(payload)
    if not records:
        raise ValueError("fork manifest contains no child records")

    start_comparisons = [
        compare_simulator_states(
            records[0]["start_snapshot"], record["start_snapshot"]
        )
        for record in records
    ]
    family_reference: dict[str, dict[str, Any]] = {}
    for record in records:
        family_reference.setdefault(str(record["action_family"]), record)
    end_comparisons = []
    for record in records:
        reference = family_reference[str(record["action_family"])]
        end_comparisons.append(
            compare_simulator_states(
                reference["end_snapshot"], record["end_snapshot"]
            )
        )

    prefix_values = [
        json.dumps(record.get("prefix_runtime_sha256"), sort_keys=True)
        for record in records
    ]
    prefix_canonical_exact = all(
        record.get("prefix_runtime_sha256")
        == record.get("canonical_prefix_runtime_sha256")
        for record in records
    )
    pre_action_values = [
        json.dumps(record.get("pre_action_runtime_sha256"), sort_keys=True)
        for record in records
    ]
    student_action_values = [str(record.get("student_action_sha256")) for record in records]
    common_exact = (
        len(set(prefix_values)) == 1
        and len(set(pre_action_values)) == 1
        and len(set(student_action_values)) == 1
        and all(record.get("source_student_exact") is True for record in records)
    )
    family_observable_exact = True
    for family, reference in family_reference.items():
        for record in records:
            if str(record["action_family"]) != family:
                continue
            if (
                record.get("executed_action_sha256")
                != reference.get("executed_action_sha256")
                or record.get("end_observation_sha256")
                != reference.get("end_observation_sha256")
            ):
                family_observable_exact = False

    start_full = all(item["exact"] for item in start_comparisons)
    end_full = all(item["exact"] for item in end_comparisons)
    oracle_path = Path(oracle_path)
    result = {
        "schema": (
            "flashwam_same_state_ab_oracle_v1"
            if len(family_reference) > 1
            else "flashwam_same_state_aa_oracle_v3"
        ),
        "clone_method": manifest["clone_method"],
        "task": manifest.get("task"),
        "seed": int(manifest["seed"]),
        "prompt": manifest.get("prompt"),
        "intervention_frame": int(manifest["intervention_frame"]),
        "repeats": int(manifest["repeats"]),
        "symmetric_restore": False,
        "parent_setup_snapshot": manifest.get("parent_setup_snapshot"),
        "parent_setup_snapshot_sha256": (
            simulator_state_sha256(
                torch.load(
                    manifest["parent_setup_snapshot"],
                    map_location="cpu",
                    weights_only=False,
                )
            )
            if manifest.get("parent_setup_snapshot")
            else None
        ),
        "prefix_reconstruction_exact": (
            len(set(prefix_values)) == 1 and prefix_canonical_exact
        ),
        "common_causal_inputs_exact": common_exact,
        "within_family_observable_exact": family_observable_exact,
        "start_full_state_exact": start_full,
        "start_core_state_exact": start_full,
        "within_family_end_full_state_exact": end_full,
        "within_family_end_core_state_exact": end_full,
        "strict_causal_contract_go": bool(
            len(set(prefix_values)) == 1
            and prefix_canonical_exact
            and common_exact
            and family_observable_exact
            and start_full
            and end_full
        ),
        "near_strict_causal_contract_go": bool(
            len(set(prefix_values)) == 1
            and prefix_canonical_exact
            and common_exact
            and family_observable_exact
            and start_full
            and end_full
        ),
        "fork_parent_cuda_rng_inherited": bool(
            manifest.get("parent_cuda_rng_inherited")
        ),
        "fork_child_cuda_rng_api": "disabled; inherited at fork",
        "branches": [
            {
                "repeat": int(record["repeat"]),
                "branch": record["branch"],
                "start_simulator_sha256": simulator_state_sha256(
                    record["start_snapshot"]
                ),
                "start_observation_sha256": record["start_observation_sha256"],
                "prefix_runtime_sha256": record["prefix_runtime_sha256"],
                "pre_action_runtime_sha256": record.get(
                    "pre_action_runtime_sha256"
                ),
                "student_action_sha256": record["student_action_sha256"],
                "action_family": record["action_family"],
                "executed_action_sha256": record["executed_action_sha256"],
                "source_student_action_exact": record["source_student_exact"],
                "planner_mode": "fork_child_full_prefix",
                "end_simulator_sha256": simulator_state_sha256(
                    record["end_snapshot"]
                ),
                "end_observation_sha256": record["end_observation_sha256"],
                "progress": None,
            }
            for record in records
        ],
    }
    oracle_path.parent.mkdir(parents=True, exist_ok=True)
    parent_setup = None
    if manifest.get("parent_setup_snapshot"):
        parent_setup = torch.load(
            manifest["parent_setup_snapshot"],
            map_location="cpu",
            weights_only=False,
        )
    torch.save(
        {**result, "records": records, "parent_setup": parent_setup},
        oracle_path.with_suffix(".pt"),
    )
    oracle_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
