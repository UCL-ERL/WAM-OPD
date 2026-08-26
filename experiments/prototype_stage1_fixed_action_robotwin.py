"""PROTOTYPE — execute one fixed action chunk in a reproducible RoboTwin state.

The evaluator supports three operations:

1. capture the initial observation for a fixed task seed;
2. execute a saved student or Teacher-Bridge action while recording Curobo
   joint paths;
3. execute the same action while replaying those joint paths.

Planner replay is required because Curobo 0.7.8 can return different
interpolated paths for identical start qpos and end-effector targets.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml


DEFAULT_PROJECT_ROOT = Path(
    os.environ.get(
        "WAVE_RL_ROOT",
        os.environ.get(
            "PROJECT_ROOT", str(Path(__file__).resolve().parents[2] / "wave-rl")
        ),
    )
).expanduser().resolve()
DEFAULT_PROMPT = "Activate the the switch for electrical control using the left arm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--task", default="turn_switch")
    parser.add_argument("--task-config", default="demo_randomized")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--capture-observation", type=Path)
    parser.add_argument("--capture-context", type=Path)
    parser.add_argument("--prefix-artifact", type=Path)
    parser.add_argument("--prefix-key", default="prefix_env_actions")
    parser.add_argument(
        "--prefix-chunks",
        type=int,
        help="Use only the first N chunks from --prefix-artifact.",
    )
    parser.add_argument(
        "--suffix-start-chunk",
        type=int,
        help=(
            "After the candidate branch, execute native prefix-artifact chunks "
            "starting at this zero-based index."
        ),
    )
    parser.add_argument(
        "--prefix-planner-trace-mode",
        choices=("none", "record", "replay"),
        default="none",
    )
    parser.add_argument("--prefix-planner-trace-dir", type=Path)
    parser.add_argument("--action-artifact", type=Path)
    parser.add_argument("--action-key")
    parser.add_argument(
        "--planner-trace-mode",
        choices=("none", "record", "replay"),
        default="none",
    )
    parser.add_argument("--planner-trace-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def install_enhanced_determinism() -> None:
    import envs._base_task as base_task

    original_scene_config = base_task.sapien.SceneConfig

    def enhanced_scene_config(*args, **kwargs):
        config = original_scene_config(*args, **kwargs)
        config.enable_enhanced_determinism = True
        return config

    base_task.sapien.SceneConfig = enhanced_scene_config


def build_task_args(robotwin_root: Path, task: str, config_name: str) -> dict:
    from envs import CONFIGS_PATH

    with (robotwin_root / "task_config" / f"{config_name}.yml").open(
        "r", encoding="utf-8"
    ) as handle:
        args = yaml.safe_load(handle)
    args["task_name"] = task
    args["task_config"] = config_name
    args["eval_mode"] = True
    args["eval_video_log"] = False

    with Path(CONFIGS_PATH, "_embodiment_config.yml").open(
        "r", encoding="utf-8"
    ) as handle:
        embodiment_types = yaml.safe_load(handle)
    embodiment_type = args["embodiment"]
    if len(embodiment_type) != 1:
        raise NotImplementedError("prototype currently covers one dual-arm embodiment")
    robot_file = Path(embodiment_types[embodiment_type[0]]["file_path"])
    with (robot_file / "config.yml").open("r", encoding="utf-8") as handle:
        embodiment_config = yaml.safe_load(handle)
    args["left_robot_file"] = str(robot_file)
    args["right_robot_file"] = str(robot_file)
    args["left_embodiment_config"] = embodiment_config
    args["right_embodiment_config"] = embodiment_config
    args["dual_arm_embodied"] = True

    with Path(CONFIGS_PATH, "_camera_config.yml").open(
        "r", encoding="utf-8"
    ) as handle:
        camera_config = yaml.safe_load(handle)
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]
    return args


def install_planner_trace(
    task_env,
    *,
    prefix_mode: str,
    prefix_trace_dir: Path | None,
    prefix_calls_per_arm: int,
    branch_mode: str,
    branch_trace_dir: Path | None,
) -> None:
    for mode, trace_dir, phase in (
        (prefix_mode, prefix_trace_dir, "prefix"),
        (branch_mode, branch_trace_dir, "branch"),
    ):
        if mode != "none" and trace_dir is None:
            raise ValueError(f"{phase} planner trace directory is required")
        if mode == "record":
            trace_dir.mkdir(parents=True, exist_ok=True)
    counters = {"left": 0, "right": 0}

    def wrap(arm: str, original):
        def traced(target_pose, *args, **kwargs):
            absolute_index = counters[arm]
            counters[arm] += 1
            if absolute_index < prefix_calls_per_arm:
                mode = prefix_mode
                trace_dir = prefix_trace_dir
                trace_index = absolute_index
            else:
                mode = branch_mode
                trace_dir = branch_trace_dir
                trace_index = absolute_index - prefix_calls_per_arm
            if mode == "none":
                return original(target_pose, *args, **kwargs)

            entity = (
                task_env.robot.left_entity
                if arm == "left"
                else task_env.robot.right_entity
            )
            start_qpos = np.asarray(entity.get_qpos()).copy()
            path = trace_dir / f"{arm}_{trace_index:04d}.npz"
            if mode == "replay":
                saved = np.load(path)
                saved_start = saved["start_qpos"]
                saved_target = saved["target_pose"]
                if not np.array_equal(start_qpos, saved_start):
                    raise AssertionError(
                        f"{arm} call {trace_index} start qpos differs; "
                        f"max_abs={np.max(np.abs(start_qpos - saved_start))}"
                    )
                target_array = np.asarray(target_pose)
                if not np.array_equal(target_array, saved_target):
                    raise AssertionError(
                        f"{arm} call {trace_index} target differs; "
                        f"max_abs={np.max(np.abs(target_array - saved_target))}"
                    )
                return {
                    "status": str(saved["status"].item()),
                    "position": saved["position"],
                    "velocity": saved["velocity"],
                }

            result = original(target_pose, *args, **kwargs)
            if mode == "record":
                np.savez(
                    path,
                    start_qpos=start_qpos,
                    target_pose=np.asarray(target_pose).copy(),
                    status=np.asarray(str(result["status"])),
                    position=np.asarray(result.get("position", [])),
                    velocity=np.asarray(result.get("velocity", [])),
                )
            return result

        return traced

    task_env.robot.left_plan_path = wrap(
        "left", task_env.robot.left_plan_path
    )
    task_env.robot.right_plan_path = wrap(
        "right", task_env.robot.right_plan_path
    )


def switch_progress(task_env) -> dict[str, float | bool]:
    qpos = float(task_env.switch.get_qpos()[0])
    limits = np.asarray(task_env.switch.get_qlimits()[0], dtype=np.float64)
    normalized = (qpos - limits[0]) / max(limits[1] - limits[0], 1e-12)
    switch_position = np.asarray(task_env.switch.get_pose().p, dtype=np.float64)
    left_position = np.asarray(task_env.get_arm_pose("left")[:3], dtype=np.float64)
    right_position = np.asarray(task_env.get_arm_pose("right")[:3], dtype=np.float64)
    return {
        "switch_qpos": qpos,
        "switch_lower_limit": float(limits[0]),
        "switch_upper_limit": float(limits[1]),
        "switch_normalized_progress": float(normalized),
        "left_eef_to_switch_origin": float(
            np.linalg.norm(left_position - switch_position)
        ),
        "right_eef_to_switch_origin": float(
            np.linalg.norm(right_position - switch_position)
        ),
        "success": bool(task_env.check_success()),
    }


def execute_action_chunk(
    task_env,
    action: np.ndarray,
    initial_eef_pose: np.ndarray,
    add_init_pose_fn,
    *,
    first_chunk: bool,
) -> tuple[int, list[dict]]:
    if action.shape != (16, 2, 16):
        raise ValueError(f"expected action shape (16, 2, 16), got {action.shape}")
    observations: list[dict] = []
    executed_actions = 0
    start_frame = 1 if first_chunk else 0
    for frame_index in range(start_frame, action.shape[1]):
        for action_index in range(action.shape[2]):
            ee_action = add_init_pose_fn(
                action[:, frame_index, action_index],
                initial_eef_pose,
            )
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
            executed_actions += 1
            if (action_index + 1) % 4 == 0:
                observations.append(task_env.get_obs())
            if task_env.eval_success:
                return executed_actions, observations
    return executed_actions, observations


def main() -> None:
    args = parse_args()
    for path_arg in (
        "capture_observation",
        "capture_context",
        "prefix_artifact",
        "prefix_planner_trace_dir",
        "action_artifact",
        "planner_trace_dir",
        "output",
    ):
        path = getattr(args, path_arg)
        if path is not None:
            setattr(args, path_arg, path.expanduser().resolve())
    project_root = args.project_root.expanduser().resolve()
    robotwin_root = project_root / "third_party" / "RoboTwin-lingbot-native"
    lingbot_root = project_root / "third_party" / "lingbot-va"
    sys.path[:0] = [str(robotwin_root), str(lingbot_root)]
    os.environ["ROBOTWIN_ROOT"] = str(robotwin_root)
    os.chdir(robotwin_root)

    from evaluation.robotwin.eval_polict_client_openpi import (
        add_init_pose,
        class_decorator,
        format_obs,
    )

    install_enhanced_determinism()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    task_args = build_task_args(
        robotwin_root, args.task, args.task_config
    )
    task_env = class_decorator(args.task)
    task_env.setup_demo(
        now_ep_num=0,
        seed=args.seed,
        is_test=True,
        **task_args,
    )
    task_env.set_instruction(instruction=args.prompt)
    try:
        initial_obs = task_env.get_obs()
        formatted_obs = format_obs(initial_obs, args.prompt)
        if args.capture_observation:
            capture_path = args.capture_observation
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save([formatted_obs], capture_path)

        initial_eef_pose = np.asarray(
            initial_obs["endpose"]["left_endpose"]
            + [initial_obs["endpose"]["left_gripper"]]
            + initial_obs["endpose"]["right_endpose"]
            + [initial_obs["endpose"]["right_gripper"]],
            dtype=np.float64,
        )
        result: dict[str, object] = {
            "task": args.task,
            "task_config": args.task_config,
            "seed": args.seed,
            "prompt": args.prompt,
            "initial": switch_progress(task_env),
        }
        all_native_actions: list[np.ndarray] = []
        prefix_actions: list[np.ndarray] = []
        if args.prefix_artifact:
            prefix_artifact = torch.load(
                args.prefix_artifact,
                map_location="cpu",
                weights_only=False,
            )
            all_native_actions = [
                np.asarray(chunk)
                for chunk in prefix_artifact[args.prefix_key]
            ]
            prefix_actions = all_native_actions
            if args.prefix_chunks is not None:
                prefix_actions = prefix_actions[: args.prefix_chunks]

        prefix_calls_per_arm = sum(
            16 if index == 0 else 32
            for index in range(len(prefix_actions))
        )
        if prefix_actions or args.action_artifact:
            install_planner_trace(
                task_env,
                prefix_mode=args.prefix_planner_trace_mode,
                prefix_trace_dir=args.prefix_planner_trace_dir,
                prefix_calls_per_arm=prefix_calls_per_arm,
                branch_mode=args.planner_trace_mode,
                branch_trace_dir=args.planner_trace_dir,
            )

        context_chunks: list[dict[str, object]] = []
        progress_timeline: list[dict[str, object]] = []
        total_executed_actions = 0
        for chunk_index, prefix_action in enumerate(prefix_actions):
            executed, raw_observations = execute_action_chunk(
                task_env,
                prefix_action,
                initial_eef_pose,
                add_init_pose,
                first_chunk=chunk_index == 0,
            )
            formatted_observations = [
                format_obs(observation, args.prompt)
                for observation in raw_observations
            ]
            context_chunks.append(
                {
                    "frame_st_id": chunk_index * 2,
                    "observations": formatted_observations,
                    "env_action": prefix_action,
                }
            )
            total_executed_actions += executed
            progress_timeline.append(
                {
                    "phase": "prefix",
                    "chunk_index": chunk_index,
                    "executed_actions": total_executed_actions,
                    **switch_progress(task_env),
                }
            )
            if task_env.eval_success:
                break

        if args.capture_context:
            args.capture_context.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "task": args.task,
                    "task_config": args.task_config,
                    "seed": args.seed,
                    "prompt": args.prompt,
                    "initial_observation": formatted_obs,
                    "chunks": context_chunks,
                },
                args.capture_context,
            )

        result.update(
            {
                "prefix_artifact": (
                    str(args.prefix_artifact) if args.prefix_artifact else None
                ),
                "prefix_chunks": len(prefix_actions),
                "prefix_planner_trace_mode": args.prefix_planner_trace_mode,
                "prefix_executed_actions": total_executed_actions,
                "prefix_final": switch_progress(task_env),
            }
        )
        if args.action_artifact:
            if not args.action_key:
                raise ValueError("--action-key is required with --action-artifact")
            artifact = torch.load(
                args.action_artifact,
                map_location="cpu",
                weights_only=False,
            )
            action = np.asarray(artifact[args.action_key])
            if action.shape[0] != 16:
                raise ValueError(f"expected 16 action channels, got {action.shape}")
            executed_actions, _ = execute_action_chunk(
                task_env,
                action,
                initial_eef_pose,
                add_init_pose,
                first_chunk=not prefix_actions,
            )
            total_executed_actions += executed_actions
            progress_timeline.append(
                {
                    "phase": "branch",
                    "executed_actions": total_executed_actions,
                    **switch_progress(task_env),
                }
            )
            suffix_executed_actions = 0
            if args.suffix_start_chunk is not None:
                if not all_native_actions:
                    raise ValueError(
                        "--suffix-start-chunk requires --prefix-artifact"
                    )
                for suffix_index in range(
                    args.suffix_start_chunk,
                    len(all_native_actions),
                ):
                    executed, _ = execute_action_chunk(
                        task_env,
                        all_native_actions[suffix_index],
                        initial_eef_pose,
                        add_init_pose,
                        first_chunk=False,
                    )
                    suffix_executed_actions += executed
                    total_executed_actions += executed
                    progress_timeline.append(
                        {
                            "phase": "suffix",
                            "chunk_index": suffix_index,
                            "executed_actions": total_executed_actions,
                            **switch_progress(task_env),
                        }
                    )
                    if task_env.eval_success:
                        break
            result.update(
                {
                    "action_artifact": str(args.action_artifact),
                    "action_key": args.action_key,
                    "planner_trace_mode": args.planner_trace_mode,
                    "executed_actions": executed_actions,
                    "suffix_start_chunk": args.suffix_start_chunk,
                    "suffix_executed_actions": suffix_executed_actions,
                    "total_executed_actions": total_executed_actions,
                    "final": switch_progress(task_env),
                }
            )
        result["progress_timeline"] = progress_timeline

        rendered = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            output = args.output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    finally:
        task_env.close_env()


if __name__ == "__main__":
    main()
