"""PROTOTYPE — Teacher Bridge on a real RoboTwin observation.

Question
--------
Does the non-degenerate action bridge observed on synthetic states persist when
the released 1v/1a Flash student itself generates the plan and one-step action
chain from a real three-camera RoboTwin observation and task prompt?

The observation was saved by the existing LingBot-VA evaluation pipeline.  This
script does not step the simulator or train either model.

One-command run
---------------
CUDA_VISIBLE_DEVICES=0 WAVE_RL_ROOT=/path/to/wave-rl \
  /path/to/lingbot-python \
  experiments/prototype_real_obs_action_teacher_bridge.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from einops import rearrange

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(
    os.environ.get(
        "WAVE_RL_ROOT",
        os.environ.get("PROJECT_ROOT", str(WORKSPACE_ROOT.parent / "wave-rl")),
    )
).expanduser().resolve()
for path in (
    WORKSPACE_ROOT,
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "third_party" / "lingbot-va",
):
    sys.path.insert(0, str(path))

from experiments.goal1_exact_condition import (
    GOAL1_PRODUCTION_FIELD_POLICY,
    GOAL1_PRODUCTION_SCHEMA_VERSION,
    ConditionContractError,
    assert_cache_semantics,
    assert_fingerprint_match,
    build_canonical_context,
    build_condition_fingerprint,
    cache_valid_length,
    capture_prepared_plan,
    prepare_plan_input,
    sequence_hash,
    stable_hash,
    tensor_hash,
    tensor_diff,
)
from experiments.opd_task_specs import require_training_task_config

DEFAULT_ARTIFACT_ROOT = Path(
    os.environ.get("WAM_OPD_ARTIFACT_ROOT", WORKSPACE_ROOT / ".artifacts")
).expanduser().resolve()
DEFAULT_STUDENT = Path(
    os.environ.get(
        "WAM_OPD_STUDENT_ROOT",
        DEFAULT_ARTIFACT_ROOT / "models" / "FlashWAM-RoboTwin",
    )
).expanduser().resolve()
DEFAULT_TEACHER_TRANSFORMER = Path(
    os.environ.get(
        "WAM_OPD_TEACHER_ROOT",
        DEFAULT_ARTIFACT_ROOT / "models" / "lingbot-va-posttrain-robotwin",
    )
).expanduser().resolve() / "transformer"
DEFAULT_OBSERVATION = Path(
    os.environ.get(
        "WAM_OPD_OBSERVATION",
        DEFAULT_ARTIFACT_ROOT / "inputs" / "saved_observation.pt",
    )
).expanduser().resolve()
DEFAULT_PROMPT = (
    "Grab red block, position it on the left. Next, grab green block to place "
    "beside red block, then grab blue block and set it to the right of green block."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--student-delta", type=Path)
    parser.add_argument(
        "--teacher-transformer",
        type=Path,
        default=DEFAULT_TEACHER_TRANSFORMER,
    )
    parser.add_argument("--observation", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--replay-context", type=Path)
    parser.add_argument(
        "--context-chunks",
        type=int,
        help="Use only the first N executed chunks from --replay-context.",
    )
    parser.add_argument(
        "--prompt",
        help=(
            "Prompt override. With --replay-context this defaults to the "
            "captured current-Student prompt and an explicit mismatch fails."
        ),
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--teacher-noise-seed",
        type=int,
        help=(
            "Independent seed for the Teacher-on-Teacher video/action noise. "
            "If omitted, a deterministic offset from --seed is used."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--student-action-steps", type=int, default=1)
    parser.add_argument("--stage-g-collection-id")
    parser.add_argument("--stage-g-round-index", type=int, default=0)
    parser.add_argument("--stage-m-live-context-id")
    parser.add_argument("--save-actions", type=Path)
    parser.add_argument("--runtime-state-audit-output", type=Path)
    return parser.parse_args()


def resolve_prompt(
    requested_prompt: str | None,
    replay_context: dict[str, object] | None,
) -> str:
    if replay_context is None:
        return requested_prompt if requested_prompt is not None else DEFAULT_PROMPT
    captured_prompt = str(replay_context["prompt"])
    if requested_prompt is not None and requested_prompt != captured_prompt:
        raise ValueError("replay context prompt does not match --prompt")
    return captured_prompt


def validate_context_policy_delta(
    replay_context: dict[str, object],
    requested_delta: Path | None,
) -> None:
    captured = replay_context.get("policy_delta_path")
    captured_path = Path(str(captured)).resolve() if captured else None
    requested_path = requested_delta.resolve() if requested_delta is not None else None
    if captured_path != requested_path:
        raise ValueError(
            "replay context policy delta does not match --student-delta: "
            f"captured={captured_path} requested={requested_path}"
        )


def synchronize_replayed_frame_position(server: object, frame_st_id: int) -> None:
    """Keep audit/runtime metadata aligned with a manually replayed KV prefix."""

    server.frame_st_id = int(frame_st_id)


def euler_endpoint(
    state: torch.Tensor,
    velocity: torch.Tensor,
    sigma_start: float,
    sigma_end: float,
) -> torch.Tensor:
    return state + (sigma_end - sigma_start) * velocity


def masked_metrics(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    left_valid = left.float()[mask]
    right_valid = right.float()[mask]
    delta = left_valid - right_valid
    return {
        "rmse": delta.square().mean().sqrt().item(),
        "relative_l2": (
            delta.norm() / right_valid.norm().clamp_min(1e-12)
        ).item(),
        "cosine": torch.nn.functional.cosine_similarity(
            left_valid[None],
            right_valid[None],
        ).item(),
    }


def main() -> None:
    args = parse_args()
    if args.replay_context is None and not args.observation.is_file():
        raise FileNotFoundError(args.observation)
    managed_collection = bool(
        args.stage_g_collection_id is not None
        or args.stage_m_live_context_id is not None
    )
    if args.student_delta is not None and not managed_collection:
        raise ValueError(
            "--student-delta requires a managed Stage G/M collection so policy "
            "provenance cannot silently fall back to schema v1"
        )
    if (
        args.stage_g_collection_id is not None
        and args.stage_m_live_context_id is not None
    ):
        raise ValueError("Stage G and Stage M collection modes are exclusive")
    if args.stage_g_collection_id is not None and args.replay_context is None:
        raise ValueError("Stage G labels require --replay-context")
    if args.stage_m_live_context_id is not None and args.replay_context is None:
        raise ValueError("Stage M live labels require --replay-context")
    if args.stage_g_round_index < 0:
        raise ValueError("--stage-g-round-index must be non-negative")

    replay_context_payload = (
        torch.load(
            args.replay_context,
            map_location="cpu",
            weights_only=False,
        )
        if args.replay_context is not None
        else None
    )
    if replay_context_payload is not None:
        require_training_task_config(
            str(replay_context_payload.get("task_config", "unknown"))
        )
    if args.stage_m_live_context_id is not None:
        from experiments.stage_m_live_bridge_contract import validate_live_context

        validate_live_context(replay_context_payload)
        if (
            replay_context_payload["live_context_id"]
            != args.stage_m_live_context_id
        ):
            raise ValueError(
                "Stage M live context id mismatch: "
                f"expected={args.stage_m_live_context_id!r} "
                f"actual={replay_context_payload['live_context_id']!r}"
            )
    prompt = resolve_prompt(args.prompt, replay_context_payload)
    if replay_context_payload is not None:
        validate_context_policy_delta(replay_context_payload, args.student_delta)

    from wave_rl.adapters.lingbot_va.inference_wrapper import (
        LingBotVANativeWrapper,
        LingBotVAWrapperConfig,
    )
    from wan_va.modules.model import WanTransformer3DModel
    from wan_va.utils import data_seq_to_patch

    wrapper = LingBotVANativeWrapper(
        LingBotVAWrapperConfig(
            benchmark="robotwin",
            checkpoint_path=args.student,
            save_root=Path("/tmp/flashwam_real_obs_bridge"),
            local_rank=0,
            video_num_inference_steps=1,
            action_num_inference_steps=args.student_action_steps,
        )
    )
    if args.student_delta is not None:
        from experiments.prototype_flashwam_robotwin_server import (
            apply_transformer_delta,
        )

        delta = torch.load(
            args.student_delta,
            map_location="cpu",
            weights_only=False,
        )
        apply_transformer_delta(wrapper.server.transformer, delta)
    wrapper.reset(prompt)
    server = wrapper.server
    device = torch.device(args.device)
    dtype = server.dtype
    frame_chunk_size = server.job_config.frame_chunk_size

    torch.manual_seed(args.seed)
    context_inputs: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    task_id = "unknown"
    task_config = "unknown"
    env_seed = -1
    context_chunks_used = 0
    if args.replay_context:
        replay_context = replay_context_payload
        if replay_context is None:
            raise AssertionError("replay context was not loaded")
        context_chunks = replay_context["chunks"]
        if args.context_chunks is not None:
            if args.context_chunks > len(context_chunks):
                raise ValueError(
                    "requested --context-chunks exceeds captured current-Student "
                    f"history: requested={args.context_chunks} available={len(context_chunks)}"
                )
            context_chunks = context_chunks[: args.context_chunks]
        task_id = str(replay_context.get("task", task_id))
        task_config = str(replay_context.get("task_config", task_config))
        env_seed = int(replay_context.get("seed", env_seed))
        context_chunks_used = len(context_chunks)
        init_latent = server._encode_obs(
            {"obs": replay_context["initial_observation"]}
        )
        server.init_latent = init_latent
        frame_st_id = 0
        with torch.inference_mode():
            for chunk in context_chunks:
                if chunk["frame_st_id"] != frame_st_id:
                    raise ValueError(
                        "non-contiguous replay context: "
                        f"expected frame_st_id={frame_st_id}, "
                        f"got {chunk['frame_st_id']}"
                    )
                latent_model_input = server._encode_obs(
                    {"obs": chunk["observations"]}
                )
                if frame_st_id == 0:
                    latent_model_input = torch.cat(
                        [init_latent, latent_model_input],
                        dim=2,
                    )
                action_model_input = server.preprocess_action(
                    chunk["env_action"]
                ).to(latent_model_input)
                model_input = server._prepare_latent_input(
                    latent_model_input,
                    action_model_input,
                    frame_st_id=frame_st_id,
                )
                server.transformer(
                    server._repeat_input_for_cfg(model_input["latent_res_lst"]),
                    update_cache=2,
                    cache_name=server.cache_name,
                    action_mode=False,
                )
                server.transformer(
                    server._repeat_input_for_cfg(model_input["action_res_lst"]),
                    update_cache=2,
                    cache_name=server.cache_name,
                    action_mode=True,
                )
                context_inputs.append(
                    (
                        frame_st_id,
                        latent_model_input.detach().clone(),
                        action_model_input.detach().clone(),
                    )
                )
                frame_st_id += latent_model_input.shape[2]
        synchronize_replayed_frame_position(server, frame_st_id)
        observation_history_length = sum(
            len(chunk["observations"]) for chunk in context_chunks
        )
        observation_description = str(args.replay_context)
    else:
        observation_history = torch.load(
            args.observation,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(observation_history, list):
            raise TypeError(
                "expected the saved native observation history to be a list"
            )
        # ``obs_data_*.pt`` contains a history list.  At reset the native
        # inference path encodes one current observation.
        current_observation = observation_history[-1]
        init_latent = server._encode_obs({"obs": current_observation})
        frame_st_id = 0
        observation_history_length = len(observation_history)
        observation_description = str(args.observation)

    if args.runtime_state_audit_output is not None:
        from experiments.flashwam_runtime_state_audit import (
            runtime_state_component_sha256,
            runtime_state_sha256,
        )

        args.runtime_state_audit_output.parent.mkdir(
            parents=True, exist_ok=True
        )
        args.runtime_state_audit_output.write_text(
            json.dumps(
                {
                    "schema": "flashwam_offline_prefix_runtime_audit_v1",
                    "frame_st_id": frame_st_id,
                    "runtime_state_sha256": runtime_state_sha256(server),
                    "component_sha256": runtime_state_component_sha256(
                        server
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    server.init_latent = init_latent
    latents = torch.randn(
        1,
        48,
        frame_chunk_size,
        server.latent_height,
        server.latent_width,
        device=device,
        dtype=dtype,
    )
    initial_video_noise = latents.detach().clone()
    actions = torch.randn(
        1,
        server.job_config.action_dim,
        frame_chunk_size,
        server.action_per_frame,
        1,
        device=device,
        dtype=dtype,
    )

    # Cross-stream protocol noise is model-family scoped: SS/ST share the
    # Student noise, while TS/TT share a separate Teacher noise.  The old
    # prototype silently reused ``initial_video_noise`` and ``actions`` for
    # Teacher-on-Teacher control, making a nominal TT result confounded with
    # Student noise.  Persist and use these actual tensors below.
    teacher_noise_seed = (
        int(args.teacher_noise_seed)
        if args.teacher_noise_seed is not None
        else int(args.seed) + 1_000_003
    )
    teacher_noise_generator = torch.Generator(device=device)
    teacher_noise_generator.manual_seed(teacher_noise_seed)
    teacher_video_noise = torch.randn(
        1,
        48,
        frame_chunk_size,
        server.latent_height,
        server.latent_width,
        device=device,
        dtype=dtype,
        generator=teacher_noise_generator,
    )
    teacher_action_base_noise = torch.randn(
        1,
        server.job_config.action_dim,
        frame_chunk_size,
        server.action_per_frame,
        1,
        device=device,
        dtype=dtype,
        generator=teacher_noise_generator,
    )
    server.scheduler.set_timesteps(1)
    server.action_scheduler.set_timesteps(args.student_action_steps)
    video_timesteps = F.pad(server.scheduler.timesteps, (0, 1), value=0)
    action_timesteps = F.pad(server.action_scheduler.timesteps, (0, 1), value=0)

    with torch.inference_mode():
        student_prepared_plan = None
        student_plan_timestep = None
        student_plan_latent_cond = None
        for index, timestep in enumerate(video_timesteps):
            last_step = index == len(video_timesteps) - 1
            latent_cond = (
                init_latent[:, :, 0:1].to(dtype)
                if frame_st_id == 0
                else None
            )
            model_input = server._prepare_latent_input(
                latents,
                None,
                timestep,
                timestep,
                latent_cond,
                None,
                frame_st_id=frame_st_id,
            )
            if last_step:
                # This is the tensor actually supplied to the Student video
                # branch.  ``student_plan`` alone is not sufficient evidence:
                # the native helper mutates the first slice when a condition
                # is supplied.
                student_prepared_plan = model_input["latent_res_lst"][
                    "noisy_latents"
                ].detach().clone()
                student_plan_timestep = model_input["latent_res_lst"][
                    "timesteps"
                ].detach().clone()
                student_plan_latent_cond = (
                    latent_cond.detach().clone()
                    if latent_cond is not None
                    else None
                )
            velocity = server.transformer(
                server._repeat_input_for_cfg(model_input["latent_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=server.cache_name,
                action_mode=False,
            )
            if not last_step:
                velocity = data_seq_to_patch(
                    server.job_config.patch_size,
                    velocity,
                    frame_chunk_size,
                    server.latent_height,
                    server.latent_width,
                    batch_size=2 if server.use_cfg else 1,
                )
                velocity = velocity[1:] + server.job_config.guidance_scale * (
                    velocity[:1] - velocity[1:]
                )
                latents = server.scheduler.step(velocity, timestep, latents)
            if latent_cond is not None:
                latents[:, :, 0:1] = latent_cond

        student_plan = latents.detach().clone()
        if student_prepared_plan is None or student_plan_timestep is None:
            raise ConditionContractError("Student produced no final prepared plan")
        prepared_plan_capture = capture_prepared_plan(
            student_plan,
            student_prepared_plan,
            student_plan_timestep,
            frame_st_id=frame_st_id,
            latent_cond=student_plan_latent_cond,
        )
        student_cache_valid_length = cache_valid_length(
            server.transformer, server.cache_name
        )
        action_cond = (
            torch.zeros(
                1,
                server.job_config.action_dim,
                1,
                server.action_per_frame,
                1,
                device=device,
                dtype=dtype,
            )
            if frame_st_id == 0
            else None
        )
        if action_cond is not None:
            actions[:, :, 0:1] = action_cond
            teacher_action_base_noise[:, :, 0:1] = action_cond
        teacher_action_base_noise[:, ~server.action_mask] *= 0
        # The native helper applies this mask in place when the action branch
        # is prepared.  Apply it before recording epsilon_a as well, so the
        # canonical noise tensor is exactly what the branch sees, including
        # inactive channels.
        actions[:, ~server.action_mask] *= 0
        student_chain = [actions.detach().clone()]
        student_action_branch_input = None
        for index, timestep in enumerate(action_timesteps):
            last_step = index == len(action_timesteps) - 1
            model_input = server._prepare_latent_input(
                None,
                actions,
                timestep,
                timestep,
                None,
                action_cond,
                frame_st_id=frame_st_id,
            )
            if student_action_branch_input is None:
                student_action_branch_input = {
                    key: value.detach().clone()
                    for key, value in model_input["action_res_lst"].items()
                    if isinstance(value, torch.Tensor)
                }
            velocity = server.transformer(
                server._repeat_input_for_cfg(model_input["action_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=server.cache_name,
                action_mode=True,
            )
            if not last_step:
                velocity = rearrange(
                    velocity,
                    "b (f n) c -> b c f n 1",
                    f=frame_chunk_size,
                )[:1]
                actions = server.action_scheduler.step(
                    velocity,
                    timestep,
                    actions,
                )
                if action_cond is not None:
                    actions[:, :, 0:1] = action_cond
                student_chain.append(actions.detach().clone())

    teacher = WanTransformer3DModel.from_pretrained(
        args.teacher_transformer,
        torch_dtype=dtype,
        attn_mode="torch",
    ).to(device)
    teacher.eval()
    def create_model_cache(model: object, cache_name: str) -> None:
        model.create_empty_cache(
            cache_name,
            server.job_config.attn_window,
            (
                frame_chunk_size
                * server.latent_height
                * server.latent_width
                // 4
            ),
            frame_chunk_size * server.action_per_frame,
            device=device,
            dtype=dtype,
            batch_size=2 if server.use_cfg else 1,
        )
        with torch.inference_mode():
            for context_frame_st_id, latent_input, action_input in context_inputs:
                history_input = server._prepare_latent_input(
                    latent_input,
                    action_input,
                    frame_st_id=context_frame_st_id,
                )
                model(
                    server._repeat_input_for_cfg(
                        history_input["latent_res_lst"]
                    ),
                    update_cache=2,
                    cache_name=cache_name,
                    action_mode=False,
                )
                model(
                    server._repeat_input_for_cfg(
                        history_input["action_res_lst"]
                    ),
                    update_cache=2,
                    cache_name=cache_name,
                    action_mode=True,
                )

    def create_teacher_cache(cache_name: str) -> None:
        create_model_cache(teacher, cache_name)

    teacher_cache = "real_obs_student_plan"
    create_teacher_cache(teacher_cache)
    # Teacher consumes the exact prepared tensor produced by the Student
    # deployment path.  In particular, this passes no episode initial latent
    # for a nonzero prefix and cannot silently rewrite the first plan slice.
    plan_input, teacher_plan_capture = prepare_plan_input(
        server,
        prepared_plan_capture.prepared_z_s,
        frame_st_id=frame_st_id,
        already_prepared=True,
        latent_t=0,
    )
    if not torch.equal(
        teacher_plan_capture.prepared_z_s,
        prepared_plan_capture.prepared_z_s,
    ):
        raise ConditionContractError("Teacher did not receive canonical prepared_z_s")
    with torch.inference_mode():
        teacher(
            server._repeat_input_for_cfg(plan_input["latent_res_lst"]),
            update_cache=1,
            cache_name=teacher_cache,
            action_mode=False,
        )
    teacher_cache_valid_length = cache_valid_length(teacher, teacher_cache)

    teacher_action_branch_input = None

    @torch.inference_mode()
    def teacher_action_velocity(
        state: torch.Tensor,
        sigma: float,
        cache_name: str = teacher_cache,
    ) -> torch.Tensor:
        nonlocal teacher_action_branch_input
        model_input = server._prepare_latent_input(
            None,
            state.clone(),
            0,
            sigma * 1000.0,
            None,
            action_cond,
            frame_st_id=frame_st_id,
        )
        if teacher_action_branch_input is None:
            teacher_action_branch_input = {
                key: value.detach().clone()
                for key, value in model_input["action_res_lst"].items()
                if isinstance(value, torch.Tensor)
            }
        output = teacher(
            server._repeat_input_for_cfg(model_input["action_res_lst"]),
            update_cache=0,
            cache_name=cache_name,
            action_mode=True,
        )
        velocity = rearrange(
            output,
            "b (f n) c -> b c f n 1",
            f=frame_chunk_size,
        )[:1]
        velocity[:, ~server.action_mask] = 0
        if action_cond is not None:
            velocity[:, :, 0:1] = 0
        return velocity

    loss_mask = server.action_mask.to(device)[None, :, None, None, None].expand_as(
        student_chain[0]
    )
    loss_mask = loss_mask.clone()
    if action_cond is not None:
        loss_mask[:, :, 0:1] = False

    teacher_boundaries = [1.0 - index / 50.0 for index in range(51)]
    macro_boundaries = tuple(
        float(timestep.item()) / 1000.0 for timestep in action_timesteps
    )
    interval_results: list[dict[str, object]] = []
    final_bridge_endpoint = None
    with torch.inference_mode():
        for macro_index, (sigma_start, sigma_end) in enumerate(
            zip(macro_boundaries[:-1], macro_boundaries[1:])
        ):
            student_state = student_chain[macro_index]
            student_endpoint = student_chain[macro_index + 1]
            direct_velocity = teacher_action_velocity(student_state, sigma_start)
            direct_endpoint = euler_endpoint(
                student_state,
                direct_velocity,
                sigma_start,
                sigma_end,
            )
            bridge_endpoint = student_state.clone()
            teacher_micro_start = round((1.0 - sigma_start) * 50)
            teacher_micro_end = round((1.0 - sigma_end) * 50)
            if (
                abs(teacher_boundaries[teacher_micro_start] - sigma_start) > 1e-6
                or abs(teacher_boundaries[teacher_micro_end] - sigma_end) > 1e-6
            ):
                raise ValueError(
                    "Student macro boundary does not align with the native "
                    "50-step teacher schedule"
                )
            for micro_index in range(teacher_micro_start, teacher_micro_end):
                micro_start = teacher_boundaries[micro_index]
                micro_end = teacher_boundaries[micro_index + 1]
                bridge_endpoint = euler_endpoint(
                    bridge_endpoint,
                    teacher_action_velocity(bridge_endpoint, micro_start),
                    micro_start,
                    micro_end,
                )
                if action_cond is not None:
                    bridge_endpoint[:, :, 0:1] = 0
            final_bridge_endpoint = bridge_endpoint
            interval_results.append(
                {
                    "macro_index": macro_index,
                    "sigma_start": sigma_start,
                    "sigma_end": sigma_end,
                    "student_vs_teacher_direct": masked_metrics(
                        student_endpoint,
                        direct_endpoint,
                        loss_mask,
                    ),
                    "student_vs_teacher_bridge": masked_metrics(
                        student_endpoint,
                        bridge_endpoint,
                        loss_mask,
                    ),
                    "teacher_direct_vs_bridge": masked_metrics(
                        direct_endpoint,
                        bridge_endpoint,
                        loss_mask,
                    ),
                }
            )

    if student_action_branch_input is None or teacher_action_branch_input is None:
        raise ConditionContractError("Student/Teacher action branches were not observed")

    tokenized_prompt = server.tokenizer(prompt, add_special_tokens=True)
    prompt_ids = (
        tokenized_prompt["input_ids"]
        if hasattr(tokenized_prompt, "__getitem__")
        and not isinstance(tokenized_prompt, (list, tuple))
        else tokenized_prompt
    )
    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids.reshape(-1).tolist()
    elif prompt_ids and isinstance(prompt_ids[0], (list, tuple)):
        prompt_ids = prompt_ids[0]
    prompt_ids = tuple(int(item) for item in prompt_ids)
    history_hash = sequence_hash(
        {
            "frame_st_id": frame_id,
            "latent": latent_input,
            "action": action_input,
        }
        for frame_id, latent_input, action_input in context_inputs
    )
    observation_hash = sequence_hash(
        {"frame_st_id": frame_id, "latent": latent_input}
        for frame_id, latent_input, _action_input in context_inputs
    )
    model_action_history_hash = sequence_hash(
        {"frame_st_id": frame_id, "action": action_input}
        for frame_id, _latent_input, action_input in context_inputs
    )
    normalization_metadata = {
        "method": str(server.action_norm_method),
        "actions_q01_hash": tensor_hash(server.actions_q01),
        "actions_q99_hash": tensor_hash(server.actions_q99),
        "action_mask_hash": tensor_hash(server.action_mask),
        "used_action_channel_ids": list(server.job_config.used_action_channel_ids),
    }
    student_fingerprint = build_condition_fingerprint(
        checkpoint_owner="student",
        history_hash=history_hash,
        prompt_hash=stable_hash(prompt_ids),
        observation_hash=observation_hash,
        model_action_history_hash=model_action_history_hash,
        prepared_plan=prepared_plan_capture.prepared_z_s,
        prepared_plan_timestep=prepared_plan_capture.prepared_z_s_timestep,
        action_base_noise=student_chain[0],
        action_timestep=student_action_branch_input["timesteps"],
        mask=loss_mask,
        normalization_metadata=normalization_metadata,
        frame_st_id=frame_st_id,
        token_positions=student_action_branch_input["grid_id"].reshape(-1).tolist(),
        cache_valid_length=student_cache_valid_length,
        sigma_start=macro_boundaries[0],
        sigma_end=macro_boundaries[-1],
    )
    teacher_fingerprint = build_condition_fingerprint(
        checkpoint_owner="teacher",
        history_hash=history_hash,
        prompt_hash=stable_hash(prompt_ids),
        observation_hash=observation_hash,
        model_action_history_hash=model_action_history_hash,
        prepared_plan=teacher_plan_capture.prepared_z_s,
        prepared_plan_timestep=teacher_plan_capture.prepared_z_s_timestep,
        action_base_noise=student_chain[0],
        action_timestep=teacher_action_branch_input["timesteps"],
        mask=loss_mask,
        normalization_metadata=normalization_metadata,
        frame_st_id=frame_st_id,
        token_positions=teacher_action_branch_input["grid_id"].reshape(-1).tolist(),
        cache_valid_length=teacher_cache_valid_length,
        sigma_start=macro_boundaries[0],
        sigma_end=macro_boundaries[-1],
    )
    # This is the labeler's fail-closed gate.  Owner and KV values are allowed
    # to differ; chronology and all semantic input fields are not.
    from experiments.goal1_exact_condition import assert_fingerprint_match

    assert_fingerprint_match(student_fingerprint, teacher_fingerprint)
    assert_cache_semantics(student_fingerprint, teacher_fingerprint)
    executed_physical_actions = []
    if replay_context_payload is not None:
        for chunk in replay_context_payload.get("chunks", ())[:context_chunks_used]:
            physical = torch.as_tensor(chunk["env_action"])
            executed_physical_actions.append(
                {
                    "frame_st_id": int(chunk["frame_st_id"]),
                    "shape": list(physical.shape),
                    "tensor_hash": tensor_hash(physical),
                }
            )
    canonical_context = build_canonical_context(
        behavior_checkpoint_id=str(args.student),
        task=task_id,
        seed=env_seed,
        episode_id=f"{task_id}:seed{env_seed}",
        prefix_id=f"c{context_chunks_used}",
        frame_st_id=frame_st_id,
        prompt_token_ids=prompt_ids,
        observation_history=observation_hash,
        model_format_action_history=model_action_history_hash,
        history_hash=history_hash,
        executed_physical_actions=executed_physical_actions,
        prepared_plan=prepared_plan_capture,
        action_base_noise=student_chain[0],
        valid_action_mask=loss_mask,
        sigma_start=macro_boundaries[0],
        sigma_end=macro_boundaries[-1],
        normalization_metadata=normalization_metadata,
    )
    canonical_context.student_fingerprint = student_fingerprint
    canonical_context.teacher_fingerprint = teacher_fingerprint
    canonical_context.action_token_positions = tuple(
        int(item) for item in student_fingerprint.token_positions
    )
    canonical_context.cache_valid_length = int(
        student_fingerprint.cache_valid_length
    )

    if final_bridge_endpoint is None:
        raise RuntimeError("Teacher Bridge produced no macro endpoint")

    # Teacher-on-teacher-plan control: reconstruct the same causal history in a
    # separate cache, solve the current video plan with the native 25-step
    # teacher scheduler, then solve the action interval with 50 teacher steps.
    teacher_plan_cache = "real_obs_teacher_plan"
    create_teacher_cache(teacher_plan_cache)
    teacher_plan = teacher_video_noise.clone()
    server.scheduler.set_timesteps(25)
    teacher_video_timesteps = F.pad(server.scheduler.timesteps, (0, 1), value=0)
    with torch.inference_mode():
        for index, timestep in enumerate(teacher_video_timesteps):
            last_step = index == len(teacher_video_timesteps) - 1
            latent_cond = (
                init_latent[:, :, 0:1].to(dtype)
                if frame_st_id == 0
                else None
            )
            model_input = server._prepare_latent_input(
                teacher_plan,
                None,
                timestep,
                timestep,
                latent_cond,
                None,
                frame_st_id=frame_st_id,
            )
            velocity = teacher(
                server._repeat_input_for_cfg(model_input["latent_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=teacher_plan_cache,
                action_mode=False,
            )
            if not last_step:
                velocity = data_seq_to_patch(
                    server.job_config.patch_size,
                    velocity,
                    frame_chunk_size,
                    server.latent_height,
                    server.latent_width,
                    batch_size=2 if server.use_cfg else 1,
                )
                velocity = velocity[1:] + server.job_config.guidance_scale * (
                    velocity[:1] - velocity[1:]
                )
                teacher_plan = server.scheduler.step(
                    velocity,
                    timestep,
                    teacher_plan,
                )
            if latent_cond is not None:
                teacher_plan[:, :, 0:1] = latent_cond

    teacher_plan_action = teacher_action_base_noise.clone()
    for micro_index in range(50):
        micro_start = teacher_boundaries[micro_index]
        micro_end = teacher_boundaries[micro_index + 1]
        teacher_plan_action = euler_endpoint(
            teacher_plan_action,
            teacher_action_velocity(
                teacher_plan_action,
                micro_start,
                cache_name=teacher_plan_cache,
            ),
            micro_start,
            micro_end,
        )
        if action_cond is not None:
            teacher_plan_action[:, :, 0:1] = 0

    # TS interface probe: rebuild the same semantic history in a separate
    # cache owned by the released Student transformer, inject the actual
    # Teacher-generated zT as an already-prepared video condition, and solve
    # the action branch with the Student 1-step schedule.  This is deliberately
    # separate from ``server.cache_name`` (Student-on-zS) and from both Teacher
    # caches.  No KV tensor is shared between model owners.
    ts_cache = "real_obs_student_teacher_plan"
    create_model_cache(server.transformer, ts_cache)
    ts_plan_input, ts_plan_capture = prepare_plan_input(
        server,
        teacher_plan,
        frame_st_id=frame_st_id,
        already_prepared=True,
        latent_t=0,
    )
    if not torch.equal(ts_plan_capture.prepared_z_s, teacher_plan):
        raise ConditionContractError(
            "TS Student cache preparation modified the canonical Teacher zT"
        )
    with torch.inference_mode():
        server.transformer(
            server._repeat_input_for_cfg(ts_plan_input["latent_res_lst"]),
            update_cache=1,
            cache_name=ts_cache,
            action_mode=False,
        )
    ts_cache_valid_length = cache_valid_length(server.transformer, ts_cache)
    ts_action = teacher_action_base_noise.clone()
    ts_student_action_branch_input = None
    ts_student_action_chain = [ts_action.detach().clone()]
    with torch.inference_mode():
        for index, timestep in enumerate(action_timesteps):
            last_step = index == len(action_timesteps) - 1
            model_input = server._prepare_latent_input(
                None,
                ts_action,
                timestep,
                timestep,
                None,
                action_cond,
                frame_st_id=frame_st_id,
            )
            if ts_student_action_branch_input is None:
                ts_student_action_branch_input = {
                    key: value.detach().clone()
                    for key, value in model_input["action_res_lst"].items()
                    if isinstance(value, torch.Tensor)
                }
            velocity = server.transformer(
                server._repeat_input_for_cfg(model_input["action_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=ts_cache,
                action_mode=True,
            )
            if not last_step:
                velocity = rearrange(
                    velocity,
                    "b (f n) c -> b c f n 1",
                    f=frame_chunk_size,
                )[:1]
                ts_action = server.action_scheduler.step(
                    velocity,
                    timestep,
                    ts_action,
                )
                if action_cond is not None:
                    ts_action[:, :, 0:1] = action_cond
                ts_student_action_chain.append(ts_action.detach().clone())
    if ts_student_action_branch_input is None:
        raise ConditionContractError("TS Student action branch was not observed")
    ts_student_model_action = ts_action.clone()
    ts_student_model_action[:, ~server.action_mask] = 0
    ts_student_env_action = server.postprocess_action(ts_student_model_action)
    teacher_teacher_plan_action_branch_input = None
    teacher_teacher_plan_action_probe = server._prepare_latent_input(
        None,
        teacher_plan_action.clone(),
        0,
        1000.0,
        None,
        action_cond,
        frame_st_id=frame_st_id,
    )
    teacher_teacher_plan_action_branch_input = {
        key: value.detach().clone()
        for key, value in teacher_teacher_plan_action_probe["action_res_lst"].items()
        if isinstance(value, torch.Tensor)
    }

    teacher_teacher_plan_cache_valid_length = cache_valid_length(
        teacher, teacher_plan_cache
    )
    teacher_teacher_plan_fingerprint = build_condition_fingerprint(
        checkpoint_owner="teacher",
        history_hash=history_hash,
        prompt_hash=stable_hash(prompt_ids),
        observation_hash=observation_hash,
        model_action_history_hash=model_action_history_hash,
        prepared_plan=ts_plan_capture.prepared_z_s,
        prepared_plan_timestep=ts_plan_capture.prepared_z_s_timestep,
        action_base_noise=teacher_action_base_noise,
        action_timestep=teacher_teacher_plan_action_branch_input["timesteps"],
        mask=loss_mask,
        normalization_metadata=normalization_metadata,
        frame_st_id=frame_st_id,
        token_positions=teacher_teacher_plan_action_branch_input["grid_id"].reshape(-1).tolist(),
        cache_valid_length=teacher_teacher_plan_cache_valid_length,
        sigma_start=macro_boundaries[0],
        sigma_end=macro_boundaries[-1],
    )
    ts_student_fingerprint = build_condition_fingerprint(
        checkpoint_owner="student",
        history_hash=history_hash,
        prompt_hash=stable_hash(prompt_ids),
        observation_hash=observation_hash,
        model_action_history_hash=model_action_history_hash,
        prepared_plan=ts_plan_capture.prepared_z_s,
        prepared_plan_timestep=ts_plan_capture.prepared_z_s_timestep,
        action_base_noise=teacher_action_base_noise,
        action_timestep=ts_student_action_branch_input["timesteps"],
        mask=loss_mask,
        normalization_metadata=normalization_metadata,
        frame_st_id=frame_st_id,
        token_positions=ts_student_action_branch_input["grid_id"].reshape(-1).tolist(),
        cache_valid_length=ts_cache_valid_length,
        sigma_start=macro_boundaries[0],
        sigma_end=macro_boundaries[-1],
    )
    assert_fingerprint_match(
        ts_student_fingerprint, teacher_teacher_plan_fingerprint
    )
    assert_cache_semantics(
        ts_student_fingerprint, teacher_teacher_plan_fingerprint
    )

    cache_provenance = {
        "student": {
            "owner": "student",
            "cache_name": str(server.cache_name),
            "valid_length": int(student_cache_valid_length),
            "semantic_history_rebuilt": True,
        },
        "teacher_student_plan": {
            "owner": "teacher",
            "cache_name": str(teacher_cache),
            "valid_length": int(teacher_cache_valid_length),
            "semantic_history_rebuilt": True,
        },
        "teacher_teacher_plan": {
            "owner": "teacher",
            "cache_name": str(teacher_plan_cache),
            "valid_length": int(teacher_teacher_plan_cache_valid_length),
            "semantic_history_rebuilt": True,
            "kv_values_compared": False,
        },
        "student_teacher_plan": {
            "owner": "student",
            "cache_name": str(ts_cache),
            "valid_length": int(ts_cache_valid_length),
            "semantic_history_rebuilt": True,
            "kv_values_compared": False,
        },
    }

    teacher_plan_metrics = {
        "student_vs_teacher_on_teacher_plan": masked_metrics(
            student_chain[-1],
            teacher_plan_action,
            loss_mask,
        ),
        "teacher_on_student_plan_vs_teacher_on_teacher_plan": masked_metrics(
            final_bridge_endpoint,
            teacher_plan_action,
            loss_mask,
        ),
        "student_plan_vs_teacher_plan_rmse": (
            student_plan.float() - teacher_plan.float()
        ).square().mean().sqrt().item(),
    }

    if args.save_actions:
        student_action_model = student_chain[-1].clone()
        bridge_action_model = final_bridge_endpoint.clone()
        direct_action_model = direct_endpoint.clone()
        teacher_plan_action_model = teacher_plan_action.clone()
        student_action_model[:, ~server.action_mask] = 0
        bridge_action_model[:, ~server.action_mask] = 0
        direct_action_model[:, ~server.action_mask] = 0
        teacher_plan_action_model[:, ~server.action_mask] = 0
        is_stage_g = args.stage_g_collection_id is not None
        is_stage_m = args.stage_m_live_context_id is not None
        action_artifact = {
            "schema_version": 4 if is_stage_m else (3 if is_stage_g else 1),
            "task_id": task_id,
            "task_config": task_config,
            "env_seed": env_seed,
            "diffusion_seed": args.seed,
            "policy_version": str(args.student),
            "student_checkpoint": str(args.student),
            "teacher_transformer": str(args.teacher_transformer),
            "replay_context_path": observation_description,
            "context_chunks_used": context_chunks_used,
            "video_base_noise": initial_video_noise.cpu(),
            "teacher_video_base_noise": teacher_video_noise.cpu(),
            "student_plan_x0": student_plan.cpu(),
            "prepared_student_plan": canonical_context.prepared_z_s.cpu(),
            "prepared_student_plan_timestep": canonical_context.prepared_z_s_timestep.cpu(),
            "action_base_noise": student_chain[0].cpu(),
            "teacher_action_base_noise": teacher_action_base_noise.cpu(),
            "student_action_chain": torch.stack(student_chain, dim=1).cpu(),
            "teacher_direct_model_action": direct_action_model.cpu(),
            "valid_action_mask": loss_mask.cpu(),
            "action_channel_mask": server.action_mask.cpu(),
            "actions_q01": server.actions_q01.cpu(),
            "actions_q99": server.actions_q99.cpu(),
            "student_model_action": student_action_model.cpu(),
            "teacher_bridge_model_action": bridge_action_model.cpu(),
        "teacher_on_teacher_plan_model_action": (
                teacher_plan_action_model.cpu()
            ),
            "ts_student_model_action": ts_student_model_action.cpu(),
            "student_env_action": server.postprocess_action(
                student_action_model
            ),
            "teacher_bridge_env_action": server.postprocess_action(
                bridge_action_model
            ),
            "teacher_on_teacher_plan_env_action": server.postprocess_action(
                teacher_plan_action_model
            ),
            "ts_student_env_action": ts_student_env_action,
            # Legacy aliases retained for the existing online-intervention tools.
            "student_plan": student_plan.cpu(),
            "teacher_plan": teacher_plan.cpu(),
            "prompt": prompt,
            "observation": observation_description,
            "frame_st_id": frame_st_id,
            "seed": args.seed,
            "student_action_steps": args.student_action_steps,
            "teacher_noise_seed": teacher_noise_seed,
            "macro_boundaries": macro_boundaries,
            "goal1_condition_schema": canonical_context.schema_version,
            "goal1_production_schema": GOAL1_PRODUCTION_SCHEMA_VERSION,
            "goal1_production_field_policy": GOAL1_PRODUCTION_FIELD_POLICY,
            "artifact_kind": "goal1_canonical_action_context",
            "canonical_action_context": canonical_context.to_dict(),
            "condition_fingerprint": {
                "student": student_fingerprint.to_dict(),
                "teacher": teacher_fingerprint.to_dict(),
                "teacher_teacher_plan": teacher_teacher_plan_fingerprint.to_dict(),
                "ts_student": ts_student_fingerprint.to_dict(),
            },
            "cache_provenance": cache_provenance,
            "goal1_plan_prepare_diff": tensor_diff(
                prepared_plan_capture.raw_z_s,
                prepared_plan_capture.prepared_z_s,
            ),
            "goal1_teacher_plan_injection_diff": tensor_diff(
                prepared_plan_capture.prepared_z_s,
                teacher_plan_capture.prepared_z_s,
            ),
        }
        if is_stage_g:
            from experiments.stage_g_data_manifest import (
                TEACHER_TARGET,
                build_label_key,
                build_policy_version,
                build_state_key,
                file_sha256,
            )

            policy_version = build_policy_version(
                args.student,
                args.student_delta,
            )
            state_key = build_state_key(
                task_id=task_id,
                task_config=task_config,
                env_seed=env_seed,
                frame_st_id=frame_st_id,
                prompt=prompt,
                replay_context_path=args.replay_context,
                student_plan=student_plan,
            )
            label_key = build_label_key(
                state_key=state_key,
                policy_version=policy_version,
                action_base_noise=student_chain[0],
                teacher_transformer=args.teacher_transformer,
            )
            action_artifact.update(
                {
                    "artifact_kind": "stage_g_teacher_bridge_label",
                    "teacher_target_kind": TEACHER_TARGET,
                    "collection_id": args.stage_g_collection_id,
                    "round_index": args.stage_g_round_index,
                    "state_key": state_key,
                    "label_key": label_key,
                    "policy_version": policy_version,
                    "policy_delta_path": (
                        str(args.student_delta.resolve())
                        if args.student_delta is not None
                        else None
                    ),
                    "policy_delta_sha256": (
                        file_sha256(args.student_delta)
                        if args.student_delta is not None
                        else None
                    ),
                }
            )
        if is_stage_m:
            from experiments.stage_g_data_manifest import (
                build_policy_version,
                file_sha256,
            )

            policy_version = build_policy_version(
                args.student,
                args.student_delta,
            )
            if replay_context_payload is None:
                raise AssertionError("Stage M replay context was not loaded")
            if replay_context_payload["policy_version"] != policy_version:
                raise ValueError(
                    "Stage M live context policy_version mismatch: "
                    f"captured={replay_context_payload['policy_version']!r} "
                    f"labeler={policy_version!r}"
                )
            action_artifact.update(
                {
                    "artifact_kind": "stage_m_live_teacher_bridge_label",
                    "teacher_target_kind": (
                        "teacher_bridge_pathwise_macro_endpoint"
                    ),
                    "live_context_id": args.stage_m_live_context_id,
                    "replay_context_sha256": file_sha256(args.replay_context),
                    "replay_context_semantic_sha256": (
                        replay_context_payload["semantic_sha256"]
                    ),
                    "policy_version": policy_version,
                    "policy_delta_path": (
                        str(args.student_delta.resolve())
                        if args.student_delta is not None
                        else None
                    ),
                    "policy_delta_sha256": (
                        file_sha256(args.student_delta)
                        if args.student_delta is not None
                        else None
                    ),
                }
            )
        args.save_actions.parent.mkdir(parents=True, exist_ok=True)
        torch.save(action_artifact, args.save_actions)

    result = {
        "prototype_question": (
            "Does Teacher Bridge remain non-degenerate on a real RoboTwin "
            "observation and a student-generated plan/action chain?"
        ),
        "scope": (
            "Real saved observation and prompt; student-induced denoising states; "
            "no simulator step and no policy-quality claim."
        ),
        "observation": observation_description,
        "observation_history_length": observation_history_length,
        "frame_st_id": frame_st_id,
        "prompt": prompt,
        "student_action_steps": args.student_action_steps,
        "teacher_noise_seed": teacher_noise_seed,
        "cache_provenance": cache_provenance,
        "student_plan_shape": list(student_plan.shape),
        "student_action_chain_shape": [
            len(student_chain),
            *student_chain[0].shape,
        ],
        "macro_boundaries": macro_boundaries,
        "policy_delta": (
            str(args.student_delta) if args.student_delta is not None else None
        ),
        "stage_g_collection_id": args.stage_g_collection_id,
        "stage_g_round_index": args.stage_g_round_index,
        "action_artifact": str(args.save_actions) if args.save_actions else None,
        "intervals": interval_results,
        "teacher_plan_control": teacher_plan_metrics,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
