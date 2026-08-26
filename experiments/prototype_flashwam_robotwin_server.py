"""PROTOTYPE — Native RoboTwin Flash-WAM server with explicit inference NFE.

Question
--------
Does the released Flash-WAM checkpoint perform better in closed-loop RoboTwin
control at 1v/2a than at its published 1v/1a operating point?

This entry point only exposes the already-existing LingBot-VA server with
runtime video/action step overrides. It does not modify or train the model.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(
    os.environ.get(
        "WAVE_RL_ROOT",
        os.environ.get("PROJECT_ROOT", str(WORKSPACE_ROOT.parent / "wave-rl")),
    )
).expanduser().resolve()
for path in (
    WORKSPACE_ROOT,
    PROJECT_ROOT / "third_party" / "lingbot-va",
    PROJECT_ROOT / "third_party" / "lingbot-va" / "wan_va",
):
    sys.path.insert(0, str(path))

from experiments.flashwam_runtime_state_audit import (
    runtime_state_component_sha256,
    runtime_state_sha256,
)


def optional_env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    return None if raw is None or not raw.strip() else int(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video-steps", type=int, default=1)
    parser.add_argument(
        "--video-exec-steps",
        type=int,
        default=optional_env_int("VIDEO_EXEC_STEPS"),
        help=(
            "Optional number of leading video scheduler intervals to execute. "
            "For the LingBot-VA paper schedule, use --video-steps 5 "
            "--video-exec-steps 3 to stop at nominal flow time s=0.6."
        ),
    )
    parser.add_argument("--action-steps", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--save-root", type=Path, required=True)
    parser.add_argument(
        "--audit-root",
        type=Path,
        help="Persistent audit directory; defaults to --save-root.",
    )
    parser.add_argument(
        "--action-diffusion-seed",
        type=int,
        help=(
            "Base seed for a deterministic per-environment/per-chunk diffusion "
            "RNG schedule. The upstream implementation samples video then "
            "action noise from the same CUDA RNG, so this fixes both initial "
            "noise tensors while making the action noise reproducible."
        ),
    )
    parser.add_argument(
        "--action-diffusion-direct-seed",
        type=int,
        help=(
            "Use this exact RNG seed for each action inference. This is only "
            "for replaying offline Teacher-Bridge labels whose --seed was "
            "passed directly to torch.manual_seed."
        ),
    )
    parser.add_argument("--diffusion-noise-artifact", type=Path)
    parser.add_argument("--diffusion-noise-frame", type=int)
    parser.add_argument(
        "--save-diffusion-noise",
        type=Path,
        help=(
            "Optional diagnostic prefix. When set, save the actual video and "
            "action torch.randn tensors for every inference call."
        ),
    )
    parser.add_argument("--override-seed", type=int)
    parser.add_argument("--prefix-artifact", type=Path)
    parser.add_argument("--prefix-key", default="prefix_env_actions")
    parser.add_argument("--prefix-chunks", type=int, default=0)
    parser.add_argument("--intervention-artifact", type=Path)
    parser.add_argument("--intervention-key")
    parser.add_argument("--intervention-frame", type=int)
    parser.add_argument("--transformer-delta", type=Path)
    parser.add_argument(
        "--runtime-state-audit",
        action="store_true",
        help="Return hashes of effective transformer/VAE recurrent state.",
    )
    return parser.parse_args()


def derive_chunk_diffusion_seed(
    action_diffusion_seed: int,
    environment_seed: int,
    frame_st_id: int,
) -> int:
    """Derive a stable torch-compatible seed for one inference chunk."""
    coordinates = (
        "flashwam_action_diffusion_v1"
        f"|{int(action_diffusion_seed)}"
        f"|{int(environment_seed)}"
        f"|{int(frame_st_id)}"
    ).encode("ascii")
    digest = hashlib.blake2b(coordinates, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**63)


def resolve_action_inference_seed(
    *,
    base_seed: int | None,
    direct_seed: int | None,
    environment_seed: int,
    frame_st_id: int,
) -> int | None:
    if base_seed is not None and direct_seed is not None:
        raise ValueError("base and direct action diffusion seeds are mutually exclusive")
    if direct_seed is not None:
        return int(direct_seed)
    if base_seed is not None:
        return derive_chunk_diffusion_seed(
            base_seed,
            environment_seed,
            frame_st_id,
        )
    return None


def load_diffusion_noise_replay(path: str | Path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("diffusion noise artifact must contain a dictionary")
    required = ("video_base_noise", "action_base_noise")
    if any(key not in payload for key in required):
        raise ValueError("diffusion noise artifact lacks saved base noise")
    video = payload["video_base_noise"]
    action = payload["action_base_noise"]
    if not isinstance(video, torch.Tensor) or not isinstance(action, torch.Tensor):
        raise ValueError("saved diffusion noise must be torch tensors")
    if video.ndim != 5 or action.ndim != 5:
        raise ValueError("saved diffusion noise must be rank-5")
    return video.detach().clone(), action.detach().clone()


def array_sha256(array: np.ndarray) -> str:
    """Hash an action trace including its representation metadata."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def observation_sha256(value: object) -> str:
    """Hash a nested policy observation including types, shapes, and values."""

    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, np.ndarray):
            contiguous = np.ascontiguousarray(item)
            digest.update(b"array\0")
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(contiguous.tobytes())
            return
        if isinstance(item, np.generic):
            update(np.asarray(item))
            return
        if isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item):
                if not isinstance(key, str):
                    raise TypeError("observation dictionary keys must be strings")
                update(key)
                update(item[key])
            return
        if isinstance(item, list):
            digest.update(f"list:{len(item)}\0".encode("ascii"))
            for child in item:
                update(child)
            return
        if isinstance(item, tuple):
            digest.update(f"tuple:{len(item)}\0".encode("ascii"))
            for child in item:
                update(child)
            return
        if item is None or isinstance(item, (str, int, float, bool)):
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(b"\0")
            digest.update(
                json.dumps(item, allow_nan=False, ensure_ascii=False).encode("utf-8")
            )
            digest.update(b"\0")
            return
        raise TypeError(f"unsupported observation value: {type(item).__name__}")

    update(value)
    return digest.hexdigest()


def prefix_actions_from_artifact(
    artifact: dict[str, object], key: str, count: int
) -> list[np.ndarray]:
    """Read either a legacy action list or captured Stage G/H context chunks."""

    selected = artifact[key][:count]
    actions = []
    for item in selected:
        if isinstance(item, dict):
            if "env_action" not in item:
                raise ValueError(f"prefix chunk under {key!r} has no env_action")
            item = item["env_action"]
        actions.append(np.asarray(item))
    return actions


def load_partial_transformer_state(transformer, state_dict) -> None:
    """Copy a partial plain-Tensor state into Tensor or world-size-one DTensor params."""
    import torch

    targets = dict(transformer.named_parameters())
    targets.update(dict(transformer.named_buffers()))
    unexpected = sorted(set(state_dict) - set(targets))
    if unexpected:
        raise ValueError(
            f"transformer delta has unexpected keys: {unexpected[:8]}"
        )
    with torch.no_grad():
        for name, source in state_dict.items():
            target = targets[name]
            if tuple(target.shape) != tuple(source.shape):
                raise ValueError(
                    f"transformer delta shape mismatch for {name}: "
                    f"target={tuple(target.shape)} source={tuple(source.shape)}"
                )
            local_target = (
                target.to_local() if hasattr(target, "to_local") else target
            )
            if local_target.numel() != source.numel():
                raise ValueError(
                    f"transformer delta local shard mismatch for {name}: "
                    f"local={tuple(local_target.shape)} "
                    f"source={tuple(source.shape)}"
                )
            local_target.copy_(
                source.reshape(local_target.shape).to(
                    device=local_target.device,
                    dtype=local_target.dtype,
                )
            )


def apply_transformer_delta(transformer, delta) -> dict[str, object]:
    """Attach any required module and load a supported Student delta."""

    delta_format = delta.get("format")
    state_dict = delta.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("transformer delta has no state_dict")

    if delta_format == "flashwam_action_output_adapter_v1":
        adapter = delta.get("adapter")
        if not isinstance(adapter, dict):
            raise ValueError("action output adapter checkpoint has no metadata")
        if adapter.get("kind") != "action_output_residual":
            raise ValueError(
                "unsupported action output adapter kind: "
                f"{adapter.get('kind')!r}"
            )
        from experiments.action_output_adapter import (
            attach_action_output_adapter,
        )

        info = attach_action_output_adapter(
            transformer,
            rank=int(adapter["rank"]),
            initialization=str(
                adapter.get("initialization", "gated_random")
            ),
        )
        adapter_rank = info.rank
    elif delta_format == "flashwam_action_delta_v1":
        adapter_rank = None
    else:
        raise ValueError(
            "unsupported transformer delta format: "
            f"{delta_format!r}"
        )

    load_partial_transformer_state(transformer, state_dict)
    return {
        "format": delta_format,
        "tensor_count": len(state_dict),
        "adapter_rank": adapter_rank,
    }


def main() -> None:
    args = parse_args()
    if args.video_exec_steps is not None and not (
        1 <= args.video_exec_steps <= args.video_steps
    ):
        raise ValueError(
            "video exec steps must lie in [1, video scheduler steps]"
        )
    resolve_action_inference_seed(
        base_seed=args.action_diffusion_seed,
        direct_seed=args.action_diffusion_direct_seed,
        environment_seed=0,
        frame_st_id=0,
    )

    from configs import VA_CONFIGS
    from distributed.util import init_distributed
    from utils import init_logger, run_async_server_mode
    from wan_va_server import VA_Server

    import torch

    if (args.diffusion_noise_artifact is None) != (
        args.diffusion_noise_frame is None
    ):
        raise ValueError(
            "diffusion noise replay requires both artifact and frame"
        )
    diffusion_noise_replay = (
        load_diffusion_noise_replay(args.diffusion_noise_artifact)
        if args.diffusion_noise_artifact is not None
        else None
    )

    prefix_actions: list[np.ndarray] = []
    if args.prefix_artifact:
        prefix_artifact = torch.load(
            args.prefix_artifact,
            map_location="cpu",
            weights_only=False,
        )
        prefix_actions = prefix_actions_from_artifact(
            prefix_artifact,
            args.prefix_key,
            args.prefix_chunks,
        )

    intervention_action = None
    if args.intervention_artifact:
        if args.intervention_key is None or args.intervention_frame is None:
            raise ValueError(
                "intervention artifact requires --intervention-key and "
                "--intervention-frame"
            )
        intervention_artifact = torch.load(
            args.intervention_artifact,
            map_location="cpu",
            weights_only=False,
        )
        intervention_action = np.asarray(
            intervention_artifact[args.intervention_key]
        )
    if (prefix_actions or intervention_action is not None) and args.override_seed is None:
        raise ValueError("action overrides require --override-seed")

    persistent_audit_root = args.audit_root or args.save_root
    persistent_audit_root.mkdir(parents=True, exist_ok=True)
    audit_dir = persistent_audit_root / "online_override_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    kv_input_audit_dir = persistent_audit_root / "kv_input_audit"
    kv_input_audit_dir.mkdir(parents=True, exist_ok=True)
    diffusion_audit_path = (
        persistent_audit_root / "action_diffusion_audit.jsonl"
    )

    class PairedVA_Server(VA_Server):
        current_seed = None
        intervention_applied = False
        noise_capture_index = 0

        def _infer(self, obs, frame_st_id=0):
            if args.save_diffusion_noise is not None:
                original_randn = torch.randn
                captured_noise = []

                def capture_randn(*shape, **kwargs):
                    value = original_randn(*shape, **kwargs)
                    captured_noise.append(value.detach().cpu().clone())
                    return value

                torch.randn = capture_randn
                try:
                    result = super()._infer(obs, frame_st_id=frame_st_id)
                finally:
                    torch.randn = original_randn
                if len(captured_noise) != 2:
                    raise AssertionError(
                        "diffusion noise capture expected exactly video and action "
                        f"tensors, got {len(captured_noise)} at frame {frame_st_id}"
                    )
                output = args.save_diffusion_noise
                output.parent.mkdir(parents=True, exist_ok=True)
                capture_index = PairedVA_Server.noise_capture_index
                PairedVA_Server.noise_capture_index += 1
                output = output.with_name(
                    f"{output.stem}_{capture_index:04d}{output.suffix or '.pt'}"
                )
                torch.save(
                    {
                        "schema": "flashwam_diffusion_noise_capture_v1",
                        "environment_seed": self.current_seed,
                        "frame_st_id": int(frame_st_id),
                        "capture_index": capture_index,
                        "video_timestep_start": float(
                            self.scheduler.timesteps[0].detach().cpu()
                        ),
                        # The native loop pads the scheduler with the
                        # terminal endpoint 0; scheduler.timesteps itself
                        # contains only the denoising knots.
                        "video_timestep_end": 0.0,
                        "action_timestep_start": float(
                            self.action_scheduler.timesteps[0].detach().cpu()
                        ),
                        "action_timestep_end": 0.0,
                        "video_base_noise": captured_noise[0],
                        "action_base_noise": captured_noise[1],
                    },
                    output,
                )
                return result
            if (
                diffusion_noise_replay is None
                or frame_st_id != args.diffusion_noise_frame
            ):
                return super()._infer(obs, frame_st_id=frame_st_id)

            saved_noise = iter(diffusion_noise_replay)
            original_randn = torch.randn

            def replay_randn(*shape, **kwargs):
                try:
                    value = next(saved_noise)
                except StopIteration as error:
                    raise AssertionError(
                        "diffusion noise replay exceeded two saved tensors"
                    ) from error
                requested_shape = tuple(int(item) for item in shape)
                if requested_shape != tuple(value.shape):
                    raise AssertionError(
                        "diffusion noise replay shape differs: "
                        f"expected={tuple(value.shape)} actual={requested_shape}"
                    )
                return value.to(
                    device=kwargs.get("device", value.device),
                    dtype=kwargs.get("dtype", value.dtype),
                )

            torch.randn = replay_randn
            try:
                result = super()._infer(obs, frame_st_id=frame_st_id)
                try:
                    next(saved_noise)
                except StopIteration:
                    pass
                else:
                    raise AssertionError(
                        "diffusion noise replay did not consume both tensors"
                    )
                return result
            finally:
                torch.randn = original_randn

        def infer(self, obs):
            reset = bool(obs.get("reset"))
            compute_kv_cache = bool(obs.get("compute_kv_cache"))
            initialize_context = bool(obs.get("initialize_context"))
            if initialize_context:
                if reset or compute_kv_cache or getattr(self, "frame_st_id", 0) != 0:
                    raise AssertionError(
                        "context initialization requires a fresh reset"
                    )
                self.init_latent = self._encode_obs(obs)
                result = {}
                if args.runtime_state_audit:
                    result["initialized_runtime_sha256"] = runtime_state_sha256(
                        self
                    )
                return result
            if reset:
                self.intervention_applied = False
                if "seed" in obs:
                    environment_seed = int(obs["seed"])
                    self.current_seed = environment_seed
                    torch.manual_seed(environment_seed)
                    torch.cuda.manual_seed_all(environment_seed)
                elif (
                    args.action_diffusion_seed is not None
                    or args.action_diffusion_direct_seed is not None
                ):
                    raise ValueError(
                        "action diffusion determinism requires reset payload seed"
                    )
            frame_st_id = getattr(self, "frame_st_id", 0)
            if compute_kv_cache:
                cache_for_frame_st_id = int(frame_st_id) + 2
                kv_input_record = {
                    "schema": "flashwam_kv_input_audit_v1",
                    "environment_seed": self.current_seed,
                    "frame_st_id": cache_for_frame_st_id,
                    "observation_sha256": observation_sha256(obs.get("obs")),
                }
                torch.save(
                    {**kv_input_record, "observation": obs.get("obs")},
                    kv_input_audit_dir
                    / f"seed{self.current_seed}_frame{cache_for_frame_st_id}.pt",
                )
                print(
                    "PROTOTYPE_KV_INPUT_AUDIT "
                    + json.dumps(kv_input_record, sort_keys=True),
                    flush=True,
                )
            chunk_seed = None
            pre_action_runtime_sha256 = None
            if (
                not reset
                and not compute_kv_cache
                and not initialize_context
                and (
                    args.action_diffusion_seed is not None
                    or args.action_diffusion_direct_seed is not None
                )
            ):
                if self.current_seed is None:
                    raise ValueError(
                        "action inference occurred before a seeded reset"
                    )
                chunk_seed = resolve_action_inference_seed(
                    base_seed=args.action_diffusion_seed,
                    direct_seed=args.action_diffusion_direct_seed,
                    environment_seed=self.current_seed,
                    frame_st_id=frame_st_id,
                )
                torch.manual_seed(chunk_seed)
                torch.cuda.manual_seed_all(chunk_seed)
            if args.runtime_state_audit and not reset and not compute_kv_cache:
                pre_action_runtime_sha256 = runtime_state_sha256(self)
            result = super().infer(obs)
            if args.runtime_state_audit:
                runtime_record = {
                    "schema": "flashwam_runtime_state_audit_v1",
                    "environment_seed": self.current_seed,
                    "frame_st_id": int(getattr(self, "frame_st_id", 0)),
                    "component_sha256": runtime_state_component_sha256(self),
                }
                result["runtime_state_component_sha256"] = runtime_record[
                    "component_sha256"
                ]
                if compute_kv_cache:
                    post_kv_runtime_sha256 = runtime_state_sha256(self)
                    result["post_kv_runtime_sha256"] = post_kv_runtime_sha256
                    runtime_record["kind"] = "post_kv"
                    runtime_record["runtime_state_sha256"] = (
                        post_kv_runtime_sha256
                    )
                elif pre_action_runtime_sha256 is not None:
                    result["pre_action_runtime_sha256"] = (
                        pre_action_runtime_sha256
                    )
                    runtime_record["kind"] = "pre_action"
                    runtime_record["runtime_state_sha256"] = (
                        pre_action_runtime_sha256
                    )
                else:
                    runtime_record["kind"] = "reset"
                    runtime_record["runtime_state_sha256"] = (
                        runtime_state_sha256(self)
                    )
                print(
                    "PROTOTYPE_RUNTIME_STATE_AUDIT "
                    + json.dumps(runtime_record, sort_keys=True),
                    flush=True,
                )
            if "action" in result:
                action = np.asarray(result["action"])
                audit_record = {
                    "schema": "flashwam_action_diffusion_audit_v1",
                    "environment_seed": self.current_seed,
                    "action_diffusion_seed": args.action_diffusion_seed,
                    "action_diffusion_direct_seed": (
                        args.action_diffusion_direct_seed
                    ),
                    "chunk_diffusion_seed": chunk_seed,
                    "frame_st_id": int(frame_st_id),
                    "action_sha256": array_sha256(action),
                    "action_shape": list(action.shape),
                    "action_dtype": str(action.dtype),
                    "action_timestep_start": float(
                        self.action_scheduler.timesteps[0].detach().cpu()
                    ),
                    # This is the macro endpoint consumed by the native
                    # F.pad(..., value=0) action loop, not the last raw
                    # scheduler knot (which is still 1000 for 1-step NFE).
                    "action_timestep_end": 0.0,
                    "rng_scope": (
                        "joint_video_action_initial_noise"
                        if (
                            args.action_diffusion_seed is not None
                            or args.action_diffusion_direct_seed is not None
                        )
                        else "uncontrolled"
                    ),
                }
                with diffusion_audit_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(audit_record, sort_keys=True) + "\n")
                print(
                    "PROTOTYPE_ACTION_DIFFUSION_AUDIT "
                    + json.dumps(audit_record, sort_keys=True),
                    flush=True,
                )
            if (
                "action" not in result
                or self.current_seed != args.override_seed
            ):
                return result

            replacement = None
            replacement_kind = None
            prefix_index = frame_st_id // 2
            if frame_st_id % 2 == 0 and prefix_index < len(prefix_actions):
                replacement = prefix_actions[prefix_index]
                replacement_kind = "prefix"
            if (
                intervention_action is not None
                and frame_st_id == args.intervention_frame
                and not self.intervention_applied
            ):
                replacement = intervention_action
                replacement_kind = "intervention"
                self.intervention_applied = True
            if replacement is None:
                return result
            if result["action"].shape != replacement.shape:
                raise ValueError(
                    f"override shape {replacement.shape} does not match "
                    f"generated action {result['action'].shape}"
                )

            torch.save(
                {
                    "seed": self.current_seed,
                    "frame_st_id": frame_st_id,
                    "kind": replacement_kind,
                    "generated_student_action": result["action"],
                    "replacement_action": replacement,
                },
                audit_dir / f"seed{self.current_seed}_frame{frame_st_id}.pt",
            )
            result["action"] = replacement.copy()
            print(
                "PROTOTYPE_ONLINE_OVERRIDE "
                f"seed={self.current_seed} frame_st_id={frame_st_id} "
                f"kind={replacement_kind}",
                flush=True,
            )
            return result

    init_logger()
    config = deepcopy(VA_CONFIGS["robotwin"])
    config.wan22_pretrained_model_name_or_path = str(args.checkpoint)
    config.num_inference_steps = args.video_steps
    if args.video_exec_steps is not None:
        config.video_exec_step = args.video_exec_steps
    config.action_num_inference_steps = args.action_steps
    config.save_root = str(args.save_root)
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1

    init_distributed(config.world_size, config.local_rank, config.rank)
    server = PairedVA_Server(config)
    if args.transformer_delta:
        delta = torch.load(
            args.transformer_delta,
            map_location="cpu",
            weights_only=False,
        )
        load_result = apply_transformer_delta(server.transformer, delta)
        print(
            "PROTOTYPE_ACTION_DELTA "
            f"path={args.transformer_delta} "
            f"format={load_result['format']} "
            f"tensors={load_result['tensor_count']} "
            f"adapter_rank={load_result['adapter_rank']}",
            flush=True,
        )
    print(
        "PROTOTYPE_CONFIG "
        f"video_steps={config.num_inference_steps} "
        f"video_exec_steps={config.video_exec_step} "
        f"action_steps={config.action_num_inference_steps} "
        f"checkpoint={config.wan22_pretrained_model_name_or_path} "
        "teacher_loaded=false "
        f"action_intervention={intervention_action is not None} "
        f"override_seed={args.override_seed} "
        f"action_diffusion_seed={args.action_diffusion_seed} "
        f"action_diffusion_direct_seed={args.action_diffusion_direct_seed} "
        f"diffusion_noise_frame={args.diffusion_noise_frame} "
        "diffusion_rng_scope=joint_video_action_initial_noise "
        f"prefix_chunks={len(prefix_actions)} "
        f"intervention_frame={args.intervention_frame}",
        flush=True,
    )
    run_async_server_mode(server, config.local_rank, config.host, args.port)


if __name__ == "__main__":
    main()
