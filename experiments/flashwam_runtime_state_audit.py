"""Hash the effective recurrent state used by a FlashWAM server.

Unallocated KV pool entries come from ``torch.empty`` and are deliberately
excluded.  The digest covers every allocated transformer KV slot, allocation
metadata, both streaming-VAE caches, the initial latent, and frame position.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch


def _update_scalar(digest: Any, label: str, value: object) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(value, sort_keys=True).encode("utf-8"))
    digest.update(b"\0")


def _update_tensor(digest: Any, label: str, value: torch.Tensor) -> None:
    tensor = value.detach()
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    tensor = tensor.contiguous()
    _update_scalar(digest, f"{label}.dtype", str(tensor.dtype))
    _update_scalar(digest, f"{label}.shape", list(tensor.shape))
    digest.update(label.encode("utf-8"))
    digest.update(b"\0bytes\0")
    digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes())


def _update_optional_tensor(
    digest: Any,
    label: str,
    value: torch.Tensor | None,
) -> None:
    if value is None:
        _update_scalar(digest, label, None)
    else:
        _update_tensor(digest, label, value)


def _update_vae_cache(digest: Any, label: str, vae: object | None) -> None:
    if vae is None:
        _update_scalar(digest, label, None)
        return
    cache = getattr(vae, "feat_cache", None)
    if cache is None:
        raise ValueError(f"{label} has no feat_cache")
    _update_scalar(digest, f"{label}.count", len(cache))
    for index, value in enumerate(cache):
        _update_optional_tensor(digest, f"{label}[{index}]", value)


def runtime_state_sha256(server: object) -> str:
    """Return a deterministic digest of FlashWAM's effective next-step state."""

    digest = hashlib.sha256()
    cache_name = str(server.cache_name)
    _update_scalar(digest, "schema", "flashwam_runtime_state_v1")
    _update_scalar(digest, "cache_name", cache_name)
    _update_scalar(digest, "frame_st_id", int(server.frame_st_id))
    _update_optional_tensor(digest, "init_latent", server.init_latent)

    blocks = list(server.transformer.blocks)
    _update_scalar(digest, "block_count", len(blocks))
    for block_index, block in enumerate(blocks):
        caches = block.attn1.attn_caches
        if caches is None or cache_name not in caches or caches[cache_name] is None:
            raise ValueError(f"block {block_index} has no active {cache_name!r} cache")
        cache = caches[cache_name]
        prefix = f"block[{block_index}]"
        mask = cache["mask"].detach().to(dtype=torch.bool)
        valid = mask.nonzero(as_tuple=False).squeeze(-1)
        _update_tensor(digest, f"{prefix}.valid_slots", valid)
        _update_tensor(digest, f"{prefix}.id", cache["id"][valid])
        _update_tensor(digest, f"{prefix}.is_pred", cache["is_pred"][valid])
        _update_tensor(digest, f"{prefix}.k", cache["k"][:, valid])
        _update_tensor(digest, f"{prefix}.v", cache["v"][:, valid])

    _update_vae_cache(digest, "streaming_vae", server.streaming_vae)
    _update_vae_cache(
        digest,
        "streaming_vae_half",
        getattr(server, "streaming_vae_half", None),
    )
    return digest.hexdigest()


def runtime_state_component_sha256(server: object) -> dict[str, str]:
    """Return independent digests that localize recurrent-state differences."""

    result = {}
    cache_name = str(server.cache_name)

    digest = hashlib.sha256()
    _update_scalar(digest, "cache_name", cache_name)
    _update_scalar(digest, "frame_st_id", int(server.frame_st_id))
    result["metadata"] = digest.hexdigest()

    digest = hashlib.sha256()
    _update_optional_tensor(digest, "init_latent", server.init_latent)
    result["init_latent"] = digest.hexdigest()

    for block_index, block in enumerate(server.transformer.blocks):
        cache = block.attn1.attn_caches[cache_name]
        mask = cache["mask"].detach().to(dtype=torch.bool)
        valid = mask.nonzero(as_tuple=False).squeeze(-1)
        digest = hashlib.sha256()
        _update_tensor(digest, "valid_slots", valid)
        _update_tensor(digest, "id", cache["id"][valid])
        _update_tensor(digest, "is_pred", cache["is_pred"][valid])
        _update_tensor(digest, "k", cache["k"][:, valid])
        _update_tensor(digest, "v", cache["v"][:, valid])
        result[f"transformer_block_{block_index}"] = digest.hexdigest()

    for label, vae in (
        ("streaming_vae", server.streaming_vae),
        ("streaming_vae_half", getattr(server, "streaming_vae_half", None)),
    ):
        digest = hashlib.sha256()
        _update_vae_cache(digest, label, vae)
        result[label] = digest.hexdigest()
    return result
