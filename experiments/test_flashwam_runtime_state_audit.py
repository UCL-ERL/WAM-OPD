"""Tests for hashing the effective FlashWAM autoregressive runtime state."""

from __future__ import annotations

import unittest

import torch

from experiments.flashwam_runtime_state_audit import (
    runtime_state_component_sha256,
    runtime_state_sha256,
)


class _Attention:
    def __init__(self) -> None:
        self.attn_caches = {
            "pos": {
                "k": torch.arange(24, dtype=torch.float32).reshape(1, 6, 2, 2),
                "v": torch.arange(24, dtype=torch.float32).reshape(1, 6, 2, 2) + 1,
                "id": torch.tensor([0, 1, -1, -1, -1, -1]),
                "mask": torch.tensor([True, True, False, False, False, False]),
                "is_pred": torch.tensor([False, False, False, False, False, False]),
            }
        }


class _Block:
    def __init__(self) -> None:
        self.attn1 = _Attention()


class _Transformer:
    def __init__(self) -> None:
        self.blocks = [_Block()]


class _VAE:
    def __init__(self) -> None:
        self.feat_cache = [torch.tensor([1.0, 2.0]), None]


class _Server:
    def __init__(self) -> None:
        self.cache_name = "pos"
        self.frame_st_id = 4
        self.init_latent = torch.tensor([3.0])
        self.transformer = _Transformer()
        self.streaming_vae = _VAE()
        self.streaming_vae_half = None


class RuntimeStateAuditTest(unittest.TestCase):
    def test_component_hash_localizes_transformer_difference(self) -> None:
        left = _Server()
        right = _Server()
        right.transformer.blocks[0].attn1.attn_caches["pos"]["k"][
            0, 0, 0, 0
        ] += 1

        left_hashes = runtime_state_component_sha256(left)
        right_hashes = runtime_state_component_sha256(right)
        self.assertNotEqual(
            left_hashes["transformer_block_0"],
            right_hashes["transformer_block_0"],
        )
        self.assertEqual(
            left_hashes["streaming_vae"], right_hashes["streaming_vae"]
        )
        self.assertEqual(left_hashes["init_latent"], right_hashes["init_latent"])

    def test_hash_is_stable_and_tracks_effective_cache(self) -> None:
        left = _Server()
        right = _Server()

        self.assertEqual(runtime_state_sha256(left), runtime_state_sha256(right))
        right.transformer.blocks[0].attn1.attn_caches["pos"]["k"][0, 0, 0, 0] += 1
        self.assertNotEqual(runtime_state_sha256(left), runtime_state_sha256(right))

    def test_hash_ignores_unallocated_kv_slots(self) -> None:
        left = _Server()
        right = _Server()
        right.transformer.blocks[0].attn1.attn_caches["pos"]["k"][0, 5] += 999
        right.transformer.blocks[0].attn1.attn_caches["pos"]["v"][0, 4] -= 999

        self.assertEqual(runtime_state_sha256(left), runtime_state_sha256(right))

    def test_hash_tracks_streaming_vae_and_frame(self) -> None:
        left = _Server()
        right = _Server()
        right.streaming_vae.feat_cache[0][0] += 1
        self.assertNotEqual(runtime_state_sha256(left), runtime_state_sha256(right))

        right = _Server()
        right.frame_st_id += 2
        self.assertNotEqual(runtime_state_sha256(left), runtime_state_sha256(right))


if __name__ == "__main__":
    unittest.main()
