from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.waopd_v0j_teacher_free_behavior import (
    _resolve_adapter_load_spec,
    _validate_loaded_adapter,
)


def _all_block_checkpoint(path: Path) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    contract: dict[str, object] = {
        "rank": 8,
        "alpha": 8.0,
        "dropout": 0.0,
        "block_indices": list(range(30)),
    }
    state = {
        "transformer_blocks.0.attn1.to_q.lora_A": torch.arange(8.0),
        "transformer_blocks.29.attn1.to_q.lora_B": torch.arange(8.0) + 1.0,
    }
    torch.save(
        {
            "adapter_kind": "joint_lora",
            "adapter_contract": contract,
            "adapter_state_dict": state,
        },
        path,
    )
    return contract, state


def test_adapter_load_spec_uses_all_blocks_from_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    contract, state = _all_block_checkpoint(checkpoint)

    spec = _resolve_adapter_load_spec(checkpoint, "joint_lora")

    assert spec.kind == "joint_lora"
    assert spec.rank == 8
    assert spec.alpha == 8.0
    assert spec.dropout == 0.0
    assert spec.block_indices == tuple(range(30))
    assert spec.contract == contract
    assert set(spec.state_dict or {}) == set(state)


def test_loaded_adapter_validation_rejects_partial_strict_false_load(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    contract, state = _all_block_checkpoint(checkpoint)
    spec = _resolve_adapter_load_spec(checkpoint, None)

    class PartialRuntime:
        def adapter_contract(self) -> dict[str, object]:
            return contract

        def adapter_state(self) -> dict[str, torch.Tensor]:
            first_name = next(iter(state))
            return {first_name: state[first_name]}

    with pytest.raises(RuntimeError, match="parameter names differ"):
        _validate_loaded_adapter(PartialRuntime(), spec)  # type: ignore[arg-type]
