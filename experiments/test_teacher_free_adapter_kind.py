from pathlib import Path

import pytest
import torch

from experiments.waopd_v0j_teacher_free_behavior import _resolve_adapter_kind


def test_adapter_kind_is_auto_detected_from_dual_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "dual.pt"
    torch.save({"adapter_kind": "dual_lora", "adapter_state_dict": {}}, checkpoint)
    assert _resolve_adapter_kind(checkpoint, None) == "dual_lora"


def test_adapter_kind_override_must_match_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "dual.pt"
    torch.save({"adapter_kind": "dual_lora", "adapter_state_dict": {}}, checkpoint)
    with pytest.raises(ValueError, match="differs from checkpoint"):
        _resolve_adapter_kind(checkpoint, "joint_lora")


def test_legacy_adapter_state_defaults_to_joint_lora(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"blocks.0.lora_A": torch.ones(1)}, checkpoint)
    assert _resolve_adapter_kind(checkpoint, None) == "joint_lora"


def test_metadata_less_dual_keys_do_not_silently_fall_back_to_joint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "ambiguous-dual.pt"
    torch.save(
        {"blocks.0.attn1.to_q.video_lora_A": torch.ones(1)}, checkpoint
    )
    with pytest.raises(ValueError, match="dual-bank keys"):
        _resolve_adapter_kind(checkpoint, None)
    with pytest.raises(ValueError, match="dual-bank keys"):
        _resolve_adapter_kind(checkpoint, "joint_lora")


def test_unknown_metadata_less_mapping_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unknown.pt"
    torch.save({"optimizer_state_dict": {}}, checkpoint)
    with pytest.raises(ValueError, match="no recognized JointLoRA keys"):
        _resolve_adapter_kind(checkpoint, None)


def test_adapter_kind_without_checkpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires --adapter-state"):
        _resolve_adapter_kind(None, "dual_lora")
