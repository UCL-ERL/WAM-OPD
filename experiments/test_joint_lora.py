from __future__ import annotations

import torch
from torch import nn

from experiments.joint_lora import (
    JointLoRALinear,
    attach_joint_lora,
    joint_lora_base_parameter_hashes,
    joint_lora_state_dict,
)


class _Attention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Identity()])


class _Projection(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn1 = _Attention(dim)
        self.attn2 = _Attention(dim)
        self.ffn = nn.Module()
        self.ffn.net = nn.ModuleList([_Projection(dim), nn.Identity(), nn.Linear(dim, dim)])


class _Transformer(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(4)])


def test_joint_lora_has_exact_targets_and_zero_init_parity() -> None:
    torch.manual_seed(7)
    model = _Transformer()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    info = attach_joint_lora(model, rank=2, alpha=2.0, block_indices=(0, 1, 2, 3))
    assert len(info.modules) == 40
    assert info.trainable_parameter_count == 40 * 2 * (8 + 8)
    assert len([m for m in model.modules() if isinstance(m, JointLoRALinear)]) == 40
    after = {name: value.detach().clone() for name, value in model.named_parameters()}
    for name, value in before.items():
        normalized = name.replace(".base", "")
        if normalized in after:
            assert torch.equal(value, after[normalized])

    # B=0 makes every wrapped call bitwise equal to the clean base.
    x = torch.randn(3, 8)
    for module in model.modules():
        if isinstance(module, JointLoRALinear):
            assert torch.equal(module.base(x), module(x))

    # The same wrapper is used by both semantic streams; a nonzero B changes
    # both representative calls, with no action-mode bypass gate.
    wrappers = [m for m in model.modules() if isinstance(m, JointLoRALinear)]
    with torch.no_grad():
        for module in wrappers:
            module.lora_B.fill_(0.01)
    assert any(not torch.equal(module.base(x), module(x)) for module in wrappers)
    assert joint_lora_state_dict(model)
    assert joint_lora_base_parameter_hashes(model)
