from __future__ import annotations

import torch
from torch import nn

from experiments.joint_lora import JointLoRALinear


def test_joint_lora_keeps_fp32_master_parameters_on_bf16_base() -> None:
    torch.manual_seed(7)
    base = nn.Linear(4, 3, bias=True).to(dtype=torch.bfloat16)
    wrapper = JointLoRALinear(
        base,
        rank=2,
        alpha=2.0,
        dropout=0.0,
    )
    inputs = torch.randn(5, 4, dtype=torch.float32).to(dtype=torch.bfloat16)

    output = wrapper(inputs)
    output.float().square().mean().backward()

    assert output.dtype == torch.bfloat16
    assert wrapper.lora_A.dtype == torch.float32
    assert wrapper.lora_B.dtype == torch.float32
    assert wrapper.lora_A.grad is not None
    assert wrapper.lora_B.grad is not None
    assert wrapper.lora_A.grad.dtype == torch.float32
    assert wrapper.lora_B.grad.dtype == torch.float32
    assert bool(torch.isfinite(wrapper.lora_A.grad).all())
    assert bool(torch.isfinite(wrapper.lora_B.grad).all())
    assert float(wrapper.lora_B.grad.abs().sum().item()) > 0.0
    assert wrapper.base.weight.grad is None
