"""Compare isolated Single-GPU and two-rank success-path pilot outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch


def _close(left: float, right: float, *, atol: float, rtol: float) -> bool:
    return math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol)


def _tensor_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    delta = left.detach().double() - right.detach().double()
    reference = left.detach().double()
    delta_l2 = float(delta.square().sum().sqrt().item())
    reference_l2 = float(reference.square().sum().sqrt().item())
    return {
        "max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "relative_l2": delta_l2 / max(reference_l2, 1e-30),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-dir", type=Path, required=True)
    parser.add_argument("--distributed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scalar-atol", type=float, default=1e-7)
    parser.add_argument("--scalar-rtol", type=float, default=2e-5)
    parser.add_argument("--tensor-max-abs", type=float, default=5e-6)
    parser.add_argument("--tensor-relative-l2", type=float, default=2e-5)
    args = parser.parse_args()

    single = json.loads((args.single_dir / "result.json").read_text())
    distributed = json.loads((args.distributed_dir / "result.json").read_text())
    single_state = torch.load(
        args.single_dir / "final_state.pt", map_location="cpu", weights_only=True
    )
    distributed_state = torch.load(
        args.distributed_dir / "final_state.pt", map_location="cpu", weights_only=True
    )

    sample_order_equal = single["sample_ids"] == distributed["sample_ids"]
    step_count_equal = single["optimizer_steps"] == distributed["optimizer_steps"]
    batch_size_equal = (
        single["global_effective_batch_size"]
        == distributed["global_effective_batch_size"]
    )
    step_diagnostics: list[dict[str, Any]] = []
    scalar_equivalent = len(single["steps"]) == len(distributed["steps"])
    for left, right in zip(single["steps"], distributed["steps"], strict=False):
        fields = (
            "video_loss",
            "action_loss",
            "action_fm_loss",
            "gradient_norm_pre_clip",
            "batch_weight",
        )
        comparisons = {
            field: {
                "single": float(left[field]),
                "distributed": float(right[field]),
                "close": _close(
                    left[field],
                    right[field],
                    atol=args.scalar_atol,
                    rtol=args.scalar_rtol,
                ),
            }
            for field in fields
        }
        scalar_equivalent = scalar_equivalent and all(
            item["close"] for item in comparisons.values()
        )
        step_diagnostics.append(
            {"step_id": int(left["step_id"]), "comparisons": comparisons}
        )

    adapter_metrics: dict[str, dict[str, float]] = {}
    adapter_equivalent = True
    for name, left in single_state["adapter_state"].items():
        metrics = _tensor_metrics(left, distributed_state["adapter_state"][name])
        adapter_metrics[name] = metrics
        adapter_equivalent = adapter_equivalent and (
            metrics["max_abs"] <= args.tensor_max_abs
            and metrics["relative_l2"] <= args.tensor_relative_l2
        )

    optimizer_equivalent = True
    optimizer_worst = {"max_abs": 0.0, "relative_l2": 0.0}
    left_optimizer = single_state["optimizer_state"]["state"]
    right_optimizer = distributed_state["optimizer_state"]["state"]
    if sorted(left_optimizer) != sorted(right_optimizer):
        optimizer_equivalent = False
    else:
        for key in sorted(left_optimizer):
            for name, left in left_optimizer[key].items():
                right = right_optimizer[key][name]
                if isinstance(left, torch.Tensor):
                    metrics = _tensor_metrics(left, right)
                    optimizer_worst["max_abs"] = max(
                        optimizer_worst["max_abs"], metrics["max_abs"]
                    )
                    optimizer_worst["relative_l2"] = max(
                        optimizer_worst["relative_l2"], metrics["relative_l2"]
                    )
                    optimizer_equivalent = optimizer_equivalent and (
                        metrics["max_abs"] <= args.tensor_max_abs
                        and metrics["relative_l2"] <= args.tensor_relative_l2
                    )
                else:
                    optimizer_equivalent = optimizer_equivalent and left == right

    rank_consistent = all(
        payload is not None
        and payload["final_adapter_hash"] == distributed["final_adapter_hash"]
        for payload in distributed["rank_payloads"]
    )
    fixed_input_equal = (
        single["fixed_package"]["source_config_sha256"]
        == distributed["fixed_package"]["source_config_sha256"]
        and single["fixed_package"]["artifacts"]
        == distributed["fixed_package"]["artifacts"]
    )
    speedup = float(single["elapsed_seconds"]) / float(
        distributed["elapsed_seconds"]
    )
    passed = all(
        (
            sample_order_equal,
            step_count_equal,
            batch_size_equal,
            scalar_equivalent,
            adapter_equivalent,
            optimizer_equivalent,
            rank_consistent,
            fixed_input_equal,
        )
    )
    summary = {
        "schema": "waopd_success_path_multigpu_equivalence_verdict_v1",
        "status": "PASS" if passed else "FAIL",
        "equivalence": {
            "sample_order_equal": sample_order_equal,
            "optimizer_step_count_equal": step_count_equal,
            "global_effective_batch_size_equal": batch_size_equal,
            "per_step_scalar_equivalent": scalar_equivalent,
            "adapter_state_equivalent": adapter_equivalent,
            "optimizer_state_equivalent": optimizer_equivalent,
            "distributed_ranks_identical": rank_consistent,
            "fixed_input_equal_and_read_only": fixed_input_equal,
        },
        "tolerances": {
            "scalar_atol": args.scalar_atol,
            "scalar_rtol": args.scalar_rtol,
            "tensor_max_abs": args.tensor_max_abs,
            "tensor_relative_l2": args.tensor_relative_l2,
        },
        "timing": {
            "single_seconds": float(single["elapsed_seconds"]),
            "distributed_seconds": float(distributed["elapsed_seconds"]),
            "speedup": speedup,
        },
        "hashes": {
            "single_adapter": single["final_adapter_hash"],
            "distributed_adapter": distributed["final_adapter_hash"],
            "exact_hash_match": (
                single["final_adapter_hash"] == distributed["final_adapter_hash"]
            ),
        },
        "optimizer_worst_error": optimizer_worst,
        "adapter_worst_error": {
            field: max(metrics[field] for metrics in adapter_metrics.values())
            for field in ("max_abs", "relative_l2")
        },
        "step_diagnostics": step_diagnostics,
    }
    _write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
