"""Exact observation contract for Stage H prefix reconstruction."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _update_digest(digest: Any, key: str, value: Any) -> None:
    digest.update(key.encode("utf-8"))
    if isinstance(value, str):
        digest.update(b"str\0")
        digest.update(value.encode("utf-8"))
        return
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())


def formatted_observation_sha256(observation: dict[str, Any]) -> str:
    """Hash every formatted-observation field including representation metadata."""

    digest = hashlib.sha256()
    for key in sorted(observation):
        _update_digest(digest, key, observation[key])
    return digest.hexdigest()


def compare_formatted_observations(
    expected: dict[str, Any], actual: dict[str, Any]
) -> dict[str, Any]:
    """Return an exact, field-level comparison of two formatted observations."""

    mismatches: list[str] = []
    array_metrics: dict[str, dict[str, float]] = {}
    expected_keys = set(expected)
    actual_keys = set(actual)
    for key in sorted(expected_keys - actual_keys):
        mismatches.append(f"{key}: missing from actual")
    for key in sorted(actual_keys - expected_keys):
        mismatches.append(f"{key}: unexpected in actual")
    for key in sorted(expected_keys & actual_keys):
        left = expected[key]
        right = actual[key]
        if isinstance(left, str) or isinstance(right, str):
            if left != right:
                mismatches.append(f"{key}: values differ")
            continue
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if left_array.dtype != right_array.dtype:
            mismatches.append(
                f"{key}: dtype differs ({left_array.dtype} != {right_array.dtype})"
            )
        elif left_array.shape != right_array.shape:
            mismatches.append(
                f"{key}: shape differs ({left_array.shape} != {right_array.shape})"
            )
        else:
            difference = np.abs(
                left_array.astype(np.float64) - right_array.astype(np.float64)
            )
            array_metrics[key] = {
                "mean_abs": float(difference.mean()),
                "max_abs": float(difference.max(initial=0.0)),
            }
            if np.any(difference):
                mismatches.append(f"{key}: values differ")
    nonvisual_mismatches = [
        item
        for item in mismatches
        if not item.startswith("observation.images.")
    ]
    return {
        "exact": not mismatches,
        "nonvisual_exact": not nonvisual_mismatches,
        "mismatches": mismatches,
        "array_metrics": array_metrics,
        "sha256": formatted_observation_sha256(actual),
        "expected_sha256": formatted_observation_sha256(expected),
    }
