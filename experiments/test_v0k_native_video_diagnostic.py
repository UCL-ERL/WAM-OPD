"""Unit tests for the restored V0K diagnostic video recording module.

The module was referenced by the eval and history-capture entry points but
never shipped.  These tests pin the public contract: real MP4 output through
OpenCV when available, graceful empty-video finalization, and strict
double-finalize protection.

OpenCV is a runtime (not CI dev) dependency, so the video-output tests skip
on minimal installs; the pure contract assertions always run.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from experiments import v0k_native_video_diagnostic as MODULE


def _frame(value: int, shape: tuple[int, int] = (16, 24)) -> dict[str, np.ndarray]:
    return {
        "observation.images.head": np.full(
            (shape[0], shape[1], 3), value, dtype=np.uint8
        )
    }


def _collector(root: Path, **kwargs) -> MODULE.RecordingCollector:
    defaults = dict(task="move_stapler_pad", seed=7, condition="epoch_01")
    defaults.update(kwargs)
    return MODULE.RecordingCollector(root=root, **defaults)


def test_camera_registry_has_native_observation_keys() -> None:
    assert "observation.images.head" in MODULE.CAMERAS
    assert "observation.images.left" in MODULE.CAMERAS
    assert "observation.images.right" in MODULE.CAMERAS
    # The eval patches an observer camera onto the base registry.
    merged = {**MODULE.CAMERAS, "recording.images.observer": "observer_camera"}
    assert merged["recording.images.observer"] == "observer_camera"


def test_finalize_empty_episode_still_publishes_a_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        collector = _collector(Path(tmp))
        payload = collector.finalize({"success": False})
        assert payload["schema"] == "waopd_v0k_diagnostic_recording_v1"
        assert payload["status"] == "PASS"
        assert payload["frames"] == 0
        assert payload["dropped_frames"] == 0
        assert payload["composite_mp4_exists"] is True
        assert Path(payload["composite_mp4"]).is_file()
        assert Path(payload["video_root"]).is_dir()


def test_finalize_records_error_as_partial() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        payload = _collector(Path(tmp)).finalize({}, error="boom")
        assert payload["status"] == "PARTIAL"
        assert payload["error"] == "boom"


def test_double_finalize_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        collector = _collector(Path(tmp))
        collector.finalize({})
        with pytest.raises(RuntimeError):
            collector.finalize({})


def test_unknown_camera_drops_frames_without_failing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        collector = _collector(
            Path(tmp), camera_key="observation.images.missing"
        )
        collector.observation_callback({"observation": _frame(1)})
        payload = collector.finalize({})
        assert payload["dropped_frames"] == 1
        assert payload["frames"] == 0


def test_shape_mismatch_drops_the_divergent_frame() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        collector = _collector(Path(tmp))
        collector.observation_callback({"observation": _frame(1, (16, 24))})
        collector.observation_callback({"observation": _frame(2, (32, 48))})
        payload = collector.finalize({})
        assert payload["frames"] == 1
        assert payload["dropped_frames"] == 1


def test_disabled_collector_does_not_require_video_write() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        collector = _collector(Path(tmp), enabled=False)
        collector.observation_callback({"observation": _frame(1)})
        payload = collector.finalize({})
        assert payload["enabled"] is False
        assert payload["frames"] == 0


@pytest.mark.skipif(
    MODULE.cv2 is None,
    reason="openCV is a runtime-only dependency not installed in CI",
)
def test_frames_are_written_to_a_real_readable_mp4() -> None:
    import cv2

    with tempfile.TemporaryDirectory() as tmp:
        collector = _collector(Path(tmp))
        for index in range(5):
            collector.observation_callback(
                {"observation": _frame(index), "control_step": index}
            )
        payload = collector.finalize({"success": True})
        assert payload["frames"] == 5
        assert payload["composite_mp4_size_bytes"] > 0
        capture = cv2.VideoCapture(str(payload["composite_mp4"]))
        try:
            assert capture.isOpened()
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
        finally:
            capture.release()
