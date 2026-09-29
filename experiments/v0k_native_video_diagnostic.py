"""Diagnostic video recording for native WAM-OPD evaluation units.

The eval and history-capture entry points reference this module for
per-unit video evidence, but it was never shipped with the public checkout.
This implementation restores the missing ``RecordingCollector`` seam with a
real OpenCV-backed writer so ``--record`` / ``--record-video`` produces an
actual ``composite.mp4`` under ``<root>/video/``.

The collector is deliberately a lightweight observer.  It never feeds state
back into the policy, the simulator, or the evaluator; frames come only from
the observation callback.  A unit with no camera frames still finalizes into
a valid (empty) MP4 so the eval resumability contract
(``run_handover_mic_success_path_eval._unit_complete``) is never blocked by
the absence of renderer output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

try:  # The runtime dependency set ships OpenCV.
    import cv2
except Exception as exc:  # pragma: no cover - exercised only on minimal installs
    cv2 = None
    _CV2_IMPORT_ERROR = str(exc)


# Camera registry patched by ``run_handover_mic_success_path_eval`` to add the
# observer camera.  Default keys are the native formatted-observation image
# fields that exist without any patch.
CAMERAS: dict[str, str] = {
    "observation.images.head": "head_camera",
    "observation.images.left": "left_camera",
    "observation.images.right": "right_camera",
}


def _image_from_observation(observation: Mapping[str, Any], key: str) -> np.ndarray | None:
    """Return an HxWx3 uint8 frame for ``observation[key]``, or None."""

    if not isinstance(observation, Mapping):
        return None
    value = observation.get(key)
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3:
        return None
    if array.dtype != np.uint8:
        try:
            array = array.astype(np.uint8)
        except (ValueError, TypeError):
            return None
    return np.ascontiguousarray(array)


class RecordingCollector:
    """Buffer observation frames and publish one MP4 at finalize time.

    Frames are appended in callback order with no reordering, so the video
    chronology matches the trace exactly.  The writer is created lazily at
    finalize; a unit that emits no frames produces an empty MP4 rather than a
    missing file.
    """

    def __init__(
        self,
        *,
        root: Path,
        task: str,
        seed: int,
        condition: str,
        enabled: bool = True,
        fps: float = 10.0,
        codec: str = "mp4v",
        camera_key: str = "observation.images.head",
    ) -> None:
        self.root = Path(root)
        self.task = str(task)
        self.seed = int(seed)
        self.condition = str(condition)
        self.enabled = bool(enabled)
        self.fps = float(fps)
        self.codec = str(codec)
        self.camera_key = str(camera_key)
        self._frames: list[np.ndarray] = []
        self._frame_count = 0
        self._dropped_frames = 0
        self._first_frame_shape: tuple[int, int, int] | None = None
        self._started_at_unix_ns: int | None = None
        self._finalized = False

    def macro_callback(self, event: Mapping[str, Any]) -> None:
        # The macro boundary is retained in the summary only; frames are
        # collected from the observation callback where control steps exist.
        return None

    def observation_callback(self, event: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        if self._finalized:
            raise RuntimeError("RecordingCollector already finalized")
        if self._started_at_unix_ns is None:
            self._started_at_unix_ns = datetime.now(timezone.utc).timestamp() * 1_000_000_000
        frame = _image_from_observation(
            event.get("observation") if isinstance(event, Mapping) else None,
            self.camera_key,
        )
        if frame is None:
            self._dropped_frames += 1
            return
        if self._first_frame_shape is None:
            self._first_frame_shape = tuple(frame.shape)
        elif tuple(frame.shape) != self._first_frame_shape:
            self._dropped_frames += 1
            return
        self._frames.append(frame)
        self._frame_count += 1

    def _write_mp4(self, video_root: Path) -> Path:
        """Write the buffered frames to ``<video_root>/composite.mp4``."""

        if cv2 is None:
            raise RuntimeError(
                "v0k RecordingCollector requires opencv-python: "
                f"{_CV2_IMPORT_ERROR}"
            )
        video_root.mkdir(parents=True, exist_ok=True)
        output = video_root / "composite.mp4"
        # A fresh temp name avoids exposing a partial final file if the
        # process dies mid-write; rename is atomic on the same filesystem.
        # Note: OpenCV backends reject dot-prefixed (hidden) filenames, so the
        # temp name uses an ordinary prefix.
        temporary = video_root / f"{output.stem}.{id(self):x}.tmp{output.suffix}"
        try:
            if self._frames:
                height, width = self._first_frame_shape[:2]
                writer = cv2.VideoWriter(
                    str(temporary),
                    cv2.VideoWriter_fourcc(*self.codec),
                    float(self.fps),
                    (int(width), int(height)),
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"OpenCV VideoWriter failed to open {temporary}"
                    )
                try:
                    for frame in self._frames:
                        writer.write(frame)
                finally:
                    writer.release()
                # VideoWriter release is not guaranteed to flush the file to
                # disk synchronously on every backend; wait briefly for the
                # temp file to appear before the atomic rename.
                for _ in range(200):
                    if temporary.exists() and temporary.stat().st_size > 0:
                        break
                    time.sleep(0.01)
                if not temporary.exists():
                    raise RuntimeError(
                        f"OpenCV VideoWriter produced no file: {temporary}"
                    )
            else:
                temporary.write_bytes(b"")
            temporary.replace(output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return output

    def finalize(
        self,
        result: Mapping[str, Any] | None = None,
        *,
        error: str | None = None,
    ) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("RecordingCollector already finalized")
        result = dict(result) if isinstance(result, Mapping) else {}
        video_root = self.root / "video"
        if self.enabled:
            output = self._write_mp4(video_root)
        else:
            output = video_root / "composite.mp4"
        payload = {
            "schema": "waopd_v0k_diagnostic_recording_v1",
            "task": str(self.task),
            "seed": int(self.seed),
            "condition": str(self.condition),
            "enabled": bool(self.enabled),
            "video_root": str(video_root.resolve()),
            "composite_mp4": str(output.resolve()),
            "composite_mp4_exists": bool(output.is_file()),
            "composite_mp4_size_bytes": int(output.stat().st_size) if output.is_file() else 0,
            "frames": int(self._frame_count),
            "dropped_frames": int(self._dropped_frames),
            "fps": float(self.fps),
            "codec": str(self.codec),
            "camera_key": str(self.camera_key),
            "started_at_unix_ns": self._started_at_unix_ns,
            "status": "PASS" if error is None else "PARTIAL",
            "error": error,
        }
        self._finalized = True
        self._frames.clear()
        return payload


__all__ = ["CAMERAS", "RecordingCollector"]
