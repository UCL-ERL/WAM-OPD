"""Atomic receipt writes for the success-path eval.

A truncated receipt (e.g. process killed mid-write) makes the pipeline
classify the phase as started-but-incomplete and refuse to rerun forever, so
the eval ``_write_json`` must leave no partial file behind.  These tests run
without a model runtime and without torch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.run_handover_mic_success_path_eval import _write_json


class AtomicReceiptWriteTest(unittest.TestCase):
    def test_write_publishes_complete_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            _write_json(target, {"status": "PASS", "items": [1, 2]})
            self.assertTrue(target.is_file())
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"status": "PASS", "items": [1, 2]},
            )
            # No temp residue should remain after a successful write.
            self.assertEqual(
                [p.name for p in Path(directory).iterdir()],
                ["receipt.json"],
            )

    def test_failed_replace_removes_temp_and_leaves_no_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            with mock.patch(
                "experiments.run_handover_mic_success_path_eval.os.replace",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaises(OSError):
                    _write_json(target, {"status": "PASS"})
            # The target must never appear, and the temp file must be cleaned up.
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_failure_before_rename_leaves_no_target(self) -> None:
        from experiments import run_handover_mic_success_path_eval as eval_module

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            original_mkstemp = eval_module.tempfile.mkstemp

            def failing_mkstemp(*args, **kwargs):
                fd, name = original_mkstemp(*args, **kwargs)
                # Simulate a failure after the temp file is created but before
                # the rename by closing the descriptor early.
                os.close(fd)
                raise OSError("interrupted")

            with mock.patch(
                "experiments.run_handover_mic_success_path_eval.tempfile.mkstemp",
                side_effect=failing_mkstemp,
            ):
                with self.assertRaises(OSError):
                    _write_json(target, {"status": "PASS"})
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
