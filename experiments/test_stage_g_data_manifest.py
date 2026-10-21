"""Unit tests for Stage G policy/label identity content hashing.

The identity keys for policy versions and Teacher labels must reflect the
*contents* of the checkpoint and teacher files, not just their paths, so that
replacing weights at the same path invalidates stale keys instead of silently
reusing them.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest import mock

from experiments.stage_g_data_manifest import (
    build_label_key,
    build_policy_version,
)


class PolicyVersionContentIdentityTest(unittest.TestCase):
    def test_rewriting_base_at_same_path_changes_released_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "student.pt"
            base.write_bytes(b"weights-v1")
            first = build_policy_version(base)
            base.write_bytes(b"weights-v2")
            second = build_policy_version(base)
            self.assertEqual(first[:9], "released-")
            self.assertNotEqual(first, second)

    def test_unchanged_base_yields_stable_released_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "student.pt"
            base.write_bytes(b"weights-v1")
            self.assertEqual(
                build_policy_version(base), build_policy_version(base)
            )

    def test_rewriting_base_at_same_path_changes_adapter_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "student.pt"
            delta = Path(directory) / "delta.pt"
            base.write_bytes(b"weights-v1")
            delta.write_bytes(b"delta-v1")
            first = build_policy_version(base, delta)
            base.write_bytes(b"weights-v2")
            second = build_policy_version(base, delta)
            self.assertEqual(first[:8], "adapter-")
            self.assertNotEqual(first, second)

    def test_rewriting_delta_at_same_path_changes_adapter_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "student.pt"
            delta = Path(directory) / "delta.pt"
            base.write_bytes(b"weights-v1")
            delta.write_bytes(b"delta-v1")
            first = build_policy_version(base, delta)
            delta.write_bytes(b"delta-v2")
            second = build_policy_version(base, delta)
            self.assertNotEqual(first, second)

    def test_missing_base_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "nope.pt"
            with self.assertRaises(FileNotFoundError):
                build_policy_version(missing)

    def test_missing_delta_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "student.pt"
            base.write_bytes(b"weights-v1")
            missing = Path(directory) / "nope.pt"
            with self.assertRaises(FileNotFoundError):
                build_policy_version(base, missing)


class LabelKeyTeacherContentIdentityTest(unittest.TestCase):
    @staticmethod
    def _fake_noise() -> object:
        # tensor_sha256 is stubbed below so no torch dependency is needed.
        return object()

    def test_rewriting_teacher_at_same_path_changes_label_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            teacher = Path(directory) / "teacher.pt"
            teacher.write_bytes(b"teacher-v1")
            with mock.patch(
                "experiments.stage_g_data_manifest.tensor_sha256",
                return_value="noise-hash",
            ):
                first = build_label_key(
                    state_key="state-x",
                    policy_version="adapter-abc",
                    action_base_noise=self._fake_noise(),
                    teacher_transformer=teacher,
                )
                teacher.write_bytes(b"teacher-v2")
                second = build_label_key(
                    state_key="state-x",
                    policy_version="adapter-abc",
                    action_base_noise=self._fake_noise(),
                    teacher_transformer=teacher,
                )
            self.assertEqual(first[:6], "label-")
            self.assertNotEqual(first, second)

    def test_unchanged_teacher_yields_stable_label_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            teacher = Path(directory) / "teacher.pt"
            teacher.write_bytes(b"teacher-v1")
            with mock.patch(
                "experiments.stage_g_data_manifest.tensor_sha256",
                return_value="noise-hash",
            ):
                first = build_label_key(
                    state_key="state-x",
                    policy_version="adapter-abc",
                    action_base_noise=self._fake_noise(),
                    teacher_transformer=teacher,
                )
                second = build_label_key(
                    state_key="state-x",
                    policy_version="adapter-abc",
                    action_base_noise=self._fake_noise(),
                    teacher_transformer=teacher,
                )
            self.assertEqual(first, second)

    def test_missing_teacher_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "nope.pt"
            with mock.patch(
                "experiments.stage_g_data_manifest.tensor_sha256",
                return_value="noise-hash",
            ):
                with self.assertRaises(FileNotFoundError):
                    build_label_key(
                        state_key="state-x",
                        policy_version="adapter-abc",
                        action_base_noise=self._fake_noise(),
                        teacher_transformer=missing,
                    )


if __name__ == "__main__":
    unittest.main()
