"""A workspace scaffold is not a successfully installed model runtime."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_bootstrap_reports_missing_external_trees(tmp_path):
    # Simulate successful Git commands that leave incomplete source trees.
    binary = tmp_path / "bin"
    binary.mkdir()
    git = binary / "git"
    git.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "if sys.argv[1] == 'clone':\n"
        "    (pathlib.Path(sys.argv[-1]) / 'third_party').mkdir(parents=True)\n"
    )
    git.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(root / "scripts/bootstrap_dependencies.sh")],
        env={**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
             "WAM_OPD_RUNTIME_ROOT": str(tmp_path / "wam-runtime")},
        capture_output=True, text=True,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "INCOMPLETE" in result.stderr
    assert "lingbot-va" in result.stderr
    assert "RoboTwin-lingbot-native" in result.stderr


def test_runtime_patch_is_versioned_and_covers_required_seams():
    patch = Path(__file__).resolve().parents[1] / "patches/lingbot-va/0001-wam-opd-portability-and-runtime.patch"
    text = patch.read_text(encoding="utf-8")
    for marker in (
        "evaluation/robotwin/eval_polict_client_openpi.py",
        "wan_va/modules/model.py",
        "wan_va/wan_va_server.py",
        "LINGBOT_VA_DISABLE_FLEX_COMPILE",
        "LINGBOT_VA_ROBOTWIN_CKPT",
    ):
        assert marker in text


@pytest.mark.parametrize("source", ["lingbot-va", "RoboTwin-lingbot-native"])
def test_bootstrap_preserves_existing_sources(tmp_path, source):
    existing = tmp_path / "third_party" / source
    existing.mkdir(parents=True)
    marker = existing / "local-patch.txt"
    marker.write_text("keep this source patch")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(root / "scripts/bootstrap_dependencies.sh")],
        env={**os.environ, "WAM_OPD_RUNTIME_ROOT": str(tmp_path)},
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "refusing to overwrite" in result.stderr
    assert marker.read_text() == "keep this source patch"
    assert not (tmp_path / "third_party" / (
        "RoboTwin-lingbot-native" if source == "lingbot-va" else "lingbot-va"
    )).exists()
