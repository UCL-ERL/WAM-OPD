"""A workspace scaffold is not a successfully installed model runtime."""

import os
from pathlib import Path
import subprocess
import sys


def test_bootstrap_reports_missing_external_trees(tmp_path):
    # Simulate the pinned wave-rl tree: third_party contains only a placeholder.
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
             "WAVE_RL_ROOT": str(tmp_path / "wave")},
        capture_output=True, text=True,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "INCOMPLETE" in result.stderr
    assert "lingbot-va" in result.stderr
    assert "RoboTwin-lingbot-native" in result.stderr
