"""Independent WAM-OPD defaults and explicit legacy configuration compatibility."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("settings,expected", [
    ({}, None),
    ({"WAM_OPD_RUNTIME_ROOT": "/tmp/wam-runtime"}, "/tmp/wam-runtime"),
    ({"WAVE_RL_ROOT": "/tmp/legacy-runtime"}, "/tmp/legacy-runtime"),
    ({"PROJECT_ROOT": "/tmp/older-runtime"}, "/tmp/older-runtime"),
    ({"WAM_OPD_RUNTIME_ROOT": "/tmp/wam-runtime", "WAVE_RL_ROOT": "/tmp/legacy-runtime"}, "/tmp/wam-runtime"),
])
def test_runtime_paths(settings, expected):
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("WAM_OPD_") and k not in {"WAVE_RL_ROOT", "PROJECT_ROOT"}}
    result = subprocess.run(
        [sys.executable, "-c",
         "import json; from experiments import paths; "
         "print(json.dumps([str(paths.RUNTIME_ROOT), str(paths.PYTHON_BIN)]))"],
        cwd=root, env={**env, **settings}, capture_output=True, text=True, check=True,
    )
    runtime, python = json.loads(result.stdout)
    assert runtime == str(Path(expected).resolve() if expected else root)
    assert python == str(Path(sys.executable).absolute())


def test_python_override_preserves_venv_symlink(tmp_path):
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    result = subprocess.run(
        [sys.executable, "-c", "from experiments.paths import PYTHON_BIN; print(PYTHON_BIN)"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "WAM_OPD_PYTHON_BIN": str(interpreter)},
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == str(interpreter)
