"""Portable filesystem defaults for WAM-OPD.

Large models and experiment outputs intentionally live outside the Git
checkout. Every default can be overridden with an environment variable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


REPO_ROOT = _path("WAM_OPD_ROOT", Path(__file__).resolve().parents[1])
RUNTIME_ROOT = _path(
    "WAM_OPD_RUNTIME_ROOT",
    Path(os.environ.get("WAVE_RL_ROOT", os.environ.get("PROJECT_ROOT", str(REPO_ROOT)))),
)
# Explicit old environment settings remain readable for existing manifests.
# New installations need only WAM-OPD and its direct upstream source trees.
WAVE_RL_ROOT = RUNTIME_ROOT
ARTIFACT_ROOT = _path("WAM_OPD_ARTIFACT_ROOT", REPO_ROOT / ".artifacts")
CONFIG_ROOT = _path("WAM_OPD_CONFIG_ROOT", REPO_ROOT / "configs" / "generated")
OUTPUT_ROOT = _path("WAM_OPD_OUTPUT_ROOT", ARTIFACT_ROOT / "experiments")
SCRATCH_ROOT = _path("WAM_OPD_SCRATCH_ROOT", ARTIFACT_ROOT / "scratch")
STUDENT_ROOT = _path(
    "WAM_OPD_STUDENT_ROOT", ARTIFACT_ROOT / "models" / "FlashWAM-RoboTwin"
)
TEACHER_ROOT = _path(
    "WAM_OPD_TEACHER_ROOT",
    ARTIFACT_ROOT / "models" / "lingbot-va-posttrain-robotwin",
)
SOURCE_SWEEP = _path(
    "WAM_OPD_SOURCE_SWEEP",
    ARTIFACT_ROOT / "inputs" / "robotwin_native_sweep" / "raw" / "demo_clean",
)
PYTHON_BIN = Path(os.environ.get("WAM_OPD_PYTHON_BIN", sys.executable)).expanduser().absolute()

# A lightweight checkout may not have the external runtime yet. Callers that
# only validate orchestration can explicitly choose the current interpreter.
# Do not resolve interpreter symlinks: that can silently escape an active venv.
CURRENT_PYTHON = Path(sys.executable).absolute()
