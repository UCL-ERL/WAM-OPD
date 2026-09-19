"""Exercise the checker after its own source has been committed to Git."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def repository(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = Path(__file__).resolve().parents[1] / "scripts/check_repository_hygiene.py"
    shutil.copy2(source, scripts / source.name)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    return tmp_path


def check(root):
    return subprocess.run(
        [sys.executable, str(root / "scripts/check_repository_hygiene.py")],
        cwd=root, capture_output=True, text=True,
    )


def test_tracked_checker_does_not_reject_its_rule_definitions(repository):
    result = check(repository)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name,content", [
    ("launch.py", "root = " + repr("/" + "Users/researcher/checkpoint")),
    ("weights.pt", "not a real checkpoint"),
])
def test_rejects_tracked_private_path_or_artifact(repository, name, content):
    (repository / name).write_text(content)
    subprocess.run(["git", "add", name], cwd=repository, check=True)
    result = check(repository)
    assert result.returncode == 1
    assert name in result.stdout
