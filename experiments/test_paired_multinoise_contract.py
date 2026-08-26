from __future__ import annotations

from pathlib import Path

from experiments.run_open_microwave_paired_multinoise import (
    _episode_is_complete,
    _validate_pair,
    _write_json,
)


def _episode(
    *, student: Path, adapter: Path | None, success: bool
) -> dict[str, object]:
    return {
        "status": "PASS",
        "task": "move_stapler_pad",
        "task_config": "demo_clean",
        "seed": 10002,
        "prompt": "move stapler",
        "chunks": 13,
        "max_control_steps": 400,
        "student_checkpoint": str(student.resolve()),
        "adapter_state": None if adapter is None else str(adapter.resolve()),
        "success": success,
        "runtime_nfe": {"video": 1, "action": 1},
        "teacher_loaded": False,
        "teacher_called": False,
        "training_started": False,
        "prompt_hash": "prompt",
        "initial_snapshot_sha256": "snapshot",
        "student_checkpoint_sha256": "student",
        "noise_contract": {
            "student_base_seed": 20260820,
            "chunk_action_noise_hashes": ["a0", "a1"],
            "chunk_video_noise_hashes": ["v0", "v1"],
        },
    }


def test_episode_reuse_requires_exact_adapter_identity(tmp_path: Path) -> None:
    student = tmp_path / "student"
    adapter = tmp_path / "adapter.pt"
    output = tmp_path / "episode.json"
    _write_json(output, _episode(student=student, adapter=adapter, success=True))

    kwargs = {
        "task": "move_stapler_pad",
        "task_config": "demo_clean",
        "seed": 10002,
        "prompt": "move stapler",
        "chunks": 13,
        "max_control_steps": 400,
        "noise_base_seed": 20260820,
        "student": student,
    }
    assert _episode_is_complete(output, adapter=adapter, **kwargs)
    assert not _episode_is_complete(
        output, adapter=tmp_path / "different.pt", **kwargs
    )


def test_validate_pair_is_task_generic_and_detects_rescue(tmp_path: Path) -> None:
    student = tmp_path / "student"
    adapter = tmp_path / "adapter.pt"
    released_path = tmp_path / "released.json"
    opd_path = tmp_path / "opd.json"
    _write_json(
        released_path, _episode(student=student, adapter=None, success=False)
    )
    _write_json(opd_path, _episode(student=student, adapter=adapter, success=True))

    row = _validate_pair(
        released_path=released_path,
        opd_path=opd_path,
        seed=10002,
        prompt="move stapler",
        noise_base_seed=20260820,
        task="move_stapler_pad",
        task_config="demo_clean",
        chunks=13,
        max_control_steps=400,
        student=student,
        adapter=adapter,
    )

    assert row["task"] == "move_stapler_pad"
    assert row["rescue"] is True
    assert row["regression"] is False
