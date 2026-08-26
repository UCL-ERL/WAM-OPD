"""Select one task from preregistered strict Teacher/Student gap gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_decision(root: Path, task: str) -> tuple[dict[str, Any], str]:
    path = root / task / "decision.json"
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def select_task(root: Path, tasks: list[str]) -> dict[str, Any]:
    rows = []
    for task in tasks:
        decision, digest = _load_decision(root, task)
        if decision.get("task_id") != task:
            raise ValueError(f"task mismatch for {task}: {decision.get('task_id')!r}")
        rows.append({**decision, "decision_sha256": digest})

    passing = [row for row in rows if row.get("status") == "PASS"]
    passing.sort(
        key=lambda row: (
            -int(row["paired_net_advantage"]),
            -int(row["teacher_successes"]),
            int(row["student_successes"]),
            str(row["task_id"]),
        )
    )
    selected = passing[0] if passing else None
    return {
        "schema": "waopd_new_task_strict_gap_selection_v1",
        "status": "PASS" if selected is not None else "FAIL",
        "selection_rule": [
            "status_pass",
            "paired_net_advantage_desc",
            "teacher_successes_desc",
            "student_successes_asc",
            "task_id_asc",
        ],
        "selected_task": None if selected is None else selected["task_id"],
        "selected_decision": selected,
        "decisions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scout-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = select_task(args.scout_root, args.tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(selection, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
