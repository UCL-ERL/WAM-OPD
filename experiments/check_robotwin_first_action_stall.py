"""Detect a live RoboTwin client stalled after policy reset.

Exit status 1 is the red signal: reset completed, no first action followed,
the log is older than the configured threshold, and the client is still live.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import time


RESET_MARKER = "policy reset done"
FIRST_ACTION_MARKER = "policy first action start"


@dataclass(frozen=True)
class StallVerdict:
    status: str
    reason: str
    client_pid: int
    client_alive: bool
    reset_timestamp: str | None
    first_action_timestamp: str | None
    seconds_since_reset: float | None
    seconds_since_log_update: float
    threshold_seconds: float


def _line_timestamp(line: str) -> datetime | None:
    prefix = "[robotwin-eval] "
    if not line.startswith(prefix):
        return None
    value = line[len(prefix) :].split(" ", 1)[0]
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def check_stall(
    *,
    log_path: Path,
    client_pid: int,
    threshold_seconds: float,
    now_epoch: float | None = None,
    proc_root: Path = Path("/proc"),
) -> StallVerdict:
    now_epoch = time.time() if now_epoch is None else float(now_epoch)
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    reset_index = None
    reset_time = None
    for index, line in enumerate(lines):
        if RESET_MARKER in line:
            reset_index = index
            reset_time = _line_timestamp(line)
    first_action_time = None
    if reset_index is not None:
        for line in lines[reset_index + 1 :]:
            if FIRST_ACTION_MARKER in line:
                first_action_time = _line_timestamp(line)
                break

    client_alive = (proc_root / str(client_pid)).exists()
    log_age = max(0.0, now_epoch - log_path.stat().st_mtime)
    reset_age = None
    if reset_time is not None:
        reset_age = max(0.0, now_epoch - reset_time.timestamp())

    if reset_index is None:
        status = "NOT_STALLED"
        reason = "policy reset has not completed"
    elif first_action_time is not None:
        status = "NOT_STALLED"
        reason = "policy first action started after the latest reset"
    elif not client_alive:
        status = "INCOMPLETE_EXIT"
        reason = "client exited before the first action"
    elif reset_age is None:
        status = "INDETERMINATE"
        reason = "latest reset marker has no parseable timestamp"
    elif reset_age < threshold_seconds or log_age < threshold_seconds:
        status = "PENDING"
        reason = "latest reset/log update is still inside the grace period"
    else:
        status = "STALLED"
        reason = "live client exceeded the first-action grace period"

    return StallVerdict(
        status=status,
        reason=reason,
        client_pid=client_pid,
        client_alive=client_alive,
        reset_timestamp=(reset_time.isoformat() if reset_time else None),
        first_action_timestamp=(
            first_action_time.isoformat() if first_action_time else None
        ),
        seconds_since_reset=reset_age,
        seconds_since_log_update=log_age,
        threshold_seconds=float(threshold_seconds),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--client-pid", type=int, required=True)
    parser.add_argument("--threshold-seconds", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    verdict = check_stall(
        log_path=args.log,
        client_pid=args.client_pid,
        threshold_seconds=args.threshold_seconds,
    )
    payload = asdict(verdict)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="", flush=True)
    raise SystemExit(1 if verdict.status == "STALLED" else 0)


if __name__ == "__main__":
    main()
