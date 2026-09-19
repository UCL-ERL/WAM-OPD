#!/usr/bin/env python3
"""Check that the public checkout does not contain private runtime paths or artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path


PRIVATE_PATH_MARKERS = (
    "/home/zwwl_user/",
    "/ssd/data/zwwl_user/",
    "/Users/",
)
ARTIFACT_SUFFIXES = {
    ".ckpt",
    ".mp4",
    ".mov",
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".safetensors",
}


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [root / value for value in result.stdout.decode().split("\0") if value]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    paths = tracked_files(root)
    for path in paths:
        relative = path.relative_to(root)
        if path.suffix.lower() in ARTIFACT_SUFFIXES:
            failures.append(f"tracked model/video artifact: {relative}")
            continue
        if not path.is_file():
            continue
        # This file defines the forbidden strings; it is not a runtime config.
        # Keep the exemption exact so all other scripts remain checked.
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in PRIVATE_PATH_MARKERS:
            if marker in text:
                failures.append(f"private absolute path {marker!r}: {relative}")

    if failures:
        print("Repository hygiene check failed:")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print(f"Repository hygiene check passed ({len(paths)} tracked files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
