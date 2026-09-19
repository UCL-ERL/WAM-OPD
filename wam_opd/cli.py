"""Small, stable command-line boundary for WAM-OPD.

The research entry points remain import-compatible under ``experiments``. This
module gives new users one discoverable command for environment checks and
manifest validation without hiding the underlying protocol.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from experiments import paths
from . import __version__


def _source_checks() -> list[dict[str, Any]]:
    root = paths.RUNTIME_ROOT
    checks = [
        ("runtime_root", root, "directory"),
        ("lingbot_va_source", root / "third_party/lingbot-va/wan_va/wan_va_server.py", "file"),
        ("robotwin_source", root / "third_party/RoboTwin-lingbot-native/envs", "directory"),
    ]
    result = []
    for name, target, kind in checks:
        exists = target.is_file() if kind == "file" else target.is_dir()
        result.append({"name": name, "path": str(target), "ok": exists})
    return result


def _doctor(*, require_gpu: bool, require_models: bool) -> dict[str, Any]:
    source_checks = _source_checks()
    python_path = paths.PYTHON_BIN
    python_ok = python_path.is_file() and os.access(python_path, os.X_OK)
    package_checks = {
        name: importlib.util.find_spec(name) is not None
        for name in ("torch", "numpy", "accelerate", "diffusers", "transformers")
    }
    gpu = {"available": False, "count": 0, "name": None}
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        import torch

        gpu["available"] = bool(torch.cuda.is_available())
        gpu["count"] = int(torch.cuda.device_count())
        if gpu["available"]:
            gpu["name"] = str(torch.cuda.get_device_name(0))
    student = paths.STUDENT_ROOT
    teacher = paths.TEACHER_ROOT
    model_checks = [
        {"name": "student_root", "path": str(student), "ok": student.exists()},
        {"name": "teacher_root", "path": str(teacher), "ok": teacher.exists()},
    ]
    errors = [item["name"] for item in source_checks if not item["ok"]]
    if not python_ok:
        errors.append("python_interpreter")
    if require_gpu and not gpu["available"]:
        errors.append("cuda")
    if require_models:
        errors.extend(item["name"] for item in model_checks if not item["ok"])
    return {
        "schema": "wam_opd_doctor_v1",
        "status": "PASS" if not errors else "FAIL",
        "version": __version__,
        "python": {"executable": str(python_path), "ok": python_ok},
        "runtime_root": str(paths.RUNTIME_ROOT),
        "sources": source_checks,
        "packages": package_checks,
        "cuda": gpu,
        "models": model_checks,
        "errors": errors,
    }


def _print_result(result: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"WAM-OPD {result.get('schema', 'result')} — {result['status']}")
    if "errors" in result and result["errors"]:
        print("Errors: " + ", ".join(str(value) for value in result["errors"]))
    for key in ("runtime_root", "python", "cuda"):
        if key in result:
            print(f"{key}: {result[key]}")
    for item in result.get("sources", []) + result.get("models", []):
        print(f"{'OK' if item['ok'] else 'MISSING'} {item['name']}: {item['path']}")


def _validate_manifest(manifest: Path) -> dict[str, Any]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    schema = str(payload.get("schema", ""))
    if schema == "waopd_scaled_qualified_success_path_pipeline_v1":
        from experiments.run_scaled_qualified_success_path_pipeline import validate_contract
    elif schema == "waopd_qualified_success_path_pipeline_v1":
        from experiments.run_qualified_success_path_pipeline import validate_contract
    else:
        raise ValueError(f"unsupported manifest schema: {schema!r}")
    return dict(validate_contract(manifest))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wam-opd", description="WAM-OPD operator tools")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="check the environment without loading a model")
    doctor.add_argument("--json", action="store_true", dest="as_json")
    doctor.add_argument("--require-gpu", action="store_true")
    doctor.add_argument("--require-models", action="store_true")
    manifest = sub.add_parser("manifest", help="validate a generated pipeline manifest")
    manifest_sub = manifest.add_subparsers(dest="manifest_command", required=True)
    validate = manifest_sub.add_parser("validate")
    validate.add_argument("path", type=Path)
    validate.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        result = _doctor(require_gpu=args.require_gpu, require_models=args.require_models)
        _print_result(result, as_json=args.as_json)
        return 0 if result["status"] == "PASS" else 1
    if args.command == "manifest" and args.manifest_command == "validate":
        try:
            result = _validate_manifest(args.path.expanduser().resolve())
        except (OSError, ValueError, KeyError, TypeError) as error:
            result = {"schema": "wam_opd_manifest_validation_v1", "status": "FAIL", "errors": [str(error)]}
        _print_result(result, as_json=args.as_json)
        return 0 if result["status"] == "PASS" else 1
    raise AssertionError("unreachable command")


__all__ = ["build_parser", "main"]
