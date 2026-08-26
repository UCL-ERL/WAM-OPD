#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 MANIFEST [validate|status|run] [--dry-run]" >&2
  exit 2
fi

manifest="$1"
command="${2:-run}"
dry_run="${3:-}"
workspace="${WAM_OPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${WAM_OPD_PYTHON_BIN:-${WAOPD_PYTHON_BIN:-python3}}"

if [[ ! -f "$manifest" ]]; then
  echo "missing pipeline manifest: $manifest" >&2
  exit 2
fi
if [[ "$command" != "validate" && "$command" != "status" && "$command" != "run" ]]; then
  echo "invalid command: $command" >&2
  exit 2
fi
if [[ -n "$dry_run" && "$dry_run" != "--dry-run" ]]; then
  echo "third argument must be --dry-run" >&2
  exit 2
fi

cd "$workspace"
args=("$command" --manifest "$manifest")
if [[ -n "$dry_run" ]]; then
  args+=("$dry_run")
fi
exec "$python_bin" -u experiments/run_qualified_success_path_pipeline.py "${args[@]}"
