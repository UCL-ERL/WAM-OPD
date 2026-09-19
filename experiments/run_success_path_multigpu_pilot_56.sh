#!/usr/bin/env bash
set -euo pipefail

repo_root="${WAM_OPD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
pilot_root="${WAM_OPD_PILOT_ROOT:-${repo_root}/.artifacts/multigpu_equivalence_pilot}"
python_bin="${WAM_OPD_PYTHON_BIN:-python3}"
source_config="${WAM_OPD_PILOT_CONFIG:-}"

if [[ -z "$source_config" ]]; then
  echo "WAM_OPD_PILOT_CONFIG must point to an existing trajectory_update config" >&2
  exit 2
fi
if [[ ! -f "$source_config" ]]; then
  echo "pilot config does not exist: $source_config" >&2
  exit 2
fi

cd "$repo_root"
mkdir -p "$pilot_root"
runtime_root="${WAM_OPD_RUNTIME_ROOT:-${WAVE_RL_ROOT:-${PROJECT_ROOT:-$repo_root}}}"
export PYTHONPATH="${repo_root}:${runtime_root}/third_party/lingbot-va:${runtime_root}/third_party/RoboTwin-lingbot-native${PYTHONPATH:+:${PYTHONPATH}}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${runtime_root}/third_party/RoboTwin-lingbot-native}"
export PYTHONUNBUFFERED=1

if [[ ! -f "$pilot_root/single/result.json" ]]; then
  env CUDA_VISIBLE_DEVICES=5 "$python_bin" -u -m experiments.run_success_path_multigpu_equivalence_pilot \
    --mode single \
    --config "$source_config" \
    --output-dir "$pilot_root/single" \
    >"$pilot_root/single.log" 2>&1
fi

env CUDA_VISIBLE_DEVICES=5,6 "$python_bin" -u -m torch.distributed.run \
  --standalone \
  --nproc-per-node=2 \
  -m experiments.run_success_path_multigpu_equivalence_pilot \
  --mode distributed \
  --config "$source_config" \
  --output-dir "$pilot_root/distributed_retry1" \
  >"$pilot_root/distributed_retry1.log" 2>&1

"$python_bin" -u -m experiments.compare_success_path_multigpu_pilot \
  --single-dir "$pilot_root/single" \
  --distributed-dir "$pilot_root/distributed_retry1" \
  --output "$pilot_root/verdict.json" \
  >"$pilot_root/compare.log" 2>&1
