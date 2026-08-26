#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
  echo "usage: $0 TASK GPU_ID PAIR_MANIFEST OUTPUT_ROOT" >&2
  exit 2
fi

task="$1"
gpu_id="$2"
pair_manifest="$3"
output_root="$4"
if [[ ! "$task" =~ ^[a-z0-9_]+$ || ! "$gpu_id" =~ ^[0-7]$ ]]; then
  echo "invalid task or GPU id" >&2
  exit 2
fi
if [[ -e "$output_root" ]]; then
  echo "refusing to overwrite diagnostic root: $output_root" >&2
  exit 3
fi
if [[ ! -f "$pair_manifest" ]]; then
  echo "missing diagnostic pair manifest: $pair_manifest" >&2
  exit 3
fi

workspace="${WAM_OPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
project="${WAVE_RL_ROOT:-${PROJECT_ROOT:-$workspace/../wave-rl}}"
lingbot_root="$project/third_party/lingbot-va"
robotwin_root="$project/third_party/RoboTwin-lingbot-native"
client_python="${WAM_OPD_PYTHON_BIN:-$project/third_party/RLinf/.venv-robotwin/bin/python}"
client="$workspace/experiments/prototype_fixed_instruction_robotwin_client_diag.py"
run_name="$(basename "$output_root")"
scratch_base="${WAM_OPD_SCRATCH_ROOT:-$workspace/.artifacts/scratch}"
scratch_root="$scratch_base/runtime_diagnostics/$run_name"
client_cwd="$scratch_root/client_cwd"

if [[ -e "$scratch_root" ]]; then
  echo "refusing to reuse diagnostic scratch: $scratch_root" >&2
  exit 3
fi
mkdir -p "$output_root" "$output_root/results" "$client_cwd"
ln -sfnT "$robotwin_root/assets" "$client_cwd/assets"
ln -sfnT "$robotwin_root/task_config" "$client_cwd/task_config"

set +e
(
  cd "$client_cwd"
  timeout --signal=TERM --kill-after=20 600 \
    env \
      PROJECT_ROOT="$project" \
      ROBOTWIN_ROOT="$robotwin_root" \
      PYTHONUNBUFFERED=1 \
      PYTHONWARNINGS=ignore::UserWarning \
      TOKENIZERS_PARALLELISM=false \
      PYTHONPATH="$lingbot_root:$lingbot_root/wan_va:$robotwin_root" \
      MPLCONFIGDIR="$scratch_root/matplotlib" \
      MESA_SHADER_CACHE_DIR="$scratch_root/mesa_shader_cache" \
      CUDA_VISIBLE_DEVICES="$gpu_id" \
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
      PAIR_MANIFEST="$pair_manifest" \
      ROBOTWIN_PAIR_RECHECK_EXPERT=0 \
      ROBOTWIN_SETUP_AUDIT_ONLY=1 \
      ROBOTWIN_TRACE_INITIAL_OBS=1 \
      ROBOTWIN_DIAG_EXIT_BEFORE_INFER=1 \
      ROBOTWIN_ENHANCED_DETERMINISM=1 \
      ROBOTWIN_BRANCH_TRACE_MODE=record \
      ROBOTWIN_BRANCH_TRACE_DIR="$output_root/planner_traces" \
      "$client_python" "$client" \
        --config "$robotwin_root/policy/ACT/deploy_policy.yml" \
        --overrides \
        --task_name "$task" \
        --task_config demo_clean \
        --train_config_name 0 \
        --model_name "${task}_student_diag" \
        --ckpt_setting "${task}_student_diag" \
        --seed 0 \
        --policy_name ACT \
        --save_root "$output_root/results" \
        --video_guidance_scale 5 \
        --action_guidance_scale 1 \
        --test_num 1 \
        --port 1 \
        --instruction_type seen \
        --eval_video_log False \
        --save_video False \
        --save_visualization False
) >"$output_root/diag.log" 2>&1
exit_code=$?
set -e
printf '%s\n' "$exit_code" >"$output_root/exit_code"

if (( exit_code == 0 )) && grep -q "ROBOTWIN_DIAG_INITIAL_OBS_PASS" "$output_root/diag.log"; then
  printf 'PASS\n' >"$output_root/status"
  echo "ROBOTWIN_INITIAL_OBS_DIAG_PASS task=$task gpu=$gpu_id root=$output_root"
  exit 0
fi

printf 'FAIL\n' >"$output_root/status"
echo "ROBOTWIN_INITIAL_OBS_DIAG_FAIL task=$task gpu=$gpu_id exit=$exit_code root=$output_root" >&2
tail -n 100 "$output_root/diag.log" >&2
exit 1
