#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
wave_rl_root="${WAVE_RL_ROOT:-$repo_root/../wave-rl}"
wave_rl_commit="d7aeed296ef1daa98cfda0108fd3475946226971"

if [[ -e "$wave_rl_root" ]]; then
  echo "refusing to overwrite existing path: $wave_rl_root" >&2
  exit 2
fi

git clone https://github.com/ylhaichen/wave-rl.git "$wave_rl_root"
git -C "$wave_rl_root" checkout "$wave_rl_commit"
git -C "$wave_rl_root" submodule update --init --recursive

missing=0
for required in third_party/lingbot-va/wan_va/wan_va_server.py third_party/RoboTwin-lingbot-native/envs third_party/RLinf; do
  if [[ ! -e "$wave_rl_root/$required" ]]; then
    echo "INCOMPLETE external source: missing $required" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "The pinned wave-rl tree is a scaffold, not an installed runtime. See docs/REPRODUCIBILITY.md." >&2
  exit 2
fi

echo "wave-rl source ready at $wave_rl_root"
echo "Next: create the RoboTwin/LingBot environment and configure .env."
