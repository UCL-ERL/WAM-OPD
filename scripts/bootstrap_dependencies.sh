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

echo "wave-rl source ready at $wave_rl_root"
echo "Next: create the RoboTwin/LingBot environment and configure .env."
