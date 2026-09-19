#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${WAM_OPD_RUNTIME_ROOT:-$repo_root}"
lingbot_root="$runtime_root/third_party/lingbot-va"
robotwin_root="$runtime_root/third_party/RoboTwin-lingbot-native"

for target in "$lingbot_root" "$robotwin_root"; do
  if [[ -e "$target" || -L "$target" ]]; then
    echo "refusing to overwrite existing path: $target" >&2
    exit 2
  fi
done

mkdir -p "$runtime_root/third_party"
git clone https://github.com/robbyant/lingbot-va.git "$lingbot_root"
git -C "$lingbot_root" checkout --detach 58c2ae5bac46bd8114065bea9d7d256eb67c16c3
git -C "$lingbot_root" submodule update --init --recursive
git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$robotwin_root"
git -C "$robotwin_root" checkout --detach 2eeec322d95799f537cbfe5f291a8220d965ccb8
git -C "$robotwin_root" submodule update --init --recursive

missing=0
for required in third_party/lingbot-va/wan_va/wan_va_server.py third_party/RoboTwin-lingbot-native/envs; do
  if [[ ! -e "$runtime_root/$required" ]]; then
    echo "INCOMPLETE external source: missing $required" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "Upstream sources are incomplete. See docs/REPRODUCIBILITY.md." >&2
  exit 2
fi

runtime_patch="$repo_root/patches/lingbot-va/0001-wam-opd-portability-and-runtime.patch"
if [[ ! -f "$runtime_patch" ]]; then
  echo "missing WAM-OPD upstream patch: $runtime_patch" >&2
  exit 2
fi
git -C "$lingbot_root" apply --check "$runtime_patch"
git -C "$lingbot_root" apply "$runtime_patch"

echo "Pinned upstream source checkouts and WAM-OPD compatibility patch ready under $runtime_root/third_party."
echo "Source download only: Python/CUDA packages, model weights and simulator assets are not installed."
echo "Model weights, simulator assets and CUDA packages remain separate prerequisites."
