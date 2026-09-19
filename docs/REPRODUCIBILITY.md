# Reproducibility and release verification

## Release decision: HOLD

CPU tests and configuration validation are reproducible. A fresh-machine GPU
collection → labeling → training → screening → paired60 run is **not yet
certified**. Do not interpret a green CI badge as certification of model
weights, external simulator assets, historical evaluation results, or the full
server implementation.

The September 20, 2026 checks used a clean clone of commit `26216a8`, not the
developer's dirty training file. The same commit was exported into an isolated
server directory with the production OPD checkout excluded from `PYTHONPATH`.
GPU visibility was disabled for the CPU checks. No production experiment,
checkpoint, trajectory, running process, or server source file was modified.

## Verified layers

| Check | Result on the baseline commit | What it establishes |
| --- | --- | --- |
| Wheel build / `pip install '.[dev]'`, Python 3.11 | Wheel builds; tests initially fail because NumPy was absent | Packaging alone is insufficient. NumPy is now included in the dev extra. |
| `make test` after clean checkout | Initially fails: hygiene checker rejects its own rule definitions | The committed state differed from the previous untracked-file test. Fixed with a regression test. |
| Entire CPU suite in fresh macOS Python 3.11 environment, Torch 2.6.0 / NumPy 1.26.4 | 241 passed, 1 skipped | Public test suite works without the server's Python packages. Not a model rollout. |
| Entire CPU suite in isolated Linux server checkout | 242 passed, 33 subtests passed | Same public source passes in the existing server runtime. |
| Training, teacher-free evaluation, screening, paired evaluation, scaled controller `--help` on Linux | All five exit 0 | Imports and CLI parsing work with the server dependencies. Models were not loaded. |
| Real metadata → new scaled manifest → scaled `validate` | PASS: 24 train, 12 calibration, 4 screening, 6 held-out; 4 shards; formal12 | Binder/controller agree when given real inputs in separate output roots. |
| Scaled manifest → old fixed-pipeline shell wrapper | Schema mismatch, reproduced | README formerly selected the wrong controller. Commands are corrected below. |
| Distributed success-path hooks | Focused trainer/distributed tests pass: 47 passed | The optional equivalence pilot has an explicit config/artifact contract and is not part of the default single-GPU pipeline. |
| Historical capture dependency | `experiments.v0k_native_video_diagnostic` is missing | Static compilation and current tests do not cover every shipped entry point. |

The macOS skip is reported by pytest; use `-rs` to see platform-specific reasons.
New regression tests for the hygiene checker and incomplete bootstrap are
included in `make test`.

## Start here: CPU development

### September 20 follow-up: independent runtime defaults

Verified source: commit `cc4d566`, cloned with `git clone --no-local` into a
new directory. The dirty developer trainer was **not** included.

| Verification layer | Command / result |
| --- | --- |
| Fresh Python 3.11 venv and wheel install | `python -m pip install '<clean-checkout>[dev]'`: PASS, NumPy 2.4.6 / pytest 8.4.2 |
| Lightweight repository gates | `make test`: **54 passed**, compile and hygiene PASS |
| Broader CPU suite from clean checkout | With the separate Torch 2.6.0 test environment and `einops`: `CUDA_VISIBLE_DEVICES='' python -m pytest -q experiments tests -rs`: **256 passed, 1 skipped** |
| Skip | Native closed-loop test requires the external LingBot-VA runtime |
| Direct upstream source bootstrap | Fresh isolated source checkout plus versioned WAM-OPD compatibility patch: PASS; bounded retries and `GIT_LFS_SKIP_SMUDGE=1` avoid downloading model blobs, and the bootstrap refuses existing source directories |
| Downloaded revisions | LingBot-VA `58c2ae5bac46bd8114065bea9d7d256eb67c16c3` plus `patches/lingbot-va/0001-wam-opd-portability-and-runtime.patch`; RoboTwin `2eeec322d95799f537cbfe5f291a8220d965ccb8` |
| Shell parsing and diff integrity | `bash -n` on the three changed launch/bootstrap scripts; `git diff --check`: PASS |
| GPU / real model / simulator execution | PASS on release commit `cc4d566`: isolated server GPU7 smoke; `place_fan`, SS arm, one chunk, 16 controls, output status PASS |

Operation receipt (finding S1 — runtime ownership): the retired requirement is
a sibling research-workspace checkout and a Python interpreter inside another
framework's directory. Runtime defaults, source bootstrap, manifest binder,
launcher fallbacks, `.env.example`, dependency pins and installation docs now
use WAM-OPD-owned paths. Explicit legacy environment settings and persisted
`project_root` remain compatible. Tests cover path precedence, active-venv
interpreter identity, missing source detection and refusal to overwrite either
existing source tree. No loss, checkpoint format, artifact, server source or
running process was changed. The unrelated local trainer edits remain
uncommitted. Reverting `7662c76` and its follow-up commits restores the source-only boundary change; no
data migration is needed. Retained historical wrapper dependencies and the
other release gaps below prevent a full standalone-runtime claim.

The GPU smoke used only an unoccupied physical GPU and a separate `/tmp` output
root. It loaded the released Student and official Teacher transformer, then
completed one native closed-loop episode. The result was not counted as a
benchmark score: `success=false` after the intentionally truncated 16-control
step horizon. The output receipt had schema `waopd_native_closed_loop_run_v2`,
`status=PASS`, `training_started=false`, and `episodes_started=1`; its JSON
SHA256 was
`993408a0b7f739cada43924dda0d6d76702d46d4f796b928875487ecd329fc9b`.
The server runtime was the observed environment, not a pristine source
installation; the smoke therefore proves the public entry point and artifact
serialization against that environment, not clean-machine CUDA reproducibility.

Use Python 3.10–3.12; Python 3.11 is the version used for the fresh-environment
check. No checkpoints or GPUs are required:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev]'
make test
```

To reproduce the broader CPU suite (Torch is deliberately not a mandatory
dependency of the lightweight package):

```bash
python -m pip install 'torch==2.6.0' 'numpy==1.26.4'
CUDA_VISIBLE_DEVICES='' python -m pytest -q experiments tests -rs
```

These tests include fake runtimes and small tensors. They are regression tests,
not substitutes for loading the official models and running the simulator.

## GPU runtime prerequisites

Before attempting any GPU command, provide all of the following:

- a Linux/CUDA runtime with the appropriate PyTorch build and simulator stack;
- official released Student and Teacher weights, with their model configs and
  tokenizer/VAE/text-encoder dependencies;
- LingBot-VA and RoboTwin source trees and RoboTwin assets;
- canonical task metadata and a real qualification decision;
- distinct writable config, collection, training, screening, and eval roots;
- for historical comparisons, the original frozen formal12 + extension48
  seeds, prompts, initial-state/noise contracts and selected adapter.

Set the roots from `.env.example`. Use `WAM_OPD_PYTHON_BIN` to select the actual
runtime interpreter. The formal validators currently require relevant
artifact paths under `/ssd/data`; do not weaken these checks just to get a
configuration to pass.

### Important dependency gaps

1. The bootstrap helper now downloads pinned LingBot-VA and RoboTwin sources
   directly into WAM-OPD's `third_party/`. This replaces the former incomplete
   workspace scaffold. It is a source download, not a Python/CUDA environment,
   model-weight or simulator-asset installer. All of those remain separate
   prerequisites until an independent GPU smoke test passes.
2. The observed server LingBot-VA tree is based on commit
   `58c2ae5bac46bd8114065bea9d7d256eb67c16c3`, but has local edits in
   `wan_va/modules/model.py`, `wan_va/wan_va_server.py`,
   `evaluation/robotwin/eval_polict_client_openpi.py`, and
   `evaluation/robotwin/websocket_client_policy.py`. A pristine upstream
   checkout has not been shown equivalent to these edits.
3. The previously recorded Python 3.10 / Torch 2.9 / CUDA 12.6 environment is
   not the runtime used by this verification. The observed runtime is Python
   3.11.14 / Torch 2.6.0 / CUDA 12.4; see `repro/dependencies.yaml`.
4. The server trainer includes `offline_teacher` support absent from the
   public trainer. The local distributed-training edits are a separate,
   uncommitted change. These must be reconciled against the server source
   without silently losing either history-source behavior or compatibility.
5. Downloadable reproduction inputs (task metadata, qualified decisions,
   selected adapters and frozen 60-pair panels) have not been packaged into a
   documented public artifact release. They are not produced by `pip install`.
6. `prototype_real_obs_action_teacher_bridge.py` and several server-only
   historical diagnostics use an older internal adapter wrapper. They have not
   been certified in the independent environment and are not advertised as
   standalone entry points. The formal native runner is direct and does not
   import that historical adapter. Existing artifact provenance strings and
   explicit legacy environment settings are retained for compatibility, not as
   new-user installation requirements.

### Independent environment layout

`WAM_OPD_RUNTIME_ROOT` defaults to this repository and takes precedence over
legacy explicitly configured roots. Existing manifest `project_root` fields
remain readable. New manifests point to the WAM-OPD runtime root. The Python
default is the active interpreter; interpreter symlinks are not resolved, so a
venv remains a venv in child processes. `WAM_OPD_PYTHON_BIN` overrides it.

```text
WAM-OPD/
  .venv/                         independent Python environment
  third_party/lingbot-va/         pinned model runtime source
  third_party/RoboTwin-lingbot-native/  pinned simulator source + separate assets
```

The bootstrap refuses existing source directories rather than overwriting
them. Existing server directories, source patches, environments, running jobs
and historical manifests are not changed by this migration.

## Correct scaled-pipeline commands

After binding a manifest, use the matching controller:

```bash
manifest=configs/generated/<task>_scaled_qualified_pipeline_v1_<date>.json
"$WAM_OPD_PYTHON_BIN" -m experiments.run_scaled_qualified_success_path_pipeline \
  validate --manifest "$manifest"
"$WAM_OPD_PYTHON_BIN" -m experiments.run_scaled_qualified_success_path_pipeline \
  status --manifest "$manifest"
```

Once the dependency and input gaps above are closed, the actual GPU stages are:

```bash
# This runs real collection probes, not a dry-run.
"$WAM_OPD_PYTHON_BIN" -m experiments.run_scaled_qualified_success_path_pipeline \
  canary --manifest "$manifest" --workers-per-gpu 1
"$WAM_OPD_PYTHON_BIN" -m experiments.run_scaled_qualified_success_path_pipeline \
  run --manifest "$manifest"
```

Use the same worker count in the binder and canary for this one-worker recipe.
Do not pass a scaled manifest to `scripts/run_qualified_success_path_pipeline.sh`:
that wrapper uses the fixed-size schema. The scaled controller has no
`--dry-run` option. Its standard held-out stage is formal12, not paired60.
The remaining 48 pairs require the actual frozen extension protocol; they
must not be silently regenerated for a historical comparison.

## Remaining acceptance criteria before pushing a certified release

1. Reconcile the server trainer and its required canonical modules with the
   public tree; test all advertised entry points, including optional pilots
   or explicitly resolve their support status.
2. Publish/install the required dependency revisions **and** any necessary
   patches; build an independent runtime without borrowing a dirty server tree.
3. Document accessible model/asset/panel inputs and a small runnable example.
4. In an isolated output root, run real collection/labeling, an optimizer
   update with checkpoint save/load, screening, and an Adapted rollout; verify
   semantics against the server implementation before any full benchmark.
5. Re-run clean-clone install/tests, audit the final commit, then push. Do not
   claim paper-result reproduction until the specified full panel is run or
   matched to published, verifiable outputs.

## Recovery and scope

This verification changes packaging, the license, developer checks, bootstrap
failure reporting, and usage documentation. It does not change training losses,
checkpoint formats, simulator behavior, GPU routing, or original server files.
The existing dirty trainer is intentionally left untouched. Source-only fixes
can be reversed with a normal Git revert; no model/data rollback is required.
