# WAM-OPD

On-policy distillation for joint World-Action Models (WAMs), using a
full-step LingBot-VA Teacher to improve the released few-step Flash-WAM
RoboTwin Student on Student-visited histories.

The current method performs Student-controlled collection, coherent Teacher
video/action labeling, rank-8 JointLoRA training over all 30 shared
Transformer blocks, checkpoint screening, and exact-paired held-out
evaluation. The Teacher is used for labeling/training only; deployment loads
the Student and selected adapter.

Primary contributor: [ylhaichen](https://github.com/ylhaichen).

## What is in this repository

- `experiments/`: the current formal pipeline, JointLoRA implementation,
  training/evaluation runtime, deployment prototypes, and focused tests;
- `scripts/`: stable command-line entry points;
- `configs/`: generated-config documentation (generated files are ignored);
- `docs/method/`: method and protocol decisions;
- `docs/deployment/`: real-robot deployment contract and safety gates;
- `repro/`: external dependency versions and the curated source allowlist.

Models, checkpoints, trajectories, datasets, videos, logs, and evaluation
outputs are intentionally not stored in Git.

## External dependencies

The code expects a sibling or explicitly configured `wave-rl` checkout that
contains LingBot-VA, RoboTwin, and the RoboTwin Python environment. The source
versions used by the current experiments are recorded in
`repro/dependencies.yaml`.

```bash
git clone https://github.com/UCL-ERL/WAM-OPD.git
cd WAM-OPD

# Optional: clone the pinned wave-rl source next to this repository.
./scripts/bootstrap_dependencies.sh

cp .env.example .env
# Edit .env for the local model, dataset, artifact, and Python locations.
set -a
source .env
set +a
```

The large model/data inputs must be transferred separately. Required inputs
for the full formal pipeline are:

1. released `FlashWAM-RoboTwin` Student;
2. `lingbot-va-posttrain-robotwin` Teacher Transformer;
3. RoboTwin native source/assets through `wave-rl`;
4. a qualified task decision and outcome-free episode metadata;
5. writable artifact and scratch roots outside this repository.

## Lightweight verification

The repository-level orchestration tests do not load models or require GPUs:

```bash
python3 -m compileall -q experiments tests
python3 -m pytest -q \
  tests/test_opd_task_specs.py \
  experiments/test_qualified_success_path_pipeline.py \
  experiments/test_scaled_qualified_success_path_pipeline.py \
  experiments/test_stage_h_task_progress.py
```

For model/runtime tests, activate the pinned RoboTwin/LingBot environment and
run:

```bash
python3 -m pytest -q experiments/test_joint_lora.py \
  experiments/test_joint_lora_fp32.py \
  experiments/test_waopd_native_closed_loop_runner.py
```

## Generate and run a formal pipeline

Generated configs go to `configs/generated/` and remain untracked. The scaled
binder preserves the fixed scientific recipe while allowing a larger
train/calibration cohort.

```bash
python3 -m experiments.bind_scaled_qualified_success_path_task \
  --task place_a2b_left \
  --metadata-source /path/to/accepted_episode_metadata.json \
  --qualification-decision /path/to/decision.json \
  --gpu-ids 0,1,2,3 \
  --noise-banks 31001 31002 32001 32002 \
  --train-count 24 \
  --calibration-count 12 \
  --screening-count 4 \
  --heldout-count 6 \
  --run-date YYYYMMDD

./scripts/run_qualified_success_path_pipeline.sh \
  configs/generated/<task>_scaled_qualified_pipeline_v1_<date>.json validate

./scripts/run_qualified_success_path_pipeline.sh \
  configs/generated/<task>_scaled_qualified_pipeline_v1_<date>.json run --dry-run

./scripts/run_qualified_success_path_pipeline.sh \
  configs/generated/<task>_scaled_qualified_pipeline_v1_<date>.json run
```

The fixed formal stages are:

```text
qualification PASS
→ Student-controlled collection / Teacher labeling
→ exactly 3 JointLoRA epochs
→ E1/E2/E3 screening
→ checkpoint selection
→ exact-paired Released/Adapted held-out evaluation
```

Read `QUALIFIED_SUCCESS_PATH_PIPELINE_V1.md` before changing any split,
selection, epoch, seed, or evaluation contract.

## Real-robot deployment

Start with `docs/deployment/REAL_ROBOT_DEPLOYMENT.md`. The repository includes
model-server and offline observation prototypes, but it does not yet contain a
verified production robot SDK adapter, calibration layer, collision
supervisor, watchdog, or E-stop integration.

Do not connect policy output directly to actuators. Required progression is:

```text
offline two-chunk parity
→ WebSocket loopback
→ motors-disabled shadow mode
→ human-authorized low-speed one-chunk test
→ guarded closed-loop trials
```

Simulation tests are not evidence of hardware safety.
