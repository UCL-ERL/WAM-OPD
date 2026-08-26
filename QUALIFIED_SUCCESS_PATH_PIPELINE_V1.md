# Qualified Success-Path Pipeline V1

This is the fixed post-qualification OPD path:

```text
strict gap PASS
  -> four-shard collection
  -> exactly three JointLoRA epochs
  -> checkpoint screening
  -> exact-paired held-out
  -> optional evaluation
  -> consolidated final summary
```

The only task-specific input is one JSON manifest. The controller is
`experiments/run_qualified_success_path_pipeline.py`; the thin entry point is
`scripts/run_qualified_success_path_pipeline.sh`.

## Frozen research contract

- Split: `8 train / 4 calibration / 2 screening / 6 heldout`.
- Training: one round, exactly three epochs, JointLoRA rank 8, all 30 blocks,
  AdamW, and effective batch size 4.
- Screening: epochs 1, 2, and 3 under the fixed lexicographic selection rule.
- Held-out: two non-overlapping noise banks by six held-out records, producing
  12 exact Released/Adapted pairs.
- Optional evaluation: only after held-out and always `selection_input=false`.
- Qualification: the input decision must be a PASS from the preregistered
  eight-pair strict Teacher/Student gap gate.
- Large models, trajectories, checkpoints, logs, videos, and evaluation output
  must remain under `/ssd/data`.

## Resource contract

- Four GPUs are the stable allocation; up to eight GPUs may be used for burst
  evaluation.
- Collection uses four independent one-GPU shards.
- Training uses either the certified existing single-process backend
  (`train_world_size=1`) or the separately parity-certified four-rank DDP
  backend (`train_world_size=4`). World size may only be chosen at an epoch
  boundary. Mid-epoch elastic resizing is forbidden.
- Screening, held-out, and optional evaluation may use all currently free GPUs
  allowed by the manifest.
- A GPU is selected only when it meets the stage-specific free-memory threshold
  and has no compute process.

The backend is an explicit contract boundary. A manifest must supply its
validated `update_argv`; the controller does not reinterpret a serial trainer
as distributed training. Until `joint_lora_ddp_v1` has passed numerical-parity
and resume/checkpoint tests, production manifests use
`joint_lora_single_process_v1` while still using four-way collection and burst
parallel evaluation.

## State and recovery

Each core stage must publish its existing PASS receipt. The controller derives
state only from those receipts and expected epoch checkpoints. If a stage has
started but is incomplete or non-PASS, it stops with a concrete error. It never
deletes output, guesses whether partial output is reusable, or automatically
retries it.

Commands:

```bash
scripts/run_qualified_success_path_pipeline.sh MANIFEST validate
scripts/run_qualified_success_path_pipeline.sh MANIFEST run --dry-run
scripts/run_qualified_success_path_pipeline.sh MANIFEST status
scripts/run_qualified_success_path_pipeline.sh MANIFEST run
```

The final receipt is written to:

```text
<formal_root>/pipeline/final_summary.json
```

It binds the training summary, selected epoch/checkpoint, held-out result, GPU
policy, and optional evaluation without making optional diagnostics part of
selection.
