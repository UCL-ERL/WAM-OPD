# Repository layout and ownership

This repository is a public, portable mirror of a larger research-server
working tree. The goal of the layout is to reduce accidental coupling without
breaking established `python -m experiments...` commands.

## Ownership map

| Surface | Owner | Contract |
| --- | --- | --- |
| `experiments/opd_task_specs.py` | task contract | Task IDs, prompt/task metadata, and horizon defaults used by binders and runners. |
| `experiments/*lora*.py`, `experiments/action_output_adapter.py`, `experiments/video_output_adapter.py` | model adapters | Parameter ownership, output shape, and adapter state semantics. |
| `experiments/bind*_qualified_success_path_task.py` | manifest generation | Produces validated task manifests; generated configs are not committed. |
| `experiments/run*_qualified_success_path_pipeline.py` | orchestration | Stage ordering, receipt checks, GPU policy, and fail-closed recovery. |
| `experiments/train_*.py` | training | Collection/label artifact contracts and optimizer updates. |
| `experiments/waopd_native_*.py` | native runtime | RoboTwin/LingBot process boundary and episode serialization. |
| `experiments/test_*.py`, `tests/` | verification | Contract and smoke tests; tests are not disposable diagnostics. |
| `scripts/` | operator entry points | Thin wrappers only; no duplicated training logic. |
| `docs/` and top-level contract files | decisions | Human-readable method, deployment, and artifact decisions. |

## Compatibility rules

1. Existing module paths are public within the research workflow. Do not move a
   module solely for aesthetics; add a compatibility wrapper and a migration
   note if a move is necessary.
2. `configs/generated/`, model roots, datasets, checkpoints, logs, and videos
   are external artifacts. They must not be committed.
3. A `prototype_*` file is a historical or diagnostic surface, not evidence
   that the default formal pipeline imports it. Keep it when external server
   jobs may still reference its path; document it rather than deleting it from
   a static search result.
4. Server-only absolute paths must be supplied by environment variables or
   manifests at the public boundary. A path copied from a server command is
   not a portable default.
5. New launchers should call an existing Python entry point instead of copying
   orchestration logic into another shell script.

## Where to start

- Formal pipeline: `QUALIFIED_SUCCESS_PATH_PIPELINE_V1.md` and
  `experiments/run_qualified_success_path_pipeline.py`.
- Task binding: `experiments/bind_scaled_qualified_success_path_task.py`.
- Training contract: `experiments/train_iterative_on_policy_flow_opd.py` and
  the method documents under `docs/method/`.
- Deployment safety: `docs/deployment/REAL_ROBOT_DEPLOYMENT.md`.
- Server synchronization: `docs/SERVER_SOURCE_SYNC.md`.

## Deliberately not merged into the public tree

The server working snapshot contains many one-off audits, recovery scripts,
task-specific wrappers, output manifests, and private artifact references.
Those are useful operational records but are not automatically public API. A
server file enters this repository only when it is:

1. a canonical runtime or contract surface;
2. free of private checkpoints, credentials, and hard-coded infrastructure
   paths; and
3. covered by an import/compile or focused behavior check.

This rule keeps the public repository smaller without deleting the server's
history or making the experiment irreproducible: the external artifact and
source versions are bound by manifests and documented separately.
