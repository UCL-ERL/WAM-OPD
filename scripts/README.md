# Operator scripts

Shell scripts in this directory are intentionally thin. They resolve the
repository and external dependency roots, then delegate to a Python module.
Training, evaluation, and recovery logic belongs in `experiments/` so that it
can be imported and tested without shell-specific behavior.

Before running a script:

1. copy `.env.example` to `.env` and set external roots;
2. validate the generated manifest;
3. use a new output root for a new run;
4. keep checkpoints, trajectories, logs, and videos outside Git.

`bootstrap_dependencies.sh` checks out the pinned upstream sources and applies
the versioned WAM-OPD compatibility patch under `patches/lingbot-va/`. The
patch is explicit and reviewable; it is not a copy of a private server tree.

The formal pipeline wrapper is
`run_qualified_success_path_pipeline.sh`. The other scripts are diagnostics or
dependency/bootstrap helpers and are documented by their filenames and module
docstrings.
