# Experiments package

The Python module paths in this directory are compatibility-sensitive because
the research server launches them with `python -m experiments.<module>`.

## Formal entry points

For new users, start with the public operator command:

```bash
wam-opd doctor
wam-opd manifest validate configs/generated/<manifest>.json
```

The `experiments` module paths below remain the stable implementation and
server-compatibility surface.

- `bind_qualified_success_path_task.py` and
  `bind_scaled_qualified_success_path_task.py` create validated manifests.
- `run_qualified_success_path_pipeline.py` and
  `run_scaled_qualified_success_path_pipeline.py` control stage ordering and
  receipts.
- `train_iterative_on_policy_flow_opd.py` owns the trajectory-update training
  contract used by the formal success-path pipeline.
- `train_joint_teacher_trajectory_opd.py` and the modality-specific training
  modules provide narrower research variants.
- `waopd_native_closed_loop_runner.py` and the `waopd_native_*` modules bridge
  the Python contracts to the native RoboTwin/LingBot runtime.
- `run_paired_multinoise.py` is the task-generic alias for the exact paired
  evaluation runner; it keeps the operator-facing module name independent of
  the original task-specific implementation filename.

## Support and historical surfaces

`stage_*`, `goal*`, `prototype_*`, and `robotwin_*` modules cover diagnostics,
recovery, and earlier vertical slices. They remain at their historical paths
when server jobs may import them. They are not default entry points and should
not be used to infer the current formal protocol without reading the relevant
contract document.

## Adding a new experiment

Prefer this sequence:

1. put task-independent reusable code in an existing module;
2. add a small task contract or manifest field rather than duplicating a
   launcher;
3. keep large outputs outside the checkout;
4. add a CPU-level contract test or dry-run;
5. document the stage and compatibility boundary.
