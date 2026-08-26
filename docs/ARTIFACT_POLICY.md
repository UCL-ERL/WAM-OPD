# Artifact policy

Git tracks source code, tests, documentation, configuration templates, and
small human-readable receipts only.

The following stay in external artifact storage:

- base model and Teacher weights;
- JointLoRA checkpoints;
- collected trajectories and label tensors;
- datasets and simulator assets;
- videos, logs, caches, temporary files, and full evaluation outputs.

For a selected adapter, record its external location, task, selected epoch,
base model identity, and evaluation summary. A separate artifact service can
be introduced later; Git LFS is not required for the current repository.
