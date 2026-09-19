# WAM-OPD documentation

Use this page as the map when you are not sure which document to read first.

## Start here

| Need | Read |
| --- | --- |
| Understand the fixed research workflow | [Qualified success-path contract](../QUALIFIED_SUCCESS_PATH_PIPELINE_V1.md) |
| Install and verify a checkout | [Reproducibility](REPRODUCIBILITY.md) and [server/source boundary](SERVER_SOURCE_SYNC.md) |
| Understand what belongs in Git versus external artifact storage | [Artifact policy](ARTIFACT_POLICY.md) and [repository layout](REPOSITORY_LAYOUT.md) |
| Run or adapt a task contract | [Method and task protocol notes](method/) |
| Work with a real robot | [Deployment and safety gates](deployment/REAL_ROBOT_DEPLOYMENT.md) |

## Method notes

- [Joint training method decision](method/joint_training_method_decision_20260820.md)
- [Flash-WAM / LingBot distillation contract](method/flashwam_lingbot_distillation_contract_20260820.md)
- [VLA-OPD algorithm notes](method/vla_opd_algorithm_answer_20260825.md)
- [Iteration geometry](method/vla_opd_iteration_geometry_20260825.md)

## Engineering and release notes

- [Repository presentation survey](research/REPOSITORY_PRESENTATION_SURVEY.md)
- [Server source synchronization](SERVER_SOURCE_SYNC.md)
- [Real-robot deployment gates](deployment/REAL_ROBOT_DEPLOYMENT.md)

The public repository intentionally does not contain private checkpoints,
trajectory artifacts, generated videos, server-only paths, or simulator/model
assets. Those inputs are bound through the documented artifact and
reproducibility contracts.
