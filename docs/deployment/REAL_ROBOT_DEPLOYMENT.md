# LingBot OPD → Real-Robot Deployment Handoff

Hello — you are taking over a research codebase that adapts the released Flash-WAM/LingBot-VA RoboTwin policy with on-policy Teacher supervision. Your primary job is **not to continue babysitting GPU experiments**. Your primary job is to turn one selected adapted Student checkpoint into a reproducible, guarded real-robot inference stack on another machine.

This document separates verified facts from missing engineering. Do not command a physical robot until the offline, transport, shadow-mode, and safety gates below pass.

## 1. Project in one page

### What the base model is

LingBot-VA is a joint video/action flow model. At deployment it operates autoregressively:

1. encode the current multi-camera observation and language instruction;
2. generate a future-video latent plan;
3. generate an action chunk conditioned on that plan and cached observation/action history;
4. execute the chunk;
5. feed newly observed frames plus the executed/model action history back into the KV cache;
6. repeat.

The released fast Student used here is Flash-WAM-RoboTwin at the `1 video step / 1 action step` operating point. The frozen slow Teacher is LingBot-VA, used during training only. A real-robot deployment should load **only the Student plus the selected adapter**; it does not need the Teacher.

### What this OPD project changes

The released Student is fast but weak on some RoboTwin tasks. This project collects trajectories controlled exclusively by the current Student, then asks the frozen Teacher to label the exact Student-visited histories without executing Teacher actions.

The current successful method is best described as:

> environment-on-policy coherent video/action distillation with JointLoRA

For a Student history `h_S`, the Teacher produces a coherent video/action target pair:

```text
z_T = TeacherVideo(h_S)
a_T = TeacherAction(h_S, z_T)
```

The Student is updated through one rank-8 JointLoRA bank across all 30 shared Transformer blocks. The base model remains frozen. This is continuous flow/video-action supervision, not VLA-OPD's published action-token Reverse-KL objective.

The current formal experiment structure is:

```text
strict Teacher/Student gap qualification
→ Student-controlled collection and Teacher labeling
→ 24 train trajectories + 12 fixed calibration trajectories
→ exactly 3 epochs on the fixed package
→ screen E1/E2/E3 checkpoints
→ select one checkpoint
→ 12 exact-paired Released/Adapted held-out comparisons
```

For research history and method details, read rather than duplicate:

- `research/vla_opd_algorithm_answer_20260825.md`
- `research/joint_training_method_decision_20260820.md`
- `research/flashwam_lingbot_distillation_contract_20260820.md`
- `research/codebase_task_protocol_map_20260816.md`
- `remote_patch/QUALIFIED_SUCCESS_PATH_PIPELINE_V1.md`

### What has worked so far

The best completed deployment candidate at handoff time is `place_a2b_left`, selected at epoch 3:

- Released exact-paired held-out: `0/12`
- Adapted exact-paired held-out: `5/12`
- Rescues: `5`
- Regressions: `0`
- Screening Adapted success: `E1 1/8 → E2 7/8 → E3 8/8`
- Fixed calibration loss: `0.3651 → 0.3078 → 0.2878`
- Optimizer steps: `219`

This is meaningful task-specific RoboTwin evidence. It is **not evidence of zero-shot transfer to an arbitrary real robot**, camera rig, workspace, or object distribution.

## 2. The deployment goal

Build a real-robot stack with this boundary:

```text
three synchronized RGB cameras + language instruction
                    │
                    ▼
       robot-side observation adapter
                    │  private/local network
                    ▼
     Flash-WAM VA_Server + JointLoRA adapter
                    │
                    ▼
   16-channel × 2-frame × 16-step action chunk
                    │
                    ▼
 action decoder + frame transform + safety supervisor
                    │
                    ▼
       low-level dual-arm robot controller
```

The first deliverable is not “the robot completes the task.” The first deliverable is a reproducible package that:

1. loads the exact released base model and selected JointLoRA checkpoint;
2. reproduces the A800 server output on frozen recorded observations;
3. exposes a private loopback WebSocket endpoint;
4. runs on the robot with motors disabled and logs proposed actions;
5. rejects unsafe, stale, malformed, or out-of-envelope commands;
6. only then performs one human-authorized, low-speed action chunk.

## 3. What already exists and what does not

### Reusable components

The following are real reusable components:

- Native LingBot-VA `VA_Server` with streaming VAE, text encoder, Transformer cache, video solve, action solve, action normalization, and `reset / infer / compute_kv_cache` semantics.
- Official WebSocket/msgpack transport:
  - `wan_va/utils/Simple_Remote_Infer/deploy/websocket_policy_server.py`
  - `wan_va/utils/Simple_Remote_Infer/deploy/websocket_client_policy.py`
- RoboTwin client showing the exact policy loop and action decoding:
  - `evaluation/robotwin/eval_polict_client_openpi.py`
- This project's instrumented Student server:
  - `experiments/prototype_flashwam_robotwin_server.py`
- Current JointLoRA attach/load logic:
  - `experiments/joint_lora.py`
  - `experiments/waopd_v0_video_opd.py`
- Offline saved-observation bridge:
  - `experiments/prototype_real_obs_action_teacher_bridge.py`

### Important non-components

- `prototype_real_obs_action_teacher_bridge.py` is **not a live real-robot driver**. It reads a previously saved three-camera observation, performs model analysis, does not step RoboTwin, and does not command hardware.
- The official README's “Real-world Deployment” section reports results but does not provide the robot SDK/control implementation used for those experiments.
- The generic `Simple_Remote_Infer` demo proves transport only; its sample `qwenpi_policy.py` is not this LingBot-VA policy.
- No production hardware abstraction, camera synchronization layer, frame calibration, collision supervisor, watchdog, or E-stop integration has been verified in this repository.

The receiving agent must implement these missing pieces rather than assuming simulation code is safe on hardware.

## 4. Source and artifact package to move to the new machine

The source A800 host should be accessed through a locally configured SSH alias such as `<A800_HOST>`. Do not copy SSH keys, passwords, tokens, or the underlying public IP into documentation or scripts.

Define these source-machine placeholders after login:

```text
<OPD_REPO>      = /home/<remote-user>/ylhc/LingbotVA-OPD
<WAVE_RL>       = /home/<remote-user>/ylhc/wave-rl
<ARTIFACT_ROOT> = /ssd/data/<remote-user>/ylhc/artifacts
```

### Required source

Copy or reproduce the exact runtime versions:

- `wave-rl` commit: `d7aeed296ef1daa98cfda0108fd3475946226971`
- `third_party/lingbot-va` commit: `58c2ae5bac46bd8114065bea9d7d256eb67c16c3`
- RoboTwin parity checkout, if simulator replay is required: `2eeec322d95799f537cbfe5f291a8220d965ccb8`
- The current `LingbotVA-OPD` tree is an unversioned working snapshot on the source host. A normal clone is not sufficient. Copy the exact relevant files and create a file-hash manifest on the destination.

Minimum OPD runtime files include:

```text
experiments/prototype_flashwam_robotwin_server.py
experiments/waopd_v0_video_opd.py
experiments/waopd_native_closed_loop_runner.py
experiments/joint_lora.py
experiments/goal1_exact_condition.py
experiments/flashwam_runtime_state_audit.py
```

Include their imported local modules after resolving imports. Do not blindly copy only one script.

### Required model artifacts

Released base Student:

```text
<ARTIFACT_ROOT>/LingbotVA-OPD/models/FlashWAM-RoboTwin
```

- Source disk usage at handoff: approximately `9.5 GiB`.
- Copy the whole model directory: Transformer, VAE, tokenizer, text encoder, and configs.
- Verify that `transformer/config.json` uses inference-compatible `attn_mode`: `torch` or `flashattn`, never training-only `flex`.

Recommended first adapted checkpoint:

```text
<ARTIFACT_ROOT>/LingbotVA-OPD/experiments/
place_a2b_left_success_path_v1_scaled_formal_20260825/
update/checkpoint_trajectory_update.pt
```

- Size: `243,058,729 bytes`
- SHA-256: `8572689fd8836699d1f3365c97e1446b5990a106e3b782943ab39cacc8b9eae5`
- Adapter kind: `joint_lora`
- Selected epoch: `3`

Also copy, do not recompute:

```text
update/summary.json
eval/screen/selection.json
eval/screen/summary.json
eval/heldout/summary.json
pipeline_state.json
```

Preserve the original checkpoint and SHA. If creating a smaller inference-only adapter by extracting `adapter_state_dict`, write it as a new derivative artifact with its own SHA and a receipt pointing back to the selected checkpoint.

## 5. The first required code change: load current JointLoRA in the server

This is the most important implementation gap.

`experiments/prototype_flashwam_robotwin_server.py --transformer-delta` currently supports older formats such as `flashwam_action_output_adapter_v1` and `flashwam_action_delta_v1`. It does **not** currently expose a verified loader for the success-path `joint_lora` checkpoint.

The simulator evaluator loads it through `NativeV0VideoRuntime`:

1. instantiate the released Student `VA_Server`;
2. call `attach_joint_lora(...)` with rank 8, alpha 8, dropout 0, blocks 0–29;
3. read the checkpoint mapping;
4. select `adapter_state_dict`;
5. load that state into the Transformer;
6. put the runtime in inference/eval mode.

The deployment server should implement the same path, with stronger checks:

- checkpoint `adapter_kind == "joint_lora"`;
- adapter contract matches rank 8 and all 30 blocks;
- base model parameter hashes/identity match the expected released Student;
- every expected LoRA tensor exists and has the expected shape/dtype;
- no unrecognized trainable adapter tensors are silently ignored;
- Teacher remains unloaded;
- output hashes on frozen input match the existing evaluator/runtime.

Do not merge adapter weights into the base checkpoint until the unmerged loader passes parity. Keeping the base and adapter separate makes Released/Adapted A/B comparison and rollback simple.

## 6. Exact model I/O contract that the robot adapter must preserve

### Observation cameras

The RoboTwin config expects these exact keys:

```text
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
```

Each value is an RGB `numpy` array in `H × W × 3`, effectively uint8 in `[0,255]`. The server resizes internally:

- high camera: `256 × 320`;
- each wrist camera: `128 × 160`;
- high and wrist features are combined into the model's RoboTwin T-shaped latent layout.

The deployment adapter must explicitly verify:

- RGB versus BGR channel order;
- image orientation and mirroring;
- timestamp synchronization across all three cameras;
- no stale or duplicated frame;
- camera intrinsics/extrinsics and physical viewpoint similarity;
- exposure/white-balance stability.

Do not silently substitute a missing camera with zeros unless a separately tested model contract allows it.

### Prompt and reset

At episode start, the client sends:

```python
{"reset": True, "prompt": instruction}
```

This clears Transformer/VAE cache, resets the frame position, and computes prompt embeddings. A changed task or instruction requires a reset. Never carry cache across episodes, task changes, emergency stops, or manual robot repositioning.

### Action output

The RoboTwin model has an internal 30-dimensional action representation, but the deployed config selects 16 physical channels in this order:

```text
left arm 7 + left gripper 1 + right arm 7 + right gripper 1
```

The server output is normally shaped:

```text
[16 channels, 2 model frames, 16 low-level actions per frame]
```

The official client treats each arm's 8-vector as:

```text
delta xyz (3) + delta quaternion (4) + gripper (1)
```

It composes pose deltas with the **episode-initial** end-effector pose, normalizes the resulting quaternion, and sends a 16-value dual-arm end-effector command to RoboTwin.

This behavior must be verified against the actual robot SDK. Do not assume the real controller accepts RoboTwin's `ee_action`, quaternion order, handedness, units, gripper range, or reference frame.

### Chunk and history semantics

- First policy chunk: frame 0 is conditioning and is skipped; execute frame 1 only (`16` low-level actions).
- Later chunks: execute both frames (`32` low-level actions).
- The official client captures new observations during execution and sends them back using `compute_kv_cache=True` together with the model-format action history.
- Current native history updates use the model's predicted/executed action representation, not an arbitrary robot joint-state vector.

Clone the official loop first. Do not simplify frame skipping, action-history normalization, or cache update cadence until parity tests establish that the change is benign.

The physical low-level control frequency is not defined by this handoff. Measure and configure it from the real robot/controller contract; never infer it from the model's `16` actions per frame.

## 7. Environment on the inference machine

The official LingBot-VA baseline specifies:

```text
Python 3.10.16
PyTorch 2.9.0
torchvision 0.24.0
CUDA 12.6
diffusers 0.36.0
transformers 4.55.2
websockets
einops
accelerate
msgpack
opencv-python
matplotlib
ftfy
easydict
flash-attn
```

Use a dedicated environment. Do not combine the robot vendor SDK and the heavy model environment unless dependency compatibility is proven. The intended architecture is a model server plus a lightweight robot-side client.

Official RoboTwin evaluation reports roughly `24 GiB` VRAM with VAE/text-encoder offload. The current OPD runtime has sometimes used substantially more memory during training, but deployment should be measured independently. Record:

- peak allocated/reserved VRAM;
- CPU RAM and offload traffic;
- reset latency;
- first-action latency;
- subsequent chunk latency;
- WebSocket round-trip latency;
- cache growth across a full episode.

## 8. Recommended deployment sequence

### Phase 0 — hardware inventory, no robot motion

Before writing the robot adapter, obtain explicit answers for:

- robot manufacturer/model and vendor SDK;
- single-arm or dual-arm embodiment;
- joint count, joint ordering, and limits;
- control command type: joint position, joint velocity, Cartesian pose, impedance, etc.;
- controller rate and interpolation behavior;
- base/world/tool/camera coordinate frames and quaternion convention;
- gripper command range and sign;
- head and wrist camera models, resolutions, frame rates, and timestamps;
- calibrated intrinsics/extrinsics;
- workspace, table, object geometry, and task instruction;
- available collision checking, force/torque sensing, E-stop, and deadman switch.

If these are unavailable, stop at offline replay. Do not guess them.

### Phase 1 — destination setup and artifact verification

1. Create the pinned Python/CUDA environment.
2. Copy base Student, selected adapter, source files, and summaries.
3. Verify all file hashes.
4. Verify inference `attn_mode`.
5. Load Released Student without the adapter.
6. Load Adapted Student with the new JointLoRA server path.
7. Confirm Teacher is not loaded.

### Phase 2 — deterministic offline parity

Use a frozen saved observation/prompt from the existing evaluation artifacts.

1. Run the source A800 runtime with fixed diffusion seed and save the action tensor/hash.
2. Run the destination server on the same observation, prompt, base checkpoint, adapter, and seed.
3. Require matching shape, finite values, cache position, and numerically equivalent action output.
4. Test reset, first chunk, cache update, and second chunk — a one-shot forward is insufficient.
5. Compare Released and Adapted outputs and confirm the adapter actually changes both relevant video/action behavior paths.

This golden-vector test should become a committed deployment test/receipt.

### Phase 3 — WebSocket loopback

Run server and client on the destination machine with synthetic/recorded observations:

- health check succeeds;
- msgpack preserves array dtype/shape;
- disconnect/reconnect forces reset;
- malformed payload is rejected;
- timeout produces a safe stop;
- no cache survives a client/session reset;
- server timing is logged.

### Phase 4 — hardware shadow mode

Connect real cameras and robot state, but keep motors disabled or command output disconnected.

Log for every proposed action:

- camera timestamps and hashes;
- prompt and episode/reset ID;
- raw model action;
- decoded Cartesian/joint target;
- reference frames and units;
- every safety clamp/rejection;
- inference and network latency;
- operator decision.

Visually inspect overlays/trajectories and replay logs offline. Shadow mode must cover resets, camera dropout, network timeout, stale data, and emergency stop.

### Phase 5 — guarded one-chunk live test

Only with human authorization and a physical E-stop:

1. empty or soft-object workspace;
2. low speed/acceleration/force limits;
3. small Cartesian workspace envelope;
4. one action chunk maximum;
5. preferably one arm or disabled inactive arm if the model/adapter supports a verified mask;
6. operator deadman required;
7. automatic stop after the chunk;
8. inspect logs and physical result before another command.

Do not begin with autonomous multi-chunk execution.

### Phase 6 — closed-loop task trial

Run paired Released versus Adapted trials only after the preceding gates pass. Keep starting state, prompt, object placement, cameras, safety settings, and operator procedure matched. Report failures, interventions, partial progress, and aborts; do not report only successful videos.

## 9. Mandatory safety supervisor

The policy server must never be connected directly to a robot actuator. Insert a deterministic supervisor that enforces at least:

- finite-value and exact-shape checks;
- quaternion normalization and convention checks;
- per-arm workspace bounds;
- joint position/velocity/acceleration/jerk limits after IK;
- Cartesian translation/rotation step limits;
- gripper range and rate limits;
- self-collision and environment-collision checks;
- force/torque limits where available;
- stale-camera and timestamp-skew rejection;
- inference deadline/watchdog;
- stop on network disconnect, server exception, cache mismatch, or operator release;
- explicit reset after every stop or manual reposition;
- an independent hardware E-stop.

Do not rely on RoboTwin's `check_success`, `take_action`, or simulator collision behavior as a hardware safety mechanism.

## 10. Network security warning

The bundled server defaults to `0.0.0.0`, uses unencrypted `ws://`, and does not enforce authentication. The client can send an `api_key` header, but the inspected server does not validate it.

Therefore:

- bind to `127.0.0.1` for same-machine deployment; or
- use a private isolated LAN/VLAN plus firewall; or
- tunnel through SSH/VPN;
- never expose the inference port to the public internet;
- do not treat the health endpoint as authentication;
- add authenticated transport before operating across an untrusted network.

## 11. Sim-to-real risks that must remain explicit

The adapter was trained from RoboTwin `demo_clean` trajectories. Major deployment shifts include:

- real camera viewpoints, lenses, latency, blur, exposure, and backgrounds;
- robot geometry and kinematics;
- Cartesian and quaternion conventions;
- action normalization statistics learned from RoboTwin;
- controller dynamics and timing;
- object appearance, scale, friction, and compliance;
- calibration drift and camera/arm synchronization;
- task-language mismatch.

The current q01/q99 normalization and channel mapping live in:

```text
third_party/lingbot-va/wan_va/configs/va_robotwin_cfg.py
```

Do not change these statistics merely to make commands look smaller. First reproduce the source policy exactly. If the real robot has a different embodiment/action distribution, that is a new data/model adaptation problem requiring a separately designed contract.

## 12. Definition of deployment-ready

The deployment is ready for a controlled real-robot pilot only when all are true:

- source/model/adapter hashes recorded;
- destination offline parity passes for at least two consecutive chunks;
- Released and Adapted can both be selected without changing code;
- all three camera streams are synchronized and validated;
- action channel order, units, frames, quaternion convention, and gripper mapping are documented and tested;
- safety supervisor unit/integration tests pass;
- timeout, disconnect, NaN, stale frame, and E-stop drills pass;
- one-chunk shadow/live review is signed off by the hardware operator;
- logs are sufficient to reconstruct every command and intervention;
- no Teacher weights or training-only components are loaded in deployment.

## 13. Active research runs — secondary context only

At the last snapshot, additional task adapters were still being produced:

- `stamp_seal`: epoch 3 update running;
- `place_a2b_right`: formal collection running on GPU0–1;
- `place_bread_basket`: formal collection running on GPU2–3.

These may later produce better or more relevant checkpoints, but they should not block building the generic deployment stack. The loader/server/safety architecture must accept a selected adapter path rather than hard-code one task checkpoint.

## 14. Suggested skills

- `robotics-ai-coding-flow`: use for implementing the robot-side observation/action adapter, deployment tests, and safety boundary.
- `codebase-design`: use if introducing a hardware abstraction layer; keep model serving, robot SDK, and safety supervision as separate modules.
- `diagnosing-bugs`: use for loader parity, WebSocket/cache problems, CUDA/offload failures, latency stalls, or robot-interface mismatches.
- `experiment-provenance-auditor`: use once to verify the deployment bundle, model/adapter hashes, and golden-vector receipts before live testing.
- `mattpocock-handoff`: use again if real-robot deployment moves to another operator or machine.

## 15. Immediate instructions for the receiving agent

1. Read this document and the referenced method files.
2. Inspect the destination machine and hardware; write down the missing inventory from Phase 0.
3. Do not issue robot commands yet.
4. Package/copy the exact base Student, selected adapter, runtime source, and summaries with hashes.
5. Implement the verified JointLoRA loader in the server with tests.
6. Produce a two-chunk offline golden-vector parity receipt.
7. Build the robot adapter and deterministic safety supervisor behind a mock actuator.
8. Ask the user/hardware owner to approve the first shadow-mode session and later the first one-chunk live test.

The correct first question on the new machine is not “which command starts the robot?” It is:

> What exact robot, camera, controller, coordinate-frame, network, and safety interfaces are available, and can we reproduce the selected policy offline before connecting any actuator?
