# Flash-WAM / LingBot-VA distillation and joint-training contract

Date: 2026-08-20
Scope: Flash-WAM paper v1, LingBot-VA paper v1, pinned first-party code, released checkpoint configs, and the current repository's Handover `multi_sigma_x0` implementation. This is a source check, not a new OPD recipe or an experiment result.

## Summary

The key distinction is:

```text
LingBot-VA native SFT/post-training
  = offline demonstration teacher forcing
  + joint video/action Flow Matching targets
  + full Transformer update

Flash-WAM native distillation
  = offline demonstration occupancy
  + frozen Teacher one-Euler bridge at random sigma
  + online Student versus EMA-target-Student consistency
  + modality-specific video/action consistency maps
  + full Transformer update

Current Handover OPD round
  = Student environment occupancy
  + full Teacher video endpoint z_T
  + full Teacher action endpoint a_T conditioned on saved Student plan z_S
  + three fixed-sigma endpoint x0 regression
  + all-30 shared-block JointLoRA only
  + one optimizer update
```

The current round therefore is **not a small-data reproduction of Flash-WAM's native distillation**. It is a different, environment-on-policy endpoint-imitation objective. In particular, its video target pushes the plan toward `z_T`, while its action target is valid under the old `z_S`. Updating one always-on shared adapter with both losses can make the action condition stale as soon as the video function changes.

Flash-WAM does not establish that video/action gradients never conflict. Its paper establishes a more specific failure: standard LCM gives the low-`sigma` action stream vanishingly small gradients, and Naive Joint LCM collapses. Flash-WAM fixes the per-modality consistency parametrization and demonstrates successful joint training at scale. Neither the paper nor public code reports video/action gradient cosine.

## Source/version contract

| Source | Version used | Evidence status |
| --- | --- | --- |
| Flash-WAM paper | arXiv:2606.05254v1, Sections 3.1–4.2, Appendix A.2–A.3 | primary paper |
| Flash-WAM official code | `NU-World-Model-Embodied-AI/Flash-WAM`, commit `5b8df13e9db24fb15ce42ff5ccc60a4015195960` | first-party code; remote vendored snapshot inspected at `<WAM_OPD_ROOT>/third_party/Flash-WAM` |
| LingBot-VA paper | arXiv:2601.21998v1, Sections 3.2–3.3, 4.2 | primary paper |
| LingBot-VA official code | `Robbyant/lingbot-va`, commit `58c2ae5bac46bd8114065bea9d7d256eb67c16c3` | pristine `git show HEAD:<path>` inspected; the remote worktree itself has local edits and was not used as source |
| Released Transformer configs | local LingBot-VA Teacher and FlashWAM-RoboTwin artifacts | both record 30 layers, hidden size `24*128=3072`, `action_dim=30`, and `attn_mode="torch"` |
| Current OPD mirror | files under `remote_patch/` in this repository | current local source |

Primary links: [Flash-WAM paper](https://arxiv.org/abs/2606.05254), [Flash-WAM code](https://github.com/NU-World-Model-Embodied-AI/Flash-WAM/tree/5b8df13e9db24fb15ce42ff5ccc60a4015195960), [LingBot-VA paper](https://arxiv.org/abs/2601.21998), [LingBot-VA code](https://github.com/Robbyant/lingbot-va/tree/58c2ae5bac46bd8114065bea9d7d256eb67c16c3).

## 1. LingBot-VA native SFT/post-training learns what

### Verified facts

LingBot-VA formalizes deployment as:

```text
z_future ~ p_theta(. | z_history, a_history)                         Eq. (8)
a_future ~ g_psi(. | z_future_hat, z_history, a_history)             Eq. (9)
```

Thus the action branch is an inverse-dynamics model conditioned on the predicted future video plan plus observation/action history. During native training, however, the model uses teacher forcing: ground-truth interleaved video/action tokens from demonstrations provide history. The two supervised targets are ordinary Flow Matching velocities:

```text
L_dyn = ||v_video(noised future video, history) - video flow target||^2    Eq. (11)
L_inv = ||v_action(noised action, current/next video, history) - action flow target||^2  Eq. (12)
L = L_dyn + lambda * L_inv
```

Source: LingBot-VA Section 3.2, Eqs. (8)–(9); Section 3.3, “Teacher Forcing” and Eqs. (11)–(12), paper pp. 5–8. The public code independently noises demonstration video and action, creates the native velocity target, sums video/action MSE, and calls one backward: [`wan_va/train.py` lines 167–248, 256–329](https://github.com/Robbyant/lingbot-va/blob/58c2ae5bac46bd8114065bea9d7d256eb67c16c3/wan_va/train.py#L167-L329).

The official trainer marks the entire Transformer trainable and passes all trainable parameters to AdamW; it does not use LoRA or freeze output decoders: [`wan_va/train.py` lines 71–118](https://github.com/Robbyant/lingbot-va/blob/58c2ae5bac46bd8114065bea9d7d256eb67c16c3/wan_va/train.py#L71-L118). The RoboTwin public config uses `lr=1e-5`, `beta=(0.9,0.95)`, weight decay `0.1`; its default source config says 50K steps: [`va_robotwin_train_cfg.py`](https://github.com/Robbyant/lingbot-va/blob/58c2ae5bac46bd8114065bea9d7d256eb67c16c3/wan_va/configs/va_robotwin_train_cfg.py). The paper Section 4.3.2 also reports RoboTwin training on 2,500 clean plus 25,000 randomized demonstrations for 50K steps.

### Important paper/code discrepancy

The LingBot-VA paper describes a dual-stream MoT: a 3072-wide video stream and a 768-wide action stream with separate per-layer projections (Section 3.3; Section 4.2). The released official source and the actual Teacher/Student configs used here instead instantiate one 3072-wide `blocks` stack shared by video and action, with modality-specific input/time/output modules:

```text
shared: blocks[0:30], norm_out, scale_shift_table, text embedding path
video-specific: patch_embedding_mlp, condition_embedder, proj_out
action-specific: action_embedder, condition_embedder_action, action_proj_out
```

Source: [`wan_va/modules/model.py` lines 597–651](https://github.com/Robbyant/lingbot-va/blob/58c2ae5bac46bd8114065bea9d7d256eb67c16c3/wan_va/modules/model.py#L597-L651). For decisions about the released checkpoints, the executable code/config contract is authoritative; it is unsafe to infer separate 768-wide action blocks from the paper alone.

### Deployment conditioning

Official serving first denoises video and writes the final video representation into the shared KV cache, then denoises action with the same cache. Therefore an independently run Student consumes `z_S`; an independently run Teacher consumes `z_T`: [`wan_va/wan_va_server.py` lines 443–570](https://github.com/Robbyant/lingbot-va/blob/58c2ae5bac46bd8114065bea9d7d256eb67c16c3/wan_va/wan_va_server.py#L443-L570). This is different from teacher-forced training, where the action target is conditioned on demonstration video tokens rather than a free-running Student plan.

## 2. Flash-WAM native distillation learns what

### 2.1 Occupancy and initialization

Flash-WAM's public trainer uses an offline `MultiLatentLeRobotDataset`, not simulator rollouts. The default points to `lerobot_robotwin_eef_aug_500`: [`distillation/config.py` lines 21–27](https://github.com/NU-World-Model-Embodied-AI/Flash-WAM/blob/5b8df13e9db24fb15ce42ff5ccc60a4015195960/distillation/config.py#L21-L27), [`distillation/trainer.py` lines 177–194](https://github.com/NU-World-Model-Embodied-AI/Flash-WAM/blob/5b8df13e9db24fb15ce42ff5ccc60a4015195960/distillation/trainer.py#L177-L194).

A fresh run creates three copies of the LingBot-VA Teacher checkpoint:

1. frozen Teacher;
2. full trainable online Student;
3. frozen target Student, updated by EMA.

The online Student is **initialized from the Teacher**, not from an already degraded 1v/1a Student. It calls `student.requires_grad_(True)` and AdamW receives every trainable parameter: [`distillation/trainer.py` lines 89–168](https://github.com/NU-World-Model-Embodied-AI/Flash-WAM/blob/5b8df13e9db24fb15ce42ff5ccc60a4015195960/distillation/trainer.py#L89-L168). Hence native Flash-WAM updates shared blocks, video/action input and time embedders, norms, and both output heads. It has no LoRA scope.

### 2.2 Target construction

For clean demonstration data `x0`, video/action are independently re-noised at random native schedule indices:

```text
x_sigma = (1-sigma) * x0 + sigma * epsilon
v_FM_target = epsilon - x0
```

The code samples one timestep per frame over the native 1000-step grid: [`distillation/data.py` lines 26–38, 85–113](https://github.com/NU-World-Model-Embodied-AI/Flash-WAM/blob/5b8df13e9db24fb15ce42ff5ccc60a4015195960/distillation/data.py#L26-L113). This is not a fixed set such as `[1,.5,.25]` and is not a clean endpoint-only target.

At each sampled `sigma_start`, the frozen Teacher makes one Euler bridge to `sigma_end`; the online Student predicts at the start, while the frozen EMA target Student predicts on the Teacher-bridged joint video/action state at the end:

```text
x_end = x_start + v_T(x_start, sigma_start, condition) * (sigma_end - sigma_start)
L_CD = d(f_online(x_start, sigma_start),
         stopgrad(f_EMA(x_end, sigma_end)))
```

Source: Flash-WAM Section 3.2, Eqs. (3)–(4); official [`distillation/step.py` lines 57–187](https://github.com/NU-World-Model-Embodied-AI/Flash-WAM/blob/5b8df13e9db24fb15ce42ff5ccc60a4015195960/distillation/step.py#L57-L187).

The modality-specific maps are:

```text
video:  f_v = c_skip(sigma) * x_sigma + c_out(sigma) * x0_hat       Eq. (10)
action: f_a = x_sigma - sigma * v_theta(x_sigma,sigma)              Eq. (9)
L = L_v + lambda_a * L_a                                            Eq. (12)
```

Flash-WAM proves that standard LCM has `|b(sigma)|=O(sigma^2)` near zero while the action map above has linear `|b(sigma)|=sigma`; this prevents the action learning signal from vanishing in the action stream's low-noise regime (Section 4.1, Proposition 1; Section 4.2, Eqs. (8)–(12)). The public `flashwam` mode also adds a masked demonstration action Flow Matching MSE regularizer: [`distillation/step.py` lines 189–224](https://github.com/NU-World-Model-Embodied-AI/Flash-WAM/blob/5b8df13e9db24fb15ce42ff5ccc60a4015195960/distillation/step.py#L189-L224).

### 2.3 What video latent conditions action during distillation

Public Flash-WAM training does not first generate a free-running `z_S` or `z_T`. `forward_train` concatenates these four offline tensors into one joint sequence:

```text
noised demonstration video
clean demonstration video context
noised demonstration action
clean demonstration action context
```

Teacher and online Student receive the same raw start tensors; the EMA target receives the Teacher-bridged video/action tensors. One shared block stack produces separate video/action outputs: [`wan_va/modules/model.py` lines 711–808](https://github.com/NU-World-Model-Embodied-AI/Flash-WAM/blob/5b8df13e9db24fb15ce42ff5ccc60a4015195960/wan_va/modules/model.py#L711-L808). Therefore the native Flash-WAM action loss is neither `a_T(z_T)` endpoint imitation nor `a_T(z_S)` environment-OPD labeling. It is joint consistency on an offline demonstration-derived noised state.

### 2.4 Steps, batch, and values that are not reproducibly pinned

Paper Appendix Table 6 reports AdamW, `lr=5e-6`, EMA `0.995`, Huber `c=.001`, action loss weight `1`, action regularizer weight `0.2`, and effective batch 48; Appendix A.2 reports 2,000 distillation steps. Public code defaults differ: 10,000 steps, batch 1, gradient accumulation 8, and action regularizer weight `0.01`: [`distillation/config.py` lines 91–124](https://github.com/NU-World-Model-Embodied-AI/Flash-WAM/blob/5b8df13e9db24fb15ce42ff5ccc60a4015195960/distillation/config.py#L91-L124).

The released Hugging Face Transformer does not expose optimizer state, selected step, full run config, or whether the released weights are the online or EMA Student. It is therefore valid to transfer the objective structure, but not valid to claim an exact reproduction using any one of these conflicting public values.

## 3. Why native joint training can work without proving “no conflict”

### Verified

- Both LingBot-VA SFT and Flash-WAM public distillation update the full Transformer, including modality-specific embedders/time modules/output heads. They are not constrained to one low-rank direction shared by both modes.
- Flash-WAM jointly presents paired video/action state and target tensors in the same training example and computes both targets from one forward per model (paper Section 4.2, Eqs. (11)–(12)).
- The online and EMA Students start at the strong Teacher function; distillation learns a reduced-NFE consistency map rather than learning the task policy from a weak checkpoint.
- Flash-WAM trains over random sigma, many batches, and many optimizer steps. The paper reports 2,000 steps/effective batch 48, while public defaults are larger/different.
- Flash-WAM's Naive Joint LCM ablation collapses (paper Tables 1 and 4). Shared-backbone joint optimization is therefore not automatically safe; the per-modality objective matters.

### Inference, not a paper-proven mechanism

These differences plausibly make native joint optimization much easier: Teacher initialization keeps both modes in a compatible basin; full parameters provide modality-specific escape directions; paired random-sigma supervision averages gradients over the data/path distribution; EMA reduces target motion; the action-specific map corrects a severe gradient-scale imbalance.

None of those facts proves non-negative video/action gradient cosine. Neither paper reports per-module gradient cosine or establishes a no-conflict theorem. “Flash-WAM/SFT works, therefore our two losses cannot conflict” is unsupported. The correct claim is narrower: **their training system tolerated any remaining interference at their data, update-scope, and optimization scale**.

### Public-code conditioning caveat

`forward_train` constructs a `FlexAttnFunc` mask, but the model and released configs default to `attn_mode="torch"`; `custom_sdpa` calls `scaled_dot_product_attention` without a mask. Source: [`model.py` lines 37–40, 298–320, 774–789](https://github.com/NU-World-Model-Embodied-AI/Flash-WAM/blob/5b8df13e9db24fb15ce42ff5ccc60a4015195960/wan_va/modules/model.py). This is a verified mismatch in the public default code path, not proof of how the unpublished release run was executed. It is another reason not to copy the trainer without an execution-level mask check.

## 4. What the current Handover round actually optimized

The preregistered config uses one failed Student trajectory, rank-8 JointLoRA on blocks 0–29, fixed sigmas `[1,.5,.25]`, equal video/action weights, no action velocity regularizer, and one round: [`handover_mic_multi_sigma_trajectory_update_20260820.json`](../remote_patch/configs/handover_mic_multi_sigma_trajectory_update_20260820.json).

The always-on JointLoRA freezes the base and wraps, in every selected block, Q/K/V/O of both `attn1` and `attn2` plus MLP up/down. It does not adapt `patch_embedding_mlp`, `action_embedder`, either time embedder, `norm_out`, `proj_out`, or `action_proj_out`: [`joint_lora.py` lines 25–61, 121–175](../remote_patch/originals/joint_lora.py). The resolved target list is defined in remote `experiments/video_mode_lora.py:164–197`; with all 30 blocks it comprises 10 Linear targets per block.

For each label, current training explicitly loads:

```text
saved_student_plan = student_z_s
video target       = teacher_z_t
action target      = teacher_action
```

It trains Student video x0 toward `teacher_z_t`, while Student action is evaluated after reinjecting the saved `student_z_s` and trained toward the full Teacher action endpoint: [`train_iterative_on_policy_flow_opd.py` lines 1053–1209](../remote_patch/experiments/train_iterative_on_policy_flow_opd.py). Collection verifies that the Teacher action consumed exact Student `z_S`, with the same action noise/timestep/mask, and that only Student actions controlled the simulator: [`train_iterative_on_policy_flow_opd.py` lines 717–837](../remote_patch/experiments/train_iterative_on_policy_flow_opd.py). This part is a real `ST` exact-condition action label.

But the loss is **endpoint x0 regression after re-noising the endpoint**, not Flash-WAM's Teacher one-Euler bridge plus EMA endpoint consistency. It also supplies only three sigma values, one trajectory, and one optimizer update. Calling both methods “multi-sigma distillation” does not make their targets equivalent.

## 5. The concrete joint-target mismatch in this round

At collection time the action label is:

```text
a_label = a_T(history_S, z_S_old)
```

At the same time the video loss pushes:

```text
z_S_old -> z_T
```

Because the same JointLoRA is always active in both video and action calls, the joint update changes the plan-producing function and the action function together. After the update, deployment uses a new `z_S_new`, but the saved action label remains `a_T(z_S_old)`. Thus joint video/action endpoint fitting does not preserve the exact-condition contract beyond the point at which the label was collected.

This is a verified dependency plus a causal inference, not a measured explanation of the `0/8` behavioral result. The current evidence does **not** establish that this mismatch was dominant; data amount, one-step under-training, target quality, and update scope remain live variables.

## 6. Evidence-bounded answer to “what should learn what?”

1. For environment OPD action transfer, the Student action function should learn the Teacher action/vector field on **Student-visited histories and the exact deployed Student plan/state**: `Teacher(history_S, z_S, x^a_sigma, sigma) -> Student(same inputs)`. The current collector gets the environment/history/plan part right at collection time.
2. For Flow Matching, the strongest architecture-matched target is a local Teacher vector field or Teacher bridge on the same `(x_sigma, sigma, condition)`, not merely a clean full-solver endpoint repeated at several sigmas.
3. If video and action are updated jointly, their labels must form one coherent joint path and remain valid under the update. Native Flash-WAM does this on an offline Teacher-bridged joint state. In online OPD, this requires either fresh joint labels after policy updates or an alternating contract that freezes the condition-producing branch while consuming its labels.
4. Updating only all-30 shared-block LoRA is a legitimate capacity hypothesis, not something supported by native Flash-WAM. Native LingBot-VA SFT and Flash-WAM update the full model, including modality-specific modules. A controlled scope comparison is required before claiming shared LoRA is sufficient; the papers do not determine whether heads/embedders must be updated from a released weak Student.
5. No deterministic conclusion about this one negative round is justified. It simultaneously changed occupancy, target semantics, update scope, data volume, and optimizer regime relative to native Flash-WAM. The result rules out only the exact tested package as a demonstrated behavioral improvement.

## Verified facts versus unknowns

| Statement | Status |
| --- | --- |
| LingBot-VA SFT uses paired demonstration video/action Flow Matching losses | paper + official-code verified |
| LingBot-VA/Flash-WAM released code uses one shared 30-block stack with modality-specific I/O modules | official-code + artifact-config verified |
| Native Flash-WAM starts online and EMA Students from the Teacher | official-code verified |
| Native Flash-WAM updates all online-Student parameters, not LoRA only | official-code verified |
| Native Flash-WAM uses random-sigma Teacher Euler bridge plus EMA consistency | paper + official-code verified |
| Native Flash-WAM action is explicitly conditioned on deployed Student `z_S` | false for the public training path; it uses offline joint demo-derived tensors |
| Current OPD action label is Teacher-on-exact-Student-`z_S` | current-code verified |
| Current OPD video/action targets are the same objective as Flash-WAM | contradicted by source |
| Flash-WAM has no video/action gradient conflict | unknown; not measured/reported |
| Gradient conflict caused the Handover behavioral failure | unknown |
| One trajectory/one update was insufficient | plausible but unproven |
| Released FlashWAM-RoboTwin used public-code defaults or the exact Appendix values | unknown |
| Released checkpoint is online Student or EMA Student | unknown |
