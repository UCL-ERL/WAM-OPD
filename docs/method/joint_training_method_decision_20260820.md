# FlashWAM joint video/action post-training：method decision

Date: 2026-08-20
Status: method frozen for implementation review; no rollout, training, or GPU job was started.

## Summary decision

The next implementation should train **video and action jointly**, but it must not reuse the
previous mixed endpoint objective. A shared joint DiT is trainable because the two modes implement
different conditional functions through modality-specific inputs, time conditions, token layouts,
and output heads; it is not because their gradients are guaranteed to agree.

The first fixed baseline is:

```text
Student-controlled Easy environment histories
+ coherent Teacher pair z_T, a_T(h_S, z_T)
+ Flash-WAM modality-aware video/action consistency
+ joint LoRA update
```

The accurate name is:

```text
environment-on-policy coherent forward consistency distillation
```

It is not exact Reverse-KL OPD because the action branch is teacher-forced on `z_T` rather than
averaged over current Student plans `z_S`.

The later, separately evaluated OPD method is:

```text
Student-plan-conditioned on-policy coarse-transition distillation
```

It compares the deployed Student one-step transition with a frozen Teacher transition integrated
over the same interval and condition. It must not directly match the released `1v/1a` Student's raw
velocity to the slow Teacher's instantaneous vector field.

## 1. Why other joint DiTs can train

For the released shared-stack implementation, write the two deployed functions schematically as:

```text
z = F_theta(x_video, t_video, h; video-specific I/O)
a = F_theta(x_action, t_action, h, z; action-specific I/O)
```

The shared parameters receive

```text
g_shared = g_video + lambda_action * g_action,
```

while modality-specific parameters receive only their own branch's gradient. The two calls need not
represent the same function, so there is no logical contradiction. However,
`cos(g_video, g_action) < 0` is possible in the shared blocks. None of LingBot-VA, Flash-WAM, UVA,
UWM, or UNIVERSE reports this cosine or proves a no-conflict theorem.

The source-supported reasons their full systems can converge are:

1. **Coherent targets.** Video and action describe the same demonstration future or the same
   jointly bridged state.
2. **Private capacity.** Successful systems preserve modality-specific inputs and outputs, and often
   private heads or experts, around a shared core.
3. **Controlled train-time dependency.** They use teacher forcing, masking, independent noise
   times, or marginalization; they do not silently condition action on an old free-running plan while
   simultaneously changing the plan generator.
4. **Modality-aware objectives.** Flash-WAM uses different video and action consistency maps. Its
   Naive Joint LCM ablation collapses, directly showing that a formally joint loss can still be wrong.
5. **Initialization and scale.** Native Flash-WAM starts online and EMA Students from the strong
   Teacher and trains for thousands of steps; LingBot-VA trains broadly for thousands to tens of
   thousands of steps, not one zero-up LoRA update.

Primary evidence:

- [LingBot-VA paper](https://arxiv.org/abs/2601.21998) and
  [official trainer](https://github.com/Robbyant/lingbot-va/blob/58c2ae5bac46bd8114065bea9d7d256eb67c16c3/wan_va/train.py#L167-L329)
- [Flash-WAM paper](https://arxiv.org/abs/2606.05254) and
  [official distillation step](https://github.com/NU-World-Model-Embodied-AI/Flash-WAM/blob/5b8df13e9db24fb15ce42ff5ccc60a4015195960/distillation/step.py#L57-L224)
- [UVA paper](https://arxiv.org/abs/2503.00200)
- [UWM paper](https://arxiv.org/abs/2504.02792) and
  [official joint model/loss](https://github.com/WEIRDLabUW/unified-world-model/blob/b7d4067c8859775e5b08f681e76d538015cdd445/models/uwm/uwm.py#L92-L319)
- [UNIVERSE paper](https://arxiv.org/abs/2607.05133)

## 2. Why the previous objective is not comparable

The previous round combined:

```text
video target  = z_T
action target = a_T(h_S, z_S_old)
update        = one always-on shared LoRA bank changes both functions
deployment    = action consumes z_S_new
```

This pair is not the coherent Teacher pair used by SFT/Flash-WAM and is not a verified joint
Reverse-KL estimator. It also used fixed `[1, 0.5, 0.25]` endpoint re-noising, rather than the native
random schedule, Teacher bridge, and EMA consistency target. With `A=random, B=0`, the single
optimizer step changed only `B`; it did not test full rank-8 adaptation dynamics.

Therefore the old negative result rejects only that package. It does not show that joint training,
rank 8, or OPD is intrinsically ineffective.

## 3. Fixed baseline target contract

### Collection

1. The current Student exclusively controls RoboTwin `demo_clean / Easy` environment transitions.
2. The frozen Teacher is label-only and never advances the environment.
3. Collect complete native trajectories. Keep the success-triggering action; create no labels or
   loss after first native success/terminal.
4. Save exact causal history, paired video/action noise, native timestep/scheduler metadata, masks,
   cache layout, and terminal mask.
5. At each retained Student history `h_S`, generate one coherent pair:

   ```text
   z_T = TeacherVideo(h_S)
   a_T = TeacherAction(h_S, detach(z_T)).
   ```

### Consistency objective

1. Treat `(z_T, a_T)` as a coherent Teacher pseudo-demonstration on `h_S`.
2. Re-noise each modality with its native shifted schedule; do not use the old three fixed sigmas.
3. Use the frozen Teacher Euler bridge and an EMA target Student as in Flash-WAM.
4. Use the Flash-WAM video consistency map for video and
   `f_a(x_sigma, sigma) = x_sigma - sigma * v_theta` for action.
5. Condition Teacher, online Student, and EMA action calls on the same detached `z_T`.
6. Fix the first baseline to the published Flash-WAM weighting structure:

   ```text
   L = L_CD_video + L_CD_action + 0.2 * L_FM_action
   ```

   where both action terms use the same clean pseudo-target `a_T(h_S,z_T)`. The `0.2` anchor
   weight is the paper Appendix value; the public-code default differs, so this is a declared
   baseline choice rather than a claim about the released checkpoint.
7. Do not add `ST`, retention-to-weak-Student, PCGrad, or an unregistered loss term to this baseline.

This is an architecture-specific adaptation of Flash-WAM from offline demonstrations to
Student-visited environment histories. No primary source has already validated this exact
composition, so empirical success remains to be established.

## 4. LoRA scope decision

### Source boundary

| System | Released trainable scope |
| --- | --- |
| DiffusionOPD / Flow-OPD | depth-wide attention LoRA, `r=32`, `alpha=64`; no FFN/head/time LoRA |
| openpi pi0 low-memory | separate expert-private attention+FFN LoRA, ranks 16 and 32 |
| OpenVLA-OFT | `all-linear` VLA LoRA, rank 32, plus fully trainable private action head/projectors |
| LingBot-VA / Flash-WAM | full online Transformer, including modality-specific paths |
| VLA-OPD | scope not disclosed; official code unavailable |

Sources:

- [DiffusionOPD official LoRA scope](https://github.com/ali-vilab/DiffusionOPD/blob/0c85bf0193a6da24bb1f72152b612d353a154999/scripts/train_sd3_opd.py#L659-L674)
- [Flow-OPD official LoRA scope](https://github.com/CostaliyA/Flow-OPD/blob/434ab16911fc580f51e22107151915846222eb6a/scripts/train_sd3_opd_mix.py#L755-L763)
- [openpi official repository](https://github.com/Physical-Intelligence/openpi)
- [OpenVLA-OFT official LoRA setup](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/vla-scripts/finetune.py#L846-L935)

### Current architecture

The existing all-30 shared rank-8 JointLoRA covers 300 shared Linear modules:

```text
attn1 Q/K/V/O
attn2 Q/K/V/O
FFN up/down
across blocks 0..29
```

It has about 20.15M trainable parameters and excludes embeddings, timestep conditioners, norms,
`proj_out`, and `action_proj_out`.

### Decision

LoRA is retained. Rank 8 and all 30 shared blocks remain fixed for the first controlled capacity
check; neither rank nor time/input scope is changed yet. Before an expensive behavior round, run one
cached-data scope comparison under the **same coherent target, batches, optimizer, and steps**:

```text
S0 = all-30 shared rank-8
S1 = S0 + independent proj_out/action_proj_out adapters
```

Select the smallest scope that reduces both video and action loss on whole-trajectory calibration
data without persistent branch starvation; then freeze that scope for the behavior experiment. This
is the only parameter-scope decision, not an open-ended sweep.

Do not add timestep LoRA, change rank, or introduce separate full dual-mode banks in the same test.
If both S0 and S1 demonstrably lack capacity after multiple optimizer steps update both `A` and `B`,
the next isolated comparison is rank 8 versus rank 16.

## 5. Minimal execution checks before training

These are implementation correctness conditions, not a general audit:

1. the actual training attention/cache path matches the intended causal dependency; public
   Flash-WAM creates a FlexAttention mask while released configs default to `attn_mode="torch"`;
2. all Teacher/Student calls use identical noise, tensor layout, mask, timestep convention, and
   conditioning object required by the chosen TT contract;
3. the shared adapter is active in both video and action calls; private head adapters activate only
   on their modality;
4. base parameters remain frozen, while LoRA parameters and optimizer states remain FP32;
5. multiple optimizer steps update both LoRA `A` and `B` factors;
6. split train/calibration by whole trajectory, not by neighboring macros;
7. use one inner epoch over multiple independent trajectories, then recollect with the updated
   Student rather than repeatedly fitting a stale batch.

Before engineering changes, re-read the user-provided `CODEX2.md` and inspect every reused code
path at execution level.

## 6. Empirical acceptance boundary

Source review can establish that the method is coherent; it cannot guarantee a success-rate gain.
The next pilot is accepted only if all of the following hold:

1. both held-out video and action surrogate losses improve;
2. neither branch's gradient is persistently starved; log norms and shared-block cosine as a small
   diagnostic, not as a new optimizer method;
3. exact paired Easy evaluation over accepted RoboTwin seeds and multiple locked noise banks shows
   more rescues than regressions;
4. improvement persists after fresh Student recollection rather than only on cached histories;
5. Randomized is used only after Easy improvement, as robustness evaluation.

## 7. Later ST research method

After the TT baseline establishes joint capacity, evaluate `ST` separately:

```text
h_S, z_S = current deployed Student occupancy
a_S      = Phi_S,1step(epsilon_a; h_S, z_S)
a_T      = Phi_T,multistep(epsilon_a; h_S, z_S)
L_action = d(a_S, stopgrad(a_T))
```

Use the analogous deployed coarse transition for video, detach collected conditions, bound the
inner update, and recollect after every round. This preserves the accelerated `1v/1a` object while
placing the Teacher query on Student plan support.

Do not replace `a_T` with the Teacher's first-step raw velocity. The slow Teacher instantaneous
field generally does not equal its integrated endpoint transition.

The claim remains “on-policy coarse-transition distillation” until the common transition kernel,
covariance, and scheduler assumptions required for exact transition-KL/Reverse-KL are derived.

## Related evidence notes

- [Joint video/action DiT evidence](./joint_video_action_dit_training_evidence_20260820.md)
- [LoRA scope audit](./lora_scope_for_joint_flow_vla_20260820.md)
- [Joint objective red-team](./joint_opd_consistency_contract_redteam_20260820.md)
- [OPD primary-source contract](./opd_primary_source_training_contract_20260820.md)
- [Flash-WAM/LingBot-VA contract](./flashwam_lingbot_distillation_contract_20260820.md)
