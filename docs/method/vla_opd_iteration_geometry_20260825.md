# VLA-OPD iteration geometry：fresh rollout、fresh labeling 与一次 aggregate update

Date: 2026-08-25
Scope: Zhong et al., *VLA-OPD* (`arXiv:2603.26666v1`) 的 paper-level training contract；不把本项目 WAM-OPD 实现反推成论文事实。

## Summary

VLA-OPD 的正式 Algorithm 1 是一个 **iterative fresh on-policy loop**：每个 outer iteration 用 current Student 重新 rollout，frozen Teacher 在这些新访问的 states 上重新给 action-distribution labels，然后对该批 `B × G × T` samples 做一次 aggregate policy-gradient update；更新后的 Student 进入下一轮并重新 collection。它不是“先 collect/label 一次，再对同一 package 连续训练 3 epochs”的算法几何。

论文没有披露 optimizer、numeric learning rate、minibatch schedule、同一 batch 的训练 epochs、replay/cache 或 checkpoint-selection rule。因而最严格的表述是：**paper algorithm 显示 one fresh collection/label batch → one parameter update → recollect；没有描述多 epoch label reuse**。官方 code 未发布，不能进一步断言隐藏的工程实现。

## Details

### 1. Algorithm 1 的一轮是什么

每个 `while not converged` iteration 依次执行：

1. 从 `D_prompt` sample 一个 prompt batch；
2. 对每个 prompt，由 current Student `πθ` 生成 `G` 条 trajectories；
3. 对每条 trajectory 的每个 timestep，在同一个 Student-visited state `s_{t,i}` 查询 Student/Teacher logits；
4. 计算 token-level negative Reverse-KL reward；
5. 将 prompt、group 和 timestep 上的梯度聚合后执行一次 `θ ← θ + α∇J`；
6. 回到 outer loop，由更新后的 current Student 重新 sampling。

Primary evidence: [Algorithm 1](https://arxiv.org/html/2603.26666v1#S3.SS1), PDF p.5；local extracted text `tmp/pdfs/vla-opd-2603.26666v1.txt:237-262`。Section 3.2 进一步明确：在每个 iteration `k` 执行 current policy `π_{θ_k}` 收集 `D_k`，其中 `a_t ~ π_{θ_k}(·|s_t)`，states 来自 current Student occupancy `d^{π_{θ_k}}`（[Sec. 3.2 / Eq. 3](https://arxiv.org/html/2603.26666v1#S3.SS2), PDF p.6；local text `:293-310`）。

因此，当参数从 `θ_k` 更新为 `θ_{k+1}` 后，下一轮的 trajectories 和它们对应的 Teacher queries 都应随新 occupancy 刷新。论文没有把旧 `D_k` 当作 `D_{k+1}`，也没有展示 replay loop。

### 2. Student Sampling 与 Teacher Labeling

- Environment 由 Student actions 推进；Teacher 不执行 action。
- Teacher frozen，并在每个 Student-visited state 上提供 dense token-level action logits/distribution。
- 因此 Teacher labeling 的对象是 current Student trajectory，而不是 Teacher trajectory，也不是原始 offline demonstrations。
- 论文明确说 SFT initialization 后丢弃 static offline dataset，切换到 dynamic on-policy sampling。

Primary evidence: Figure 1 caption, PDF p.4 / local text `:195-203`；Sec. 3.1, PDF p.5 / local text `:265-283`；Sec. 3.2, PDF p.6 / local text `:299-305`；Sec. 3.3 / Eq. 4, PDF p.6 / local text `:319-330`。

### 3. Student Optimization 与 objective

论文最大化 Student occupancy 上的 negative Reverse-KL：

```text
max_θ E_{s ~ d^{πθ}}[-D_KL(πθ(·|s) || πtea(·|s))].
```

对 Student-sampled action token `a_t`，其 dense reward 为：

```text
r_t^OPD = -(log πθ(a_t|s_t) - log πtea(a_t|s_t)).
```

policy-gradient 中 reward 内的 Student `log πθ` 使用 `stop_gradient`。Group estimator 对 `G` 条 trajectories 和所有 timesteps 求平均，并直接使用 raw Reverse-KL reward 作为 advantage；不是 sparse success reward，也不做 GRPO outcome normalization。

Primary evidence: [Sec. 3.4 / Eqs. 5–7](https://arxiv.org/html/2603.26666v1#S3.SS4), PDF pp.6–7；local text `:335-360` and `:385-397`。

### 4. Batch、group、steps 与未披露项

| Item | Paper fact |
| --- | --- |
| Main experiments | `B=64`, `G=8`，按 Algorithm 1 notation 为 nominal `512 trajectories/outer update` |
| Ablations | fixed `B=32`; group-size study uses `G ∈ {2,4,8}` |
| Visible iteration scale | LIBERO-Object 超过 90% within 10 steps；LIBERO-Long 约 50 steps 接近 baseline 150-step performance |
| Epochs over one `D_k` | 未披露；Algorithm 1 只画出一次 aggregate update |
| Optimizer / numeric LR / scheduler | 未披露；仅有 symbolic `α` |
| Replay / label cache / old-data mixing | 未披露；Algorithm 1 没有这些步骤 |
| Exact stopping / checkpoint selection | 未披露 |

Primary evidence: main `B,G`, PDF p.8 / local text `:431-442`；ablation settings, PDF pp.10–11 / local text `:546-550` and `:582-609`。

### 5. 与当前 WAM-OPD formal pipeline 的边界

当前 WAM-OPD protocol 是：一次 collection 得到 `12` 条 trajectories（`8 train + 4 calibration`），在同一个 labeled package 上训练 exactly `3 epochs`，保存三个 epoch checkpoints，之后 screening selection 和 exact-paired held-out。这是一个 **fixed-package multi-epoch update**：三个 epochs 是三个 optimization passes，不是三个 VLA-OPD outer iterations。

此外，两者 objective 也不同：VLA-OPD 蒸馏的是 action-token distribution 的 Reverse-KL policy gradient；本项目 JointLoRA/WAM training 使用自己的 video/action target 和 regression-style loss。故不能把当前三个 epochs 称为论文 Algorithm 1 的三轮 iterative OPD。

若以后做 VLA-OPD-style iterative ablation，正确的几何应是：

```text
π_k → fresh Student rollout D_k → fresh Teacher labels on D_k
    → one predeclared update block → π_{k+1} → fresh rollout D_{k+1}
```

这里的关键 refresh 是 **trajectory/state occupancy 与对应 labels 一起刷新**；不是对同一 trajectories 重复 labeling。至于一个 “update block” 内能否做多个 minibatch steps/epochs，论文没有给出工程答案，必须作为本项目自己的明确 design choice 报告。

## Next steps

保留当前 `fixed-package 3-epoch` formal run 作为独立 baseline，不中途改 protocol。若比较 iterative freshness，应另开 preregistered experiment，并匹配 total collection/label/optimizer-step budget；否则收益无法区分来自 fresh occupancy 还是更大的数据/计算量。

## Source

- Zhong et al., [*VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation*](https://arxiv.org/abs/2603.26666v1), arXiv:2603.26666v1.
- Local primary artifacts: `tmp/pdfs/vla-opd-2603.26666v1.pdf`, `tmp/pdfs/vla-opd-2603.26666v1.txt`.
