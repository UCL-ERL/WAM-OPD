# LingBot-VA + Flash-WAM：面向 World-Action Model 的 On-Policy Distillation

## 0. 项目定位

本项目希望研究一种面向 **World-Action Model（WAM）** 的强化学习式后训练范式：

> 让一个高计算量、强性能的 full-step WAM teacher，在一个 few-step WAM student 自己访问到的环境状态和生成状态上提供 dense supervision，从而提升 student 的闭环任务成功率，同时保留 few-step inference 的实时性。

当前选定的第一组模型是：

- **Teacher**：LingBot-VA full-step RoboTwin checkpoint；
- **Student**：官方 released Flash-WAM RoboTwin checkpoint，项目中统一按
  **1 video step / 1 action step（1v/1a）** 使用；
- **Benchmark**：RoboTwin 2.0，优先关注 Randomized split 和 long-horizon tasks；
- **核心方法方向**：Environment-on-policy + denoising-on-policy 的 modality-aware transition distillation；
- **不依赖**：privileged simulator reward、额外 reward model、重新训练 planning model；
- **主要监督**：teacher 在 student-induced states 上提供的 video/action transition target；
- **环境 0/1 success**：主要用于评价、数据审计和可选的安全过滤，而不是作为唯一训练信号。

这不是一个已经验证有效的方法，而是一条具备现成 teacher/student、公开训练代码和明确性能缺口的研究路线。当前最重要的工作，是把“普通在线 teacher relabeling”逐步提升为具有 WAM 结构特异性和数学解释的 on-policy transition distillation。

> 2026-07-29 operating-point 决策：论文中的 1v/2a 只保留为公开结果和
> counterfactual 诊断，不再把 released 1v/1a 权重强制跑 2a 后称为 1v/2a
> checkpoint。当前主线 teacher/student budget 固定为 25v/50a → 1v/1a。

---

## 1. 研究动机

### 1.1 为什么想给 WAM 做后训练

现有 VLA 的 RL post-training 已经出现 PPO、GRPO、AWR、OPD 等工作，但真正针对 joint video-action WAM 的后训练仍非常少。WAM 与普通 VLA 的区别是：

- 不只输出 action；
- 还显式生成 future video/world latent；
- video 与 action 往往通过 shared DiT、joint attention 或 sequential conditioning 耦合；
- 两种模态可能使用不同的 flow-matching noise schedule；
- action 的任务价值与 video/world prediction 的物理一致性并不完全等价。

因此，WAM 后训练不能只照搬 token policy 的 RL objective，也不能简单把所有 joint outputs 当成同一种 diffusion variable。

### 1.2 从 DreamZero RL 得到的启发

前期围绕 DreamZero 的讨论暴露了几个普遍问题：

- action-head-only 更新范围较小，性能提升可能受限；
- 直接更新 shared joint DiT 又会遇到 video/action gradient interference；
- 任务奖励通常只有 episode-level 0/1 success，credit assignment 很弱；
- PPO/GRPO 需要定义可用的 action/process likelihood；
- joint video-action policy 的 log-probability 与实际执行 action 的概率并不天然等价；
- online success 上升不一定能转化为 held-out evaluation 提升；
- 如果 imagined video 没有被真实环境或强 teacher 校准，模型可能只改善 action，或者产生“看起来合理但不真实”的计划。

这些问题促使研究方向从“如何强行对 joint WAM 做 PPO/GRPO”扩展到：

> 能否使用一个更强 teacher，为 student 在自身访问状态上的每个决策和 denoising transition 提供 dense、低方差监督？

这正是 On-Policy Distillation（OPD）最有吸引力的地方。

### 1.3 为什么 OPD 比只用 0/1 reward 更适合这个问题

如果只使用任务成功的 0/1 reward：

- 失败 episode 内部没有明确的局部纠错方向；
- long-horizon task 的 credit assignment 很困难；
- critic 容易学习 task difficulty 或 state progress，而不是真正的 action quality；
- PPO/GRPO 的梯度方差较大；
- 对 video/world branch 很难分配合理的 task-value signal。

OPD 则把 teacher 当成一个 dense process supervisor：

- student 先产生自己的状态分布；
- teacher 不只评价最终成功与否，而是在 student 访问的每个 state 上给出“teacher 会怎样生成下一步”的局部信息；
- 对 flow/diffusion model，可以进一步在 noisy latent transition 上比较 teacher/student vector field 或 transition mean；
- 不需要把 episode-level reward硬分摊给每一个 video/action token。

它不是传统意义上依赖环境 reward 的 RL，但与 RL 的核心联系在于：

- 训练数据来自当前 student policy 的 state visitation distribution；
- teacher supervision 作用在 student-induced states，而不是固定 offline demonstrations；
- policy 更新会改变下一轮采集到的数据分布；
- 最终优化目标仍是闭环任务表现，而不是单纯拟合离线数据。

---

## 2. Ideation 的收敛过程

### 2.1 初始方向：任意 WAM + OPD

最初的问题不是“必须使用 LingBot-VA”，而是：

> 哪种 WAM 最适合立刻开展 OPD，既有足够强的 teacher，又有明显较弱的 student，并且二者的生成状态和训练接口能够对应？

为此考虑过几类组合。

### 2.2 DreamZero 14B teacher → DreamZero 5B student

优点：

- 同一模型 family；
- 都是 joint video-action flow model；
- 与已有 DreamZero infra 接近。

问题：

- 需要重新核对 14B/5B 的公开 checkpoint、benchmark specialization 和 action interface；
- hidden dimension、parameterization 和内部 latent trajectory 不一定完全一致；
- 不同尺寸模型做 process-level OPD 可能需要 representation/transition adapter；
- 5B student 并不是天然的 low-NFE student，研究问题容易变成“大模型蒸馏到小模型”，而不是 WAM few-step on-policy correction。

因此它仍然可以作为后续方向，但不是最容易启动的第一组实验。

### 2.3 StarWAM teacher → DreamZero student

优点：

- StarWAM 与 DreamZero 都包含 joint video-action modeling；
- StarWAM 有 LIBERO/RoboTwin 相关实现参考。

问题：

- 两者不是同一 backbone；
- camera preprocessing、action normalization、video latent、causal structure都可能不同；
- teacher/student 很难在完全相同的 noisy latent state 上定义 transition KL；
- 更适合 behavior-level imitation，不适合作为第一篇 WAM process-level OPD 工作的干净对象。

### 2.4 Cosmos planning mode → action mode

优点：

- planning model 和 action model之间存在自然的能力层级；
- Cosmos提供较完整的世界生成和机器人生态。

问题：

- 缺少一个开箱即用、同任务域训练好的 planning/action teacher-student pair；
- 可能需要先为 RoboCasa 或其他 benchmark 训练 planning model；
- teacher/student surface不同，不能直接复用同一个 action likelihood 或 diffusion transition；
- 工程成本和研究变量过多。

因此 Cosmos 更适合后续异构 teacher/student 扩展，不适合马上验证核心假设。

### 2.5 收敛到 LingBot-VA → Flash-WAM

LingBot-VA 与 Flash-WAM 提供了一个几乎为本项目量身定制的组合：

- Flash-WAM 本来就是从 LingBot-VA step-distill 得到；
- teacher/student 参数结构相同；
- VAE、text encoder、camera layout、action space和 normalization相同；
- student 从 teacher 初始化；
- 两者在同一个 continuous flow timeline 上定义 velocity；
- Flash-WAM 已公开 frozen teacher、trainable student、EMA target 和 modality-aware consistency distillation代码；
- 两者共用 RoboTwin server/client evaluation pipeline；
- Flash student 保留了明显的 teacher-success gap，尤其是 Randomized 和 long-horizon tasks。

因此 ideation 最终从泛化的“WAM + OPD”收敛为：

> 使用 full-step LingBot-VA teacher，在 few-step Flash-WAM student 自己访问的环境状态和 denoising states 上进行 modality-aware teacher correction，研究 aggressive step distillation 丢失的闭环能力能否被 on-policy transition distillation 恢复。

---

## 3. 必须澄清的模型版本问题

### 3.1 论文中的 separated dual-stream LingBot-VA

新版 LingBot-VA 论文描述的是 separated Mixture-of-Transformers：

- Video stream：
  - 初始化自 Wan2.2-5B；
  - hidden dimension 约 3072；
  - 30 layers。
- Action stream：
  - 与 video stream 同深度；
  - hidden dimension 约 768；
  - 独立 action transformer blocks。
- 每层先在各自空间计算 Q/K/V；
- action token投影到video维度参与 joint attention；
- 再投回action空间形成 residual；
- 最后使用独立 video/action output head。

### 3.2 目前公开 checkpoint 是 shared-backbone 版本

公开的以下 checkpoint：

- `robbyant/lingbot-va-base`
- `robbyant/lingbot-va-posttrain-robotwin`
- `robbyant/lingbot-va-posttrain-libero-long`

目前都是 shared-backbone version。

实际公开代码结构为：

```text
video latent ─ video patch embedding ─┐
                                     │
action latent ─ action embedding ─────┼─ interleaved joint sequence
                                     │
text embedding ──────────────────────┘
                       ↓
             shared 30-layer Wan DiT
          hidden=3072, heads=24, FFN=14336
                       ↓
              ┌────────┴────────┐
              ↓                 ↓
        video output head   action output head
           48D latent          30D action
```

另外：

- video/action有独立的 input embedding；
- video/action有独立 timestep conditioning；
- video/action token进入同一组 Transformer blocks；
- video/action有独立 output projection；
- 历史 token通过 causal attention 和 KV cache维护。

Flash-WAM论文和公开代码使用的正是这个 shared-backbone LingBot-VA。

因此本项目第一版必须使用公开 shared-backbone teacher/student，而不能按新版 separated MoT论文图直接重写模型。

---

## 4. LingBot-VA 的 policy 生成过程

LingBot-VA不是一个只根据当前图像直接输出 action chunk 的普通 VLA。

一次 decision大致为：

```text
current observation + language + history/KV cache
                         ↓
          multi-step future-video denoising
                         ↓
               predicted future video
                         ↓
     multi-step action denoising / inverse dynamics
                         ↓
                  action chunk
                         ↓
                execute in environment
                         ↓
       receive real observation and refresh history
```

核心因果关系是：

\[
\hat z_{t+1:t+K}
\sim
p_\theta(z\mid C_t)
\]

\[
a_{t:t+K-1}
\sim
\pi_\theta(a\mid C_t,\hat z_{t+1:t+K})
\]

其中 \(C_t\) 包含：

- task language；
- observation history；
- executed action history；
- causal token positions；
- KV-cache所代表的长期上下文。

这意味着 action teacher query并不只取决于真实 observation，还取决于“使用谁生成的 future video”。

---

## 5. Flash-WAM 的结构与作用

### 5.1 Flash-WAM不是小参数模型

Flash-WAM保留 LingBot-VA 的约5B参数 shared DiT。它的加速来自减少 Number of Function Evaluations（NFE），而不是减少参数量。

常见配置为：

- Teacher：约25 video steps / 50 action steps；
- Flash student：1 video step / 2 action steps；
- 更激进 student：1 video step / 1 action step。

因此研究目标不是“大模型压缩到小模型”，而是：

> 恢复 aggressive step distillation 在闭环控制中丢失的能力，同时保持 few-step inference。

### 5.2 为什么要 modality-aware distillation

Video 与 action 使用不同的 flow-matching noise regime：

- Video SNR shift约为5.0；
- Action SNR shift约为1.0。

Video训练样本更集中于高噪声区域；action在低噪声区域也有大量训练质量。普通 LCM consistency function 在低噪声 action区域容易产生过弱的 gradient。

Flash-WAM因此使用：

- Video：variance-preserving / Karras-style consistency function；
- Action：linear-gradient-scaling clean prediction

\[
\hat x_0^a=x_\sigma^a-\sigma v_\theta^a(x_\sigma^a,\sigma)
\]

并组合：

\[
\mathcal L_{\mathrm{Flash}}
=
\mathcal L_{\mathrm{video-consistency}}
+
\lambda_a\mathcal L_{\mathrm{action-consistency}}
+
\lambda_r\mathcal L_{\mathrm{action-FM-anchor}}.
\]

公开实现同时维护：

- frozen LingBot teacher；
- trainable Flash student；
- frozen EMA target student。

而且 student 默认是 full-parameter training，不只是 LoRA 或 action head。

---

## 6. 为什么这组 teacher/student 适合 OPD

### 6.1 相同的参数空间

Teacher/student具有相同：

- model class；
- parameter key和shape；
- hidden dimension；
- number of layers；
- video/action token layout；
- VAE latent space；
- text representation；
- action normalization；
- action mask。

因此可以直接比较：

- video velocity；
- action velocity；
- predicted clean video；
- predicted clean action；
- per-layer hidden states；
- attention/KV behavior。

不需要额外 feature projector 或 cross-model adapter。

### 6.2 相同的 continuous noise timeline

Flash-WAM training仍使用 LingBot原生的1000-step FlowMatch timeline。25/50-step teacher schedule与1/2-step student schedule，本质上是从同一个 continuous timeline选取不同离散点。

这使得 teacher可以在：

```text
相同 noisy latent
相同 flow time
相同 environment context
```

上被查询。

但“能在同一个 \(x_\sigma\) 上调用”不等于“teacher/student离散 transition kernel天然相同”，这正是本项目需要处理的数学问题。

### 6.3 有现成且明确的 capability gap

RoboTwin Randomized split 的公开结果大致为：

- LingBot teacher：91.55%；
- Flash 1v/2a：82.66%；
- Flash 1v/1a：80.26%。

在 horizon-3 randomized tasks上：

- Teacher：约93.28%；
- Flash 1v/2a：约63.25%；
- Flash 1v/1a：约60.75%。

因此存在明确可研究的问题：

> few-step student在long-horizon和distribution shift下丢失了哪些能力，on-policy teacher correction能否在不增加NFE的情况下恢复这些能力？

---

## 7. 两层 On-Policy 必须分开定义

### 7.1 Environment-level on-policy

Student在RoboTwin中执行，产生自己真实访问的context：

\[
C_t^S
=
(o_{\le t}^S,a_{<t}^S,l).
\]

Teacher随后在同一个 \(C_t^S\) 上产生 supervision：

\[
\hat z_T,a_T
\sim
\pi_T(\cdot\mid C_t^S).
\]

它解决的是 covariate shift：

- Offline demonstrations主要覆盖 expert states；
- Flash student失败时进入的状态通常不在offline dataset中；
- Teacher只在offline states上蒸馏，无法主动纠正 student自己的错误分布。

这部分工程上明确可行，接近 VLA-OPD 或 DAgger-style teacher relabeling。

### 7.2 Denoising-level on-policy

更强的定义是让 student 生成自己的 noisy-state trajectory：

\[
x_{\sigma_0}^S,x_{\sigma_1}^S,\ldots,x_{\sigma_N}^S,
\]

然后 teacher在相同状态上提供：

\[
v_T(x_{\sigma_j}^S,\sigma_j,C_t^S).
\]

这更接近 DiffusionOPD：

- trajectory来自 student；
- teacher/student在相同 Markov state上定义 transition；
- 如果 transition covariance相同，per-step reverse KL可以化为 mean matching；
- 不需要 REINFORCE 或高方差 policy gradient。

### 7.3 本项目最完整的目标

最终希望同时满足：

```text
Outer on-policy：
    context来自student真实环境rollout

Inner on-policy：
    noisy video/action state来自student few-step sampler

Teacher：
    在相同context和student noisy state上提供transition target
```

这比只做普通 online behavior cloning更有 WAM-specific contribution。

---

## 8. 为什么不能直接照搬 DiffusionOPD

DiffusionOPD的标准推导依赖 teacher/student：

- 共享 state space；
- 共享 transition-kernel structure；
- 对SDE情况具有相同 covariance；
- 在相同 denoising discretization上比较 transition mean。

LingBot teacher与Flash student共享 state space和continuous flow，但实际 inference discretization不同：

```text
Teacher: many micro-steps
Student: one or two macro-steps
```

如果直接把teacher在 \(\sigma_s\) 的velocity乘以一个很大的student步长：

\[
x_{\sigma_e}
=
x_{\sigma_s}
+
(\sigma_e-\sigma_s)v_T(x_{\sigma_s},\sigma_s),
\]

这个单次Euler step可能不是teacher经过25/50步积分后的真正结果。

所以：

- 同backbone并不足以保证严格transition KL成立；
- 直接velocity matching可以作为baseline；
- 但不能未经推导就把它称为严格的DiffusionOPD objective。

---

## 9. 核心方法候选：Student-On-Policy Teacher Bridge

### 9.1 Macro-transition定义

对student的一个macro interval \(\sigma_s\rightarrow\sigma_e\)，从student访问的state出发：

\[
x_{\sigma_s}^S.
\]

Student用一个few-step transition得到：

\[
m_S
=
F_S(x_{\sigma_s}^S,\sigma_s\rightarrow\sigma_e).
\]

Teacher从完全相同的起点出发，通过多个micro-steps积分到同一个boundary：

\[
m_T
=
\operatorname{IntegrateTeacher}
(x_{\sigma_s}^S,\sigma_s\rightarrow\sigma_e).
\]

训练：

\[
\mathcal L_{\mathrm{bridge}}
=
\left\|
m_S-\operatorname{sg}(m_T)
\right\|^2.
\]

这里 teacher target必须 stop-gradient。

### 9.2 与 OPD 的关系

如果定义 student/teacher macro-transition具有相同 covariance：

\[
p_S(x_{\sigma_e}\mid x_{\sigma_s})
=
\mathcal N(m_S,\bar\sigma^2 I),
\]

\[
p_T(x_{\sigma_e}\mid x_{\sigma_s})
=
\mathcal N(m_T,\bar\sigma^2 I),
\]

则：

\[
D_{\mathrm{KL}}(p_S\|p_T)
=
\frac{1}{2\bar\sigma^2}
\|m_S-m_T\|^2.
\]

在 deterministic ODE limit下，它退化为macro mean matching。

但需要进一步确认：

- macro covariance应该如何从micro SDE composition得到；
- video/action是否需要不同 covariance和weight；
- teacher bridge是否在joint state上积分，还是按实际 inference factorization先video后action；
- one-step student的 consistency output如何映射为macro transition mean。

这些问题构成潜在的数学贡献，而不是已经完成的结论。

### 9.3 Modality-aware bridge

Video和action不能使用同一个未经加权的MSE：

\[
\mathcal L_{\mathrm{OPD}}
=
w_v(\sigma_v)
\mathcal L_{\mathrm{bridge}}^v
+
\lambda_a
w_a(\sigma_a)
\mathcal L_{\mathrm{bridge}}^a.
\]

权重应考虑：

- Flash-WAM原有consistency parameterization；
- video/action SNR shift；
- action低噪声区的gradient attenuation；
- 有效action channel mask；
- video latent维度远高于action维度造成的scale imbalance。

---

## 10. Action teacher query 的关键分支

LingBot action依赖predicted future video，因此必须明确teacher action到底以什么video为条件。

### 10.1 Teacher-on-student-plan

\[
\hat z_S\sim p_S(z\mid C_t^S),
\]

\[
a_T\sim\pi_T(a\mid C_t^S,\hat z_S).
\]

优点：

- teacher和student共享同一个内部plan state；
- 更符合严格on-policy/local correction；
- 可以隔离inverse dynamics问题。

缺点：

- student plan可能严重偏离teacher训练分布；
- teacher在student video latent上可能也不可靠。

### 10.2 Teacher-on-teacher-plan

\[
\hat z_T\sim p_T(z\mid C_t^S),
\]

\[
a_T\sim\pi_T(a\mid C_t^S,\hat z_T).
\]

优点：

- 更接近完整teacher policy；
- action target可能质量更高。

缺点：

- teacher/student内部state不同；
- video correction与action correction发生纠缠；
- 很难解释为相同state上的local transition KL。

### 10.3 推荐顺序

1. 先做 teacher-on-student-plan action correction；
2. 验证teacher在student plan上的action是否仍优于student；
3. 再引入video teacher bridge；
4. 最后比较 teacher-on-student-plan 与 teacher-on-teacher-plan；
5. 不应一开始就把完整teacher output硬蒸馏给student。

---

## 11. 初步训练目标

建议最终目标保持四类信号：

\[
\mathcal L
=
\lambda_a\mathcal L_{\mathrm{OPD-action}}
+
\lambda_v\mathcal L_{\mathrm{OPD-video}}
+
\lambda_F\mathcal L_{\mathrm{Flash-anchor}}
+
\lambda_D\mathcal L_{\mathrm{demo-FM}}.
\]

含义为：

- \(\mathcal L_{\mathrm{OPD-action}}\)：student-induced context/plan上的teacher action transition；
- \(\mathcal L_{\mathrm{OPD-video}}\)：student video noisy state上的teacher bridge；
- \(\mathcal L_{\mathrm{Flash-anchor}}\)：保持few-step consistency能力；
- \(\mathcal L_{\mathrm{demo-FM}}\)：防止在线数据导致基础技能遗忘。

第一版不应把四项全部打开，应通过逐步实验定位有效信号。

---

## 12. 实现阶段

### Stage 0：版本、checkpoint与evaluation parity

目标：确认teacher/student确实可以被可靠比较。

需要：

- 下载并加载：
  - `robbyant/lingbot-va-posttrain-robotwin`
  - `NU-World-Model-Embodied-AI/FlashWAM-RoboTwin`
- 锁定RoboTwin commit、camera preprocessing、task prompt；
- 核对：
  - transformer config；
  - action normalization；
  - `used_action_channel_ids`；
  - action/video SNR shift；
  - VAE与text encoder；
  - frame/action chunk size；
- 复现：
  - teacher 25v/50a；
  - Flash 1v/2a；
  - Flash 1v/1a；
- 使用相同task、seed和initialization做paired evaluation。

Go/No-Go：

- 如果无法复现teacher/student gap，暂停方法开发；
- 如果预处理或action stats不一致，先解决parity。

### Stage 1：Teacher correction usefulness test

目标：在不训练student的情况下，先确认teacher在student访问状态上有用。

对student失败或高分歧状态：

1. 保存student context；
2. 重建teacher causal context；
3. 分别获得：
   - student action；
   - teacher-on-student-plan action；
   - teacher-on-teacher-plan action；
4. 从相同simulator state短程执行或重放；
5. 比较局部progress、最终success或task predicate。

这一步回答：

> Teacher是否真的能纠正student-induced states，还是teacher一旦离开expert distribution也同样失败？

如果teacher correction没有优势，OPD不会自动产生提升。

### Stage 2：Environment-on-policy action distillation

目标：构建最小可行版本。

流程：

```text
Flash student rollout
        ↓
保存真实observation/action history
        ↓
teacher在相同context上relabel action
        ↓
student action distillation
        +
offline Flash/demo anchor
```

这一阶段：

- 不更新video OPD target；
- 可保持video branch的Flash consistency anchor；
- teacher action优先使用student plan conditioning；
- 使用有效16D action mask；
- 保留teacher/student checkpoint provenance。

这更接近environment-level VLA-OPD，是必要baseline，但不是最终方法。

### Stage 3：Action denoising-level OPD

保存student action generation中的：

- base noise；
- noisy action state；
- flow timestep；
- student transition output；
- predicted video conditioning；
- causal context identity。

比较三种目标：

1. **Clean action matching**
   \[
   \|\hat a_S-\operatorname{sg}(\hat a_T)\|^2
   \]
2. **Direct teacher velocity matching**
   \[
   \|v_S(x_\sigma^S)-\operatorname{sg}(v_T(x_\sigma^S))\|^2
   \]
3. **Teacher bridge matching**
   \[
   \|m_S-\operatorname{sg}(m_T)\|^2
   \]

这一步是判断数学方法是否比普通teacher imitation更有价值的核心。

### Stage 4：Video bridge distillation

从student video noisy state出发，用teacher micro-solver积分到student boundary。

需要：

- 使用video-specific consistency parameterization；
- 保持teacher/student context一致；
- 不用action的linear scaling直接替代video loss；
- 监控held-out video latent/FM loss；
- 检查video improvement是否真的改善action，而不是只提升视觉质量。

### Stage 5：Joint modality-aware OPD

同时训练video/action bridge，但保留：

- per-modality loss scale；
- action channel mask；
- offline Flash anchor；
- demonstration FM anchor；
- gradient conflict logging；
- policy-version-balanced replay。

训练cycle可以是：

1. Student rollout；
2. Context/replay commit；
3. Teacher batched bridge labeling；
4. Joint student update；
5. EMA update；
6. Checkpoint commit；
7. Paired quick evaluation；
8. 周期性formal evaluation。

### Stage 6：正式实验

正式结果至少包含：

- RoboTwin Clean；
- RoboTwin Randomized；
- horizon 1/2/3分层；
- per-task gains/regressions；
- held-out random seeds；
- teacher/student success gap recovery；
- inference NFE与latency；
- teacher labeling和训练成本；
- offline task retention；
- video/action gradient冲突；
- 至少两个training seeds，资源允许时增加到三个。

---

## 13. 关键 Baselines 和 Ablations

### 13.1 必须有的 baselines

- LingBot full-step teacher；
- Released Flash-WAM 1v/2a；
- Released Flash-WAM 1v/1a；
- 继续训练相同步数的 offline Flash-WAM；
- Online replay + original Flash objective；
- Environment-on-policy hard action distillation；
- Direct teacher velocity matching；
- Teacher bridge matching；
- Action-only OPD；
- Video-only OPD；
- Joint video-action OPD。

### 13.2 关键 ablations

- Teacher-on-student-plan vs teacher-on-teacher-plan；
- Environment-on-policy only vs denoising-on-policy only vs both；
- Direct Euler target vs teacher multi-step bridge；
- 1v/2a student vs 1v/1a student；
- 有/无 offline Flash anchor；
- 有/无 demonstration FM anchor；
- Full-parameter vs limited block unfreeze；
- 不同teacher uncertainty filter；
- 不同policy lag和replay freshness；
- Video/action gradient surgery或loss balancing。

---

## 14. 工程方便点

### 14.1 可以直接复用

Flash-WAM已经提供：

- teacher/student/EMA三模型训练结构；
- FlowMatch schedulers；
- video/action consistency functions；
- full-parameter FSDP；
- action validity mask；
- optimizer和checkpoint；
- RoboTwin offline dataset loader；
- shared LingBot model implementation。

LingBot-VA已经提供：

- RoboTwin server/client evaluation；
- multi-GPU rollout；
- checkpoint和数据；
- action/camera preprocessing；
- KV-cache inference。

### 14.2 主要新增模块

仍需实现：

- online transition/replay schema；
- student context捕获；
- teacher cache重建；
- student denoising-state捕获；
- teacher-on-student-state forward；
- teacher micro-step bridge；
- teacher label caching；
- policy-version和label-version管理；
- mixed online/offline sampler；
- OPD/anchor loss routing；
- per-modality gradient logging；
- paired promotion evaluation。

---

## 15. 最大难点与风险

### 15.1 KV-cache与history alignment

Teacher必须看到student实际的：

- observation history；
- executed action history；
- real feedback frames；
- task prompt；
- chunk/position信息。

不能直接使用teacher自己的rollout cache，否则不再是student-state OPD。

推荐从clean replay重建teacher cache，而不是长期序列化模型内部KV tensor，因为后者依赖：

- model version；
- attention implementation；
- dtype；
- FSDP partition；
- cache window configuration。

### 15.2 Teacher在student state上也可能失败

Teacher整体更强，不代表它在student的极端OOD失败状态上始终可靠。

需要监控：

- teacher多seed action variance；
- teacher plan一致性；
- teacher/student divergence；
- teacher short-horizon correction效果；
- teacher target是否越界；
- teacher是否在某些task系统性失败。

必要时使用 uncertainty weighting，而不是无条件强蒸馏。

### 15.3 Shared backbone gradient interference

公开版本是shared DiT，因此：

- action OPD会改变video generation；
- video OPD会改变action inference；
- online loss可能破坏few-step consistency；
- long-tail online data可能造成50-task遗忘。

应记录：

- action/video gradient norm；
- shared block gradient cosine；
- embedding/head/backbone parameter delta；
- offline Flash loss；
- held-out video loss；
- per-task retention。

### 15.4 Action padding

RoboTwin action虽然保存为30D，但只有16D执行。

有效channel大致为：

```python
list(range(0, 7)) + [28] + list(range(7, 14)) + [29]
```

如果padding维参与loss reduction，会出现loss明显下降但闭环控制无提升。

### 15.5 One-step student缺少内部dense trajectory

1v/1a student每个模态只有一个macro transition，denoising-level OPD信号不够“dense”。

当前 released checkpoint 按 1v/1a 作为唯一主 operating point：

- action 只有共同 macro boundary \([1,0]\)；
- teacher bridge 使用 teacher 50 个 action micro-steps 构造完整区间 endpoint；
- Stage 3 的监督不是两个局部 transition，而是一个全区间 macro flow-map target；
- 为弥补内部监督不够 dense，应优先结合 environment-on-policy replay、teacher
  intermediate diagnostics 和 Flash/demo anchor；
- 强制 2a 仅保留为 counterfactual diagnostic，不作为主结果。

### 15.6 Teacher labeling成本

Student rollout很快，但teacher每个chunk需要多次video/action forward。

8×A100 80GB足以进行5B full-parameter FSDP训练，主要瓶颈是teacher labeling throughput。

建议阶段式执行：

```text
8-GPU student rollout
        ↓
batched online context store
        ↓
8-GPU teacher labeling/bridge integration
        ↓
8-GPU FSDP student update
```

不要一开始强求teacher、student、EMA和simulator全部长期同时驻留。

### 15.7 方法创新可能退化为普通DAgger

如果方法最终只是：

```text
student访问state → teacher输出action → student做MSE
```

它的主要贡献会被认为是把 VLA-OPD/DAgger 应用于一个新模型，创新度有限。

要形成更强贡献，需要至少解决其中一个WAM-specific问题：

- modality-aware on-policy transition objective；
- macro teacher bridge与few-step student的KL/mean-matching推导；
- action conditioning on student/teacher plan的因果分解；
- environment-on-policy与denoising-on-policy的双层统一；
- shared WAM中video/action OPD gradient conflict的系统分析；
- 多个task-specialized WAM teachers合并到一个few-step student。

---

## 16. 预期贡献

如果实验成立，项目可以形成以下贡献：

### Contribution 1：WAM-specific OPD formulation

把OPD从：

- token-level VLA policy；
- 单模态diffusion generation；

扩展到：

- autoregressive joint video-action WAM；
- asymmetric video/action flow schedule；
- few-step student与full-step teacher；
- environment state和denoising state的双层on-policy分布。

### Contribution 2：Teacher-bridge macro transition

为不同NFE的teacher/student定义共同macro boundary，由teacher micro-solver给出bridge target，避免把teacher velocity粗暴用于巨大Euler step。

### Contribution 3：Modality-aware joint correction

分别处理：

- high-dimensional video latent；
- low-dimensional precision action；
- 不同noise regime；
- shared-backbone gradient interference；
- action-conditioned/plan-conditioned causal关系。

### Contribution 4：Recovering step-distillation control loss

证明student无需恢复teacher的25/50 NFE，也可以通过on-policy teacher correction恢复部分long-horizon和randomized success。

这是比“进一步加速WAM”不同的问题：

> Flash-WAM解决如何把WAM变快；本项目解决变快之后丢失的闭环能力如何通过on-policy post-training恢复。

---

## 17. 成功标准

### 17.1 最低成功标准

- On-policy action distillation在held-out seeds上显著优于：
  - released Flash；
  - matched-step offline Flash continuation；
- 不是只在采集任务或训练seed上提升；
- 保持相同1v/2a或1v/1a inference budget；
- 未导致明显的其他任务retention collapse。

### 17.2 强成功标准

- Teacher bridge优于hard action target与direct velocity matching；
- Joint video-action OPD优于action-only；
- long-horizon/randomized gap得到明显恢复；
- teacher-student discrepancy下降与真实success提升存在稳定相关性；
- 模型仍满足实时推理要求。

### 17.3 论文级成功标准

- 方法在多个task/horizon上分布式提升，而不是只靠少数任务；
- 至少两个training seeds；
- paired evaluation显示统计上可信的gain；
- 清楚证明收益来自on-policy transition matching，而不是额外online interactions或额外SFT compute；
- 有可复现的teacher/student/context alignment实现；
- 对失败case、teacher uncertainty和gradient conflict有系统分析。

---

## 18. 当前不应宣称的内容

在完成理论和实验之前，不应宣称：

- 已经严格实现DiffusionOPD；
- teacher/student因为同backbone就天然满足closed-form KL条件；
- online teacher distillation必然恢复teacher成功率；
- video质量提升必然带来action success提升；
- full-parameter training一定优于partial unfreeze；
- teacher在所有student-induced states上都是正确专家；
- 只要training success上升就说明generalization提升；
- 只在几个困难任务上提升就足以证明通用WAM后训练能力。

---

## 19. 公开资源

### Models

- LingBot-VA teacher：
  https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin

- Flash-WAM student：
  https://huggingface.co/NU-World-Model-Embodied-AI/FlashWAM-RoboTwin

### Code

- LingBot-VA：
  https://github.com/Robbyant/lingbot-va

- Flash-WAM：
  https://github.com/NU-World-Model-Embodied-AI/Flash-WAM

### Data

- RoboTwin LeRobot dataset：
  https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot

### Papers and related methods

- LingBot-VA / Causal World Modeling for Robot Control：
  https://arxiv.org/abs/2601.21998

- Flash-WAM：
  https://arxiv.org/abs/2606.05254

- DiffusionOPD：
  https://arxiv.org/abs/2605.15055

- VLA-OPD：
  https://arxiv.org/abs/2603.26666

---

## 20. 给后续 Coding/Research Agent 的首要问题

后续进入codebase研究时，优先回答以下问题：

1. Released Flash checkpoint究竟对应1v/1a还是可同时切换1v/2a？
2. Teacher与student inference时，video/action sampler的准确boundary和scheduler定义是什么？
3. Flash distillation training中的joint forward，与实际“先生成video再生成action”的inference factorization差异有多大？
4. 能否在不序列化KV tensor的情况下，从student replay稳定重建teacher causal context？
5. Teacher能否直接接收student-generated video latent作为action conditioning？
6. Student action denoising的中间state能否完整捕获并在learner端重放？
7. Teacher从student noisy state执行micro-step bridge时，KV-cache和clean history如何固定？
8. Teacher/student macro transition如何定义共同 covariance，或者是否应明确采用ODE mean matching？
9. Video/action loss当前的实际reduction、mask和scale是否正确？
10. 三个5B模型在8×A100 80GB上的FSDP显存、吞吐和checkpoint成本是多少？
11. Online replay如何与原始offline Flash dataset混合，才能避免policy drift？
12. 哪些RoboTwin tasks最适合第一轮pilot：teacher明显强于student、但teacher在student states上仍有可靠纠错能力？

---

## 21. 一句话总结

> 本项目利用full-step LingBot-VA作为强teacher、few-step Flash-WAM作为实时student，在student自己访问的环境状态和video-action denoising states上进行modality-aware teacher-bridge distillation，目标是在不增加student inference NFE的前提下，恢复aggressive step distillation在RoboTwin randomized与long-horizon控制中丢失的成功率，并探索适用于joint video-action WAM的On-Policy Distillation数学与训练范式。
