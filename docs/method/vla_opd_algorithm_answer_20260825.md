# VLA-OPD algorithm: concise primary-source note

Date: 2026-08-25

## Verified procedure

VLA-OPD repeats the following outer iteration until convergence:

1. Sample `B` prompts.
2. The current Student samples `G` environment trajectories per prompt.
3. The frozen Teacher is queried at every Student-visited state and returns
   dense action-token logits/distributions. Teacher actions are not executed.
4. For Student-sampled action token `a_t`, construct the Monte-Carlo
   Reverse-KL reward

   `r_t = -(log pi_student(a_t | s_t) - log pi_teacher(a_t | s_t))`,

   with the reward term stopped for gradient computation.
5. Aggregate the policy gradient over `B x G x T` and apply one parameter
   update in the published Algorithm 1.
6. Discard the old occupancy batch and collect fresh trajectories with the
   updated Student.

The paper therefore specifies a fresh rollout -> dense Teacher labeling ->
aggregate update -> refreshed rollout loop. It does not specify repeated
multi-epoch training on one fixed trajectory package.

## Published geometry and evidence limits

- Main experiments state `B=64`, `G=8`, nominally `512` trajectories per
  outer update under the paper's notation.
- Group-size ablation uses `B=32`, `G in {2,4,8}`.
- The paper does not disclose optimizer, numeric learning rate, trainable
  module mask, minibatch schedule, implementation-level epoch count, replay
  cache, or checkpoint-selection protocol. Official code was unavailable at
  the checked date.

## Primary sources

- Zhong et al., *VLA-OPD: Bridging Offline SFT and Online RL for
  Vision-Language-Action Models via On-Policy Distillation*,
  arXiv:2603.26666v1: Fig. 1; Algorithm 1; Sections 3.1-3.4; Eqs. 3-7;
  Sections 4.2 and 4.4.
- Local primary-source PDF:
  `tmp/pdfs/vla-opd-2603.26666v1.pdf`.
- Full repository contract and paper/local-method comparison:
  `research/vla_opd_training_contract_20260819.md`.
