# Repository presentation survey

**Scope.** Read-only review of first-party README files from mature robotics and
machine-learning repositories, with the goal of improving WAM-OPD's public
presentation without changing its compatibility-sensitive runtime surface.

**Reviewed:** 2026-09-20. The links below point to the source README files that
were reviewed, rather than to secondary write-ups.

## Patterns worth adopting

| Pattern | Evidence in the source repository | WAM-OPD adaptation |
| --- | --- | --- |
| Hero visual, centered title, and one-sentence value proposition | [LeRobot README](https://github.com/huggingface/lerobot/blob/main/README.md) opens with a full-width logo image, centered badges, and a short project goal; [Diffusion Policy README](https://github.com/real-stanford/diffusion_policy/blob/main/README.md) leads with project/paper/data links and teaser figures. | Add a restrained WAM-OPD mark/teaser and keep the first sentence focused on the contribution: on-policy distillation for joint world-action models. |
| Short, useful navigation near the top | [OpenVLA README](https://github.com/openvla/openvla/blob/main/README.md) provides explicit links to Getting Started, pretrained models, installation, fine-tuning, evaluation, and the project website. | Add a compact `Install · Quick start · Pipeline · Docs · Citation` row. Do not duplicate the full table of contents. |
| A visual explanation before implementation detail | [LeRobot README](https://github.com/huggingface/lerobot/blob/main/README.md) uses robot-control and architecture visuals; [Isaac Lab README](https://github.com/isaac-sim/IsaacLab/blob/main/README.md) places a project image before its feature summary. | Put one WAM-OPD pipeline/teaser figure before the long method explanation. Keep the existing text pipeline as an accessible fallback. |
| Clear “try it” or smallest useful path | [Diffusion Policy README](https://github.com/real-stanford/diffusion_policy/blob/main/README.md) has a prominent Colab “Try it out” section; [openpi README](https://github.com/Physical-Intelligence/openpi/blob/main/README.md) shows a minimal inference snippet immediately after requirements and installation. | Make the CPU verification path and the smallest manifest-validation path visually prominent. Keep GPU/model setup separate and honest. |
| Structured capability and result tables | [LeRobot README](https://github.com/huggingface/lerobot/blob/main/README.md) groups policies by category; [openpi README](https://github.com/Physical-Intelligence/openpi/blob/main/README.md) uses model/checkpoint tables; [Diffusion Policy README](https://github.com/real-stanford/diffusion_policy/blob/main/README.md) documents the reproducible experiment/log tree. | Use small tables for pipeline stages, supported entry points, and artifact boundaries. Do not add unverified benchmark numbers or fake status badges. |
| Reproducibility and licensing are explicit | [OpenVLA README](https://github.com/openvla/openvla/blob/main/README.md) calls out version pins and model-license caveats; [Isaac Lab README](https://github.com/isaac-sim/IsaacLab/blob/main/README.md) distinguishes framework, dependency, and simulator licenses. | Retain the current release boundary: code under Apache-2.0, while models, simulator assets, datasets, and external sources retain their own licenses. Link to the existing reproducibility and artifact-policy docs. |
| Docs and community links are easy to find | [LeRobot README](https://github.com/huggingface/lerobot/blob/main/README.md) links to documentation, hardware guides, datasets, and community channels; [Isaac Lab README](https://github.com/isaac-sim/IsaacLab/blob/main/README.md) separates documentation, support, troubleshooting, and discussions. | Add a documentation map and a focused issue/support path after the quick start. Keep internal server paths out of the first screen. |

## What not to copy

1. Do not add badges for benchmark scores, model downloads, or coverage unless
   the repository publishes and continuously verifies the underlying source.
2. Do not make a large animated hero the only explanation. A static image or
   text pipeline must remain available for accessibility and quick review.
3. Do not put server paths, private artifact roots, or unpublished checkpoints
   in the public README. They belong in the existing reproducibility boundary.
4. Do not move `experiments.*` modules merely to make the tree look cleaner;
   their import paths are part of the public compatibility surface.
5. Do not imply that a CPU test or an isolated GPU smoke test is a complete
   benchmark reproduction. The current README's release-candidate disclaimer
   is the correct level of certainty.

## Proposed WAM-OPD presentation hierarchy

### P0 — first-screen polish

- Add a small WAM-OPD mark and centered title/tagline.
- Keep only verified badges (CI, license, Python/package status if available).
- Add one-line navigation to installation, quick start, pipeline, docs, and
  citation.
- Add one project teaser/pipeline visual with meaningful alt text.

### P1 — discoverability

- Add a compact “What this repo contains” table.
- Add a short `Quick start` block that ends in a local, deterministic command.
- Add a docs map for method, reproducibility, deployment, artifact policy, and
  server synchronization.
- Add a results/demo gallery only after checked assets are available.

### P2 — release polish

- Add a versioned release/changelog page and a citation block once the paper
  metadata is final.
- Add issue templates and contribution guidance for external users.
- Configure the GitHub social preview and repository metadata outside the code
  tree, after the README's content and assets are stable.

## Decision for the first implementation pass

The first pass should be limited to README hierarchy, a lightweight local SVG
mark, and documentation navigation. It should not change Python module paths,
training/evaluation contracts, generated-artifact rules, or claims about
benchmark reproducibility. Validation should include `git diff --check`, the
existing `make test`, and `wam-opd doctor`.
