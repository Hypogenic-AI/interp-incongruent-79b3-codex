# Outline: Interpolation for Incongruent Concepts

## Title
- Type-dependent interpolation instability for incongruent text concepts

## Abstract
- Motivate pathwise geometry beyond clean LRH endpoints.
- Describe two-stage experiment: separable-control LRH sanity check plus matched SugarCrepe++ path study.
- Report pooled reversal and subset heterogeneity.
- State main conclusion: relational changes destabilize interpolation more than lexical swaps.

## Introduction
- Hook: linear concept directions do not imply meaningful linear paths.
- Gap: LRH papers study pointwise concept directions; interpolation geometry papers focus on image latents.
- Approach: compare congruent and incongruent caption paths with manifold and dual-space metrics.
- Quantitative preview: pooled incongruent paths reduce off-manifold deviation by 0.0083, but relation replacements increase it by 0.0154.
- Contributions:
  - define matched text interpolation benchmark;
  - provide control experiment for separable directions;
  - show heterogeneity across corruption types;
  - identify relational structure as key failure mode.

## Related Work
- LRH and concept geometry.
- Extensions to ambiguous/categorical concepts.
- Geodesic and manifold-respecting interpolation.
- Position this work as path-centric text-space bridge.

## Methodology
- Formalize paired path comparison.
- Experiment 1: concept alignment margin on LRH assets.
- Experiment 2: congruent vs incongruent caption interpolation on SugarCrepe++.
- Metrics: off-manifold, dual path, dual detour, switch count, graph detour.
- Baselines and statistics.

## Results
- Control table with top concepts and interference heatmap figure.
- Main paired results table.
- Subset table with endpoint distances and interpretation.
- Qualitative figure panel references and representative examples from CSV.

## Discussion
- Why pooled results reverse the simple hypothesis.
- Structural incompatibility versus lexical corruption.
- Limitations: embeddings not hidden states, graph ratio collapsed, benchmark mismatch.
- Broader implications for steering and latent traversal.

## Conclusion
- Summarize contribution and main empirical message.
- Future work: hidden states, stronger benchmarks, better manifold estimators.
