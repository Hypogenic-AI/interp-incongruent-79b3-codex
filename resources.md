# Resources Catalog

## Summary

This document catalogs the papers, datasets, and code repositories gathered for the project on interpolation for incongruent concepts.

## Papers

Total papers downloaded: 8

| Title | Authors | Year | File | Key Info |
|------|---------|------|------|----------|
| The Linear Representation Hypothesis and the Geometry of Large Language Models | Park, Choe, Veitch | 2024 | `papers/park_2023_linear_representation_hypothesis_llms.pdf` | Core LRH formalization and causal inner product |
| The Geometry of Categorical and Hierarchical Concepts in Large Language Models | Park, Choe, Jiang, Veitch | 2025 | `papers/park_2024_geometry_categorical_hierarchical_concepts_llms.pdf` | Features, polytopes, and hierarchical orthogonality |
| Toward a Flexible Framework for LRH Using Maximum Likelihood Estimation | Nguyen, Leng | 2025 | `papers/nguyen_2025_flexible_framework_lrh_mle.pdf` | Activation-difference estimator for ambiguous concepts |
| The Geometry of Concepts: Sparse Autoencoder Feature Structure | Li et al. | 2025 | `papers/li_2024_geometry_of_concepts_sae_feature_structure.pdf` | Large-scale concept geometry in SAE feature space |
| On Linear Interpolation in the Latent Space of Deep Generative Models | Michelis, Becker | 2021 | `papers/michelis_2021_linear_interpolation_latent_space_dgms.pdf` | Straight lines can diverge sharply from geodesics |
| Feature-Based Interpolation and Geodesics in the Latent Spaces of Generative Models | Struski et al. | 2019 | `papers/struski_2019_feature_based_interpolation_geodesics.pdf` | Feature-conditioned interpolation under arbitrary densities |
| The Riemannian Geometry of Deep Generative Models | Shao, Kumar, Fletcher | 2017 | `papers/shao_2017_riemannian_geometry_deep_generative_models.pdf` | Foundational pull-back metric and geodesic machinery |
| Should Semantic Vector Composition be Explicit? Can it be Linear? | Widdows, Howell, Cohen | 2021 | `papers/cohen_2021_semantic_vector_composition_linear.pdf` | Framing paper on linear semantic composition |

See `papers/README.md` for detailed descriptions.

## Datasets

Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| CFQ (MCD1) | Hugging Face `cfq` | 95,743 train / 11,968 test | compositional semantic parsing | `datasets/cfq_mcd1/` | good separable-control benchmark |
| SugarCrepe++ subsets | Hugging Face `Aman-J/SugarCrepe_pp` | 245 + 666 + 1,406 examples | compositional caption discrimination | `datasets/sugarcrepe_pp/` | lightweight proxy for incongruent composition |

See `datasets/README.md` for download and loading instructions.

## Code Repositories

Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| linear_rep_geometry | https://github.com/KihoPark/linear_rep_geometry | LRH, probes, interventions, causal inner product | `code/linear_rep_geometry/` | official implementation |
| llm_categorical_hierarchical_representations | https://github.com/kihopark/llm_categorical_hierarchical_representations | categorical and hierarchical concept geometry | `code/llm_categorical_hierarchical_representations/` | official implementation with WordNet assets |
| generative_latent_space | https://github.com/mmichelis/GenerativeLatentSpace | linear vs geodesic interpolation comparisons | `code/generative_latent_space/` | includes pretrained MNIST models |
| feature_based_interpolation | https://github.com/gmum/feature-based-interpolation | feature-constrained interpolation and geodesics | `code/feature_based_interpolation/` | useful for non-Gaussian densities |

See `code/README.md` for detailed descriptions.

## Resource Gathering Notes

### Search Strategy

The search started with the local paper-finder workflow, but the helper was effectively blocked by the unavailable localhost service path. I then switched to direct manual search using arXiv, ACL Anthology, OpenReview, GitHub, and Hugging Face. Queries were centered on `linear representation hypothesis`, `concept geometry`, `hierarchical concepts`, `latent interpolation`, and `Riemannian geometry`.

### Selection Criteria

- Direct relevance to LRH, concept geometry, or interpolation geometry.
- Preference for papers with official code or reusable assets.
- Preference for compact datasets that enable immediate experimentation.
- Inclusion of both a clean compositional-control benchmark and a harder compositional benchmark.

### Challenges Encountered

- The `paper-finder` local service did not return results in a usable way.
- Several multimodal datasets are large, gated, or require additional image assets.
- The exact benchmark for "incongruent concept interpolation" does not currently exist as a standard single dataset.

### Gaps and Workarounds

- Workaround for missing direct benchmark: use a two-regime evaluation design.
  - Regime 1: separable compositions on CFQ.
  - Regime 2: semantically fragile or mismatched compositions on SugarCrepe++.
- Workaround for non-binary concept estimation: rely on WordNet hierarchy assets and SAND-style extensions.

## Recommendations for Experiment Design

1. Primary datasets: use `CFQ` as the separable control and `SugarCrepe++` as the incongruent-composition proxy.
2. Baseline methods: compare LRH-style linear directions with geodesic-aware interpolation and with naive Euclidean baselines.
3. Evaluation metrics: projection separation, off-target interference, orthogonality violations, and relative length improvement under pull-back metrics.
4. Code to adapt/reuse: start from `linear_rep_geometry` and `llm_categorical_hierarchical_representations`; borrow interpolation diagnostics from `generative_latent_space` and `feature_based_interpolation`.
