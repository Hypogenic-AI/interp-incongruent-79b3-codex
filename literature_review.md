# Literature Review: Interpolation for Incongruent Concepts

## Review Scope

### Research Question
What should we expect when interpolating between incongruent concepts, rather than causally separable concepts, under the linear representation hypothesis and an information-geometric view of representation space?

### Inclusion Criteria
- Papers that formalize or test the linear representation hypothesis.
- Papers that study concept geometry, semantic hierarchy, or compositional structure in representation space.
- Papers that study interpolation or geodesics in non-Euclidean latent spaces.
- Resources with runnable code or reusable benchmark assets.

### Exclusion Criteria
- Purely application-driven interpolation papers without geometric or representational analysis.
- Papers centered only on image generation quality with no useful geometric machinery.
- Broad interpretability work not tied to concept geometry or interpolation.

### Time Frame
- Foundations: 2017-present
- Main emphasis: 2021-2025

### Sources
- arXiv
- ACL Anthology
- OpenReview
- GitHub
- Hugging Face Datasets

## Search Log

| Date | Query / Method | Source | Notes |
|------|----------------|--------|-------|
| 2026-04-19 | `paper-finder` helper | local script | Local service path did not return usable results; switched to manual search. |
| 2026-04-19 | `linear representation hypothesis`, `geometry of concepts`, `latent interpolation`, `Riemannian geometry latent space` | arXiv API | Produced the core theory and interpolation paper set. |
| 2026-04-19 | title-specific repository search | GitHub | Resolved official code for the two Park papers and two interpolation papers. |
| 2026-04-19 | `cfq`, `SugarCrepe++` | Hugging Face Datasets | Downloaded compact benchmark bundles suitable for immediate experiments. |

## Screening Results

| Paper | Decision | Notes |
|------|----------|-------|
| Park et al. 2024, LRH | Include | Core formalization for separable concepts. |
| Park et al. 2025, categorical/hierarchical geometry | Include | Most direct extension toward non-binary and structured concepts. |
| Nguyen and Leng 2025 | Include | Practical extension for ambiguous concepts and activation steering. |
| Michelis and Becker 2021 | Include | Strong baseline showing straight-line interpolation can be misleading. |
| Struski et al. 2019 | Include | Feature-conditioned interpolation for arbitrary latent densities. |
| Shao et al. 2017 | Include | Foundational Riemannian geometry tools. |
| Li et al. 2025 | Include | Adjacent evidence on large-scale concept-space geometry. |
| Widdows et al. 2021 | Include | Useful framing paper on linear compositionality. |

## Research Area Overview

The literature splits naturally into two halves. One half studies whether concepts are encoded linearly in model representations and what geometric structure that implies. The other half studies whether straight-line interpolation respects the actual geometry induced by a nonlinear generator or representation map. The project hypothesis sits exactly at the intersection: if some concepts are causally separable, straight linear operations may behave predictably, but if concepts are incongruent or structurally entangled, linear interpolation may deviate from semantically meaningful paths.

## Key Papers

### The Linear Representation Hypothesis and the Geometry of Large Language Models
- Authors: Kiho Park, Yo Joong Choe, Victor Veitch
- Year: 2024
- Source: ICML 2024 / arXiv
- Key Contribution: Formalizes LRH using counterfactuals, distinguishes unembedding and embedding representations, and defines a causal inner product.
- Methodology: Uses counterfactual word pairs and LLaMA-2 representations; tests subspace structure, probe behavior, orthogonality, and interventions.
- Datasets Used: Counterfactual word-pair sets plus paired multilingual Wikipedia contexts.
- Results: Concept directions act as probes and steering vectors; causal inner product is more semantically faithful than naive Euclidean geometry.
- Code Available: Yes, `code/linear_rep_geometry/`
- Relevance to Our Research: Best reference point for the "causally separable" case and for defining what a clean concept direction should look like.

### The Geometry of Categorical and Hierarchical Concepts in Large Language Models
- Authors: Kiho Park, Yo Joong Choe, Yibo Jiang, Victor Veitch
- Year: 2025
- Source: ICLR 2025 / arXiv
- Key Contribution: Extends LRH from binary contrasts to binary features, categorical concepts, and hierarchies; represents categories as polytopes and hierarchical relations via orthogonality.
- Methodology: Estimates 900+ concept vectors from WordNet noun and verb hierarchies on Gemma and LLaMA-3.
- Datasets Used: WordNet-derived synset and hypernym resources.
- Results: Test words project near 1 on target feature vectors, random words near 0; child-parent and parent-grandparent vectors satisfy predicted orthogonality.
- Code Available: Yes, `code/llm_categorical_hierarchical_representations/`
- Relevance to Our Research: Most direct bridge from separable binary concepts to richer concept families where incongruence is likely to appear.

### Toward a Flexible Framework for LRH Using Maximum Likelihood Estimation
- Authors: Trung Nguyen, Yan Leng
- Year: 2025
- Source: arXiv
- Key Contribution: Replaces the dependency on single-token counterfactual pairs with an activation-difference estimator (SAND) based on a vMF model.
- Methodology: MLE over activation differences across multiple concepts and benchmarks.
- Datasets Used: LLaMA activation-engineering benchmarks.
- Results: Better flexibility and stronger monitoring/manipulation performance for ambiguous concepts.
- Code Available: Not resolved in this pass.
- Relevance to Our Research: Promising path for incongruent concepts, where clean counterfactual pairs may not exist.

### On Linear Interpolation in the Latent Space of Deep Generative Models
- Authors: Mike Yan Michelis, Quentin Becker
- Year: 2021
- Source: ICLR Workshop / OpenReview
- Key Contribution: Shows straight-line interpolation in latent space can deviate arbitrarily from the shortest curve under the pull-back metric.
- Methodology: Computes relative length improvement of a learned shorter curve over the straight line; compares models through Monte Carlo statistics rather than cherry-picked examples.
- Datasets Used: MNIST-based VAE/BiGAN experiments.
- Results: Non-zero relative improvement means the straight line is not a faithful proxy for geodesic interpolation.
- Code Available: Yes, `code/generative_latent_space/`
- Relevance to Our Research: Strong support for measuring interpolation quality geometrically rather than visually or by cosine alone.

### Feature-Based Interpolation and Geodesics in the Latent Spaces of Generative Models
- Authors: Lukasz Struski et al.
- Year: 2019
- Source: arXiv
- Key Contribution: General interpolation framework for arbitrary latent densities and feature-favored paths.
- Methodology: Defines a quality measure over curves and interprets maximization as geodesic search under a modified Riemannian metric.
- Datasets Used: Mixtures of Gaussians, MNIST generative models, chemical compounds.
- Results: Handles interpolation under non-Gaussian densities and constrained feature preferences.
- Code Available: Yes, `code/feature_based_interpolation/`
- Relevance to Our Research: Useful template when incongruent concepts require interpolation that explicitly prefers or avoids semantic features.

### The Riemannian Geometry of Deep Generative Models
- Authors: Hang Shao, Abhishek Kumar, P. Thomas Fletcher
- Year: 2017
- Source: arXiv
- Key Contribution: Establishes pull-back geometry, geodesics, and parallel transport for deep generators.
- Methodology: Derives algorithms on generated manifolds and tests them on image data.
- Datasets Used: Image datasets for generative models.
- Results: Learned manifolds were often close to low curvature, but not guaranteed to be Euclidean.
- Code Available: Not gathered here.
- Relevance to Our Research: Provides the right language and tools for a non-Euclidean interpolation analysis.

## Common Methodologies

- Counterfactual-pair estimation: used in Park et al. 2024 for binary contrasts.
- Feature-vector estimation from token sets: used in Park et al. 2025 for WordNet attributes and categories.
- Activation-difference estimation: used in Nguyen and Leng 2025 to relax strict counterfactual assumptions.
- Pull-back metric / geodesic computation: used in Shao et al. 2017 and Michelis and Becker 2021.
- Feature-constrained curve optimization: used in Struski et al. 2019.

## Standard Baselines

- Euclidean inner product versus causal inner product.
- Straight-line interpolation versus geodesic or shorter-curve interpolation.
- Random pair / shuffled-unembedding controls in concept geometry papers.
- Simple linear probing or steering vectors from contrast pairs.

## Evaluation Metrics

- Projection score onto a concept vector.
- Cosine similarity / orthogonality between concept directions or difference vectors.
- Relative length improvement of a geodesic-like curve over a straight line.
- Downstream monitoring/manipulation accuracy for steering vectors.
- Retrieval or ranking accuracy for true versus corrupted compositional pairs.

## Datasets in the Literature

- Word-pair counterfactual sets: used for binary concept estimation.
- WordNet noun/verb hierarchies: used for feature and hierarchy geometry.
- Wikipedia paired contexts: used for probe validation.
- MNIST and related latent-space image datasets: used for interpolation geometry.
- Compositional caption benchmarks such as SugarCrepe / attribute-object datasets: useful for semantically incongruent composition tests, though not central in the LRH papers.

## Gaps and Opportunities

- LRH is well developed for clean binary contrasts, but much less settled for semantically awkward or incongruent concept combinations.
- Existing LRH work mostly studies point estimates of directions and orthogonality, not interpolation trajectories between incompatible concepts.
- Information-geometry papers study geodesics, but mostly in generative image models rather than LLM activation spaces.
- There is no obvious off-the-shelf benchmark exactly matching "incongruent concept interpolation"; a mixed benchmark design is likely required.

## Recommendations for Our Experiment

- Recommended datasets:
  - `datasets/cfq_mcd1` as the separable-control benchmark.
  - `datasets/sugarcrepe_pp` as a lightweight incongruent-composition benchmark.
  - WordNet and word-pair assets inside the cloned repos for concept construction.
- Recommended baselines:
  - Euclidean interpolation versus causal-inner-product-aware interpolation.
  - Straight-line interpolation versus shortest-curve / geodesic surrogate.
  - Clean contrast pairs versus ambiguous concept sets estimated with SAND-style activation differences.
- Recommended metrics:
  - Orthogonality violations between target and off-target concepts.
  - Relative length improvement under pull-back metrics.
  - Probe separability before and after interpolation.
  - Ranking accuracy on true versus corrupted compositional examples.
- Methodological considerations:
  - Distinguish pointwise concept arithmetic from pathwise interpolation.
  - Treat "incongruent" as a measurable property, e.g. low causal separability, low orthogonality, or high path curvature.
  - Expect category-level or polytope structure to be more stable than a single direction for complex concepts.
