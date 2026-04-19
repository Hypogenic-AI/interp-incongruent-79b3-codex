# Interpolation for Incongruent Concepts

## 1. Executive Summary
This project asked what interpolation looks like when the endpoints are not cleanly separable concepts, but instead incongruent or contradictory semantic descriptions. I tested this with a real embedding model (`text-embedding-3-large`) using two stages: a separable-control experiment on LRH word-pair assets, and a matched congruent-vs-incongruent path study on SugarCrepe++ caption triplets.

The main finding is not a simple global “incongruent is more curved” story. Instead, interpolation instability is **type-dependent**. Relation replacements (`replace_relation`) behave like genuine incongruence: they yield larger off-manifold deviation and larger dual-space path length than paraphrase controls. Attribute and object swaps do not; in those subsets, the negative caption is often lexically closer to the source than the paraphrase, so the linear path can look *more* locally regular than the congruent control.

Practical implication: interpolation failure is not triggered by contradiction alone. It appears when the endpoint change disrupts **relational structure**, not merely when a caption is corrupted by a nearby lexical substitution.

## 2. Research Question & Motivation
The motivating question was: under the linear representation hypothesis and an information-geometric view of representation space, what should we expect when interpolating between incongruent concepts rather than causally separable ones?

This matters because most LRH work studies curated counterfactual contrasts and mostly asks whether concept directions exist. That leaves open the pathwise question: even if endpoints are meaningful, does the straight line between them remain semantically meaningful when the concepts are incompatible? Bridging LRH with interpolation geometry matters for any attempt to use representation arithmetic, steering, or latent traversal outside clean toy settings.

## 3. Literature Review Summary
The pre-gathered literature suggested a specific gap:

- Park et al. (2024) formalize LRH and causal inner products for clean concept directions.
- Park et al. (2025) extend this to categorical and hierarchical concepts.
- Nguyen and Leng (2025) argue that ambiguous concepts need more flexible estimation than strict counterfactual pairs.
- Michelis and Becker (2021), Struski et al. (2019), and Shao et al. (2017) show that straight lines can diverge from manifold-respecting paths, but mostly in generative image latents.

What was missing was a direct text-space test of whether incongruent endpoints produce visibly different interpolation behavior from congruent controls.

## 4. Methodology

### 4.1 Experimental Setup
- Model: `text-embedding-3-large` via the OpenAI embeddings API
- Local resources used:
  - `code/linear_rep_geometry/word_pairs/` for separable-control concepts
  - `datasets/sugarcrepe_pp/` for matched caption triplets
- Python: 3.12.8
- Hardware detected: 4x NVIDIA RTX A6000 (unused for the final API-based run)
- Random seed: 42
- SugarCrepe sample: 75 examples each from `swap_object`, `swap_atribute`, and `replace_relation` for 225 matched examples total
- Interpolation grid: 11 points per path
- k-NN graph: `k=15`
- Bootstrap samples: 2,000

I initially attempted a local transformer stack, but the isolated `.venv` developed a broken `torch`/`scipy` state during install. Because the task explicitly required real models rather than simulations, I switched to a real embedding API with on-disk caching in `results/cache/`.

### 4.2 Experiment 1: Separable-Control LRH Sanity Check
For each of 27 concept files from the LRH repository, I split word pairs into train/test, estimated a mean concept vector on the train split, and evaluated held-out pair differences against that vector versus other concept vectors.

This is not a full replication of causal-inner-product LRH, but it is a sanity check that the representation model still shows a regime where some concepts behave approximately linearly and with limited off-target interference.

### 4.3 Experiment 2: Congruent vs Incongruent Caption Interpolation
Each SugarCrepe example provides:
- `caption`: the source description
- `caption2`: a congruent paraphrase
- `negative_caption`: an incongruent or corrupted alternative

For each example I compared:
- Congruent path: `caption -> caption2`
- Incongruent path: `caption -> negative_caption`

Metrics:
- `off_manifold_mean`: average nearest-neighbor distance from interior interpolation points to the sampled caption set
- `dual_path_length`: cumulative Jensen-Shannon path length in a local similarity-softmax simplex
- `dual_detour_ratio`: dual path length divided by endpoint dual distance
- `switch_count`: number of changes in source/target/alternate anchor assignment along the path
- `detour_ratio`: graph geodesic divided by direct endpoint distance

### 4.4 Statistical Plan
- Paired permutation sign test on congruent vs incongruent metrics
- Bootstrap 95% confidence intervals for mean differences
- Benjamini-Hochberg correction across the main metric family
- Paired standardized effect size (`d_z`)

## 5. Results

### 5.1 Separable-Control Experiment
The control experiment did recover a strong separable regime for several curated concepts.

| Concept | Same-concept alignment | Off-target alignment | Margin |
|---|---:|---:|---:|
| `Ving - 3pSg` | 0.629 | -0.018 | 0.647 |
| `adj - superlative` | 0.668 | 0.052 | 0.616 |
| `adj - comparative` | 0.620 | 0.033 | 0.587 |
| `3pSg - Ved` | 0.568 | -0.000 | 0.568 |
| `verb - Ving` | 0.587 | 0.061 | 0.525 |

This supports the claim that the representation model can express clean linear contrasts in at least some curated settings.

Figure files:
- `figures/control_alignment_margin.png`
- `figures/control_interference_heatmap.png`

### 5.2 Main Paired Comparison

| Metric | Congruent mean | Incongruent mean | Difference (inc - cong) | 95% CI | p | q |
|---|---:|---:|---:|---:|---:|---:|
| Off-manifold deviation | 0.1325 | 0.1242 | -0.0083 | [-0.0143, -0.0020] | 0.0105 | 0.0262 |
| Dual path length | 0.5108 | 0.4645 | -0.0462 | [-0.0777, -0.0129] | 0.0075 | 0.0262 |
| Dual detour ratio | 1.0612 | 1.0567 | -0.0045 | [-0.0133, 0.0047] | 0.3193 | 0.3992 |
| Switch count | 1.0133 | 1.0000 | -0.0133 | [-0.0311, 0.0000] | 0.2654 | 0.3992 |
| Graph detour ratio | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 1.0000 |

At the pooled level, incongruent paths were actually *less* off-manifold and had *shorter* dual paths than congruent ones. This contradicts the original simple hypothesis.

### 5.3 Subset Breakdown
The pooled reversal is caused by strong heterogeneity across subsets.

#### `replace_relation`
- Off-manifold mean difference: `+0.0154`
- Dual path length difference: `+0.0785`
- Dual detour ratio difference: `+0.0280`
- In 64% of examples, incongruent paths were larger than congruent paths on these metrics.

This is the subset that behaves like the original hypothesis: changing a relation produces a less stable interpolation trajectory.

#### `swap_atribute`
- Off-manifold mean difference: `-0.0235`
- Dual path length difference: `-0.1316`
- Dual detour ratio difference: `-0.0199`

#### `swap_object`
- Off-manifold mean difference: `-0.0167`
- Dual path length difference: `-0.0857`
- Dual detour ratio difference: `-0.0216`

These two subsets go the other way. The endpoint analysis explains why:

| Subset | Congruent direct distance | Incongruent direct distance |
|---|---:|---:|
| `replace_relation` | 0.4876 | 0.5429 |
| `swap_atribute` | 0.4672 | 0.3826 |
| `swap_object` | 0.4762 | 0.4161 |

For object and attribute swaps, the negative caption is often closer to the source than the paraphrase. That means the “incongruent” endpoint is frequently a shorter lexical perturbation, not a stronger semantic detour.

### 5.4 Qualitative Trajectories
Representative `replace_relation` failures:

1. Source: “Two people are holding phones together to create a linking.”
   Target: “Two men are dropping phones, causing them to unlink.”
   The midpoint retrieval jumps directly from the source wording to the contradictory target, with `off_manifold_mean = 0.2706` and `dual_detour_ratio = 1.3243`.

2. Source: “The back view of an airplane on a runway.”
   Target: “An airplane is flying over a runway.”
   Again the midpoint flips between incompatible event descriptions, with `off_manifold_mean = 0.2672` and `dual_detour_ratio = 1.3323`.

Representative `swap_object` counterexample:

1. Source: “A city scape scene.”
   Incongruent target: “A train yard scene with a city scape in the foreground.”
   Congruent target: “a train yard is visible in the foreground of a cityscape scene.”
   Here the congruent paraphrase is actually the more off-manifold path (`0.2660`) than the incongruent target (`0.2464`), because the negative caption remains very close to the source wording.

Raw outputs:
- `results/sugarcrepe_interpolation_results.csv`
- `results/paired_metric_summary.csv`
- `results/qualitative_retrieval_examples.csv`

Figures:
- `figures/off_manifold_mean_by_pair_type.png`
- `figures/dual_detour_ratio_by_pair_type.png`
- `figures/detour_ratio_by_pair_type.png`

## 6. Analysis & Discussion
The clearest answer to the research question is:

**Interpolation between incongruent concepts does not have a single generic signature. It depends on what makes them incongruent.**

Three conclusions follow:

1. Clean separable concepts still exist.
   The LRH-style control recovered strong within-concept alignment margins for several curated transformations. So the model does express a regime in which linear concept arithmetic is plausible.

2. Relation incongruence is the strongest failure mode.
   When the corruption changes the relation or event structure, interpolation becomes less stable: more off-manifold and longer in the induced dual space.

3. Lexically local corruption is not enough.
   Attribute and object swaps often look easier than paraphrase interpolation because the negative caption stays close in wording and local geometry, even if it is semantically wrong.

This suggests that “incongruence” should not be defined only by truth-condition mismatch. A better operationalization for interpolation studies is **structural incompatibility**, especially relational incompatibility.

## 7. Limitations
- This is an embedding-space study, not a hidden-state study with the causal inner product from Park et al. (2024). It tests the broader hypothesis, not the full LRH formalism.
- The graph geodesic ratio collapsed to 1.0 in this sampled k-NN graph, so it did not discriminate conditions. The meaningful evidence came from off-manifold and dual-space measures instead.
- SugarCrepe++ is a proxy benchmark. Its three subsets do not represent the same kind of incongruence.
- API token usage was cached locally, but exact monetary cost was not logged by the request layer.
- The environment had available GPUs, but the final experiment was API-based, so GPU acceleration was not used.

## 8. Conclusions & Next Steps
The answer to “what does interpolation between incongruent concepts look like?” is: **it looks unstable when the conflict is relational, but not necessarily when the conflict is a short lexical swap.** In the latter case, the corrupted endpoint can still sit on a short, locally smooth path.

The theoretical implication is that interpolation failure tracks structural entanglement more than contradiction in the abstract. For future work:
- rerun the same analysis on hidden states from an open LLM with explicit LRH-style concept vectors,
- replace SugarCrepe object/attribute subsets with a benchmark where the incongruent endpoint is not lexically closer than the paraphrase, and
- use a stronger manifold estimator so graph-geodesic detour becomes informative.

## 9. Reproducibility
Setup and run:

```bash
source .venv/bin/activate
python src/run_research.py
```

The script is deterministic with `seed=42` aside from external API service nondeterminism. Embeddings are cached in `results/cache/`.

## 10. References
- Park, Choe, Veitch. *The Linear Representation Hypothesis and the Geometry of Large Language Models*. 2024.
- Park, Choe, Jiang, Veitch. *The Geometry of Categorical and Hierarchical Concepts in Large Language Models*. 2025.
- Nguyen, Leng. *Toward a Flexible Framework for LRH Using Maximum Likelihood Estimation*. 2025.
- Michelis, Becker. *On Linear Interpolation in the Latent Space of Deep Generative Models*. 2021.
- Struski et al. *Feature-Based Interpolation and Geodesics in the Latent Spaces of Generative Models*. 2019.
- Shao, Kumar, Fletcher. *The Riemannian Geometry of Deep Generative Models*. 2017.
