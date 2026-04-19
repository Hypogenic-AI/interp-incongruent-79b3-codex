# Research Plan: Interpolation for Incongruent Concepts

## Motivation & Novelty Assessment

### Why This Research Matters
The linear representation hypothesis (LRH) and related information-geometric work largely analyze concepts that are intentionally clean, separable, and counterfactually well behaved. Real semantic systems are not restricted to such cases: many concepts are awkwardly composed, mutually interfering, or only locally meaningful, so understanding interpolation there is necessary if representation geometry is meant to explain actual model behavior rather than curated toy contrasts.

### Gap in Existing Work
The literature review shows a clear split. LRH papers establish linear concept directions and orthogonality for curated concepts, while interpolation papers show that straight lines can diverge from manifold-respecting paths, but mainly in generative image latents rather than text representations. What is missing is a direct test of what happens when endpoints are semantically incongruent in a text representation space and whether the resulting paths become more off-manifold, more circuitous, or more semantically unstable than congruent controls.

### Our Novel Contribution
We test interpolation under matched congruent versus incongruent endpoint pairs in a real text representation model. Concretely, we combine:
- an LRH-style separable-concept control using curated word-pair directions from `code/linear_rep_geometry/word_pairs`, and
- a path study on SugarCrepe++ where each example gives both a congruent pair (`caption`, `caption2`) and an incongruent pair (`caption`, `negative_caption`).

This allows us to ask not just whether a point estimate of a concept is linear, but whether an interpolation path remains close to the empirical semantic manifold when the endpoint concepts disagree.

### Experiment Justification
- Experiment 1: Estimate concept directions on curated LRH word pairs and measure within-concept projection strength and cross-concept interference.
  Why needed: This establishes a separable-control regime and verifies that the chosen representation model reproduces the expected low-interference geometry at least approximately.
- Experiment 2: Compare congruent and incongruent caption interpolation on SugarCrepe++ using manifold detour, off-manifold deviation, and induced dual-space path length.
  Why needed: This directly tests the user’s question about what interpolation between incongruent concepts looks like.
- Experiment 3: Robustness and qualitative retrieval analysis on representative cases.
  Why needed: Aggregate metrics alone can hide whether the path collapses to generic meanings, jumps between neighborhoods, or passes through semantically unstable regions.

## Research Question
What changes when we interpolate between incongruent concepts instead of causally separable concepts: do the paths become more off-manifold, more geometrically circuitous, and less semantically coherent in both a representation space and an induced dual probability simplex?

## Background and Motivation
Prior LRH work focuses on concept directions that are cleanly estimable from counterfactual pairs and often approximately orthogonal under a causal inner product. Prior interpolation geometry work shows that straight-line interpolation can be a poor proxy for geodesics on curved manifolds. The missing connection is whether semantic incongruence itself predicts interpolation failure: if endpoints encode incompatible or fragile compositions, the straight line may leave the semantic manifold even when each endpoint individually lies on it.

## Hypothesis Decomposition
- H1: Curated separable concepts from LRH assets will show stronger within-concept alignment and lower off-target interference than incongruent caption differences.
- H2: Linear interpolation between congruent caption pairs will stay closer to the empirical caption manifold than interpolation between incongruent pairs.
- H3: Incongruent pairs will have larger graph-geodesic detour ratios than congruent pairs, indicating that the manifold-respecting route bends away from the Euclidean chord.
- H4: Incongruent pairs will produce longer paths in an induced dual-space probability simplex over local semantic neighborhoods, reflecting higher semantic instability.

Independent variables:
- Pair type: separable-control, congruent, incongruent
- SugarCrepe subset: `swap_object`, `swap_atribute`, `replace_relation`

Dependent variables:
- Projection margin and cross-concept interference
- Mean off-manifold nearest-neighbor distance of interpolation points
- Graph-geodesic detour ratio
- Dual-space path length / detour ratio
- Neighborhood switching count and qualitative retrieval trajectory

Alternative explanations:
- Effects could arise from lexical overlap or caption length rather than incongruence.
- Effects could reflect encoder limitations rather than a genuine geometric phenomenon.
- A sparse benchmark manifold could inflate detour ratios for all pairs.

Mitigations:
- Use matched pairs from the same SugarCrepe example for congruent versus incongruent comparison.
- Report within-subset and pooled results.
- Include a separable-control concept experiment to verify the encoder is not uniformly noisy.

## Proposed Methodology

### Approach
Use a real transformer representation model to encode words and captions. First, reproduce a lightweight separability sanity check using curated LRH word pairs. Then build an empirical semantic manifold from sampled SugarCrepe captions, paraphrases, and negatives. Compare linear interpolation outcomes between congruent and incongruent endpoints by measuring how far the chord departs from the manifold and how much longer the manifold-respecting path becomes.

The analysis is deliberately path-centric rather than probe-centric. This is the novel piece relative to the cited LRH work.

### Experimental Steps
1. Verify environment, GPU, dataset schemas, and repository assets.
   Rationale: avoid assumptions and ensure the plan matches local resources.
2. Install missing dependencies in `.venv` using `uv add`.
   Rationale: keep the run isolated and reproducible.
3. Load a modern open transformer representation model and implement pooled text encoding.
   Rationale: use a real model with stable sentence-level geometry.
4. Run Experiment 1 on LRH word-pair assets.
   Rationale: establish a low-interference control regime for separable concepts.
5. Sample a balanced subset from SugarCrepe++, encode all unique texts, and construct a k-NN graph manifold.
   Rationale: approximate the empirical semantic manifold with tractable computation.
6. For each example, compare the congruent path (`caption` -> `caption2`) and incongruent path (`caption` -> `negative_caption`).
   Rationale: matched-pair design controls for image identity and surface form.
7. Compute primal and dual metrics, run paired statistical tests, and save figures/tables.
   Rationale: quantify whether incongruence predicts geometric and semantic instability.
8. Inspect representative trajectories by nearest-neighbor retrieval along the path.
   Rationale: answer what the interpolation “looks like,” not only whether metrics differ.

### Baselines
- Random other-concept directions for the separable-control experiment.
- Congruent caption interpolation as the primary baseline for incongruent interpolation.
- Direct Euclidean chord distance as the baseline against graph-geodesic distance.

### Evaluation Metrics
- Projection score margin: same-concept versus other-concept alignment.
- Cross-concept interference: similarity between concept directions.
- Off-manifold deviation: mean nearest-neighbor distance of interpolated points to the sampled caption set.
- Geodesic detour ratio: shortest-path distance on the k-NN graph divided by direct embedding distance.
- Dual-space path length: cumulative Jensen-Shannon distance across interpolation steps after softmaxing similarities to a local anchor set.
- Neighborhood switching count: number of changes in nearest retrieved caption family across the path.

### Statistical Analysis Plan
- Paired Wilcoxon signed-rank tests for congruent versus incongruent metrics on matched SugarCrepe examples.
- Bootstrap 95% confidence intervals for mean differences and ratios.
- Benjamini-Hochberg correction across the main family of paired tests.
- Report effect sizes using paired standardized mean differences where sensible.

## Expected Outcomes
Results supporting the hypothesis would show:
- low interference for curated LRH concept directions,
- significantly larger off-manifold deviation for incongruent paths,
- significantly larger graph-geodesic detour ratios for incongruent pairs, and
- longer dual-space paths with more neighborhood switching for incongruent pairs.

Results against the hypothesis would show little or no difference between congruent and incongruent pairs after matching, suggesting that interpolation failure is driven more by generic manifold sparsity than by semantic incongruence.

## Timeline and Milestones
- Phase 0-1 planning and resource review: complete first
- Environment and implementation: one coding pass
- Experiment execution: one full run on sampled SugarCrepe data plus LRH control
- Analysis and figure generation: immediately after experiment run
- Documentation and validation: final pass after outputs are stable

## Potential Challenges
- The chosen encoder may not reproduce strong LRH-like separability.
  Mitigation: treat Experiment 1 as a sanity check rather than a full replication.
- Hugging Face model download or memory issues.
  Mitigation: use a smaller modern encoder and batch conservatively.
- Graph-geodesic metrics may be sensitive to k-NN graph sparsity.
  Mitigation: test a modest sensitivity range for `k` and report the default setting.

## Success Criteria
- `planning.md`, code, results, figures, `REPORT.md`, and `README.md` are all produced.
- At least one real representation model is run successfully.
- The main congruent versus incongruent comparison is completed with paired statistics and qualitative examples.
- The final report gives a clear answer to what interpolation between incongruent concepts looks like in this operationalization.
