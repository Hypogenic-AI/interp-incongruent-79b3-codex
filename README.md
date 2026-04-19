# Interpolation for Incongruent Concepts

This project studies what interpolation looks like when concept endpoints are semantically incongruent instead of cleanly separable. The final pipeline uses a real embedding model (`text-embedding-3-large`) with cached API calls, a separable-control experiment on LRH word-pair assets, and a matched congruent-vs-incongruent comparison on SugarCrepe++.

## Key Findings
- The LRH-style control still showed a clear separable regime: several curated concepts had held-out alignment margins above `0.5`.
- Pooled across all SugarCrepe subsets, incongruent paths were **not** more unstable than congruent paths; off-manifold deviation and dual path length were slightly lower.
- The pooled result is misleading: `replace_relation` behaves like true incongruence and increases instability, while `swap_object` and `swap_atribute` often make the negative caption *closer* to the source than the paraphrase.
- The most useful interpretation is that interpolation failure tracks **structural/relational incompatibility**, not contradiction alone.
- The graph geodesic detour metric was uninformative in this sampled k-NN graph and should be replaced in follow-up work.

## Reproduce
```bash
source .venv/bin/activate
python src/run_research.py
```

Outputs are written to:
- `results/control_concept_alignment.csv`
- `results/control_concept_interference.csv`
- `results/sugarcrepe_interpolation_results.csv`
- `results/paired_metric_summary.csv`
- `results/qualitative_retrieval_examples.csv`
- `figures/*.png`

Embeddings are cached in `results/cache/`.

## File Structure
- `planning.md`: research plan and novelty assessment
- `REPORT.md`: full report with methods, results, discussion, and limitations
- `src/run_research.py`: end-to-end experiment pipeline
- `literature_review.md`: pre-gathered literature synthesis
- `resources.md`: catalog of papers, datasets, and code assets

## Notes
- GPUs were available in the workspace, but the final experiment used a real API model rather than local GPU inference.
- The `.venv` was used throughout; dependency tracking lives in `pyproject.toml`.
