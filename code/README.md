# Cloned Repositories

## Repo 1: linear_rep_geometry
- URL: https://github.com/KihoPark/linear_rep_geometry
- Purpose: official code for the LRH and causal-inner-product paper
- Location: `code/linear_rep_geometry/`
- Key files:
  - `store_matrices.py`
  - `linear_rep_geometry.py`
  - `1_subspace.ipynb`
  - `2_heatmap.ipynb`
  - `3_measurement.ipynb`
  - `4_intervention.ipynb`
- Notes:
  - Contains the counterfactual word-pair resources used to estimate concept directions.
  - Best baseline for reproducing separability, probing, and steering experiments.
  - Requires `transformers`, `torch`, and GPU access for full reproduction.

## Repo 2: llm_categorical_hierarchical_representations
- URL: https://github.com/kihopark/llm_categorical_hierarchical_representations
- Purpose: official code for categorical/hierarchical concept geometry in LLMs
- Location: `code/llm_categorical_hierarchical_representations/`
- Key files:
  - `01_eval_noun.ipynb`
  - `02_eval_verb.ipynb`
  - `03_eval_noun_llama.ipynb`
  - `04_intervention.ipynb`
  - `06_subgraph.ipynb`
  - `data/noun_synsets_wordnet_gemma.json`
  - `data/noun_synsets_wordnet_hypernym_graph_gemma.adjlist`
- Notes:
  - Most directly reusable codebase for experiments on non-binary, hierarchical, or potentially incongruent concepts.
  - Provides WordNet-derived concept sets and graph assets.
  - Useful for testing whether interpolation paths respect hierarchy or violate orthogonality constraints.

## Repo 3: generative_latent_space
- URL: https://github.com/mmichelis/GenerativeLatentSpace
- Purpose: compare straight-line interpolation with shorter/geodesic-like curves in generative latent spaces
- Location: `code/generative_latent_space/`
- Key files:
  - `Models/VAE.py`
  - `Geometry/metric.py`
  - `Geometry/geodesic.py`
  - `MNIST_interpolation.py`
  - `MNIST_featureInterpolation.py`
  - `MNIST_MCimprovement.py`
  - `MNIST_MCcompare.py`
- Notes:
  - Includes pretrained model files and scripts for MNIST interpolation experiments.
  - Strong starting point for information-geometry evaluations of interpolation quality.
  - Focused on image latent spaces rather than LLM activations, but the geometry tooling is relevant.

## Repo 4: feature_based_interpolation
- URL: https://github.com/gmum/feature-based-interpolation
- Purpose: feature-constrained interpolation and geodesic search under arbitrary latent densities
- Location: `code/feature_based_interpolation/`
- Key files:
  - `main.py`
  - `interpolation_model.py`
  - `interpolation_visualization.py`
  - `geodesics_manifolds.ipynb`
- Notes:
  - Useful when the latent prior is non-Gaussian or when interpolation should favor specific semantic features.
  - Provides a more flexible interpolation objective than plain straight-line latent traversal.

## Recommended Reuse

- For separable-concept LRH experiments: start with `linear_rep_geometry`.
- For categorical or hierarchical concept structure: adapt `llm_categorical_hierarchical_representations`.
- For geodesic versus linear interpolation diagnostics: use `generative_latent_space`.
- For feature-conditioned interpolation paths: use `feature_based_interpolation`.
