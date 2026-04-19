# Downloaded Datasets

This directory contains local datasets for the research project. The full local dataset artifacts are intentionally ignored by git; only this documentation and small sample files are intended to remain visible in version control.

## Dataset 1: CFQ (MCD1)

### Overview
- Source: Hugging Face `cfq` config `mcd1`
- Size: train 95,743 / test 11,968
- Format: Hugging Face DatasetDict saved to disk
- Task: compositional semantic parsing
- Splits: `train`, `test`
- License: see dataset card on Hugging Face / original Google release

### Why It Matters Here
CFQ is a good proxy for the "causally separable" side of the hypothesis. Its compositional structure is explicit and symbolic, making it useful for measuring whether interpolation behaves predictably when concept factors are relatively cleanly decomposed.

### Download Instructions

Using Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("cfq", "mcd1")
dataset.save_to_disk("datasets/cfq_mcd1")
```

### Loading the Dataset

```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/cfq_mcd1")
print(dataset["train"][0])
```

### Sample Data
- Local sample: `datasets/cfq_mcd1/samples/examples.json`

### Notes
- Text-only and compact enough for quick theory-driven experiments.
- Useful as a control benchmark against harder, mismatched, or semantically odd compositions.

## Dataset 2: SugarCrepe++ Caption Contrasts

### Overview
- Source: Hugging Face `Aman-J/SugarCrepe_pp`
- Size downloaded locally:
  - `swap_object`: 245 examples
  - `swap_atribute`: 666 examples
  - `replace_relation`: 1,406 examples
- Format: Hugging Face DatasetDict saved to disk
- Task: vision-language compositionality evaluation via true/false caption pairs
- Splits: benchmark releases expose these subsets as `train`
- License: see dataset card / originating benchmark materials

### Why It Matters Here
These caption-pair subsets are a practical proxy for the "incongruent concept" regime: they encode minimally different concept combinations where one caption is plausible and the other is compositional but wrong. They are useful for building interpolation tests over embeddings without requiring immediate download of the full image assets.

### Download Instructions

```python
from datasets import load_dataset, DatasetDict

parts = {
    "swap_object": load_dataset("Aman-J/SugarCrepe_pp", "swap_object", split="train"),
    "swap_atribute": load_dataset("Aman-J/SugarCrepe_pp", "swap_atribute", split="train"),
    "replace_relation": load_dataset("Aman-J/SugarCrepe_pp", "replace_relation", split="train"),
}

DatasetDict(parts).save_to_disk("datasets/sugarcrepe_pp")
```

### Loading the Dataset

```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/sugarcrepe_pp")
print(dataset["swap_object"][0])
```

### Sample Data
- Local samples:
  - `datasets/sugarcrepe_pp/samples/swap_object.json`
  - `datasets/sugarcrepe_pp/samples/swap_atribute.json`
  - `datasets/sugarcrepe_pp/samples/replace_relation.json`

### Notes
- The downloaded artifact contains caption-level benchmark data, not the full underlying image collection.
- This is a useful lightweight benchmark for initial representation-space experiments before moving to larger visual compositional datasets such as MIT-States or ImageNet-AO.

## Recommended Additional Dataset Assets

These were not downloaded because the image assets are larger and less direct to reproduce in this pass, but they are strong next-step benchmarks:

- MIT-States: classic attribute-object compositional generalization benchmark.
- WordNet-derived feature sets from `code/llm_categorical_hierarchical_representations/data/`: directly aligned with the hierarchical concept geometry paper.
- Counterfactual word-pair sets from `code/linear_rep_geometry/word_pairs/`: directly aligned with LRH measurement/intervention experiments.
