#!/usr/bin/env python3
"""Run the interpolation-for-incongruent-concepts research pipeline.

This script:
1. runs a separable-concept control using LRH word-pair assets,
2. runs congruent vs incongruent interpolation experiments on SugarCrepe++,
3. saves raw outputs, summary tables, and figures for the final report.

The representation model is a real embedding API (`text-embedding-3-large`)
with local caching for reproducibility and cost control.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from datasets import load_from_disk


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
LOGS_DIR = ROOT / "logs"
CACHE_DIR = RESULTS_DIR / "cache"
MODEL_NAME = "text-embedding-3-large"
API_URL = "https://api.openai.com/v1/embeddings"


@dataclass
class Config:
    seed: int = 42
    sample_per_split: int = 75
    interpolation_steps: int = 11
    knn_k: int = 15
    anchor_neighbors: int = 12
    batch_size: int = 128
    temperature: float = 15.0
    bootstrap_samples: int = 2000
    control_train_fraction: float = 0.7
    max_word_pairs_per_concept: int = 80
    embedding_model: str = MODEL_NAME


def ensure_dirs() -> None:
    for path in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR, CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required but not set.")
    return key


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    total = np.sum(exp_x)
    return exp_x / max(total, 1e-12)


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))
    return float(np.sqrt(max(0.0, 0.5 * (kl_pm + kl_qm))))


def benjamini_hochberg(pvalues: Sequence[float]) -> List[float]:
    order = np.argsort(pvalues)
    ranked = np.array(pvalues, dtype=float)[order]
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = min(prev, ranked[i] * n / rank)
        adjusted[i] = value
        prev = value
    output = np.empty(n, dtype=float)
    output[order] = adjusted
    return output.tolist()


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> Tuple[float, float]:
    means = []
    n = len(values)
    for _ in range(samples):
        indices = rng.integers(0, n, size=n)
        means.append(float(np.mean(values[indices])))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_sign_permutation_test(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, samples: int) -> float:
    diff = a - b
    observed = abs(float(np.mean(diff)))
    perms = []
    for _ in range(samples):
        signs = rng.choice([-1.0, 1.0], size=len(diff))
        perms.append(abs(float(np.mean(diff * signs))))
    perms = np.array(perms)
    return float((np.sum(perms >= observed) + 1) / (len(perms) + 1))


def paired_effect_size(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    sd = float(np.std(diff, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diff) / sd)


def save_json(obj: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def load_cache(path: Path) -> Dict[str, List[float]]:
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    return {}


def save_cache(cache: Dict[str, List[float]], path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(cache, handle)


def fetch_embeddings(texts: Sequence[str], config: Config) -> np.ndarray:
    cache_path = CACHE_DIR / f"{config.embedding_model.replace('/', '_')}_cache.pkl"
    cache = load_cache(cache_path)
    missing = [text for text in texts if text not in cache]
    api_key = load_api_key()

    if missing:
        session = requests.Session()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        for start in range(0, len(missing), config.batch_size):
            batch = missing[start : start + config.batch_size]
            payload = {"input": batch, "model": config.embedding_model}
            for attempt in range(6):
                response = session.post(API_URL, headers=headers, json=payload, timeout=120)
                if response.status_code == 200:
                    data = response.json()["data"]
                    for item, text in zip(data, batch):
                        cache[text] = item["embedding"]
                    save_cache(cache, cache_path)
                    break
                if response.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Embedding API error {response.status_code}: {response.text[:500]}")
            else:
                raise RuntimeError("Embedding API retries exhausted.")

    matrix = np.array([cache[text] for text in texts], dtype=np.float32)
    return l2_normalize(matrix)


def sample_sugarcrepe(config: Config) -> pd.DataFrame:
    dataset = load_from_disk(str(ROOT / "datasets" / "sugarcrepe_pp"))
    rows = []
    rng = np.random.default_rng(config.seed)
    for split_name in dataset.keys():
        frame = dataset[split_name].to_pandas()
        sample_n = min(len(frame), config.sample_per_split)
        chosen = frame.iloc[rng.choice(len(frame), size=sample_n, replace=False)].copy()
        chosen["subset"] = split_name
        rows.append(chosen)
    return pd.concat(rows, ignore_index=True)


def load_word_pair_concepts(config: Config) -> Dict[str, List[Tuple[str, str]]]:
    concept_dir = ROOT / "code" / "linear_rep_geometry" / "word_pairs"
    concepts: Dict[str, List[Tuple[str, str]]] = {}
    for path in sorted(concept_dir.glob("*.txt")):
        concept = path.stem.strip("[]")
        pairs = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                pairs.append((parts[0], parts[1]))
        if pairs:
            concepts[concept] = pairs[: config.max_word_pairs_per_concept]
    return concepts


def run_control_experiment(config: Config, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame]:
    concepts = load_word_pair_concepts(config)
    all_words = sorted({word for pairs in concepts.values() for pair in pairs for word in pair})
    word_embeddings = fetch_embeddings(all_words, config)
    word_to_idx = {word: idx for idx, word in enumerate(all_words)}

    concept_vectors: Dict[str, np.ndarray] = {}
    concept_rows = []
    heldout_diffs: Dict[str, np.ndarray] = {}

    for concept, pairs in concepts.items():
        indices = np.arange(len(pairs))
        rng.shuffle(indices)
        split = max(1, int(len(indices) * config.control_train_fraction))
        train_idx = indices[:split]
        test_idx = indices[split:] if split < len(indices) else indices[-1:]

        train_diffs = []
        test_diffs = []
        for idx in train_idx:
            a, b = pairs[idx]
            train_diffs.append(word_embeddings[word_to_idx[b]] - word_embeddings[word_to_idx[a]])
        for idx in test_idx:
            a, b = pairs[idx]
            test_diffs.append(word_embeddings[word_to_idx[b]] - word_embeddings[word_to_idx[a]])

        vector = np.mean(np.stack(train_diffs), axis=0)
        vector = vector / max(np.linalg.norm(vector), 1e-12)
        concept_vectors[concept] = vector
        heldout = np.stack(test_diffs)
        heldout = heldout / np.maximum(np.linalg.norm(heldout, axis=1, keepdims=True), 1e-12)
        heldout_diffs[concept] = heldout

    concepts_sorted = sorted(concept_vectors)
    vectors = np.stack([concept_vectors[name] for name in concepts_sorted])
    interference = vectors @ vectors.T
    interference_df = pd.DataFrame(interference, index=concepts_sorted, columns=concepts_sorted)

    for concept in concepts_sorted:
        same_scores = heldout_diffs[concept] @ concept_vectors[concept]
        other_scores = []
        for other in concepts_sorted:
            if other == concept:
                continue
            other_scores.extend((heldout_diffs[concept] @ concept_vectors[other]).tolist())
        concept_rows.append(
            {
                "concept": concept,
                "same_mean_alignment": float(np.mean(same_scores)),
                "other_mean_alignment": float(np.mean(other_scores)),
                "alignment_margin": float(np.mean(same_scores) - np.mean(other_scores)),
                "heldout_pairs": int(len(same_scores)),
            }
        )

    return pd.DataFrame(concept_rows).sort_values("alignment_margin", ascending=False), interference_df


def build_text_inventory(sampled: pd.DataFrame) -> Tuple[List[str], Dict[str, int]]:
    texts = sorted(set(sampled["caption"]).union(sampled["caption2"]).union(sampled["negative_caption"]))
    return texts, {text: idx for idx, text in enumerate(texts)}


def build_knn_graph(embeddings: np.ndarray, k: int) -> nx.Graph:
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(embeddings)))
    for idx in range(len(embeddings)):
        neighbors = np.argpartition(-similarity[idx], k)[:k]
        for neighbor in neighbors:
            weight = float(np.linalg.norm(embeddings[idx] - embeddings[neighbor]))
            graph.add_edge(idx, int(neighbor), weight=weight)
    return graph


def prepare_anchor_ids(
    src_idx: int,
    dst_idx: int,
    alt_idx: int,
    embeddings: np.ndarray,
    anchor_neighbors: int,
) -> List[int]:
    scores_src = embeddings @ embeddings[src_idx]
    scores_dst = embeddings @ embeddings[dst_idx]
    top_src = np.argpartition(-scores_src, anchor_neighbors + 1)[: anchor_neighbors + 1]
    top_dst = np.argpartition(-scores_dst, anchor_neighbors + 1)[: anchor_neighbors + 1]
    anchor_ids = list(dict.fromkeys([src_idx, dst_idx, alt_idx, *top_src.tolist(), *top_dst.tolist()]))
    return anchor_ids


def interpolation_metrics(
    src_idx: int,
    dst_idx: int,
    alt_idx: int,
    embeddings: np.ndarray,
    all_pairs_distances: np.ndarray,
    config: Config,
) -> Dict[str, float | int | str]:
    src = embeddings[src_idx]
    dst = embeddings[dst_idx]
    ts = np.linspace(0.0, 1.0, config.interpolation_steps)
    points = np.stack([(1.0 - t) * src + t * dst for t in ts])

    interior = points[1:-1]
    diff_to_nodes = interior[:, None, :] - embeddings[None, :, :]
    nearest_distances = np.linalg.norm(diff_to_nodes, axis=2).min(axis=1)
    off_manifold = float(np.mean(nearest_distances))

    direct_distance = float(np.linalg.norm(src - dst))
    geodesic_length = float(all_pairs_distances[src_idx, dst_idx])
    detour_ratio = geodesic_length / max(direct_distance, 1e-12)

    anchor_ids = prepare_anchor_ids(src_idx, dst_idx, alt_idx, embeddings, config.anchor_neighbors)
    anchor_emb = embeddings[anchor_ids]
    probs = []
    for point in points:
        sims = config.temperature * (anchor_emb @ point)
        probs.append(softmax(sims))
    dual_length = sum(js_distance(probs[i], probs[i + 1]) for i in range(len(probs) - 1))
    dual_endpoint = js_distance(probs[0], probs[-1])
    dual_detour_ratio = dual_length / max(dual_endpoint, 1e-12)

    triad_emb = embeddings[[src_idx, dst_idx, alt_idx]]
    triad_labels = ["source", "target", "alternate"]
    triad_assignments = []
    for point in points:
        triad_scores = triad_emb @ point
        triad_assignments.append(triad_labels[int(np.argmax(triad_scores))])
    switch_count = sum(
        1 for left, right in zip(triad_assignments[:-1], triad_assignments[1:]) if left != right
    )
    midpoint_label = triad_assignments[len(triad_assignments) // 2]

    nearest_node_ids = np.argmin(np.linalg.norm(points[:, None, :] - embeddings[None, :, :], axis=2), axis=1)

    return {
        "direct_distance": direct_distance,
        "off_manifold_mean": off_manifold,
        "geodesic_length": geodesic_length,
        "detour_ratio": detour_ratio,
        "dual_path_length": float(dual_length),
        "dual_detour_ratio": float(dual_detour_ratio),
        "switch_count": int(switch_count),
        "midpoint_label": midpoint_label,
        "path_nearest_nodes": json.dumps(nearest_node_ids.tolist()),
    }


def run_sugarcrepe_experiment(config: Config, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sampled = sample_sugarcrepe(config)
    texts, text_to_idx = build_text_inventory(sampled)
    embeddings = fetch_embeddings(texts, config)
    graph = build_knn_graph(embeddings, config.knn_k)
    distance_frame = nx.floyd_warshall_numpy(graph, weight="weight")
    all_pairs_distances = np.asarray(distance_frame, dtype=np.float32)

    rows = []
    qualitative_rows = []
    for index, row in enumerate(sampled.itertuples(index=False), start=1):
        src_idx = text_to_idx[row.caption]
        para_idx = text_to_idx[row.caption2]
        neg_idx = text_to_idx[row.negative_caption]

        congruent_metrics = interpolation_metrics(
            src_idx, para_idx, neg_idx, embeddings, all_pairs_distances, config
        )
        incongruent_metrics = interpolation_metrics(
            src_idx, neg_idx, para_idx, embeddings, all_pairs_distances, config
        )

        rows.append(
            {
                "example_id": int(row.id),
                "subset": row.subset,
                "pair_type": "congruent",
                "source_text": row.caption,
                "target_text": row.caption2,
                "alternate_text": row.negative_caption,
                **congruent_metrics,
            }
        )
        rows.append(
            {
                "example_id": int(row.id),
                "subset": row.subset,
                "pair_type": "incongruent",
                "source_text": row.caption,
                "target_text": row.negative_caption,
                "alternate_text": row.caption2,
                **incongruent_metrics,
            }
        )

        for pair_type, target_text, metrics in [
            ("congruent", row.caption2, congruent_metrics),
            ("incongruent", row.negative_caption, incongruent_metrics),
        ]:
            nearest_ids = json.loads(metrics["path_nearest_nodes"])
            snapshot_steps = [0, 2, 5, 8, config.interpolation_steps - 1]
            qualitative_rows.append(
                {
                    "example_id": int(row.id),
                    "subset": row.subset,
                    "pair_type": pair_type,
                    "source_text": row.caption,
                    "target_text": target_text,
                    "alternate_text": row.negative_caption if pair_type == "congruent" else row.caption2,
                    "retrieval_t0": texts[nearest_ids[snapshot_steps[0]]],
                    "retrieval_t02": texts[nearest_ids[snapshot_steps[1]]],
                    "retrieval_t05": texts[nearest_ids[snapshot_steps[2]]],
                    "retrieval_t08": texts[nearest_ids[snapshot_steps[3]]],
                    "retrieval_t10": texts[nearest_ids[snapshot_steps[4]]],
                    "off_manifold_mean": metrics["off_manifold_mean"],
                    "detour_ratio": metrics["detour_ratio"],
                    "dual_detour_ratio": metrics["dual_detour_ratio"],
                }
            )

        if index % 25 == 0:
            print(f"Processed {index}/{len(sampled)} SugarCrepe examples", flush=True)

    return pd.DataFrame(rows), pd.DataFrame(qualitative_rows)


def summarize_paired_metrics(results: pd.DataFrame, config: Config, rng: np.random.Generator) -> pd.DataFrame:
    metrics = [
        "off_manifold_mean",
        "detour_ratio",
        "dual_path_length",
        "dual_detour_ratio",
        "switch_count",
    ]
    pivots = {
        metric: results.pivot_table(index=["subset", "example_id"], columns="pair_type", values=metric)
        for metric in metrics
    }
    rows = []
    pvals = []
    for metric, pivot in pivots.items():
        pivot = pivot.dropna()
        congruent = pivot["congruent"].to_numpy(dtype=float)
        incongruent = pivot["incongruent"].to_numpy(dtype=float)
        diff = incongruent - congruent
        ci_low, ci_high = bootstrap_mean_ci(diff, rng, config.bootstrap_samples)
        p_value = paired_sign_permutation_test(incongruent, congruent, rng, config.bootstrap_samples)
        pvals.append(p_value)
        rows.append(
            {
                "metric": metric,
                "n": int(len(diff)),
                "congruent_mean": float(np.mean(congruent)),
                "incongruent_mean": float(np.mean(incongruent)),
                "mean_difference_incongruent_minus_congruent": float(np.mean(diff)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "p_value": p_value,
                "effect_size_dz": paired_effect_size(incongruent, congruent),
            }
        )

    adjusted = benjamini_hochberg(pvals)
    for row, q_value in zip(rows, adjusted):
        row["q_value_bh"] = q_value
    return pd.DataFrame(rows)


def make_figures(control_df: pd.DataFrame, interference_df: pd.DataFrame, results: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 8))
    heatmap_slice = interference_df.iloc[:15, :15]
    sns.heatmap(heatmap_slice, cmap="coolwarm", center=0.0)
    plt.title("Concept Interference Heatmap (Top 15 Concepts)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "control_interference_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    control_plot = control_df.head(15).sort_values("alignment_margin")
    plt.barh(control_plot["concept"], control_plot["alignment_margin"], color="#2b6cb0")
    plt.xlabel("Held-out alignment margin")
    plt.ylabel("Concept")
    plt.title("Separable-Concept Control: Same-Concept vs Off-Target Alignment")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "control_alignment_margin.png", dpi=200)
    plt.close()

    metric_specs = [
        ("off_manifold_mean", "Off-manifold deviation"),
        ("detour_ratio", "Graph geodesic detour ratio"),
        ("dual_detour_ratio", "Dual-space detour ratio"),
    ]
    for metric, title in metric_specs:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=results, x="pair_type", y=metric, hue="subset")
        plt.title(title)
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{metric}_by_pair_type.png", dpi=200)
        plt.close()


def write_metadata(config: Config) -> None:
    metadata = {
        "python": sys.version,
        "config": asdict(config),
        "hardware": {
            "gpu_query": "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv",
            "gpu_summary": os.popen(
                "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null"
            ).read().strip(),
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_json(metadata, RESULTS_DIR / "run_metadata.json")


def main() -> None:
    ensure_dirs()
    config = Config()
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    write_metadata(config)

    control_df, interference_df = run_control_experiment(config, rng)
    control_df.to_csv(RESULTS_DIR / "control_concept_alignment.csv", index=False)
    interference_df.to_csv(RESULTS_DIR / "control_concept_interference.csv")

    sugar_results, qualitative = run_sugarcrepe_experiment(config, rng)
    sugar_results.to_csv(RESULTS_DIR / "sugarcrepe_interpolation_results.csv", index=False)
    qualitative.to_csv(RESULTS_DIR / "qualitative_retrieval_examples.csv", index=False)

    paired_summary = summarize_paired_metrics(sugar_results, config, rng)
    paired_summary.to_csv(RESULTS_DIR / "paired_metric_summary.csv", index=False)

    make_figures(control_df, interference_df, sugar_results)


if __name__ == "__main__":
    main()
