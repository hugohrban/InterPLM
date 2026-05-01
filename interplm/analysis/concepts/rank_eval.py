"""
Rank-based evaluation of SAE features for biological concepts.

Two analysis modes:
  - f1_guided: look up top features from F1 CSV, then run rank eval
  - annotation_enrichment: find features that fire at annotated residues,
    bypassing F1/TP/FP analysis entirely
"""

import heapq
import json
import re
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from sklearn.metrics import auc, precision_recall_curve
from tqdm import tqdm


# ─── Shared metric helpers ────────────────────────────────────────────────────

def _auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    p, r, _ = precision_recall_curve((labels > 0).astype(np.int8), scores)
    return float(auc(r[::-1], p[::-1]))


def _random_baseline(L: int, n_pos: int, k_list: List[int]) -> dict:
    out = {}
    for k in k_list:
        if k >= L:
            out[f"hit@{k}"] = 1.0
        else:
            denom = comb(L, k)
            num = comb(L - n_pos, k) if (L - n_pos) >= k else 0
            out[f"hit@{k}"] = 1.0 - num / denom
    return out


def _per_protein_metrics(
    activations: np.ndarray, labels: np.ndarray, k_list: List[int]
) -> Optional[dict]:
    """
    activations, labels: 1-D arrays length L (one protein).
    labels may be domain IDs — we binarize.
    Returns None if no positive residues.
    """
    bin_labels = (labels > 0).astype(np.int8)
    n_pos = int(bin_labels.sum())
    if n_pos == 0:
        return None
    L = len(activations)
    order = np.argsort(-activations, kind="stable")
    rank_of_pos = np.empty(L, dtype=np.int64)
    rank_of_pos[order] = np.arange(1, L + 1)
    pos_idx = np.flatnonzero(bin_labels)
    pos_ranks = rank_of_pos[pos_idx]
    best_rank = int(pos_ranks.min())
    out = {f"hit@{k}": int(best_rank <= k) for k in k_list}
    for k in k_list:
        keff = min(k, L)
        n_in_topk = int((pos_ranks <= keff).sum())
        out[f"prec@{k}"] = n_in_topk / keff
        out[f"rec@{k}"] = n_in_topk / n_pos
    out["mrr"] = 1.0 / best_rank
    out["best_rank"] = best_rank
    out["L"] = L
    out["n_pos"] = n_pos
    out["auprc"] = _auprc(bin_labels, activations) if n_pos < L else float("nan")
    return out


# ─── Concept search ───────────────────────────────────────────────────────────

def search_concepts(query: str, concept_names: List[str]) -> List[Tuple[int, str]]:
    """Case-insensitive substring match. Returns [(filt_idx, name), ...]."""
    q = query.lower()
    return [(i, name) for i, name in enumerate(concept_names) if q in name.lower()]


# ─── Mode 1: F1-guided feature discovery ─────────────────────────────────────

def get_top_features_for_concept(
    concept_name: str,
    f1_df: pd.DataFrame,
    top_k: int = 10,
    sort_by: str = "f1_per_domain",
) -> pd.DataFrame:
    """
    Filter f1_df for exact concept name, dedupe on feature (keep best threshold row),
    sort descending by sort_by, return top-k rows.
    Works when all F1 scores are near 0 — returns the least-bad features.
    """
    df = f1_df[f1_df["concept"] == concept_name].copy()
    if df.empty:
        raise ValueError(f"Concept '{concept_name}' not found in F1 DataFrame.")
    df = df.sort_values(sort_by, ascending=False).drop_duplicates(subset=["feature"])
    return df.head(top_k).reset_index(drop=True)


# ─── Mode 2: Annotation enrichment ───────────────────────────────────────────

def compute_annotation_enrichment(
    raw_concept_col: int,
    sae,
    embed_dir: Path,
    annot_dir: Path,
    shards: List[int],
    top_k_features: int = 20,
    device: str = "cuda:0",
    chunk_size: int = 2048,
) -> pd.DataFrame:
    """
    Find SAE features that consistently activate at annotated residue positions.

    Performs a single pass over shards, accumulating mean feature activation at
    concept-annotated residues vs. background.  Returns top features by enrichment
    ratio (mean_at_concept / mean_at_background), bypassing any F1/TP/FP analysis.

    Uses encode_feat_subset with all features and normalize_features=True so that
    activations are on the same scale as the rest of the pipeline.
    """
    dict_size = sae.dict_size
    all_feats = list(range(dict_size))

    concept_sum = np.zeros(dict_size, dtype=np.float64)
    background_sum = np.zeros(dict_size, dtype=np.float64)
    concept_count = 0
    background_count = 0

    for shard in tqdm(shards, desc="Enrichment pass"):
        emb_path = embed_dir / f"shard_{shard}" / "embeddings.pt"
        annot_path = annot_dir / f"shard_{shard}" / "aa_concepts.npz"
        if not emb_path.exists() or not annot_path.exists():
            print(f"  Skipping shard {shard}: missing data")
            continue

        bundle = torch.load(emb_path, map_location="cpu", weights_only=False)
        emb = bundle["embeddings"]  # (n_aa, d_model) fp16
        labels_sparse = load_npz(annot_path)
        label_col = labels_sparse[:, raw_concept_col].toarray().ravel()
        n_aa = emb.shape[0]

        with torch.no_grad():
            for s in range(0, n_aa, chunk_size):
                e = emb[s : s + chunk_size].to(device).float()
                acts = sae.encode_feat_subset(e, all_feats, normalize_features=True)
                acts_np = acts.cpu().numpy().astype(np.float64)
                lab = label_col[s : s + chunk_size] > 0

                if lab.any():
                    concept_sum += acts_np[lab].sum(axis=0)
                    concept_count += int(lab.sum())
                not_lab = ~lab
                if not_lab.any():
                    background_sum += acts_np[not_lab].sum(axis=0)
                    background_count += int(not_lab.sum())

    if concept_count == 0:
        raise RuntimeError(
            "No concept-annotated residues found in evaluated shards. "
            "Check that the raw_concept_col and shard list are correct."
        )

    mean_concept = concept_sum / concept_count
    mean_background = background_sum / max(background_count, 1)
    enrichment = mean_concept / (mean_background + 1e-8)

    df = pd.DataFrame(
        {
            "feature": np.arange(dict_size, dtype=int),
            "mean_act_at_concept": mean_concept,
            "mean_act_at_background": mean_background,
            "enrichment_ratio": enrichment,
        }
    )
    df["n_concept_residues"] = concept_count
    df["n_background_residues"] = background_count
    return (
        df.sort_values("enrichment_ratio", ascending=False)
        .head(top_k_features)
        .reset_index(drop=True)
    )


# ─── Fixed-size max-heap for example tracking ────────────────────────────────

class _MaxHeap:
    """Fixed-size tracker of top-N entries by a float key (highest keys win)."""

    def __init__(self, capacity: int):
        self._cap = capacity
        self._heap: list = []  # min-heap of (key, tiebreak, data); heap[0] = worst of top-N
        self._count = 0

    def push(self, key: float, data: dict) -> None:
        if np.isnan(key):
            key = 0.0
        entry = (key, self._count, data)
        self._count += 1
        if len(self._heap) < self._cap:
            heapq.heappush(self._heap, entry)
        elif key > self._heap[0][0]:  # new entry beats current worst in top-N
            heapq.heapreplace(self._heap, entry)

    def to_list(self) -> List[dict]:
        """Return entries sorted best-first (highest key first)."""
        return [e[2] for e in sorted(self._heap, key=lambda e: -e[0])]


# ─── Rank eval (shared by both modes) ────────────────────────────────────────

def run_rank_eval_for_concept(
    concept_name: str,
    concept_filt_idx: int,
    analysis_mode: str,
    features_df: pd.DataFrame,
    sae,
    embed_dir: Path,
    annot_dir: Path,
    keep_idx: List[int],
    shards: List[int],
    k_list: Optional[List[int]] = None,
    n_examples: int = 10,
    device: str = "cuda:0",
    chunk_size: int = 32768,
) -> dict:
    """
    For each feature in features_df, iterate over all proteins and:
      - Compute per-protein rank metrics for labeled (concept-annotated) proteins
      - Track top-n_examples positive examples (by AUPRC) with full residue activations
      - Track top-n_examples high-activation examples (by max act, any annotation status)

    Returns a JSON-serializable dict.
    """
    if k_list is None:
        k_list = [1, 5, 10, 20]

    raw_concept_col = keep_idx[concept_filt_idx]
    target_feats = sorted(features_df["feature"].astype(int).tolist())
    feat_to_col = {f: i for i, f in enumerate(target_feats)}

    labeled_results: Dict[int, list] = {f: [] for f in target_feats}
    pos_heaps: Dict[int, _MaxHeap] = {f: _MaxHeap(n_examples) for f in target_feats}
    act_heaps: Dict[int, _MaxHeap] = {f: _MaxHeap(n_examples) for f in target_feats}
    n_labeled_proteins = 0

    for shard in tqdm(shards, desc="Rank eval"):
        emb_path = embed_dir / f"shard_{shard}" / "embeddings.pt"
        annot_path = annot_dir / f"shard_{shard}" / "aa_concepts.npz"
        if not emb_path.exists() or not annot_path.exists():
            print(f"  Skipping shard {shard}: missing data")
            continue

        bundle = torch.load(emb_path, map_location="cpu", weights_only=False)
        emb = bundle["embeddings"]  # (n_aa, d_model) fp16
        boundaries = bundle["boundaries"]
        protein_ids = bundle["protein_ids"]
        labels_sparse = load_npz(annot_path)
        label_col = labels_sparse[:, raw_concept_col].toarray().ravel()
        n_aa = emb.shape[0]

        # Compute activations for target features only (efficient subset)
        feat_acts = np.empty((n_aa, len(target_feats)), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, n_aa, chunk_size):
                e = emb[s : s + chunk_size].to(device).float()
                acts = sae.encode_feat_subset(e, target_feats, normalize_features=True)
                feat_acts[s : s + chunk_size] = acts.cpu().numpy()

        for (start, end), pid in zip(boundaries, protein_ids):
            lab = label_col[start:end]
            is_annotated = bool(lab.sum() > 0)
            bin_lab = (lab > 0).astype(np.int8)
            pos_residue_indices = np.flatnonzero(bin_lab).tolist()

            if is_annotated:
                n_labeled_proteins += 1

            for feat, col in feat_to_col.items():
                acts_prot = feat_acts[start:end, col]
                max_act = float(acts_prot.max())
                # Activation at each annotated position — compact, threshold-independent,
                # lets the dashboard compute overlap at any threshold.
                acts_at_sites = [float(acts_prot[i]) for i in pos_residue_indices]

                act_heaps[feat].push(
                    max_act,
                    {
                        "protein_id": pid,
                        "L": int(end - start),
                        "max_activation": max_act,
                        "is_annotated": is_annotated,
                        "n_annotated": len(pos_residue_indices),
                        "positive_residue_indices": pos_residue_indices,
                        "activations_at_annotation_sites": acts_at_sites,
                        "feature_activations": acts_prot.tolist(),
                    },
                )

                if not is_annotated:
                    continue

                m = _per_protein_metrics(acts_prot, lab, k_list)
                if m is None:
                    continue

                labeled_results[feat].append(m)
                auprc_key = m["auprc"] if not np.isnan(m["auprc"]) else m["mrr"]
                pos_heaps[feat].push(
                    auprc_key,
                    {
                        "protein_id": pid,
                        "L": m["L"],
                        "n_positive": m["n_pos"],
                        "metrics": {
                            k: m[k]
                            for k in m
                            if k not in ("L", "n_pos")
                        },
                        "positive_residue_indices": pos_residue_indices,
                        "activations_at_annotation_sites": acts_at_sites,
                        "feature_activations": acts_prot.tolist(),
                    },
                )

    # n_labeled_proteins is double-counted across features — use single-feature count
    first_feat = target_feats[0] if target_feats else None
    n_labeled_proteins = len(labeled_results[first_feat]) if first_feat else 0

    features_out = []
    for _, row in features_df.iterrows():
        feat = int(row["feature"])
        rs = labeled_results[feat]
        n_prot = len(rs)

        agg: dict = {"n_proteins": n_prot}
        if n_prot > 0:
            for k in k_list:
                agg[f"hit@{k}"] = float(np.mean([r[f"hit@{k}"] for r in rs]))
                agg[f"prec@{k}"] = float(np.mean([r[f"prec@{k}"] for r in rs]))
                agg[f"rec@{k}"] = float(np.mean([r[f"rec@{k}"] for r in rs]))
            agg["mrr"] = float(np.mean([r["mrr"] for r in rs]))
            auprcs = [r["auprc"] for r in rs if not np.isnan(r["auprc"])]
            agg["auprc_mean"] = float(np.mean(auprcs)) if auprcs else float("nan")
            agg["auprc_median"] = float(np.median(auprcs)) if auprcs else float("nan")
            agg["auprc_baseline"] = float(
                np.mean([r["n_pos"] / r["L"] for r in rs])
            )
            agg["median_L"] = int(np.median([r["L"] for r in rs]))
            agg["median_n_positive"] = float(np.median([r["n_pos"] for r in rs]))

        rand_base: dict = {}
        if rs:
            rb = _random_baseline(
                int(np.median([r["L"] for r in rs])),
                max(1, int(np.median([r["n_pos"] for r in rs]))),
                k_list,
            )
            rand_base = {f"hit@{k}": rb[f"hit@{k}"] for k in k_list}
        else:
            rand_base = {f"hit@{k}": float("nan") for k in k_list}

        features_out.append(
            {
                "feature_id": feat,
                "discovery_stats": _discovery_stats_from_row(row),
                "aggregate_metrics": agg,
                "random_baseline": rand_base,
                "positive_examples": pos_heaps[feat].to_list(),
                "high_activation_examples": act_heaps[feat].to_list(),
            }
        )

    return {
        "concept": concept_name,
        "filt_idx": concept_filt_idx,
        "analysis_mode": analysis_mode,
        "n_proteins_with_concept": n_labeled_proteins,
        "shards_evaluated": list(shards),
        "k_list": k_list,
        "features": features_out,
    }


def _discovery_stats_from_row(row: pd.Series) -> dict:
    if "enrichment_ratio" in row.index:
        return {
            "source": "annotation_enrichment",
            "mean_act_at_concept": float(row.get("mean_act_at_concept", float("nan"))),
            "mean_act_at_background": float(
                row.get("mean_act_at_background", float("nan"))
            ),
            "enrichment_ratio": float(row.get("enrichment_ratio", float("nan"))),
            "n_concept_residues": int(row.get("n_concept_residues", 0)),
        }
    return {
        "source": "f1_guided",
        "f1_per_domain": float(row.get("f1_per_domain", float("nan"))),
        "f1": float(row.get("f1", float("nan"))),
        "precision": float(row.get("precision", float("nan"))),
        "recall": float(row.get("recall", float("nan"))),
        "threshold_pct": float(row.get("threshold_pct", float("nan"))),
        "tp": int(row.get("tp", 0)),
        "fp": int(row.get("fp", 0)),
    }


# ─── JSON serialisation helper ────────────────────────────────────────────────

def save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, allow_nan=True)


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def concept_to_filename(concept_name: str, mode: str) -> str:
    """Convert concept name and mode to a filesystem-safe filename."""
    safe = re.sub(r"[^\w\-]", "_", concept_name)
    return f"{safe}_{mode}.json"
