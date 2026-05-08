"""
Per-protein SAE feature analysis.

Given a protein ID, computes which SAE features activate in it, characterizes
them by cross-dataset specificity and within-protein activation pattern, and
optionally reports overlap with biological concept annotations.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.sparse import load_npz

from interplm.data_processing.embedding_loader import (
    detect_and_create_loader,
    load_shard_embeddings,
)
from interplm.sae.inference import get_sae_feats_in_batches

# ── Specificity thresholds (dataset_frequency, 0–1 scale) ────────────────────
_SPECIFIC_THRESH = 0.10   # fires in < 10% of proteins
_GENERAL_THRESH = 0.50    # fires in ≥ 50% of proteins

# ── Activation-pattern thresholds (fraction of residues activated) ────────────
_FOCAL_THRESH = 0.20      # < 20% residues → focal
_BROAD_THRESH = 0.50      # ≥ 50% residues → broad

# ── Module-level stat cache ───────────────────────────────────────────────────
_STATS_CACHE: Dict[Path, pd.DataFrame] = {}


# ─── Shard index ──────────────────────────────────────────────────────────────

def _shard_index_path(embed_dir: Path) -> Path:
    return embed_dir / ".protein_shard_index.json"


def _index_is_stale(index_path: Path, embed_dir: Path) -> bool:
    """Return True if any shard file is newer than the index."""
    idx_mtime = index_path.stat().st_mtime
    for pt in embed_dir.rglob("embeddings.pt"):
        if pt.stat().st_mtime > idx_mtime:
            return True
    for pt in embed_dir.glob("shard_*.pt"):
        if pt.stat().st_mtime > idx_mtime:
            return True
    return False


def build_protein_shard_index(
    embed_dir: Path,
    annot_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """
    Build (or load) a mapping of protein_id → shard_idx for all shards in embed_dir.

    Writes the index to embed_dir/.protein_shard_index.json.  On subsequent calls
    the cached file is returned unless any shard file is newer.

    Cheap path: reads annot_dir/shard_N/protein_data.tsv Entry column (CSV, fast).
    Fallback: loads each shard's embeddings.pt and extracts protein_ids list.
    Old-format shards (raw tensor, no protein_ids) are skipped with a warning.
    """
    embed_dir = Path(embed_dir)
    index_path = _shard_index_path(embed_dir)

    if index_path.exists() and not _index_is_stale(index_path, embed_dir):
        with open(index_path) as f:
            data = json.load(f)
        data.pop("_meta", None)
        return data

    print("Building protein shard index …")
    loader = detect_and_create_loader(embed_dir)
    shard_indices = loader.get_available_shard_indices()

    mapping: Dict[str, int] = {}
    for shard_idx in shard_indices:
        # Cheap path: protein_data.tsv from annotation dir
        if annot_dir is not None:
            tsv = Path(annot_dir) / f"shard_{shard_idx}" / "protein_data.tsv"
            if tsv.exists():
                df = pd.read_csv(tsv, sep="\t", usecols=["Entry"])
                for pid in df["Entry"].dropna().unique():
                    mapping[str(pid).upper()] = shard_idx
                continue

        # Fallback: load embeddings.pt
        try:
            data = load_shard_embeddings(embed_dir, shard_idx, return_tensor_only=False)
        except FileNotFoundError:
            warnings.warn(f"Shard {shard_idx} not found; skipping.")
            continue

        if not isinstance(data, dict) or "protein_ids" not in data or data["protein_ids"] is None:
            warnings.warn(
                f"Shard {shard_idx} has no protein_ids (old format); skipping in index."
            )
            continue

        for pid in data["protein_ids"]:
            mapping[str(pid).upper()] = shard_idx

    # Persist
    out = {
        "_meta": {
            "n_shards": len(shard_indices),
            "built_at": datetime.now(timezone.utc).isoformat(),
        },
        **mapping,
    }
    with open(index_path, "w") as f:
        json.dump(out, f)
    print(f"Index written to {index_path} ({len(mapping)} proteins)")
    return mapping


def find_protein_in_shards(
    protein_id: str,
    embed_dir: Path,
    annot_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, int, int]:
    """
    Locate a protein in the pre-computed embedding shards.

    Returns
    -------
    protein_embeddings : np.ndarray, shape (L, d_model), float32
    shard_idx          : int
    protein_idx        : int   (index of this protein within the shard's protein list)

    Raises
    ------
    ValueError  if the protein is not found in any shard, or if the shard
                uses old format (raw tensor without protein_ids).
    """
    embed_dir = Path(embed_dir)
    pid_upper = protein_id.upper()

    mapping = build_protein_shard_index(embed_dir, annot_dir)
    if pid_upper not in mapping:
        raise ValueError(
            f"Protein '{protein_id}' not found in any shard under {embed_dir}. "
            "Pass --sequence to embed it on the fly."
        )

    shard_idx = mapping[pid_upper]
    data = load_shard_embeddings(embed_dir, shard_idx, return_tensor_only=False)

    if not isinstance(data, dict) or "protein_ids" not in data or data["protein_ids"] is None:
        raise ValueError(
            f"Shard {shard_idx} has no protein_ids (old format). "
            "Cannot extract embeddings for a specific protein."
        )

    protein_ids = data["protein_ids"]
    boundaries = data["boundaries"]
    embeddings = data["embeddings"]  # Tensor (n_aa, d_model)

    for i, pid in enumerate(protein_ids):
        if str(pid).upper() == pid_upper:
            start, end = boundaries[i]
            prot_emb = embeddings[start:end].float().numpy()
            return prot_emb, shard_idx, i

    raise ValueError(
        f"Protein '{protein_id}' listed in index for shard {shard_idx} "
        "but not found in shard's protein_ids list. Rebuild the index."
    )


# ─── SAE forward pass ─────────────────────────────────────────────────────────

def compute_full_activations(
    protein_embeddings: np.ndarray,
    sae,
    device: str,
    chunk_size: int = 2048,
) -> np.ndarray:
    """
    Run the full SAE forward pass (all dict_size features) for one protein.

    Returns
    -------
    np.ndarray of shape (L, dict_size), float32
    """
    result = get_sae_feats_in_batches(
        sae,
        device,
        protein_embeddings,
        chunk_size,
        feat_list=None,
        normalize_features=True,
    )
    return result.cpu().numpy().astype(np.float32)


# ─── Feature stats ────────────────────────────────────────────────────────────

def load_feature_stats(sae_dir: Path) -> pd.DataFrame:
    """
    Load Per_feature_statistics.yaml and return a DataFrame indexed by feature_id.

    Columns:
      dataset_frequency        — fraction (0–1) of dataset proteins where feature fires
      dataset_pct_when_present — fraction (0–1) of residues activated when feature is present

    Values in the YAML are on a 0–100 percentage scale; this function divides by 100.
    Result is cached in _STATS_CACHE so repeated calls within a session are free.
    """
    sae_dir = Path(sae_dir).resolve()
    if sae_dir in _STATS_CACHE:
        return _STATS_CACHE[sae_dir]

    stats_path = sae_dir / "Per_feature_statistics.yaml"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Per_feature_statistics.yaml not found in {sae_dir}. "
            "Run the per-feature statistics step first."
        )

    with open(stats_path) as f:
        raw = yaml.safe_load(f)

    freq = np.array(raw["Per_prot_frequency_of_any_activation"], dtype=np.float32) / 100.0
    pct = np.array(raw["Per_prot_pct_activated_when_present"], dtype=np.float32) / 100.0

    df = pd.DataFrame(
        {"dataset_frequency": freq, "dataset_pct_when_present": pct},
        index=pd.RangeIndex(len(freq), name="feature_id"),
    )
    _STATS_CACHE[sae_dir] = df
    return df


# ─── Feature summary ──────────────────────────────────────────────────────────

def _specificity_label(freq: float) -> str:
    if freq < _SPECIFIC_THRESH:
        return "specific"
    if freq < _GENERAL_THRESH:
        return "semi-specific"
    return "general"


def _pattern_label(pct: float) -> str:
    if pct < _FOCAL_THRESH:
        return "focal"
    if pct < _BROAD_THRESH:
        return "semi-focal"
    return "broad"


def _nanmax_suppress(arr: np.ndarray, axis: int) -> np.ndarray:
    """np.nanmax with the all-NaN-slice warning suppressed (expected for dead features)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(arr, axis=axis)


def summarize_active_features(
    activations: np.ndarray,
    threshold: float,
    top_k: int,
    cached_stats: pd.DataFrame,
    sort_by_specificity: bool = True,
) -> pd.DataFrame:
    """
    Build a summary DataFrame for features that fire above threshold.

    Parameters
    ----------
    activations       : (L, dict_size) float32; may contain NaN for dead features
    threshold         : scalar; features with max < threshold are dropped
    top_k             : maximum rows to return
    cached_stats      : output of load_feature_stats()
    sort_by_specificity : if True, sort specific → semi-specific → general,
                          then max_activation desc within each tier.
                          if False, sort purely by max_activation desc.

    Returns
    -------
    DataFrame with columns:
      feature_id, max_activation, mean_activation, pct_residues_activated,
      dataset_frequency, dataset_pct_when_present, specificity, activation_pattern
    """
    max_act = _nanmax_suppress(activations, axis=0)  # (dict_size,); NaN for dead features
    active_mask = max_act > threshold
    active_feats = np.where(active_mask)[0]

    if len(active_feats) == 0:
        return pd.DataFrame(columns=[
            "feature_id", "max_activation", "mean_activation",
            "pct_residues_activated", "dataset_frequency",
            "dataset_pct_when_present", "specificity", "activation_pattern",
        ])

    acts_active = activations[:, active_feats]  # (L, n_active)
    records = {
        "feature_id": active_feats.astype(int),
        "max_activation": max_act[active_feats],
        "mean_activation": np.nanmean(acts_active, axis=0),
        "pct_residues_activated": np.nanmean(acts_active > threshold, axis=0),
    }
    df = pd.DataFrame(records)
    df = df.join(cached_stats[["dataset_frequency", "dataset_pct_when_present"]], on="feature_id")

    df["specificity"] = df["dataset_frequency"].apply(_specificity_label)
    df["activation_pattern"] = df["pct_residues_activated"].apply(_pattern_label)

    if sort_by_specificity:
        tier_order = {"specific": 0, "semi-specific": 1, "general": 2}
        df["_tier"] = df["specificity"].map(tier_order)
        df = df.sort_values(["_tier", "max_activation"], ascending=[True, False]).drop(columns="_tier")
    else:
        df = df.sort_values("max_activation", ascending=False)

    return df.head(top_k).reset_index(drop=True)


# ─── Concept associations ─────────────────────────────────────────────────────

def get_concept_associations(
    feature_ids: List[int],
    sae_dir: Path,
    split: str = "test",
) -> Dict[int, str]:
    """
    Return a mapping of feature_id → top concept name (by f1_per_domain).

    Tries sae_dir/results_{split}_counts/concept_f1_scores.csv, then the other split.
    Returns "-" for features with no match, and for all features if CSV is absent.
    """
    sae_dir = Path(sae_dir)
    csv_path = None
    for s in (split, "valid" if split == "test" else "test"):
        candidate = sae_dir / f"results_{s}_counts" / "concept_f1_scores.csv"
        if candidate.exists():
            csv_path = candidate
            break

    if csv_path is None:
        return {f: "-" for f in feature_ids}

    usecols = ["concept", "feature", "f1_per_domain"]
    f1_df = pd.read_csv(csv_path, usecols=usecols)
    best = (
        f1_df.sort_values("f1_per_domain", ascending=False)
        .drop_duplicates(subset=["feature"])
        .set_index("feature")["concept"]
    )
    return {f: str(best.get(f, "-")) for f in feature_ids}


# ─── Annotation shard lookup ──────────────────────────────────────────────────

def _annot_shard_index_path(annot_dir: Path) -> Path:
    return annot_dir / ".protein_annot_shard_index.json"


def build_annotation_shard_index(annot_dir: Path) -> Dict[str, Tuple[int, int, int]]:
    """
    Build (or load) a mapping of protein_id → (shard_idx, row_start, row_end)
    for all proteins in the annotation shards.

    Reads protein_data.tsv from each shard_N/ directory.  The row positions
    are computed from the cumulative sum of protein lengths within each shard.

    The index is written to annot_dir/.protein_annot_shard_index.json.
    """
    annot_dir = Path(annot_dir)
    index_path = _annot_shard_index_path(annot_dir)

    # Load cached index if fresh
    if index_path.exists():
        newest_tsv = max(
            (p.stat().st_mtime for p in annot_dir.rglob("protein_data.tsv")),
            default=0.0,
        )
        if index_path.stat().st_mtime >= newest_tsv:
            with open(index_path) as f:
                raw = json.load(f)
            raw.pop("_meta", None)
            # Convert lists back to tuples
            return {k: tuple(v) for k, v in raw.items()}

    print("Building annotation shard index …")
    mapping: Dict[str, Tuple[int, int, int]] = {}
    shard_dirs = sorted(
        [d for d in annot_dir.glob("shard_*") if d.is_dir()],
        key=lambda d: int(d.name.split("_")[1]),
    )
    for shard_dir in shard_dirs:
        shard_idx = int(shard_dir.name.split("_")[1])
        tsv = shard_dir / "protein_data.tsv"
        if not tsv.exists():
            continue
        df = pd.read_csv(tsv, sep="\t", usecols=["Entry", "Length"])
        df = df.dropna(subset=["Entry", "Length"])
        df["Length"] = df["Length"].astype(int)
        row = 0
        for _, entry_row in df.iterrows():
            pid = str(entry_row["Entry"]).upper()
            L = int(entry_row["Length"])
            mapping[pid] = (shard_idx, row, row + L)
            row += L

    out = {
        "_meta": {"built_at": datetime.now(timezone.utc).isoformat()},
        **{k: list(v) for k, v in mapping.items()},
    }
    with open(index_path, "w") as f:
        json.dump(out, f)
    print(f"Annotation index written to {index_path} ({len(mapping)} proteins)")
    return mapping


# ─── Annotation overlap ───────────────────────────────────────────────────────

def get_annotation_overlap(
    protein_id: str,
    annot_dir: Path,
    keep_idx: List[int],
    concept_names: List[str],
    activations: np.ndarray,
    threshold: float,
) -> List[dict]:
    """
    Compute per-concept overlap between annotated residues and SAE-activated residues.

    Looks up the protein independently in the annotation shards (protein_data.tsv),
    so the annotation shard index does not need to match the embedding shard index.

    Parameters
    ----------
    protein_id     : protein to look up
    annot_dir      : annotation root (contains shard_N/aa_concepts.npz + protein_data.tsv)
    keep_idx       : column indices in the raw annotation matrix to select
    concept_names  : name of each kept concept (len == len(keep_idx)); already filtered
    activations    : (L, dict_size) float32 for this protein
    threshold      : activation threshold

    Returns
    -------
    List of dicts sorted by n_annotated_residues desc, one entry per concept
    that has at least one annotated residue in this protein.
    """
    annot_dir = Path(annot_dir)
    annot_index = build_annotation_shard_index(annot_dir)
    pid_upper = protein_id.upper()

    if pid_upper not in annot_index:
        warnings.warn(
            f"Protein '{protein_id}' not found in any annotation shard under {annot_dir}; "
            "skipping annotation overlap."
        )
        return []

    shard_idx, row_start, row_end = annot_index[pid_upper]
    L_annot = row_end - row_start
    L_act = activations.shape[0]

    if L_annot != L_act:
        warnings.warn(
            f"Length mismatch for '{protein_id}': annotation L={L_annot}, "
            f"activation L={L_act}. Skipping overlap."
        )
        return []

    annot_path = annot_dir / f"shard_{shard_idx}" / "aa_concepts.npz"
    if not annot_path.exists():
        warnings.warn(f"Annotation file not found: {annot_path}; skipping overlap.")
        return []

    labels_sparse = load_npz(annot_path)
    label_rows = labels_sparse[row_start:row_end, :]   # sparse (L, n_raw)
    label_filtered = label_rows[:, keep_idx].toarray() # dense (L, n_kept)

    # Residues where any SAE feature fires above threshold.
    # _nanmax_suppress ignores NaN entries from dead features (rescale_factor=0).
    any_active = _nanmax_suppress(activations, axis=1) > threshold  # (L,) bool

    results = []
    for col_i, concept_name in enumerate(concept_names):
        ann_mask = label_filtered[:, col_i] > 0
        n_annotated = int(ann_mask.sum())
        if n_annotated == 0:
            continue

        n_activated = int(any_active.sum())
        n_overlap = int((ann_mask & any_active).sum())
        overlap_frac = n_overlap / n_annotated

        top_feat = _best_feature_for_concept(activations, ann_mask, threshold)

        results.append({
            "concept_name": concept_name,
            "n_annotated_residues": n_annotated,
            "n_activated_residues": n_activated,
            "n_overlap": n_overlap,
            "overlap_fraction": overlap_frac,
            "top_feature": top_feat,
        })

    results.sort(key=lambda r: r["n_annotated_residues"], reverse=True)
    return results


def _best_feature_for_concept(
    activations: np.ndarray,
    ann_mask: np.ndarray,
    threshold: float,
) -> Optional[int]:
    """Feature with highest recall at annotated residues; ties broken by max activation."""
    n_pos = int(ann_mask.sum())
    if n_pos == 0:
        return None
    acts_at_pos = activations[ann_mask, :]                       # (n_pos, dict_size)
    recall = np.nan_to_num((acts_at_pos > threshold).sum(axis=0), nan=0.0)
    max_at_pos = np.nan_to_num(acts_at_pos.max(axis=0), nan=0.0)
    best = int(np.lexsort((max_at_pos, recall))[-1])
    return best
