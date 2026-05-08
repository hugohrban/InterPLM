#!/usr/bin/env python
"""
Per-protein SAE feature analysis.

Given a protein ID, finds which SAE features activate in it, characterizes
them by cross-dataset specificity (specific / semi-specific / general) and
within-protein activation pattern (focal / semi-focal / broad), and optionally
reports overlap with biological concept annotations.

Results can be saved to JSON for downstream use in the dashboard.
"""

import json
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from interplm.analysis.concepts.compare_activations import load_concept_names
from interplm.analysis.per_protein_analysis import (
    compute_full_activations,
    find_protein_in_shards,
    get_annotation_overlap,
    get_concept_associations,
    load_feature_stats,
    summarize_active_features,
)
from interplm.sae.inference import load_sae


def protein_feature_analysis(
    protein_id: str,
    sae_dir: Path,
    embed_dir: Path,
    annot_dir: Optional[Path] = None,
    sequence: Optional[str] = None,
    threshold: float = 0.05,
    top_k: int = 30,
    sort_by: str = "specificity",
    specific_only: bool = False,
    device: str = "cuda:0",
    chunk_size: int = 2048,
    save_json: bool = False,
    split: str = "test",
) -> None:
    """
    Analyse which SAE features activate in a given protein.

    Args:
        protein_id:    UniProt (or other) ID to query
        sae_dir:       Path to trained SAE directory (contains ae_normalized.pt, config.yaml)
        embed_dir:     Directory containing per-shard embedding files
        annot_dir:     Annotation root directory (optional; enables concept overlap report)
        sequence:      Raw amino-acid sequence; used as fallback when protein is not in shards
        threshold:     Activation threshold for counting a feature as "firing"
        top_k:         Maximum features to display / save
        sort_by:       "specificity" (specific → semi-specific → general, then max_act) or
                       "max_activation" (purely by activation magnitude)
        specific_only: Only show features with dataset_frequency < 0.10
        device:        CUDA device string
        chunk_size:    SAE forward-pass batch size (residues per chunk)
        save_json:     Save full results to sae_dir/protein_analysis/{protein_id}.json
        split:         Split whose concept F1 scores and annotation metadata to use
    """
    if sort_by not in ("specificity", "max_activation"):
        print(f"Error: --sort_by must be 'specificity' or 'max_activation', got '{sort_by}'")
        sys.exit(1)

    sae_dir = Path(sae_dir)
    embed_dir = Path(embed_dir)

    # ── Load SAE ──────────────────────────────────────────────────────────────
    model_name = "ae_normalized.pt" if (sae_dir / "ae_normalized.pt").exists() else "ae.pt"
    print(f"Loading SAE from {sae_dir}/{model_name} …")
    sae = load_sae(sae_dir, model_name=model_name, device=device)
    sae.eval()
    print(f"  {sae.__class__.__name__}  dict_size={sae.dict_size}\n")

    # ── Locate protein embeddings ─────────────────────────────────────────────
    protein_embeddings: np.ndarray
    shard_idx: Optional[int] = None

    try:
        protein_embeddings, shard_idx, _protein_idx = find_protein_in_shards(
            protein_id, embed_dir, annot_dir
        )
        print(f"Protein: {protein_id}  (L={len(protein_embeddings)}), found in shard_{shard_idx}")

    except ValueError as exc:
        if sequence is None:
            print(f"Error: {exc}")
            sys.exit(1)

        print(f"  Protein not found in shards. Embedding from provided sequence …")
        protein_embeddings = _embed_sequence(sequence, sae_dir, device)
        if protein_embeddings is None:
            sys.exit(1)
        print(f"Protein: {protein_id}  (L={len(protein_embeddings)}, embedded on-the-fly)")

    L = len(protein_embeddings)

    # ── Full SAE forward pass ─────────────────────────────────────────────────
    print(f"Running SAE forward pass (chunk_size={chunk_size}) …")
    activations = compute_full_activations(protein_embeddings, sae, device, chunk_size)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        n_active = int((np.nanmax(activations, axis=0) > threshold).sum())
    print(f"Features active above threshold={threshold}: {n_active}\n")

    if n_active == 0:
        print(f"Warning: no features fire above threshold={threshold} for this protein.")

    # ── Feature summary ───────────────────────────────────────────────────────
    cached_stats = load_feature_stats(sae_dir)
    sort_by_specificity = (sort_by == "specificity")
    df = summarize_active_features(
        activations, threshold, top_k, cached_stats, sort_by_specificity
    )

    if specific_only:
        df = df[df["specificity"] == "specific"].head(top_k).reset_index(drop=True)
        if df.empty:
            print("No specific features (dataset_frequency < 0.10) fire above threshold.")

    # ── Concept associations ──────────────────────────────────────────────────
    if not df.empty:
        concept_map = get_concept_associations(
            df["feature_id"].tolist(), sae_dir, split
        )
        df["top_concept"] = df["feature_id"].map(concept_map).fillna("-")
    else:
        df["top_concept"] = pd.Series(dtype=str)

    # ── Print feature table ───────────────────────────────────────────────────
    _print_feature_table(df, protein_id, L, sort_by)

    # ── Annotation overlap ────────────────────────────────────────────────────
    overlaps: List[dict] = []
    if annot_dir is not None:
        annot_dir = Path(annot_dir)
        split_dir = annot_dir / split
        meta_path = split_dir / "metadata.json"
        col_path = split_dir / "aa_concepts_columns.txt"
        if meta_path.exists() and col_path.exists():
            meta = json.loads(meta_path.read_text())
            keep_idx: List[int] = meta["indices_of_concepts_to_keep"]
            # aa_concepts_columns.txt in the split dir is already the filtered list
            # (one name per entry in keep_idx); concept_names[i] maps to keep_idx[i].
            concept_names = load_concept_names(col_path)
            overlaps = get_annotation_overlap(
                protein_id, annot_dir, keep_idx, concept_names,
                activations, threshold,
            )
            _print_overlap_table(overlaps)
        else:
            warnings.warn(
                f"Annotation metadata not found in {split_dir}; skipping overlap."
            )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if save_json:
        out_dir = sae_dir / "protein_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{protein_id}.json"
        _save_results(
            out_path, protein_id, L, shard_idx, threshold,
            n_active, df, activations, overlaps,
        )
        print(f"\nResults saved to: {out_path}")


# ─── Sequence embedding fallback ──────────────────────────────────────────────

def _embed_sequence(
    sequence: str,
    sae_dir: Path,
    device: str,
) -> Optional[np.ndarray]:
    """
    Embed a raw amino-acid sequence using the PLM inferred from the SAE config.
    Returns per-residue embeddings as (L, d_model) float32 ndarray, or None on failure.
    """
    import yaml as _yaml

    config_path = sae_dir / "config.yaml"
    if not config_path.exists():
        print(f"Error: cannot embed on the fly — config.yaml not found in {sae_dir}.")
        return None

    with open(config_path) as f:
        cfg = _yaml.safe_load(f)

    plm_embd_dir = cfg.get("dataloader_cfg", {}).get("plm_embd_dir", "")
    # Infer layer from the path (…/layer_24 → 24)
    parts = Path(plm_embd_dir).parts
    layer: Optional[int] = None
    for p in reversed(parts):
        if p.startswith("layer_"):
            try:
                layer = int(p.split("_")[1])
                break
            except ValueError:
                pass

    if layer is None:
        print(
            "Error: could not infer layer from SAE config plm_embd_dir. "
            "The path should contain a 'layer_N' component."
        )
        return None

    plm_path_lower = plm_embd_dir.lower()
    if "progen2" in plm_path_lower:
        return _embed_progen2(sequence, plm_embd_dir, layer, device)
    if "esm" in plm_path_lower:
        return _embed_esm(sequence, plm_embd_dir, layer, device)

    print(
        "Error: cannot determine PLM type from config plm_embd_dir. "
        f"Value: {plm_embd_dir!r}"
    )
    return None


def _embed_progen2(
    sequence: str,
    plm_embd_dir: str,
    layer: int,
    device: str,
) -> Optional[np.ndarray]:
    try:
        from interplm.embedders.progen2 import ProGen2

        # Guess the model name from the path (progen2_large → progen2-large)
        for part in reversed(Path(plm_embd_dir).parts):
            if "progen" in part.lower():
                model_name = part.replace("_", "-")
                break
        else:
            model_name = "progen2-large"

        print(f"  Embedding with ProGen2 model={model_name} layer={layer} …")
        embedder = ProGen2(model_name, device=device)
        embedder.load_model()
        return embedder.embed_single_sequence(sequence, layer)
    except Exception as e:
        print(f"Error embedding with ProGen2: {e}")
        return None


def _embed_esm(
    sequence: str,
    plm_embd_dir: str,
    layer: int,
    device: str,
) -> Optional[np.ndarray]:
    try:
        from interplm.embedders.esm import ESM

        # Guess model name; fall back to esm2_t33_650M_UR50D
        for part in reversed(Path(plm_embd_dir).parts):
            if "esm" in part.lower():
                model_name = part
                break
        else:
            model_name = "facebook/esm2_t33_650M_UR50D"

        print(f"  Embedding with ESM model={model_name} layer={layer} …")
        embedder = ESM(model_name, device=device)
        embedder.load_model()
        return embedder.embed_single_sequence(sequence, layer)
    except Exception as e:
        print(f"Error embedding with ESM: {e}")
        return None


# ─── Printing helpers ─────────────────────────────────────────────────────────

_COL_WIDTHS = {
    "feat":        6,
    "max_act":     8,
    "pct_res":     7,
    "ds_freq":     7,
    "specificity": 13,
    "act_pattern": 10,
    "top_concept": 30,
}


def _print_feature_table(
    df: pd.DataFrame,
    protein_id: str,
    L: int,
    sort_by: str,
) -> None:
    sort_desc = (
        "specificity then max activation"
        if sort_by == "specificity"
        else "max activation"
    )
    print(f"Top {len(df)} features sorted by {sort_desc}:")

    hdr = (
        f"{'feat':>6}  {'max_act':>7}  {'pct_res':>7}  {'ds_freq':>7}  "
        f"{'specificity':<13}  {'act_pattern':<11}  top_concept"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for _, row in df.iterrows():
        concept = str(row.get("top_concept", "-"))
        if len(concept) > 32:
            concept = concept[:29] + "…"
        print(
            f"{int(row['feature_id']):>6}  "
            f"{row['max_activation']:>7.4f}  "
            f"{row['pct_residues_activated']:>7.3f}  "
            f"{row['dataset_frequency']:>7.4f}  "
            f"{row['specificity']:<13}  "
            f"{row['activation_pattern']:<11}  "
            f"{concept}"
        )
    print()


def _print_overlap_table(overlaps: List[dict]) -> None:
    if not overlaps:
        print("Annotation overlap: (no annotated concepts found for this protein)\n")
        return

    print("Annotation overlap:")
    print(
        f"  {'concept':<35}  {'n_ann':>5}  {'n_act':>5}  "
        f"{'n_ovlp':>6}  {'ovlp_frac':>9}  {'top_feat':>8}"
    )
    print("  " + "-" * 82)
    for r in overlaps:
        name = r["concept_name"]
        if len(name) > 35:
            name = name[:32] + "…"
        top_f = r["top_feature"]
        top_f_str = str(top_f) if top_f is not None else "-"
        print(
            f"  {name:<35}  {r['n_annotated_residues']:>5}  "
            f"{r['n_activated_residues']:>5}  {r['n_overlap']:>6}  "
            f"{r['overlap_fraction']:>9.3f}  {top_f_str:>8}"
        )
    print()


# ─── JSON serialisation ───────────────────────────────────────────────────────

def _save_results(
    path: Path,
    protein_id: str,
    length: int,
    shard_idx: Optional[int],
    threshold: float,
    n_active: int,
    df: pd.DataFrame,
    activations: np.ndarray,
    overlaps: List[dict],
) -> None:
    top_features = []
    for _, row in df.iterrows():
        fid = int(row["feature_id"])
        profile = activations[:, fid].tolist()
        top_features.append(
            {
                "feature_id": fid,
                "max_activation": float(row["max_activation"]),
                "mean_activation": float(row["mean_activation"]),
                "pct_residues_activated": float(row["pct_residues_activated"]),
                "dataset_frequency": float(row["dataset_frequency"]),
                "dataset_pct_when_present": float(row["dataset_pct_when_present"]),
                "specificity": str(row["specificity"]),
                "activation_pattern": str(row["activation_pattern"]),
                "top_concept": str(row.get("top_concept", "-")),
                "activation_profile": profile,
            }
        )

    result = {
        "protein_id": protein_id,
        "length": length,
        "shard_idx": shard_idx,
        "threshold": threshold,
        "n_active_features": n_active,
        "top_features": top_features,
        "annotation_overlaps": overlaps,
    }
    with open(path, "w") as f:
        json.dump(result, f, allow_nan=True)


if __name__ == "__main__":
    from tap import tapify
    tapify(protein_feature_analysis)
