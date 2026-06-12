"""
Positional analysis: where in a protein do concept-associated features fire?

For N-terminal concepts (signal peptide, transit peptide) we compare:
- ESM-2-650M L24  (bidirectional)
- ProGen2-large L24  (causal / left-to-right)

Hypothesis: causal attention anchors N-terminal features more sharply at
the sequence start because later tokens cannot dilute early context.

Output: per-protein activation centroid + annotation centroid, saved to
comparison_results/positional_analysis.json
"""
import json
import sys
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from interplm.sae.dictionary import BatchTopKSAE, ReLUSAE

OUT_DIR = ROOT / "comparison_results"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda:0"

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODELS = {
    "ESM-2-650M L24": {
        "sae_dir": ROOT / "trained_saes/pretrained_esm2-650M-layer24",
        "embd_dir": ROOT / "data/analysis_embeddings/esm2_650m/layer_24",
        "arch": "relu",
    },
    "ProGen2-large L24 k=350": {
        "sae_dir": ROOT / "trained_saes/best_progen_large_24",
        "embd_dir": ROOT / "data/analysis_embeddings/progen2_large/layer_24",
        "arch": "btk",
    },
    "ProGen2-large L24 k=10": {
        "sae_dir": ROOT / "trained_saes/best_progen_large_24_k10",
        "embd_dir": ROOT / "data/analysis_embeddings/progen2_large/layer_24",
        "arch": "btk",
    },
}

# N-terminal concept columns in the annotation matrix
NTERMINAL_CONCEPTS = [
    "Signal peptide_any",
    "Signal peptide_Tat-type signal",
    "Transit peptide_any",
    "Transit peptide_Mitochondrion",
    "Transit peptide_Chloroplast",
]

# Test shards (25-49 based on results_test_counts filenames)
TEST_SHARDS = list(range(25, 50))
ANN_SHARD_DIR = ROOT / "data/annotations/uniprotkb/processed"
CONCEPT_COLS_FILE = ANN_SHARD_DIR / "uniprotkb_aa_concepts_columns.txt"


def load_concept_columns():
    with open(CONCEPT_COLS_FILE) as f:
        return [l.strip() for l in f]


def load_sae(cfg):
    sae_dir = cfg["sae_dir"]
    ae_path = sae_dir / "ae_normalized.pt"
    if not ae_path.exists():
        ae_path = sae_dir / "ae.pt"

    import yaml
    json_config_path = sae_dir / "config.json"
    yaml_config_path = sae_dir / "config.yaml"
    if json_config_path.exists():
        # ESM-2 pretrained uses JSON config
        with open(json_config_path) as f:
            config = json.load(f)
        arch = config.get("architecture", {})
        dim = arch.get("esm_dim", 1280)
        ef = arch.get("expansion_factor", 8)
        dict_size = arch.get("feature_dim", dim * ef)
        k = None
    elif yaml_config_path.exists():
        with open(yaml_config_path) as f:
            config = yaml.safe_load(f)
        trainer = config.get("trainer_cfg", {})
        dim = trainer.get("activation_dim", 1280)
        ef = trainer.get("expansion_factor", 4)
        dict_size = trainer.get("dictionary_size", dim * ef)
        k = trainer.get("k", None)
    else:
        raise FileNotFoundError(f"No config found in {sae_dir}")

    if cfg["arch"] == "btk" and k is not None:
        sae = BatchTopKSAE(dim, dict_size, k=k)
    else:
        sae = ReLUSAE(dim, dict_size)

    state = torch.load(ae_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and any(k.startswith("ae.") for k in state):
        state = {k.replace("ae.", ""): v for k, v in state.items() if k.startswith("ae.")}
    sae.load_state_dict(state, strict=False)
    sae.to(DEVICE).eval()
    return sae


def get_best_features(sae_dir, concept_names):
    """Return best feature ID per concept from heldout_top_pairings."""
    top_path = sae_dir / "results_test_counts/heldout_top_pairings.csv"
    if not top_path.exists():
        return {}
    df = pd.read_csv(top_path)
    result = {}
    for concept in concept_names:
        subset = df[df["concept"] == concept]
        if len(subset) == 0:
            continue
        best_row = subset.loc[subset["f1"].idxmax()]
        result[concept] = {"feature": int(best_row["feature"]),
                           "f1": float(best_row["f1"]),
                           "threshold_pct": float(best_row["threshold_pct"])}
    return result


def load_shard_data(shard_idx, embd_dir, ann_shard_dir, concept_col_indices):
    """Load embeddings and annotation mask for one shard."""
    embd_path = embd_dir / f"shard_{shard_idx}" / "embeddings.pt"
    ann_path = ann_shard_dir / f"shard_{shard_idx}" / "aa_concepts.npz"

    if not embd_path.exists() or not ann_path.exists():
        return None

    embd_data = torch.load(embd_path, map_location="cpu", weights_only=False)
    embeddings = embd_data["embeddings"]          # (total_residues, dim)
    boundaries = embd_data["boundaries"]           # list of (start, end)
    protein_ids = embd_data["protein_ids"]

    ann_raw = np.load(ann_path, allow_pickle=True)
    ann_mat = scipy.sparse.csr_matrix(
        (ann_raw["data"], ann_raw["indices"], ann_raw["indptr"]),
        shape=tuple(ann_raw["shape"])
    )
    # Extract columns for our concepts
    ann_cols = ann_mat[:, concept_col_indices].toarray().astype(np.float32)

    return embeddings, boundaries, protein_ids, ann_cols


@torch.no_grad()
def compute_activation_profile(sae, embeddings_chunk):
    """
    Run SAE on a protein's embedding chunk. Returns per-residue activation
    for each feature as numpy array (n_residues, dict_size).
    Uses chunked inference to avoid OOM.
    """
    chunk_size = 512
    n = embeddings_chunk.shape[0]
    all_acts = []
    for i in range(0, n, chunk_size):
        batch = embeddings_chunk[i:i+chunk_size].to(DEVICE, dtype=torch.float32)
        acts = sae.encode(batch)  # (chunk, dict_size)
        all_acts.append(acts.cpu().numpy())
    return np.concatenate(all_acts, axis=0)  # (n_residues, dict_size)


def normalized_centroid(profile):
    """Mean position (0=N-term, 1=C-term) weighted by activation magnitude."""
    n = len(profile)
    if n == 0:
        return None
    positions = np.arange(n) / max(n - 1, 1)
    total = profile.sum()
    if total < 1e-9:
        return None
    return float((positions * profile).sum() / total)


def annotation_centroid(ann_col):
    """Mean annotated position (0..1) for a binary annotation vector."""
    n = len(ann_col)
    positions = np.arange(n) / max(n - 1, 1)
    total = ann_col.sum()
    if total < 1e-9:
        return None
    return float((positions * ann_col).sum() / total)


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------
concept_cols = load_concept_columns()
print(f"Total concept columns: {len(concept_cols)}")

# Find indices for N-terminal concepts
concept_col_indices = []
concept_col_map = {}
for c in NTERMINAL_CONCEPTS:
    matches = [i for i, col in enumerate(concept_cols) if col == c]
    if matches:
        concept_col_indices.append(matches[0])
        concept_col_map[c] = matches[0]
    else:
        print(f"  WARNING: concept '{c}' not found in columns")

print(f"N-terminal concepts found: {list(concept_col_map.keys())}")

results = {}

for model_name, cfg in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    sae_dir = cfg["sae_dir"]
    embd_dir = cfg["embd_dir"]

    # Get best feature per concept
    best_features = get_best_features(sae_dir, list(concept_col_map.keys()))
    print(f"Best features: {best_features}")

    if not best_features:
        print("  No concept features found, skipping.")
        results[model_name] = {}
        continue

    # Load SAE
    print("Loading SAE...")
    sae = load_sae(cfg)
    print(f"  SAE loaded: {type(sae).__name__}")

    model_results = {c: [] for c in best_features}

    for shard_idx in TEST_SHARDS:
        data = load_shard_data(shard_idx, embd_dir, ANN_SHARD_DIR,
                               concept_col_indices)
        if data is None:
            continue

        embeddings, boundaries, protein_ids, ann_cols = data

        for prot_idx, (pid, (start, end)) in enumerate(zip(protein_ids, boundaries)):
            prot_embd = embeddings[start:end]   # (L, dim)
            prot_len = end - start

            # Only process proteins that have at least one N-terminal annotation
            prot_ann = ann_cols[start:end]       # (L, n_nterminal_concepts)
            has_any = prot_ann.sum() > 0
            if not has_any:
                continue

            # Run SAE
            act_profile = compute_activation_profile(sae, prot_embd)  # (L, dict_size)

            for concept, feat_info in best_features.items():
                feat_id = feat_info["feature"]
                local_col_idx = list(concept_col_map.keys()).index(concept)
                ann_vec = prot_ann[:, local_col_idx]

                if ann_vec.sum() == 0:
                    continue

                feat_acts = act_profile[:, feat_id]     # (L,)
                act_cent = normalized_centroid(feat_acts)
                ann_cent = annotation_centroid(ann_vec)

                if act_cent is None or ann_cent is None:
                    continue

                model_results[concept].append({
                    "protein_id": pid,
                    "length": prot_len,
                    "act_centroid": act_cent,
                    "ann_centroid": ann_cent,
                    "feature_id": feat_id,
                    "feature_f1": feat_info["f1"],
                    "act_mean": float(feat_acts.mean()),
                    "act_max": float(feat_acts.max()),
                    "act_width_pct": float((feat_acts > 0).mean() * 100),
                    "ann_frac": float(ann_vec.mean() * 100),
                })

        total_processed = sum(len(v) for v in model_results.values())
        if total_processed > 0:
            sys.stdout.write(f"\r  Shard {shard_idx}: found {total_processed} annotated proteins")
            sys.stdout.flush()

    print()
    # Summarize per concept
    for concept, records in model_results.items():
        if not records:
            print(f"  {concept}: no annotated proteins found")
            continue
        df = pd.DataFrame(records)
        act_centroids = df["act_centroid"].values
        ann_centroids = df["ann_centroid"].values
        widths = df["act_width_pct"].values

        print(f"\n  Concept: {concept}")
        print(f"    Best feature: {best_features[concept]['feature']}  F1={best_features[concept]['f1']:.3f}")
        print(f"    N proteins: {len(df)}")
        print(f"    Annotation centroid: {ann_centroids.mean():.3f} ± {ann_centroids.std():.3f}")
        print(f"    Activation centroid: {act_centroids.mean():.3f} ± {act_centroids.std():.3f}")
        print(f"    Centroid bias (act - ann): {(act_centroids - ann_centroids).mean():.3f}")
        print(f"    Activation width: {widths.mean():.2f}% ± {widths.std():.2f}%")

    results[model_name] = {
        concept: [{"protein_id": r["protein_id"], "length": r["length"],
                   "act_centroid": r["act_centroid"], "ann_centroid": r["ann_centroid"],
                   "act_width_pct": r["act_width_pct"], "ann_frac": r["ann_frac"]}
                  for r in recs]
        for concept, recs in model_results.items()
    }

    del sae
    torch.cuda.empty_cache()

# --------------------------------------------------------------------------
# Save results
# --------------------------------------------------------------------------
out_path = OUT_DIR / "positional_analysis.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved positional results to {out_path}")

# --------------------------------------------------------------------------
# Cross-model comparison summary
# --------------------------------------------------------------------------
print("\n" + "="*60)
print("CROSS-MODEL POSITIONAL COMPARISON")
print("="*60)

summary_rows = []
for concept in NTERMINAL_CONCEPTS:
    for model_name in MODELS:
        recs = results.get(model_name, {}).get(concept, [])
        if not recs:
            continue
        act_cents = [r["act_centroid"] for r in recs]
        ann_cents = [r["ann_centroid"] for r in recs]
        widths = [r["act_width_pct"] for r in recs]
        bias = np.array(act_cents) - np.array(ann_cents)
        summary_rows.append({
            "Model": model_name,
            "Concept": concept,
            "N": len(recs),
            "Ann centroid": f"{np.mean(ann_cents):.3f}",
            "Act centroid": f"{np.mean(act_cents):.3f}",
            "Centroid bias": f"{bias.mean():.3f}",
            "Bias std": f"{bias.std():.3f}",
            "Width%": f"{np.mean(widths):.2f}",
        })

df_pos = pd.DataFrame(summary_rows)
if not df_pos.empty:
    print(df_pos.to_string(index=False))
    df_pos.to_csv(OUT_DIR / "table5_positional_summary.csv", index=False)
else:
    print("No positional results computed.")
