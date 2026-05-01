"""
Quick-and-dirty per-protein rank evaluation for rare local concepts.

For a chosen (concept, feature) pair, ask:
- For each protein with at least one annotated residue of `concept`,
  where do the annotated residue(s) rank within that protein's
  feature activation profile?
- Report Hits@1, Hits@5, Hits@10, MRR, AUPRC (per protein).
- Compare to a random baseline (uniform random ranking).
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import load_npz
from sklearn.metrics import precision_recall_curve, auc


def auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUPRC computed manually via precision_recall_curve to dodge an
    array-shape API quirk in this sklearn version."""
    p, r, _ = precision_recall_curve((labels > 0).astype(np.int8), scores)
    # auc expects monotonic x — recall returned by sklearn is descending
    # so reverse it
    return float(auc(r[::-1], p[::-1]))

from interplm.sae.inference import load_sae

DEVICE = "cuda:0"
SAE_DIR = Path("/scratch/tmp/hrbanh/InterPLM/trained_saes/best_progen_large_24")
EMBED_DIR = Path("/scratch/tmp/hrbanh/InterPLM/data/analysis_embeddings/progen2_large/layer_24")
ANNOT_DIR = Path("/scratch/tmp/hrbanh/InterPLM/data/annotations/uniprotkb/processed")
TEST_META = json.loads((ANNOT_DIR / "test" / "metadata.json").read_text())
KEEP_IDX = TEST_META["indices_of_concepts_to_keep"]  # filtered_idx -> raw_col

# Filtered concept index → top feature (from earlier F1 analysis)
TARGETS = [
    {"name": "Binding site_ATP",        "filt_idx": 29, "feature": 7196},
    {"name": "Active site_any",         "filt_idx": 27, "feature": 5222},
    {"name": "Active site_Nucleophile", "filt_idx": 2,  "feature": 5222},
    {"name": "Binding site_Zn(2+)",     "filt_idx": 30, "feature": 7229},
]
SHARDS = [25, 26, 27, 28]  # subset of test shards for speed
K_LIST = [1, 5, 10, 20]


def per_protein_metrics(activations: np.ndarray, labels: np.ndarray):
    """
    activations, labels: 1D arrays of length L (residues in one protein).
    `labels` may contain domain IDs (>0) rather than 0/1 — we binarize.
    Returns dict with hits@k, precision@k, recall@k, MRR, AUPRC.
    """
    bin_labels = (labels > 0).astype(np.int8)
    n_pos = int(bin_labels.sum())
    if n_pos == 0:
        return None
    L = len(activations)
    # Sort positions by activation desc; ranks are 1-indexed
    order = np.argsort(-activations, kind="stable")
    rank_of_pos = np.empty(L, dtype=np.int64)
    rank_of_pos[order] = np.arange(1, L + 1)
    pos_idx = np.flatnonzero(bin_labels)
    pos_ranks = rank_of_pos[pos_idx]
    best_rank = int(pos_ranks.min())
    out = {f"hit@{k}": int(best_rank <= k) for k in K_LIST}
    # precision@k / recall@k: how many positives fall in top-k
    for k in K_LIST:
        keff = min(k, L)
        n_in_topk = int((pos_ranks <= keff).sum())
        out[f"prec@{k}"] = n_in_topk / keff
        out[f"rec@{k}"] = n_in_topk / n_pos
    out["mrr"] = 1.0 / best_rank
    out["best_rank"] = best_rank
    out["L"] = L
    out["n_pos"] = n_pos
    # AUPRC needs at least one negative
    if n_pos < L:
        out["auprc"] = auprc(bin_labels, activations)
    else:
        out["auprc"] = float("nan")
    return out


def random_baseline(L: int, n_pos: int, n_pos_idx: np.ndarray = None):
    """Expected hits@k under uniform random ranking with n_pos positives."""
    out = {}
    for k in K_LIST:
        # P(at least one of n_pos positives in top-k) = 1 - C(L-n_pos, k)/C(L,k)
        if k >= L:
            out[f"hit@{k}"] = 1.0
        else:
            from math import comb
            denom = comb(L, k)
            num = comb(L - n_pos, k) if (L - n_pos) >= k else 0
            out[f"hit@{k}"] = 1 - num / denom
    return out


def main():
    print(f"Loading SAE from {SAE_DIR}")
    sae = load_sae(SAE_DIR, model_name="ae_normalized.pt", device=DEVICE)
    sae.eval()
    target_feats = sorted({t["feature"] for t in TARGETS})

    # Per-target accumulators
    results = {t["name"]: [] for t in TARGETS}
    rand_results = {t["name"]: [] for t in TARGETS}

    for shard in SHARDS:
        emb_path = EMBED_DIR / f"shard_{shard}" / "embeddings.pt"
        annot_path = ANNOT_DIR / f"shard_{shard}" / "aa_concepts.npz"
        if not emb_path.exists() or not annot_path.exists():
            print(f"Skipping shard {shard}: missing data")
            continue
        print(f"\n=== Shard {shard} ===")
        bundle = torch.load(emb_path, map_location="cpu", weights_only=False)
        emb = bundle["embeddings"]  # (n_aa, d_model) fp16
        boundaries = bundle["boundaries"]
        protein_ids = bundle["protein_ids"]
        labels_sparse = load_npz(annot_path)  # (n_aa_shard, 1038)
        assert emb.shape[0] == labels_sparse.shape[0], (
            f"emb {emb.shape[0]} vs labels {labels_sparse.shape[0]}"
        )

        # Compute SAE feature activations for the target features
        # in chunks to avoid OOM
        chunk = 32768
        n_aa = emb.shape[0]
        feat_acts = np.empty((n_aa, len(target_feats)), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, n_aa, chunk):
                e = emb[s:s + chunk].to(DEVICE).float()
                # encode_feat_subset returns (chunk, len(feat_list))
                acts = sae.encode_feat_subset(
                    e, target_feats, normalize_features=True
                )
                feat_acts[s:s + chunk] = acts.cpu().numpy()

        feat_to_col = {f: i for i, f in enumerate(target_feats)}

        # For each target concept, gather per-protein metrics
        for tgt in TARGETS:
            raw_col = KEEP_IDX[tgt["filt_idx"]]
            feat_col = feat_to_col[tgt["feature"]]
            label_col = labels_sparse[:, raw_col].toarray().ravel()

            n_proteins_with_concept = 0
            for (start, end), pid in zip(boundaries, protein_ids):
                lab = label_col[start:end]
                if lab.sum() == 0:
                    continue
                n_proteins_with_concept += 1
                acts = feat_acts[start:end, feat_col]
                m = per_protein_metrics(acts, lab)
                if m is not None:
                    results[tgt["name"]].append(m)
                    rb = random_baseline(m["L"], m["n_pos"])
                    rand_results[tgt["name"]].append(rb)
            print(
                f"  {tgt['name']}: {n_proteins_with_concept} proteins with concept"
            )

    print("\n========== RESULTS ==========\n")
    for tgt in TARGETS:
        rs = results[tgt["name"]]
        rrs = rand_results[tgt["name"]]
        if not rs:
            print(f"{tgt['name']}: no proteins with concept in evaluated shards")
            continue
        n = len(rs)
        print(f"\n[{tgt['name']}] feature {tgt['feature']} | n_proteins={n}")
        for k in K_LIST:
            hk = np.mean([r[f"hit@{k}"] for r in rs])
            rhk = np.mean([r[f"hit@{k}"] for r in rrs])
            pk = np.mean([r[f"prec@{k}"] for r in rs])
            rck = np.mean([r[f"rec@{k}"] for r in rs])
            # random precision@k = n_pos / L (expected fraction of positives in any k positions)
            rpk = np.mean([r["n_pos"] / r["L"] for r in rs])
            print(
                f"  k={k:<2}: hit@k={hk:.3f}(rand {rhk:.3f}) "
                f"prec@k={pk:.3f}(rand {rpk:.3f}) rec@k={rck:.3f}"
            )
        mrr = np.mean([r["mrr"] for r in rs])
        print(f"  MRR    : {mrr:.3f}")
        auprcs = [r["auprc"] for r in rs if not np.isnan(r["auprc"])]
        if auprcs:
            print(f"  AUPRC  : mean={np.mean(auprcs):.3f}  median={np.median(auprcs):.3f}")
        med_L = int(np.median([r["L"] for r in rs]))
        med_n = float(np.median([r["n_pos"] for r in rs]))
        print(f"  median protein length={med_L}, median positives/protein={med_n}")


if __name__ == "__main__":
    main()
