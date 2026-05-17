#!/usr/bin/env python
"""
Find SAE features enriched in non-targeting-peptide (noTP) sequences
compared to transit-peptide (TP) sequences from a steering experiment.

Usage (TargetP2 mode):
    python scripts/find_antitarget_features.py \
        --fasta steering_experiments/transit_v2/fasta/f5900x5.0+f3884x10.0.fasta \
        --targetp steering_experiments/transit_v2/targetp/f5900x5.0+f3884x10.0_summary.targetp2 \
        --sae_dir trained_saes/best_progen_large_24 \
        --top_k 30

Usage (ps_scan mode):
    python scripts/find_antitarget_features.py \
        --fasta steering_experiments/sushi/combo_r1.fasta \
        --ps_scan steering_experiments/sushi/combo_r1.pff \
        --sae_dir trained_saes/best_progen_large_24 \
        --top_k 20
"""

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def parse_fasta(path: str) -> list[tuple[str, str]]:
    """Returns list of (header, sequence) tuples."""
    entries = []
    header = None
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        entries.append((header, "".join(seq_parts)))
    return entries


def parse_targetp(path: str) -> dict[str, str]:
    """Returns {seq_id: prediction} where prediction is 'SP', 'mTP', or 'noTP'."""
    labels = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            seq_id = parts[0]
            pred = parts[1]
            labels[seq_id] = pred
    return labels


def parse_ps_scan(path: str) -> set[str]:
    """Parse ps_scan PFF output (tab-separated). Returns set of sequence IDs with ≥1 match."""
    matched: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts:
                matched.add(parts[0])
    return matched


def embed_sequences_batch(
    sequences: list[str],
    embedder,
    layer: int,
    batch_size: int = 128,
) -> list[np.ndarray]:
    """
    Embed each sequence independently (variable-length → can't stack).
    Returns list of (L, d_model) arrays.
    """
    results = []
    for i, seq in enumerate(sequences):
        if i % 50 == 0:
            print(f"  Embedding {i}/{len(sequences)}...", end="\r")
        emb = embedder.embed_single_sequence(seq, layer)
        results.append(emb)
    print(f"  Embedding {len(sequences)}/{len(sequences)}... done")
    return results


def sae_activations(embeddings: list[np.ndarray], sae, device: str, chunk_size: int = 4096) -> list[np.ndarray]:
    """Run SAE encode on each embedding array. Returns list of (L, dict_size) arrays."""
    results = []
    for emb in embeddings:
        t = torch.from_numpy(emb).to(device)
        with torch.no_grad():
            acts = sae.encode(t).cpu().numpy()
        results.append(acts)
    return results


def compute_group_stats(
    activations_list: list[np.ndarray],
    stat: str = "max",  # "max" or "mean"
) -> np.ndarray:
    """
    For each sequence in the list, reduce (L, dict_size) -> (dict_size,) using `stat`.
    Returns (N, dict_size) matrix.
    """
    all_stats = []
    for acts in activations_list:
        if stat == "max":
            all_stats.append(np.max(acts, axis=0))
        else:
            all_stats.append(np.mean(acts, axis=0))
    return np.stack(all_stats, axis=0)  # (N, dict_size)


def find_antitarget_features(
    fasta: str,
    targetp: Optional[str] = None,
    ps_scan: Optional[str] = None,
    sae_dir: str = "trained_saes/best_progen_large_24",
    top_k: int = 30,
    stat: str = "max",
    min_tp: int = 10,
    device: str = "cuda:0",
    layer: Optional[int] = None,
    steered_features: Optional[list[int]] = None,
) -> None:
    """
    Main analysis: identify features enriched in positive vs negative sequences.

    Args:
        fasta:            Path to FASTA file from steering experiment
        targetp:          Path to TargetP2 output (use this OR --ps_scan)
        ps_scan:          Path to ps_scan PFF output (use this OR --targetp)
        sae_dir:          Path to SAE directory
        top_k:            Number of top anti-features to report
        stat:             'max' or 'mean' per-sequence SAE feature statistic
        min_tp:           Minimum positive sequences required to run analysis
        device:           CUDA device
        layer:            Override layer index (reads from config.yaml if None)
        steered_features: Feature IDs to report in summary section (default: [5900, 3884, 7356])
    """
    if targetp is None and ps_scan is None:
        raise ValueError("Provide either --targetp or --ps_scan")
    if targetp is not None and ps_scan is not None:
        raise ValueError("Provide either --targetp or --ps_scan, not both")

    sae_dir = Path(sae_dir)

    # ── Parse inputs ─────────────────────────────────────────────────────────
    if ps_scan is not None:
        print("Parsing FASTA and ps_scan PFF output...")
        seqs = parse_fasta(fasta)
        matched_ids = parse_ps_scan(ps_scan)
        tp_seqs, notp_seqs = [], []
        for header, seq in seqs:
            if header in matched_ids:
                tp_seqs.append(seq)
            else:
                notp_seqs.append(seq)
        label_source = f"ps_scan ({len(matched_ids)} matched IDs)"
    else:
        print("Parsing FASTA and TargetP outputs...")
        seqs = parse_fasta(fasta)
        labels = parse_targetp(targetp)
        tp_seqs, notp_seqs = [], []
        for header, seq in seqs:
            pred = labels.get(header, "noTP")
            is_tp = pred in ("SP", "mTP")
            if is_tp:
                tp_seqs.append(seq)
            else:
                notp_seqs.append(seq)
        label_source = "TargetP2"

    print(f"  Label source:   {label_source}")
    print(f"  TP sequences:   {len(tp_seqs)}")
    print(f"  noTP sequences: {len(notp_seqs)}")

    if len(tp_seqs) < min_tp:
        print(f"Error: only {len(tp_seqs)} TP sequences (need >= {min_tp}). Aborting.")
        sys.exit(1)

    # ── Load config ───────────────────────────────────────────────────────────
    import yaml
    config_path = sae_dir / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    eval_cfg = cfg.get("eval_cfg", cfg)
    model_name = eval_cfg.get("model_name")
    if layer is None:
        layer = eval_cfg.get("layer_idx")
    print(f"\nModel: {model_name}  layer: {layer}")

    # ── Load embedder ─────────────────────────────────────────────────────────
    print("Loading PLM embedder...")
    from interplm.embedders import get_embedder
    embedder_type = "progen2" if "progen" in model_name.lower() else "esm"
    embedder = get_embedder(embedder_type, model_name=model_name, device=device)
    embedder.load_model()

    # ── Load SAE ──────────────────────────────────────────────────────────────
    print("Loading SAE...")
    from interplm.sae.inference import load_sae
    sae_filename = "ae_normalized.pt" if (sae_dir / "ae_normalized.pt").exists() else "ae.pt"
    sae = load_sae(sae_dir, model_name=sae_filename, device=device)
    sae.eval()
    print(f"  dict_size={sae.dict_size}\n")

    # ── Embed and encode ──────────────────────────────────────────────────────
    print(f"Embedding {len(tp_seqs)} TP sequences...")
    tp_embs = embed_sequences_batch(tp_seqs, embedder, layer)
    print(f"Embedding {len(notp_seqs)} noTP sequences...")
    notp_embs = embed_sequences_batch(notp_seqs, embedder, layer)

    print("Running SAE encode...")
    tp_acts_list   = sae_activations(tp_embs, sae, device)
    notp_acts_list = sae_activations(notp_embs, sae, device)

    # ── Per-sequence statistics ───────────────────────────────────────────────
    print(f"Computing per-sequence {stat} activations...")
    tp_mat   = compute_group_stats(tp_acts_list, stat)    # (N_tp,   dict_size)
    notp_mat = compute_group_stats(notp_acts_list, stat)  # (N_notp, dict_size)

    n_tp   = tp_mat.shape[0]
    n_notp = notp_mat.shape[0]
    dict_size = tp_mat.shape[1]

    # ── Enrichment: mean(noTP) - mean(TP) ────────────────────────────────────
    mean_tp   = tp_mat.mean(axis=0)    # (dict_size,)
    mean_notp = notp_mat.mean(axis=0)  # (dict_size,)
    diff = mean_notp - mean_tp         # positive = enriched in noTP

    # Fraction of sequences where feature fires above threshold
    threshold = 0.05
    frac_tp_fire   = (tp_mat   > threshold).mean(axis=0)
    frac_notp_fire = (notp_mat > threshold).mean(axis=0)
    frac_diff = frac_notp_fire - frac_tp_fire

    # Combine: sort by mean_notp_act - mean_tp_act (anti-feature score)
    order = np.argsort(-diff)

    # ── Print top anti-features ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"TOP {top_k} ANTI-TARGET FEATURES (enriched in noTP, n_tp={n_tp}, n_notp={n_notp})")
    print(f"stat={stat}, threshold={threshold}")
    print(f"{'='*70}")
    print(f"{'feat':>6}  {'mean_noTP':>10}  {'mean_TP':>8}  {'diff':>8}  "
          f"{'%fire_noTP':>10}  {'%fire_TP':>8}  {'frac_diff':>10}")
    print("-" * 78)

    top_features = []
    for fid in order[:top_k]:
        m_notp = float(mean_notp[fid])
        m_tp   = float(mean_tp[fid])
        d      = float(diff[fid])
        f_notp = float(frac_notp_fire[fid])
        f_tp   = float(frac_tp_fire[fid])
        fd     = float(frac_diff[fid])
        top_features.append(fid)
        print(f"{fid:>6}  {m_notp:>10.4f}  {m_tp:>8.4f}  {d:>8.4f}  "
              f"{f_notp:>9.1%}  {f_tp:>7.1%}  {fd:>+9.1%}")

    print()

    # ── Print bottom (features enriched in TP = potential positive features) ──
    print(f"\n{'='*70}")
    print(f"TOP {top_k} PRO-TARGET FEATURES (enriched in TP, i.e. lower in noTP)")
    print(f"{'='*70}")
    print(f"{'feat':>6}  {'mean_TP':>8}  {'mean_noTP':>10}  {'diff':>8}  "
          f"{'%fire_TP':>8}  {'%fire_noTP':>10}  {'frac_diff':>10}")
    print("-" * 78)
    for fid in order[-top_k:][::-1]:
        m_notp = float(mean_notp[fid])
        m_tp   = float(mean_tp[fid])
        d      = float(diff[fid])
        f_notp = float(frac_notp_fire[fid])
        f_tp   = float(frac_tp_fire[fid])
        fd     = float(frac_diff[fid])
        print(f"{fid:>6}  {m_tp:>8.4f}  {m_notp:>10.4f}  {d:>8.4f}  "
              f"{f_tp:>7.1%}  {f_notp:>9.1%}  {fd:>+9.1%}")

    # ── Summary of existing steered features ─────────────────────────────────
    print()
    steered = steered_features if steered_features else [5900, 3884, 7356]
    print(f"{'='*70}")
    print("STEERED FEATURE BEHAVIOUR in noTP vs TP")
    print(f"{'feat':>6}  {'mean_TP':>8}  {'mean_noTP':>10}  {'diff':>8}  "
          f"{'%fire_TP':>8}  {'%fire_noTP':>10}")
    print("-" * 70)
    for fid in steered:
        m_notp = float(mean_notp[fid])
        m_tp   = float(mean_tp[fid])
        d      = float(diff[fid])
        f_tp   = float(frac_tp_fire[fid])
        f_notp = float(frac_notp_fire[fid])
        print(f"{fid:>6}  {m_tp:>8.4f}  {m_notp:>10.4f}  {d:>8.4f}  "
              f"{f_tp:>7.1%}  {f_notp:>9.1%}")

    print()
    print(f"Suggested negative steering candidates (top anti-features):")
    print(f"  {top_features[:10]}")
    print(f"\nNormalised anti-feature max activations (for choosing clamp values):")
    for fid in top_features[:10]:
        rescale = sae.activation_rescale_factor[fid].item()
        print(f"  feature {fid}: max_act={rescale:.4f}")


if __name__ == "__main__":
    from tap import tapify
    tapify(find_antitarget_features)
