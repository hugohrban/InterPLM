#!/usr/bin/env python
"""
Systematically evaluate SAE features for a biological concept.

Two modes:
  f1_guided          — use concept_f1_scores.csv to rank features, then run rank eval
  annotation_enrichment — find features that fire at annotated residues (no F1 involved)

Both modes produce the same structured JSON output saved under:
  {sae_dir}/rank_eval/{concept}_{mode}.json

Results are cached: use --skip_if_cached to skip recomputation when the JSON exists.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from interplm.analysis.concepts.compare_activations import load_concept_names
from interplm.analysis.concepts.rank_eval import (
    augment_with_f1,
    concept_to_filename,
    get_or_build_enrichment_cache,
    get_top_features_for_concept,
    load_results,
    query_enrichment_cache,
    run_rank_eval_for_concept,
    save_results,
    search_concepts,
)
from interplm.sae.inference import load_sae


def rank_eval_concepts(
    sae_dir: Path,
    embed_dir: Path,
    annot_dir: Path,
    concept_query: str,
    mode: str = "f1_guided",
    split: str = "test",
    top_k_features: int = 5,
    shards: Optional[List[int]] = None,
    n_examples: int = 10,
    device: str = "cuda:0",
    chunk_size_rank: int = 32768,
    chunk_size_enrich: int = 2048,
    k_list: Optional[List[int]] = None,
    print_examples: bool = True,
    act_threshold: float = 0.05,
    skip_if_cached: bool = False,
):
    """
    Evaluate SAE features for a biological concept using rank-based metrics.

    Args:
        sae_dir: Path to trained SAE directory (contains ae_normalized.pt, config.yaml)
        embed_dir: Directory containing per-shard embeddings.pt files
        annot_dir: Annotations root directory (contains shard_* subdirs with aa_concepts.npz,
                   and {split}/ subdir with metadata.json + aa_concepts_columns.txt)
        concept_query: Search string to find concept (e.g. "ATP" or exact "Binding site_ATP")
        mode: "f1_guided" or "annotation_enrichment"
        split: Which split's metadata/F1 CSV to use ("test" or "valid")
        top_k_features: Number of top features to evaluate
        shards: Explicit list of shard indices; if None, uses all shards from metadata
        n_examples: Number of example proteins to store per feature (both positive and high-activation)
        device: CUDA device string
        chunk_size_rank: Batch size for rank eval SAE inference (residues per batch)
        chunk_size_enrich: Batch size for enrichment SAE inference (smaller due to full dict output)
        k_list: Ranks to evaluate Hits@k at; default [1, 5, 10, 20]
        print_examples: Print per-feature high-activation example table after summary
        act_threshold: Activation threshold for counting activated residues in the examples table
        skip_if_cached: Skip computation if result JSON already exists; load and print cached results
    """
    if k_list is None:
        k_list = [1, 5, 10, 20]
    if mode not in ("f1_guided", "annotation_enrichment"):
        raise ValueError(f"mode must be 'f1_guided' or 'annotation_enrichment', got '{mode}'")

    sae_dir = Path(sae_dir)
    embed_dir = Path(embed_dir)
    annot_dir = Path(annot_dir)
    split_dir = annot_dir / split

    # ── Concept resolution (needed even for cache hit to get the filename) ────
    concept_col_file = split_dir / "aa_concepts_columns.txt"
    concept_names = load_concept_names(concept_col_file)

    matches = search_concepts(concept_query, concept_names)
    if len(matches) == 0:
        concept_types = sorted({n.split("_")[0] for n in concept_names if n})
        print(f"No concepts matching '{concept_query}'. Available types:")
        for ct in concept_types:
            print(f"  {ct}")
        sys.exit(1)
    elif len(matches) > 1:
        print(f"Multiple concepts match '{concept_query}'. Be more specific:")
        for filt_idx, name in matches:
            print(f"  [{filt_idx:4d}] {name}")
        sys.exit(1)

    filt_idx, concept_name = matches[0]
    out_path = sae_dir / "rank_eval" / concept_to_filename(concept_name, mode)

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if skip_if_cached and out_path.exists():
        print(f"Loading cached results from {out_path}")
        results = load_results(out_path)
        _print_summary(results)
        if print_examples:
            _print_examples_table(results, act_threshold=act_threshold)
        return

    print(f"\nConcept: '{concept_name}' (filt_idx={filt_idx})")

    # ── Load metadata ─────────────────────────────────────────────────────────
    meta = json.loads((split_dir / "metadata.json").read_text())
    keep_idx: List[int] = meta["indices_of_concepts_to_keep"]
    raw_concept_col = keep_idx[filt_idx]

    if shards is None:
        shards = sorted(meta["shard_source"])
    print(f"Shards: {shards}")

    # ── Load SAE ──────────────────────────────────────────────────────────────
    model_name = "ae_normalized.pt" if (sae_dir / "ae_normalized.pt").exists() else "ae.pt"
    print(f"Loading SAE from {sae_dir}/{model_name}")
    sae = load_sae(sae_dir, model_name=model_name, device=device)
    sae.eval()
    print(f"SAE: {sae.__class__.__name__}, dict_size={sae.dict_size}\n")

    # ── Feature discovery ─────────────────────────────────────────────────────
    if mode == "f1_guided":
        f1_csv = sae_dir / f"results_{split}_counts" / "concept_f1_scores.csv"
        if not f1_csv.exists():
            raise FileNotFoundError(
                f"F1 CSV not found at {f1_csv}. Run concept analysis first."
            )
        print(f"Loading F1 scores from {f1_csv} ...")
        usecols = [
            "concept", "feature", "f1_per_domain", "f1",
            "precision", "recall", "threshold_pct", "tp", "fp",
        ]
        f1_df = pd.read_csv(f1_csv, usecols=usecols)
        features_df = get_top_features_for_concept(concept_name, f1_df, top_k=top_k_features)
        print(f"Top {len(features_df)} features by f1_per_domain:")
        for _, r in features_df.iterrows():
            print(
                f"  feature {int(r.feature):6d}  f1_per_domain={r.f1_per_domain:.4f}"
                f"  precision={r.precision:.4f}  tp={int(r.tp)}"
            )

    else:  # annotation_enrichment
        cache = get_or_build_enrichment_cache(
            sae=sae,
            embed_dir=embed_dir,
            annot_dir=annot_dir,
            shards=shards,
            sae_dir=sae_dir,
            device=device,
            chunk_size=chunk_size_enrich,
            sae_model_name=model_name,
        )
        enrich_df = query_enrichment_cache(
            raw_concept_col=raw_concept_col,
            cache=cache,
            top_k_features=top_k_features,
        )
        f1_csv = sae_dir / f"results_{split}_counts" / "concept_f1_scores.csv"
        features_df = augment_with_f1(enrich_df, concept_name, f1_csv)
        print(f"Top {len(features_df)} features by enrichment ratio:")
        for _, r in features_df.iterrows():
            f1_str = ""
            if "f1_per_domain" in r.index and pd.notna(r.get("f1_per_domain")):
                f1_str = f"  f1={r.f1:.4f}  f1d={r.f1_per_domain:.4f}  prec={r.precision:.4f}"
            print(
                f"  feature {int(r.feature):6d}  enrichment={r.enrichment_ratio:.2f}x"
                f"  mean_at_concept={r.mean_act_at_concept:.4f}"
                f"  mean_at_bg={r.mean_act_at_background:.4f}"
                f"{f1_str}"
            )

    print()

    # ── Rank eval ─────────────────────────────────────────────────────────────
    print("Running per-protein rank evaluation ...")
    results = run_rank_eval_for_concept(
        concept_name=concept_name,
        concept_filt_idx=filt_idx,
        analysis_mode=mode,
        features_df=features_df,
        sae=sae,
        embed_dir=embed_dir,
        annot_dir=annot_dir,
        keep_idx=keep_idx,
        shards=shards,
        k_list=k_list,
        n_examples=n_examples,
        device=device,
        chunk_size=chunk_size_rank,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results(results, out_path)
    print(f"\nResults saved to: {out_path}")

    # ── Print ─────────────────────────────────────────────────────────────────
    _print_summary(results)
    if print_examples:
        _print_examples_table(results, act_threshold=act_threshold)


def _print_summary(results: dict) -> None:
    concept = results["concept"]
    mode = results["analysis_mode"]
    n_prot = results["n_proteins_with_concept"]
    k_list = results["k_list"]
    print(f"\n{'='*60}")
    print(f"RESULTS — {concept} [{mode}]")
    print(f"Proteins with concept in evaluated shards: {n_prot}")
    print(f"{'='*60}\n")

    for feat_data in results["features"]:
        feat_id = feat_data["feature_id"]
        ds = feat_data["discovery_stats"]
        agg = feat_data["aggregate_metrics"]
        rand = feat_data["random_baseline"]
        n = agg.get("n_proteins", 0)

        if ds["source"] == "f1_guided":
            print(
                f"Feature {feat_id}  [f1_per_domain={ds['f1_per_domain']:.4f}"
                f"  f1={ds['f1']:.4f}  prec={ds['precision']:.4f}"
                f"  tp={ds['tp']}]  n_proteins={n}"
            )
        else:
            f1_str = ""
            if "f1_per_domain" in ds:
                f1_str = f"  f1={ds['f1']:.4f}  f1d={ds['f1_per_domain']:.4f}  prec={ds['precision']:.4f}"
            print(
                f"Feature {feat_id}  [enrichment={ds['enrichment_ratio']:.2f}x"
                f"  mean_concept={ds['mean_act_at_concept']:.4f}"
                f"{f1_str}]  n_proteins={n}"
            )

        if n == 0:
            print("  (no labeled proteins in evaluated shards)\n")
            continue

        header = f"  {'k':<4}" + "".join(
            f"  {'hit@'+str(k):<12}{'prec@'+str(k):<12}{'rec@'+str(k):<10}"
            for k in k_list
        )
        print(header)

        row_model = "  " + " " * 4
        for k in k_list:
            hk = agg.get(f"hit@{k}", float("nan"))
            pk = agg.get(f"prec@{k}", float("nan"))
            rk = agg.get(f"rec@{k}", float("nan"))
            row_model += f"  {hk:.3f}       {pk:.3f}      {rk:.3f}    "
        print(row_model)

        mrr = agg.get("mrr", float("nan"))
        auprc_mean = agg.get("auprc_mean", float("nan"))
        auprc_med = agg.get("auprc_median", float("nan"))
        baseline = agg.get("auprc_baseline", float("nan"))
        med_L = agg.get("median_L", "?")
        med_n = agg.get("median_n_positive", "?")
        print(f"  MRR={mrr:.3f}  AUPRC mean={auprc_mean:.3f} median={auprc_med:.3f}"
              f" (baseline={baseline:.4f})")
        print(f"  median protein length={med_L}  median positives/protein={med_n}")

        pos_ex = feat_data.get("positive_examples", [])
        print(f"  Positive examples: {len(pos_ex)}"
              + (f" (best AUPRC={pos_ex[0]['metrics']['auprc']:.3f},"
                 f" best_rank={pos_ex[0]['metrics']['best_rank']})" if pos_ex else ""))
        hi_ex = feat_data.get("high_activation_examples", [])
        n_unannotated = sum(1 for e in hi_ex if not e["is_annotated"])
        print(f"  High-act examples: {len(hi_ex)} ({n_unannotated} unannotated)\n")


def _print_examples_table(results: dict, act_threshold: float = 0.05) -> None:
    """Print per-feature table of high-activation examples with localization metrics."""
    for feat_data in results["features"]:
        feat_id = feat_data["feature_id"]
        examples = feat_data.get("high_activation_examples", [])
        if not examples:
            continue

        print(f"Feature {feat_id} — high-activation examples (threshold={act_threshold})")
        print(f"  {'protein_id':<14} {'max_act':>7}  {'ann?':>6}  {'n_act':>5}  {'n_ann':>5}  "
              f"{'n_ovlp':>6}  {'peak_rank':>9}  {'prec@peak':>9}  note")
        print("  " + "-" * 90)

        for e in examples:
            acts = np.array(e["feature_activations"])
            pos_idx = set(e["positive_residue_indices"])
            is_ann = e["is_annotated"]
            ann_flag = "[ANN]" if is_ann else "     "
            n_activated = int((acts > act_threshold).sum())
            n_annotated = e.get("n_annotated", len(pos_idx))
            # n_overlap: annotated residues that are also activated above threshold
            acts_at_sites = e.get("activations_at_annotation_sites")
            if acts_at_sites is not None:
                n_overlap = sum(1 for v in acts_at_sites if v > act_threshold)
            else:
                n_overlap = sum(1 for i in pos_idx if acts[i] > act_threshold)

            if is_ann and pos_idx:
                order = np.argsort(-acts, kind="stable")
                rank_of_residue = {r: i + 1 for i, r in enumerate(order)}
                best_site_rank = min(rank_of_residue[p] for p in pos_idx)
                top_n = set(order[:best_site_rank].tolist())
                prec_at_peak = len(top_n & pos_idx) / best_site_rank
                peak_on_site = int(np.argmax(acts)) in pos_idx
                note = "peak ON site" if peak_on_site else f"peak off site (nearest site rank={best_site_rank})"
                rank_str = f"{best_site_rank:>9}"
                prec_str = f"{prec_at_peak:>9.3f}"
            else:
                rank_str = "        -"
                prec_str = "        -"
                note = "no annotation"

            print(f"  {e['protein_id']:<14} {e['max_activation']:>7.4f}  {ann_flag}  "
                  f"{n_activated:>5}  {n_annotated:>5}  {n_overlap:>6}  "
                  f"{rank_str}  {prec_str}  {note}")
        print()


if __name__ == "__main__":
    from tap import tapify
    tapify(rank_eval_concepts)
