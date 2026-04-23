#!/usr/bin/env python
"""
Batch concept analysis for all trained SAEs.

Runs the following per-SAE steps, skipping any that are already complete:
  1. Normalize SAE (ae_normalized.pt)
  2. Compare activations vs validation set (results_valid_counts/)
  3. Calculate F1 on validation set
  4. Compare activations vs test set (results_test_counts/)
  5. Calculate F1 on test set
  6. Report metrics (heldout_top_pairings.csv)

Pre-computed shared inputs (annotation matrices, eval-set metadata, analysis
embeddings) must already exist. Steps 1/2/4 of the README walkthrough are
assumed done before calling this script.

Usage examples:
    # Dry run — print status table, no side effects
    python scripts/run_concept_analysis.py --dry_run

    # Single SAE
    python scripts/run_concept_analysis.py \
        --sae trained_saes/relu_ef16_steps100000_bs512_lr1e-05_l1-6e-02

    # All unevaluated SAEs
    python scripts/run_concept_analysis.py
"""

import argparse
import json
import shutil
import traceback
from pathlib import Path

import pandas as pd
import yaml

from interplm.analysis.concepts.calculate_f1 import combine_metrics_across_shards
from interplm.analysis.concepts.compare_activations import analyze_all_shards_in_set
from interplm.analysis.concepts.report_metrics import report_metrics, report_valid_metrics
from interplm.sae.normalize import normalize_sae_features

# ---------------------------------------------------------------------------
# Model → analysis-embeddings mapping
# Maps a model name fragment (substring match against model_name.lower())
# to a base directory under embeddings_base_dir.  The layer is appended as
# "layer_{layer_idx}" so any layer is supported without enumerating each one.
# ---------------------------------------------------------------------------
EMBEDDINGS_BASE_MAP: dict[str, str] = {
    "esm2_t6_8m_ur50d": "esm2_8m",   # facebook/esm2_t6_8M_UR50D or esm2_t6_8M_UR50D
    "esm2-8m":          "esm2_8m",   # shorthand used with published pretrained SAEs
    "esm2-8M":          "esm2_8m",   # shorthand used with published pretrained SAEs
    "esm2-650m":        "esm2_650m", # published ESM-2-650M SAEs
    "esm2-650M":        "esm2_650m", # published ESM-2-650M SAEs
    "progen2-small":    "progen2_small",
    "progen2-large":    "progen2_large",
}

# Published SAE model name → dummy config key used in interplm/sae/migration/
DUMMY_CONFIG_MAP: dict[str, str] = {
    "esm2-8m":  "esm2-8m",
    "esm2-650m": "esm2-650m",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_config(sae_dir: Path, model_name_override: str | None) -> bool:
    """
    Ensure config.yaml exists in sae_dir.  If it's missing and a
    model_name_override is provided that matches a known dummy config
    (e.g. 'esm2-8m'), copy the migration dummy config into place.
    Returns True if config is available afterwards.
    """
    cfg_path = sae_dir / "config.yaml"
    if cfg_path.exists():
        return True
    if model_name_override is None:
        return False
    key = model_name_override.lower().strip()
    if key not in DUMMY_CONFIG_MAP:
        return False
    migration_cfg = (
        Path(__file__).parent.parent
        / "interplm" / "sae" / "migration"
        / f"dummy_config_{DUMMY_CONFIG_MAP[key]}.yaml"
    )
    if not migration_cfg.exists():
        return False
    shutil.copy(migration_cfg, cfg_path)
    print(f"  Copied dummy config ({DUMMY_CONFIG_MAP[key]}) → {cfg_path}")
    return True


def load_config(sae_dir: Path) -> dict | None:
    """
    Load config.yaml as a plain dict.  Returns None if the file is absent or
    uses Python-object YAML tags (e.g. migration dummy configs) — callers fall
    back to CLI overrides in that case.  load_sae() handles those configs
    internally via TrainingRunConfig.from_yaml.
    """
    cfg_path = sae_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        try:
            return yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            return None


def resolve_embeddings(
    config: dict | None,
    embeddings_base_dir: Path,
    model_name_override: str | None = None,
    layer_override: int | None = None,
) -> Path | None:
    """Return the analysis embeddings directory for this SAE, or None if unknown."""
    # CLI overrides take priority; fall back to config
    if model_name_override and layer_override is not None:
        model_name = model_name_override
        layer_idx = layer_override
    elif config is not None:
        eval_cfg = config.get("eval_cfg") or {}
        model_name = eval_cfg.get("model_name") or model_name_override or ""
        layer_idx = eval_cfg.get("layer_idx") if eval_cfg.get("layer_idx") is not None else layer_override
    else:
        model_name = model_name_override or ""
        layer_idx = layer_override

    if not model_name or layer_idx is None:
        return None

    model_lower = model_name.lower()
    for fragment, base_dir in EMBEDDINGS_BASE_MAP.items():
        if fragment in model_lower:
            return embeddings_base_dir / base_dir / f"layer_{layer_idx}"

    return None


def shards_complete(output_dir: Path, eval_set_dir: Path) -> bool:
    """Return True if every shard_N_counts.npz required by the eval set exists."""
    meta_path = eval_set_dir / "metadata.json"
    if not meta_path.exists():
        return False
    with open(meta_path) as f:
        shards = json.load(f)["shard_source"]
    return all((output_dir / f"shard_{s}_counts.npz").exists() for s in shards)


def step_status(sae_dir: Path, annotations_dir: Path) -> dict[str, bool]:
    """Return a dict of {step_name: is_done} for the status table."""
    valid_dir = sae_dir / "results_valid_counts"
    test_dir = sae_dir / "results_test_counts"
    return {
        "normalize":    (sae_dir / "ae_normalized.pt").exists(),
        "valid_counts": shards_complete(valid_dir, annotations_dir / "valid"),
        "valid_f1":     (valid_dir / "concept_f1_scores.csv").exists(),
        "test_counts":  shards_complete(test_dir, annotations_dir / "test"),
        "test_f1":      (test_dir / "concept_f1_scores.csv").exists(),
        "report":       (test_dir / "heldout_top_pairings.csv").exists(),
    }


# ---------------------------------------------------------------------------
# Status table
# ---------------------------------------------------------------------------

def print_status_table(
    saes: list[Path],
    annotations_dir: Path,
    embeddings_base_dir: Path,
    model_name_override: str | None = None,
    layer_override: int | None = None,
) -> None:
    header = f"{'SAE':<55} {'model':<20} {'L':>3}  nrm  v_cnt  v_f1  t_cnt  t_f1  rpt"
    print(header)
    print("-" * len(header))

    for sae_dir in saes:
        config = load_config(sae_dir)
        embds_path = resolve_embeddings(
            config, embeddings_base_dir, model_name_override, layer_override
        )

        # Determine display model name / layer
        if model_name_override:
            model_name = model_name_override
            layer_idx = layer_override
        elif config is not None:
            eval_cfg = config.get("eval_cfg") or {}
            model_name = eval_cfg.get("model_name") or "?"
            layer_idx = eval_cfg.get("layer_idx")
        else:
            model_name = "?"
            layer_idx = None

        model_short = model_name.split("/")[-1][:18]

        if embds_path is None and config is None:
            print(f"  {sae_dir.name:<53}  (no config — use --model_name / --layer)")
            continue

        if embds_path is None:
            print(f"  {sae_dir.name:<53}  {model_short:<20} {str(layer_idx):>3}  (unknown model — skipped)")
            continue

        status = step_status(sae_dir, annotations_dir)

        def mark(done: bool) -> str:
            return " ok " if done else " -- "

        print(
            f"  {sae_dir.name:<53}  {model_short:<20} {str(layer_idx):>3}"
            f"  {mark(status['normalize'])}"
            f"  {mark(status['valid_counts'])}"
            f"  {mark(status['valid_f1'])}"
            f"  {mark(status['test_counts'])}"
            f"  {mark(status['test_f1'])}"
            f"  {mark(status['report'])}"
        )
    print()


# ---------------------------------------------------------------------------
# Per-SAE analysis
# ---------------------------------------------------------------------------

def run_sae(
    sae_dir: Path,
    annotations_dir: Path,
    embeddings_base_dir: Path,
    force: bool = False,
    model_name_override: str | None = None,
    layer_override: int | None = None,
) -> dict | None:
    """
    Run the full concept-analysis pipeline for one SAE directory.
    Returns the metrics summary dict (from report_metrics) or None if skipped.
    """
    name = sae_dir.name

    # If config is missing and we have a known model name, copy a dummy config
    ensure_config(sae_dir, model_name_override)

    config = load_config(sae_dir)
    embds_dir = resolve_embeddings(
        config, embeddings_base_dir, model_name_override, layer_override
    )

    if embds_dir is None:
        if config is None:
            print(f"[{name}] No config.yaml and no --model_name/--layer provided — skipping.")
        else:
            eval_cfg = config.get("eval_cfg") or {}
            print(
                f"[{name}] Unknown model ({eval_cfg.get('model_name')!r}, "
                f"layer {eval_cfg.get('layer_idx')}) — skipping."
            )
        return None
    print(f"[{name}] Using analysis embeddings from: {embds_dir}")

    valid_dir = sae_dir / "results_valid_counts"
    test_dir  = sae_dir / "results_test_counts"
    valid_eval = annotations_dir / "valid"
    test_eval  = annotations_dir / "test"

    def log(step: str, skipped: bool = False) -> None:
        suffix = "skipped (already done)" if skipped else "done"
        print(f"  [{name}] {step} ... {suffix}")

    # Step 1 — Normalize
    if not force and (sae_dir / "ae_normalized.pt").exists():
        log("normalize", skipped=True)
    else:
        print(f"  [{name}] normalize ...")
        normalize_sae_features(sae_dir=sae_dir, aa_embds_dir=embds_dir)
        log("normalize")

    # Step 2 — Compare activations (valid)
    if not force and shards_complete(valid_dir, valid_eval):
        log("compare activations [valid]", skipped=True)
    else:
        print(f"  [{name}] compare activations [valid] ...")
        analyze_all_shards_in_set(
            sae_dir=sae_dir,
            aa_embds_dir=embds_dir,
            eval_set_dir=valid_eval,
            output_dir=valid_dir,
        )
        log("compare activations [valid]")

    # Step 3 — Calculate F1 (valid)
    if not force and (valid_dir / "concept_f1_scores.csv").exists():
        log("calculate F1 [valid]", skipped=True)
    else:
        print(f"  [{name}] calculate F1 [valid] ...")
        combine_metrics_across_shards(
            eval_res_dir=valid_dir,
            eval_set_dir=valid_eval,
        )
        log("calculate F1 [valid]")

    # Step 3b — Report valid metrics
    valid_summary_path = valid_dir / "valid_metrics_summary.json"
    if not force and valid_summary_path.exists():
        log("report valid metrics", skipped=True)
    else:
        print(f"  [{name}] report valid metrics ...")
        report_valid_metrics(
            valid_path=valid_dir / "concept_f1_scores.csv",
            eval_set_dir=valid_eval,
        )
        log("report valid metrics")

    # Step 4 — Compare activations (test)
    if not force and shards_complete(test_dir, test_eval):
        log("compare activations [test]", skipped=True)
    else:
        print(f"  [{name}] compare activations [test] ...")
        analyze_all_shards_in_set(
            sae_dir=sae_dir,
            aa_embds_dir=embds_dir,
            eval_set_dir=test_eval,
            output_dir=test_dir,
        )
        log("compare activations [test]")

    # Step 5 — Calculate F1 (test)
    if not force and (test_dir / "concept_f1_scores.csv").exists():
        log("calculate F1 [test]", skipped=True)
    else:
        print(f"  [{name}] calculate F1 [test] ...")
        combine_metrics_across_shards(
            eval_res_dir=test_dir,
            eval_set_dir=test_eval,
        )
        log("calculate F1 [test]")

    # Step 6 — Report metrics
    summary_path = test_dir / "metrics_summary.json"
    if not force and summary_path.exists():
        log("report metrics", skipped=True)
        with open(summary_path) as f:
            return json.load(f)
    else:
        print(f"  [{name}] report metrics ...")
        summary = report_metrics(
            valid_path=valid_dir / "concept_f1_scores.csv",
            test_path=test_dir  / "concept_f1_scores.csv",
            eval_set_dir=test_eval,
        )
        log("report metrics")
        return summary


# ---------------------------------------------------------------------------
# SAE collection
# ---------------------------------------------------------------------------

def collect_saes(saes_dir: Path, force: bool, annotations_dir: Path) -> list[Path]:
    """Return all SAE dirs in saes_dir that have a config.yaml."""
    candidates = sorted(
        d for d in saes_dir.iterdir()
        if d.is_dir() and (d / "config.yaml").exists()
    )
    if force:
        return candidates
    # If not forcing, still return all — skip logic is handled per-step inside run_sae.
    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch concept analysis for all trained SAEs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sae",
        type=Path,
        default=None,
        help="Path to a single SAE directory. Overrides --saes_dir.",
    )
    parser.add_argument(
        "--saes_dir",
        type=Path,
        default=Path("trained_saes"),
        help="Root directory containing trained SAE subdirectories.",
    )
    parser.add_argument(
        "--annotations_dir",
        type=Path,
        default=Path("data/annotations/uniprotkb/processed"),
        help="Directory containing processed annotations and eval sets (valid/, test/).",
    )
    parser.add_argument(
        "--embeddings_base_dir",
        type=Path,
        default=Path("data/analysis_embeddings"),
        help="Base directory for analysis embeddings. Model/layer appended automatically.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=(
            "Override model name when config.yaml is missing. "
            "For published SAEs use 'esm2-8m' or 'esm2-650m'. "
            "For custom SAEs use the full model identifier matching EMBEDDINGS_MAP "
            "(e.g. 'esm2_t6_8M_UR50D' or 'progen2-small')."
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Override layer index when config.yaml is missing (e.g. 4 for ESM-2-8M).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all steps even if outputs already exist.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print status table and exit without running anything.",
    )
    args = parser.parse_args()

    # Resolve SAE list
    if args.sae is not None:
        saes = [args.sae.resolve()]
    else:
        if not args.saes_dir.exists():
            parser.error(f"--saes_dir not found: {args.saes_dir}")
        saes = collect_saes(args.saes_dir, args.force, args.annotations_dir)

    if not saes:
        print("No SAE directories found.")
        return

    # Print status table
    print(f"\nFound {len(saes)} SAE(s)\n")
    print_status_table(
        saes, args.annotations_dir, args.embeddings_base_dir,
        model_name_override=args.model_name,
        layer_override=args.layer,
    )

    if args.dry_run:
        print("Dry run — exiting without running analysis.")
        return

    # Run analysis
    n_completed = 0
    n_errored = 0
    all_results: list[dict] = []

    for sae_dir in saes:
        print(f"\n{'='*70}")
        print(f"  SAE: {sae_dir.name}")
        print(f"{'='*70}")
        try:
            summary = run_sae(
                sae_dir=sae_dir,
                annotations_dir=args.annotations_dir,
                embeddings_base_dir=args.embeddings_base_dir,
                force=args.force,
                model_name_override=args.model_name,
                layer_override=args.layer,
            )
            n_completed += 1
            if summary is not None:
                config = load_config(sae_dir)
                eval_cfg = (config.get("eval_cfg") or {}) if config else {}
                model_name = args.model_name or eval_cfg.get("model_name") or ""
                layer_idx = args.layer if args.layer is not None else eval_cfg.get("layer_idx")
                row = {k: v for k, v in summary.items() if not isinstance(v, dict)}
                # Merge in valid metrics if available
                valid_summary_path = sae_dir / "results_valid_counts" / "valid_metrics_summary.json"
                valid_row: dict = {}
                if valid_summary_path.exists():
                    with open(valid_summary_path) as _f:
                        valid_summary = json.load(_f)
                    valid_row = {
                        f"valid_{k}": v
                        for k, v in valid_summary.items()
                        if not isinstance(v, dict)
                    }
                all_results.append({
                    "sae": sae_dir.name,
                    "model_name": model_name,
                    "layer": layer_idx,
                    **row,
                    **valid_row,
                })
        except Exception:
            print(f"\n  [{sae_dir.name}] ERROR:")
            traceback.print_exc()
            n_errored += 1

    # Write consolidated results table
    if all_results:
        results_path = args.saes_dir / "concept_analysis_results.csv"
        # Merge with any existing rows so reruns of a subset don't wipe prior entries
        if results_path.exists():
            existing = pd.read_csv(results_path)
            updated = pd.concat(
                [existing[~existing["sae"].isin({r["sae"] for r in all_results})],
                 pd.DataFrame(all_results)],
                ignore_index=True,
            ).sort_values("sae")
        else:
            updated = pd.DataFrame(all_results).sort_values("sae")
        updated.to_csv(results_path, index=False)
        print(f"\nConsolidated results saved to {results_path}")

    print(f"\n{'='*70}")
    print(f"  Summary: {n_completed} completed, {n_errored} errored")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
