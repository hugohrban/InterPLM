#!/usr/bin/env python
"""
Build a dataset-wide concept prevalence table from the annotation metadata files.

Reads metadata.json + aa_concepts_columns.txt from the valid and test eval sets
and writes a single CSV with per-concept AA and domain counts for both splits.

Output: <annotations_dir>/concept_prevalence.csv

This file is dataset-specific, not SAE-specific — regenerate it if the annotation
set changes (e.g. different filtering threshold or UniProtKB snapshot).

Usage:
    python scripts/build_concept_prevalence.py
    python scripts/build_concept_prevalence.py \
        --annotations_dir data/annotations/uniprotkb/processed
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from interplm.analysis.concepts.report_metrics import extract_concept_type


def load_split(split_dir: Path) -> tuple[list[str], list[float], list[float], int]:
    """Return (concept_names, n_aa, n_domains, total_aa) for one split."""
    with open(split_dir / "metadata.json") as f:
        meta = json.load(f)
    with open(split_dir / "aa_concepts_columns.txt") as f:
        concepts = [line.strip() for line in f.read().split("\n") if line.strip()]
    return (
        concepts,
        meta["n_positive_aa_per_concept"],
        meta["n_positive_domains_per_concept"],
        int(meta["n_amino_acids"]),
    )


def build_prevalence_table(annotations_dir: Path) -> pd.DataFrame:
    valid_concepts, v_aa, v_dom, v_total = load_split(annotations_dir / "valid")
    test_concepts,  t_aa, t_dom, t_total = load_split(annotations_dir / "test")

    assert valid_concepts == test_concepts, (
        "Concept lists differ between valid and test — were they built from the same run?"
    )

    rows = []
    for concept, n_aa_v, n_dom_v, n_aa_t, n_dom_t in zip(
        valid_concepts, v_aa, v_dom, t_aa, t_dom
    ):
        rows.append({
            "concept": concept,
            "concept_type": extract_concept_type(concept),
            "n_aa_valid": int(n_aa_v),
            "n_domains_valid": int(n_dom_v),
            "frac_aa_valid": round(n_aa_v / v_total, 6),
            "n_aa_test": int(n_aa_t),
            "n_domains_test": int(n_dom_t),
            "frac_aa_test": round(n_aa_t / t_total, 6),
        })

    df = pd.DataFrame(rows).sort_values("n_aa_valid", ascending=False).reset_index(drop=True)

    # Metadata row at top of printout (not in CSV)
    print(f"Total AAs — valid: {v_total:,}  test: {t_total:,}")
    print(f"Concepts: {len(df)}")
    print(f"\nTop 10 by valid AA count:")
    print(df[["concept", "concept_type", "n_aa_valid", "n_domains_valid"]].head(10).to_string(index=False))
    print(f"\nBottom 10 by valid AA count:")
    print(df[["concept", "concept_type", "n_aa_valid", "n_domains_valid"]].tail(10).to_string(index=False))

    print(f"\nCounts by concept type (valid AAs):")
    type_summary = (
        df.groupby("concept_type")
        .agg(n_concepts=("concept", "count"), total_aa_valid=("n_aa_valid", "sum"))
        .sort_values("total_aa_valid", ascending=False)
    )
    print(type_summary.to_string())

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build concept prevalence table from annotation metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotations_dir",
        type=Path,
        default=Path("data/annotations/uniprotkb/processed"),
    )
    args = parser.parse_args()

    df = build_prevalence_table(args.annotations_dir)

    out_path = args.annotations_dir / "concept_prevalence.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
