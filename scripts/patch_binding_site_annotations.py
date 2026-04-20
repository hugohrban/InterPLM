#!/usr/bin/env python
"""
Patch mis-extracted annotation columns in existing aa_concepts.npz shard files.

The original extraction pipeline had a bug in process_categorical_feature: it hardcoded
a /note="..." regex, but some UniProtKB feature types use a different field:
  - Binding site  → /ligand="..."
  - Cofactor      → /Name="..."

This meant all columns for those types were silently left as zeros even though the column
names were correctly enumerated.  This script re-extracts the affected columns from the
already-saved protein_data.tsv files and patches them into each shard's aa_concepts.npz.

Usage:
    python scripts/patch_binding_site_annotations.py
    python scripts/patch_binding_site_annotations.py --uniprot_dir data/annotations/uniprotkb/processed
    python scripts/patch_binding_site_annotations.py --shard 0   # single shard for testing
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from interplm.analysis.concepts.concept_constants import categorical_concepts
from interplm.analysis.concepts.extract_annotations import add_sequence_features
from interplm.analysis.concepts.parsing_utils import process_categorical_feature

# Only these concept types need patching — they use a non-"note" field in the raw data.
CONCEPTS_TO_PATCH = {
    name: sep
    for name, _, sep in categorical_concepts
    if sep != "note"
}
# e.g. {"Binding site": "ligand", "Cofactor": "Name"}


def get_concept_info(
    uniprot_dir: Path,
) -> dict[str, tuple[list[str], list[int]]]:
    """
    For each concept type that needs patching, return (subcategories, col_indices).
    subcategories: list of subcat names as stored in the column file (including "any")
    col_indices:   column positions in aa_concepts.npz
    """
    with open(uniprot_dir / "uniprotkb_aa_concepts_columns.txt") as f:
        all_cols = f.read().splitlines()

    result = {}
    for concept_name in CONCEPTS_TO_PATCH:
        prefix = f"{concept_name}_"
        subcats = [c.replace(prefix, "") for c in all_cols if c.startswith(prefix)]
        indices = [i for i, c in enumerate(all_cols) if c.startswith(prefix)]
        if subcats:
            result[concept_name] = (subcats, indices)
    return result


def extract_concept_columns(
    protein_tsv: Path,
    concept_name: str,
    separator_name: str,
    subcategories: list[str],
) -> np.ndarray:
    """
    Re-extract columns for one concept type from a protein_data.tsv shard.
    Returns a dense uint32 array of shape (n_aa, len(subcategories)).

    subcategories already includes the "any" catch-all as the last element.
    """
    df = pd.read_csv(protein_tsv, sep="\t")

    total_aa = int(df["Length"].sum())

    if concept_name not in df.columns or df[concept_name].isnull().all():
        return np.zeros((total_aa, len(subcategories)), dtype=np.uint32)

    category_options = subcategories  # already includes "any"
    current_index = {cat: 1 for cat in category_options}
    col_data: dict[str, list[list]] = {cat: [] for cat in category_options}

    col_short = df[concept_name].dropna().iloc[0].split(" ")[0]

    for _, row in df.iterrows():
        results, current_index = process_categorical_feature(
            row[concept_name],
            col_short,
            category_options,
            int(row["Length"]),
            current_index,
            separator_name=separator_name,
        )
        for cat, result in zip(category_options, results):
            col_data[cat].append(result)

    arrays = {
        cat: np.concatenate([np.array(arr, dtype=np.uint32) for arr in col_data[cat]])
        for cat in subcategories
    }
    return np.column_stack([arrays[cat] for cat in subcategories])


def patch_shard(
    shard_id: int,
    uniprot_dir: Path,
    concept_info: dict[str, tuple[list[str], list[int]]],
) -> None:
    shard_dir = uniprot_dir / f"shard_{shard_id}"
    tsv_path = shard_dir / "protein_data.tsv"
    npz_path = shard_dir / "aa_concepts.npz"

    if not tsv_path.exists():
        print(f"  Shard {shard_id}: protein_data.tsv missing — skipping")
        return

    mat = sparse.load_npz(npz_path).toarray()

    total_patched = 0
    for concept_name, (subcats, col_indices) in concept_info.items():
        separator = CONCEPTS_TO_PATCH[concept_name]
        new_cols = extract_concept_columns(tsv_path, concept_name, separator, subcats)

        assert new_cols.shape[0] == mat.shape[0], (
            f"Shard {shard_id} / {concept_name}: AA count mismatch — "
            f"matrix {mat.shape[0]} vs extracted {new_cols.shape[0]}"
        )

        mat[:, col_indices] = new_cols
        total_patched += int((new_cols > 0).sum())

    sparse.save_npz(npz_path, sparse.csr_matrix(mat.astype(np.uint32)))
    print(f"  Shard {shard_id}: {total_patched:,} non-zero entries patched across {list(concept_info)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch mis-extracted annotation columns into existing aa_concepts.npz shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--uniprot_dir",
        type=Path,
        default=Path("data/annotations/uniprotkb/processed"),
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=None,
        help="Process a single shard (for testing). Omit to process all.",
    )
    parser.add_argument("--num_shards", type=int, default=None, help="Number of shards to process (for testing).")
    args = parser.parse_args()

    concept_info = get_concept_info(args.uniprot_dir)
    print(f"Concepts to patch: {list(CONCEPTS_TO_PATCH)}")
    for name, (subcats, indices) in concept_info.items():
        print(f"  {name}: {len(subcats)} subcategories, columns {indices[0]}–{indices[-1]}")

    if args.shard is not None:
        shards = [args.shard]
    else:
        shards = sorted(
            int(d.name.split("_")[1])
            for d in args.uniprot_dir.iterdir()
            if d.is_dir() and d.name.startswith("shard_") and (d / "aa_concepts.npz").exists()
        )
        if args.num_shards is not None:
            shards = shards[:args.num_shards]
        print(f"\nFound {len(shards)} shards to patch")

    for shard_id in tqdm(shards, desc="Patching shards"):
        patch_shard(shard_id, args.uniprot_dir, concept_info)

    print("\nDone. Next steps:")
    print("  1. python scripts/build_concept_prevalence.py")
    print("  2. Re-run make_eval_subset with --force_include_patterns 'Binding site' 'Cofactor'")
    print("  3. Re-run concept analysis for all SAEs")


if __name__ == "__main__":
    main()
