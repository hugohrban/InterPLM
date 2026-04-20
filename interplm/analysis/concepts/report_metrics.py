import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from interplm.analysis.concepts.concept_constants import (
    binary_meta_cols,
    categorical_concepts,
    categorical_meta_cols,
    paired_binary_cols,
)

# While we do find these useful to examine, we don't want to report them in the final
# metrics as they do not necessarily represent biological concepts (as each amino acid
# is just a single token).
concept_types_to_ignore = ["amino_acid"]

# Ordered longest-first so startswith matching is unambiguous
_TYPE_PREFIXES: list[str] = sorted(
    [c[0] for c in categorical_concepts]
    + categorical_meta_cols
    + binary_meta_cols
    + paired_binary_cols
    + ["amino_acid"],
    key=len,
    reverse=True,
)

# _TYPE_PREFIXES = ['Compositional bias', 'Compositional bias', 'Modified residue', 'Modified residue', 'Transit peptide', 'Signal peptide', 'Disulfide bond', 'Glycosylation', 'Binding site', 'Active site', 'Domain [FT]', 'Zinc finger', 'Zinc finger', 'Domain [FT]', 'Beta strand', 'Coiled coil', 'Lipidation', 'amino_acid', 'Cofactor', 'Region', 'Region', 'Motif', 'Motif', 'Helix', 'Turn']


def extract_concept_type(concept: str) -> str:
    """Return the high-level annotation type for a concept string."""
    for prefix in _TYPE_PREFIXES:
        if concept == prefix or concept.startswith(prefix + "_"):
            return prefix
    return concept.split("_")[0] if "_" in concept else concept


def identify_top_feature_per_concept(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the top feature per concept based on the maximum F1 score.

    Args:
        df: DataFrame containing F1 scores for each feature and concept

    Returns:
        DataFrame containing the top feature per concept
    """
    # Get indices of feature concept pairs for best feature per concept
    df = df[
        ~df["concept"].str.contains(
            "|".join(concept_types_to_ignore), case=False, na=False
        )
    ]
    top_feat_per_concept = df.sort_values(
        by=["f1_per_domain", "f1"], ascending=False
    ).drop_duplicates("concept")
    return top_feat_per_concept[["feature", "concept"]]


def identify_all_top_pairings(
    df: pd.DataFrame, top_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Identify all feature-concept pairs above a threshold F1 score.

    Args:
        df: DataFrame containing F1 scores for each feature and concept
        top_threshold: Minimum F1 score threshold for considering a pairing

    Returns:
        DataFrame containing all feature-concept pairs above threshold
    """
    df = df[
        ~df["concept"].str.contains(
            "|".join(concept_types_to_ignore), case=False, na=False
        )
    ]

    print(
        f"Compared {df['feature'].nunique():,} features (with 1+ true positive) to {df['concept'].nunique():,} concepts (that are not amino acids)"
    )

    top_feat_concept_pairs = (
        df[df["f1_per_domain"] > top_threshold]
        .sort_values(["f1_per_domain", "f1"], ascending=False)
        .drop_duplicates(subset=["feature", "concept"], keep="first")
    )
    return top_feat_concept_pairs


def find_top_heldout_feat_per_concept(
    df_valid: pd.DataFrame, df_test: pd.DataFrame
) -> pd.Series:
    """
    Calculate the best F1 score per concept based on the held-out test set.

    Args:
        df_valid: DataFrame containing F1 scores for each feature and concept in the validation set
        df_test: DataFrame containing F1 scores for each feature and concept in the test set

    Returns:
        Series containing best F1 scores per concept in test set
    """
    top_feat_per_concept_valid = identify_top_feature_per_concept(df_valid)

    # Merge test set with validation top pairs to get matching feature-concept pairs
    matched_pairs = pd.merge(
        df_test, top_feat_per_concept_valid, on=["feature", "concept"], how="inner"
    )

    return matched_pairs.sort_values(
        ["f1_per_domain", "f1"], ascending=False
    ).drop_duplicates(subset="concept", keep="first")


def find_all_top_heldout_feats(
    df_valid: pd.DataFrame, df_test: pd.DataFrame, top_threshold: float = 0.5
) -> int:
    """
    Calculate the number of top feature-concept pairs in the held-out test set.

    Args:
        df_valid: DataFrame containing F1 scores for each feature and concept in the validation set
        df_test: DataFrame containing F1 scores for each feature and concept in the test set
        top_threshold: Minimum F1 score threshold for considering a pairing

    Returns:
        Number of feature-concept pairs above threshold in test set
    """
    top_feat_per_concept_valid = identify_all_top_pairings(df_valid, top_threshold)

    # Merge test set with validation top pairs to get matching feature-concept pairs
    matched_pairs = pd.merge(
        df_test,
        top_feat_per_concept_valid[["concept", "feature"]],
        on=["feature", "concept"],
        how="inner",
    )

    matched_pairs = matched_pairs[matched_pairs["f1_per_domain"] > top_threshold]
    matched_pairs = matched_pairs.sort_values(
        ["f1_per_domain", "f1"], ascending=False
    ).drop_duplicates(subset=["feature", "concept"], keep="first")
    return matched_pairs


def _compute_f1_distribution(f1_series: pd.Series) -> dict:
    """Return distribution statistics for a series of per-concept best F1 scores."""
    vals = f1_series.dropna().values
    if len(vals) == 0:
        return {}
    return {
        "median_f1_per_concept": round(float(np.median(vals)), 6),
        "p90_f1_per_concept": round(float(np.percentile(vals, 90)), 6),
        "frac_f1_above_0.7": round(float((vals > 0.7).mean()), 4),
        "frac_f1_above_0.9": round(float((vals > 0.9).mean()), 4),
    }


def _compute_valid_test_gap(
    df_valid: pd.DataFrame,
    top_feat_per_concept_test: pd.DataFrame,
) -> float | None:
    """
    Mean signed F1 drop from validation to test for the valid-selected feature per concept.
    Positive = valid F1 is higher than test F1 (expected).
    """
    top_selection = identify_top_feature_per_concept(df_valid)  # (feature, concept)

    # Best valid F1 for each selected (feature, concept) pair
    df_valid_deduped = (
        df_valid.sort_values(["f1_per_domain", "f1"], ascending=False)
        .drop_duplicates(["feature", "concept"])
    )
    valid_f1s = pd.merge(
        top_selection,
        df_valid_deduped[["feature", "concept", "f1_per_domain"]],
        on=["feature", "concept"],
    ).rename(columns={"f1_per_domain": "f1_valid"})

    test_f1s = top_feat_per_concept_test[["concept", "f1_per_domain"]].rename(
        columns={"f1_per_domain": "f1_test"}
    )

    gap_df = pd.merge(valid_f1s[["concept", "f1_valid"]], test_f1s, on="concept")
    if gap_df.empty:
        return None
    return round(float((gap_df["f1_valid"] - gap_df["f1_test"]).mean()), 6)


def _compute_concept_coverage(
    all_top_feats: pd.DataFrame,
    eval_set_dir: Path,
) -> tuple[int | None, float | None, dict | None]:
    """
    Return (n_total_concepts, overall_coverage, per_type_coverage_dict).
    Returns (None, None, None) if eval_set_dir / aa_concepts_columns.txt is missing.
    """
    concept_names_path = eval_set_dir / "aa_concepts_columns.txt"
    if not concept_names_path.exists():
        return None, None, None

    with open(concept_names_path) as f:
        all_concepts = [line.strip() for line in f.read().split("\n") if line.strip()]

    non_aa = [
        c for c in all_concepts
        if not any(ignore in c for ignore in concept_types_to_ignore)
    ]
    n_total = len(non_aa)
    identified = set(all_top_feats["concept"].unique())
    n_identified = len(identified & set(non_aa))
    coverage = round(n_identified / n_total, 4) if n_total else None

    # Per-type breakdown
    type_totals: dict[str, int] = {}
    for c in non_aa:
        t = extract_concept_type(c)
        type_totals[t] = type_totals.get(t, 0) + 1

    type_identified: dict[str, int] = {}
    for c in identified:
        t = extract_concept_type(c)
        type_identified[t] = type_identified.get(t, 0) + 1

    per_type = {
        t: {
            "identified": type_identified.get(t, 0),
            "total": total,
            "fraction": round(type_identified.get(t, 0) / total, 4),
        }
        for t, total in sorted(type_totals.items())
    }

    return n_total, coverage, per_type


def _compute_polysemanticity(all_top_feats: pd.DataFrame) -> dict:
    """
    For each feature that passes the threshold, count how many distinct concepts it
    is paired with.  We split this into two flavours:

    - within-type: a feature is paired with >1 concept of the *same* annotation type
      (e.g. two different Domain subtypes).  This is the suspicious kind — it suggests
      the feature is genuinely mixing unrelated biology.

    - cross-type only: the feature covers multiple annotation types but only one concept
      per type.  This is largely expected: a residue in a kinase active site will
      simultaneously have labels for Domain [FT], Binding site, Active site, etc.,
      so a feature that detects "kinase active site" will look polysemantic here even
      though it is monosemantic in any meaningful sense.

    Caveat: because our annotation types describe overlapping coordinate spaces rather
    than disjoint partitions, the raw mean_concepts_per_feature number will always be
    inflated.  Use within-type polysemanticity as the primary signal and treat
    cross-type as expected background.
    """
    df = all_top_feats[["feature", "concept"]].copy()
    df["concept_type"] = df["concept"].map(extract_concept_type)

    feature_concept_counts = df.groupby("feature")["concept"].nunique()
    n_features_total = len(feature_concept_counts)

    # Within-type: any (feature, type) group that has more than one distinct concept
    per_feature_type = df.groupby(["feature", "concept_type"])["concept"].nunique()
    within_type_poly_features = (
        per_feature_type[per_feature_type > 1]
        .index.get_level_values("feature")
        .unique()
    )
    frac_within = len(within_type_poly_features) / n_features_total if n_features_total else 0.0

    # Cross-type only: polysemantic overall but not within any single type
    overall_poly = feature_concept_counts[feature_concept_counts > 1].index
    cross_type_only = set(overall_poly) - set(within_type_poly_features)
    frac_cross_only = len(cross_type_only) / n_features_total if n_features_total else 0.0

    return {
        "mean_concepts_per_feature": round(float(feature_concept_counts.mean()), 4),
        "median_concepts_per_feature": round(float(feature_concept_counts.median()), 4),
        "frac_features_polysemantic": round(float((feature_concept_counts > 1).mean()), 4),
        "frac_features_within_type_polysemantic": round(frac_within, 4),
        "frac_features_cross_type_only": round(frac_cross_only, 4),
    }


def _compute_rank_stability(
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    top_k: int = 20,
    min_features: int = 5,
) -> dict | None:
    """
    For each concept, rank its features by validation F1 and compute Spearman
    correlation with how those same features rank on the test set.

    Why: if a feature genuinely detects a concept, its rank among competing features
    should be consistent across splits.  A high mean correlation means our
    validation-based feature selection is reliable; a low one means the rankings are
    noisy and we should distrust which feature "won" on validation.

    Why top-K only (default 20):
    The full feature ranking per concept has a long tail of features with F1 near zero
    whose precise ordering is pure noise (a single TP difference can flip ranks).
    Restricting to the top-K focuses the correlation on the part of the ranking that
    actually matters for downstream decisions, and avoids inflating or deflating the
    correlation with meaningless tail comparisons.

    Caveats:
    - Only computed for concepts with >= min_features features having any TP on valid,
      since Spearman on very few points is unreliable.
    - Features in top-K on valid that have *zero* TP on test (and thus don't appear in
      df_test at all) are dropped from the merged set, which can reduce effective K.
      This is conservative but appropriate: if the feature had zero TP on test, it truly
      did not generalise.
    """
    # Exclude amino acid concepts (same as rest of the pipeline)
    ignore_pat = "|".join(concept_types_to_ignore)
    df_v = df_valid[~df_valid["concept"].str.contains(ignore_pat, case=False, na=False)]
    df_t = df_test[~df_test["concept"].str.contains(ignore_pat, case=False, na=False)]

    # Best F1 per (feature, concept) across thresholds
    df_v_best = (
        df_v.sort_values(["f1_per_domain", "f1"], ascending=False)
        .drop_duplicates(["feature", "concept"])
    )
    df_t_best = (
        df_t.sort_values(["f1_per_domain", "f1"], ascending=False)
        .drop_duplicates(["feature", "concept"])
    )

    correlations: list[float] = []
    for concept, valid_grp in df_v_best.groupby("concept"):
        if len(valid_grp) < min_features:
            continue

        top_valid = valid_grp.nlargest(top_k, "f1_per_domain")
        test_grp = df_t_best[
            (df_t_best["concept"] == concept)
            & (df_t_best["feature"].isin(top_valid["feature"]))
        ]

        merged = pd.merge(
            top_valid[["feature", "f1_per_domain"]].rename(columns={"f1_per_domain": "f1_valid"}),
            test_grp[["feature", "f1_per_domain"]].rename(columns={"f1_per_domain": "f1_test"}),
            on="feature",
            how="inner",
        )
        if len(merged) < min_features:
            continue

        corr, _ = spearmanr(merged["f1_valid"], merged["f1_test"])
        if not np.isnan(corr):
            correlations.append(float(corr))

    if not correlations:
        return None

    return {
        "rank_stability_mean_spearman": round(float(np.mean(correlations)), 4),
        "rank_stability_median_spearman": round(float(np.median(correlations)), 4),
        "rank_stability_n_concepts_evaluated": len(correlations),
        "rank_stability_top_k": top_k,
        "rank_stability_min_features": min_features,
    }


def report_metrics(
    valid_path: Path,
    test_path: Path,
    eval_set_dir: Optional[Path] = None,
    top_threshold: float = 0.5,
) -> dict:
    """
    Report the best F1 scores per concept in the held-out test set.

    Args:
        valid_path: Path to validation F1 scores
        test_path: Path to test F1 scores
        eval_set_dir: Eval set directory (for aa_concepts_columns.txt); enables concept coverage
        top_threshold: Minimum F1 score threshold for considering a pairing

    Returns:
        Dict with summary statistics
    """
    df_valid = pd.read_csv(valid_path)
    df_test = pd.read_csv(test_path)

    top_feat_per_concept_path = test_path.parent / "heldout_top_pairings.csv"
    all_top_feats_path = test_path.parent / "heldout_all_top_pairings.csv"
    summary_path = test_path.parent / "metrics_summary.json"

    top_feat_per_concept = find_top_heldout_feat_per_concept(df_valid, df_test)
    top_feat_per_concept.to_csv(top_feat_per_concept_path, index=True, header=True)

    all_top_feats = find_all_top_heldout_feats(df_valid, df_test, top_threshold)
    all_top_feats.to_csv(all_top_feats_path, index=False, header=True)

    avg_f1 = float(top_feat_per_concept["f1_per_domain"].mean())
    n_concepts = int(all_top_feats["concept"].nunique())
    n_features = int(all_top_feats["feature"].nunique())

    f1_dist = _compute_f1_distribution(top_feat_per_concept["f1_per_domain"])
    valid_test_gap = _compute_valid_test_gap(df_valid, top_feat_per_concept)
    polysemanticity = _compute_polysemanticity(all_top_feats)
    rank_stability = _compute_rank_stability(df_valid, df_test)

    n_total_concepts, coverage, per_type_coverage = (
        _compute_concept_coverage(all_top_feats, eval_set_dir)
        if eval_set_dir is not None
        else (None, None, None)
    )

    summary: dict = {
        "avg_f1_per_concept": round(avg_f1, 6),
        **f1_dist,
        "n_concepts_identified": n_concepts,
        "n_features_with_concept": n_features,
        "valid_test_f1_gap": valid_test_gap,
        **polysemanticity,
        **(rank_stability or {}),
        "top_threshold": top_threshold,
    }
    if n_total_concepts is not None:
        summary["n_total_concepts"] = n_total_concepts
        summary["concept_coverage"] = coverage
        summary["per_type_coverage"] = per_type_coverage

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Saved best pairings per concept to {top_feat_per_concept_path} and all top pairings to {all_top_feats_path}"
    )
    print(f"Saved summary metrics to {summary_path}")
    print("-" * 50)
    print(f"Average best F1 per concept in test set: {avg_f1:.3f}")
    print(f"  median={f1_dist.get('median_f1_per_concept', '?'):.3f}  "
          f"p90={f1_dist.get('p90_f1_per_concept', '?'):.3f}  "
          f">0.7: {f1_dist.get('frac_f1_above_0.7', '?'):.1%}  "
          f">0.9: {f1_dist.get('frac_f1_above_0.9', '?'):.1%}")
    if valid_test_gap is not None:
        print(f"Valid→test F1 gap (mean): {valid_test_gap:+.3f}")
    print(f"Number of concepts identified: {n_concepts}", end="")
    if coverage is not None:
        print(f"  ({coverage:.1%} of {n_total_concepts} total)")
    else:
        print()
    print(f"Number of features associated with a concept: {n_features}")
    if per_type_coverage:
        print("Per-type coverage:")
        for t, stats in per_type_coverage.items():
            print(f"  {t:<30} {stats['identified']:>4}/{stats['total']:<4} ({stats['fraction']:.1%})")
    print(f"Polysemanticity: mean {polysemanticity['mean_concepts_per_feature']:.2f} concepts/feature  "
          f"within-type {polysemanticity['frac_features_within_type_polysemantic']:.1%}  "
          f"cross-type only {polysemanticity['frac_features_cross_type_only']:.1%}")
    if rank_stability:
        print(f"Rank stability (top-{rank_stability['rank_stability_top_k']}): "
              f"mean Spearman={rank_stability['rank_stability_mean_spearman']:.3f}  "
              f"median={rank_stability['rank_stability_median_spearman']:.3f}  "
              f"over {rank_stability['rank_stability_n_concepts_evaluated']} concepts")

    return summary


if __name__ == "__main__":
    from tap import tapify

    tapify(report_metrics)
