"""
Systematic comparison of ESM-2 vs ProGen2 SAEs.

Outputs tables and per-concept metrics to a results directory.
Advisor note: concept sets differ (734 vs 666) across model pairs;
we always intersect before computing aggregate metrics.
"""
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
SAE_DIR = ROOT / "trained_saes"
OUT_DIR = ROOT / "comparison_results"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# SAE registry
# ---------------------------------------------------------------------------
SAES = {
    # Large pair (734-concept set)
    "ESM-2-650M L24": {
        "dir": "pretrained_esm2-650M-layer24",
        "model": "ESM-2-650M", "layer": 24, "arch": "ReLU",
        "dim": 1280, "ef": 8, "dict_size": 10240, "k": None,
        "pair": "large",
    },
    "ProGen2-large L24 k=350": {
        "dir": "best_progen_large_24",
        "model": "ProGen2-large", "layer": 24, "arch": "BTK",
        "dim": 2560, "ef": 4, "dict_size": 10240, "k": 350,
        "pair": "large",
    },
    "ProGen2-large L24 k=64": {
        "dir": "best_progen_large_24_k64",
        "model": "ProGen2-large", "layer": 24, "arch": "BTK",
        "dim": 2560, "ef": 4, "dict_size": 10240, "k": 64,
        "pair": "large",
    },
    "ProGen2-large L24 k=10": {
        "dir": "best_progen_large_24_k10",
        "model": "ProGen2-large", "layer": 24, "arch": "BTK",
        "dim": 2560, "ef": 4, "dict_size": 10240, "k": 10,
        "pair": "large",
    },
    # Layer 9 (same 734-set for ProGen2, but ESM-2 L9 is 666-set — no cross-pair row)
    "ESM-2-650M L9": {
        "dir": "pretrained_esm2-650M-layer9",
        "model": "ESM-2-650M", "layer": 9, "arch": "ReLU",
        "dim": 1280, "ef": 8, "dict_size": 10240, "k": None,
        "pair": "large_l9_esm",
    },
    "ProGen2-large L9": {
        "dir": "best_progen_large_9",
        "model": "ProGen2-large", "layer": 9, "arch": "ReLU",
        "dim": 2560, "ef": 4, "dict_size": 10240, "k": None,
        "pair": "large_l9_progen",
    },
    # Small pair (666-concept set)
    "ESM-2-8M L4": {
        "dir": "pretrained_esm2-8M-layer4",
        "model": "ESM-2-8M", "layer": 4, "arch": "ReLU",
        "dim": 320, "ef": 32, "dict_size": 10240, "k": None,
        "pair": "small",
    },
    "ProGen2-small L7": {
        "dir": "best_progen_small_7",
        "model": "ProGen2-small", "layer": 7, "arch": "BTK",
        "dim": 1024, "ef": 16, "dict_size": 10240, "k": 50,
        "pair": "small",
    },
}


def load_sae_data(name, cfg):
    d = SAE_DIR / cfg["dir"]
    result = {"name": name, **cfg}

    # --- metrics_summary ---
    ms_path = d / "results_test_counts" / "metrics_summary.json"
    if ms_path.exists():
        with open(ms_path) as f:
            ms = json.load(f)
        result["raw_metrics"] = ms
    else:
        result["raw_metrics"] = None

    # --- fidelity ---
    fe_path = d / "final_evaluation.yaml"
    if fe_path.exists():
        with open(fe_path) as f:
            fe = yaml.safe_load(f)
        result["pct_loss_recovered"] = fe["fidelity"]["pct_loss_recovered"]
        result["ce_with_sae"] = fe["fidelity"]["CE_w_sae_patching"]
    else:
        result["pct_loss_recovered"] = None
        result["ce_with_sae"] = None

    # --- concept F1 scores ---
    f1_path = d / "results_test_counts" / "concept_f1_scores.csv"
    if f1_path.exists():
        df = pd.read_csv(f1_path)
        # Keep only the best (feature, threshold) pair per concept
        best = df.groupby("concept")["f1"].max().reset_index()
        result["concept_f1_df"] = best
    else:
        result["concept_f1_df"] = None

    # --- feature statistics (for locality) ---
    fs_path = d / "Per_feature_statistics.yaml"
    if fs_path.exists():
        with open(fs_path) as f:
            fs = yaml.safe_load(f)
        result["pct_active_per_prot"] = np.array(fs["Per_prot_frequency_of_any_activation"])
        result["pct_width_when_present"] = np.array(fs["Per_prot_pct_activated_when_present"])
    else:
        result["pct_active_per_prot"] = None
        result["pct_width_when_present"] = None

    # --- heldout top pairings (best feature per concept) ---
    top_path = d / "results_test_counts" / "heldout_top_pairings.csv"
    if top_path.exists():
        result["top_pairings"] = pd.read_csv(top_path)
    else:
        result["top_pairings"] = None

    return result


def compute_intersection_metrics(data_a, data_b, threshold=0.5):
    """Recompute avg_F1 and coverage on the concept intersection of two SAEs."""
    if data_a["concept_f1_df"] is None or data_b["concept_f1_df"] is None:
        return None, None

    df_a = data_a["concept_f1_df"].set_index("concept")["f1"]
    df_b = data_b["concept_f1_df"].set_index("concept")["f1"]
    common = df_a.index.intersection(df_b.index)
    n_common = len(common)

    a_vals = df_a.loc[common]
    b_vals = df_b.loc[common]

    return {
        "n_common_concepts": n_common,
        f"{data_a['name']}_avg_f1": float(a_vals.mean()),
        f"{data_a['name']}_median_f1": float(a_vals.median()),
        f"{data_a['name']}_coverage": float((a_vals >= threshold).mean()),
        f"{data_a['name']}_n_identified": int((a_vals >= threshold).sum()),
        f"{data_b['name']}_avg_f1": float(b_vals.mean()),
        f"{data_b['name']}_median_f1": float(b_vals.median()),
        f"{data_b['name']}_coverage": float((b_vals >= threshold).mean()),
        f"{data_b['name']}_n_identified": int((b_vals >= threshold).sum()),
    }, common.tolist()


def per_type_coverage(data, common_concepts=None, threshold=0.5):
    """Coverage fraction per concept type (Domain, Signal peptide, etc.)."""
    if data["concept_f1_df"] is None:
        return {}
    df = data["concept_f1_df"].copy()
    if common_concepts is not None:
        df = df[df["concept"].isin(common_concepts)]
    df["type"] = df["concept"].str.split("_").str[0]
    result = {}
    for t, grp in df.groupby("type"):
        result[t] = {
            "identified": int((grp["f1"] >= threshold).sum()),
            "total": len(grp),
            "fraction": float((grp["f1"] >= threshold).mean()),
            "avg_f1": float(grp["f1"].mean()),
        }
    return result


def locality_for_concept_features(data, threshold=0.5):
    """
    For concept-associated features, return average activation width
    (% of residues activated when present).
    Lower = more local/specific.
    """
    if data["top_pairings"] is None or data["pct_width_when_present"] is None:
        return None
    tp = data["top_pairings"]
    # top_pairings has 'feature' column (int)
    widths = []
    for _, row in tp.iterrows():
        feat_id = int(row["feature"])
        if feat_id < len(data["pct_width_when_present"]):
            widths.append(data["pct_width_when_present"][feat_id])
    return {
        "mean_width_pct": float(np.mean(widths)) if widths else None,
        "median_width_pct": float(np.median(widths)) if widths else None,
        "n_features": len(widths),
    }


def per_type_locality(data, common_concepts=None, threshold=0.5):
    """Mean activation width for best features, grouped by concept type."""
    if data["top_pairings"] is None or data["pct_width_when_present"] is None:
        return {}
    tp = data["top_pairings"].copy()
    if common_concepts is not None:
        tp = tp[tp["concept"].isin(common_concepts)]
    tp["type"] = tp["concept"].str.split("_").str[0]
    widths_arr = data["pct_width_when_present"]
    result = {}
    for t, grp in tp.groupby("type"):
        feats = grp["feature"].astype(int).values
        valid_feats = feats[feats < len(widths_arr)]
        if len(valid_feats) == 0:
            continue
        w = widths_arr[valid_feats]
        result[t] = {"mean_width_pct": float(w.mean()), "n": len(valid_feats)}
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading SAE data...")
all_data = {name: load_sae_data(name, cfg) for name, cfg in SAES.items()}

# ---------------------------------------------------------------------------
# Table 1: Overall metrics (raw + intersection-corrected)
# ---------------------------------------------------------------------------
print("\n=== TABLE 1: Overall SAE metrics ===")

rows = []
for name, d in all_data.items():
    ms = d["raw_metrics"]
    row = {
        "Model": d["model"],
        "Layer": d["layer"],
        "Arch": d["arch"],
        "k": d["k"] if d["k"] else ("L1" if d["arch"] == "ReLU" else "—"),
        "Dict size": d["dict_size"],
        "Embed dim": d["dim"],
        "Fidelity (%)": f"{d['pct_loss_recovered']:.1f}" if d["pct_loss_recovered"] else "—",
        "Raw avg F1": f"{ms['avg_f1_per_concept']:.4f}" if ms else "—",
        "Raw median F1": f"{ms['median_f1_per_concept']:.4f}" if ms else "—",
        "Raw coverage": f"{ms['concept_coverage']:.4f}" if ms else "—",
        "N concepts": ms["n_total_concepts"] if ms else "—",
        "N identified": ms["n_concepts_identified"] if ms else "—",
        "Polysemantic %": f"{ms['frac_features_polysemantic']*100:.2f}" if ms else "—",
        "Rank stability": f"{ms['rank_stability_mean_spearman']:.4f}" if ms else "—",
    }
    rows.append(row)

df_overall = pd.DataFrame(rows)
print(df_overall.to_string(index=False))
df_overall.to_csv(OUT_DIR / "table1_overall_metrics.csv", index=False)

# ---------------------------------------------------------------------------
# Table 2: Intersection-corrected pairwise comparison
# ---------------------------------------------------------------------------
print("\n=== TABLE 2: Pairwise comparison on common concept set ===")

esm_large = all_data["ESM-2-650M L24"]
progen_k350 = all_data["ProGen2-large L24 k=350"]
progen_k64  = all_data["ProGen2-large L24 k=64"]
progen_k10  = all_data["ProGen2-large L24 k=10"]
esm_small   = all_data["ESM-2-8M L4"]
progen_small = all_data["ProGen2-small L7"]

# Large pair: ESM-2-650M vs ProGen2-large (k=10 as best ProGen)
inter_large, common_large = compute_intersection_metrics(esm_large, progen_k10)
# Small pair
inter_small, common_small = compute_intersection_metrics(esm_small, progen_small)
# ProGen k-sweep vs ESM (common concepts to ESM-2-650M-L24)
inter_k350, _ = compute_intersection_metrics(esm_large, progen_k350)
inter_k64,  _ = compute_intersection_metrics(esm_large, progen_k64)

print(f"\nLarge pair — {inter_large['n_common_concepts']} common concepts")
for k, v in inter_large.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

print(f"\nSmall pair — {inter_small['n_common_concepts']} common concepts")
for k, v in inter_small.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# K-sweep table
print("\n=== ProGen2-large L24 k-sweep vs ESM-2-650M L24 (common concepts) ===")
sweep_rows = []
for label, inter in [("k=350", inter_k350), ("k=64", inter_k64), ("k=10", inter_large)]:
    sweep_rows.append({
        "ProGen2 k": label,
        "N common": inter["n_common_concepts"],
        "ESM-2-650M avg F1": inter[f"ESM-2-650M L24_avg_f1"],
        "ESM-2-650M coverage": inter[f"ESM-2-650M L24_coverage"],
        "ProGen2 avg F1": inter[f"ProGen2-large L24 {label}_avg_f1"],
        "ProGen2 coverage": inter[f"ProGen2-large L24 {label}_coverage"],
    })
df_sweep = pd.DataFrame(sweep_rows)
print(df_sweep.to_string(index=False))
df_sweep.to_csv(OUT_DIR / "table2_k_sweep.csv", index=False)

# Save intersection metrics
with open(OUT_DIR / "intersection_metrics.json", "w") as f:
    json.dump({"large_pair": inter_large, "small_pair": inter_small,
               "k350_vs_esm": inter_k350, "k64_vs_esm": inter_k64,
               "k10_vs_esm": inter_large}, f, indent=2)

# ---------------------------------------------------------------------------
# Table 3: Per-type coverage on common concept set
# ---------------------------------------------------------------------------
print("\n=== TABLE 3: Per-type coverage on common concept set ===")

type_esm    = per_type_coverage(esm_large, common_large)
type_pg_k10 = per_type_coverage(progen_k10, common_large)
type_pg_k350= per_type_coverage(progen_k350, common_large)

type_rows = []
all_types = sorted(set(type_esm) | set(type_pg_k10))
for t in all_types:
    e  = type_esm.get(t, {})
    p10= type_pg_k10.get(t, {})
    p350=type_pg_k350.get(t, {})
    type_rows.append({
        "Type": t,
        "N concepts": e.get("total", p10.get("total", "—")),
        "ESM-2-650M cov": f"{e.get('fraction',0):.2%}" if e else "—",
        "ESM-2-650M avg F1": f"{e.get('avg_f1',0):.3f}" if e else "—",
        "ProGen2 k=10 cov": f"{p10.get('fraction',0):.2%}" if p10 else "—",
        "ProGen2 k=10 avg F1": f"{p10.get('avg_f1',0):.3f}" if p10 else "—",
        "ProGen2 k=350 cov": f"{p350.get('fraction',0):.2%}" if p350 else "—",
        "ProGen2 k=350 avg F1": f"{p350.get('avg_f1',0):.3f}" if p350 else "—",
    })
df_type = pd.DataFrame(type_rows)
print(df_type.to_string(index=False))
df_type.to_csv(OUT_DIR / "table3_per_type_coverage.csv", index=False)

# Small pair per-type
type_esm_s = per_type_coverage(esm_small, common_small)
type_pg_s  = per_type_coverage(progen_small, common_small)
type_rows_s = []
for t in sorted(set(type_esm_s) | set(type_pg_s)):
    e = type_esm_s.get(t, {})
    p = type_pg_s.get(t, {})
    type_rows_s.append({
        "Type": t,
        "N concepts": e.get("total", p.get("total", "—")),
        "ESM-2-8M cov": f"{e.get('fraction',0):.2%}" if e else "—",
        "ESM-2-8M avg F1": f"{e.get('avg_f1',0):.3f}" if e else "—",
        "ProGen2-small cov": f"{p.get('fraction',0):.2%}" if p else "—",
        "ProGen2-small avg F1": f"{p.get('avg_f1',0):.3f}" if p else "—",
    })
df_type_s = pd.DataFrame(type_rows_s)
print("\nSmall pair:")
print(df_type_s.to_string(index=False))
df_type_s.to_csv(OUT_DIR / "table3b_per_type_coverage_small.csv", index=False)

# ---------------------------------------------------------------------------
# Table 4: Activation locality (width) per concept type
# ---------------------------------------------------------------------------
print("\n=== TABLE 4: Activation width (% residues activated) per concept type ===")

loc_esm    = per_type_locality(esm_large, common_large)
loc_pg_k10 = per_type_locality(progen_k10, common_large)
loc_pg_k350= per_type_locality(progen_k350, common_large)

loc_rows = []
for t in sorted(set(loc_esm) | set(loc_pg_k10)):
    e   = loc_esm.get(t, {})
    p10 = loc_pg_k10.get(t, {})
    p350= loc_pg_k350.get(t, {})
    loc_rows.append({
        "Type": t,
        "ESM-2-650M width%": f"{e.get('mean_width_pct', float('nan')):.2f}" if e else "—",
        "ProGen2 k=10 width%": f"{p10.get('mean_width_pct', float('nan')):.2f}" if p10 else "—",
        "ProGen2 k=350 width%": f"{p350.get('mean_width_pct', float('nan')):.2f}" if p350 else "—",
        "ESM-2 N feats": e.get("n", "—"),
        "PG k=10 N feats": p10.get("n", "—"),
    })

df_loc = pd.DataFrame(loc_rows)
print(df_loc.to_string(index=False))
df_loc.to_csv(OUT_DIR / "table4_activation_locality.csv", index=False)

# --- Overall locality (all concept-associated features) ---
print("\n=== Overall activation locality ===")
for name, d in all_data.items():
    loc = locality_for_concept_features(d)
    if loc:
        print(f"  {name}: mean_width={loc['mean_width_pct']:.2f}%, "
              f"median={loc['median_width_pct']:.2f}%, n_feats={loc['n_features']}")

# Save locality
loc_summary = {}
for name, d in all_data.items():
    loc_summary[name] = locality_for_concept_features(d)
with open(OUT_DIR / "locality_summary.json", "w") as f:
    json.dump(loc_summary, f, indent=2)

print(f"\nAll tables saved to {OUT_DIR}/")
