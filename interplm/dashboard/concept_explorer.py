"""
Concept Explorer dashboard view.

Lets users search for SAE features by biological concept using the rank eval
pipeline. Results are loaded from cached JSON files or computed on-the-fly
when a GPU is available.
"""

import io
import json
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from interplm.analysis.concepts.compare_activations import load_concept_names
from interplm.analysis.concepts.rank_eval import (
    compute_annotation_enrichment,
    concept_to_filename,
    get_top_features_for_concept,
    load_results,
    run_rank_eval_for_concept,
    save_results,
    search_concepts,
)
from interplm.dashboard.colors import get_structure_palette_and_colormap
from interplm.dashboard.feature_activation_vis import visualize_protein_feature
from interplm.dashboard.view_structures import view_single_protein

_DEFAULT_ANNOT_DIR = "data/annotations/uniprotkb/processed"


def render_concept_explorer(cache, layer_name: str, layer_data: dict) -> None:
    """Main entry point for the Concept Explorer, called from app.py."""
    sae_dir = _coerce_path(layer_data.get("sae_dir"))
    annot_dir = _coerce_path(layer_data.get("annot_dir"))
    embed_dir = _coerce_path(layer_data.get("aa_embeds_dir"))
    sae = layer_data.get("SAE")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Concept Explorer")

        if sae_dir is None:
            sae_dir_str = st.text_input(
                "SAE directory",
                key="ce_sae_dir",
                placeholder="trained_saes/best_progen_large_24_k10",
            )
            sae_dir = Path(sae_dir_str) if sae_dir_str else None

        if annot_dir is None:
            annot_dir_str = st.text_input(
                "Annotations directory",
                key="ce_annot_dir",
                value=_DEFAULT_ANNOT_DIR,
            )
            annot_dir = Path(annot_dir_str) if annot_dir_str else None

        concept_query = st.text_input(
            "Concept query",
            key="ce_concept_query",
            placeholder='e.g. "ATP" or "Binding site_ATP"',
        )
        mode = st.selectbox(
            "Analysis mode",
            ["f1_guided", "annotation_enrichment"],
            key="ce_mode",
            help=(
                "**f1_guided**: ranks features using pre-computed F1 scores.\n\n"
                "**annotation_enrichment**: finds features that fire at annotated "
                "residues directly (GPU required if not cached)."
            ),
        )
        top_k = st.number_input(
            "Top K features", min_value=1, max_value=20, value=5, key="ce_top_k"
        )

        has_gpu = torch.cuda.is_available()
        max_examples_limit = 50 if has_gpu else 20
        n_examples = st.number_input(
            "Examples per feature",
            min_value=1,
            max_value=max_examples_limit,
            value=5,
            key="ce_n_examples",
            help=(
                "How many proteins to show per feature. "
                f"Up to {max_examples_limit} available "
                f"({'GPU detected' if has_gpu else 'no GPU — cached only'})."
            ),
        )

        can_run = bool(concept_query and sae_dir)
        run_btn = st.button(
            "Load / Run Analysis",
            disabled=not can_run,
            help="Loads cached results if available; otherwise computes on GPU.",
        )
        if not has_gpu:
            st.caption("No GPU detected — cached results only.")

        # Cached analyses as one-click loaders
        if sae_dir:
            _sidebar_cached_list(Path(sae_dir))

    # ── Main area ──────────────────────────────────────────────────────────────
    st.header("Concept Explorer")

    # One-click load from cached list
    cached_load_path = st.session_state.pop("ce_load_cached", None)
    if cached_load_path:
        loaded = load_results(Path(cached_load_path))
        if loaded:
            st.session_state["ce_results"] = loaded
            st.session_state["ce_results_key"] = cached_load_path

    # Load/compute via form button
    if run_btn and concept_query and sae_dir:
        result, err = _load_or_compute(
            sae_dir=Path(sae_dir),
            annot_dir=annot_dir,
            embed_dir=embed_dir,
            sae=sae,
            concept_query=concept_query,
            mode=mode,
            top_k=int(top_k),
            has_gpu=has_gpu,
        )
        if err:
            st.error(err)
        if result is not None:
            results_key = f"{sae_dir}|{concept_query}|{mode}"
            st.session_state["ce_results"] = result
            st.session_state["ce_results_key"] = results_key

    # Display results
    current_key = f"{sae_dir}|{concept_query}|{mode}" if (sae_dir and concept_query) else ""
    stored_key = st.session_state.get("ce_results_key", "")
    # Accept either a form-based key or a file-path key (from one-click cached load)
    has_matching_results = (
        "ce_results" in st.session_state
        and stored_key
        and (stored_key == current_key or (cached_load_path is None and stored_key.endswith(".json")))
    )

    if has_matching_results:
        _render_results(st.session_state["ce_results"], layer_name, cache, int(n_examples))
    else:
        if sae_dir:
            _show_cached_analyses_main(Path(sae_dir))
        if not run_btn:
            st.info(
                "Select a cached analysis above or enter a concept query in the sidebar "
                "and click **Load / Run Analysis**."
            )


# ── Load or compute ───────────────────────────────────────────────────────────

def _load_or_compute(
    sae_dir: Path,
    annot_dir: Optional[Path],
    embed_dir: Optional[Path],
    sae,
    concept_query: str,
    mode: str,
    top_k: int,
    has_gpu: bool,
    split: str = "test",
):
    """Return (results_dict, error_str). Exactly one of them is None."""
    if annot_dir is None:
        return None, "Annotations directory not set. Add it in the sidebar."

    concept_col_file = annot_dir / split / "aa_concepts_columns.txt"
    if not concept_col_file.exists():
        return None, f"Concept list not found: {concept_col_file}"

    concept_names = load_concept_names(concept_col_file)
    matches = search_concepts(concept_query, concept_names)

    if len(matches) == 0:
        types = sorted({n.split("_")[0] for n in concept_names if n})
        return None, (
            f"No concepts match '{concept_query}'. "
            f"Available types: {', '.join(types[:20])}"
        )
    if len(matches) > 1:
        lines = "\n".join(f"• [{i}] {name}" for i, name in matches[:20])
        return None, f"Multiple matches — be more specific:\n{lines}"

    filt_idx, concept_name = matches[0]
    out_path = sae_dir / "rank_eval" / concept_to_filename(concept_name, mode)

    if out_path.exists():
        return load_results(out_path), None

    if not has_gpu:
        return None, (
            f"No cached results for '{concept_name}' [{mode}] and no GPU available."
        )

    meta_path = annot_dir / split / "metadata.json"
    if not meta_path.exists():
        return None, f"metadata.json not found: {meta_path}"

    meta = json.loads(meta_path.read_text())
    keep_idx: List[int] = meta["indices_of_concepts_to_keep"]
    raw_concept_col = keep_idx[filt_idx]

    shards = sorted(
        int(p.name.split("_")[1])
        for p in annot_dir.iterdir()
        if p.is_dir() and p.name.startswith("shard_")
    )
    device = "cuda:0"

    if mode == "f1_guided":
        f1_csv = sae_dir / f"results_{split}_counts" / "concept_f1_scores.csv"
        if not f1_csv.exists():
            return None, f"F1 CSV not found: {f1_csv}. Run concept analysis first."
        f1_df = pd.read_csv(
            f1_csv,
            usecols=["concept", "feature", "f1_per_domain", "f1",
                     "precision", "recall", "threshold_pct", "tp", "fp"],
        )
        features_df = get_top_features_for_concept(concept_name, f1_df, top_k=top_k)
    else:
        with st.status("Running annotation enrichment…", expanded=True):
            features_df = compute_annotation_enrichment(
                raw_concept_col=raw_concept_col,
                sae=sae,
                embed_dir=embed_dir,
                annot_dir=annot_dir,
                shards=shards,
                top_k_features=top_k,
                device=device,
                chunk_size=2048,
            )

    with st.status("Running per-protein rank evaluation…", expanded=True):
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
            device=device,
            chunk_size=32768,
        )

    save_results(results, out_path)
    return results, None


# ── Results rendering ─────────────────────────────────────────────────────────

def _render_results(results: dict, layer_name: str, cache, n_examples: int = 5) -> None:
    concept = results["concept"]
    mode = results["analysis_mode"]
    n_prot = results["n_proteins_with_concept"]

    st.subheader(f"{concept}  [{mode}]")
    st.caption(
        f"{n_prot} proteins with this concept · "
        f"{len(results['shards_evaluated'])} shards evaluated"
    )

    # Summary table
    rows = []
    for feat_data in results["features"]:
        feat_id = feat_data["feature_id"]
        ds = feat_data["discovery_stats"]
        agg = feat_data["aggregate_metrics"]
        if ds["source"] == "f1_guided":
            disc = f"f1_pd={ds.get('f1_per_domain', float('nan')):.3f}"
        else:
            disc = f"enrich={ds.get('enrichment_ratio', float('nan')):.1f}x"
        rows.append({
            "Feature": f"f/{feat_id}",
            "Discovery": disc,
            "Hit@1": f"{agg.get('hit@1', float('nan')):.3f}",
            "MRR": f"{agg.get('mrr', float('nan')):.3f}",
            "AUPRC mean": f"{agg.get('auprc_mean', float('nan')):.3f}",
            "baseline": f"{agg.get('auprc_baseline', float('nan')):.4f}",
            "N proteins": agg.get("n_proteins", 0),
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("---")

    colormap_fn, _ = get_structure_palette_and_colormap((0, 0.4, 0.85))

    for feat_idx, feat_data in enumerate(results["features"]):
        try:
            _render_feature_expander(feat_data, feat_idx, layer_name, colormap_fn, cache, concept, n_examples)
        except Exception as e:
            st.error(f"Error rendering feature {feat_data.get('feature_id', '?')}: {e}")


def _render_feature_expander(feat_data, feat_idx, layer_name, colormap_fn, cache, concept, n_examples: int = 5):
    feat_id = feat_data["feature_id"]
    ds = feat_data["discovery_stats"]
    agg = feat_data["aggregate_metrics"]

    if ds["source"] == "f1_guided":
        header = (
            f"Feature {feat_id} — "
            f"f1_per_domain={ds.get('f1_per_domain', 0):.4f}  "
            f"hit@1={agg.get('hit@1', 0):.3f}  "
            f"MRR={agg.get('mrr', 0):.3f}  "
            f"AUPRC={agg.get('auprc_mean', 0):.3f}"
        )
    else:
        header = (
            f"Feature {feat_id} — "
            f"enrichment={ds.get('enrichment_ratio', 0):.1f}x  "
            f"hit@1={agg.get('hit@1', 0):.3f}  "
            f"MRR={agg.get('mrr', 0):.3f}  "
            f"AUPRC={agg.get('auprc_mean', 0):.3f}"
        )

    with st.expander(header, expanded=(feat_idx == 0)):
        if st.button(
            f"Open f/{feat_id} in Feature Explorer",
            key=f"goto_{feat_idx}_{feat_id}",
        ):
            st.session_state["dashboard_mode"] = "Feature Explorer"
            st.session_state[f"feature_id_{layer_name}"] = feat_id
            st.rerun()

        pos_examples = feat_data.get("positive_examples", [])[:n_examples]
        hi_examples = feat_data.get("high_activation_examples", [])[:n_examples]

        # Gate: only render examples when the user has loaded them
        # (auto-load for the first feature since it's expanded by default)
        load_key = f"load_examples_{feat_idx}"
        examples_loaded = st.session_state.get(load_key, feat_idx == 0)

        if not examples_loaded:
            if st.button(
                f"Load {len(pos_examples) + len(hi_examples)} protein examples",
                key=f"load_btn_{feat_idx}",
            ):
                st.session_state[load_key] = True
                st.rerun()
            return

        section_choice = st.radio(
            "Show examples",
            options=["Best Localization", "Highest Activation"],
            horizontal=True,
            key=f"section_{feat_idx}",
            label_visibility="collapsed",
        )

        if section_choice == "Best Localization":
            st.caption(
                f"Annotated proteins ranked by AUPRC — how precisely the feature "
                f"activates at the binding site residues · {len(pos_examples)} shown"
            )
            for ex_idx, ex in enumerate(pos_examples):
                _render_example_protein(ex, feat_id, feat_idx, "pos", ex_idx, colormap_fn, cache)
        else:
            st.caption(
                f"Proteins with the strongest overall activation — includes unannotated "
                f"proteins that may have missing annotations · {len(hi_examples)} shown"
            )
            for ex_idx, ex in enumerate(hi_examples):
                _render_example_protein(ex, feat_id, feat_idx, "hi", ex_idx, colormap_fn, cache)


def _render_example_protein(
    example: dict,
    feat_id: int,
    feat_idx: int,
    section: str,
    ex_idx: int,
    colormap_fn,
    cache,
) -> None:
    pid = example["protein_id"]
    L = example["L"]
    feature_activations = np.array(example["feature_activations"], dtype=np.float32)
    annotation_indices = example.get("positive_residue_indices", [])
    # not used, for some this was not compputed properly but we can just use annotation indices being non-empty
    # is_annotated = example.get("is_annotated", False)
    max_act = example.get("max_activation", float(feature_activations.max()))
    n_ann = example.get("n_annotated", len(annotation_indices))
    acts_at_sites = example.get("activations_at_annotation_sites", [])
    n_ovlp = sum(1 for v in acts_at_sites if v > 0.05) if acts_at_sites else 0

    ann_tag = "**[ANN]**" if annotation_indices else ""
    st.markdown(
        f"[{pid}](https://www.uniprot.org/uniprotkb/{pid}) {ann_tag} — "
        f"L={L}  max_act={max_act:.4f}  n_ann={n_ann}  n_ovlp(>0.05)={n_ovlp}"
    )

    sequence = _get_sequence(pid, L, cache)

    max_val = float(feature_activations.max())
    normalized = feature_activations / max_val if max_val > 1e-8 else feature_activations

    chart_key = f"chart_{feat_idx}_{section}_{ex_idx}_{pid}"
    struct_key = f"struct_{feat_idx}_{section}_{ex_idx}_{pid}"

    col_chart, col_struct = st.columns([4, 5])

    with col_chart:
        if st.session_state.get(chart_key, False):
            fig = visualize_protein_feature(
                feature_acts=feature_activations,
                sequence=sequence,
                metadata=None,
                annotation_indices=annotation_indices if annotation_indices else None,
            )
            st.plotly_chart(fig, width='stretch')
        else:
            img_bytes = _static_activation_image(
                feature_activations,
                tuple(annotation_indices) if annotation_indices else None,
            )
            st.image(img_bytes, width='stretch')
            if st.button("Show interactive plot", key=f"btn_{chart_key}"):
                st.session_state[chart_key] = True
                st.rerun()

    with col_struct:
        if st.session_state.get(struct_key, False):
            try:
                structure_html = view_single_protein(
                    uniprot_id=pid,
                    values_to_color=normalized,
                    colormap_fn=colormap_fn,
                    residues_to_highlight=None,
                    pymol_params={"width": 500, "height": 350},
                )
                st.components.v1.html(structure_html, height=350)
            except Exception:
                st.caption("Structure not available.")
        else:
            if st.button("Load 3D structure", key=f"btn_{struct_key}"):
                st.session_state[struct_key] = True
                st.rerun()

    st.divider()


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(max_entries=512)
def _static_activation_image(
    feature_activations: np.ndarray,
    annotation_indices: Optional[tuple] = None,
) -> bytes:
    """Render a lightweight matplotlib PNG of the activation profile.

    Returns bytes (not BytesIO) so Streamlit's media cache stays valid across reruns.
    annotation_indices must be a tuple (not list) to be hashable for st.cache_data.
    """
    L = len(feature_activations)
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(12, 7.0))

    # Green background spans at annotated positions (drawn first, behind bars)
    if annotation_indices:
        for idx in annotation_indices:
            if 0 <= idx < L:
                ax.axvspan(idx - 0.5, idx + 0.5, color="#2CA02C", alpha=0.25, zorder=0, linewidth=0)

    ax.bar(x, feature_activations, color="#4C9BE8", width=1.0, linewidth=0, zorder=1)
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Residue", fontsize=8)
    ax.set_ylabel("Activation", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def _coerce_path(p) -> Optional[Path]:
    if p is None:
        return None
    return Path(p)


def _sidebar_cached_list(sae_dir: Path) -> None:
    """Show available cached analyses as one-click buttons in the sidebar."""
    rank_eval_dir = sae_dir / "rank_eval"
    if not rank_eval_dir.exists():
        return
    json_files = sorted(rank_eval_dir.glob("*.json"))
    if not json_files:
        return
    st.markdown(f"**Cached analyses ({len(json_files)})**")
    for f in json_files:
        if st.button(f.stem, key=f"sidebar_cached_{f.stem}", width="stretch"):
            st.session_state["ce_load_cached"] = str(f)


def _show_cached_analyses_main(sae_dir: Path) -> None:
    """Show available cached analyses as buttons in the main area."""
    rank_eval_dir = sae_dir / "rank_eval"
    if not rank_eval_dir.exists():
        return
    json_files = sorted(rank_eval_dir.glob("*.json"))
    if not json_files:
        return
    st.markdown(f"**{len(json_files)} cached analyses available — click to load:**")
    cols = st.columns(min(len(json_files), 3))
    for i, f in enumerate(json_files):
        with cols[i % 3]:
            if st.button(f.stem, key=f"main_cached_{f.stem}", width="stretch"):
                st.session_state["ce_load_cached"] = str(f)
                st.rerun()


def _get_sequence(pid: str, L: int, cache) -> str:
    """Look up protein sequence from dashboard metadata; fall back to placeholders."""
    try:
        metadata = cache.protein_metadata
        seq_col = getattr(metadata, "sequence_col", "sequence")
        seq = metadata.data.loc[pid.upper(), seq_col]
        if isinstance(seq, str) and len(seq) == L:
            return seq
    except Exception:
        pass
    return "?" * L
