"""
Protein Zoom dashboard page.

Given a single protein and a handful of SAE feature IDs, runs one SAE forward
pass and compares the features' per-residue activation profiles (bar chart,
overlaid or stacked). Optionally shades residues annotated for a concept.

Reachable directly via the sidebar, or via a "Zoom in" link from Concept
Explorer, which pre-fills the protein ID, feature IDs, and concept context
through session_state.
"""

import json
from pathlib import Path
from typing import List, Optional

import streamlit as st

from interplm.analysis.concepts.compare_activations import load_concept_names
from interplm.analysis.concepts.rank_eval import search_concepts
from interplm.analysis.per_protein_analysis import (
    compute_full_activations,
    find_protein_in_shards,
    get_annotation_indices_for_concept,
)
from interplm.dashboard.feature_activation_vis import (
    MAX_ZOOM_FEATURES,
    visualize_multi_feature_activations,
)

_DEFAULT_ANNOT_DIR = "data/annotations/uniprotkb/processed"


def render_protein_zoom(cache, layer_name: str, layer_data: dict) -> None:
    """Main entry point for the Protein Zoom page, called from app.py."""
    sae_dir = _coerce_path(layer_data.get("sae_dir"))
    annot_dir = _coerce_path(layer_data.get("annot_dir"))
    embed_dir = _coerce_path(layer_data.get("aa_embeds_dir"))
    sae = layer_data.get("SAE")

    with st.sidebar:
        st.markdown("## Protein Zoom")

        if sae_dir is None:
            sae_dir_str = st.text_input(
                "SAE directory", key="pz_sae_dir",
                placeholder="trained_saes/best_progen_large_24_k10",
            )
            sae_dir = Path(sae_dir_str) if sae_dir_str else None

        if annot_dir is None:
            annot_dir_str = st.text_input(
                "Annotations directory", key="pz_annot_dir", value=_DEFAULT_ANNOT_DIR,
            )
            annot_dir = Path(annot_dir_str) if annot_dir_str else None

        st.session_state.setdefault("pz_protein_id", "")
        protein_id = st.text_input(
            "Protein ID", key="pz_protein_id", placeholder="e.g. P0DUT8",
        )

        st.session_state.setdefault("pz_feature_ids_text", "")
        feat_ids_text = st.text_input(
            "Feature IDs (comma-separated)",
            key="pz_feature_ids_text",
            placeholder="e.g. 7196, 7514, 5243",
            help=f"Up to {MAX_ZOOM_FEATURES} at a time (categorical color limit).",
        )

        st.session_state.setdefault("pz_concept_query", "")
        concept_query = st.text_input(
            "Shade concept (optional)",
            key="pz_concept_query",
            placeholder='e.g. "Binding site_ATP"',
            help="Shades residues annotated for this concept, if found.",
        )

        view_mode = st.radio(
            "Layout", ["Stacked", "Overlay"], key="pz_mode", horizontal=True,
        )

        can_run = bool(protein_id and feat_ids_text and sae_dir and embed_dir)
        run_btn = st.button("Analyze protein", disabled=not can_run)

    st.header("Protein Zoom")
    st.caption(
        "Run several SAE features on one protein and compare their per-residue "
        "activation profiles."
    )

    if sae is None:
        st.error(
            "No SAE loaded for this layer. Preload it via Feature Explorer / "
            "Concept Explorer first, or select a layer with a loaded SAE."
        )
        return

    feature_ids = _parse_feature_ids(feat_ids_text)
    if feat_ids_text and not feature_ids:
        st.error("Could not parse any feature IDs. Use a comma-separated list of integers.")
        return
    if len(feature_ids) > MAX_ZOOM_FEATURES:
        st.warning(
            f"Showing the first {MAX_ZOOM_FEATURES} of {len(feature_ids)} feature IDs."
        )
        feature_ids = feature_ids[:MAX_ZOOM_FEATURES]

    zoom_key = f"{protein_id.strip().upper()}|{','.join(str(f) for f in feature_ids)}"

    if run_btn:
        device = "cuda:0" if _has_gpu() else "cpu"
        with st.spinner(f"Running SAE on {protein_id} …"):
            result, err = _compute_protein_zoom(
                protein_id=protein_id.strip(),
                feature_ids=feature_ids,
                sae=sae,
                embed_dir=embed_dir,
                annot_dir=annot_dir,
                concept_query=concept_query.strip() or None,
                device=device,
            )
        if err:
            st.error(err)
            st.session_state.pop("pz_result", None)
            st.session_state.pop("pz_result_key", None)
        else:
            st.session_state["pz_result"] = result
            st.session_state["pz_result_key"] = zoom_key

    if st.session_state.get("pz_result_key") == zoom_key:
        result = st.session_state.get("pz_result")
        if result:
            _render_zoom_result(result, feature_ids, view_mode, cache)
            return

    if not can_run:
        st.info(
            "Enter a protein ID and feature IDs in the sidebar (or click "
            "**Zoom in** on an example in Concept Explorer) and click "
            "**Analyze protein**."
        )


def _compute_protein_zoom(
    protein_id: str,
    feature_ids: List[int],
    sae,
    embed_dir: Path,
    annot_dir: Optional[Path],
    concept_query: Optional[str],
    device: str,
    chunk_size: int = 2048,
    split: str = "test",
):
    """Return (result_dict, error_str). Exactly one of them is None."""
    try:
        embeddings, shard_idx, _ = find_protein_in_shards(protein_id, embed_dir, annot_dir)
    except ValueError as exc:
        return None, str(exc)

    activations = compute_full_activations(embeddings, sae, device, chunk_size)
    dict_size = activations.shape[1]
    bad_ids = [f for f in feature_ids if not (0 <= f < dict_size)]
    if bad_ids:
        return None, f"Feature ID(s) out of range [0, {dict_size}): {bad_ids}"

    L = activations.shape[0]
    acts_by_feat = {fid: activations[:, fid] for fid in feature_ids}

    annotation_indices: List[int] = []
    concept_name = None
    if annot_dir is not None and concept_query:
        annotation_indices, concept_name = _resolve_annotation_indices(
            protein_id, annot_dir, concept_query, split
        )

    return {
        "protein_id": protein_id,
        "shard_idx": shard_idx,
        "L": L,
        "activations": acts_by_feat,
        "annotation_indices": annotation_indices,
        "concept_name": concept_name,
    }, None


def _resolve_annotation_indices(
    protein_id: str, annot_dir: Path, concept_query: str, split: str,
):
    """Best-effort concept lookup; returns ([], None) on any ambiguity or failure."""
    try:
        annot_dir = Path(annot_dir)
        split_dir = annot_dir / split
        concept_names = load_concept_names(split_dir / "aa_concepts_columns.txt")
        matches = search_concepts(concept_query, concept_names)
        if len(matches) != 1:
            return [], None
        filt_idx, concept_name = matches[0]
        meta = json.loads((split_dir / "metadata.json").read_text())
        keep_idx: List[int] = meta["indices_of_concepts_to_keep"]
        raw_concept_col = keep_idx[filt_idx]
        indices = get_annotation_indices_for_concept(protein_id, annot_dir, raw_concept_col)
        return indices, concept_name
    except Exception:
        return [], None


def _render_zoom_result(result: dict, feature_ids: List[int], view_mode: str, cache) -> None:
    pid = result["protein_id"]
    L = result["L"]
    ann = result["annotation_indices"]
    concept_name = result.get("concept_name")

    if concept_name:
        ann_note = f"{len(ann)} residue(s) annotated for '{concept_name}'" if ann \
            else f"no residues annotated for '{concept_name}'"
    else:
        ann_note = "no concept shading"

    st.markdown(f"[{pid}](https://www.uniprot.org/uniprotkb/{pid}) — L={L}  ({ann_note})")

    sequence = _get_sequence(pid, L, cache)
    acts = {fid: result["activations"][fid] for fid in feature_ids if fid in result["activations"]}
    fig = visualize_multi_feature_activations(
        acts,
        annotation_indices=ann,
        mode="overlay" if view_mode == "Overlay" else "stacked",
        sequence=sequence,
    )
    st.plotly_chart(fig, width="stretch")


def _get_sequence(pid: str, L: int, cache) -> Optional[str]:
    """Look up protein sequence from dashboard metadata; None if unavailable/mismatched."""
    try:
        metadata = cache.protein_metadata
        seq_col = getattr(metadata, "sequence_col", "sequence")
        seq = metadata.data.loc[pid.upper(), seq_col]
        if isinstance(seq, str) and len(seq) == L:
            return seq
    except Exception:
        pass
    return None


def _parse_feature_ids(text: str) -> List[int]:
    ids = []
    for tok in text.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            ids.append(int(tok))
        except ValueError:
            continue
    return ids


def _has_gpu() -> bool:
    import torch
    return torch.cuda.is_available()


def _coerce_path(p) -> Optional[Path]:
    if p is None:
        return None
    return Path(p)
