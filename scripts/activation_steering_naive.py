#!/usr/bin/env python
"""
Experimental activation steering via SAE feature intervention.

Intercepts a PLM layer's hidden state during generation, encodes it with a
trained SAE, modifies a single feature's activation, then decodes back into
hidden-state space. Two steering methods are supported:

  direct     — h_steered = decode(modified_acts)
  with_error — h_steered = decode(modified_acts) + (h - decode(original_acts))
               (adds back the SAE residual to preserve unmodelled signal)

Clamp values are normalised by default: 1.0 = max observed activation for
that feature during training (stored in ae_normalized.pt). Pass
--use_raw_clamp_values to supply literal activation-space values instead.

Usage:
    # Default: clamp feature 42 to its max observed activation, 10 sequences
    python scripts/activation_steering_naive.py \
        --sae_dir trained_saes/best_progen_large_24 \
        --feature_id 42 \
        --output_fasta out/steered.fasta

    # Ablate feature 42
    python scripts/activation_steering_naive.py \
        --sae_dir trained_saes/best_progen_large_24 \
        --feature_id 42 --mode ablate \
        --output_fasta out/ablated.fasta

    # Half max, with-error reconstruction, seeded prefix
    python scripts/activation_steering_naive.py \
        --sae_dir trained_saes/best_progen_large_24 \
        --feature_id 42 --clamp_value 0.5 \
        --steering_method with_error --prefix MKTAY \
        --output_fasta out/steered_half.fasta
"""

from pathlib import Path
from typing import Optional

import json
import yaml
import torch
from tqdm import tqdm
import torch.nn.functional as F
from tap import tapify

from interplm.embedders import get_embedder
from interplm.sae.inference import load_sae
from interplm.utils import get_device


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _infer_embedder_type(model_name: str) -> str:
    name = model_name.lower()
    if "progen" in name:
        return "progen2"
    if "esm" in name:
        return "esm"
    raise ValueError(
        f"Cannot infer embedder type from model name {model_name!r}. "
        "Expected 'progen2' or 'esm' in the name."
    )


def load_config_info(sae_dir: Path) -> tuple[str, int, str]:
    """Return (model_name, layer_idx, embedder_type) from sae_dir/config.yaml.

    Uses yaml.safe_load on the raw dict so we don't need to instantiate the
    full dataclass hierarchy (eval_cfg can be ESMFidelityConfig or
    ProGenFidelityConfig with extra fields not present in EvaluationConfig).
    """
    cfg_path = sae_dir / "config.yaml"
    with open(cfg_path) as f:
        data = yaml.safe_load(f)
    eval_cfg = (data or {}).get("eval_cfg") or {}
    model_name = eval_cfg.get("model_name")
    layer_idx = eval_cfg.get("layer_idx")
    if not model_name or layer_idx is None:
        raise ValueError(
            f"config.yaml in {sae_dir} is missing eval_cfg.model_name or "
            "eval_cfg.layer_idx. Cannot infer which PLM and layer to hook."
        )
    embedder_type = _infer_embedder_type(model_name)
    return model_name, layer_idx, embedder_type


# ---------------------------------------------------------------------------
# Layer access
# ---------------------------------------------------------------------------

def get_layer_module(model: torch.nn.Module, embedder_type: str, layer_idx: int) -> torch.nn.Module:
    if embedder_type == "progen2":
        return model.transformer.h[layer_idx]
    raise NotImplementedError(
        f"Activation steering during generation is not yet supported for "
        f"embedder_type={embedder_type!r}. ESM-2 is a masked LM and does not "
        "support autoregressive generation."
    )


# ---------------------------------------------------------------------------
# Steering hooks
# ---------------------------------------------------------------------------

def make_steering_hook(
    sae: torch.nn.Module,
    feature_id: int,
    raw_steer_value: float,
    mode: str,
    use_error: bool,
) -> callable:
    """
    Returns a forward hook that intercepts a layer's hidden state and steers
    feature `feature_id` to `raw_steer_value` (in raw SAE activation space).
    """
    def hook_fn(module, input, output):
        h = output[0]                              # [batch, seq_len, d_model]
        orig_shape = h.shape
        flat_h = h.reshape(-1, orig_shape[-1])     # [N, d_model]

        acts = sae.encode(flat_h)                  # [N, n_features], raw values

        if use_error:
            error = flat_h - sae.decode(acts.clone())  # SAE residual

        if mode == "clamp":
            acts[:, feature_id] = raw_steer_value
        elif mode == "add":
            acts[:, feature_id] = acts[:, feature_id] + raw_steer_value
        elif mode == "ablate":
            acts[:, feature_id] = 0.0
        else:
            raise ValueError(f"Unknown mode {mode!r}. Choose: clamp, add, ablate.")

        h_steered = sae.decode(acts)
        if use_error:
            h_steered = h_steered + error

        h_steered = h_steered.reshape(orig_shape).to(h.dtype)
        return (h_steered,) + output[1:]

    return hook_fn


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _get_eos_token_id(tokenizer) -> int:
    """Return the token id corresponding to ProGen2 C-terminus token '2'."""
    ids = tokenizer.encode("2", add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(
            f"Expected '2' to map to a single token, got ids={ids}. "
            "Check tokenizer compatibility."
        )
    return ids[0]


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


def _aa_probs(probs: torch.Tensor, vocab: dict) -> list[float]:
    """Return per-AA probabilities in AA_ORDER (ACDEFGHIKLMNPQRSTVWY) order."""
    return [
        probs[vocab[aa]].item() if aa in vocab else 0.0
        for aa in AA_ORDER
    ]


def _entropy_bits(probs: torch.Tensor) -> float:
    """Shannon entropy in bits (log base 2)."""
    p = probs.clamp(min=1e-12)
    return float(-(p * torch.log2(p)).sum().item())


def generate_one(
    model: torch.nn.Module,
    tokenizer,
    layer_module: torch.nn.Module,
    hook_fn: callable,
    prefix: Optional[str],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    eos_token_id: int,
    device: str,
    collect_debug: bool = False,
    vocab: Optional[dict] = None,
) -> tuple[str, list[dict]]:
    """
    Generate one sequence with the steering hook active throughout.

    Returns (sequence, debug_steps) where debug_steps is [] when collect_debug=False.

    Note: uses a simple left-to-right loop rather than model.generate()
    because the latter is incompatible with ProGen2's remote config.
    """
    seq_str = ("1" + prefix) if prefix else "1"
    ids = tokenizer(
        seq_str, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    debug_steps: list[dict] = []
    inv_vocab = {v: k for k, v in vocab.items()} if vocab else {}
    handle = layer_module.register_forward_hook(hook_fn)
    try:
        with tqdm(total=max_new_tokens, desc="tokens", unit="tok", leave=False) as pbar:
            for step in range(max_new_tokens):
                # Steered logits — hook is active
                with torch.no_grad():
                    logits_steered = model(input_ids=ids).logits[0, -1]   # [vocab_size]

                if collect_debug:
                    # Base logits — temporarily remove hook for one clean forward pass
                    handle.remove()
                    with torch.no_grad():
                        logits_base = model(input_ids=ids).logits[0, -1]
                    handle = layer_module.register_forward_hook(hook_fn)

                logits = logits_steered
                if not torch.isfinite(logits).all():
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

                if do_sample:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs, 1).unsqueeze(0)
                else:
                    next_id = logits.argmax().view(1, 1)

                if collect_debug:
                    t = temperature if temperature != 0 else 1.0
                    p_steered = F.softmax(logits_steered.float() / t, dim=-1).cpu()
                    p_base = F.softmax(logits_base.float() / t, dim=-1).cpu()
                    kl = F.kl_div(
                        F.log_softmax(logits_base.float() / t, dim=-1).cpu(),
                        p_steered,
                        reduction="sum",
                    ).item()
                    debug_steps.append({
                        "step": step,
                        "chosen_token": inv_vocab.get(next_id.item(), "?"),
                        "base": {
                            "aa_probs": _aa_probs(p_base, vocab),
                            "entropy": _entropy_bits(p_base),
                        },
                        "steered": {
                            "aa_probs": _aa_probs(p_steered, vocab),
                            "entropy": _entropy_bits(p_steered),
                        },
                        "kl_steered_from_base": round(kl, 6),
                    })

                ids = torch.cat([ids, next_id], dim=1)
                pbar.update(1)
                if next_id.item() == eos_token_id:
                    break
    finally:
        handle.remove()

    # Decode and strip terminus tokens ("1" and "2")
    raw = tokenizer.decode(ids[0], skip_special_tokens=False)
    return raw.strip().strip("12").strip(), debug_steps


# ---------------------------------------------------------------------------
# Sequence scoring (no intervention)
# ---------------------------------------------------------------------------

def score_sequence(
    model: torch.nn.Module,
    tokenizer,
    sequence: str,
    device: str,
) -> tuple[float, float]:
    """Return (neg_log_likelihood, perplexity) of sequence under the unsteered model."""
    seq_str = "1" + sequence + "2"
    ids = tokenizer(seq_str, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        loss = model(input_ids=ids, labels=ids).loss  # mean NLL per predicted token
    return loss.item(), torch.exp(loss).item()


# ---------------------------------------------------------------------------
# FASTA output
# ---------------------------------------------------------------------------

def write_fasta(sequences: list[str], headers: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for header, seq in zip(headers, sequences):
            f.write(f">{header}\n{seq}\n")


# ---------------------------------------------------------------------------
# Distribution visualization
# ---------------------------------------------------------------------------

def plot_steering_debug(
    debug_data: dict,
    output_png: Path,
    sequence_idx: int = 0,
    max_steps: Optional[int] = None,
) -> None:
    """
    Save a heatmap PNG comparing base vs steered next-token distributions.

    Columns are the 20 standard AAs in fixed ACDEFGHIKLMNPQRSTVWY order,
    colored by probability (fixed 0–1 scale). For each generation step, the
    base (B) row is drawn above the steered (S) row, with a white gap row
    separating consecutive steps. Y-axis labels carry step index, chosen
    token, and per-step KL divergence.

    Args:
        debug_data: Dict with 'meta' and 'sequences' keys — same structure as
            the JSON written by the steering script, or assembled in-memory.
        output_png: Destination path for the PNG file.
        sequence_idx: Which entry in debug_data['sequences'] to visualize.
        max_steps: Cap the number of steps shown (useful for long sequences).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError as e:
        raise ImportError("matplotlib and numpy are required for plotting.") from e

    meta     = debug_data["meta"]
    seq_data = debug_data["sequences"][sequence_idx]
    steps    = seq_data["steps"]
    if max_steps is not None:
        steps = steps[:max_steps]

    n_steps = len(steps)
    if n_steps == 0:
        raise ValueError("No steps to plot.")

    n_aa = len(AA_ORDER)

    base_probs  = np.zeros((n_steps, n_aa))
    steer_probs = np.zeros((n_steps, n_aa))
    chosen:  list[str]   = []
    kl_vals: list[float] = []

    for i, step in enumerate(steps):
        chosen.append(step["chosen_token"])
        kl_vals.append(step["kl_steered_from_base"])
        base_probs[i]  = step["base"]["aa_probs"]
        steer_probs[i] = step["steered"]["aa_probs"]

    # Matrix layout (per step): [base_row, steered_row, gap_row]
    # except no trailing gap after the last step
    # row indices: step i → base at 3i, steered at 3i+1, gap at 3i+2
    n_rows = 3 * n_steps - 1
    matrix = np.full((n_rows, n_aa), np.nan)
    for i in range(n_steps):
        matrix[3 * i]     = base_probs[i]
        matrix[3 * i + 1] = steer_probs[i]
        # gap row (3i+2) stays NaN → rendered white

    cmap_name = "YlOrRd"
    cmap_obj  = plt.get_cmap(cmap_name).copy()
    cmap_obj.set_bad(color="white")        # NaN gap rows → white
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    CELL_W, CELL_H = 0.58, 0.32
    hmap_w = n_aa * CELL_W + 2.8
    hmap_h = min(n_rows * CELL_H + 1.0, 32.0)
    kl_h   = 2.0
    fig_w  = hmap_w + 0.8
    fig_h  = hmap_h + kl_h + 1.5

    fig = plt.figure(figsize=(fig_w, fig_h), layout="constrained")
    gs  = fig.add_gridspec(
        2, 2,
        height_ratios=[hmap_h, kl_h],
        width_ratios=[hmap_w, 0.45],
    )
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_kl   = fig.add_subplot(gs[1, 0])

    im = ax_heat.imshow(matrix, aspect="auto", cmap=cmap_obj,
                        vmin=0.0, vmax=1.0, interpolation="nearest")

    # Annotate data cells and highlight the chosen AA
    for i in range(n_steps):
        chosen_aa = chosen[i]
        for row_off, prob_mat in ((0, base_probs), (1, steer_probs)):
            row = 3 * i + row_off
            for j, aa in enumerate(AA_ORDER):
                prob = prob_mat[i, j]
                rgba = cmap_obj(norm(prob))
                brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "white" if brightness < 0.55 else "black"
                is_chosen = aa == chosen_aa
                # Shift AA letter up in highlighted cells to make room for prob text
                aa_y = row - 0.18 if is_chosen else row
                ax_heat.text(j, aa_y, aa, ha="center", va="center", fontsize=7,
                             color=txt_color, fontfamily="monospace")
                if is_chosen:
                    ax_heat.text(j, row + 0.26, f"{prob:.3f}", ha="center", va="center",
                                 fontsize=4.5, fontstyle="italic", color=txt_color)
                    ax_heat.add_patch(mpatches.FancyBboxPatch(
                        (j - 0.48, row - 0.48), 0.96, 0.96,
                        boxstyle="square,pad=0",
                        linewidth=2.0, edgecolor="deepskyblue", facecolor="none",
                        clip_on=False,
                    ))

    # Y-ticks only on data rows; labels carry step info and KL
    data_rows = []
    y_labels  = []
    for i in range(n_steps):
        data_rows += [3 * i, 3 * i + 1]
        y_labels  += [f"{i}:{chosen[i]}  B", f"KL={kl_vals[i]:.3f}  S"]

    ax_heat.set_yticks(data_rows)
    ax_heat.set_yticklabels(y_labels, fontsize=7, fontfamily="monospace")
    ax_heat.set_xticks(range(n_aa))
    ax_heat.set_xticklabels(list(AA_ORDER), fontsize=8, fontfamily="monospace")
    ax_heat.xaxis.tick_top()
    ax_heat.xaxis.set_label_position("top")
    # Extend xlim slightly so edge patches aren't clipped
    ax_heat.set_xlim(-0.6, n_aa - 0.4)

    fig.colorbar(im, cax=ax_cbar, label="Probability (0–1)")

    # KL divergence bar chart
    kl_arr = np.array(kl_vals)
    ax_kl.bar(range(n_steps), kl_arr, color="steelblue", alpha=0.8, width=0.85)
    ax_kl.axhline(kl_arr.mean(), color="tomato", linestyle="--", linewidth=1.5,
                  label=f"mean = {kl_arr.mean():.3f}")
    ax_kl.set_xlim(-0.5, n_steps - 0.5)
    ax_kl.set_xlabel("Generation step", fontsize=9)
    ax_kl.set_ylabel("KL (steered ∥ base)", fontsize=9)
    ax_kl.set_title("Per-step KL divergence", fontsize=9)
    ax_kl.legend(fontsize=8)

    seq_str = seq_data["sequence"]
    fig.suptitle(
        f"Feature {meta['feature_id']} | {meta['mode']} | {meta['steering_method']} | "
        f"clamp={meta['clamp_value']:.3f}   [seq {sequence_idx}]\n"
        f"{seq_str[:80]}{'...' if len(seq_str) > 80 else ''}  "
        f"nll={seq_data['nll']:.1f}  ppl={seq_data['ppl']:.1f}\n"
        "Base (B) / Steered (S) — 20 standard AAs in fixed order",
        fontsize=9,
    )

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    sae_dir: Path,
    feature_id: int,
    output_fasta: Path,
    clamp_value: float = 1.0,
    use_raw_clamp_values: bool = False,
    mode: str = "clamp",
    steering_method: str = "direct",
    prefix: Optional[str] = None,
    n_sequences: int = 10,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    do_sample: bool = True,
    use_normalized_sae: bool = True,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    debug_output: Optional[Path] = None,
    plot_output: Optional[Path] = None,
    plot_max_steps: Optional[int] = None,
) -> None:
    """
    Generate protein sequences with SAE-guided activation steering.

    Args:
        sae_dir: Directory containing a trained SAE (config.yaml + ae*.pt).
        feature_id: Index of the SAE feature to steer.
        output_fasta: Path to write the generated sequences in FASTA format.
        clamp_value: Steering magnitude. By default normalised: 1.0 = max
            observed activation for this feature; 0.0 = ablate; >1.0 = amplify
            beyond training max. Use --use_raw_clamp_values for literal units.
        use_raw_clamp_values: If True, clamp_value is used directly in raw
            SAE activation units instead of being scaled by the per-feature max.
        mode: How to modify the feature: 'clamp' (set to value), 'add' (shift
            from current), or 'ablate' (set to 0).
        steering_method: 'direct' replaces the hidden state with the SAE
            reconstruction. 'with_error' adds back the SAE residual.
        prefix: Optional amino-acid prefix string. Generation continues from
            this prefix. If None, generation starts de novo from the BOS token.
        n_sequences: Number of sequences to generate.
        max_new_tokens: Maximum tokens to generate per sequence.
        temperature: Sampling temperature (ignored when do_sample=False).
        do_sample: Sample from the distribution; if False, use greedy decoding.
        use_normalized_sae: Load ae_normalized.pt (has activation_rescale_factor
            populated). Falls back to ae.pt when False.
        seed: Optional random seed for reproducibility.
        device: Device override ('cuda', 'cpu', etc.). Auto-detected if None.
        debug_output: If provided, write a JSON file comparing base vs steered
            next-token distributions (all 20 standard AAs) at every generation step.
        plot_output: If provided, save a heatmap PNG of base vs steered distributions.
            For multiple sequences, index is appended: plot_output_seq0.png, etc.
        plot_max_steps: Cap the number of steps shown in the plot (default: all).
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = device or get_device()
    sae_dir = Path(sae_dir)

    # --- Load SAE and resolve raw steer value ---
    sae_filename = "ae_normalized.pt" if use_normalized_sae else "ae.pt"
    print(f"Loading SAE from {sae_dir / sae_filename} ...")
    sae = load_sae(sae_dir, model_name=sae_filename, device=device)
    sae.eval()

    rescale = sae.activation_rescale_factor[feature_id].item()
    if not use_normalized_sae and not use_raw_clamp_values and abs(rescale - 1.0) < 1e-6:
        print(
            "WARNING: ae.pt has activation_rescale_factor≈1 (not populated). "
            "Normalised clamp_value will not scale to the true per-feature max. "
            "Either use ae_normalized.pt (default) or pass --use_raw_clamp_values."
        )

    if mode == "ablate":
        raw_steer_value = 0.0
        display_value = "0.0 (ablate)"
    elif use_raw_clamp_values:
        raw_steer_value = clamp_value
        display_value = f"{clamp_value:.4f} (raw)"
    else:
        raw_steer_value = clamp_value * rescale
        display_value = f"{clamp_value:.4f} (normalised) = {raw_steer_value:.4f} raw"

    print(f"Feature {feature_id}: max observed activation = {rescale:.4f}")
    print(f"Steering value: {display_value}  mode={mode}  method={steering_method}")

    # --- Infer PLM from config and load ---
    model_name, layer_idx, embedder_type = load_config_info(sae_dir)
    print(f"Loading {embedder_type} model {model_name!r}, layer {layer_idx} ...")
    embedder = get_embedder(embedder_type, model_name=model_name, device=device)
    model = embedder.model
    tokenizer = embedder.tokenizer

    layer_module = get_layer_module(model, embedder_type, layer_idx)
    eos_token_id = _get_eos_token_id(tokenizer)
    use_error = steering_method == "with_error"

    hook_fn = make_steering_hook(
        sae=sae,
        feature_id=feature_id,
        raw_steer_value=raw_steer_value,
        mode=mode,
        use_error=use_error,
    )

    collect_debug = debug_output is not None or plot_output is not None
    vocab = tokenizer.get_vocab() if collect_debug else None

    # --- Generate ---
    sequences: list[str] = []
    scores: list[tuple[float, float]] = []
    all_debug_steps: list[list[dict]] = []
    print(f"\nGenerating {n_sequences} sequence(s) ...")
    for i in range(n_sequences):
        seq, debug_steps = generate_one(
            model=model,
            tokenizer=tokenizer,
            layer_module=layer_module,
            hook_fn=hook_fn,
            prefix=prefix,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
            device=device,
            collect_debug=collect_debug,
            vocab=vocab,
        )
        nll, ppl = score_sequence(model, tokenizer, seq, device)
        sequences.append(seq)
        scores.append((nll, ppl))
        all_debug_steps.append(debug_steps)
        print(f"  [{i+1}/{n_sequences}] len={len(seq)}  nll={nll:.2f}  ppl={ppl:.2f}  {seq[:60]}{'...' if len(seq) > 60 else ''}")

    # --- Write FASTA ---
    clamp_tag = f"{clamp_value:.4f}{'_raw' if use_raw_clamp_values else '_norm'}"
    headers = [
        f"steered_{i}|feature_{feature_id}|{mode}|{steering_method}|value_{clamp_tag}|nll={nll:.2f}|ppl={ppl:.2f}"
        for i, (nll, ppl) in enumerate(scores)
    ]
    write_fasta(sequences, headers, Path(output_fasta))
    print(f"\nWrote {n_sequences} sequence(s) to {output_fasta}")

    # --- Build debug payload (shared by JSON writer and plotter) ---
    payload = None
    if collect_debug:
        payload = {
            "meta": {
                "sae_dir": str(sae_dir),
                "feature_id": feature_id,
                "mode": mode,
                "steering_method": steering_method,
                "clamp_value": clamp_value,
                "raw_steer_value": raw_steer_value,
            },
            "sequences": [
                {
                    "index": i,
                    "sequence": seq,
                    "nll": round(nll, 4),
                    "ppl": round(ppl, 4),
                    "steps": steps,
                }
                for i, (seq, (nll, ppl), steps) in enumerate(
                    zip(sequences, scores, all_debug_steps)
                )
            ],
        }

    if debug_output and payload:
        debug_output = Path(debug_output)
        debug_output.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote debug distributions to {debug_output}")

    if plot_output and payload:
        plot_output = Path(plot_output)
        for i in range(n_sequences):
            out = (
                plot_output.with_name(f"{plot_output.stem}_seq{i}{plot_output.suffix}")
                if n_sequences > 1
                else plot_output
            )
            plot_steering_debug(payload, out, sequence_idx=i, max_steps=plot_max_steps)


if __name__ == "__main__":
    tapify(main)
