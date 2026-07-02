#!/usr/bin/env python
"""
Experimental activation steering via SAE feature intervention.

Intercepts a PLM layer's hidden state during generation, encodes it with a
trained SAE, modifies one or more feature activations, then decodes back into
hidden-state space. Two steering methods are supported:

  direct     — h_steered = decode(modified_acts)
  with_error — h_steered = decode(modified_acts) + (h - decode(original_acts))
               (adds back the SAE residual to preserve unmodelled signal)

Clamp values are normalised by default: 1.0 = max observed activation for
that feature during training (stored in ae_normalized.pt). Pass
--use_raw_clamp_values to supply literal activation-space values instead.

Single-feature usage:
    python scripts/activation_steering_naive.py \
        --sae_dir trained_saes/best_progen_large_24 \
        --feature_id 42 --clamp_value 5.0

    python scripts/activation_steering_naive.py \
        --sae_dir trained_saes/best_progen_large_24 \
        --feature_id 42 --mode ablate

Multi-feature usage (--features overrides --feature_id / --clamp_value):
    python scripts/activation_steering_naive.py \
        --sae_dir trained_saes/best_progen_large_24 \
        --features 5900:5.0 42:2.0 --mode clamp
"""

from pathlib import Path
from typing import Optional

import json
import math
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
        # hidden_states[k] from output_hidden_states=True equals the output of h[k-1],
        # so to hook the same activations the SAE was trained on we must hook h[layer_idx - 1].
        if layer_idx < 1:
            raise ValueError(
                f"layer_idx must be >= 1 for progen2 (layer 0 is the raw embedding with no "
                f"transformer block to hook). Got layer_idx={layer_idx}."
            )
        n_layers = len(model.transformer.h)
        if layer_idx > n_layers:
            raise ValueError(
                f"layer_idx={layer_idx} is out of range for this model ({n_layers} blocks). "
                f"Valid range: 1–{n_layers}."
            )
        if layer_idx == n_layers:
            # Final hidden state passes through ln_f; hook that to match hidden_states[n_layers].
            return model.transformer.ln_f
        return model.transformer.h[layer_idx - 1]
    raise NotImplementedError(
        f"Activation steering during generation is not yet supported for "
        f"embedder_type={embedder_type!r}. ESM-2 is a masked LM and does not "
        "support autoregressive generation."
    )


def verify_hook_target(
    model: torch.nn.Module,
    layer_module: torch.nn.Module,
    layer_idx: int,
    tokenizer,
    device: str,
    atol: float = 1e-5,
) -> None:
    """Assert that the forward hook on layer_module captures hidden_states[layer_idx].

    Runs one forward pass with a short dummy sequence, compares the value captured
    by a hook on layer_module against outputs.hidden_states[layer_idx].
    Raises AssertionError with max |diff| if they disagree.
    """
    seq_str = "1MKTAY2"
    ids = tokenizer(seq_str, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    captured: dict = {}

    def _cap(m, inp, out):
        is_tuple = isinstance(out, tuple)
        captured["h"] = (out[0] if is_tuple else out).detach().clone()

    handle = layer_module.register_forward_hook(_cap)
    try:
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
    finally:
        handle.remove()

    hs = out.hidden_states[layer_idx]
    if not torch.allclose(captured["h"], hs, atol=atol):
        max_diff = (captured["h"] - hs).abs().max().item()
        raise AssertionError(
            f"Hook capture does not match hidden_states[{layer_idx}]! "
            f"max |diff| = {max_diff:.2e}  (atol={atol:.0e}). "
            "Check get_layer_module() for this embedder type and layer index."
        )
    print(f"[verify] Hook on layer_module == hidden_states[{layer_idx}]  (max |diff| < {atol:.0e})")


# ---------------------------------------------------------------------------
# Steering hooks
# ---------------------------------------------------------------------------

def make_steering_hook(
    sae: torch.nn.Module,
    feature_specs: list[tuple[int, float]],
    mode: str,
    use_error: bool,
    steer_from_pos: int = 0,
) -> callable:
    """
    Returns a forward hook that intercepts a layer's hidden state and steers
    one or more features. `feature_specs` is a list of (feature_id, raw_value)
    pairs; all features are modified with the same `mode`.

    `steer_from_pos` controls which token positions are steered: positions
    0..(steer_from_pos-1) are passed through unchanged, only positions
    steer_from_pos..seq_len-1 are encoded/modified/decoded. Set to the
    initial prefix length so the input prompt is never steered.
    """
    def hook_fn(module, input, output):
        # ln_f returns a bare tensor; all other blocks return a tuple
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output      # [batch, seq_len, d_model]
        batch_size, seq_len, d_model = h.shape

        # Absolute position of this chunk's first token. With KV caching, after
        # the prefix is processed the chunk is a single new token whose absolute
        # position is tracked via hook_fn.token_offset. Without caching the full
        # sequence is re-fed every step and token_offset stays 0, recovering the
        # original behaviour.
        offset = getattr(hook_fn, "token_offset", 0)
        # Local index within this chunk where steering begins.
        pos_start = steer_from_pos - offset
        if pos_start < 0:
            pos_start = 0
        if pos_start >= seq_len:
            # Entire chunk is still inside the unsteered prefix; nothing to do.
            return output

        flat_h = h[:, pos_start:, :].reshape(-1, d_model)  # [N_new, d_model]
        acts = sae.encode(flat_h)                           # [N_new, n_features]

        if use_error:
            error = flat_h - sae.decode(acts)  # SAE residual; acts unmodified here

        for feature_id, raw_steer_value in feature_specs:
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

        h_steered = h_steered.reshape(batch_size, seq_len - pos_start, d_model).to(h.dtype)
        if pos_start > 0:
            h_out = torch.cat([h[:, :pos_start, :], h_steered], dim=1)
        else:
            h_out = h_steered

        return (h_out,) + output[1:] if is_tuple else h_out

    # Absolute position of the first token in the chunk currently being fed to
    # the model. The generation loop updates this when KV caching is on.
    hook_fn.token_offset = 0
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
    layer_module: Optional[torch.nn.Module],
    hook_fn: Optional[callable],
    prefix: Optional[str],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    eos_token_id: int,
    device: str,
    collect_debug: bool = False,
    vocab: Optional[dict] = None,
    verbose: bool = False,
    use_cache: bool = True,
) -> tuple[str, list[dict]]:
    """
    Generate one sequence, optionally with a steering hook active throughout.

    Pass layer_module=None / hook_fn=None to generate from the base model
    without any SAE intervention (mode="none").

    Returns (sequence, debug_steps) where debug_steps is [] when collect_debug=False.

    With use_cache=True (default) the model's attention KV cache is reused across
    steps: the full prefix is processed once, then only the single newest token is
    fed each step (model(input_ids=x, past_key_values=past)). With use_cache=False
    the full growing sequence is re-fed every step — slower, kept as a correctness
    reference. The two paths are numerically equivalent because the steering hook
    modifies a layer's *output*, which only affects deeper layers; each token's
    hidden state at the steered layer is therefore a fixed function of the original
    tokens, so steering it once (cached) equals re-steering it every step.

    Note: uses a simple left-to-right loop rather than model.generate()
    because the latter is incompatible with ProGen2's remote config.
    """
    seq_str = ("1" + prefix) if prefix else "1"
    ids = tokenizer(
        seq_str, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    debug_steps: list[dict] = []
    inv_vocab = {v: k for k, v in vocab.items()} if vocab else {}
    steered = layer_module is not None and hook_fn is not None
    handle = layer_module.register_forward_hook(hook_fn) if steered else None

    # KV cache state. `x` is the chunk fed to the model on each step: the full
    # prefix on the first step, then a single new token. `past` holds the cached
    # attention K/V. `past_base` is a parallel cache for the unsteered base pass
    # used only when collect_debug is True (it is fed the same chosen tokens but
    # never sees the steering hook).
    past = None
    past_base = None
    x = ids
    x_base = ids
    if steered:
        hook_fn.token_offset = 0
    try:
        with tqdm(total=max_new_tokens, desc="tokens", unit="tok", leave=False, disable=not verbose) as pbar:
            for step in range(max_new_tokens):
                with torch.no_grad():
                    if use_cache:
                        out = model(input_ids=x, past_key_values=past, use_cache=True)
                        past = out.past_key_values
                        if past is None:
                            raise RuntimeError(
                                "Model did not return past_key_values; KV caching is "
                                "unsupported for this model. Re-run with use_cache=False."
                            )
                        logits_steered = out.logits[0, -1]                # [vocab_size]
                    else:
                        logits_steered = model(input_ids=ids).logits[0, -1]

                if collect_debug:
                    # Base logits — temporarily remove hook for one clean forward pass
                    if steered:
                        handle.remove()
                    with torch.no_grad():
                        if use_cache:
                            out_base = model(input_ids=x_base, past_key_values=past_base, use_cache=True)
                            past_base = out_base.past_key_values
                            logits_base = out_base.logits[0, -1]
                        else:
                            logits_base = model(input_ids=ids).logits[0, -1]
                    if steered:
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
                if use_cache:
                    # Next step feeds only the newest token; advance the hook's
                    # absolute-position offset so it steers this generated token.
                    x = next_id
                    x_base = next_id
                    if steered:
                        hook_fn.token_offset = ids.shape[1] - 1
                pbar.update(1)
                if next_id.item() == eos_token_id:
                    break
    finally:
        if handle is not None:
            handle.remove()

    # Decode and strip terminus tokens ("1" and "2")
    raw = tokenizer.decode(ids[0], skip_special_tokens=False)
    return raw.strip().strip("12").strip(), debug_steps


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------

def generate_batch(
    model: torch.nn.Module,
    tokenizer,
    layer_module: Optional[torch.nn.Module],
    hook_fn: Optional[callable],
    prefix: Optional[str],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    eos_token_id: int,
    device: str,
    collect_debug: bool = False,
    vocab: Optional[dict] = None,
    verbose: bool = False,
    use_cache: bool = True,
) -> tuple[list[str], list[list[dict]]]:
    """
    Generate `batch_size` sequences in parallel, optionally with a steering hook.

    All sequences start from the same prefix (or BOS if None). Sequences that
    reach EOS early are frozen while the rest of the batch continues. Returns
    (sequences, debug_steps_per_seq); debug_steps_per_seq[i] is empty when
    collect_debug=False.

    use_cache mirrors generate_one: True (default) reuses the attention KV cache
    and feeds only the newest token each step; False re-feeds the full sequence
    every step as a correctness reference. The two are numerically equivalent.
    """
    seq_str = ("1" + prefix) if prefix else "1"
    single_ids = tokenizer(seq_str, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    ids = single_ids.expand(batch_size, -1).clone()  # [B, prefix_len]

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    inv_vocab = {v: k for k, v in vocab.items()} if vocab else {}

    steered = layer_module is not None and hook_fn is not None
    handle = layer_module.register_forward_hook(hook_fn) if steered else None

    batch_debug: list[list[dict]] = [[] for _ in range(batch_size)]

    # KV cache state (see generate_one for details).
    past = None
    past_base = None
    x = ids
    x_base = ids
    if steered:
        hook_fn.token_offset = 0

    try:
        with tqdm(total=max_new_tokens, desc="tokens", unit="tok", leave=False, disable=not verbose) as pbar:
            for step in range(max_new_tokens):
                if finished.all():
                    break

                with torch.no_grad():
                    if use_cache:
                        out = model(input_ids=x, past_key_values=past, use_cache=True)
                        past = out.past_key_values
                        if past is None:
                            raise RuntimeError(
                                "Model did not return past_key_values; KV caching is "
                                "unsupported for this model. Re-run with use_cache=False."
                            )
                        logits_steered = out.logits[:, -1]               # [B, vocab]
                    else:
                        logits_steered = model(input_ids=ids).logits[:, -1]

                if collect_debug:
                    if steered:
                        handle.remove()
                    with torch.no_grad():
                        if use_cache:
                            out_base = model(input_ids=x_base, past_key_values=past_base, use_cache=True)
                            past_base = out_base.past_key_values
                            logits_base = out_base.logits[:, -1]
                        else:
                            logits_base = model(input_ids=ids).logits[:, -1]
                    if steered:
                        handle = layer_module.register_forward_hook(hook_fn)

                logits = logits_steered
                if not torch.isfinite(logits).all():
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

                if do_sample:
                    probs = F.softmax(logits / temperature, dim=-1)  # [B, vocab]
                    next_ids = torch.multinomial(probs, 1)            # [B, 1]
                else:
                    next_ids = logits.argmax(dim=-1, keepdim=True)    # [B, 1]

                # Freeze finished sequences so their tokens don't corrupt decoding
                next_ids[finished] = eos_token_id

                if collect_debug:
                    t = temperature if temperature != 0 else 1.0
                    p_s = F.softmax(logits_steered.float() / t, dim=-1).cpu()
                    p_b = F.softmax(logits_base.float() / t, dim=-1).cpu()
                    for b_idx in range(batch_size):
                        if finished[b_idx]:
                            continue
                        kl = F.kl_div(
                            F.log_softmax(logits_base[b_idx].float() / t, dim=-1).cpu(),
                            p_s[b_idx],
                            reduction="sum",
                        ).item()
                        batch_debug[b_idx].append({
                            "step": step,
                            "chosen_token": inv_vocab.get(next_ids[b_idx].item(), "?"),
                            "base": {
                                "aa_probs": _aa_probs(p_b[b_idx], vocab),
                                "entropy": _entropy_bits(p_b[b_idx]),
                            },
                            "steered": {
                                "aa_probs": _aa_probs(p_s[b_idx], vocab),
                                "entropy": _entropy_bits(p_s[b_idx]),
                            },
                            "kl_steered_from_base": round(kl, 6),
                        })

                ids = torch.cat([ids, next_ids], dim=1)
                if use_cache:
                    x = next_ids
                    x_base = next_ids
                    if steered:
                        hook_fn.token_offset = ids.shape[1] - 1
                finished |= (next_ids.squeeze(1) == eos_token_id)
                pbar.update(1)
    finally:
        if handle is not None:
            handle.remove()

    sequences = []
    for b_idx in range(batch_size):
        raw = tokenizer.decode(ids[b_idx], skip_special_tokens=False)
        sequences.append(raw.strip().strip("12").strip())

    return sequences, batch_debug


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
    output_fasta: Optional[Path] = None,
    feature_id: Optional[int] = None,
    clamp_value: float = 1.0,
    features: Optional[list[str]] = None,
    use_raw_clamp_values: bool = False,
    mode: str = "clamp",
    steering_method: str = "with_error",
    prefix: Optional[str] = None,
    n_sequences: int = 10,
    batch_size: int = 1,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    do_sample: bool = True,
    use_cache: bool = True,
    use_normalized_sae: bool = True,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    debug_output: Optional[Path] = None,
    plot_output: Optional[Path] = None,
    plot_max_steps: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """
    Generate protein sequences with SAE-guided activation steering.

    Args:
        sae_dir: Directory containing a trained SAE (config.yaml + ae*.pt).
        feature_id: Index of the SAE feature to steer (single-feature interface).
            Not required when mode="none" or when --features is used.
        clamp_value: Steering magnitude for --feature_id. By default normalised:
            1.0 = max observed activation; >1.0 = amplify beyond training max.
            Use --use_raw_clamp_values for literal units.
        features: Multi-feature interface. Each entry is "feature_id:clamp_value"
            (e.g. "5900:5.0 42:2.0"). Overrides --feature_id / --clamp_value.
            All features share the global --mode.
        output_fasta: Path to write the generated sequences in FASTA format.
            If not provided, auto-named from run parameters and written to
            steering_outputs/.
        use_raw_clamp_values: If True, clamp values are used directly in raw
            SAE activation units instead of being scaled by per-feature maxima.
        mode: How to modify features: 'clamp' (set to value), 'add' (shift
            from current), 'ablate' (set to 0), or 'none' (bypass SAE entirely).
        steering_method: 'with_error' (default) adds back the SAE residual to
            preserve unmodelled signal. 'direct' replaces the hidden state
            purely with the SAE reconstruction.
        prefix: Optional amino-acid prefix string. Generation continues from
            this prefix. If None, generation starts de novo from the BOS token.
        n_sequences: Minimum number of sequences to generate. The actual count
            is ceil(n_sequences / batch_size) * batch_size when batch_size > 1.
        batch_size: Number of sequences to generate in parallel per forward
            pass. Values > 1 use generate_batch() and are faster on GPU.
        max_new_tokens: Maximum tokens to generate per sequence.
        temperature: Sampling temperature (ignored when do_sample=False).
        do_sample: Sample from the distribution; if False, use greedy decoding.
        use_cache: Reuse the attention KV cache across generation steps (feed only
            the newest token each step instead of re-running the full sequence).
            Numerically equivalent to use_cache=False but much faster; disable
            only to fall back to the reference path for debugging.
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
    # --- Resolve feature list ---
    # --features takes precedence over --feature_id / --clamp_value
    if features is not None:
        feat_clamp_pairs: list[tuple[int, float]] = []
        for spec in features:
            parts = spec.split(":")
            if len(parts) != 2:
                raise ValueError(
                    f"--features entries must be 'feature_id:clamp_value', got {spec!r}"
                )
            feat_clamp_pairs.append((int(parts[0]), float(parts[1])))
    elif feature_id is not None:
        feat_clamp_pairs = [(feature_id, clamp_value)]
    else:
        feat_clamp_pairs = []

    if mode != "none" and not feat_clamp_pairs:
        raise ValueError("--feature_id (or --features) is required unless --mode none")

    if seed is not None:
        torch.manual_seed(seed)

    device = device or get_device()
    sae_dir = Path(sae_dir)

    if mode == "none":
        # --- No steering: load PLM only ---
        model_name, layer_idx, embedder_type = load_config_info(sae_dir)
        print(f"Loading {embedder_type} model {model_name!r} (no steering) ...")
        embedder = get_embedder(embedder_type, model_name=model_name, device=device)
        model = embedder.model
        tokenizer = embedder.tokenizer
        layer_module = None
        hook_fn = None
    else:
        # --- Load SAE and resolve raw steer values ---
        sae_filename = "ae_normalized.pt" if use_normalized_sae else "ae.pt"
        print(f"Loading SAE from {sae_dir / sae_filename} ...")
        sae = load_sae(sae_dir, model_name=sae_filename, device=device)
        sae.eval()

        raw_feature_specs: list[tuple[int, float]] = []
        for fid, cv in feat_clamp_pairs:
            rescale = sae.activation_rescale_factor[fid].item()
            if not use_normalized_sae and not use_raw_clamp_values and abs(rescale - 1.0) < 1e-6:
                print(
                    f"WARNING: feature {fid}: ae.pt has activation_rescale_factor≈1 (not populated). "
                    "Normalised clamp_value will not scale to the true per-feature max. "
                    "Either use ae_normalized.pt (default) or pass --use_raw_clamp_values."
                )
            if mode == "ablate":
                raw_val = 0.0
                display = "0.0 (ablate)"
            elif use_raw_clamp_values:
                raw_val = cv
                display = f"{cv:.4f} (raw)"
            else:
                raw_val = cv * rescale
                display = f"{cv:.4f} (normalised) = {raw_val:.4f} raw"
            print(f"Feature {fid}: max activation = {rescale:.4f}  steer = {display}")
            raw_feature_specs.append((fid, raw_val))

        print(f"mode={mode}  method={steering_method}")

        # --- Infer PLM from config and load ---
        model_name, layer_idx, embedder_type = load_config_info(sae_dir)
        print(f"Loading {embedder_type} model {model_name!r}, layer {layer_idx} ...")
        embedder = get_embedder(embedder_type, model_name=model_name, device=device)
        model = embedder.model
        tokenizer = embedder.tokenizer

        layer_module = get_layer_module(model, embedder_type, layer_idx)
        verify_hook_target(model, layer_module, layer_idx, tokenizer, device)
        use_error = steering_method == "with_error"

        # Compute the initial prompt length so the hook skips prefix positions.
        _prefix_str = ("1" + prefix) if prefix else "1"
        _initial_seq_len = tokenizer(
            _prefix_str, add_special_tokens=False, return_tensors="pt"
        ).input_ids.shape[1]

        hook_fn = make_steering_hook(
            sae=sae,
            feature_specs=raw_feature_specs,
            mode=mode,
            use_error=use_error,
            steer_from_pos=_initial_seq_len,
        )

    eos_token_id = _get_eos_token_id(tokenizer)
    collect_debug = debug_output is not None or plot_output is not None
    vocab = tokenizer.get_vocab() if collect_debug else None

    # --- Generate ---
    n_batches = math.ceil(n_sequences / batch_size)
    total = n_batches * batch_size
    sequences: list[str] = []
    scores: list[tuple[float, float]] = []
    all_debug_steps: list[list[dict]] = []
    print(f"\nGenerating {total} sequence(s) ({n_batches} batch(es) of {batch_size}) ...")

    for batch_idx in range(n_batches):
        if batch_size > 1:
            batch_seqs, batch_debug = generate_batch(
                model=model,
                tokenizer=tokenizer,
                layer_module=layer_module,
                hook_fn=hook_fn,
                prefix=prefix,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                eos_token_id=eos_token_id,
                device=device,
                collect_debug=collect_debug,
                vocab=vocab,
                verbose=verbose,
                use_cache=use_cache,
            )
        else:
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
                verbose=verbose,
                use_cache=use_cache,
            )
            batch_seqs, batch_debug = [seq], [debug_steps]

        for b, (seq, debug_steps) in enumerate(zip(batch_seqs, batch_debug)):
            seq_idx = batch_idx * batch_size + b
            nll, ppl = score_sequence(model, tokenizer, seq, device)
            sequences.append(seq)
            scores.append((nll, ppl))
            all_debug_steps.append(debug_steps)

    # --- Write FASTA ---
    suffix = "_raw" if use_raw_clamp_values else "_norm"
    if not feat_clamp_pairs:
        feat_header_tag = "no_feature"
        feat_file_tag = "no_feature"
        value_tag = f"{clamp_value:.4f}{suffix}"
    elif len(feat_clamp_pairs) == 1:
        fid, cv = feat_clamp_pairs[0]
        feat_header_tag = f"feature_{fid}"
        feat_file_tag = f"feature_{fid}"
        value_tag = f"{cv:.4f}{suffix}"
    else:
        feat_header_tag = "features_" + "-".join(str(fid) for fid, _ in feat_clamp_pairs)
        feat_file_tag = feat_header_tag
        value_tag = "_".join(f"{fid}x{cv:.4f}" for fid, cv in feat_clamp_pairs) + suffix

    sae_name = Path(sae_dir).name
    prefix_tag_hdr = f"|prefix_{prefix[:6]}" if prefix else ""
    headers = [
        f"steered_{i}|sae_{sae_name}|{feat_header_tag}|{mode}|{steering_method}|value_{value_tag}{prefix_tag_hdr}|nll={nll:.2f}|ppl={ppl:.2f}"
        for i, (nll, ppl) in enumerate(scores)
    ]
    if output_fasta is None:
        if prefix:
            if len(prefix) <= 8:
                prefix_tag = f"_prefix_{prefix}"
            else:
                prefix_tag = f"_prefix_{prefix[:4]}..{prefix[-4:]}"
        else:
            prefix_tag = ""
        fname = f"{feat_file_tag}_{mode}_{steering_method}_value_{value_tag}{prefix_tag}.fasta"
        output_fasta = Path("steering_outputs") / fname
    write_fasta(sequences, headers, Path(output_fasta))
    print(f"\nWrote {total} sequence(s) to {output_fasta}")

    # --- Build debug payload (shared by JSON writer and plotter) ---
    payload = None
    if collect_debug:
        payload = {
            "meta": {
                "sae_dir": str(sae_dir),
                "features": [{"feature_id": fid, "clamp_value": cv} for fid, cv in feat_clamp_pairs],
                "mode": mode,
                "steering_method": steering_method,
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
