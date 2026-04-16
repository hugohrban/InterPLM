#!/usr/bin/env python
"""
Diagnostic script for ProGen2 loss computation.

Checks two things:
  1. Attention mask / padding correctness — are we accidentally computing
     loss on padding tokens?
  2. Effect of ProGen2 special tokens "1" (N-terminus) and "2" (C-terminus)
     on per-sequence cross-entropy.

Usage:
  python scripts/check_progen_loss.py sequences.fasta
  python scripts/check_progen_loss.py sequences.fasta --n-seqs 4 --max-len 128
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from interplm.embedders.progen2 import ProGen2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_fasta(fasta_path: Path, max_seqs: int) -> List[Tuple[str, str]]:
    """Read up to max_seqs (header, sequence) pairs from a FASTA file."""
    records = []
    with open(fasta_path) as f:
        header, parts = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(parts)))
                    if len(records) >= max_seqs:
                        return records
                header, parts = line[1:], []
            else:
                parts.append(line)
        if header and len(records) < max_seqs:
            records.append((header, "".join(parts)))
    return records


def tokenize_batch(tokenizer, sequences: List[str], device: str, max_length: int):
    """Tokenize a list of sequences with padding, no special tokens added."""
    return tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    ).to(device)


def compute_per_token_loss(
    model, tokenizer, sequences: List[str], device: str, max_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for a batch and return per-token CE loss.

    For a causal LM the loss at position i is -log P(x_{i+1} | x_{0..i}),
    so the returned tensors have shape [B, T-1] (one fewer than input length).

    Returns:
        per_token_loss  [B, T-1]  — CE at each (non-shifted) prediction step
        real_token_mask [B, T-1]  — 1 where the *target* token is real (not padding)
    """
    inputs = tokenize_batch(tokenizer, sequences, device, max_length)
    input_ids      = inputs["input_ids"]       # [B, T]
    attention_mask = inputs["attention_mask"]  # [B, T]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits  # [B, T, V]

    # Causal shift: logits[i] predicts input_ids[i+1]
    shift_logits = logits[:, :-1, :].contiguous()    # [B, T-1, V]
    shift_labels = input_ids[:, 1:].contiguous()     # [B, T-1]
    shift_mask   = attention_mask[:, 1:].contiguous()  # [B, T-1]

    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(input_ids.shape[0], -1)  # [B, T-1]

    return per_token_loss, shift_mask


def masked_mean_ce(per_token_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean CE over non-padding positions, one scalar per sequence."""
    return (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


# ---------------------------------------------------------------------------
# Section 1 — padding / attention mask audit
# ---------------------------------------------------------------------------

def check_padding_masking(
    model, tokenizer, sequences: List[str], device: str, max_length: int
) -> None:
    print("\n" + "=" * 65)
    print("SECTION 1: Attention mask / padding correctness")
    print("=" * 65)

    inputs = tokenize_batch(tokenizer, sequences, device, max_length)
    input_ids      = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    B, T = input_ids.shape

    # 1a — show real vs padding token counts
    print(f"\n[1a] Batch shape: {B} sequences × {T} tokens (after padding)")
    print(f"     {'seq':>4}  {'seq_len':>8}  {'real_tokens':>12}  {'padding':>8}")
    for i, seq in enumerate(sequences):
        n_real = int(attention_mask[i].sum())
        n_pad  = T - n_real
        print(f"     {i:>4}  {len(seq):>8}  {n_real:>12}  {n_pad:>8}")

    # 1b — inspect loss values at padding positions
    per_token_loss, shift_mask = compute_per_token_loss(
        model, tokenizer, sequences, device, max_length
    )
    pad_positions = shift_mask == 0

    print(f"\n[1b] Loss at padding positions (informational — should be excluded from mean):")
    if pad_positions.any():
        pad_losses = per_token_loss[pad_positions]
        print(f"     mean={pad_losses.mean():.4f}  "
              f"min={pad_losses.min():.4f}  max={pad_losses.max():.4f}")
        print(f"     Note: pad token = eos token, so these are non-zero but should")
        print(f"     not be included when averaging. Masking is critical.")
    else:
        print("     No padding in this batch (all sequences same length).")

    # 1c — masked vs unmasked mean CE per sequence
    print(f"\n[1c] Mean CE per sequence — masked vs unmasked:")
    print(f"     {'seq':>4}  {'masked (correct)':>18}  {'unmasked (wrong)':>18}  {'delta':>8}")
    for i in range(B):
        masked   = masked_mean_ce(per_token_loss[i:i+1], shift_mask[i:i+1]).item()
        unmasked = per_token_loss[i].mean().item()
        delta    = unmasked - masked
        flag     = "  <-- padding inflates loss" if abs(delta) > 0.01 else ""
        print(f"     {i:>4}  {masked:>18.4f}  {unmasked:>18.4f}  {delta:>+8.4f}{flag}")

    # 1d — cross-check against HF's built-in labels=-100 mechanism
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    with torch.no_grad():
        hf_loss = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss.item()

    our_loss = (per_token_loss * shift_mask).sum().item() / shift_mask.sum().item()
    match = abs(hf_loss - our_loss) < 1e-3
    print(f"\n[1d] Cross-check: HF loss (labels=-100 at pad) vs our masked mean")
    print(f"     HF  loss: {hf_loss:.4f}")
    print(f"     Our loss: {our_loss:.4f}")
    print(f"     Match:    {'YES — masking is consistent' if match else 'NO — discrepancy, investigate'}")


# ---------------------------------------------------------------------------
# Section 2 — special token effect on CE
# ---------------------------------------------------------------------------

def check_special_tokens(
    model, tokenizer, sequences: List[str], device: str, max_length: int
) -> None:
    print("\n" + "=" * 65)
    print("SECTION 2: Effect of special tokens '1' (N-term) and '2' (C-term)")
    print("=" * 65)
    print("  ProGen2 was trained on sequences formatted as: 1<seq>2")
    print("  This section compares CE loss with vs without those tokens.\n")

    seqs_raw = sequences
    seqs_fwd = ["1" + s + "2" for s in sequences]           # N→C forward
    seqs_rev = ["2" + s[::-1] + "1" for s in sequences]     # C→N reverse

    loss_raw, mask_raw = compute_per_token_loss(model, tokenizer, seqs_raw, device, max_length)
    loss_fwd, mask_fwd = compute_per_token_loss(model, tokenizer, seqs_fwd, device, max_length)
    loss_rev, mask_rev = compute_per_token_loss(model, tokenizer, seqs_rev, device, max_length)

    ce_raw = masked_mean_ce(loss_raw, mask_raw)
    ce_fwd = masked_mean_ce(loss_fwd, mask_fwd)
    ce_rev = masked_mean_ce(loss_rev, mask_rev)

    print(f"  {'seq':>4}  {'no tokens':>10}  {'1<seq>2 (fwd)':>14}  {'2<rev>1 (rev)':>14}  {'fwd-raw':>8}  {'rev-raw':>8}  {'fwd-rev':>8}")
    for i in range(len(sequences)):
        print(f"  {i:>4}"
              f"  {ce_raw[i].item():>10.4f}"
              f"  {ce_fwd[i].item():>14.4f}"
              f"  {ce_rev[i].item():>14.4f}"
              f"  {ce_fwd[i].item() - ce_raw[i].item():>+8.4f}"
              f"  {ce_rev[i].item() - ce_raw[i].item():>+8.4f}"
              f"  {ce_fwd[i].item() - ce_rev[i].item():>+8.4f}")

    print(f"\n  Summary across {len(sequences)} sequences:")
    print(f"    No tokens:     mean CE = {ce_raw.mean():.4f} ± {ce_raw.std():.4f}")
    print(f"    1<seq>2 (fwd): mean CE = {ce_fwd.mean():.4f} ± {ce_fwd.std():.4f}  (delta vs raw: {(ce_fwd-ce_raw).mean():+.4f})")
    print(f"    2<rev>1 (rev): mean CE = {ce_rev.mean():.4f} ± {ce_rev.std():.4f}  (delta vs raw: {(ce_rev-ce_raw).mean():+.4f})")
    print(f"    fwd vs rev:    mean delta = {(ce_fwd - ce_rev).mean():+.4f} ± {(ce_fwd - ce_rev).std():.4f}")

    # Per-position breakdown for first sequence — all three conditions
    print(f"\n  Per-position loss for seq[0] (first 20 positions):")
    n_show  = min(20, loss_raw.shape[1])
    seq0    = sequences[0]
    seq0rev = seq0[::-1]

    raw_tgts = list(seq0[1:n_show+1])
    fwd_tgts = list(("1" + seq0 + "2")[1:n_show+1])
    rev_tgts = list(("2" + seq0rev + "1")[1:n_show+1])

    for label, tgts, loss in [
        ("no tokens", raw_tgts, loss_raw),
        ("1<seq>2  ", fwd_tgts, loss_fwd),
        ("2<rev>1  ", rev_tgts, loss_rev),
    ]:
        print(f"\n    {label} — target at each step:")
        print(f"    token:  " + "  ".join(f"{c:>4}" for c in tgts))
        print(f"    loss:   " + "  ".join(f"{v:>4.2f}" for v in loss[0, :n_show].tolist()))


# ---------------------------------------------------------------------------
# Section 3 — reverse-engineer the true N/C-terminus token IDs
# ---------------------------------------------------------------------------

def check_token_ids(model, tokenizer, sequences: List[str], device: str) -> None:
    print("\n" + "=" * 65)
    print("SECTION 3: Reverse-engineering the terminus token IDs")
    print("=" * 65)

    # 3a — full vocabulary dump, highlighting non-standard tokens
    vocab = tokenizer.get_vocab()
    std_aa = set("ACDEFGHIKLMNPQRSTVWY")
    special = {tok: tid for tok, tid in sorted(vocab.items(), key=lambda x: x[1])
               if tok not in std_aa}
    print(f"\n[3a] Full vocab size: {len(vocab)}")
    print(f"     Non-amino-acid tokens (id → token):")
    for tok, tid in sorted(special.items(), key=lambda x: x[1]):
        extra = ""
        if tid == tokenizer.pad_token_id:
            extra = "  ← pad_token"
        if tid == tokenizer.eos_token_id:
            extra += "  ← eos_token"
        if hasattr(tokenizer, "bos_token_id") and tid == tokenizer.bos_token_id:
            extra += "  ← bos_token"
        print(f"       id={tid:>4}  repr={repr(tok)}{extra}")

    # 3b — what do the strings "1" and "2" encode to?
    id_1 = tokenizer.encode("1", add_special_tokens=False)
    id_2 = tokenizer.encode("2", add_special_tokens=False)
    print(f"\n[3b] tokenizer.encode('1') = {id_1}")
    print(f"     tokenizer.encode('2') = {id_2}")
    # Reverse lookup: what token string is at those IDs?
    inv_vocab = {v: k for k, v in vocab.items()}
    for ids, name in [(id_1, "'1'"), (id_2, "'2'")]:
        for i in ids:
            print(f"     token id {i} → repr={repr(inv_vocab.get(i, '???'))}")

    # 3c — unconditional first-token prediction
    # Ask: what token does the model predict before seeing anything?
    # Proxy: logits from a single-token input of each candidate.
    # Specifically, use the PAD/EOS token as a "neutral" seed and
    # look at the next-token distribution.
    print(f"\n[3c] Top-10 predicted next tokens from a neutral seed (pad/eos):")
    seed_id = tokenizer.pad_token_id
    seed = torch.tensor([[seed_id]], device=device)
    with torch.no_grad():
        logits = model(input_ids=seed).logits[0, -1]  # [V]
    probs = F.softmax(logits, dim=-1)
    top = torch.topk(probs, 10)
    for p, idx in zip(top.values.tolist(), top.indices.tolist()):
        tok_str = inv_vocab.get(idx, "???")
        print(f"       p={p:.4f}  id={idx:>4}  token={repr(tok_str)}")

    # 3d — try each candidate "1" encoding as seed and show next-token distribution
    print(f"\n[3d] Top-10 predicted next tokens when seeded with the '1'-string token(s):")
    for i in id_1:
        seed = torch.tensor([[i]], device=device)
        with torch.no_grad():
            logits = model(input_ids=seed).logits[0, -1]
        probs = F.softmax(logits, dim=-1)
        top = torch.topk(probs, 10)
        print(f"     Seed token id={i} (repr={repr(inv_vocab.get(i, '???'))}):")
        for p, idx in zip(top.values.tolist(), top.indices.tolist()):
            tok_str = inv_vocab.get(idx, "???")
            print(f"       p={p:.4f}  id={idx:>4}  token={repr(tok_str)}")

    # 3e — exhaustive search: for every non-AA token, seed with it and measure
    #       how amino-acid-like the next-token distribution is (AA mass = sum of
    #       P(token) for token in std_aa).  The true N-term token should have the
    #       highest AA mass.
    print(f"\n[3e] AA-mass of next-token dist for each non-AA seed token")
    print(f"     (higher = model expects an amino acid to follow = likely N-term token):")
    aa_ids = torch.tensor([vocab[aa] for aa in std_aa if aa in vocab], device=device)
    rows = []
    for tok, tid in special.items():
        seed = torch.tensor([[tid]], device=device)
        with torch.no_grad():
            logits = model(input_ids=seed).logits[0, -1]
        probs = F.softmax(logits, dim=-1)
        aa_mass = probs[aa_ids].sum().item()
        rows.append((aa_mass, tid, tok))
    rows.sort(reverse=True)
    for aa_mass, tid, tok in rows[:15]:
        flag = "  ← likely N-term" if aa_mass > 0.8 else ""
        print(f"       id={tid:>4}  token={repr(tok):<12}  AA-mass={aa_mass:.4f}{flag}")

    # 3f — generate a short sequence from each top candidate and show it
    # Use manual greedy decode (model.generate incompatible with ProGen2 config)
    print(f"\n[3f] Greedy generation (20 tokens) from top-3 AA-mass seeds:")
    for _, tid, tok in rows[:3]:
        ids = torch.tensor([[tid]], device=device)
        for _ in range(20):
            with torch.no_grad():
                next_logits = model(input_ids=ids).logits[0, -1]
            next_id = next_logits.argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, next_id], dim=1)
        generated = [inv_vocab.get(i, "?") for i in ids[0].tolist()]
        print(f"     seed id={tid} ({repr(tok)}):  {''.join(generated)}")


# ---------------------------------------------------------------------------
# Section 4 — embedding shift due to special tokens
# ---------------------------------------------------------------------------

def check_embedding_shift(
    model, tokenizer, sequences: List[str], device: str, max_length: int
) -> None:
    print("\n" + "=" * 65)
    print("SECTION 4: Embedding shift due to special tokens")
    print("=" * 65)
    print("  Comparing last-layer hidden states for matching amino acid positions.")
    print("  With '1..2', each AA is shifted right by 1, so we compare:")
    print("  hidden[raw][pos i]  vs  hidden[tok][pos i+1]\n")

    def get_hidden_states(seqs):
        inputs = tokenize_batch(tokenizer, seqs, device, max_length)
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
        # last layer hidden states [B, T, D]
        return out.hidden_states[-1], inputs["attention_mask"]

    seqs_raw = sequences
    seqs_tok = ["1" + s + "2" for s in sequences]

    hidden_raw, mask_raw = get_hidden_states(seqs_raw)
    hidden_tok, mask_tok = get_hidden_states(seqs_tok)

    cos_sims, l2_dists = [], []
    for i, seq in enumerate(sequences):
        n = len(seq)
        # raw:  positions 0..n-1  correspond to AA 0..n-1
        # tok:  positions 1..n    correspond to AA 0..n-1 (pos 0 is "1")
        h_raw = hidden_raw[i, :n, :]       # [n, D]
        h_tok = hidden_tok[i, 1:n + 1, :]  # [n, D]

        cos = F.cosine_similarity(h_raw, h_tok, dim=-1)  # [n]
        l2  = (h_raw - h_tok).norm(dim=-1)               # [n]

        cos_sims.append(cos)
        l2_dists.append(l2)

    all_cos = torch.cat(cos_sims)
    all_l2  = torch.cat(l2_dists)

    print(f"  Cosine similarity (raw vs 1..2), per amino acid position:")
    print(f"    mean={all_cos.mean():.4f}  std={all_cos.std():.4f}  "
          f"min={all_cos.min():.4f}  max={all_cos.max():.4f}")

    print(f"\n  L2 distance (raw vs 1..2), per amino acid position:")
    print(f"    mean={all_l2.mean():.4f}  std={all_l2.std():.4f}  "
          f"min={all_l2.min():.4f}  max={all_l2.max():.4f}")

    # Per-sequence summary
    print(f"\n  Per-sequence mean cosine similarity:")
    print(f"  {'seq':>4}  {'seq_len':>8}  {'mean_cos':>10}  {'mean_l2':>10}")
    for i, seq in enumerate(sequences):
        print(f"  {i:>4}  {len(seq):>8}  {cos_sims[i].mean():.4f}      {l2_dists[i].mean():.4f}")

    # Positional trend: does the shift effect grow or shrink along the sequence?
    min_len = min(len(s) for s in sequences)
    n_pos = min(min_len, 50)
    stacked_cos = torch.stack([c[:n_pos] for c in cos_sims])  # [B, n_pos]
    pos_mean_cos = stacked_cos.mean(dim=0)

    print(f"\n  Position-wise mean cosine similarity (first {n_pos} positions):")
    print("  pos:  " + " ".join(f"{i:>5}" for i in range(0, n_pos, 5)))
    print("  cos:  " + " ".join(f"{pos_mean_cos[i]:.3f}" for i in range(0, n_pos, 5)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ProGen2 loss diagnostic: check padding masking and special token effect"
    )
    parser.add_argument("fasta", type=Path, help="Input FASTA file")
    parser.add_argument(
        "--model", default="hugohrban/progen2-small",
        help="Model name or local path (default: hugohrban/progen2-small)"
    )
    parser.add_argument(
        "--n-seqs", type=int, default=8,
        help="Number of sequences to load from FASTA (default: 8)"
    )
    parser.add_argument(
        "--max-len", type=int, default=256,
        help="Truncate sequences to this length before tokenizing (default: 256)"
    )
    parser.add_argument("--device", default=None, help="cuda / cpu / mps (default: auto)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model}")

    print("\nLoading tokenizer and model...")
    embedder = ProGen2(model_name=args.model, device=device)
    model     = embedder.model
    tokenizer = embedder.tokenizer

    records   = read_fasta(args.fasta, max_seqs=args.n_seqs)
    sequences = [
        "".join(seq.strip().upper().split())[: args.max_len]
        for _, seq in records
    ]
    print(f"\nLoaded {len(sequences)} sequences, lengths: {[len(s) for s in sequences]}")

    check_padding_masking(model, tokenizer, sequences, device, args.max_len + 2)
    check_special_tokens(model, tokenizer, sequences, device, args.max_len + 2)
    check_token_ids(model, tokenizer, sequences, device)
    check_embedding_shift(model, tokenizer, sequences, device, args.max_len + 2)


if __name__ == "__main__":
    main()
