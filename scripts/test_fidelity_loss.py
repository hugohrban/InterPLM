"""
Quick sanity-check script for the two cross-entropy helpers in fidelity.py,
and a CLI for computing average sequence loss from a FASTA or plain-text file.

Usage
-----
Run unit tests (no arguments):
    python scripts/test_fidelity_loss.py

Evaluate average loss over sequences:
    python scripts/test_fidelity_loss.py --eval \\
        --file seqs.fasta \\
        --model esm \\
        --model-name facebook/esm2_t6_8M_UR50D \\
        --batch-size 32

    python scripts/test_fidelity_loss.py --eval \\
        --file seqs.txt \\
        --model progen \\
        --model-name hugohrban/progen2-small \\
        --batch-size 8

File formats accepted (auto-detected):
  FASTA  — lines starting with '>' are headers, next line(s) are sequence
  plain  — one sequence per non-empty line

Notes on the two loss functions
---------------------------------
  ESM (masked-LM):   logit[i] predicts token[i]
                     CE computed over interior tokens [1:length-1]
  ProGen (causal-LM): logit[i] predicts token[i+1]
                     CE computed over logits[0:length-2] vs tokens[1:length-1]
"""

import argparse
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from interplm.train.fidelity import (
    calculate_cross_entropy,
    calculate_cross_entropy_causal,
    calculate_loss_recovered,
)

VOCAB = 32  # small vocab for fast construction
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _one_hot_logits(token_ids: torch.Tensor, vocab: int = VOCAB) -> torch.Tensor:
    """Return logits that perfectly predict token_ids (one-hot, pre-softmax)."""
    logits = torch.zeros(*token_ids.shape, vocab)
    logits.scatter_(-1, token_ids.unsqueeze(-1), 100.0)
    return logits


def _check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        raise AssertionError(f"FAILED: {name}")


# ---------------------------------------------------------------------------
# 1. ESM (masked-LM): perfect predictor → loss ≈ 0
# ---------------------------------------------------------------------------
def test_esm_perfect_predictor():
    print("\n=== ESM / masked-LM ===")
    # Sequence layout: [CLS] aa0 aa1 aa2 [EOS]  (length = 5)
    tokens = torch.tensor([[1, 5, 6, 7, 2]])        # shape (1, 5)
    mask   = torch.ones(1, 5, dtype=torch.int)
    # Perfect logits: logit[i] → token[i]  (ESM convention)
    logits = _one_hot_logits(tokens)                # shape (1, 5, VOCAB)
    losses = calculate_cross_entropy(logits, tokens, mask)
    _check("single seq, perfect pred → loss ≈ 0", math.isclose(losses[0], 0.0, abs_tol=1e-3),
           f"loss={losses[0]:.6f}")


def test_esm_known_value():
    """Construct a 1-token interior, uniform logits → loss = log(VOCAB)."""
    # [CLS] aa0 [EOS]  →  only aa0 is scored
    tokens = torch.tensor([[0, 3, 0]])
    mask   = torch.ones(1, 3, dtype=torch.int)
    logits = torch.zeros(1, 3, VOCAB)               # uniform → log(VOCAB)
    losses = calculate_cross_entropy(logits, tokens, mask)
    expected = math.log(VOCAB)
    _check("uniform logits → loss = log(VOCAB)", math.isclose(losses[0], expected, rel_tol=1e-4),
           f"got={losses[0]:.4f}, expected={expected:.4f}")


def test_esm_excludes_boundary_tokens():
    """Replacing only the boundary-token logits with wrong predictions must not affect loss."""
    tokens = torch.tensor([[1, 5, 6, 7, 2]])
    mask   = torch.ones(1, 5, dtype=torch.int)
    # Perfect logits for interior; deliberately wrong for positions 0 and 4
    logits = _one_hot_logits(tokens)
    logits[0, 0, :] = -100.0   # wrong at CLS  → should be ignored
    logits[0, 4, :] = -100.0   # wrong at EOS  → should be ignored
    losses = calculate_cross_entropy(logits, tokens, mask)
    _check("boundary logit errors ignored → loss ≈ 0", math.isclose(losses[0], 0.0, abs_tol=1e-3),
           f"loss={losses[0]:.6f}")


def test_esm_padding_ignored():
    """Padded positions must not influence the loss."""
    # Two sequences; second is shorter (padded to length 5)
    tokens = torch.tensor([[1, 5, 6, 7, 2],
                            [1, 8, 9, 0, 0]])   # pad token = 0
    mask   = torch.tensor([[1, 1, 1, 1, 1],
                            [1, 1, 1, 0, 0]], dtype=torch.int)
    logits = _one_hot_logits(tokens)
    losses = calculate_cross_entropy(logits, tokens, mask)
    _check("both seqs perfect → loss ≈ 0 each", all(math.isclose(l, 0.0, abs_tol=1e-3) for l in losses),
           f"losses={losses}")


# ---------------------------------------------------------------------------
# 2. ProGen / causal-LM: perfect predictor → loss ≈ 0
# ---------------------------------------------------------------------------
def test_causal_perfect_predictor():
    print("\n=== ProGen / causal-LM ===")
    # Sequence layout: [BOS=1] aa0 aa1 aa2 [EOS=2]  (length = 5)
    # Causal: logit[i] predicts token[i+1]
    # So perfect logits need logit[0]→token[1], logit[1]→token[2], ...
    tokens = torch.tensor([[1, 5, 6, 7, 2]])
    mask   = torch.ones(1, 5, dtype=torch.int)
    # Build shifted-perfect logits: position k should be confident about token[k+1]
    logits = torch.zeros(1, 5, VOCAB)
    for k in range(4):           # k = 0..3, predicts token k+1
        logits[0, k, tokens[0, k + 1].item()] = 100.0
    losses = calculate_cross_entropy_causal(logits, tokens, mask)
    _check("single seq, perfect pred → loss ≈ 0", math.isclose(losses[0], 0.0, abs_tol=1e-3),
           f"loss={losses[0]:.6f}")


def test_causal_known_value():
    """[BOS] aa0 [EOS], uniform logits → loss = log(VOCAB) for the one AA position."""
    tokens = torch.tensor([[1, 5, 2]])
    mask   = torch.ones(1, 3, dtype=torch.int)
    logits = torch.zeros(1, 3, VOCAB)
    losses = calculate_cross_entropy_causal(logits, tokens, mask)
    expected = math.log(VOCAB)
    _check("uniform logits → loss = log(VOCAB)", math.isclose(losses[0], expected, rel_tol=1e-4),
           f"got={losses[0]:.4f}, expected={expected:.4f}")


def test_causal_excludes_boundary_tokens():
    """Wrong logits at boundary positions must not affect causal loss."""
    tokens = torch.tensor([[1, 5, 6, 7, 2]])
    mask   = torch.ones(1, 5, dtype=torch.int)
    # Shifted-perfect interior logits
    logits = torch.zeros(1, 5, VOCAB)
    for k in range(4):
        logits[0, k, tokens[0, k + 1].item()] = 100.0
    # Corrupt the last position (which is excluded from causal targets)
    logits[0, 4, :] = -100.0
    losses = calculate_cross_entropy_causal(logits, tokens, mask)
    _check("last-position logit error ignored → loss ≈ 0", math.isclose(losses[0], 0.0, abs_tol=1e-3),
           f"loss={losses[0]:.6f}")


def test_causal_padding_ignored():
    tokens = torch.tensor([[1, 5, 6, 7, 2],
                            [1, 8, 9, 0, 0]])
    mask   = torch.tensor([[1, 1, 1, 1, 1],
                            [1, 1, 1, 0, 0]], dtype=torch.int)
    logits = torch.zeros(2, 5, VOCAB)
    # Seq 0: shifted-perfect for length-5
    for k in range(4):
        logits[0, k, tokens[0, k + 1].item()] = 100.0
    # Seq 1: shifted-perfect for length-3 (positions 0,1 only)
    for k in range(2):
        logits[1, k, tokens[1, k + 1].item()] = 100.0
    losses = calculate_cross_entropy_causal(logits, tokens, mask)
    _check("both seqs perfect → loss ≈ 0 each", all(math.isclose(l, 0.0, abs_tol=1e-3) for l in losses),
           f"losses={losses}")


# ---------------------------------------------------------------------------
# 3. Demonstrate and quantify the difference between the two functions
# ---------------------------------------------------------------------------
def test_esm_vs_causal_same_input_differ():
    """
    With the *same* logit tensor both functions should give different losses,
    because ESM reads logit[i]→token[i] while causal reads logit[i]→token[i+1].

    We build logits that are ESM-perfect (logit[i] confident about token[i]).
    ESM loss should be ≈ 0; causal loss should be > 0.
    """
    print("\n=== ESM vs Causal with same inputs ===")
    tokens = torch.tensor([[1, 5, 6, 7, 2]])
    mask   = torch.ones(1, 5, dtype=torch.int)

    # Logits aligned ESM-style
    esm_logits = _one_hot_logits(tokens)

    esm_loss   = calculate_cross_entropy(esm_logits, tokens, mask)[0]
    causal_loss = calculate_cross_entropy_causal(esm_logits, tokens, mask)[0]

    _check("ESM-aligned logits → ESM loss ≈ 0",  math.isclose(esm_loss,   0.0, abs_tol=1e-3),
           f"esm_loss={esm_loss:.6f}")
    _check("ESM-aligned logits → causal loss > 0", causal_loss > 0.1,
           f"causal_loss={causal_loss:.4f}")
    print(f"    ESM loss={esm_loss:.6f}   causal loss={causal_loss:.4f}  (differ by {abs(causal_loss - esm_loss):.4f})")

    # Flip: causal-perfect logits → causal ≈ 0, ESM > 0
    causal_logits = torch.zeros(1, 5, VOCAB)
    for k in range(4):
        causal_logits[0, k, tokens[0, k + 1].item()] = 100.0

    esm_loss2    = calculate_cross_entropy(causal_logits, tokens, mask)[0]
    causal_loss2 = calculate_cross_entropy_causal(causal_logits, tokens, mask)[0]

    _check("Causal-aligned logits → causal loss ≈ 0", math.isclose(causal_loss2, 0.0, abs_tol=1e-3),
           f"causal_loss={causal_loss2:.6f}")
    _check("Causal-aligned logits → ESM loss > 0", esm_loss2 > 0.1,
           f"esm_loss={esm_loss2:.4f}")
    print(f"    ESM loss={esm_loss2:.4f}   causal loss={causal_loss2:.6f}  (differ by {abs(esm_loss2 - causal_loss2):.4f})")


def test_causal_shift_off_by_one():
    """
    Explicitly verify the +1 token shift in the causal function.

    We construct logits where only position k=0 is informative (confident about
    some token X).  We check that:
      - causal: the target at position 0 is tokens[1]  (shift by +1)
      - ESM:    the target at position 1 is tokens[1]  (no shift, but interior slice)
    """
    print("\n=== Shift alignment check ===")
    # [BOS=1] [A=5] [B=6] [EOS=2]
    tokens = torch.tensor([[1, 5, 6, 2]])
    mask   = torch.ones(1, 4, dtype=torch.int)

    # Logit[0] is confident about token id 5 (= tokens[1])
    logits = torch.full((1, 4, VOCAB), -10.0)
    logits[0, 0, 5] = 100.0

    causal_loss = calculate_cross_entropy_causal(logits, tokens, mask)[0]
    # Interior for causal: logits[0:2] → targets tokens[1:3] = [5, 6]
    # logit[0] correctly predicts 5, logit[1] is flat → partial loss
    manual_loss_0 = F.cross_entropy(logits[0, 0], tokens[0, 1]).item()  # ≈ 0
    manual_loss_1 = F.cross_entropy(logits[0, 1], tokens[0, 2]).item()  # high
    expected_causal = (manual_loss_0 + manual_loss_1) / 2
    _check("causal loss matches manual shift calculation",
           math.isclose(causal_loss, expected_causal, rel_tol=1e-4),
           f"got={causal_loss:.6f}, expected={expected_causal:.6f}")


# ---------------------------------------------------------------------------
# 4. calculate_loss_recovered
# ---------------------------------------------------------------------------
def test_loss_recovered_perfect():
    print("\n=== calculate_loss_recovered ===")
    r = calculate_loss_recovered(ce_autoencoder=1.0, ce_identity=1.0, ce_zero_ablation=3.0)
    _check("SAE = identity → 100% recovered", math.isclose(r, 100.0, rel_tol=1e-6),
           f"got={r}")


def test_loss_recovered_zero():
    r = calculate_loss_recovered(ce_autoencoder=3.0, ce_identity=1.0, ce_zero_ablation=3.0)
    _check("SAE = zero-ablation → 0% recovered", math.isclose(r, 0.0, abs_tol=1e-6),
           f"got={r}")


def test_loss_recovered_half():
    r = calculate_loss_recovered(ce_autoencoder=2.0, ce_identity=1.0, ce_zero_ablation=3.0)
    _check("halfway → 50% recovered", math.isclose(r, 50.0, rel_tol=1e-6),
           f"got={r}")


def test_loss_recovered_clipped():
    # SAE better than identity → clipped to 100
    r = calculate_loss_recovered(ce_autoencoder=0.5, ce_identity=1.0, ce_zero_ablation=3.0)
    _check("SAE better than identity → clipped to 100%", math.isclose(r, 100.0, rel_tol=1e-6),
           f"got={r}")


def test_loss_recovered_zero_division():
    r = calculate_loss_recovered(ce_autoencoder=1.0, ce_identity=1.0, ce_zero_ablation=1.0)
    _check("zero denominator → 0.0 (no crash)", math.isclose(r, 0.0, abs_tol=1e-9),
           f"got={r}")


# ---------------------------------------------------------------------------
# Sequence file loading
# ---------------------------------------------------------------------------

def load_sequences(path: str) -> list[str]:
    """Load sequences from a FASTA or plain-text (one-per-line) file.

    Auto-detects format: if any line starts with '>' it's treated as FASTA,
    otherwise each non-empty line is a sequence.
    """
    lines = open(path).read().splitlines()
    is_fasta = any(l.startswith(">") for l in lines)

    if is_fasta:
        seqs, current = [], []
        for line in lines:
            if line.startswith(">"):
                if current:
                    seqs.append("".join(current))
                current = []
            else:
                stripped = line.strip()
                if stripped:
                    current.append(stripped)
        if current:
            seqs.append("".join(current))
        return seqs
    else:
        return [l.strip() for l in lines if l.strip()]


# ---------------------------------------------------------------------------
# Per-model loss calculators (no nnsight, no intervention — plain forward pass)
# ---------------------------------------------------------------------------

def eval_esm_loss(sequences: list[str], model_name: str, batch_size: int) -> list[float]:
    """Compute per-sequence CE loss with an ESM masked-LM."""
    from transformers import AutoTokenizer, EsmForMaskedLM
    from interplm.utils import get_device

    device = get_device()
    print(f"Loading ESM model {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name).to(device).eval()

    all_losses = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="ESM batches"):
        batch_seqs = sequences[i : i + batch_size]
        enc = tokenizer(batch_seqs, return_tensors="pt", padding=True,
                        truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attn_mask).logits  # (B, L, V)

        batch_losses = calculate_cross_entropy(logits, input_ids, attn_mask)
        all_losses.extend(batch_losses)

    return all_losses


def eval_progen_loss(sequences: list[str], model_name: str, batch_size: int) -> list[float]:
    """Compute per-sequence causal CE loss with a ProGen2 model."""
    from interplm.embedders import get_embedder
    from interplm.utils import get_device

    device = get_device()
    print(f"Loading ProGen2 model {model_name} …")
    embedder = get_embedder("progen", model_name=model_name, device=device)
    model = embedder.model.eval()
    pad_id = embedder.tokenizer.pad_token_id

    all_losses = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="ProGen batches"):
        batch_seqs = sequences[i : i + batch_size]
        enc = embedder.tokenize(batch_seqs)
        input_ids = enc["input_ids"].to(device)
        attn_mask = (input_ids != pad_id).to(torch.int).to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask=attn_mask)
            logits = output.logits  # (B, L, V)

        batch_losses = calculate_cross_entropy_causal(logits, input_ids, attn_mask)
        all_losses.extend(batch_losses)

    return all_losses


def run_eval(args) -> None:
    seqs = load_sequences(args.file)
    print(f"Loaded {len(seqs)} sequences from {args.file}")

    if args.model == "esm":
        model_name = args.model_name or "facebook/esm2_t6_8M_UR50D"
        losses = eval_esm_loss(seqs, model_name, args.batch_size)
        ce_fn = "calculate_cross_entropy (masked-LM, no shift)"
    elif args.model == "progen":
        model_name = args.model_name or "hugohrban/progen2-small"
        losses = eval_progen_loss(seqs, model_name, args.batch_size)
        ce_fn = "calculate_cross_entropy_causal (causal-LM, +1 shift)"
    else:
        sys.exit(f"Unknown model type: {args.model}")

    avg = float(np.mean(losses))
    std = float(np.std(losses))
    avg_ppl = float(np.exp(avg))

    print(f"\nModel      : {model_name}")
    print(f"CE function: {ce_fn}")
    print(f"Sequences  : {len(losses)}")
    print(f"Avg CE loss: {avg:.4f}  ±  {std:.4f}")
    print(f"Avg PPL    : {avg_ppl:.4f}")
    print(f"Min / Max  : {min(losses):.4f} / {max(losses):.4f}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate average loss instead of running unit tests")
    parser.add_argument("--file", help="Path to FASTA or plain-text sequence file")
    parser.add_argument("--model", choices=["esm", "progen"],
                        help="Model family to use")
    parser.add_argument("--model-name", dest="model_name", default=None,
                        help="HuggingFace model ID (defaults: esm2_t6_8M, progen2-small)")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=32,
                        help="Batch size for inference (default: 32)")
    args = parser.parse_args()

    if args.eval:
        if not args.file or not args.model:
            parser.error("--eval requires --file and --model")
        run_eval(args)
        sys.exit(0)

    # --- unit tests ---
    tests = [
        test_esm_perfect_predictor,
        test_esm_known_value,
        test_esm_excludes_boundary_tokens,
        test_esm_padding_ignored,
        test_causal_perfect_predictor,
        test_causal_known_value,
        test_causal_excludes_boundary_tokens,
        test_causal_padding_ignored,
        test_esm_vs_causal_same_input_differ,
        test_causal_shift_off_by_one,
        test_loss_recovered_perfect,
        test_loss_recovered_zero,
        test_loss_recovered_half,
        test_loss_recovered_clipped,
        test_loss_recovered_zero_division,
    ]

    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed.append(str(e))

    print(f"\n{'='*60}")
    if failed:
        print(f"FAILED {len(failed)}/{len(tests)} tests:")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")
