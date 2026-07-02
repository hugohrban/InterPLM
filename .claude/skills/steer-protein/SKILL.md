---
description: Guide for SAE activation steering on progen2-large to generate proteins satisfying a biological condition. Use when asked to steer the model, generate proteins with specific properties (transit peptides, binding sites, domains), find features for a concept, run anti-target analysis, discover pro/anti-condition SAE features, or interpret activation steering results.
---

Activation Steering Skill: Generating Proteins that Satisfy a Condition

This document describes how to use the InterPLM codebase to steer progen2-large to generate proteins with desired properties via SAE feature intervention. The workflow applies to any generatable condition — transit peptides, binding sites, specific domains — not just the transit peptide example the scripts were developed on.

---

## 1. System Overview

**Model:** `hugohrban/progen2-large` (1.2B params, autoregressive, 32 layers). Generation is prefix-conditioned; a single methionine `M` is the standard neutral prefix.

**SAE:** A sparse autoencoder trained on the hidden states of one layer. The recommended checkpoint is `trained_saes/best_progen_large_24`, which covers layer 24. It has:
- Dictionary size: 10,240 features
- k = 350 active features per token (BatchTopK)
- Checkpoint file: `ae_normalized.pt` (use this, not `ae.pt`)
- Config: `config.yaml` (records `model_name`, `layer_idx`, activation dim, etc.)

**Steering mechanism:** A forward hook intercepts the hidden state at the SAE's layer, encodes it, modifies one or more feature activations, then decodes back to the residual stream. experiments here can use `--steering_method direct` (full replacement) or `--steering_method with_error` (adding the difference between the reconstruction and tru values, may be good if the SAE has high reconstruction loss), it is good to compare both steering methods. and `--mode clamp` (set feature to a fixed value).

---

## 2. SAE Directory Structure

```
trained_saes/best_progen_large_24/
├── ae_normalized.pt                 # SAE weights (use this)
├── ae.pt                            # unnormalized weights (avoid)
├── config.yaml                      # training config; has model_name and layer_idx
├── max_activations_per_feature.pt   # per-feature max activation tensor (dict_size,)
├── Per_feature_statistics.yaml      # mean, std, sparsity per feature
├── Per_feature_max_examples.yaml    # example sequences where each feature fires strongly
├── rank_eval/                       # pre-computed rank-eval results for specific concepts
│   ├── Transit_peptide_any_f1_guided.json
│   ├── Transit_peptide_Mitochondrion_f1_guided.json
│   └── ...
└── results_test_counts/
    └── concept_f1_scores.csv        # F1 of every feature × every concept (large file)
```

**Reading max activations in Python:**
```python
import torch
from interplm.sae.inference import load_sae
from pathlib import Path

sae = load_sae(Path("trained_saes/best_progen_large_24"), model_name="ae_normalized.pt", device="cuda:0")
# sae.activation_rescale_factor is a (dict_size,) tensor
max_act = sae.activation_rescale_factor[feature_id].item()
```
This is the value that 1.0× clamp maps to. Steering at N× means the raw value injected is `N * max_act`.

---

## 3. Finding Candidate Features for a Concept

### 3a. Query the concept F1 CSV

`results_test_counts/concept_f1_scores.csv` contains F1 scores for every (feature, concept, threshold) combination. Each row has:

| column | meaning |
|---|---|
| `concept` | concept string, e.g. `"Transit peptide_any"` or `"Binding site_ATP"` |
| `feature` | SAE feature index |
| `f1` | standard F1 (global threshold) |
| `f1_per_domain` | per-residue F1 inside annotated domains — **the most useful metric for position-specific features** |
| `precision`, `recall` | standard precision/recall |
| `is_aa_level_concept` | True for residue-level concepts (binding sites, active sites); False for sequence-level |

Query example:
```python
import pandas as pd
df = pd.read_csv("trained_saes/best_progen_large_24/results_test_counts/concept_f1_scores.csv")
results = df[df["concept"].str.contains("Transit peptide_any")]
top = results.sort_values("f1_per_domain", ascending=False).head(20)
print(top[["concept", "feature", "f1_per_domain", "f1", "recall", "precision"]])
```

**Important:** high `f1_per_domain` means the feature is a good *detector* of the concept in natural proteins. It does **not** guarantee the feature is a good *causal driver* of generation — that must be verified by steering.

### 3b. Use the rank_eval script for richer analysis

`scripts/rank_eval_concepts.py` computes rank-based enrichment metrics and prints examples:
```bash
python scripts/rank_eval_concepts.py \
    --sae_dir trained_saes/best_progen_large_24 \
    --embed_dir data/embeddings/progen2_large/layer_24 \
    --annot_dir data/annotations \
    --concept_query "Transit peptide" \
    --top_k_features 10 \
    --print_examples
```
Pre-computed results for some concepts are cached in `rank_eval/*.json`.

### 3c. Inspect activation profiles of candidate features

Before steering, compare what the feature actually *does* in natural proteins. Use `scripts/protein_feature_analysis.py` to see which features activate on a known protein of interest:

```bash
python scripts/protein_feature_analysis.py \
    --protein_id P12345 \
    --sae_dir trained_saes/best_progen_large_24 \
    --embed_dir data/embeddings/progen2_large/layer_24 \
    --annot_dir data/annotations \
    --top_k 30 \
    --sort_by specificity
```

This prints per-feature activation statistics and, if `--annot_dir` is provided, overlaps with known annotations. The `specificity` sort order prioritises features that fire rarely across the dataset (specific features) over features that fire on most proteins (global features).

**Local vs global features — why it matters for steering:**

- **Local/specific features** (low dataset frequency, fires in <10% of proteins): these tend to encode a precise biological function or structural motif. Steering them strongly biases generation toward that motif. They are the best targets for condition-driven generation.
- **Global/general features** (fires in >50% of proteins): these often encode broad sequence properties (overall hydrophobicity, charge pattern, amino acid composition). Steering them may shift gross sequence statistics rather than functional identity. You can see this in the output: global features appear in both TP and noTP groups in anti-target analysis.

Check `Per_feature_statistics.yaml` for each feature's `mean_activation`, `fraction_nonzero`, and `max_activation`. A feature with `fraction_nonzero < 0.05` is a good specific candidate; one with `fraction_nonzero > 0.5` is likely global and less useful as a steering target.

---

## 4. Running Steering Experiments

### 4a. Single-feature baseline

```bash
python scripts/activation_steering_naive.py \
    --sae_dir trained_saes/best_progen_large_24 \
    --feature_id 5900 \
    --clamp_value 5.0 \
    --mode clamp \
    --steering_method direct \
    --prefix M \
    --n_sequences 512 \
    --batch_size 256 \
    --max_new_tokens 30 \
    --seed 42 \
    --output_fasta outputs/f5900x5.0.fasta
```

`--clamp_value 5.0` means 5× the feature's max training activation. The FASTA headers record per-sequence NLL and perplexity under the unsteered model.

### 4b. Multi-feature steering

Use `--features` to steer multiple features simultaneously. Positive values amplify; negative values suppress:

```bash
python scripts/activation_steering_naive.py \
    --sae_dir trained_saes/best_progen_large_24 \
    --features "5900:5.0" "3884:10.0" "8227:7.0" "8839:-1.0" \
    --mode clamp \
    --steering_method direct \
    --prefix M \
    --n_sequences 512 \
    --batch_size 256 \
    --max_new_tokens 30 \
    --seed 42 \
    --output_fasta outputs/combo.fasta
```

Feature IDs must be plain integers (no `f` prefix). Clamp values can be negative.

### 4c. Choosing clamp values

| value | meaning |
|---|---|
| 0.0 | ablation (same as `--mode ablate`) |
| 1.0× | max observed activation during SAE training |
| 2–10× | extrapolation; effective for most features without degeneration |
| >15× | risk of degeneration — check NLL and fragment rate |
| negative | subtracts the feature's direction from the hidden state; suppresses its contribution |

**Practical approach:** start at 2× and double until either the condition rate plateaus or fragment rate rises above ~8%. The plateau tells you the feature is saturated; the fragment rise tells you degeneration is starting.

### 4d. Steering method: `direct` vs `with_error`

- `direct`: the full hidden state is replaced by SAE decode of the modified activations. Removes the SAE residual (reconstruction error). Simpler and more interpretable.
- `with_error`: SAE residual is added back after decode. More faithful to normal model operation.

Use `direct` for exploratory steering and benchmarking. The difference is usually small at moderate clamp values.

### 4e. Output files

By default, FASTAs go to `steering_outputs/` with an auto-generated filename encoding all parameters. Use `--output_fasta` to specify an explicit path. Each FASTA header encodes:
```
>steered_N|features_...|clamp|direct|value_...|nll=X.XX|ppl=Y.YY
```

---

## 5. Evaluating Results

**Define a condition evaluator.** For transit peptides we use TargetP 2.0:
```bash
targetp -fasta outputs/combo.fasta -org non-pl -stdout > outputs/combo_summary.targetp2
```
For other conditions, any sequence classifier (HMMer, a fine-tuned classifier, ESMFold secondary structure, etc.) can serve this role. The output just needs to label each generated sequence as passing/failing.

**Quality metrics — always check these:**

| metric | how to compute | threshold for concern |
|---|---|---|
| NLL | from FASTA headers (`nll=` field) | >4.5 vs baseline ~2.8 |
| Fragment rate | TargetP "Probable protein fragment" flag, or directly `(length <= 5).mean()` | >10% |
| Mean sequence length | from sequences | should stay near `max_new_tokens` cap |

The NLL in the FASTA header is computed under the **unsteered** model. A small increase (~0.3 nats) is normal for functional steering; doubling (~+3 nats) signals degenerate outputs. Fragment rate is the most immediate signal: if it exceeds ~10%, reduce the clamp value.

---

## 6. Iterative Feature Discovery via Anti-Target Analysis

This is the most powerful part of the workflow. Once you have a baseline steering condition, you can mine your own generated sequences to find additional features that distinguish successful from failed generations.

**The idea:** embed generated sequences through the same layer used by the SAE, encode with the SAE, compare feature activations between the "passing" and "failing" subsets. Features enriched in the passing group are candidates for additional positive steering; features enriched in the failing group are candidates for negative steering.

### Step 1: Generate sequences with an initial condition

```bash
python scripts/activation_steering_naive.py ... --output_fasta outputs/round1.fasta
```

### Step 2: Run your condition evaluator

```bash
targetp -fasta outputs/round1.fasta -org non-pl -stdout > outputs/round1_summary.targetp2
# or substitute your own evaluator, producing a file that labels each sequence
```

### Step 3: Run anti-target analysis

`scripts/find_antitarget_features.py` takes the FASTA and its labels, embeds both groups, and reports feature enrichment statistics:

```bash
python scripts/find_antitarget_features.py \
    --fasta outputs/round1.fasta \
    --targetp outputs/round1_summary.targetp2 \
    --sae_dir trained_saes/best_progen_large_24 \
    --top_k 20 \
    --stat max
```

Output includes:
- **Top anti-condition features** (enriched in failing sequences): `mean_failing`, `mean_passing`, `diff`, `%fire_failing`, `%fire_passing`, `frac_diff`
- **Top pro-condition features** (enriched in passing sequences): same columns
- **Max activations** for top features (for choosing clamp values)

**Reading the output:**
- `frac_diff` = `%fire_passing − %fire_failing`. Large positive means strongly enriched in passing (pro-condition). Large negative means enriched in failing (anti-condition).
- Features where both groups fire at 100% are global and useless for steering — skip them.
- Focus on features where `frac_diff > 30%` and the passing fire rate is high (>60%).

### Step 4: Add new features and repeat

Take the strongest pro-condition features from step 3 and add them at 2–3× to your existing combo. Re-generate, re-evaluate, re-run anti-target analysis. Each round typically squeezes out 2–5 percentage points.

**When negative steering helps:** features enriched in the failing group are *candidates* for suppression, but not all are useful. A feature may be enriched in failures because it is downstream of the failure, not upstream. Test negative steering empirically (at −1× to −2×) and check whether condition rate improves. If it drops, the feature is a consequence; skip it. If it improves, it is a genuine upstream driver.

---

## 7. Practical Tips

**GPU allocation:** with two A100s, run positive-feature experiments on GPU 0 and negative/ablation experiments on GPU 1 in parallel using `CUDA_VISIBLE_DEVICES=0/1`.

**Working directory:** always run scripts from `/scratch/tmp/hrbanh/InterPLM`. Relative paths in all scripts assume this root. The `cd` command persists in the Bash session — if you `cd` into a subdirectory for `targetp`, subsequent Python calls will fail. Use absolute paths for TargetP:
```bash
BASE=/scratch/tmp/hrbanh/InterPLM/steering_experiments/my_experiment
targetp -fasta $BASE/fasta/run1.fasta -org non-pl -stdout > $BASE/targetp/run1_summary.targetp2
```

**Batch size vs n_sequences:** `--batch_size 256` is efficient on A100 for progen2-large at 30 tokens. `--n_sequences 512` gives enough sequences for TargetP statistics to be meaningful (±3% noise at 512).

**Seed reproducibility:** fix `--seed 42` across all conditions to control for sampling variance.

**Degeneration check without an evaluator:** if you want a quick sanity check without running a condition evaluator, inspect the FASTA directly:
```python
# Quick degeneration check from FASTA
import re, numpy as np
from pathlib import Path

seqs, nlls = [], []
with open("outputs/run.fasta") as f:
    seq = ""
    for line in f:
        if line.startswith(">"):
            if seq: seqs.append(seq); seq = ""
            m = re.search(r'nll=([0-9.]+)', line)
            if m: nlls.append(float(m.group(1)))
        else:
            seq += line.strip()
    if seq: seqs.append(seq)

lengths = np.array([len(s) for s in seqs])
print(f"NLL: {np.mean(nlls):.3f}  frag(≤5): {(lengths<=5).mean():.1%}  mean_len: {lengths.mean():.1f}")
```

NLL >4.5 or frag rate >15% are reliable signals of degeneration.

---

## 8. Full Workflow Summary

```
1. Query concept_f1_scores.csv  →  candidate feature IDs ranked by f1_per_domain
2. Check feature specificity     →  Per_feature_statistics.yaml, fraction_nonzero
3. Check max_act                 →  sae.activation_rescale_factor[fid].item()
4. Generate baseline             →  activation_steering_naive.py, mode=none
5. Test single features          →  clamp at 2×, 5×, 10×; check condition rate + NLL + frag
6. Build multi-feature combo     →  add features greedily; each addition should improve
7. Anti-target analysis          →  find_antitarget_features.py on current best output
8. Add new pro-condition features and test negative suppression of anti-condition features
9. Repeat from step 7 until gains diminish (<1pp per round)
10. Record NLL and fragment rates for all conditions as quality evidence
```
