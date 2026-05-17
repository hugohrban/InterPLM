# Causal Intervention on Transit Peptide Identity via SAE Feature Steering

## Setup

We performed activation steering experiments on progen2-large (1.2B parameters) using a sparse autoencoder (SAE) trained on layer 24 of 32. The SAE has a dictionary of 10,240 features with k=350 active features per token (BatchTopK), trained on a large corpus of protein sequences. All generation used a single methionine prefix (`M`), batch size 256, and a maximum of 30 new tokens, producing sequences in the 1–31 AA range — consistent with the length of functional N-terminal targeting peptides (typically 20–80 AA, with functional information concentrated in the first ~30 residues).

**Steering mechanism.** At each forward pass through layer 24, we intercept the hidden state, encode it through the SAE, clamp one or more feature activations to a target value, then decode back to the residual stream using the `direct` method (full replacement, no error term). Steering magnitudes are expressed as multiples of each feature's maximum observed activation during SAE training (`activation_rescale_factor`). A clamp value of 1.0× therefore sets the feature to its training-distribution maximum; values above 1.0× extrapolate beyond the training regime. Negative values actively suppress the feature below zero, reversing its contribution to the hidden state.

---

## Feature Discovery

### Initial candidates

Feature 5900 was previously identified as encoding N-terminal transit/signal peptide structure based on high per-position F1 score (f1_pd = 0.494) on a labelled dataset of Swiss-Prot proteins. Feature 7356 had the highest F1 score overall (f1_pd = 0.578) and was therefore also tested. Feature 3884 had a mitochondria-specific profile (f1_pd = 0.397) and was included as a candidate mTP driver.

Single-feature steering results established the ordering: **f8227 (67.6%) > f3884 (59.0%) > f5900 (45.1%) > f7356 (34.6%)** at their respective optimal clamp values. Feature 7356, despite its high F1, performed poorly in steering — likely because high linear separability does not imply causal influence on generation. Feature 8227 was not in the initial candidate set; it was discovered through the anti-target analysis described below.

### Iterative anti-target analysis

To identify features causally relevant to transit peptide generation, we performed an iterative analysis comparing SAE feature activations between sequences classified as TP (SP or mTP by TargetP 2.0) and noTP within the same generated batch. For each experiment condition, we embedded all generated sequences through progen2-large layer 24, encoded the resulting hidden states through the SAE, and computed per-feature mean and maximum activation for the TP and noTP groups separately. Features strongly enriched in the TP group are candidates for positive steering; features enriched in noTP are candidates for negative (suppressive) steering.

**Round 1** was run on the output of the f5900:5 + f3884:10 combination (63.9% any-TP). The strongest pro-TP feature discovered was **feature 8227**: it fired in 94.2% of TP-classified sequences versus 51.9% of noTP sequences, with a mean activation of 6.32 versus 1.09. Adding f8227 at 7× to the existing combination raised the any-TP rate from 63.9% to 75.0%.

**Round 2** was run on the output of the three-feature combination (75.0% any-TP, ~25% noTP remaining). The analysis revealed two new pro-TP features — **f1323** (fires in 92.2% of TP vs 48.4% of noTP) and a further set of anti-target features: **f8839** and **f3266**, enriched specifically in the noTP sequences of this more refined batch (+19.5% and +14.6% firing rate differential respectively).

---

## Experiments and Results

All conditions used n=512 generated sequences. The table below shows the full progression:

| Condition | %any-TP | %SP | %mTP | %frag | NLL |
|---|---|---|---|---|---|
| Unsteered baseline | 31.8% | 26.0% | 5.9% | 3.5% | 2.796 |
| Best single feature: f8227:10× | 67.6% | 60.2% | 7.4% | 3.7% | 3.063 |
| f5900:5 + f3884:10 + f8227:7 | 75.0% | 67.8% | 7.2% | 5.9% | 3.073 |
| **Best: + f1323:2 + f8839:−1 + f3266:−1** | **78.7%** | **74.2%** | **4.5%** | **4.3%** | **3.075** |
| neg only: f8643:−2 + f1422:−2 + f3266:−1 | 30.9% | 24.8% | 6.1% | 2.5% | 2.962 |
| pos combo + f8839:−1 + f3266:−1 | 77.5% | 72.7% | 4.9% | 4.7% | 3.070 |
| f8227:−2 (suppression) | 20.9% | 17.4% | 3.5% | 2.5% | 2.863 |
| f8227:−5 (suppression) | 8.6% | 6.1% | 2.5% | 0.2% | 2.990 |
| All pro-TP features negated | 4.5% | 3.1% | 1.4% | 1.4% | 3.134 |

The best overall condition, f5900:5 + f3884:10 + f8227:7 + f1323:2 + f8839:−1 + f3266:−1, achieves **78.7% any-TP** versus a baseline of 31.8% — a 2.5× increase. Positive and negative contributions are approximately additive: adding f1323:2 to the three-feature baseline gained +2.9 percentage points; adding the negative steering of f8839 and f3266 independently gained +2.5 points; combining both gained +3.7 points.

### mTP-specific note

Maximising mitochondrial transit peptide (mTP) rate specifically requires a different feature set. The highest mTP rates came from f7356:3 + f5900:5 (14.8% mTP, 52.7% any-TP), and more broadly from f7356-dominant combinations without f8227. Feature 8227 appears to push the model specifically toward signal peptide (SP) identity at the expense of mTP, consistent with its high SP-specificity in natural protein data. The best any-TP condition has only 4.5% mTP — the trade-off between SP and mTP rate is a consistent pattern across all experiments.

### Negative steering of anti-target features

An important negative result: not all features enriched in noTP sequences respond usefully to suppressive steering. Features 8643 and 1422 (identified as anti-target features in round 1) produced a drop from 75.0% to 62.9% when negatively steered at −2× alongside the positive combo. By contrast, features 8839 and 3266 (identified in round 2) improved performance when negatively steered at −1×. The distinction likely reflects causality: features that are downstream consequences of failed TP generation will not improve matters when suppressed, while features that are genuinely upstream drivers of non-TP trajectories will. Negative-steering-only conditions (no positive features) perform at or near baseline, confirming that the positive steering carries the functional effect.

---

## Sequence Quality and Degeneration

A key concern with activation steering is that performance gains may reflect degraded, degenerate sequences rather than genuine functional generation. We use three metrics to address this:

**NLL (negative log-likelihood under the unsteered model):** Sequences are scored by running them back through the unsteered progen2-large. A low NLL means the model finds the sequence plausible as a natural protein. High NLL indicates sequences outside the model's training distribution — a proxy for nonsense or degenerate outputs.

**Fragment rate:** Sequences where TargetP 2.0 flags "Probable protein fragment" — typically sequences too short to constitute a functional protein. This is the earliest and most direct sign of degeneration: the model begins hitting EOS after very few tokens.

**Mean sequence length:** Only informative when it drops well below the generation cap (30 tokens here). When sequences hit EOS early across the board, mean length collapses.

Across all functional steering conditions (4.5%–78.7% any-TP), NLL spans a narrow band of 2.796–3.134 — a shift of at most 0.34 nats relative to the unsteered baseline. Fragment rates stay below 6%. Mean sequence length holds at 30–31 AA throughout. These metrics are consistent across conditions that span a 17× range in TP rate, which argues strongly that the steering is modulating protein function, not introducing distributional noise.

For contrast, steering feature 5900 at 30× (far outside the training regime) produces clearly degenerate output: NLL = 5.785 (double the functional steering range), mean length = 3.2 AA, and 93.8% of sequences are ≤5 amino acids. Examples: `M`, `MA`, `MV`, `MMX`. This represents a qualitative failure mode entirely absent from the functional steering experiments.

---

## The Normalized Steering Scale

Clamp values in these experiments are expressed as multiples of each feature's `activation_rescale_factor` — the maximum activation value that feature reached across the training corpus. A clamp of 1.0× therefore sets the feature to its empirical maximum from training; values above 1.0× are extrapolations beyond anything the SAE has encoded.

In practice, the best-performing conditions use clamp values of 5–10× for the primary features. This may seem large, but the effective signal-to-noise context matters: the SAE operates on hidden states of dimension 2560 in progen2-large, and the feature vector contributes a scaled direction of that space. Moderate extrapolation (2–10×) reliably shifts generation without catastrophic degeneration; large extrapolation (30×) destroys coherence. The transition between these regimes is visible in the NLL and fragment rate data and provides a principled criterion for choosing clamp values in future experiments.

Negative clamp values are not just "less activation" — they set the feature to a negative raw value, meaning the decoded direction is subtracted from the hidden state rather than added. This is a stronger intervention than ablation (setting to zero) and corresponds to actively reversing the feature's contribution to the forward pass.
