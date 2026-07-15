# Changelog (fork additions)

Covers commits authored on top of the upstream `InterPLM` fork point
(`47d9d78`, first ProGen2 commit) through `HEAD`. Ordered by significance:
major features first, then bug fixes, then usability/convenience polish.

## Major New Features

- **ProGen2 support end-to-end**: new ProGen2 embedder registered as a PLM
  backend alongside ESM-2 (`interplm/embedders/progen2.py`,
  `interplm/embedders/__init__.py`), plus ProGen-specific fidelity
  computation and activation intervention
  (`interplm/train/fidelity.py`, `interplm/sae/intervention.py`,
  `interplm/analysis/concepts/extract_annotations.py`,
  `scripts/evaluate_sae.py`), and support for the large ProGen2 checkpoint
  (`interplm/embedders/progen2.py`, `scripts/run_concept_analysis.py`).
  `47d9d78` `27f6602` `54db295`
- **Vectorized concept metrics + parallelized annotation extraction**:
  replaced per-concept loops with a single matmul for metric calculation
  and switched shard processing to `ProcessPoolExecutor`
  (`interplm/analysis/concepts/compare_activations.py`,
  `interplm/analysis/concepts/extract_annotations.py`,
  `interplm/embedders/esm.py`). Took the concept-analysis pipeline from
  ~24 hours to ~20 minutes in practice — the single biggest practical
  speedup in the fork.
  `c097eb3`
- **Multi-GPU, resumable embedding/annotation extraction pipeline**:
  sharded work across all available GPUs via `ProcessPoolExecutor`,
  auto-halved batch size on OOM and retried, skipped shards already
  processed, and kept the embedder resident in worker processes instead
  of reloading per shard (`scripts/extract_embeddings.py`,
  `scripts/embed_annotations.py`). Also added single-pass multi-layer
  extraction instead of one forward pass per layer
  (`scripts/embed_annotations.py`, `interplm/embedders/esm.py`,
  `interplm/embedders/progen2.py`).
  `274f8e5` `40d919e` `bb92ee9` `40e4f43` `14fb8d9`
- **CLI-driven SAE training** (`scripts/train_sae.py`): trains any of the
  four SAE variants (relu/topk/batch_topk/jump_relu) via `tapify`, with a
  `--dry_run` profiling mode and auto-generated hyperparameter-encoded
  save directories.
  `bdc1ebb` `17ce727`
- **Rank-based concept evaluation framework**: scores SAE features against
  a biological concept via F1-guided ranking or annotation-enrichment
  (`interplm/analysis/concepts/rank_eval.py`, `scripts/quick_rank_eval.py`,
  `scripts/rank_eval_concepts.py`), with a precomputed sufficient-statistics
  cache so repeat concept queries are instant instead of rerunning a full
  GPU pass (`interplm/analysis/concepts/rank_eval.py`,
  `scripts/rank_eval_concepts.py`).
  `c25425b` `88f2950` `42e7049`
- **Concept Explorer & Protein Zoom dashboard modes**: a searchable
  dashboard view of SAE features by biological concept, backed by rank
  eval results (`interplm/dashboard/concept_explorer.py`,
  `interplm/dashboard/app.py`, `interplm/dashboard/dashboard_cache.py`,
  `interplm/dashboard/feature_activation_vis.py`,
  `scripts/create_dashboard.py`), plus a "Protein Zoom" page comparing
  per-residue activation profiles of chosen features on a chosen protein,
  with cross-navigation between dashboard pages
  (`interplm/dashboard/protein_zoom.py`, `interplm/dashboard/app.py`,
  `interplm/dashboard/concept_explorer.py`,
  `interplm/dashboard/feature_activation_vis.py`,
  `interplm/analysis/per_protein_analysis.py`).
  `704854a` `0da9f36` `e57320d`
- **Per-protein SAE feature analysis module**: identifies which features
  fire in a given protein, classifies them by cross-dataset specificity
  and within-protein activation pattern, and reports concept-annotation
  overlap, with a lazy shard index for fast lookup over large embedding
  sets (`interplm/analysis/per_protein_analysis.py`,
  `scripts/protein_feature_analysis.py`).
  `963a8bd`
- **Activation steering for SAE feature intervention**: hooks a PLM layer
  during generation, edits an SAE feature (clamp/ablate), and decodes back
  (direct or with-error reconstruction); later extended with multi-feature
  steering, batched generation, hook verification, KV-cache generation,
  and prefix-skip steering so only newly generated tokens are steered
  (`scripts/activation_steering_naive.py`,
  `.claude/skills/steer-protein/SKILL.md`).
  `bcc8d31` `1d4033d` `ca4341c` `e91fc6d`
- **Anti-target feature discovery + steering evaluation tooling**: finds
  SAE features enriched in negative vs. positive examples (e.g.
  no-transit-peptide vs. transit-peptide sequences) using TargetP2,
  ps_scan, or SignalP6 as label sources (`scripts/find_antitarget_features.py`),
  plus MobiDB-lite-based evaluation of steered outputs — disorder
  fraction, compositional bias, sequence diversity stats
  (`scripts/eval_mobidb_steering.py`) — and a written skill guide for the
  steering workflow (`.claude/skills/steer-protein/SKILL.md`).
  `70f9ad4` `d14f3e2` `9cab7e1` `cfb4665` `5387e00`
- **ESM-2 vs. ProGen2 comparison research tooling**: systematic
  concept-coverage comparison — precision/recall/F1/AUROC across
  checkpoints (`scripts/compare_esm_progen.py`) — and a positional
  analysis testing whether causal attention anchors N-terminal features
  more sharply at sequence start (`scripts/positional_analysis.py`).
  `f12c6d2` `6fae729`
- **Consolidated concept-analysis metrics & reporting**: in-sample
  validation metrics (best F1 per concept without cross-split selection)
  and a richer report — F1 distribution, valid→test F1 gap, concept
  coverage, polysemanticity, Spearman rank stability, JSON summary
  (`interplm/analysis/concepts/report_metrics.py`,
  `scripts/run_concept_analysis.py`) — plus aggregation of per-SAE
  metrics into one consolidated results CSV
  (`scripts/run_concept_analysis.py`).
  `8a9443c` `a8f53db` `b0a8e91`
- **Dead-neuron resampling on by default + full-batch fidelity checks**:
  changed SAE training defaults to resample dead neurons and run fidelity
  evaluation across all batches after training instead of a subset
  (`interplm/train/training_run.py`, `scripts/train_sae.py`).
  `3ba23f9`

## Bug Fixes

- Fixed categorical annotation parsing to use the correct per-feature
  separator field — `Binding site` and `Cofactor` columns were silently
  zeroed out because the parser hardcoded `/note=` when those feature
  types actually use `/ligand=` and `/Name=`
  (`interplm/analysis/concepts/extract_annotations.py`,
  `interplm/analysis/concepts/parsing_utils.py`).
  `f6cf69d`
- Fixed a threshold-count mismatch in F1 calculation (hardcoded 10
  thresholds vs. the actual 5 used when generating shard counts), which
  caused an out-of-bounds index
  (`interplm/analysis/concepts/calculate_f1.py`,
  `interplm/analysis/concepts/compare_activations.py`).
  `8ad8513`
- Fixed SAE features not being normalized before thresholding in concept
  activation comparison (`interplm/analysis/concepts/compare_activations.py`).
  `a4b4604`
- Fixed a dtype cast bug in dead-neuron resampling weights
  (`interplm/train/trainers/relu.py`).
  `2de1c52`
- Fixed fidelity batch sampling to use a batch count instead of a
  training-step interval, and a related numpy-array slicing `TypeError`
  (`interplm/train/fidelity.py`).
  `7f3009c`
- Reverted an accidental regression where note normalization had become
  dead code, restoring trailing-instance-index stripping (e.g. "Sushi 1"
  → "Sushi") (`interplm/analysis/concepts/parsing_utils.py`).
  `c4b3718`
- Fixed the AFDB structure download to use the current prediction API and
  model v6 after the old endpoint broke (`interplm/dashboard/view_structures.py`).
  `03c298e`
- Fixed a broken ProGen2 model reference by prefixing it with the
  `hugohrban/` HF namespace (`scripts/protein_feature_analysis.py`).
  `43403e7`
- Fixed Streamlit warnings and broken walkthrough/tutorial steps
  (`interplm/dashboard/app.py`, `interplm/sae/intervention.py`,
  `interplm/train/fidelity.py`).
  `5213ee0` `c04eef9` `801531c`

## Usability / Convenience Improvements

- Replaced noisy per-shard progress bars with a single unified bar
  (`interplm/analysis/per_protein_tracking.py`).
  `22a769f`
- Made the dashboard load its embedder lazily on first use instead of at
  startup (`interplm/dashboard/app.py`).
  `ea47198`
- Auto-discover all shards in an embeddings directory when none are
  explicitly specified (`scripts/collect_feature_activations.py`).
  `9e5ad02`
- Auto-detect the SAE class from checkpoint keys instead of requiring it
  to be specified (`scripts/create_dashboard.py`).
  `889b5e0`
- `evaluate_sae` now accepts a directory of FASTA files (not just a
  single file); fidelity parses FASTA format directly via BioPython
  (`scripts/evaluate_sae.py`, `interplm/train/fidelity.py`).
  `8135ff1`
- Added `force_include_patterns` to guarantee specific annotation types
  (e.g. rare domains) bypass min-count thresholds and are always included
  in eval subsets / annotation extraction
  (`interplm/analysis/concepts/prepare_eval_set.py`,
  `interplm/analysis/concepts/extract_annotations.py`).
  `9b84a65` `e7de635`
- Normalized trailing instance indices in UniProtKB domain notes so
  repeated instances of a domain are grouped together
  (`interplm/analysis/concepts/parsing_utils.py`).
  `fcb9bc7`
- Centralized default threshold percentages into a single shared
  constant instead of duplicating the list across callers
  (`interplm/analysis/concepts/concept_constants.py`,
  `interplm/analysis/concepts/calculate_f1.py`,
  `interplm/analysis/concepts/compare_activations.py`).
  `a28e7e1`
- Increased the dead-feature threshold to 20M tokens (fewer
  false-positive "dead" features on longer runs)
  (`interplm/train/trainers/batch_top_k.py`).
  `46e41c1`
- Added a `verbose` flag to suppress per-token progress bars during
  batched steering generation (`scripts/activation_steering_naive.py`).
  `e91fc6d`
- Cleaned up progress reporting and debug noise: tqdm progress and
  silenced debug prints (`interplm/sae/normalize.py`,
  `interplm/train/configs.py`).
  `b566b99`
