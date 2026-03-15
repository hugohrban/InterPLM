#!/bin/bash
set -e  # Exit on any error

# Full InterPLM Walkthrough Script for ProGen2
# This script runs all steps from the README walkthrough using the ProGen2 model.
# It uses smaller dataset sizes for faster testing.
#
# All intermediate data goes under $INTERPLM_DATA/walkthrough_progen2/
# Steps are idempotent via checkpoint files stored in .checkpoints/
#
# IMPORTANT: Run this script with the interplm conda environment activated:
#   conda activate interplm
#   export INTERPLM_DATA=/path/to/data
#   bash examples/run_full_walkthrough_progen.sh

echo "============================================"
echo "InterPLM Full Walkthrough Test (ProGen2)"
echo "============================================"
echo ""

# Check conda environment
if ! python -c "import tap" 2>/dev/null; then
    echo "Error: interplm conda environment not activated"
    echo "Please run:"
    echo "  conda activate interplm"
    echo "  bash examples/run_full_walkthrough_progen.sh"
    exit 1
fi

# Check required environment variables
if [ -z "$INTERPLM_DATA" ]; then
    echo "Error: INTERPLM_DATA environment variable not set"
    echo "Please run: export INTERPLM_DATA=/path/to/data"
    exit 1
fi

if [ -z "$LAYER" ]; then
    echo "Warning: LAYER not set, using default LAYER=4"
    export LAYER=4
fi

# ============================================
# CHECKPOINT MECHANISM
# ============================================

BASE="$INTERPLM_DATA" # /walkthrough_progen2"
CHECKPOINT_DIR="$BASE/.checkpoints"
mkdir -p "$CHECKPOINT_DIR"

step_done() { [ -f "$CHECKPOINT_DIR/$1" ]; }
mark_done() { date > "$CHECKPOINT_DIR/$1"; echo "  [checkpoint saved: $1]"; }

# ============================================
# HYPERPARAMETERS - Configure walkthrough here
# ============================================

# Training data parameters
NUM_PROTEINS=2000
PROTEINS_PER_SHARD=500

# Protein embedder parameters (ProGen2)
EMBEDDER_TYPE="progen2"
MODEL_NAME="hugohrban/progen2-small"
BATCH_SIZE=8

# Swiss-Prot annotation parameters
ANNOTATION_INPUT="${INTERPLM_DATA}/annotations/uniprotkb/proteins.tsv.gz"
MIN_REQUIRED_INSTANCES=25
N_SHARDS=6

# Evaluation set parameters
MIN_AA_PER_CONCEPT=1000
MIN_DOMAINS_PER_CONCEPT=25
VALID_SHARD_START=0; VALID_SHARD_END=$((N_SHARDS/2 - 1)); TEST_SHARD_START=$((N_SHARDS/2)); TEST_SHARD_END=$((N_SHARDS - 1))

# Output directories
MODEL_DIR="models/walkthrough_model_progen"
RESULTS_DIR="results/valid_counts"
DASHBOARD_NAME="walkthrough_progen2"

echo "Configuration:"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo "  Python: $(which python)"
echo "  INTERPLM_DATA: $INTERPLM_DATA"
echo "  BASE: $BASE"
echo "  LAYER: $LAYER"
echo ""
echo "Hyperparameters:"
echo "  NUM_PROTEINS: $NUM_PROTEINS"
echo "  PROTEINS_PER_SHARD: $PROTEINS_PER_SHARD"
echo "  EMBEDDER: $EMBEDDER_TYPE ($MODEL_NAME)"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  N_SHARDS: $N_SHARDS"
echo "  MODEL_DIR: $MODEL_DIR"
echo ""

# ============================================
# Step 0: Setup
# ============================================

if step_done "step_0_setup"; then
    echo "Step 0: Setup directories (already done, skipping)"
else
    echo "Step 0: Setup directories"
    mkdir -p $BASE/{uniprot,uniprot_shards,eval_shards,walkthrough_progen/training_embeddings,walkthrough_progen/eval_shards,analysis_embeddings,annotations}
    echo "✓ Directories created"
    mark_done "step_0_setup"
fi
echo ""

# ============================================
# Step 1: Data preparation
# ============================================

# Step 1a: Download sequences
if step_done "step_1a_download"; then
    echo "Step 1a: Download Swiss-Prot sequences (already done, skipping)"
else
    UNIPROT_FILE="$BASE/uniprot/uniprot_sprot.fasta.gz"
    if [ ! -f "$UNIPROT_FILE" ]; then
        echo "Step 1a: Download Swiss-Prot sequences"
        wget -P $BASE/uniprot/ https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
        echo "✓ Downloaded Swiss-Prot"
    fi
    mark_done "step_1a_download"
fi
echo ""

# Step 1b: Subset FASTA
if step_done "step_1b_subset"; then
    echo "Step 1b: Create protein subset (already done, skipping)"
else
    echo "Step 1b: Create protein subset ($NUM_PROTEINS proteins for testing)"
    python scripts/subset_fasta.py \
        --input_file $BASE/uniprot/uniprot_sprot.fasta.gz \
        --output_file $BASE/uniprot/subset.fasta \
        --num_proteins $NUM_PROTEINS
    echo "✓ Subset created"
    mark_done "step_1b_subset"
fi
echo ""

# Step 1c: Shard FASTA
if step_done "step_1c_shard"; then
    echo "Step 1c: Shard FASTA files (already done, skipping)"
else
    echo "Step 1c: Shard FASTA files ($PROTEINS_PER_SHARD proteins/shard)"
    python scripts/shard_fasta.py \
        --input_file $BASE/uniprot/subset.fasta \
        --output_dir $BASE/uniprot_shards/ \
        --proteins_per_shard $PROTEINS_PER_SHARD
    echo "✓ FASTA sharded"
    mark_done "step_1c_shard"
fi
echo ""

# Step 1d: Set aside eval shard
if step_done "step_1d_eval_shard"; then
    echo "Step 1d: Move shard_0 to eval (already done, skipping)"
else
    echo "Step 1d: Move shard_0 to eval"
    mkdir -p $BASE/walkthrough_progen/eval_shards/
    mv $BASE/uniprot_shards/shard_0.fasta $BASE/walkthrough_progen/eval_shards/
    echo "✓ Eval shard separated"
    mark_done "step_1d_eval_shard"
fi
echo ""

# Step 1e: Extract embeddings for training
if step_done "step_1e_embeddings"; then
    echo "Step 1e: Extract training embeddings (already done, skipping)"
else
    echo "Step 1e: Extract training embeddings (layer $LAYER)"
    python scripts/extract_embeddings.py \
        --fasta_dir $BASE/uniprot_shards/ \
        --output_dir $BASE/walkthrough_progen/training_embeddings/progen2_small/ \
        --embedder_type $EMBEDDER_TYPE \
        --model_name $MODEL_NAME \
        --layers $LAYER \
        --batch_size $BATCH_SIZE
    echo "✓ Training embeddings extracted"
    mark_done "step_1e_embeddings"
fi
echo ""

# ============================================
# Step 2: Train SAE
# ============================================

if step_done "step_2_train_sae"; then
    echo "Step 2: Train basic SAE (already done, skipping)"
else
    echo "Step 2: Train basic SAE (this will take a few minutes)"
    INTERPLM_DATA=$BASE python examples/train_basic_sae_progen.py
    echo "✓ SAE trained"
    mark_done "step_2_train_sae"
fi
echo ""

# ============================================
# Step 3: Concept analysis setup
# ============================================

echo "Step 3: Setup concept analysis"
echo ""

# Step 3a: Extract annotations
if step_done "step_3a_annotations"; then
    echo "Step 3a: Extract UniProtKB annotations (already done, skipping)"
else
    echo "Step 3a: Extract UniProtKB annotations"
    python -m interplm.analysis.concepts.extract_annotations \
        --input_uniprot_path $ANNOTATION_INPUT \
        --output_dir $BASE/annotations/uniprotkb/processed \
        --n_shards $N_SHARDS \
        --min_required_instances $MIN_REQUIRED_INSTANCES \
        --overwrite
    echo "✓ Annotations extracted"
    mark_done "step_3a_annotations"
fi
echo ""

# Step 3b: Embed annotations
if step_done "step_3b_embed_annotations"; then
    echo "Step 3b: Extract embeddings for annotated proteins (already done, skipping)"
else
    echo "Step 3b: Extract embeddings for annotated proteins"
    python scripts/embed_annotations.py \
        --input_dir $BASE/annotations/uniprotkb/processed/ \
        --output_dir $BASE/analysis_embeddings/progen2_small/layer_$LAYER \
        --embedder_type $EMBEDDER_TYPE \
        --model_name $MODEL_NAME \
        --layer $LAYER \
        --batch_size $BATCH_SIZE
    echo "✓ Annotation embeddings extracted"
    mark_done "step_3b_embed_annotations"
fi
echo ""

# Step 3c: Normalize SAE
if step_done "step_3c_normalize"; then
    echo "Step 3c: Normalize SAE (already done, skipping)"
else
    echo "Step 3c: Normalize SAE"
    python -m interplm.sae.normalize \
        --sae_dir $MODEL_DIR/layer_$LAYER \
        --aa_embds_dir $BASE/analysis_embeddings/progen2_small/layer_$LAYER
    echo "✓ SAE normalized"
    mark_done "step_3c_normalize"
fi
echo ""

# Step 3d: Create evaluation sets
if step_done "step_3d_eval_sets"; then
    echo "Step 3d: Create validation and test sets (already done, skipping)"
else
    echo "Step 3d: Create validation and test sets"
    python -m interplm.analysis.concepts.prepare_eval_set \
        --valid_shard_range $VALID_SHARD_START $VALID_SHARD_END \
        --test_shard_range $TEST_SHARD_START $TEST_SHARD_END \
        --uniprot_dir $BASE/annotations/uniprotkb/processed \
        --min_aa_per_concept $MIN_AA_PER_CONCEPT \
        --min_domains_per_concept $MIN_DOMAINS_PER_CONCEPT
    echo "✓ Eval sets created"
    mark_done "step_3d_eval_sets"
fi
echo ""

# Step 3e: Compare activations
if step_done "step_3e_compare_activations"; then
    echo "Step 3e: Compare feature activations to concepts (already done, skipping)"
else
    echo "Step 3e: Compare feature activations to concepts (validation set only)"
    python -m interplm.analysis.concepts.compare_activations \
        --sae_dir $MODEL_DIR/layer_$LAYER \
        --aa_embds_dir $BASE/analysis_embeddings/progen2_small/layer_$LAYER \
        --eval_set_dir $BASE/annotations/uniprotkb/processed/valid/ \
        --output_dir $RESULTS_DIR/
    echo "✓ Activations compared"
    mark_done "step_3e_compare_activations"
fi
echo ""

# Step 3f: Calculate F1 scores
if step_done "step_3f_f1_scores"; then
    echo "Step 3f: Calculate F1 scores (already done, skipping)"
else
    echo "Step 3f: Calculate F1 scores"
    python -m interplm.analysis.concepts.calculate_f1 \
        --eval_res_dir $RESULTS_DIR \
        --eval_set_dir $BASE/annotations/uniprotkb/processed/valid/
    echo "✓ F1 scores calculated"
    mark_done "step_3f_f1_scores"
fi
echo ""

# ============================================
# Step 4: Collect feature activations
# ============================================

if step_done "step_4_collect_activations"; then
    echo "Step 4: Collect feature activations for dashboard (already done, skipping)"
else
    echo "Step 4: Collect feature activations for dashboard"
    python scripts/collect_feature_activations.py \
        --sae_dir $MODEL_DIR/layer_$LAYER/ \
        --embeddings_dir $BASE/analysis_embeddings/progen2_small/layer_$LAYER \
        --metadata_dir $BASE/annotations/uniprotkb/processed \
        --shard_range $VALID_SHARD_START $TEST_SHARD_END
    echo "✓ Feature activations collected"
    mark_done "step_4_collect_activations"
fi
echo ""

# ============================================
# Step 5: Create dashboard
# ============================================

if step_done "step_5_dashboard"; then
    echo "Step 5: Create dashboard cache (already done, skipping)"
else
    echo "Step 5: Create dashboard cache"
    python scripts/create_dashboard.py \
        --sae_path $MODEL_DIR/layer_$LAYER/ae.pt \
        --embeddings_dir $BASE/analysis_embeddings/progen2_small/layer_$LAYER \
        --layer $LAYER \
        --metadata_path $ANNOTATION_INPUT \
        --dashboard_name $DASHBOARD_NAME \
        --model_name $EMBEDDER_TYPE \
        --model_type $MODEL_NAME \
        --shard_range $VALID_SHARD_START $TEST_SHARD_END \
        --concept_enrichment_path $RESULTS_DIR/concept_f1_scores.csv
    echo "✓ Dashboard created"
    mark_done "step_5_dashboard"
fi
echo ""

echo "============================================"
echo "✅ Full Walkthrough Complete!"
echo "============================================"
echo ""
echo "Results saved to:"
echo "  - Base directory: $BASE"
echo "  - Trained SAE: $MODEL_DIR/layer_$LAYER/ae.pt"
echo "  - Evaluation: $BASE/evaluation_results.yaml"
echo "  - Concept F1: $RESULTS_DIR/concept_f1_scores.csv"
echo "  - Dashboard cache: $BASE/dashboard_cache/$DASHBOARD_NAME/"
echo ""
echo "To view the dashboard, run:"
echo "  streamlit run interplm/dashboard/app.py -- --cache_dir $BASE/dashboard_cache/$DASHBOARD_NAME"
echo ""
echo "To reset and re-run all steps, remove checkpoints:"
echo "  rm -rf $CHECKPOINT_DIR/*"
echo ""
