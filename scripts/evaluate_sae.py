#!/usr/bin/env python
"""
Evaluate a trained SAE on held-out protein sequences.

This script calculates three key metrics:
1. Reconstruction quality (variance explained, MSE)
2. Sparsity (L0, feature activation statistics)
3. Downstream task fidelity (loss recovered on masked language modeling)

Usage:
    # Single FASTA file
    python scripts/evaluate_sae.py \
        --sae_path trained_saes/pretrained_esm2-650M-layer24/ae_normalized.pt \
        --fasta_file data/uniref50_subset500k_shards/test/shard_0.fasta \
        --model_name esm2_t33_650M_UR50D \
        --layer 24

    # Directory of FASTA files (all *.fasta files are used)
    python scripts/evaluate_sae.py \
        --sae_path trained_saes/pretrained_esm2-650M-layer24/ae_normalized.pt \
        --fasta_file data/uniref50_subset500k_shards/test \
        --model_name esm2_t33_650M_UR50D \
        --layer 24
"""

import tempfile
from pathlib import Path
from typing import Optional
import yaml
import torch
from tqdm import tqdm

from interplm.sae.inference import load_sae
from interplm.embedders import get_embedder
from interplm.train.fidelity import ESMFidelityConfig, ProGenFidelityConfig
from interplm.utils import get_device


def calculate_reconstruction_metrics(sae, embeddings):
    """Calculate reconstruction quality metrics.

    Args:
        sae: Trained SAE model
        embeddings: Original embeddings tensor [n_tokens, d_model]

    Returns:
        dict: Reconstruction metrics
    """
    with torch.no_grad():
        reconstructed = sae(embeddings)

        # MSE
        mse = torch.mean((embeddings - reconstructed) ** 2).item()

        # Variance explained
        total_variance = torch.var(embeddings, dim=0).sum()
        residual_variance = torch.var(embeddings - reconstructed, dim=0).sum()
        variance_explained = (1 - residual_variance / total_variance).item()

        # Per-dimension MSE statistics
        per_dim_mse = torch.mean((embeddings - reconstructed) ** 2, dim=0)

    return {
        "mse": mse,
        "variance_explained": variance_explained,
        "per_dim_mse_mean": per_dim_mse.mean().item(),
        "per_dim_mse_std": per_dim_mse.std().item(),
        "per_dim_mse_max": per_dim_mse.max().item(),
    }


def calculate_sparsity_metrics(sae, embeddings):
    """Calculate sparsity metrics.

    Args:
        sae: Trained SAE model
        embeddings: Original embeddings tensor [n_tokens, d_model]

    Returns:
        dict: Sparsity metrics
    """
    with torch.no_grad():
        features = sae.encode(embeddings)   # [n_tokens, sae.dict_size]

        # L0 sparsity (average number of active features per token)
        l0_sparsity = (features != 0).float().sum(dim=-1).mean().item()

        # Feature activation frequency (% of tokens that activate each feature)
        feature_active = (features != 0).float().mean(dim=0)

        # Dead features (never activate)
        dead_features = (feature_active == 0).sum().item()

        # Highly active features (activate on >50% of tokens)
        highly_active = (feature_active > 0.5).sum().item()

    return {
        "l0_sparsity": l0_sparsity,
        "l0_sparsity_pct": (l0_sparsity / sae.dict_size) * 100,
        "dead_features": dead_features,
        "dead_features_pct": (dead_features / sae.dict_size) * 100,
        "highly_active_features": highly_active,
        "highly_active_pct": (highly_active / sae.dict_size) * 100,
        "mean_feature_activation_freq": feature_active.mean().item() * 100,
    }


def extract_embeddings_from_fastas(fasta_files, embedder, layer, max_proteins=None):
    """Extract embeddings from one or more FASTA files.

    Args:
        fasta_files: Single Path or list of Paths to FASTA files
        embedder: Protein embedder instance
        layer: Layer to extract
        max_proteins: Maximum total proteins to process (default: all)

    Returns:
        torch.Tensor: Concatenated embeddings [n_tokens, d_model]
    """
    from Bio import SeqIO

    if isinstance(fasta_files, (str, Path)):
        fasta_files = [Path(fasta_files)]

    all_embeddings = []
    n_proteins = 0
    n_skipped = 0

    for fasta_file in fasta_files:
        if max_proteins is not None and n_proteins >= max_proteins:
            break

        with open(fasta_file) as f:
            for record in SeqIO.parse(f, "fasta"):
                if max_proteins is not None and n_proteins >= max_proteins:
                    break

                sequence = str(record.seq)
                embeddings = embedder.embed_single_sequence(sequence, layer=layer)

                if not isinstance(embeddings, torch.Tensor):
                    embeddings = torch.from_numpy(embeddings)

                if torch.isnan(embeddings).any():
                    n_skipped += 1
                    continue

                all_embeddings.append(embeddings)
                n_proteins += 1

    all_embeddings = torch.cat(all_embeddings, dim=0)

    print(f"Extracted {all_embeddings.shape[0]:,} tokens from {n_proteins} proteins across {len(fasta_files)} file(s)")
    if n_skipped > 0:
        print(f"  (Skipped {n_skipped} proteins with NaN embeddings)")

    return all_embeddings


def evaluate_sae(
    sae_path: str,
    fasta_file: Path,
    model_name: str,
    layer: int,
    output_file: Optional[Path] = None,
    skip_fidelity: bool = False,
    fidelity_batch_size: int = 8,
    max_proteins: Optional[int] = None,
    device: Optional[str] = None,
):
    """
    Evaluate SAE on held-out protein sequences.

    Args:
        sae_path: Path to SAE model (local path or hf://org/model:layer_N)
        fasta_file: Path to a FASTA file, or a directory containing *.fasta files
        model_name: ESM model name (e.g., esm2_t6_8M_UR50D)
        layer: Layer index to evaluate
        output_file: Path to save results (default: print to stdout)
        skip_fidelity: Skip downstream task fidelity evaluation (faster)
        fidelity_batch_size: Batch size for fidelity evaluation (default: 8)
        max_proteins: Maximum number of proteins to evaluate (default: all)
        device: Device for embedder (cpu/cuda/mps, default: auto-detect)
    """
    # Resolve fasta_file to a list of paths
    fasta_path = Path(fasta_file)
    if fasta_path.is_dir():
        fasta_files = sorted(fasta_path.glob("*.fasta"))
        if not fasta_files:
            raise FileNotFoundError(f"No .fasta files found in directory: {fasta_path}")
        print(f"Found {len(fasta_files)} FASTA file(s) in {fasta_path}")
    else:
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        fasta_files = [fasta_path]

    print("=" * 70)
    print("SAE Evaluation Script")
    print("=" * 70)
    print(f"SAE: {sae_path}")
    print(f"FASTA: {fasta_file} ({len(fasta_files)} file(s))")
    print(f"Model: {model_name}")
    print(f"Layer: {layer}")
    print()

    # Load SAE
    device_str = get_device()
    print(f"Loading SAE on device: {device_str}")

    # Parse sae_path - could be directory or .pt file
    sae_path_obj = Path(sae_path)
    if sae_path_obj.is_file():
        # If it's a .pt file, use the parent directory
        sae_dir = sae_path_obj.parent
        model_name_sae = sae_path_obj.name
    else:
        # If it's a directory, use default model name
        sae_dir = sae_path_obj
        model_name_sae = "ae.pt"

    sae = load_sae(sae_dir, model_name=model_name_sae, device=device_str)

    print(f"SAE loaded: {sae.dict_size} features, {sae.activation_dim}D embeddings")
    print()

    # Load embedder
    print("Loading protein embedder...")
    embedder_type = "esm"
    if "progen" in model_name:
        embedder_type="progen"

    # MPS has known issues with ESM-2 producing NaN embeddings in loops
    # Force CPU for embedder if MPS is detected and user didn't specify device
    embedder_device = device if device else device_str
    if embedder_device == "mps" and device is None:
        print("⚠️  WARNING: MPS device detected. ESM-2 has known NaN issues on MPS.")
        print("   Using CPU for embedder instead (SAE will still use MPS).")
        embedder_device = "cpu"

    embedder = get_embedder(
        embedder_type,
        model_name=f"{model_name}",
        device=embedder_device
    )
    print(f"Embedder loaded: {embedder_type} on {embedder_device}")
    print()

    # Extract embeddings
    embeddings = extract_embeddings_from_fastas(
        fasta_files,
        embedder,
        layer,
        max_proteins=max_proteins,
    )

    embeddings = embeddings.to(device_str)

    # Calculate metrics
    results = {
        "sae_path": str(sae_path),
        "fasta_input": str(fasta_file),
        "n_fasta_files": len(fasta_files),
        "model_name": model_name,
        "layer": layer,
        "n_tokens": embeddings.shape[0],
        "sae_dict_size": sae.dict_size,
        "sae_activation_dim": sae.activation_dim,
    }

    # 1. Reconstruction metrics
    print("\n" + "=" * 70)
    print("1. RECONSTRUCTION QUALITY")
    print("=" * 70)
    recon_metrics = calculate_reconstruction_metrics(sae, embeddings)
    results["reconstruction"] = recon_metrics

    for key, value in recon_metrics.items():
        print(f"  {key}: {value:.6f}")

    # 2. Sparsity metrics
    print("\n" + "=" * 70)
    print("2. SPARSITY")
    print("=" * 70)
    sparsity_metrics = calculate_sparsity_metrics(sae, embeddings)
    results["sparsity"] = sparsity_metrics

    for key, value in sparsity_metrics.items():
        if "pct" in key or "freq" in key:
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value}")

    # 3. Fidelity metrics
    if not skip_fidelity:
        print("\n" + "=" * 70)
        print("3. DOWNSTREAM TASK FIDELITY")
        print("=" * 70)
        print("This measures how well the SAE preserves model's ability to predict")
        print("masked tokens. Higher is better (100% = perfect preservation).")
        print()

        from Bio import SeqIO

        # Fidelity needs a single sequence file. Write a temp combined FASTA
        # when the input is a directory, reuse the file directly otherwise.
        _tmp_fasta = None
        if len(fasta_files) == 1:
            fidelity_seq_path = fasta_files[0]
        else:
            _tmp_fasta = tempfile.NamedTemporaryFile(
                mode="w", suffix=".fasta", delete=False
            )
            n_written = 0
            for fasta_f in fasta_files:
                for record in SeqIO.parse(fasta_f, "fasta"):
                    if max_proteins is not None and n_written >= max_proteins:
                        break
                    _tmp_fasta.write(f">{record.id}\n{record.seq}\n")
                    n_written += 1
                if max_proteins is not None and n_written >= max_proteins:
                    break
            _tmp_fasta.close()
            fidelity_seq_path = Path(_tmp_fasta.name)
            print(f"  Combined {n_written} sequences into temp file for fidelity")

        try:
            fidelity_config_cls = ESMFidelityConfig if embedder_type == "esm" else ProGenFidelityConfig
            fidelity_config = fidelity_config_cls(
                eval_seq_path=fidelity_seq_path,
                model_name=model_name,
                layer_idx=layer,
                eval_batch_size=fidelity_batch_size,
            )

            fidelity_eval = fidelity_config.build()
            fidelity_metrics = fidelity_eval._calculate_fidelity(sae)
            results["fidelity"] = fidelity_metrics

            for key, value in fidelity_metrics.items():
                if "pct" in key:
                    print(f"  {key}: {value:.2f}%")
                else:
                    print(f"  {key}: {value:.6f}")
        finally:
            if _tmp_fasta is not None:
                import os
                os.unlink(_tmp_fasta.name)
    else:
        print("\n(Skipping fidelity evaluation)")

    # Save or print results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        print(f"Results saved to: {output_file}")
    else:
        print("\nFull results:")
        print(yaml.dump(results, default_flow_style=False))

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    from tap import tapify
    tapify(evaluate_sae)
