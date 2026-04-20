"""
Compares each feature to each concept across all proteins in an evaluation set. Calculates
metrics for each shard individualy (tp, fp, tp_per_domain) and saves the results to disk.
These metrics need to be combined across all shards to get the final metrics for the evaluation set.

Because the number of comparisons can become very large, and both the feature activations
and the concept labels are sparse, we use sparse matrix operations to calculate the metrics
and this really speeds things up.

However, the neuron activations are actually quite dense so running the sparse calculations
on neurons disguising as SAE features via identity SAEs is quite slow. To address that, we
use a dense implementation for the neuron activations that can be set via the is_sparse flag.
"""

from email.policy import default
import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from scipy import sparse
from tqdm import tqdm

from interplm.analysis.concepts.concept_constants import (
    is_aa_level_concept,
    default_thresholds_percent
)
from interplm.sae.dictionary import Dictionary
from interplm.sae.inference import get_sae_feats_in_batches, load_sae
from interplm.data_processing.embedding_loader import load_shard_embeddings


def count_unique_nonzero_sparse(
    matrix: Union[np.ndarray, sparse.spmatrix],
) -> List[int]:
    """
    Count unique non-zero values in each column of a sparse matrix.

    Args:
        matrix: Input matrix, either as a NumPy array or a SciPy sparse matrix.
               Will be converted to sparse CSC format if not already sparse.

    Returns:
        List of integers where each element represents the count of unique
        non-zero values in the corresponding column.
    """
    # Convert input to CSC (Compressed Sparse Column) format if not already sparse
    if not sparse.issparse(matrix):
        matrix = sparse.csc_matrix(matrix)
    else:
        # Ensure matrix is in CSC format for efficient column access
        matrix = matrix.tocsc()

    unique_counts = []
    # Iterate through each column
    for i in range(matrix.shape[1]):
        # Extract the current column
        col = matrix.getcol(i)
        # Count unique values:
        # 1. Get the non-zero data values in the column
        # 2. Convert to set to get unique values
        # 3. Subtract 1 if 0 is present in the data
        # Note: col.data contains only explicitly stored values
        unique_counts.append(len(set(col.data)) - (0 in col.data))

    return unique_counts


def calc_metrics_sparse(
    sae_feats_sparse: sparse.spmatrix,
    per_token_labels_sparse: sparse.spmatrix,
    threshold_percents: List[float],
    is_aa_level_concept_list: List[bool],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate various metrics for sparse feature matrices across different thresholds.

    Args:
        sae_feats_sparse: Sparse matrix of features (samples x features)
        per_token_labels_sparse: Sparse matrix of labels (samples x concepts)
        threshold_percents: List of threshold values to evaluate
        is_aa_level_concept_list: Boolean flags indicating if each concept is AA-level

    Returns:
        Tuple containing:
        - tp: True positives array (concepts x features x thresholds)
        - fp: False positives array (concepts x features x thresholds)
        - tp_per_domain: True positives per domain array (concepts x features x thresholds)
    """
    _, n_features = sae_feats_sparse.shape
    n_concepts = per_token_labels_sparse.shape[1]
    n_thresholds = len(threshold_percents)

    tp = np.zeros((n_concepts, n_features, n_thresholds))
    fp = np.zeros((n_concepts, n_features, n_thresholds))
    tp_per_domain = np.zeros((n_concepts, n_features, n_thresholds))

    sae_feats_csr = sae_feats_sparse.tocsr()
    labels_csc = per_token_labels_sparse.tocsc()
    # Binary label matrix for vectorised tp/fp: (n_tokens, n_concepts)
    labels_binary = (per_token_labels_sparse > 0).astype(np.float32)
    non_aa_indices = [i for i, v in enumerate(is_aa_level_concept_list) if not v]

    for threshold_idx, threshold in enumerate(threshold_percents):
        feats_bin = sae_feats_csr.copy()
        feats_bin.data = (feats_bin.data > threshold).astype(np.float32)
        feats_bin.eliminate_zeros()

        # Single matmul replaces the per-concept loop for tp and fp.
        # labels_binary.T: (n_concepts, n_tokens) @ feats_bin: (n_tokens, n_features)
        # → (n_concepts, n_features)
        tp_mat = (labels_binary.T @ feats_bin).toarray()
        tp[:, :, threshold_idx] = tp_mat

        # fp = tokens where feature fires but concept is absent
        total_active = np.asarray(feats_bin.sum(axis=0))  # (1, n_features)
        fp[:, :, threshold_idx] = total_active - tp_mat

        # tp_per_domain only needed for non-AA-level concepts
        for concept_idx in non_aa_indices:
            tp_per_domain[concept_idx, :, threshold_idx] = (
                count_unique_nonzero_sparse(feats_bin.multiply(labels_csc[:, concept_idx]))
            )

    return tp, fp, tp_per_domain


def count_unique_nonzero_dense(matrix: torch.Tensor) -> List[int]:
    """
    Count unique non-zero values in each column of a dense matrix.

    Args:
        matrix: Dense PyTorch tensor to analyze

    Returns:
        List of counts of unique non-zero values for each column
    """
    # Initialize list to store counts
    unique_counts = []

    # Iterate through each column
    for col in range(matrix.shape[1]):
        # Get unique values in the column
        unique_values = torch.unique(matrix[:, col])
        # Count how many unique values are non-zero
        count = torch.sum(unique_values != 0).item()
        unique_counts.append(count)

    return unique_counts


def calc_metrics_dense(
    sae_feats: torch.Tensor,
    per_token_labels_sparse: Union[np.ndarray, sparse.spmatrix],
    threshold_percents: List[float],
    is_aa_level_concept: List[bool],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate metrics for dense feature matrices

    Args:
        sae_feats: Dense tensor of features from SAE
        per_token_labels_sparse: Label matrix in sparse format
        threshold_percents: List of threshold values to evaluate
        is_aa_level_concept: Boolean flags indicating if each concept is AA-level

    Returns:
        Tuple of numpy arrays (tp, fp, tp_per_domain) containing metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_feats = sae_feats.to(device)
    per_token_labels = torch.tensor(
        per_token_labels_sparse.astype(np.float32), device=device
    )

    # Get dimensions from input tensors
    _, n_features = sae_feats.shape
    n_concepts = per_token_labels.shape[1]
    n_thresholds = len(threshold_percents)

    per_feat_adjusted_thresholds = torch.tensor(
        threshold_percents, dtype=torch.float32, device=device
    )

    tp = torch.zeros((n_concepts, n_features, n_thresholds), device=device)
    fp = torch.zeros((n_concepts, n_features, n_thresholds), device=device)
    tp_per_domain = torch.zeros((n_concepts, n_features, n_thresholds), device=device)

    labels_binary = (per_token_labels > 0).float()  # (n_tokens, n_concepts)
    non_aa_indices = [i for i, v in enumerate(is_aa_level_concept) if not v]

    for threshold_idx in range(n_thresholds):
        threshold = per_feat_adjusted_thresholds[threshold_idx]
        sae_feats_binarized = (sae_feats > threshold).float()

        # Single matmul replaces the per-concept loop for tp and fp.
        # labels_binary.T: (n_concepts, n_tokens) @ sae_feats_binarized: (n_tokens, n_features)
        # → (n_concepts, n_features)
        tp_mat = labels_binary.T @ sae_feats_binarized
        tp[:, :, threshold_idx] = tp_mat

        total_active = sae_feats_binarized.sum(dim=0, keepdim=True)  # (1, n_features)
        fp[:, :, threshold_idx] = total_active - tp_mat

        for concept_idx in non_aa_indices:
            concept_labels = per_token_labels[:, concept_idx].unsqueeze(1)
            tp_per_domain[concept_idx, :, threshold_idx] = torch.tensor(
                count_unique_nonzero_dense(sae_feats_binarized * concept_labels),
                device=device,
            )

    return tp.cpu().numpy(), fp.cpu().numpy(), tp_per_domain.cpu().numpy()


def process_shard(
    sae: Dictionary,
    device: torch.device,
    aa_embeddings: torch.Tensor,
    per_token_labels: Union[np.ndarray, sparse.spmatrix],
    threshold_percents: List[float],
    is_aa_concept_list: List[bool],
    token_chunk_size: int = 4096,
    # kept for API compatibility, no longer used
    feat_chunk_max: int = 512,
    is_sparse: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a shard by iterating over token batches rather than feature chunks.

    Each token batch runs the SAE once (all features) and immediately accumulates
    tp/fp via GPU dense matmul — eliminating the previous 40× redundant SAE passes
    and CPU sparse conversions.

    Args:
        sae: Normalized SAE model
        device: PyTorch device
        aa_embeddings: Amino acid embeddings (n_tokens, d_model)
        per_token_labels: Label matrix (n_tokens, n_concepts)
        threshold_percents: Activation thresholds to evaluate
        is_aa_concept_list: True for AA-level concepts (no domain counting needed)
        token_chunk_size: Tokens processed per iteration (tune to GPU memory)
        feat_chunk_max: Unused, kept for backward compatibility
        is_sparse: Unused, kept for backward compatibility

    Returns:
        Tuple of (tp, fp, tp_per_domain), each (n_concepts, n_features, n_thresholds)
    """
    if isinstance(aa_embeddings, dict) and "embeddings" in aa_embeddings:
        aa_embeddings = aa_embeddings["embeddings"]

    aa_embeddings = aa_embeddings.to(device)

    n_tokens = aa_embeddings.shape[0]
    n_features = sae.dict_size
    n_concepts = per_token_labels.shape[1]
    n_thresholds = len(threshold_percents)

    print(f"{n_tokens=} tokens, {n_features=} features, {n_concepts=} concepts")

    # Labels on GPU as dense float — (n_tokens, n_concepts)
    if sparse.issparse(per_token_labels):
        labels_gpu = torch.tensor(
            per_token_labels.toarray(), dtype=torch.float32, device=device
        )
    else:
        labels_gpu = torch.tensor(
            per_token_labels.astype(np.float32), device=device
        )
    labels_binary = (labels_gpu > 0).float()  # (n_tokens, n_concepts)

    thresholds = torch.tensor(threshold_percents, dtype=torch.float32, device=device)

    # Accumulators on GPU
    tp = torch.zeros((n_concepts, n_features, n_thresholds), device=device)
    total_active = torch.zeros((n_features, n_thresholds), device=device)

    # For tp_per_domain: per non-AA concept, track which domain IDs each feature catches.
    # caught_ever[c][t_idx][d, f] = True if feature f fired on any token in domain d.
    non_aa_indices = [i for i, v in enumerate(is_aa_concept_list) if not v]
    domain_info: dict = {}  # concept_idx → (unique_domains, caught_ever)
    for c_idx in non_aa_indices:
        labels_col = labels_gpu[:, c_idx]
        unique_domains = torch.unique(labels_col[labels_col > 0])
        if len(unique_domains) > 0:
            caught_ever = torch.zeros(
                (n_thresholds, len(unique_domains), n_features),
                dtype=torch.bool,
                device=device,
            )
            domain_info[c_idx] = (unique_domains, caught_ever)

    all_features = list(range(n_features))

    with torch.no_grad():
        for start in range(0, n_tokens, token_chunk_size):
        # for start in tqdm(range(0, n_tokens, token_chunk_size), desc="Token chunks"):
            end = min(start + token_chunk_size, n_tokens)
            aa_chunk = aa_embeddings[start:end]  # (chunk, d_model)

            # One SAE pass for all features — replaces the 40-chunk feature loop
            sae_feats = sae.encode_feat_subset(aa_chunk, all_features)  # (chunk, n_features)

            labels_bin_chunk = labels_binary[start:end]  # (chunk, n_concepts)
            labels_chunk = labels_gpu[start:end]          # (chunk, n_concepts)

            for t_idx in range(n_thresholds):
                feats_bin = (sae_feats > thresholds[t_idx]).float()  # (chunk, n_features)

                # (n_concepts, chunk) @ (chunk, n_features) → (n_concepts, n_features)
                tp[:, :, t_idx] += labels_bin_chunk.T @ feats_bin
                total_active[:, t_idx] += feats_bin.sum(dim=0)

                # tp_per_domain: for each non-AA concept, OR in which domains were caught
                for c_idx, (unique_domains, caught_ever) in domain_info.items():
                    labels_col_chunk = labels_chunk[:, c_idx]  # (chunk,)
                    # (chunk, n_domains) — which tokens belong to each domain
                    domain_mask = (
                        labels_col_chunk.unsqueeze(1) == unique_domains.unsqueeze(0)
                    ).float()
                    # (n_domains, chunk) @ (chunk, n_features) → (n_domains, n_features)
                    caught_ever[t_idx] |= (domain_mask.T @ feats_bin) > 0

    # fp = total activations minus true positives
    fp = total_active.unsqueeze(0) - tp  # (n_concepts, n_features, n_thresholds)

    tp_per_domain = torch.zeros((n_concepts, n_features, n_thresholds), device=device)
    for c_idx, (_, caught_ever) in domain_info.items():
        # caught_ever: (n_thresholds, n_domains, n_features) → sum over domains
        # → (n_thresholds, n_features) → transpose → (n_features, n_thresholds)
        tp_per_domain[c_idx] = caught_ever.sum(dim=1).T

    return tp.cpu().numpy(), fp.cpu().numpy(), tp_per_domain.cpu().numpy()


def analyze_concepts(
    sae_dir: Path,
    aa_embds_dir: Path = Path("../../data/processed/embeddings"),
    eval_set_dir: Path = Path("../../data/processed/valid"),
    output_dir: Path = "concept_results",
    threshold_percents: List[float] = default_thresholds_percent,
    shard: int | None = None,
    is_sparse: bool = True,
):
    """
    Analyzes concepts in protein sequences using a Sparse Autoencoder (SAE) model.

    Args:
        sae_dir (Path): Directory containing the normalized SAE model file 'ae_normalized.pt'
        aa_embds_dir (Path, optional): Directory containing amino acid embeddings.
        eval_set_dir (Path, optional): Directory containing validation dataset and metadata.
        output_dir (Path, optional): Directory where results will be saved.
        threshold_percents (List[float], optional): List of threshold values for concept detection.
        shard (int | None): Specific shard number to process. Must exist in evaluation set.
        is_sparse (bool, optional): Whether to use sparse matrix operations.

    Returns:
        None: Results are saved to disk as NPZ file with following arrays:
            - tp: True positives counts
            - fp: False positives counts
            - tp_per_domain: True positives counts per domain

    Raises:
        ValueError: If normalized SAE model is not found in sae_dir
        ValueError: If specified shard is not in the evaluation set
    """

    # Load evaluation set metadata from JSON file
    with open(eval_set_dir / "metadata.json", "r") as f:
        eval_set_metadata = json.load(f)

    # Verify that the normalized SAE model exists
    if not (sae_dir / "ae_normalized.pt").exists():
        raise ValueError(f"Normalized SAE model not found in {sae_dir}")

    # Validate that the specified shard exists in the evaluation set
    if shard not in eval_set_metadata["shard_source"]:
        raise ValueError(f"Shard {shard} is not in this evaluation set")

    # Load concept names and identify amino acid level concepts
    concept_names = load_concept_names(eval_set_dir / "aa_concepts_columns.txt")
    is_aa_concept_list = [
        is_aa_level_concept(concept_name) for concept_name in concept_names
    ]

    # Load and process labels for the specified shard
    per_token_labels = sparse.load_npz(eval_set_metadata["path_to_shards"][str(shard)])
    per_token_labels = per_token_labels[
        :, eval_set_metadata["indices_of_concepts_to_keep"]
    ]

    # Set up device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the normalized SAE model
    sae = load_sae(model_dir=sae_dir, model_name="ae_normalized.pt", device=device)

    # Load embeddings using centralized loader (auto-detects format)
    embeddings = load_shard_embeddings(aa_embds_dir, shard, device=str(device))

    # Extract just the embeddings tensor if it's in dict format
    if isinstance(embeddings, dict) and 'embeddings' in embeddings:
        embeddings = embeddings['embeddings']

    # Process the shard and get results (true positives, false positives, and true positives per domain)
    (tp, fp, tp_per_domain) = process_shard(
        sae,
        device,
        embeddings,
        per_token_labels,
        threshold_percents,
        is_aa_concept_list,
        feat_chunk_max=250,
        is_sparse=is_sparse,
    )

    # Create output directory if it doesn't exist and save results
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"shard_{shard}_counts.npz",
        tp=tp,
        fp=fp,
        tp_per_domain=tp_per_domain,
    )


def analyze_all_shards_in_set(
    sae_dir: Path,
    aa_embds_dir: Path,
    eval_set_dir: Path,
    output_dir: Path = "concept_results",
    threshold_percents: List[float] = default_thresholds_percent,
    is_sparse: bool = True,
):
    """Wrapper to scan calculate metrics across all shards in an evaluation set.

    Args:
        sae_dir (Path): Directory containing the normalized SAE model file 'ae_normalized.pt'
        aa_embds_dir (Path): Directory containing amino acid embeddings.
        eval_set_dir (Path): Directory containing validation dataset and metadata
        output_dir (Path, optional): Directory where results will be saved.
        threshold_percents (List[float], optional): List of threshold values for concept detection.
        is_sparse (bool, optional): Whether to use sparse matrix operations.

    Returns:
        None: Results for each shard are saved to disk in the output_dir

    Raises:
        FileNotFoundError: If metadata.json is not found in eval_set_dir
        ValueError: If any individual shard analysis fails (inherited from analyze_concepts)
    """
    # Load list of shards to evaluate from metadata
    with open(eval_set_dir / "metadata.json", "r") as f:
        shards_to_eval = json.load(f)["shard_source"]
        print(f"Analyzing set {eval_set_dir.stem} with {len(shards_to_eval)} shards")

    # Process each shard sequentially
    for shard in tqdm(shards_to_eval, desc="Processing shards"):
        analyze_concepts(
            sae_dir,
            aa_embds_dir,
            eval_set_dir,
            output_dir,
            threshold_percents,
            shard,
            is_sparse,
        )


def load_concept_names(concept_name_path: Path) -> List[str]:
    """Load concept names from a file."""
    with open(concept_name_path, "r") as f:
        return f.read().split("\n")


if __name__ == "__main__":
    from tap import tapify

    tapify(analyze_all_shards_in_set)

    # Note: If you want to split this up and run each shard individually,
    # you can do so by instead calling:
    # tapify(analyze_concepts)
