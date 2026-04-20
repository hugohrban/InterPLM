#!/usr/bin/env python
"""Extract protein embeddings for annotated proteins from UniProtKB."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from tqdm import tqdm

from interplm.embedders import get_embedder


def _output_file(shard_file: Path, output_dir: Path, layer: int, multi_layer: bool) -> Path:
    shard_name = shard_file.parent.name if "shard_" in shard_file.parent.name else shard_file.stem
    if multi_layer:
        return output_dir / f"layer_{layer}" / shard_name / "embeddings.pt"
    return output_dir / shard_name / "embeddings.pt"


def is_shard_done(shard_file: Path, output_dir: Path, layers: List[int]) -> bool:
    multi = len(layers) > 1
    return all(_output_file(shard_file, output_dir, layer, multi).exists() for layer in layers)


def _load_shard(shard_file: Path, sequence_column: str) -> tuple[list, list]:
    """Load sequences and protein IDs from a shard file."""
    df = pd.read_csv(shard_file, sep="\t" if shard_file.suffix == ".tsv" else ",")

    seq_col = next(
        (col for col in df.columns if col.lower() == sequence_column.lower()), None
    )
    if seq_col is None:
        raise ValueError(
            f"Column '{sequence_column}' not found in {shard_file}. "
            f"Available columns: {list(df.columns)}"
        )

    sequences = df[seq_col].tolist()
    if "Entry" in df.columns:
        protein_ids = df["Entry"].tolist()
    elif df.index.name:
        protein_ids = df[seq_col].index.tolist()
    else:
        protein_ids = list(range(len(sequences)))

    return sequences, protein_ids


def _embed_and_save(
    embedder,
    shard_file: Path,
    output_dir: Path,
    layers: List[int],
    batch_size: int,
    sequence_column: str,
) -> None:
    sequences, protein_ids = _load_shard(shard_file, sequence_column)
    print(f"\nProcessing {shard_file.name} with {len(sequences)} sequences, layers={layers}")

    multi = len(layers) > 1
    result = embedder.extract_embeddings_with_boundaries_multiple_layers(
        sequences, layers=layers, batch_size=batch_size
    )
    boundaries = result["boundaries"]
    embeddings_by_layer: dict = result["embeddings"]  # Dict[int, Tensor]

    for layer, embeddings in embeddings_by_layer.items():
        output_file = _output_file(shard_file, output_dir, layer, multi)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings": embeddings,
                "boundaries": boundaries,
                "protein_ids": protein_ids,
            },
            output_file,
        )
        print(f"Saved layer {layer} embeddings to {output_file}")


# Module-level state for worker processes (one embedder per process, loaded once)
_worker_embedder = None
_worker_device = None


def _init_worker(embedder_type: str, model_name: str, gpu_queue) -> None:
    global _worker_embedder, _worker_device
    gpu_id = gpu_queue.get()
    _worker_device = f"cuda:{gpu_id}"
    print(f"[worker] Loading {embedder_type} on {_worker_device}...")
    _worker_embedder = get_embedder(embedder_type, model_name=model_name, device=_worker_device)


def _worker(
    shard_file: Path,
    output_dir: Path,
    layers: List[int],
    batch_size: int,
    sequence_column: str,
) -> str:
    current_batch_size = batch_size
    while True:
        try:
            _embed_and_save(_worker_embedder, shard_file, output_dir, layers, current_batch_size, sequence_column)
            print(f"[✅] {shard_file.name} with batch_size={current_batch_size} on {_worker_device}")
            return shard_file.name
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            if current_batch_size <= 1:
                raise RuntimeError(f"OOM on {shard_file.name} with batch_size=1 on {_worker_device}")
            current_batch_size = max(1, current_batch_size // 2)
            print(f"[OOM] {shard_file.name} on {_worker_device}: retrying with batch_size={current_batch_size}")


def _process_sequential(
    shard_files: List[Path],
    output_dir: Path,
    embedder_type: str,
    model_name: str,
    layers: List[int],
    batch_size: int,
    sequence_column: str,
    device: str,
) -> tuple[int, int]:
    embedder = get_embedder(embedder_type, model_name=model_name, device=device)
    processed = 0
    failed = 0
    for shard_file in tqdm(shard_files, desc="Processing shards", unit="shard"):
        current_batch_size = batch_size
        while True:
            try:
                _embed_and_save(embedder, shard_file, output_dir, layers, current_batch_size, sequence_column)
                processed += 1
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                    print(f"[FAIL] {shard_file.name}: {e}")
                    failed += 1
                    break
                torch.cuda.empty_cache()
                if current_batch_size <= 1:
                    print(f"[FAIL] {shard_file.name}: OOM with batch_size=1 on {device}")
                    failed += 1
                    break
                current_batch_size = max(1, current_batch_size // 2)
                print(f"[OOM] {shard_file.name} on {device}: retrying with batch_size={current_batch_size}")
    return processed, failed


def embed_annotations(
    input_dir: Path,
    output_dir: Path,
    embedder_type: str = "esm",
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    layers: List[int] = [3],
    batch_size: int = 8,
    sequence_column: str = "sequence",
):
    """
    Extract PLM embeddings for proteins with annotations.

    Args:
        input_dir: Directory containing annotation CSV files (shard_*.csv)
        output_dir: Directory to save embeddings
        embedder_type: Type of protein embedder to use (default: esm)
        model_name: Model name/identifier (default: facebook/esm2_t6_8M_UR50D)
        layers: Layer(s) to extract embeddings from. Single layer saves to
                output_dir/shard_N/embeddings.pt (backward compatible).
                Multiple layers save to output_dir/layer_N/shard_N/embeddings.pt.
        batch_size: Batch size for processing (default: 8)
        sequence_column: Name of the column containing sequences (default: sequence)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover shard files
    shard_files = sorted(input_dir.glob("shard_*/protein_data.tsv"))
    if not shard_files:
        shard_files = sorted(input_dir.glob("shard_*.csv"))
    if not shard_files:
        shard_files = sorted(input_dir.glob("*.csv"))
    if not shard_files:
        raise FileNotFoundError(f"No protein data files found in {input_dir}")

    # Skip already-processed shards
    pending = [f for f in shard_files if not is_shard_done(f, output_dir, layers)]
    skipped = len(shard_files) - len(pending)
    print(
        f"{len(shard_files)} total shards, {skipped} skipped (already done), "
        f"{len(pending)} to process"
    )
    print(f"Extracting layers: {layers}")

    if not pending:
        print("All shards already processed.")
        return

    print(f"Loading {embedder_type} embedder: {model_name}")
    n_gpus = torch.cuda.device_count()

    if n_gpus <= 1:
        device = "cuda:0" if n_gpus == 1 else "cpu"
        print(f"Device: {device}")
        processed, failed = _process_sequential(
            pending, output_dir, embedder_type, model_name, layers, batch_size, sequence_column, device
        )
    else:
        print(f"Using {n_gpus} GPUs (one model loaded per GPU, kept resident)")
        ctx = multiprocessing.get_context("spawn")
        gpu_queue = ctx.Queue()
        for i in range(n_gpus):
            gpu_queue.put(i)
        processed = 0
        failed = 0
        with ProcessPoolExecutor(
            max_workers=n_gpus,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(embedder_type, model_name, gpu_queue),
        ) as executor:
            future_to_shard = {
                executor.submit(_worker, f, output_dir, layers, batch_size, sequence_column): f
                for f in pending
            }
            for future in tqdm(as_completed(future_to_shard), total=len(pending), desc="Shards"):
                shard_file = future_to_shard[future]
                try:
                    future.result()
                    processed += 1
                except Exception as e:
                    print(f"[FAIL] {shard_file.name}: {e}")
                    failed += 1

    print(f"\nDone. Processed: {processed}, Failed: {failed}, Skipped: {skipped}")
    if failed == 0:
        print(f"Embeddings saved to {output_dir}")


if __name__ == "__main__":
    from tap import tapify
    tapify(embed_annotations)
