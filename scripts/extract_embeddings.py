#!/usr/bin/env python
"""
Extract protein embeddings from FASTA files for SAE training.
Supports multi-GPU distribution, OOM-resilient batch size auto-tuning, and skip logic.
"""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from interplm.embedders import get_embedder


def is_shard_done(fasta_file: Path, output_dir: Path, layers: List[int]) -> bool:
    """Return True if all layer activation files already exist for this shard."""
    return all(
        (output_dir / f"layer_{layer}" / fasta_file.stem / "activations.pt").exists()
        for layer in layers
    )


# Module-level state for worker processes (one embedder per process, loaded once)
_worker_embedder = None
_worker_device = None


def _init_worker(embedder_type: str, model_name: str, gpu_queue) -> None:
    """Initializer run once per worker process. Loads the embedder onto its GPU."""
    import torch
    from interplm.embedders import get_embedder

    global _worker_embedder, _worker_device
    gpu_id = gpu_queue.get()
    _worker_device = f"cuda:{gpu_id}"
    print(f"[worker] Loading {embedder_type} on {_worker_device}...")
    _worker_embedder = get_embedder(embedder_type, model_name=model_name, device=_worker_device)


def _worker(
    fasta_file: Path,
    output_dir: Path,
    layers: List[int],
    batch_size: int,
) -> str:
    """Worker function for multi-GPU processing. Reuses the per-process embedder."""
    import torch

    current_batch_size = batch_size
    while True:
        try:
            _worker_embedder.embed_fasta_file_multiple_layers(
                fasta_file,
                layers=layers,
                output_dir=output_dir,
                batch_size=current_batch_size,
            )
            print(f"[✅] {fasta_file.name} with batch_size={current_batch_size} on {_worker_device}")
            return fasta_file.name
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            if current_batch_size <= 1:
                raise RuntimeError(
                    f"OOM on {fasta_file.name} with batch_size=1 on {_worker_device}"
                )
            current_batch_size = max(1, current_batch_size // 2)
            print(
                f"[OOM] {fasta_file.name} on {_worker_device}: "
                f"retrying with batch_size={current_batch_size}"
            )


def _process_sequential(
    fasta_files: List[Path],
    output_dir: Path,
    embedder_type: str,
    model_name: str,
    layers: List[int],
    batch_size: int,
    device: str,
) -> tuple[int, int]:
    """Process shards sequentially on a single device. Returns (processed, failed)."""
    embedder = get_embedder(embedder_type, model_name=model_name, device=device)
    processed = 0
    failed = 0
    for fasta_file in tqdm(fasta_files, desc="Processing shards", unit="shard"):
        current_batch_size = batch_size
        while True:
            try:
                embedder.embed_fasta_file_multiple_layers(
                    fasta_file,
                    layers=layers,
                    output_dir=output_dir,
                    batch_size=current_batch_size,
                )
                processed += 1
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                    print(f"[FAIL] {fasta_file.name}: {e}")
                    failed += 1
                    break
                torch.cuda.empty_cache()
                if current_batch_size <= 1:
                    print(f"[FAIL] {fasta_file.name}: OOM with batch_size=1 on {device}")
                    failed += 1
                    break
                current_batch_size = max(1, current_batch_size // 2)
                print(
                    f"[OOM] {fasta_file.name} on {device}: "
                    f"retrying with batch_size={current_batch_size}"
                )
    return processed, failed


def main(
    fasta_dir: Path,
    output_dir: Path,
    embedder_type: str = "esm",
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    layers: List[int] = [6],
    batch_size: int = 32,
    shard_index: Optional[int] = None,
    dry_run: bool = False,
):
    """
    Extract protein embeddings from FASTA files for SAE training.

    Args:
        fasta_dir: Directory containing FASTA files (sharded)
        output_dir: Directory to save embeddings
        embedder_type: Type of protein embedder to use (default: esm)
        model_name: Model name/identifier (default: facebook/esm2_t6_8M_UR50D)
        layers: Layers to extract (default: [6])
        batch_size: Batch size for processing. ESM doubles this on GPU internally,
                    so start high and let OOM recovery find the right size.
        shard_index: Optional: process only a specific shard by index (0-based)
        dry_run: If True, process only the first 3 pending shards (useful for
                 finding optimal batch size without running the full job)
    """
    # Discover shards
    fasta_files = sorted(fasta_dir.glob("*.fasta"))
    if not fasta_files:
        fasta_files = sorted(fasta_dir.glob("*.fa"))
    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found in {fasta_dir}")

    # Filter to specific shard if requested
    if shard_index is not None:
        if shard_index < 0 or shard_index >= len(fasta_files):
            raise ValueError(
                f"Shard index {shard_index} out of range. "
                f"Found {len(fasta_files)} shards (0-{len(fasta_files)-1})"
            )
        fasta_files = [fasta_files[shard_index]]
        print(f"Processing only shard {shard_index}: {fasta_files[0].name}")

    # Skip already-done shards
    pending = [f for f in fasta_files if not is_shard_done(f, output_dir, layers)]
    skipped = len(fasta_files) - len(pending)
    print(
        f"{len(fasta_files)} total shards, {skipped} skipped (already done), "
        f"{len(pending)} to process"
    )

    if not pending:
        print("All shards already processed.")
        return

    if dry_run:
        pending = pending[:12]
        print(f"[DRY RUN] Processing only {len(pending)} shard(s) to test batch size")

    print(f"Extracting embeddings for layers: {layers}")
    print(f"Batch size: {batch_size}")

    n_gpus = torch.cuda.device_count()

    # Single GPU or explicit shard_index → sequential in-process
    if n_gpus <= 1 or shard_index is not None:
        device = "cuda:0" if n_gpus >= 1 else "cpu"
        print(f"Device: {device}")
        processed, failed = _process_sequential(
            pending, output_dir, embedder_type, model_name, layers, batch_size, device
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
                executor.submit(_worker, fasta_file, output_dir, layers, batch_size): fasta_file
                for fasta_file in pending
            }
            for future in tqdm(
                as_completed(future_to_shard), total=len(pending), desc="Shards"
            ):
                fasta_file = future_to_shard[future]
                try:
                    future.result()
                    processed += 1
                except Exception as e:
                    print(f"[FAIL] {fasta_file.name}: {e}")
                    failed += 1

    print(
        f"\nDone. Processed: {processed}, Failed: {failed}, Skipped: {skipped}"
    )
    if failed == 0:
        for layer in layers:
            print(f"Layer {layer} embeddings saved to {output_dir / f'layer_{layer}'}")


if __name__ == "__main__":
    from tap import tapify
    tapify(main)
