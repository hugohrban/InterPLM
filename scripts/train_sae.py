#!/usr/bin/env python
"""
CLI entry point for SAE training.

Supports all four trainer types (relu, topk, batch_topk, jump_relu) and
exposes all meaningful training parameters with sensible defaults.

Usage:
    python scripts/train_sae.py --embeddings_dir data/training_embeddings/esm2_8m/layer_4 \
        --save_dir models/my_sae

    # TopK with explicit k
    python scripts/train_sae.py --embeddings_dir ... --save_dir models/topk \
        --trainer_type topk --k 64 --steps 5000

    # Dry run (20 steps, no model saved)
    python scripts/train_sae.py --embeddings_dir ... --save_dir /tmp/test \
        --dry_run

    # W&B enabled
    python scripts/train_sae.py ... --use_wandb --wandb_entity myorg \
        --wandb_project interplm --wandb_name my-run
"""

import tempfile
from pathlib import Path
from typing import Optional

from tap import tapify

from interplm.train.configs import TrainingRunConfig
from interplm.train.checkpoint_manager import CheckpointConfig
from interplm.train.data_loader import DataloaderConfig
from interplm.train.evaluation import EvaluationConfig
from interplm.train.fidelity import ESMFidelityConfig, ProGenFidelityConfig
from interplm.train.trainers.relu import ReLUTrainerConfig
from interplm.train.trainers.top_k import TopKTrainerConfig
from interplm.train.trainers.batch_top_k import BatchTopKTrainerConfig
from interplm.train.trainers.jump_relu import JumpReLUTrainerConfig
from interplm.train.training_run import SAETrainingRun
from interplm.train.wandb_manager import WandbConfig


def main(
    embeddings_dir: Path,
    save_dir: Path,
    # Architecture
    trainer_type: str = "relu",
    expansion_factor: int = 4,
    # Core training
    steps: int = 10_000,
    batch_size: int = 512,
    lr: Optional[float] = None,
    warmup_steps: Optional[int] = None,
    decay_start: Optional[int] = None,
    seed: int = 0,
    n_shards_to_include: Optional[int] = None,
    dry_run: bool = False,
    # ReLU-specific
    l1_penalty: float = 0.06,
    resample_steps: Optional[int] = None,
    # TopK / BatchTopK-specific
    k: int = 32,
    auxk_alpha: float = 1 / 32,
    # JumpReLU-specific
    target_l0: float = 20.0,
    sparsity_penalty: float = 1.0,
    # Evaluation (all optional)
    eval_seq_path: Optional[Path] = None,
    embedder_type: str = "esm",
    model_name: Optional[str] = None,
    layer_idx: Optional[int] = None,
    # W&B
    use_wandb: bool = False,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    # Checkpointing
    save_steps: int = 2_000,
    max_ckpts_to_keep: int = 1,
):
    """Train a Sparse Autoencoder (SAE) on pre-extracted protein language model embeddings."""

    if dry_run:
        print("[DRY RUN] Overriding steps=20, disabling checkpointing and model save.")
        steps = 20
        save_steps = steps + 1
        max_ckpts_to_keep = 0

    # Build trainer config
    trainer_type = trainer_type.lower()
    if trainer_type == "relu":
        trainer_cfg = ReLUTrainerConfig(
            expansion_factor=expansion_factor,
            lr=lr,
            steps=steps,
            warmup_steps=warmup_steps,
            decay_start=decay_start,
            l1_penalty=l1_penalty,
            resample_steps=resample_steps,
        )
    elif trainer_type == "topk":
        trainer_cfg = TopKTrainerConfig(
            expansion_factor=expansion_factor,
            lr=lr,
            steps=steps,
            warmup_steps=warmup_steps,
            decay_start=decay_start,
            k=k,
            auxk_alpha=auxk_alpha,
        )
    elif trainer_type == "batch_topk":
        trainer_cfg = BatchTopKTrainerConfig(
            expansion_factor=expansion_factor,
            lr=lr,
            steps=steps,
            warmup_steps=warmup_steps,
            decay_start=decay_start,
            k=k,
            auxk_alpha=auxk_alpha,
        )
    elif trainer_type == "jump_relu":
        trainer_cfg = JumpReLUTrainerConfig(
            expansion_factor=expansion_factor,
            lr=lr,
            steps=steps,
            warmup_steps=warmup_steps,
            decay_start=decay_start,
            target_l0=target_l0,
            sparsity_penalty=sparsity_penalty,
        )
    else:
        raise ValueError(
            f"Unknown trainer_type: {trainer_type!r}. "
            "Choose from: relu, topk, batch_topk, jump_relu"
        )

    # Build eval config
    if eval_seq_path is not None and model_name is not None and layer_idx is not None:
        if embedder_type == "esm":
            eval_cfg = ESMFidelityConfig(
                eval_seq_path=eval_seq_path,
                model_name=model_name,
                layer_idx=layer_idx,
            )
        elif embedder_type == "progen2":
            eval_cfg = ProGenFidelityConfig(
                eval_seq_path=eval_seq_path,
                model_name=model_name,
                layer_idx=layer_idx,
            )
        else:
            raise ValueError(
                f"Unknown embedder_type: {embedder_type!r}. Choose from: esm, progen2"
            )
    else:
        eval_cfg = EvaluationConfig()

    # Build remaining configs
    dataloader_cfg = DataloaderConfig(
        plm_embd_dir=embeddings_dir,
        batch_size=batch_size,
        seed=seed,
        n_shards_to_include=n_shards_to_include,
    )

    wandb_cfg = WandbConfig(
        use_wandb=use_wandb,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )

    def _run(actual_save_dir: Path):
        checkpoint_cfg = CheckpointConfig(
            save_dir=actual_save_dir,
            save_steps=save_steps,
            max_ckpts_to_keep=max_ckpts_to_keep,
        )
        config = TrainingRunConfig(
            dataloader_cfg=dataloader_cfg,
            trainer_cfg=trainer_cfg,
            eval_cfg=eval_cfg,
            wandb_cfg=wandb_cfg,
            checkpoint_cfg=checkpoint_cfg,
        )
        SAETrainingRun.from_config(config).run()

    if dry_run:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _run(Path(tmp_dir))
        print("[DRY RUN] No model saved.")
    else:
        _run(Path(save_dir))
        print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    tapify(main)
