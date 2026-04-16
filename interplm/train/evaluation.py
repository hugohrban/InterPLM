from dataclasses import dataclass
from pathlib import Path

import torch
import torch as t

from interplm.sae.dictionary import Dictionary
from interplm.train.data_loader import DataloaderConfig, ShardedActivationsDataset


@dataclass
class EvaluationConfig:
    # List of sequences to evaluate fidelity on
    eval_seq_path: Path | None = None
    # Directory of embeddings to calculate eval metrics on
    eval_embd_dir: Path | None = None
    # Batch size for evaluation
    eval_batch_size: int | None = None
    # Steps to evaluate on (reconstruction/sparsity on held-out embeddings)
    eval_steps: int | None = None
    # Steps between fidelity evaluations (nnsight intervention through LM)
    fidelity_steps: int | None = None
    # Number of batches to use per training-time fidelity call (None = all)
    fidelity_n_batches: int | None = None
    # Normalization files for evaluation data (should match training)
    zscore_means_file: Path | None = None
    zscore_vars_file: Path | None = None
    target_dtype: torch.dtype = torch.float32
    device: str | None = None  # Override device; None means auto-detect via get_device()

    def build(self) -> "EvaluationManager":
        return EvaluationManager(self)


class EvaluationManager:
    def __init__(self, eval_config: EvaluationConfig):
        self.config = eval_config
        self.eval_steps = eval_config.eval_steps
        self.fidelity_steps = eval_config.fidelity_steps
        self.fidelity_n_batches = eval_config.fidelity_n_batches
        self.eval_seq_path = eval_config.eval_seq_path
        self.eval_embd_dir = eval_config.eval_embd_dir
        self.eval_batch_size = eval_config.eval_batch_size

        self.eval_activations = (
            DataloaderConfig(
                plm_embd_dir=self.eval_embd_dir,
                batch_size=self.eval_batch_size,
                zscore_means_file=eval_config.zscore_means_file,
                zscore_vars_file=eval_config.zscore_vars_file,
                target_dtype=eval_config.target_dtype,
            ).build()
            if self.eval_embd_dir is not None
            else None
        )

    def _calculate_fidelity(self, sae_model, use_all_batches: bool = False):
        """By default, we don't calculate fidelity (subclass should override)"""
        return None

    def _should_run_evals_on_valid(self, step):
        return (
            self.eval_embd_dir is not None
            and self.eval_steps is not None
            and step % self.eval_steps == 0
        )

    def _should_run_fidelity(self, step):
        return self.fidelity_steps is not None and step % self.fidelity_steps == 0

    def calculate_monitoring_metrics(
        self,
        features: t.Tensor,
        activations: t.Tensor,
        reconstructions: t.Tensor,
        sae_model: Dictionary,
    ):
        metrics = {
            "performance/variance_explained": self._calculate_variance_explained(
                activations, reconstructions
            ),
            "performance/mse": self._calculate_mse(activations, reconstructions),
        }

        # Only include l0 sparsity for non-BatchTopK SAEs (where it's meaningful)
        if not hasattr(sae_model, 'k'):
            metrics["performance/l0_sparsity"] = self._calculate_sparsity(features)

        metrics.update(self._calculate_feature_stats(features, sae_model))

        return metrics

    def _calculate_mse(self, activations, reconstructions):
        return t.mean((activations - reconstructions) ** 2).item()

    def _calculate_sparsity(self, features):
        n_nonzero_per_example = (features != 0).float().sum(dim=-1)
        return n_nonzero_per_example.mean().item()

    def _calculate_feature_stats(self, features, sae_model):
        """Dead features and activation frequency stats."""
        feature_active = (features != 0).float().mean(dim=0)  # [dict_size]
        dict_size = feature_active.shape[0]

        dead_features = (feature_active == 0).sum().item()
        highly_active = (feature_active > 0.5).sum().item()

        return {
            "features/dead": dead_features,
            "features/dead_pct": (dead_features / dict_size) * 100,
            "features/highly_active": highly_active,
            "features/highly_active_pct": (highly_active / dict_size) * 100,
            "features/mean_activation_freq": feature_active.mean().item() * 100,
        }

    def _calculate_variance_explained(self, activations, reconstructed):
        total_variance = t.var(activations, dim=0).sum()
        residual_variance = t.var(activations - reconstructed, dim=0).sum()
        return (1 - residual_variance / total_variance).item()
