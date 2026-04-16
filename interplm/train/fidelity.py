"""
Calculate the loss (cross entropy) fidelity metric for a Sparse Autoencoder (SAE) trained on ESM embeddings
1. Calculates original cross entropy and cross entropy after zero-ablation.
2. Creates a function to calculate cross entropy using SAE reconstructions.
3. During each evaluation step, uses step 2 to evaluate the current SAE model
   then compares to the original and zero-ablation cross entropy to calculate
   the loss recovered metric.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from esm import pretrained
from nnsight import NNsight
from tqdm import tqdm
from transformers import EsmForMaskedLM, AutoModelForCausalLM

# from interplm.esm.embed import shuffle_individual_parameters  # Not available in public repo


# ---------------------------------------------------------------------------
# Fidelity baseline cache
#
# orig_loss and zero_loss are purely a function of (model, eval sequences,
# layer).  Computing them is expensive (full forward pass over all eval seqs
# twice) and identical across every training run with the same config.  We
# cache them on disk next to the eval sequence file so subsequent runs load
# instantly.
#
# Cache validity is guarded by a SHA-256 hash of (model_name + layer_idx +
# raw file content).  If anything changes the cache is silently invalidated
# and recomputed.
# ---------------------------------------------------------------------------


def _fidelity_cache_hash(model_name: str, layer_idx: int, eval_seq_content: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(str(layer_idx).encode())
    h.update(eval_seq_content.encode())
    return h.hexdigest()


def _fidelity_cache_path(eval_seq_path: Path, model_name: str, layer_idx: int) -> Path:
    model_slug = model_name.replace("/", "_").replace("\\", "_")
    # cache_dir = eval_seq_path.parent / ".fidelity_cache"
    cache_dir = Path.home() / "InterPLM" / "data" / ".fidelity_cache"
    return cache_dir / f"{eval_seq_path.stem}__{model_slug}__layer{layer_idx}.json"


def _load_fidelity_cache(
    cache_path: Path, expected_hash: str
) -> tuple[float, float] | None:
    """Return (orig_loss, zero_loss) from cache, or None if missing/stale."""
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
    except Exception as e:
        print(f"  [fidelity cache] Could not read {cache_path}: {e} — recomputing.")
        return None
    if data.get("hash") != expected_hash:
        print(f"  [fidelity cache] Hash mismatch for {cache_path.name} — recomputing.")
        return None
    orig_loss = float(data["orig_loss"])
    zero_loss = float(data["zero_loss"])
    print(
        f"  [fidelity cache] Loaded from {cache_path}\n"
        f"    orig_loss={orig_loss:.4f}  zero_loss={zero_loss:.4f}"
    )
    return orig_loss, zero_loss


def _save_fidelity_cache(
    cache_path: Path,
    cache_hash: str,
    orig_loss: float,
    zero_loss: float,
    model_name: str,
    layer_idx: int,
    eval_seq_path: Path,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "hash": cache_hash,
        "orig_loss": orig_loss,
        "zero_loss": zero_loss,
        "model_name": model_name,
        "layer_idx": layer_idx,
        "eval_seq_path": str(eval_seq_path),
    }
    cache_path.write_text(json.dumps(data, indent=2))
    print(f"  [fidelity cache] Saved to {cache_path}")
from interplm.sae.intervention import get_esm_output_with_intervention, get_progen_output_with_intervention
from interplm.train.evaluation import EvaluationConfig, EvaluationManager
from interplm.utils import get_device
from interplm.embedders import get_embedder


@dataclass
class ESMFidelityConfig(EvaluationConfig):
    model_name: str | None = None
    layer_idx: int | None = None
    corrupt: bool = False

    def build(self) -> "ESMFidelityFunction":
        return ESMFidelityFunction(self)


def calculate_cross_entropy(model_output, batch_tokens, batch_attn_mask):
    """Calculate cross entropy for each sequence in batch, excluding start/end tokens.

    For masked LMs (ESM): logit[i] predicts token[i], so we slice [1:length-1] for both.
    """
    losses = []
    for j, mask in enumerate(batch_attn_mask):
        length = mask.sum()
        seq_logits = model_output[j, 1 : length - 1]
        seq_tokens = batch_tokens[j, 1 : length - 1]
        loss = F.cross_entropy(seq_logits, seq_tokens)
        losses.append(loss.item())
    return losses


def calculate_cross_entropy_causal(model_output, batch_tokens, batch_attn_mask):
    """Calculate cross entropy for each sequence in batch for a causal LM (e.g. ProGen2).

    Sequences are formatted as "1<AA>2" where "1"/"2" are boundary tokens.
    For causal LMs, logit[i] predicts token[i+1], so we shift:
      - logits: positions 0..n_aa-1  (model_output[j, 0 : length-2])
      - targets: positions 1..n_aa   (batch_tokens[j, 1 : length-1])
    This gives next-token CE over AA positions only, excluding the boundary tokens.
    """
    losses = []
    for j, mask in enumerate(batch_attn_mask):
        length = mask.sum()
        seq_logits = model_output[j, 0 : length - 2]
        seq_tokens = batch_tokens[j, 1 : length - 1]
        loss = F.cross_entropy(seq_logits, seq_tokens)
        losses.append(loss.item())
    return losses


def calculate_loss_recovered(ce_autoencoder, ce_identity, ce_zero_ablation):
    """
    Calculate the loss recovered metric for a Sparse Autoencoder (SAE).

    If the recovered loss is as good as  to the original loss as possible, the
    metric will be 100%. If the recovered loss is as bad as zero-ablating, the
    metric will be 0%.

    Parameters:
    ce_autoencoder (float): Cross-entropy loss when using the SAE's reconstructions
    ce_identity (float): Cross-entropy loss when using the identity function
    ce_zero_ablation (float): Cross-entropy loss when using the zero-ablation function

    Returns:
    float: The loss recovered metric as a percentage
    """

    numerator = ce_autoencoder - ce_identity
    denominator = ce_zero_ablation - ce_identity

    # Avoid division by zero
    if np.isclose(denominator, 0):
        return 0.0

    loss_recovered = 1 - (numerator / denominator)

    # Clip the result to be between 0 and 1
    loss_recovered = np.clip(loss_recovered, 0, 1).item()

    # Convert to percentage
    return loss_recovered * 100.0


class ESMFidelityFunction(EvaluationManager):
    def __init__(
        self,
        eval_config: "ESMFidelityConfig",
    ):
        # The super class just sets the config
        super().__init__(eval_config)

        print("Prepping loss fidelity_fn")
        self.device = eval_config.device or get_device()

        # Extract config values
        self.model_name = eval_config.model_name
        self.eval_seq_path = eval_config.eval_seq_path
        self.layer_idx = eval_config.layer_idx
        self.batch_size = eval_config.eval_batch_size or 256
        self.corrupt = eval_config.corrupt

        # Load the ESM model and alphabet
        _, alphabet = pretrained.load_model_and_alphabet(self.model_name)
        self.model = EsmForMaskedLM.from_pretrained(f"facebook/{self.model_name}").to(
            self.device
        )

        if self.corrupt:
            raise NotImplementedError(
                "The 'corrupt' parameter requires shuffle_individual_parameters() "
                "which is not available in the public repo"
            )

        batch_converter = alphabet.get_batch_converter()

        # Load evaluation sequences
        eval_seq_content = Path(self.eval_seq_path).read_text()
        eval_seqs = [line.strip() for line in eval_seq_content.splitlines() if line.strip()]

        # Prepare data in the format expected by the batch converter
        data = [(f"protein{i}", seq) for i, seq in enumerate(eval_seqs)]

        # Pre-tokenize and create batches
        self.tokenized_batches = []
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i : i + self.batch_size]
            _, _, batch_tokens = batch_converter(batch_data)
            batch_mask = (batch_tokens != alphabet.padding_idx).to(int)
            self.tokenized_batches.append((batch_tokens, batch_mask))

        self.nnsight_model = NNsight(self.model).to(self.device)
        self.layer_idx = self.layer_idx

        # orig_loss and zero_loss are fixed for (model, eval_seqs, layer) —
        # load from cache when available to skip the expensive forward passes.
        _cache_hash = _fidelity_cache_hash(self.model_name, self.layer_idx, eval_seq_content)
        _cache_path = _fidelity_cache_path(Path(self.eval_seq_path), self.model_name, self.layer_idx)
        _cached = _load_fidelity_cache(_cache_path, _cache_hash)
        if _cached is not None:
            self.orig_loss, self.zero_loss = _cached
        else:
            self.orig_loss, self.zero_loss = self._CE_for_orig_and_zero_ablation(
                self.tokenized_batches
            )
            _save_fidelity_cache(
                _cache_path, _cache_hash, self.orig_loss, self.zero_loss,
                self.model_name, self.layer_idx, Path(self.eval_seq_path),
            )
        print("Finished initializing ESM Fidelity Function")

    def _calculate_fidelity(self, sae_model, use_all_batches: bool = False) -> dict:
        batches = (
            self.tokenized_batches
            if use_all_batches or self.fidelity_n_batches is None or len(self.tokenized_batches) <= self.fidelity_steps
            else self.tokenized_batches[:np.random.choice(len(self.tokenized_batches), self.fidelity_steps, replace=False)]
        )
        sae_loss = self._CE_from_sae_recon(batches, sae_model)

        loss_recovered = calculate_loss_recovered(
            ce_autoencoder=sae_loss,
            ce_identity=self.orig_loss,
            ce_zero_ablation=self.zero_loss,
        )

        return {"pct_loss_recovered": loss_recovered, "CE_w_sae_patching": sae_loss}

    def _CE_for_orig_and_zero_ablation(self, tokenized_batches):
        """Calculate cross entropy for original and zero-ablated outputs."""
        orig_losses, zero_losses = [], []

        for batch_tokens, batch_attn_mask in tqdm(tokenized_batches):
            batch_tokens = batch_tokens.to(self.device)
            batch_attn_mask = batch_attn_mask.to(self.device)

            orig_logits, orig_hidden = get_esm_output_with_intervention(
                self.model,
                self.nnsight_model,
                batch_tokens,
                batch_attn_mask,
                self.layer_idx,
            )

            zero_logits, _ = get_esm_output_with_intervention(
                self.model,
                self.nnsight_model,
                batch_tokens,
                batch_attn_mask,
                self.layer_idx,
                torch.zeros_like(orig_hidden),
            )

            orig_losses.extend(
                calculate_cross_entropy(orig_logits, batch_tokens, batch_attn_mask)
            )
            zero_losses.extend(
                calculate_cross_entropy(zero_logits, batch_tokens, batch_attn_mask)
            )

        return np.mean(orig_losses).item(), np.mean(zero_losses).item()

    def _CE_from_sae_recon(self, tokenized_batches, sae_model):
        """Calculate cross entropy using SAE reconstructions."""
        sae_losses = []

        for batch_tokens, batch_attn_mask in tqdm(tokenized_batches):
            batch_tokens = batch_tokens.to(self.device)
            batch_attn_mask = batch_attn_mask.to(self.device)

            _, orig_hidden = get_esm_output_with_intervention(
                self.model,
                self.nnsight_model,
                batch_tokens,
                batch_attn_mask,
                self.layer_idx,
            )

            # Get reconstructions with unnormalize=True for injection into the model
            reconstructions = sae_model(orig_hidden, unnormalize=True)
            sae_logits, _ = get_esm_output_with_intervention(
                self.model,
                self.nnsight_model,
                batch_tokens,
                batch_attn_mask,
                self.layer_idx,
                reconstructions,
            )

            sae_losses.extend(
                calculate_cross_entropy(sae_logits, batch_tokens, batch_attn_mask)
            )

        return np.mean(sae_losses).item()


@dataclass
class ProGenFidelityConfig(EvaluationConfig):
    model_name: str | None = None
    layer_idx: int | None = None
    corrupt: bool = False

    def build(self) -> "ProGenFidelityFunction":
        return ProGenFidelityFunction(self)


class ProGenFidelityFunction(EvaluationManager):
    def __init__(
        self,
        eval_config: "ProGenFidelityConfig",
    ):
        # The super class just sets the config
        super().__init__(eval_config)

        print("Prepping loss fidelity_fn")
        self.device = eval_config.device or get_device()

        # Extract config values
        self.model_name = eval_config.model_name
        self.eval_seq_path = eval_config.eval_seq_path
        self.layer_idx = eval_config.layer_idx
        self.batch_size = eval_config.eval_batch_size or 16
        self.corrupt = eval_config.corrupt

        self.embedder = get_embedder("progen", model_name=self.model_name, device=self.device)
        self.model = self.embedder.model

        if self.corrupt:
            raise NotImplementedError(
                "The 'corrupt' parameter requires shuffle_individual_parameters() "
                "which is not available in the public repo"
            )

        # Load evaluation sequences
        eval_seq_content = Path(self.eval_seq_path).read_text()
        eval_seq_path_suffix = self.eval_seq_path.suffix
        if eval_seq_path_suffix in [".txt"]:
            data: list[str] = [line.strip() for line in eval_seq_content.splitlines() if line.strip()]
        elif eval_seq_path_suffix in [".fasta", ".fa"]:
            from Bio import SeqIO
            data = []
            for record in SeqIO.parse(self.eval_seq_path, "fasta"):
                seq = str(record.seq).strip()
                if seq:
                    data.append(seq)
        else:
            raise ValueError(f"Unsupported eval sequence file format: {eval_seq_path_suffix}")
        

        # Pre-tokenize and create batches
        self.tokenized_batches = []
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i : i + self.batch_size]
            batch_tokens: torch.Tensor = self.embedder.tokenize(batch_data).input_ids
            batch_mask: torch.Tensor = (batch_tokens != self.embedder.tokenizer.pad_token_id).to(int)
            self.tokenized_batches.append((batch_tokens, batch_mask))

        print(f"Loaded {len(data)} sequences, created {len(self.tokenized_batches)} batches")

        self.nnsight_model = NNsight(self.model).to(self.device)
        self.layer_idx = self.layer_idx

        # orig_loss and zero_loss are fixed for (model, eval_seqs, layer) —
        # load from cache when available to skip the expensive forward passes.
        _cache_hash = _fidelity_cache_hash(self.model_name, self.layer_idx, eval_seq_content)
        _cache_path = _fidelity_cache_path(Path(self.eval_seq_path), self.model_name, self.layer_idx)
        _cached = _load_fidelity_cache(_cache_path, _cache_hash)
        if _cached is not None:
            self.orig_loss, self.zero_loss = _cached
        else:
            self.orig_loss, self.zero_loss = self._CE_for_orig_and_zero_ablation(
                self.tokenized_batches
            )
            _save_fidelity_cache(
                _cache_path, _cache_hash, self.orig_loss, self.zero_loss,
                self.model_name, self.layer_idx, Path(self.eval_seq_path),
            )
        print("Finished initializing ProGen2 Fidelity Function")

    def _calculate_fidelity(self, sae_model, use_all_batches: bool = False) -> dict:
        batches = (
            self.tokenized_batches
            if use_all_batches or self.fidelity_n_batches is None or len(self.tokenized_batches) <= self.fidelity_steps
            else self.tokenized_batches[:np.random.choice(len(self.tokenized_batches), self.fidelity_steps, replace=False)]
        )
        sae_loss = self._CE_from_sae_recon(batches, sae_model)

        loss_recovered = calculate_loss_recovered(
            ce_autoencoder=sae_loss,
            ce_identity=self.orig_loss,
            ce_zero_ablation=self.zero_loss,
        )

        return {"pct_loss_recovered": loss_recovered, "CE_w_sae_patching": sae_loss}

    def _CE_for_orig_and_zero_ablation(self, tokenized_batches):
        """Calculate cross entropy for original and zero-ablated outputs."""
        orig_losses, zero_losses = [], []

        for batch_tokens, batch_attn_mask in tqdm(tokenized_batches, total=len(tokenized_batches), desc="Calculating original and zero-ablated cross entropy", unit="batch"):
            batch_tokens = batch_tokens.to(self.device)
            batch_attn_mask = batch_attn_mask.to(self.device)

            orig_logits, orig_hidden = get_progen_output_with_intervention(
                self.model,
                self.nnsight_model,
                batch_tokens,
                batch_attn_mask,
                self.layer_idx,
            )

            zero_logits, _ = get_progen_output_with_intervention(
                self.model,
                self.nnsight_model,
                batch_tokens,
                batch_attn_mask,
                self.layer_idx,
                torch.zeros_like(orig_hidden),
            )

            orig_losses.extend(
                # calculate_cross_entropy(orig_logits, batch_tokens, batch_attn_mask)
                calculate_cross_entropy_causal(orig_logits, batch_tokens, batch_attn_mask)
            )
            zero_losses.extend(
                # calculate_cross_entropy(zero_logits, batch_tokens, batch_attn_mask)
                calculate_cross_entropy_causal(zero_logits, batch_tokens, batch_attn_mask)
            )

        return np.mean(orig_losses).item(), np.mean(zero_losses).item()

    def _CE_from_sae_recon(self, tokenized_batches, sae_model):
        """Calculate cross entropy using SAE reconstructions."""
        sae_losses = []

        for batch_tokens, batch_attn_mask in tqdm(tokenized_batches, total=len(tokenized_batches), desc="Calculating SAE cross entropy", unit="batch"):
            batch_tokens = batch_tokens.to(self.device)
            batch_attn_mask = batch_attn_mask.to(self.device)

            _, orig_hidden = get_progen_output_with_intervention(
                self.model,
                self.nnsight_model,
                batch_tokens,
                batch_attn_mask,
                self.layer_idx,
            )

            # Get reconstructions with unnormalize=True for injection into the model
            reconstructions = sae_model(orig_hidden, unnormalize=True)
            sae_logits, _ = get_progen_output_with_intervention(
                self.model,
                self.nnsight_model,
                batch_tokens,
                batch_attn_mask,
                self.layer_idx,
                reconstructions,
            )

            sae_losses.extend(
                calculate_cross_entropy_causal(sae_logits, batch_tokens, batch_attn_mask)
            )

        return np.mean(sae_losses).item()
