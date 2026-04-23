"""ProGen2 embedder implementation for InterPLM."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from interplm.embedders.base import BaseEmbedder
from interplm.utils import get_device


class ProGen2(BaseEmbedder):
    """ProGen2 embedder for extracting protein language model embeddings."""

    MODEL_DIMS = {
        "hugohrban/progen2-small": 1024,
        "hugohrban/progen2-large": 2560,
    }

    def __init__(
        self,
        model_name: str = "hugohrban/progen2-small",
        device: Optional[str] = None,
        max_length: int = 1024,
    ):
        """Initialize ProGen2 embedder and load model weights."""
        if device is None:
            device = get_device()

        super().__init__(model_name, device)
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.load_model()

    def _legacy_get_head_mask(
        self,
        head_mask: Optional[torch.Tensor],
        num_hidden_layers: int,
        dtype: torch.dtype,
    ):
        """Compatibility shim for remote ProGen code on transformers>=5."""
        if head_mask is None:
            return [None] * num_hidden_layers
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError("head_mask must have dimension 1 or 2")
        return head_mask.to(dtype=dtype)

    def _install_transformers_compat_shims(self) -> None:
        """Patch methods expected by legacy remote ProGen code."""
        transformer = getattr(self.model, "transformer", None)
        if transformer is not None and not hasattr(transformer, "get_head_mask"):
            model_dtype = self.model.dtype

            def _get_head_mask(head_mask, num_hidden_layers):
                return self._legacy_get_head_mask(head_mask, num_hidden_layers, model_dtype)

            transformer.get_head_mask = _get_head_mask

    def _fix_meta_attention_scale(self) -> None:
        """Patch legacy ProGen attention buffers left on meta device or corrupted after loading.

        Three persistent=False buffers are not saved in the checkpoint and must be
        re-initialised after from_pretrained:
          - scale_attn:   sqrt(head_dim) scaling factor
          - bias:         lower-triangular causal mask  [1, 1, n_pos, n_pos]
          - masked_bias:  large negative value added to masked positions (~-1e9)
        """
        transformer = getattr(self.model, "transformer", None)
        if transformer is None or not hasattr(transformer, "h"):
            return

        n_positions = self.model.config.n_positions

        for block in transformer.h:
            attn = getattr(block, "attn", None)
            if attn is None:
                continue

            # Fix scale_attn
            if hasattr(attn, "scale_attn"):
                scale_attn = attn.scale_attn
                if isinstance(scale_attn, torch.Tensor) and (
                    scale_attn.device.type == "meta" or scale_attn.item() == 0.0
                ):
                    attn.scale_attn = torch.sqrt(
                        torch.tensor(attn.head_dim, dtype=torch.float32, device=self.device)
                    ).to(torch.get_default_dtype())

            # Unconditionally reinitialize persistent=False buffers — they are not
            # saved in the checkpoint and reliably end up corrupted after loading.
            # bias:        lower-triangular causal mask [1, 1, n_pos, n_pos]
            # masked_bias: large negative added to future positions (~-1e9)
            if hasattr(attn, "bias"):
                attn.bias = torch.tril(
                    torch.ones((n_positions, n_positions), dtype=torch.bool, device=self.device)
                ).view(1, 1, n_positions, n_positions)

            if hasattr(attn, "masked_bias"):
                attn.masked_bias = torch.tensor(-1e9, device=self.device)

    def load_model(self) -> None:
        """Load ProGen2 tokenizer and model from Hugging Face."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        ).to(self.device)
        self._install_transformers_compat_shims()
        self._fix_meta_attention_scale()
        self.model.eval()

    def _validate_layer(self, layer: int) -> None:
        max_layer = self.available_layers[-1]
        if layer < 0 or layer > max_layer:
            raise ValueError(f"Layer {layer} out of range [0, {max_layer}]")

    def extract_embeddings(
        self,
        sequences: List[str],
        layer: int,
        batch_size: int = 8,
        return_contacts: bool = False,
    ) -> np.ndarray:
        """Extract flattened token embeddings for one layer."""
        if return_contacts:
            raise ValueError("ProGen2 does not support contact prediction output.")
        embeddings_dict = self.extract_embeddings_multiple_layers(
            sequences=sequences,
            layers=[layer],
            batch_size=batch_size,
            shuffle=False,
        )
        return embeddings_dict[layer]

    def extract_embeddings_multiple_layers(
        self,
        sequences: List[str],
        layers: List[int],
        batch_size: int = 8,
        shuffle: bool = False,
    ) -> Dict[int, torch.Tensor]:
        """Extract flattened token embeddings for multiple layers in one forward pass."""
        for layer in layers:
            self._validate_layer(layer)

        all_embeddings = {layer: [] for layer in layers}
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        batch_iterator = range(0, len(sequences), batch_size)
        if num_batches > 1:
            batch_iterator = tqdm(batch_iterator, desc="Processing batches", total=num_batches)

        for i in batch_iterator:
            batch_sequences = sequences[i : i + batch_size]
            if len(batch_sequences) == 0:
                continue

            inputs = self.tokenize(batch_sequences)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            attention_mask = inputs["attention_mask"].detach().cpu()

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states

            for layer in layers:
                layer_output = hidden_states[layer].detach().cpu()
                for seq_idx in range(layer_output.shape[0]):
                    seq_len = int(attention_mask[seq_idx].sum().item())
                    # seq_len includes the leading "1" and trailing "2" boundary
                    # tokens; strip them so only AA-position embeddings are kept.
                    seq_embeddings = layer_output[seq_idx, 1 : seq_len - 1, :]
                    all_embeddings[layer].append(seq_embeddings)

        result = {}
        for layer in layers:
            layer_tensor = torch.cat(all_embeddings[layer], dim=0)
            if shuffle:
                perm = torch.randperm(layer_tensor.size(0))
                layer_tensor = layer_tensor[perm]
            result[layer] = layer_tensor
        return result

    def extract_embeddings_with_boundaries_multiple_layers(
        self,
        sequences: List[str],
        layers: List[int],
        batch_size: int = 8,
    ) -> Dict[str, Union[Dict[int, torch.Tensor], List[Tuple[int, int]]]]:
        """Extract embeddings for multiple layers and track protein boundaries.

        Returns:
            Dictionary with:
                'embeddings': Dict[int, Tensor] mapping layer → (total_tokens, d_model)
                'boundaries': List of (start, end) tuples for each protein
        """
        embeddings_dict = self.extract_embeddings_multiple_layers(
            sequences, layers, batch_size=batch_size, shuffle=False
        )
        boundaries = []
        current_pos = 0
        for sequence in sequences:
            seq_len = len(self.preprocess_sequence(sequence))
            boundaries.append((current_pos, current_pos + seq_len))
            current_pos += seq_len
        return {"embeddings": embeddings_dict, "boundaries": boundaries}

    def extract_embeddings_with_boundaries(
        self,
        sequences: List[str],
        layer: int,
        batch_size: int = 8,
    ) -> Dict[str, Union[torch.Tensor, List[Tuple[int, int]]]]:
        """Extract embeddings and return per-sequence token boundaries."""
        embeddings_dict = self.extract_embeddings_multiple_layers(
            sequences,
            [layer],
            batch_size=batch_size,
            shuffle=False,
        )
        boundaries = []
        current_pos = 0
        for sequence in sequences:
            seq_len = len(self.preprocess_sequence(sequence))
            boundaries.append((current_pos, current_pos + seq_len))
            current_pos += seq_len
        return {
            "embeddings": embeddings_dict[layer],
            "boundaries": boundaries,
        }

    def embed_single_sequence(self, sequence: str, layer: int) -> np.ndarray:
        """Extract per-residue embeddings for one protein sequence."""
        self._validate_layer(layer)
        inputs = self.tokenize([sequence])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            layer_output = outputs.hidden_states[layer][0].detach().cpu()
        seq_len = int(inputs["attention_mask"][0].sum().item())
        # Strip boundary tokens "1" (pos 0) and "2" (pos seq_len-1).
        return layer_output[1 : seq_len - 1, :].numpy()

    def _read_fasta_sequences(self, fasta_path: Path) -> List[str]:
        """Read protein sequences from a FASTA file."""
        sequences: List[str] = []
        with open(fasta_path, "r", encoding="utf-8") as f:
            current_seq = []
            for line in f:
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_seq:
                sequences.append("".join(current_seq))
        return sequences

    def embed_fasta_file(
        self,
        fasta_path: Path,
        layer: int,
        output_path: Optional[Path] = None,
        batch_size: int = 8,
    ) -> Union[np.ndarray, None]:
        """Extract flattened embeddings for all proteins in a FASTA file."""
        sequences = self._read_fasta_sequences(fasta_path)
        embeddings = self.extract_embeddings(sequences, layer, batch_size=batch_size)
        # embeddings = embeddings.to(torch.fp16)
        if output_path:
            output_path = Path(output_path)
            if not str(output_path).endswith(".pt"):
                output_path = output_path.with_suffix(".pt")
            if isinstance(embeddings, torch.Tensor):
                torch.save(embeddings, output_path)
            else:
                torch.save(torch.from_numpy(embeddings), output_path)
            return None
        return embeddings

    def embed_fasta_file_multiple_layers(
        self,
        fasta_path: Path,
        layers: List[int],
        output_dir: Optional[Path] = None,
        batch_size: int = 8,
        shuffle: bool = False,
    ) -> Union[Dict[int, torch.Tensor], None]:
        """Extract flattened embeddings for multiple layers from one FASTA shard."""
        sequences = self._read_fasta_sequences(fasta_path)
        embeddings_dict = self.extract_embeddings_multiple_layers(
            sequences,
            layers=layers,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        if output_dir:
            output_dir = Path(output_dir)
            import yaml

            for layer, embeddings in embeddings_dict.items():
                layer_dir = output_dir / f"layer_{layer}" / fasta_path.stem
                layer_dir.mkdir(parents=True, exist_ok=True)
                torch.save(embeddings, layer_dir / "activations.pt")
                metadata = {
                    "model": self.model_name,
                    "layer": layer,
                    "d_model": int(embeddings.shape[1]),
                    "total_tokens": int(embeddings.shape[0]),
                    "dtype": "float32",
                }
                with open(layer_dir / "metadata.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(metadata, f, default_flow_style=False)
            return None
        return embeddings_dict

    def get_embedding_dim(self, layer: int) -> int:
        """Return embedding width for a given layer."""
        self._validate_layer(layer)
        if self.model is not None and hasattr(self.model.config, "hidden_size"):
            return int(self.model.config.hidden_size)
        if self.model is not None and hasattr(self.model.config, "embed_dim"):
            return int(self.model.config.embed_dim)
        if self.model_name in self.MODEL_DIMS:
            return self.MODEL_DIMS[self.model_name]
        raise ValueError(f"Unknown embedding dimension for {self.model_name}")

    @property
    def available_layers(self) -> List[int]:
        """Return all valid hidden-state layer indices."""
        if self.model is not None:
            if hasattr(self.model.config, "num_hidden_layers"):
                n_layers = int(self.model.config.num_hidden_layers)
            elif hasattr(self.model.config, "n_layer"):
                n_layers = int(self.model.config.n_layer)
            else:
                raise ValueError("Unable to infer number of ProGen2 layers from config.")
            return list(range(n_layers + 1))
        return []

    @property
    def max_sequence_length(self) -> int:
        """Return maximum sequence length used during tokenization."""
        return self.max_length

    def tokenize(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize protein sequences for ProGen2 input.

        ProGen2 was trained on sequences bracketed by boundary tokens:
          forward:  1<AA_seq>2
          reverse:  2<rev_AA_seq>1
        We always use the forward format here so the model sees the same
        context it was trained on.  The boundary tokens are stripped from
        the returned embeddings in the extraction methods.
        """
        processed = ["1" + self.preprocess_sequence(seq) + "2" for seq in sequences]
        return self.tokenizer(
            processed,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

    def preprocess_sequence(self, sequence: str) -> str:
        """Clean raw protein sequence for ProGen2 tokenization."""
        return "".join(sequence.strip().upper().split())
