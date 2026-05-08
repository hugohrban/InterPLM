import shutil
from pathlib import Path
from typing import List, Optional, Type

import pandas as pd
from interplm.utils import get_device
import torch as t
import yaml

from interplm.analysis.activation_sampling import (
    get_random_sample_of_sae_feats,
)
from interplm.sae.dictionary import Dictionary
from interplm.dashboard.protein_metadata import UniProtMetadata
from interplm.data_processing.embedding_loader import load_shard_embeddings


class DashboardCache:
    """Class representing a dashboard cache containing SAE feature data across layers"""

    def __init__(
        self,
        cache_dir: Path,
    ):
        self.cache_dir = Path(cache_dir)
        # Extract dashboard name from the cache directory name
        self.dashboard_name = self.cache_dir.name

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Dashboard cache not found at {self.cache_dir}")

    @classmethod
    def load_cache(cls, cache_dir: Path) -> "DashboardCache":
        """Load the cache from disk

        Args:
            cache_dir: Full path to the dashboard cache directory
        """
        cache = cls(cache_dir)
        cache._load_cache()
        return cache

    def _load_cache(self) -> None:
        """Load the cache from disk"""
        cache_level_metadata = self._load_cache_level_metadata()
        self.model_name = cache_level_metadata["model_name"]
        self.model_type = cache_level_metadata["model_type"]
        self.protein_metadata = cache_level_metadata["protein_metadata"]

        self.layers = self._list_layers()

        self.data = {}
        for layer in self.layers:
            self.data[layer] = self._load_layer(layer_name=layer)

    def _list_layers(self) -> List[str]:
        """List all layers in the dashboard cache"""
        return [layer.name for layer in self.cache_dir.iterdir() if layer.is_dir()]

    def _load_cache_level_metadata(self) -> None:
        """Load cache level metadata from disk"""
        with open(self.cache_dir / "cache_level_metadata.yaml", "r") as f:
            return yaml.unsafe_load(f)

    def _write_cache_level_metadata(
        self, model_name: str, model_type: str, protein_metadata: UniProtMetadata
    ) -> None:
        """Write cache level metadata to disk"""

        cache_level_metadata = {
            "model_name": model_name,
            "model_type": model_type,
            "protein_metadata": protein_metadata,
        }
        with open(self.cache_dir / "cache_level_metadata.yaml", "w") as f:
            yaml.dump(cache_level_metadata, f)

    def _load_layer(self, layer_name: str) -> dict:
        """Load a single layer's data from disk"""
        layer_dir = self.cache_dir / layer_name

        if not layer_dir.exists():
            raise FileNotFoundError(f"Layer {layer_name} not found in {self.cache_dir}")

        layer_data = {}

        with open(layer_dir / "SAE_features.yaml", "r") as f:
            layer_data["SAE_features"] = {
                int(k): v for k, v in yaml.unsafe_load(f).items()
            }

        with open(layer_dir / "Per_feature_max_examples.yaml", "r") as f:
            layer_data["Per_feature_max_examples"] = yaml.unsafe_load(f)

        with open(layer_dir / "Per_feature_quantile_examples.yaml", "r") as f:
            layer_data["Per_feature_quantile_examples"] = yaml.unsafe_load(f)

        # load per_feature_statistics
        with open(layer_dir / "Per_feature_statistics.yaml", "r") as f:
            layer_data["Per_feature_statistics"] = yaml.unsafe_load(f)

        with open(layer_dir / "layer_info.yaml", "r") as f:
            layer_info = yaml.unsafe_load(f)

        ae_cls = layer_info["SAE_cls"]
        layer_data["aa_embeds_dir"] = layer_info["aa_embeds_dir"]
        layer_data["aa_metadata_dir"] = layer_info["aa_metadata_dir"]
        layer_data["sae_dir"] = layer_info.get("sae_dir")
        layer_data["annot_dir"] = layer_info.get("annot_dir")

        # Load SAE on the correct device to avoid device mismatch
        device = get_device()
        sae = ae_cls.from_pretrained(layer_dir / "SAE.pt", device=device)
        layer_data["SAE"] = sae

        optional_files = {
            "Sig_concepts_per_feature.csv": ("Sig_concepts_per_feature", pd.read_csv),
            "LLM_Autointerp.csv": ("LLM Autointerp", self._load_llm_autointerp),
            "UMAP.csv": ("UMAP", pd.read_csv),
            "Structure_feats.csv": ("Structure_feats", pd.read_csv),
        }

        for filename, (key, loader) in optional_files.items():
            file_path = layer_dir / filename
            if file_path.exists():
                with open(file_path, "r" if filename.endswith("csv") else "rb") as f:
                    layer_data[key] = loader(f)

        return layer_data

    def _load_llm_autointerp(self, file):
        """Helper to load LLM autointerp data with proper index"""
        df = pd.read_csv(file)
        if df.index.name != "Feature":
            df.set_index("Feature", inplace=True)
        return df

    @classmethod
    def create_dashboard_cache(
        cls,
        cache_dir: Path,
        model_name: str,
        model_type: str,
        protein_metadata: UniProtMetadata,
        overwrite: bool = False,
    ):
        """Create a new dashboard cache

        Args:
            cache_dir: Full path to the dashboard cache directory
            model_name: Name of the model (e.g., "esm")
            model_type: Type/variant of the model (e.g., "esm2_t6_8M_UR50D")
            protein_metadata: Protein metadata object
            overwrite: If True, overwrite existing cache
        """
        cache_dir = Path(cache_dir)
        if cache_dir.exists() and not overwrite:
            # Confirm that the model_metadata.yaml exists and matches the model_name and model_type
            if not (cache_dir / "model_metadata.yaml").exists():

                raise FileNotFoundError(f"Model metadata not found at {cache_dir / 'model_metadata.yaml'}")
            with open(cache_dir / "model_metadata.yaml", "r") as f:
                existing_metadata = yaml.load(f, Loader=yaml.FullLoader)
            if (
                existing_metadata["model_name"] != model_name
                or existing_metadata["model_type"] != model_type
            ):
                raise ValueError(
                    "Model metadata does not match the provided model_name and model_type"
                )

            # If none of these errors occured and everything matches, just load the model info
            return cls(cache_dir)

        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cls(cache_dir)
        cache.model_type = model_type
        cache.model_name = model_name
        cache._write_cache_level_metadata(model_name, model_type, protein_metadata)

        return cache

    def add_layer(
        self,
        # Layer level info
        layer_name: str,
        sae_cls: Type[Dictionary],
        sae_path: Path,
        # Pre-computed feature statistics (required)
        feature_stats_dir: Path,
        # Info for sorting proteins by activation (only needed if feature_stats_dir not provided)
        aa_embeds_dir: Optional[Path] = None,
        aa_metadata_dir: Optional[Path] = None,
        shards_to_search: Optional[List[int]] = None,
        # Additional optional things to plot
        concept_enrichment_path: Optional[Path] = None,
        # Paths needed for Concept Explorer
        sae_dir: Optional[Path] = None,
        annot_dir: Optional[Path] = None,
        overwrite: bool = True,
    ) -> None:
        """Add a new layer to the dashboard cache"""
        layer_dir = self.cache_dir / f"{layer_name}"
        if layer_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Layer {layer_name} already exists in {self.cache_dir}"
            )
        elif not layer_dir.exists():
            layer_dir.mkdir(parents=True, exist_ok=True)

        # Load and copy SAE model
        sae_path = Path(sae_path)
        if not sae_path.exists():
            raise FileNotFoundError(f"No SAE model found at {sae_path}")

        # save sae_cls to disk as yaml
        with open(layer_dir / "layer_info.yaml", "w") as f:
            yaml.dump(
                {
                    "SAE_cls": sae_cls,
                    "aa_embeds_dir": aa_embeds_dir,
                    "aa_metadata_dir": aa_metadata_dir,
                    "shards_to_search": shards_to_search,
                    "sae_dir": str(sae_dir) if sae_dir is not None else None,
                    "annot_dir": str(annot_dir) if annot_dir is not None else None,
                },
                f,
            )

        # shutil.copy(sae_path, layer_dir / "SAE.pt")

        # Load feature statistics
        device = get_device()
        sae = sae_cls.from_pretrained(sae_path, device=device)

        # Load pre-computed feature statistics (required)
        precomputed_stats = feature_stats_dir / "Per_feature_statistics.yaml"
        precomputed_max_examples = feature_stats_dir / "Per_feature_max_examples.yaml"
        precomputed_quantile = feature_stats_dir / "Per_feature_quantile_examples.yaml"
        precomputed_max_activations = feature_stats_dir / "max_activations_per_feature.pt"

        if not all([precomputed_stats.exists(), precomputed_max_examples.exists(),
                   precomputed_quantile.exists(), precomputed_max_activations.exists()]):
            raise FileNotFoundError(
                f"Missing required files in feature_stats_dir: {feature_stats_dir}\n"
                f"Expected:\n"
                f"  - Per_feature_statistics.yaml\n"
                f"  - Per_feature_max_examples.yaml\n"
                f"  - Per_feature_quantile_examples.yaml\n"
                f"  - max_activations_per_feature.pt\n\n"
                f"Run this command first:\n"
                f"  python examples/collect_feature_activations.py \\\n"
                f"    --sae_path {sae_path} \\\n"
                f"    --embeddings_dir <path> \\\n"
                f"    --metadata_dir <path> \\\n"
                f"    --shards 0 1 2 ..."
            )

        print("✓ Loading pre-computed feature statistics...")

        # Load pre-computed results
        with open(precomputed_stats, "r") as f:
            per_feature_statistics = yaml.safe_load(f)
        with open(precomputed_max_examples, "r") as f:
            max_examples = yaml.safe_load(f)
        with open(precomputed_quantile, "r") as f:
            # Use unsafe_load because quantile keys are Python tuples
            per_quantile_examples = yaml.unsafe_load(f)
        max_activations = t.load(precomputed_max_activations, map_location=device)

        # Reconstruct per_protein_tracker dict format
        per_protein_tracker = {
            "max_activation_per_feature": max_activations.tolist() if isinstance(max_activations, t.Tensor) else max_activations,
            "pct_proteins_with_activation": per_feature_statistics["Per_prot_frequency_of_any_activation"],
            "avg_pct_activated_when_present": per_feature_statistics["Per_prot_pct_activated_when_present"],
            "max": max_examples,
            "lower_quantile": per_quantile_examples,
        }

        # add the max activation per feature to the SAE as normalization factor
        sae.activation_rescale_factor = t.tensor(
            per_protein_tracker["max_activation_per_feature"], device=device
        )

        # save the SAE to disk
        t.save(sae.state_dict(), layer_dir / "SAE.pt")

        per_feature_statistics = {
            "Per_prot_frequency_of_any_activation": per_protein_tracker[
                "pct_proteins_with_activation"
            ],
            "Per_prot_pct_activated_when_present": per_protein_tracker[
                "avg_pct_activated_when_present"
            ],
        }

        with open(layer_dir / "Per_feature_statistics.yaml", "w") as f:
            yaml.dump(per_feature_statistics, f)

        with open(layer_dir / "Per_feature_max_examples.yaml", "w") as f:
            yaml.dump(per_protein_tracker["max"], f)

        per_quantile_examples = per_protein_tracker["lower_quantile"]

        with open(layer_dir / "Per_feature_quantile_examples.yaml", "w") as f:
            yaml.dump(per_quantile_examples, f)

        # Generate and save SAE features (only if embeddings directory provided)
        if aa_embeds_dir is not None and shards_to_search is not None:
            # Use centralized embedding loader (returns tensor only)
            sae_feats = get_random_sample_of_sae_feats(
                sae=sae,
                aa_embds_dir=aa_embeds_dir,
                shards_to_search=shards_to_search,
                get_aa_activations_fn=load_shard_embeddings,  # Already returns tensor by default
            )
        else:
            # Create empty SAE features if embeddings not provided
            sae_feats = {}

        with open(layer_dir / "SAE_features.yaml", "w") as f:
            yaml.dump(sae_feats, f)

        # Copy concept F1 results if provided (Sig_concepts_per_feature.csv)
        if concept_enrichment_path:
            concept_enrichment_path = Path(concept_enrichment_path)
            if not concept_enrichment_path.exists():
                print(f"Warning: Concept F1 file not found at {concept_enrichment_path}")
            else:
                print(f"✓ Adding concept F1 results from {concept_enrichment_path}")
                # This should be the Sig_concepts_per_feature.csv file
                # Just copy it directly - no need to rename columns
                output_path = layer_dir / "Sig_concepts_per_feature.csv"
                shutil.copy(concept_enrichment_path, output_path)
                # Count features for logging
                df = pd.read_csv(output_path)
                print(f"  Saved concept F1 results for {df['feature'].nunique()} features to dashboard")
