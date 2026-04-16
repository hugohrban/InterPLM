#!/usr/bin/env python
"""
Launch N independent SAE training runs in parallel, one per GPU.

Each run is a fully isolated subprocess with its own CUDA_VISIBLE_DEVICES, so
there is no shared state, no DDP communication, and no CUDA context bleed
between jobs.  When the number of configs exceeds the number of GPUs the
executor queues jobs automatically.

Usage examples
--------------
# YAML-defined sweep (recommended for complex configs)
python scripts/sweep_sae.py --sweep_config sweep.yaml --gpus 0 1 2 3

# Quick multi-layer sweep (reuses a single set of extra args for every layer)
python scripts/sweep_sae.py \\
    --embeddings_base_dir /data/esm2_650m \\
    --layers 4 8 12 16 \\
    --save_base_dir /models/sweep_001 \\
    --gpus 0 1 2 3 \\
    --trainer_type topk --k 32 --steps 50000

# Dry run: print the commands that would be executed, then exit
python scripts/sweep_sae.py --sweep_config sweep.yaml --gpus 0 1 --dry_run

Sweep YAML format
-----------------
defaults:          # merged into every run (lower priority than per-run keys)
  trainer_type: topk
  steps: 50000
  batch_size: 512

runs:              # each entry overrides / extends defaults
  - embeddings_dir: /data/esm2/layer_4
    save_dir: /models/sweep/layer4
    k: 32
  - embeddings_dir: /data/esm2/layer_8
    save_dir: /models/sweep/layer8
    k: 64
"""

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import cycle
from pathlib import Path
from typing import Optional

import yaml
from tap import tapify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_sweep_yaml(path: Path) -> list[dict]:
    """Parse sweep YAML and return a list of fully-resolved run configs."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    defaults = raw.get("defaults", {})
    runs = raw.get("runs", [])
    if not runs:
        raise ValueError(f"No 'runs' found in {path}")
    return [{**defaults, **run} for run in runs]


def _build_command(run_cfg: dict) -> list[str]:
    """Turn a run-config dict into a train_sae.py argv list."""
    script = str(Path(__file__).parent / "train_sae.py")
    cmd = [sys.executable, script]
    for key, value in run_cfg.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif value is not None:
            cmd.extend([flag, str(value)])
    return cmd


def _run_single(cmd: list[str], gpu_id: int) -> tuple[bool, str, str]:
    """Execute one training run on a specific GPU (called in subprocess pool)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(
    # YAML-based sweep
    sweep_config: Optional[Path] = None,
    # Quick multi-layer sweep (alternative to YAML)
    embeddings_base_dir: Optional[Path] = None,
    layers: Optional[list[int]] = None,
    save_base_dir: Optional[Path] = None,
    # GPU assignment
    gpus: list[int] = None,
    # Passthrough args for all runs (used with --layers mode)
    trainer_type: str = "relu",
    steps: int = 10_000,
    batch_size: int = 512,
    k: int = 32,
    expansion_factor: int = 4,
    # Misc
    dry_run: bool = False,
):
    """Run N SAE training jobs in parallel, one per GPU.

    Specify either --sweep_config (YAML) or --embeddings_base_dir + --layers
    for a quick multi-layer sweep using identical hyperparameters.
    """
    if gpus is None:
        import torch
        n = torch.cuda.device_count()
        gpus = list(range(n)) if n > 0 else [0]
        print(f"No --gpus specified; using all available: {gpus}")

    # ------------------------------------------------------------------ #
    # Build run configs                                                    #
    # ------------------------------------------------------------------ #
    if sweep_config is not None:
        run_configs = _load_sweep_yaml(Path(sweep_config))
        print(f"Loaded {len(run_configs)} runs from {sweep_config}")
    elif embeddings_base_dir is not None and layers is not None:
        if save_base_dir is None:
            raise ValueError("--save_base_dir is required with --layers")
        run_configs = []
        for layer in layers:
            run_configs.append({
                "embeddings_dir": str(Path(embeddings_base_dir) / f"layer_{layer}"),
                "save_dir": str(Path(save_base_dir) / f"layer_{layer}"),
                "trainer_type": trainer_type,
                "steps": steps,
                "batch_size": batch_size,
                "k": k,
                "expansion_factor": expansion_factor,
            })
        print(f"Built {len(run_configs)} runs for layers {layers}")
    else:
        raise ValueError(
            "Provide either --sweep_config <yaml> or "
            "--embeddings_base_dir + --layers + --save_base_dir"
        )

    # ------------------------------------------------------------------ #
    # Assign GPUs (cycle when more runs than GPUs)                        #
    # ------------------------------------------------------------------ #
    gpu_cycle = cycle(gpus)
    assignments: list[tuple[dict, int]] = [
        (cfg, next(gpu_cycle)) for cfg in run_configs
    ]

    # ------------------------------------------------------------------ #
    # Dry run: just print commands                                         #
    # ------------------------------------------------------------------ #
    if dry_run:
        print("\n[DRY RUN] Commands that would be executed:\n")
        for cfg, gpu in assignments:
            cmd = _build_command(cfg)
            print(f"  GPU {gpu}:  CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}\n")
        return

    # ------------------------------------------------------------------ #
    # Execute                                                              #
    # ------------------------------------------------------------------ #
    print(f"\nLaunching {len(assignments)} jobs on {len(gpus)} GPU(s): {gpus}\n")
    results: list[tuple[bool, str]] = []

    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(_run_single, _build_command(cfg), gpu): (cfg, gpu)
            for cfg, gpu in assignments
        }
        for future in as_completed(futures):
            cfg, gpu = futures[future]
            ok, stdout, stderr = future.result()
            save_dir = cfg.get("save_dir", "?")
            status = "✅" if ok else "❌"
            msg = "done" if ok else f"FAILED\n{stderr[-600:]}"
            print(f"{status} GPU {gpu}  {save_dir}: {msg}")
            results.append((ok, save_dir))

    n_ok = sum(ok for ok, _ in results)
    n_fail = len(results) - n_ok
    print(f"\nSweep complete: {n_ok}/{len(results)} succeeded"
          + (f", {n_fail} failed" if n_fail else "") + ".")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    tapify(main)
