"""Model Checkpointing and State Persistence.
Provenance: [REFERENCE IMPLEMENTATION]
Saves and restores model weights, optimizer states, progressive resolution stage, and training step.
"""

import os
from typing import Dict, Any, Optional
import torch


def save_checkpoint(
    filepath: str,
    step: int,
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    config: Dict[str, Any],
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Save complete training checkpoint to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state": model.state_dict(),
        "config": config,
        "metrics": metrics or {},
        "grid_res": [int(x) for x in model.sdf_grid.res],
    }

    for opt_name, opt in optimizers.items():
        checkpoint[f"{opt_name}_state"] = opt.state_dict()

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load checkpoint from disk and restore model and optimizer states."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=device or "cpu", weights_only=False)

    # Check if grid resolution in checkpoint differs from current model resolution
    saved_res = checkpoint.get("grid_res", None)
    if saved_res is not None:
        curr_res = [int(x) for x in model.sdf_grid.res]
        if saved_res != curr_res:
            model.sdf_grid.scale_volume_grid(saved_res)

    model.load_state_dict(checkpoint["model_state"])

    if optimizers is not None:
        for opt_name, opt in optimizers.items():
            state_key = f"{opt_name}_state"
            if state_key in checkpoint:
                opt.load_state_dict(checkpoint[state_key])

    return checkpoint
