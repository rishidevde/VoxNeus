"""VoxNeuS Main Training CLI Entrypoint.
Provenance: [PAPER] (Sec. 5.1: 40k steps, 2048 rays/batch)
Usage:
    python scripts/train.py --config configs/dtu_default.yaml --data_dir /path/to/DTU/scan24 --output_dir logs/scan24
"""

import os
import sys
import argparse
import yaml
import torch

# Ensure package root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from voxneus.models.voxneus import VoxNeuS
from voxneus.datasets.dtu import DTUDataset
from voxneus.datasets.blendedmvs import BlendedMVSDataset
from voxneus.training.trainer import VoxNeuSTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train VoxNeuS model")
    parser.add_argument("--config", type=str, default="configs/dtu_default.yaml", help="Path to YAML configuration")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory (e.g., DTU scan folder)")
    parser.add_argument("--output_dir", type=str, default="logs/default_run", help="Output directory for checkpoints and logs")
    parser.add_argument("--total_steps", type=int, default=None, help="Override total steps (e.g., 5 for debug smoke test)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override ray batch size (default 2048)")
    parser.add_argument("--gradient_mode", type=str, default=None, choices=["interpolated", "analytical"], help="Gradient mode")
    parser.add_argument("--use_mask", action="store_true", help="Enable foreground mask supervision")
    parser.add_argument("--no_mask", action="store_true", help="Explicitly disable foreground mask supervision (w/o mask paper setting)")
    parser.add_argument("--reso_level", type=int, default=2, help="Image downsample level (1 for full, 2 for half)")
    parser.add_argument("--device", type=str, default=None, help="Target device: 'cuda', 'cpu', or None for auto")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # CLI Overrides
    if args.total_steps is not None:
        config.setdefault("training", {})["total_steps"] = args.total_steps
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.gradient_mode is not None:
        config.setdefault("model", {})["gradient_mode"] = args.gradient_mode
    if args.use_mask:
        config.setdefault("training", {})["use_mask"] = True
    elif args.no_mask:
        config.setdefault("training", {})["use_mask"] = False

    # 2. Setup device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VoxNeuS] Using device: {device}")

    # 3. Load Dataset
    dataset_name = config.get("dataset", {}).get("name", "dtu").lower()
    if dataset_name == "dtu":
        dataset = DTUDataset(
            basedir=args.data_dir,
            reso_level=args.reso_level,
            normalize=config.get("dataset", {}).get("normalize", True),
            use_mask=config.get("training", {}).get("use_mask", False),
        )
    elif dataset_name == "blendedmvs":
        dataset = BlendedMVSDataset(basedir=args.data_dir)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # 4. Instantiate VoxNeuS Model
    model_cfg = config.get("model", {})
    model = VoxNeuS(
        grid_res=model_cfg.get("grid_res", 96),
        xyz_min=tuple(model_cfg.get("xyz_min", [-1.0, -1.0, -1.0])),
        xyz_max=tuple(model_cfg.get("xyz_max", [1.0, 1.0, 1.0])),
        gradient_mode=model_cfg.get("gradient_mode", "interpolated"),
        init_s_val=model_cfg.get("init_s_val", 0.3),
        s_max=model_cfg.get("s_max", 300.0),             # [PAPER SOTA] Bounded s parameter (200-300) prevents Dirac delta explosion
        hash_n_levels=model_cfg.get("hash_n_levels", 16),
        hash_n_features=model_cfg.get("hash_n_features", 2),
        hash_log2_size=model_cfg.get("hash_log2_size", 19),
        geom_feature_dim=model_cfg.get("geom_feature_dim", 0),
        mlp_hidden_dim=model_cfg.get("mlp_hidden_dim", 64),
        view_embed_type=model_cfg.get("view_embed_type", "sh"),
        entangled=model_cfg.get("entangled", False),
    )

    # 5. Build Trainer & Run
    trainer = VoxNeuSTrainer(
        model=model,
        dataset=dataset,
        config=config,
        output_dir=args.output_dir,
        device=device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
