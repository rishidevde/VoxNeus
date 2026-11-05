"""Novel View Synthesis & Test Image Rendering CLI.
Provenance: [REFERENCE IMPLEMENTATION]
Usage:
    python scripts/render.py --checkpoint logs/scan24/checkpoint_final.pt --data_dir /path/to/DTU/scan24 --output_dir renders/scan24
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from voxneus.models.voxneus import VoxNeuS
from voxneus.datasets.dtu import DTUDataset
from voxneus.rendering.sampler import RaySampler
from voxneus.rendering.rays import near_far_from_sphere
from voxneus.training.checkpoint import load_checkpoint
from voxneus.evaluation.metrics import compute_psnr, compute_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Render test views from VoxNeuS model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="renders", help="Directory to save rendered images")
    parser.add_argument("--chunk_size", type=int, default=2048, help="Ray chunk size for rendering")
    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    return parser.parse_args()


def render_full_image(model, sampler, rays_o, rays_d, chunk_size=2048, device="cpu"):
    """Render full image in chunks to prevent OOM."""
    N_rays = rays_o.shape[0]
    rgb_chunks = []
    depth_chunks = []

    model.eval()
    with torch.no_grad():
        for i in range(0, N_rays, chunk_size):
            o = rays_o[i:i + chunk_size].to(device)
            d = rays_d[i:i + chunk_size].to(device)
            near, far = near_far_from_sphere(o, d, radius=1.0)

            out = model.render(o, d, sampler, near, far)
            rgb_chunks.append(out["rgb"].cpu())
            depth_chunks.append(out["depth"].cpu())

    full_rgb = torch.cat(rgb_chunks, dim=0)
    full_depth = torch.cat(depth_chunks, dim=0)
    return full_rgb, full_depth


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load model
    model = VoxNeuS().to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    sampler = RaySampler(n_samples=64, n_importance=64, perturb=False).to(device)

    # 2. Load dataset
    dataset = DTUDataset(basedir=args.data_dir, reso_level=2)

    psnrs = []
    ssims = []

    print(f"[VoxNeuS] Rendering {len(dataset.test_views)} test views...")
    for idx in dataset.test_views:
        data = dataset.get_test_rays(idx)
        H, W = data["H"], data["W"]
        rays_o = data["rays_o"]
        rays_d = data["rays_d"]
        gt_rgb = data["rgb"].numpy().reshape(H, W, 3)

        pred_rgb, _ = render_full_image(model, sampler, rays_o, rays_d, chunk_size=args.chunk_size, device=device)
        pred_rgb_np = pred_rgb.numpy().reshape(H, W, 3)
        pred_rgb_np = np.clip(pred_rgb_np, 0.0, 1.0)

        mse = np.mean((pred_rgb_np - gt_rgb) ** 2)
        psnr_val = compute_psnr(mse)
        ssim_val = compute_ssim(torch.from_numpy(pred_rgb_np), torch.from_numpy(gt_rgb))
        psnrs.append(psnr_val)
        ssims.append(ssim_val)

        # Save side-by-side comparison
        vis = np.concatenate([gt_rgb, pred_rgb_np], axis=1)
        save_path = os.path.join(args.output_dir, f"view_{idx:03d}_psnr_{psnr_val:.2f}.png")
        Image.fromarray((vis * 255).astype(np.uint8)).save(save_path)
        print(f"View {idx:02d}: PSNR = {psnr_val:.2f} dB, SSIM = {ssim_val:.4f} -> {save_path}")

    mean_psnr = float(np.mean(psnrs))
    mean_ssim = float(np.mean(ssims))
    print(f"\n[VoxNeuS] Test Set Average: PSNR = {mean_psnr:.2f} dB, SSIM = {mean_ssim:.4f}\n")


if __name__ == "__main__":
    main()
