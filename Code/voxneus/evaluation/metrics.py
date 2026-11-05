"""Evaluation Metrics: PSNR, SSIM, and LPIPS.
Provenance: [REFERENCE IMPLEMENTATION]
"""

from typing import Union, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn.functional as F


def compute_psnr(mse: Union[float, torch.Tensor]) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) from Mean Squared Error."""
    if isinstance(mse, torch.Tensor):
        mse_val = mse.item()
    else:
        mse_val = float(mse)
    if mse_val <= 0.0:
        return 100.0
    return -10.0 * math.log10(max(mse_val, 1e-10))


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    channel: int = 3,
) -> float:
    """Compute Structural Similarity Index (SSIM) between two images in [0, 1].

    Args:
        img1: (H, W, C) or (1, C, H, W) Tensor.
        img2: (H, W, C) or (1, C, H, W) Tensor.

    Returns:
        ssim: Scalar SSIM value in [0, 1].
    """
    if img1.ndim == 3:
        img1 = img1.permute(2, 0, 1).unsqueeze(0)
    if img2.ndim == 3:
        img2 = img2.permute(2, 0, 1).unsqueeze(0)

    # Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    window_1d = (gauss / gauss.sum()).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous().to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()
