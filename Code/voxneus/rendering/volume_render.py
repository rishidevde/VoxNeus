"""NeuS Volume Rendering Compositing.
Provenance: [PAPER] / [REFERENCE IMPLEMENTATION] (NeuS code/models/renderer.py)
Implements Eq. 1: C_hat = sum(T_i * alpha_i * c_i) with transmittance, depth, and normal accumulation.
"""

from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
from voxneus.rendering.neus_alpha import compute_neus_alpha


def composite_weights(
    alpha: torch.Tensor,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate accumulated transmittance T_i and volume rendering weights w_i = T_i * alpha_i.

    Args:
        alpha: (batch_size, n_samples) Discrete opacities in [0, 1].
        eps: Small epsilon for numerical stability.

    Returns:
        weights: (batch_size, n_samples) Rendering weights w_i.
        transmittance: (batch_size, n_samples) Accumulated transmittance T_i.
    """
    batch_size = alpha.shape[0]

    # 1 - alpha + eps
    one_minus_alpha = torch.clamp(1.0 - alpha + eps, min=0.0, max=1.0)

    # T_i = prod_{j=1}^{i-1} (1 - alpha_j)
    # prepend 1.0 for the first sample along the ray
    padded = torch.cat([torch.ones([batch_size, 1], device=alpha.device), one_minus_alpha], dim=-1)
    cumprod = torch.cumprod(padded, dim=-1)
    transmittance = cumprod[:, :-1]

    weights = alpha * transmittance
    # NeuS2 early termination: zero out weights for rays that have already saturated opacity (T < 1e-4)
    weights = torch.where(transmittance < 1e-4, torch.zeros_like(weights), weights)
    return weights, transmittance


def compute_huber_color_loss(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    delta: float = 0.1,
) -> torch.Tensor:
    """NeuS2 Huber color loss: robust to specularities and multi-view exposure differences.
    Provenance: [PAPER] (NeuS2 Sec. 3.1 Eq. 4)
    """
    diff = torch.abs(pred_rgb - target_rgb)
    is_small = diff < delta
    loss = torch.where(is_small, 0.5 * (diff ** 2) / delta, diff - 0.5 * delta)
    return torch.mean(loss)


def volume_render_from_samples(
    colors: torch.Tensor,
    sdf: torch.Tensor,
    normals: torch.Tensor,
    ray_dirs: torch.Tensor,
    z_vals: torch.Tensor,
    dists: torch.Tensor,
    s: torch.Tensor,
    pts: Optional[torch.Tensor] = None,
    sphere_radius: Optional[float] = 1.0,
    background_rgb: Optional[torch.Tensor] = None,
    cos_anneal_ratio: float = 0.0,
    use_iter_cos_annealing: bool = False,
) -> Dict[str, torch.Tensor]:
    """Execute complete volume rendering along a batch of rays.

    Args:
        colors: (n_rays, n_samples, 3) Radiance predictions at sample points.
        sdf: (n_rays, n_samples) or (n_rays, n_samples, 1) Signed distance values.
        normals: (n_rays, n_samples, 3) Unit normal vectors.
        ray_dirs: (n_rays, 3) or (n_rays, n_samples, 3) Unit ray direction vectors.
        z_vals: (n_rays, n_samples) Distances along rays.
        dists: (n_rays, n_samples) Sample interval lengths.
        s: Variance parameter.
        pts: Optional (n_rays, n_samples, 3) Sample point positions in world space.
        sphere_radius: Unit bounding sphere radius for foreground masking (default 1.0 per NeuS/Voxurf).
        background_rgb: Optional (3,) or (n_rays, 3) background color.
        cos_anneal_ratio: Annealing ratio for iter_cos (if enabled).
        use_iter_cos_annealing: Whether to use NeuS iter_cos annealing.

    Returns:
        Dictionary containing:
            'rgb': (n_rays, 3) Composited RGB color.
            'depth': (n_rays, 1) Composited depth.
            'normal': (n_rays, 3) Composited surface normal.
            'opacity': (n_rays, 1) Accumulated opacity (sum of weights).
            'weights': (n_rays, n_samples) Sample weights along rays.
            'alpha': (n_rays, n_samples) Discrete opacities.
    """
    if ray_dirs.ndim == 2:
        ray_dirs_expanded = ray_dirs.unsqueeze(1).expand(-1, colors.shape[1], -1)
    else:
        ray_dirs_expanded = ray_dirs

    # 1. Compute discrete opacity alpha_i using VoxNeuS Eq. 2
    alpha = compute_neus_alpha(
        sdf=sdf,
        normals=normals,
        ray_dirs=ray_dirs_expanded,
        dists=dists,
        s=s,
        cos_anneal_ratio=cos_anneal_ratio,
        use_iter_cos_annealing=use_iter_cos_annealing,
    )

    # 2. Inside-sphere masking: zero out alpha outside unit sphere (NeuS lines 249-251 / Voxurf line 828)
    # Prevents background table/distant environment from polluting the foreground SDF grid
    if pts is not None and sphere_radius is not None and sphere_radius > 0:
        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1)
        inside_sphere = (pts_norm < sphere_radius).float().detach()
        alpha = alpha * inside_sphere

    # 3. Compute weights w_i = T_i * alpha_i
    weights, transmittance = composite_weights(alpha)

    # 3. Composite RGB color: sum(w_i * c_i)
    rgb = torch.sum(weights.unsqueeze(-1) * colors, dim=1)

    # 4. Composite depth: sum(w_i * z_i)
    depth = torch.sum(weights * z_vals, dim=-1, keepdim=True)

    # 5. Composite surface normal: sum(w_i * n_i)
    normal = torch.sum(weights.unsqueeze(-1) * normals, dim=1)
    normal = normal / (torch.linalg.norm(normal, dim=-1, keepdim=True) + 1e-8)

    # 6. Foreground opacity
    opacity = torch.sum(weights, dim=-1, keepdim=True)

    # 7. Apply background color if provided
    if background_rgb is not None:
        if isinstance(background_rgb, (float, int)):
            bg = torch.full_like(rgb, background_rgb)
        elif background_rgb.ndim == 1:
            bg = background_rgb.unsqueeze(0).expand_as(rgb)
        else:
            bg = background_rgb
        rgb = rgb + (1.0 - opacity) * bg

    return {
        "rgb": rgb,
        "depth": depth,
        "normal": normal,
        "opacity": opacity,
        "weights": weights,
        "alpha": alpha,
        "transmittance": transmittance,
    }
