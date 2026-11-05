"""Ray Point Sampler: Stratified and Hierarchical Inverse-CDF Sampling.
Provenance: [REFERENCE IMPLEMENTATION]
Adapted directly from NeRF (Code/nerf/run_nerf.py) and NeuS (code/models/renderer.py).
"""

from typing import Tuple, Optional, Any
import torch
import torch.nn as nn


def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    det: bool = False,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Sample points from a piecewise constant Probability Density Function (PDF)
    using inverse transform sampling (CDF inversion).
    Provenance: [REFERENCE IMPLEMENTATION] (NeRF Eq. 5 / NeuS renderer.py lines 39-69)

    Args:
        bins: (batch_size, n_bins) Bin edge locations.
        weights: (batch_size, n_bins - 1) Weights of each interval.
        n_samples: Number of samples to draw per ray.
        det: If True, uses deterministic uniform linspace; else stochastic uniform rand.
        eps: Small epsilon to prevent zero division and NaN.

    Returns:
        samples: (batch_size, n_samples) Sampled locations along ray.
    """
    # Prevent NaNs from zero weights
    weights = weights + eps
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (batch, n_bins)

    # Sample uniform values u in [0, 1]
    batch_shape = list(cdf.shape[:-1])
    if det:
        u = torch.linspace(0.5 / n_samples, 1.0 - 0.5 / n_samples, steps=n_samples, device=cdf.device)
        u = u.expand(batch_shape + [n_samples])
    else:
        u = torch.rand(batch_shape + [n_samples], device=cdf.device)

    u = u.contiguous()

    # Invert CDF with searchsorted
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)  # (batch, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_expanded = cdf.unsqueeze(1).expand(matched_shape)
    bins_expanded = bins.unsqueeze(1).expand(matched_shape)

    cdf_g = torch.gather(cdf_expanded, 2, inds_g)
    bins_g = torch.gather(bins_expanded, 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class RaySampler(nn.Module):
    """Hierarchical Ray Sampler for neural volumetric rendering.
    Supports uniform stratified sampling and coarse-to-fine importance sampling.
    """

    def __init__(
        self,
        n_samples: int = 64,
        n_importance: int = 64,
        perturb: bool = True,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.perturb = perturb

    def sample_stratified(
        self,
        near: torch.Tensor,
        far: torch.Tensor,
        n_samples: Optional[int] = None,
        perturb: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw stratified samples along rays between near and far planes.

        Args:
            near: (n_rays, 1) Near plane distances.
            far: (n_rays, 1) Far plane distances.
            n_samples: Optional override for sample count.
            perturb: If True, applies stratified jittering; else deterministic linspace.

        Returns:
            z_vals: (n_rays, n_samples) Sample distances along ray.
            dists: (n_rays, n_samples) Segment lengths.
        """
        n_samples = n_samples or self.n_samples
        perturb = self.perturb if perturb is None else perturb

        device = near.device
        t_vals = torch.linspace(0.0, 1.0, steps=n_samples, device=device)
        z_vals = near * (1.0 - t_vals) + far * t_vals

        if perturb:
            # Get intervals between samples
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            # Stratified samples in those intervals
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        # Append last distance as equal to previous or mean
        last_dist = dists[..., -1:] if dists.shape[-1] > 0 else (far - near) / n_samples
        dists = torch.cat([dists, last_dist], dim=-1)

        return z_vals, dists

    def sample_multistep_sdf(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        sdf_fn,
        n_samples: Optional[int] = None,
        n_importance: Optional[int] = None,
        up_sample_steps: int = 4,
        perturb: Optional[bool] = None,
        occupancy_grid: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw samples using multi-step progressive SDF upsampling.
        Provenance: [REFERENCE IMPLEMENTATION] (NeuS Sec. 3 / renderer.py lines 318-350)
        Upgraded with NeuS2 occupancy grid empty-space awareness.

        Args:
            rays_o: (n_rays, 3) Ray origins.
            rays_d: (n_rays, 3) Ray directions.
            near: (n_rays, 1) Near plane bounds.
            far: (n_rays, 1) Far plane bounds.
            sdf_fn: Callable querying SDF value f(x) for points (N, 3).
            n_samples: Number of coarse samples (default self.n_samples = 64).
            n_importance: Number of fine samples (default self.n_importance = 64).
            up_sample_steps: Number of progressive iterations (default 4).
            perturb: Stratified jittering flag.
            occupancy_grid: Optional NeuS2 OccupancyGrid instance for empty space skipping.

        Returns:
            z_vals: (n_rays, n_total) Combined sorted sample locations.
            dists: (n_rays, n_total) Segment lengths.
            mid_z_vals: (n_rays, n_total) Segment midpoints for volume integration.
        """
        n_samples = n_samples or self.n_samples
        n_importance = n_importance if n_importance is not None else self.n_importance
        perturb = self.perturb if perturb is None else perturb

        # 1. Stratified coarse pass
        z_vals, dists = self.sample_stratified(near, far, n_samples=n_samples, perturb=perturb)
        batch_size = z_vals.shape[0]

        if n_importance > 0 and up_sample_steps > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = sdf_fn(pts.reshape(-1, 3)).reshape(batch_size, n_samples)
                samples_per_step = max(n_importance // up_sample_steps, 1)

                for i in range(up_sample_steps):
                    inv_s = 64.0 * (2.0 ** i)
                    new_z_vals = up_sample(
                        rays_o=rays_o,
                        rays_d=rays_d,
                        z_vals=z_vals,
                        sdf=sdf,
                        n_importance=samples_per_step,
                        inv_s=inv_s,
                        det=not perturb,
                        occupancy_grid=occupancy_grid,
                    )
                    z_vals, sdf = cat_z_vals(
                        rays_o=rays_o,
                        rays_d=rays_d,
                        z_vals=z_vals,
                        new_z_vals=new_z_vals,
                        sdf=sdf,
                        sdf_fn=sdf_fn,
                        last=(i + 1 == up_sample_steps),
                    )

        # 2. Compute final segment intervals and midpoints (Paper Eq. 2)
        total_samples = z_vals.shape[-1]
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        last_dist = dists[..., -1:] if dists.shape[-1] > 0 else (far - near) / total_samples
        dists = torch.cat([dists, last_dist], dim=-1)
        mid_z_vals = z_vals + dists * 0.5

        return z_vals, dists, mid_z_vals

    def sample_hierarchical(
        self,
        z_vals: torch.Tensor,
        weights: torch.Tensor,
        n_importance: Optional[int] = None,
        det: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform importance sampling along rays guided by volume rendering weights.

        Args:
            z_vals: (n_rays, n_samples) Coarse sample locations.
            weights: (n_rays, n_samples) Opacity / rendering weights.
            n_importance: Number of fine samples to draw.
            det: Deterministic vs stochastic.

        Returns:
            all_z_vals: (n_rays, n_samples + n_importance) Sorted combined distances.
            all_dists: (n_rays, n_samples + n_importance) Combined segment lengths.
        """
        n_importance = n_importance or self.n_importance
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            bins=z_vals_mid,
            weights=weights[..., 1:-1],
            n_samples=n_importance,
            det=det,
        )

        # Merge and sort
        all_z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        all_dists = all_z_vals[..., 1:] - all_z_vals[..., :-1]
        last_dist = all_dists[..., -1:] if all_dists.shape[-1] > 0 else torch.zeros_like(all_dists[..., :1])
        all_dists = torch.cat([all_dists, last_dist], dim=-1)

        return all_z_vals, all_dists


def up_sample(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    z_vals: torch.Tensor,
    sdf: torch.Tensor,
    n_importance: int,
    inv_s: float,
    det: bool = True,
    occupancy_grid: Optional[Any] = None,
) -> torch.Tensor:
    """Up sampling given a fixed inv_s value according to NeuS Sec. 3.
    Provenance: [REFERENCE IMPLEMENTATION] (NeuS renderer.py lines 132-176)
    Upgraded with NeuS2 OccupancyGrid importance modulation.
    """
    batch_size, n_samples = z_vals.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # (n_rays, n_samples, 3)
    radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
    inside_sphere = ((radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)).float()

    sdf = sdf.reshape(batch_size, n_samples)
    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

    # Use min value of [ cos, prev_cos ] to make sampling robust across surface interfaces
    prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=z_vals.device), cos_val[:, :-1]], dim=-1)
    cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
    cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
    cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

    dist = next_z_vals - prev_z_vals
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones([batch_size, 1], device=z_vals.device), 1.0 - alpha + 1e-7], -1), -1
    )[:, :-1]

    # NeuS2 occupancy grid empty-space awareness: modulate weights to focus on occupied voxels
    if occupancy_grid is not None:
        mid_z = (prev_z_vals + next_z_vals) * 0.5
        mid_pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z[..., :, None]
        occ_mask = occupancy_grid.is_occupied(mid_pts)
        weights = weights * (occ_mask.float() * 0.90 + 0.10)

    z_samples = sample_pdf(z_vals, weights, n_importance, det=det).detach()
    return z_samples


def cat_z_vals(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    z_vals: torch.Tensor,
    new_z_vals: torch.Tensor,
    sdf: torch.Tensor,
    sdf_fn,
    last: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate and sort new z_vals along rays, querying new SDF values.
    Provenance: [REFERENCE IMPLEMENTATION] (NeuS renderer.py lines 178-193)
    """
    batch_size, n_samples = z_vals.shape
    _, n_importance = new_z_vals.shape
    z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
    z_vals, index = torch.sort(z_vals, dim=-1)

    if not last:
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        new_sdf = sdf_fn(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
        sdf = torch.cat([sdf, new_sdf], dim=-1)
        xx = torch.arange(batch_size, device=z_vals.device)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
        index_flat = index.reshape(-1)
        sdf = sdf[(xx, index_flat)].reshape(batch_size, n_samples + n_importance)

    return z_vals, sdf

