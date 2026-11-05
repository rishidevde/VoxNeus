"""Occupancy / Density Grid for Accelerated Ray Marching.
Provenance: [PAPER] (NeuS2 Sec. 3.1 & Sec. 3.2 / Instant-NGP)
Maintains an EMA-updated 3D occupancy grid using NeuS2 logistic density phi_s(d)
to skip empty space and accelerate volume rendering by 5-10x.
"""

from typing import Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def logistic_density(sdf: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """NeuS2 logistic density distribution: phi_s(x) = s * e^{-sx} / (1 + e^{-sx})^2.
    Equivalently: phi_s(x) = s * sigmoid(s * x) * (1 - sigmoid(s * x)).
    Provenance: [PAPER] (NeuS2 Sec. 3.1)
    """
    sig = torch.sigmoid(s * sdf)
    density = s * sig * (1.0 - sig)
    return density


class OccupancyGrid(nn.Module):
    """3D Occupancy Grid for empty space skipping during volume rendering.
    Provenance: [PAPER] (NeuS2 Sec. 3.1 & Instant-NGP)
    """

    def __init__(
        self,
        resolution: int = 128,
        xyz_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
        xyz_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        decay: float = 0.95,
        threshold: float = 0.01,
    ):
        super().__init__()
        self.resolution = resolution
        self.register_buffer("xyz_min", torch.tensor(xyz_min, dtype=torch.float32))
        self.register_buffer("xyz_max", torch.tensor(xyz_max, dtype=torch.float32))
        self.decay = decay
        self.threshold = threshold

        # Store continuous float density values for EMA updating
        self.register_buffer(
            "density_grid",
            torch.zeros((resolution, resolution, resolution), dtype=torch.float32),
        )
        # Store binary occupancy mask (1 = occupied / near surface, 0 = empty space)
        self.register_buffer(
            "occupancy_mask",
            torch.ones((resolution, resolution, resolution), dtype=torch.bool),
        )
        self.total_updates = 0

    @property
    def occupied_percentage(self) -> float:
        """Return percentage of grid volume currently marked as occupied."""
        return (self.occupancy_mask.sum().item() / self.occupancy_mask.numel()) * 100.0

    @torch.no_grad()
    def update_from_sdf(
        self,
        query_sdf_fn,
        s_val: torch.Tensor,
        subsample_ratio: float = 0.25,
    ) -> None:
        """Update occupancy grid with Exponential Moving Average (EMA) using logistic density.
        Provenance: [PAPER] (NeuS2 Sec. 3.1)

        Args:
            query_sdf_fn: Callable taking (N, 3) world coords and returning (N, 1) SDF values.
            s_val: Current variance scaling factor s.
            subsample_ratio: Fraction of grid cells to evaluate this step (for speed).
        """
        device = self.density_grid.device
        res = self.resolution

        # Subsample random voxel centers or full grid
        total_cells = res * res * res
        n_samples = max(int(total_cells * subsample_ratio), 1024)

        # Uniform random cell coordinates in [0, res - 1]
        idx_x = torch.randint(0, res, (n_samples,), device=device)
        idx_y = torch.randint(0, res, (n_samples,), device=device)
        idx_z = torch.randint(0, res, (n_samples,), device=device)

        # Map to physical world space
        step = (self.xyz_max - self.xyz_min) / float(res)
        norm_x = (idx_x.float() + 0.5) / float(res)
        norm_y = (idx_y.float() + 0.5) / float(res)
        norm_z = (idx_z.float() + 0.5) / float(res)

        pts_world = torch.stack([
            self.xyz_min[0] + norm_x * (self.xyz_max[0] - self.xyz_min[0]),
            self.xyz_min[1] + norm_y * (self.xyz_max[1] - self.xyz_min[1]),
            self.xyz_min[2] + norm_z * (self.xyz_max[2] - self.xyz_min[2]),
        ], dim=-1)

        # Query SDF and compute logistic density: phi_s(d) = s * sigmoid(sd) * (1 - sigmoid(sd))
        sdf_vals = query_sdf_fn(pts_world).squeeze(-1)
        densities = logistic_density(sdf_vals, s_val)

        # Also consider surface proximity: if |d| < 2 * voxel_size, mark as occupied
        voxel_diag = torch.linalg.norm(step).item()
        near_surface = (torch.abs(sdf_vals) < 2.0 * voxel_diag).float()
        effective_density = torch.maximum(densities, near_surface * (self.threshold * 2.0))

        # Update density grid with EMA: density = max(decay * prev, new)
        prev_densities = self.density_grid[idx_x, idx_y, idx_z]
        if self.total_updates == 0:
            updated = effective_density
        else:
            updated = torch.maximum(prev_densities * self.decay, effective_density)

        self.density_grid[idx_x, idx_y, idx_z] = updated
        self.occupancy_mask[idx_x, idx_y, idx_z] = updated > self.threshold
        self.total_updates += 1

    @torch.no_grad()
    def is_occupied(self, pts_world: torch.Tensor) -> torch.Tensor:
        """Test whether points in world space reside in occupied voxels.

        Args:
            pts_world: (..., 3) 3D world coordinates.

        Returns:
            occupied: (..., ) Boolean tensor indicating occupied cells.
        """
        res = self.resolution
        shape = pts_world.shape[:-1]
        flat_pts = pts_world.reshape(-1, 3)

        # Normalized coordinates in [0, 1]^3
        norm = (flat_pts - self.xyz_min) / (self.xyz_max - self.xyz_min + 1e-8)
        in_bounds = (norm >= 0.0) & (norm <= 1.0)
        in_bounds_all = in_bounds.all(dim=-1)

        # Integer indices in [0, res - 1]
        coords = torch.clamp(norm * float(res), 0.0, float(res - 1.0001)).to(torch.int64)

        occupied = self.occupancy_mask[coords[:, 0], coords[:, 1], coords[:, 2]]
        # Points outside the bounding box are considered unoccupied
        occupied = occupied & in_bounds_all

        return occupied.reshape(shape)

    # Alias for API flexibility
    query_occupancy = is_occupied
