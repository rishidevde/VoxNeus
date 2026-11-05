"""Explicit Dense 3D SDF Voxel Grid.
Provenance: [PAPER] (Sec. 4.1 & 4.4) / [REFERENCE IMPLEMENTATION] (Voxurf grid.py DenseGrid)
Stores signed distance function values directly in a 3D float32 tensor.
"""

from typing import Tuple, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from voxneus.geometry.coordinates import (
    world_to_grid_index,
    world_to_normalized_device_coords,
    get_voxel_corners_and_weights,
)
from voxneus.geometry.gradients import (
    interpolate_gradients,
    compute_analytical_gradient,
)
from voxneus.geometry.upsample import upsample_sdf_grid


class DenseSDFGrid(nn.Module):
    """Dense 3D SDF Voxel Grid representing scene geometry explicitly.
    Provenance: [PAPER] / [REFERENCE IMPLEMENTATION] (Voxurf DenseGrid)
    """

    def __init__(
        self,
        resolution: Union[int, List[int], Tuple[int, int, int]] = 96,
        xyz_min: Union[List[float], torch.Tensor] = (-1.0, -1.0, -1.0),
        xyz_max: Union[List[float], torch.Tensor] = (1.0, 1.0, 1.0),
        init_mode: str = "sphere",
        sphere_radius: float = 0.5,
        feat_dim: int = 0,
    ):
        super().__init__()
        if isinstance(resolution, int):
            self.res = torch.tensor([resolution, resolution, resolution], dtype=torch.float32)
        else:
            self.res = torch.tensor(resolution, dtype=torch.float32)

        self.register_buffer("xyz_min", torch.tensor(xyz_min, dtype=torch.float32))
        self.register_buffer("xyz_max", torch.tensor(xyz_max, dtype=torch.float32))
        self.init_mode = init_mode
        self.sphere_radius = sphere_radius
        self.feat_dim = feat_dim

        res_x, res_y, res_z = int(self.res[0]), int(self.res[1]), int(self.res[2])
        self.grid = nn.Parameter(torch.zeros([1, 1, res_x, res_y, res_z], dtype=torch.float32))
        if feat_dim > 0:
            # NeuS2 geometry feature grid g in R^15 (NeuS2 Sec. 3.1 Eq. 1)
            self.feat_grid = nn.Parameter(torch.randn([1, feat_dim, res_x, res_y, res_z], dtype=torch.float32) * 0.01)
        else:
            self.feat_grid = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize SDF grid values.
        Provenance: [REFERENCE IMPLEMENTATION] (Sphere initialization standard in NeuS/VolSDF/Voxurf).
        """
        res_x, res_y, res_z = int(self.res[0]), int(self.res[1]), int(self.res[2])
        if self.init_mode == "sphere":
            # Compute physical 3D world coordinates for each grid vertex
            x = torch.linspace(self.xyz_min[0].item(), self.xyz_max[0].item(), res_x)
            y = torch.linspace(self.xyz_min[1].item(), self.xyz_max[1].item(), res_y)
            z = torch.linspace(self.xyz_min[2].item(), self.xyz_max[2].item(), res_z)
            gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
            coords = torch.stack([gx, gy, gz], dim=-1)  # (X, Y, Z, 3)

            # SDF of sphere: ||x|| - radius
            dist = torch.linalg.norm(coords, dim=-1) - self.sphere_radius
            with torch.no_grad():
                self.grid.data.copy_(dist.unsqueeze(0).unsqueeze(0))
        elif self.init_mode == "zero":
            with torch.no_grad():
                self.grid.data.zero_()
        else:
            with torch.no_grad():
                self.grid.data.uniform_(-0.05, 0.05)

    @property
    def voxel_size(self) -> torch.Tensor:
        """Physical spacing epsilon = (xyz_max - xyz_min) / (resolution - 1).
        Provenance: [DERIVED]
        """
        return (self.xyz_max - self.xyz_min) / (self.res.to(self.xyz_min.device) - 1.0)

    def query_sdf(self, xyz: torch.Tensor) -> torch.Tensor:
        """Query SDF values at arbitrary 3D continuous world coordinates via trilinear interpolation.
        Provenance: [PAPER] (Eq. 3 of VoxNeuS paper)

        Args:
            xyz: (..., 3) Continuous world coordinates.

        Returns:
            sdf: (..., 1) Interpolated signed distance values.
        """
        orig_shape = xyz.shape[:-1]
        xyz_flat = xyz.reshape(1, 1, 1, -1, 3)

        # Map to [-1, 1]^3 for torch.nn.functional.grid_sample
        # Note: F.grid_sample expects grid coordinates in (x, y, z) order where tensor is (N, C, D, H, W)
        # corresponding to (Z, Y, X).
        norm = (xyz_flat - self.xyz_min) / (self.xyz_max - self.xyz_min + 1e-8)
        norm = norm * 2.0 - 1.0
        # Flip coordinates from (X, Y, Z) to (Z, Y, X) for grid_sample
        norm_sample = norm.flip(-1)

        sdf_sampled = F.grid_sample(
            self.grid,
            norm_sample,
            mode="bilinear",  # PyTorch 5D grid_sample requires "bilinear" string to perform 3D trilinear
            padding_mode="border",
            align_corners=True,
        )

        sdf = sdf_sampled.reshape(1, -1).T.reshape(*orig_shape, 1)
        return sdf

    def query_normals(
        self,
        xyz: torch.Tensor,
        gradient_mode: str = "interpolated",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute surface gradients and unit normals at world coordinates.

        Args:
            xyz: (..., 3) World coordinates.
            gradient_mode: "interpolated" (VoxNeuS Eq. 7) or "analytical" (Eq. 4 baseline).

        Returns:
            grad: (..., 3) Unnormalized gradient.
            normal: (..., 3) Normalized unit surface normal.
            corner_indices: (..., 8, 3) Traversed corner indices.
            weights: (..., 8) Trilinear interpolation weights.
        """
        # Map world coordinates to continuous grid coordinates [0, Res - 1]^3
        coords = world_to_grid_index(xyz, self.xyz_min, self.xyz_max, self.res.to(xyz.device))

        if gradient_mode.lower() == "interpolated":
            grad, normal, corners, weights = interpolate_gradients(
                grid=self.grid,
                continuous_coords=coords,
                voxel_size=self.voxel_size,
                grid_res=self.res.to(xyz.device),
            )
            return grad, normal, corners, weights
        elif gradient_mode.lower() == "analytical":
            grad, normal = compute_analytical_gradient(
                grid=self.grid,
                continuous_coords=coords,
                voxel_size=self.voxel_size,
                grid_res=self.res.to(xyz.device),
            )
            corners, weights = get_voxel_corners_and_weights(coords, self.res.to(xyz.device))
            return grad, normal, corners, weights
        else:
            raise ValueError(f"Unknown gradient mode: {gradient_mode}")

    def query_features(self, xyz: torch.Tensor) -> Optional[torch.Tensor]:
        """Query 15D geometry features at continuous world coordinates via trilinear interpolation.
        Provenance: [PAPER] (NeuS2 Sec. 3.1 Eq. 1: (d, g) = f(e))
        """
        if self.feat_grid is None:
            return None
        orig_shape = xyz.shape[:-1]
        xyz_flat = xyz.reshape(1, 1, 1, -1, 3)
        norm = (xyz_flat - self.xyz_min) / (self.xyz_max - self.xyz_min + 1e-8)
        norm = norm * 2.0 - 1.0
        norm_sample = norm.flip(-1)

        feat_sampled = F.grid_sample(
            self.feat_grid,
            norm_sample,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return feat_sampled.squeeze(0).squeeze(1).squeeze(1).T.reshape(*orig_shape, self.feat_dim)

    def scale_volume_grid(self, new_resolution: Union[int, List[int], Tuple[int, int, int]]) -> None:
        """Upsample SDF grid (and geometry feature grid if present) to new resolution in-place.
        Provenance: [PAPER] (Sec. 4.4 progressive super-resolution).
        """
        upsampled = upsample_sdf_grid(self.grid, new_resolution)
        self.grid = nn.Parameter(upsampled)
        if self.feat_grid is not None:
            upsampled_feats = upsample_sdf_grid(self.feat_grid, new_resolution)
            self.feat_grid = nn.Parameter(upsampled_feats)
        if isinstance(new_resolution, int):
            self.res = torch.tensor([new_resolution, new_resolution, new_resolution], dtype=torch.float32)
        else:
            self.res = torch.tensor(new_resolution, dtype=torch.float32)

    def get_numpy_grid(self) -> np.ndarray:
        """Return 3D NumPy array of the current SDF grid."""
        return self.grid.detach().squeeze().cpu().numpy()
