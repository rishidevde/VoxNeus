"""Coordinate transformations and voxel grid indexing utilities.
Handles world-to-grid mapping, continuous voxel coordinate decomposition,
and 8-corner neighbor queries for trilinear interpolation.
"""

from typing import Tuple, List
import torch


def world_to_grid_index(
    xyz: torch.Tensor,
    xyz_min: torch.Tensor,
    xyz_max: torch.Tensor,
    grid_res: torch.Tensor,
) -> torch.Tensor:
    """Map global continuous world coordinates to continuous grid index space [0, Res - 1]^3.

    Args:
        xyz: (..., 3) Global coordinates in world space.
        xyz_min: (3,) Minimum bounding box boundary in world space.
        xyz_max: (3,) Maximum bounding box boundary in world space.
        grid_res: (3,) Resolution along (X, Y, Z).

    Returns:
        coords: (..., 3) Continuous indices in [0, Res - 1]^3.
    """
    norm = (xyz - xyz_min) / (xyz_max - xyz_min + 1e-8)
    coords = norm * (grid_res - 1.0)
    return coords


def grid_index_to_world(
    coords: torch.Tensor,
    xyz_min: torch.Tensor,
    xyz_max: torch.Tensor,
    grid_res: torch.Tensor,
) -> torch.Tensor:
    """Map continuous grid index coordinates back to global world coordinates."""
    norm = coords / (grid_res - 1.0 + 1e-8)
    xyz = xyz_min + norm * (xyz_max - xyz_min)
    return xyz


def world_to_normalized_device_coords(
    xyz: torch.Tensor,
    xyz_min: torch.Tensor,
    xyz_max: torch.Tensor,
) -> torch.Tensor:
    """Map world coordinates to normalized [-1, 1]^3 device coordinates for F.grid_sample."""
    norm = (xyz - xyz_min) / (xyz_max - xyz_min + 1e-8)
    ndc = norm * 2.0 - 1.0
    return ndc


def get_voxel_corners_and_weights(
    continuous_coords: torch.Tensor,
    grid_res: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose continuous grid coordinates into 8 integer corner indices and trilinear weights.
    Provenance: [DERIVED] from Eq. 3 & Eq. 7 of VoxNeuS paper.

    Args:
        continuous_coords: (..., 3) Continuous coordinates in [0, Res - 1]^3.
        grid_res: (3,) Grid resolution (X, Y, Z).

    Returns:
        corner_indices: (..., 8, 3) Int64 tensor of 8 corner voxel coordinates.
        weights: (..., 8) Float32 tensor of trilinear interpolation weights w_i.
    """
    res_x, res_y, res_z = int(grid_res[0]), int(grid_res[1]), int(grid_res[2])

    # Clamp continuous coords within valid grid domain [0, Res - 1 - 1e-5]
    max_bound = torch.tensor([res_x - 1.0001, res_y - 1.0001, res_z - 1.0001], device=continuous_coords.device)
    min_bound = torch.zeros_like(max_bound)
    coords_clamped = torch.clamp(continuous_coords, min=min_bound, max=max_bound)

    c0 = torch.floor(coords_clamped).to(torch.int64)  # (..., 3)
    c1 = c0 + 1

    # Clamp to boundaries
    c0[..., 0] = torch.clamp(c0[..., 0], 0, res_x - 1)
    c0[..., 1] = torch.clamp(c0[..., 1], 0, res_y - 1)
    c0[..., 2] = torch.clamp(c0[..., 2], 0, res_z - 1)

    c1[..., 0] = torch.clamp(c1[..., 0], 0, res_x - 1)
    c1[..., 1] = torch.clamp(c1[..., 1], 0, res_y - 1)
    c1[..., 2] = torch.clamp(c1[..., 2], 0, res_z - 1)

    # Fractional offsets in [0, 1]
    d = coords_clamped - c0.to(coords_clamped.dtype)
    dx = d[..., 0:1]
    dy = d[..., 1:2]
    dz = d[..., 2:3]

    om_dx = 1.0 - dx
    om_dy = 1.0 - dy
    om_dz = 1.0 - dz

    # 8 corners in binary order:
    # 0: (0,0,0), 1: (0,0,1), 2: (0,1,0), 3: (0,1,1),
    # 4: (1,0,0), 5: (1,0,1), 6: (1,1,0), 7: (1,1,1)
    c000 = torch.stack([c0[..., 0], c0[..., 1], c0[..., 2]], dim=-1)
    c001 = torch.stack([c0[..., 0], c0[..., 1], c1[..., 2]], dim=-1)
    c010 = torch.stack([c0[..., 0], c1[..., 1], c0[..., 2]], dim=-1)
    c011 = torch.stack([c0[..., 0], c1[..., 1], c1[..., 2]], dim=-1)
    c100 = torch.stack([c1[..., 0], c0[..., 1], c0[..., 2]], dim=-1)
    c101 = torch.stack([c1[..., 0], c0[..., 1], c1[..., 2]], dim=-1)
    c110 = torch.stack([c1[..., 0], c1[..., 1], c0[..., 2]], dim=-1)
    c111 = torch.stack([c1[..., 0], c1[..., 1], c1[..., 2]], dim=-1)

    corner_indices = torch.stack([c000, c001, c010, c011, c100, c101, c110, c111], dim=-2)

    # Trilinear weights: w_i = prod_j (1 - |x - c_i|_j)
    w000 = om_dx * om_dy * om_dz
    w001 = om_dx * om_dy * dz
    w010 = om_dx * dy * om_dz
    w011 = om_dx * dy * dz
    w100 = dx * om_dy * om_dz
    w101 = dx * om_dy * dz
    w110 = dx * dy * om_dz
    w111 = dx * dy * dz

    weights = torch.cat([w000, w001, w010, w011, w100, w101, w110, w111], dim=-1)
    return corner_indices, weights
