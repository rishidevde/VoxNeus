"""VoxNeuS Continuous Interpolated Gradients Engine.
Provenance: [PAPER]
Implements Eq. 6 (Vertex Finite Differences) and Eq. 7 (Interpolated Gradients),
plus Eq. 4 (Analytical Gradient) for ablation study E1.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from voxneus.geometry.coordinates import (
    world_to_grid_index,
    get_voxel_corners_and_weights,
)
from voxneus.cuda import cuda_gradient_interp


def compute_vertex_finite_differences(
    grid: torch.Tensor,
    voxel_size: torch.Tensor,
) -> torch.Tensor:
    """Compute central finite differences at all grid vertices.
    Provenance: [PAPER] (Eq. 6 of VoxNeuS paper)
    Interior uses central differences; boundaries use one-sided differences.

    Args:
        grid: (1, 1, Res_x, Res_y, Res_z) Dense 3D SDF grid tensor.
        voxel_size: (3,) Physical spacing epsilon = [eps_x, eps_y, eps_z].

    Returns:
        vertex_normals: (Res_x, Res_y, Res_z, 3) Estimated normal vector field at grid vertices.
    """
    # grid shape: (1, 1, X, Y, Z) -> squeeze to (X, Y, Z)
    data = grid.squeeze(0).squeeze(0)
    res_x, res_y, res_z = data.shape
    eps_x, eps_y, eps_z = voxel_size[0], voxel_size[1], voxel_size[2]

    grad_x = torch.empty_like(data)
    if res_x > 2:
        grad_x[1:-1] = (data[2:] - data[:-2]) / (2.0 * eps_x)
    grad_x[0] = (data[1] - data[0]) / eps_x
    grad_x[-1] = (data[-1] - data[-2]) / eps_x

    grad_y = torch.empty_like(data)
    if res_y > 2:
        grad_y[:, 1:-1] = (data[:, 2:] - data[:, :-2]) / (2.0 * eps_y)
    grad_y[:, 0] = (data[:, 1] - data[:, 0]) / eps_y
    grad_y[:, -1] = (data[:, -1] - data[:, -2]) / eps_y

    grad_z = torch.empty_like(data)
    if res_z > 2:
        grad_z[:, :, 1:-1] = (data[:, :, 2:] - data[:, :, :-2]) / (2.0 * eps_z)
    grad_z[:, :, 0] = (data[:, :, 1] - data[:, :, 0]) / eps_z
    grad_z[:, :, -1] = (data[:, :, -1] - data[:, :, -2]) / eps_z

    vertex_normals = torch.stack([grad_x, grad_y, grad_z], dim=-1)
    return vertex_normals


def query_vertex_normals_at_corners(
    grid: torch.Tensor,
    corner_indices: torch.Tensor,
    voxel_size: torch.Tensor,
) -> torch.Tensor:
    """Query finite difference vertex gradients n[c_i(x)] for given corner indices.
    Extracts 6-neighborhood directly for the queried corners to avoid computing full volume if sparse.
    Interior uses central differences; boundaries use one-sided differences matching the CUDA kernel.

    Args:
        grid: (1, 1, Res_x, Res_y, Res_z) Dense SDF grid.
        corner_indices: (..., 8, 3) Corner indices in [0, Res - 1].
        voxel_size: (3,) Voxel spacing epsilon.

    Returns:
        corner_normals: (..., 8, 3) Central finite difference gradients at 8 corners.
    """
    data = grid.squeeze(0).squeeze(0)
    res_x, res_y, res_z = data.shape
    eps_x, eps_y, eps_z = voxel_size[0], voxel_size[1], voxel_size[2]

    ix = corner_indices[..., 0]
    iy = corner_indices[..., 1]
    iz = corner_indices[..., 2]

    ix_p = torch.clamp(ix + 1, max=res_x - 1)
    ix_m = torch.clamp(ix - 1, min=0)
    d_x = torch.clamp((ix_p - ix_m).float() * eps_x, min=eps_x)
    gx = (data[ix_p, iy, iz] - data[ix_m, iy, iz]) / d_x

    iy_p = torch.clamp(iy + 1, max=res_y - 1)
    iy_m = torch.clamp(iy - 1, min=0)
    d_y = torch.clamp((iy_p - iy_m).float() * eps_y, min=eps_y)
    gy = (data[ix, iy_p, iz] - data[ix, iy_m, iz]) / d_y

    iz_p = torch.clamp(iz + 1, max=res_z - 1)
    iz_m = torch.clamp(iz - 1, min=0)
    d_z = torch.clamp((iz_p - iz_m).float() * eps_z, min=eps_z)
    gz = (data[ix, iy, iz_p] - data[ix, iy, iz_m]) / d_z

    corner_normals = torch.stack([gx, gy, gz], dim=-1)
    return corner_normals


def interpolate_gradients(
    grid: torch.Tensor,
    continuous_coords: torch.Tensor,
    voxel_size: torch.Tensor,
    grid_res: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute continuous trilinear interpolated gradients.
    Provenance: [PAPER] (Eq. 7 of VoxNeuS paper)
    grad^{(i)}(x) = sum_{k=1}^8 w_k(x) * n[c_k(x)]

    Args:
        grid: (1, 1, Res_x, Res_y, Res_z) Dense SDF grid.
        continuous_coords: (..., 3) Continuous coordinates in [0, Res - 1]^3.
        voxel_size: (3,) Voxel physical spacing epsilon.
        grid_res: (3,) Grid resolution.

    Returns:
        grad_interp: (..., 3) Unnormalized continuous interpolated gradient grad^{(i)}(x).
        normals: (..., 3) Normalized unit surface normals.
        corner_indices: (..., 8, 3) Queried corner indices.
        weights: (..., 8) Trilinear weights w_i.
    """
    corner_indices, weights = get_voxel_corners_and_weights(continuous_coords, grid_res)

    # CUDA accelerated path if compiled backend is present
    cuda_grad = cuda_gradient_interp(grid, continuous_coords.reshape(-1, 3), voxel_size) if grid.is_cuda else None
    if cuda_grad is not None:
        grad_interp = cuda_grad.reshape(*continuous_coords.shape[:-1], 3)
    else:
        corner_normals = query_vertex_normals_at_corners(grid, corner_indices, voxel_size)
        # grad^{(i)}(x) = sum_{k=1}^8 w_k(x) * n[c_k(x)]
        grad_interp = torch.sum(weights.unsqueeze(-1) * corner_normals, dim=-2)

    norm = torch.linalg.norm(grad_interp, dim=-1, keepdim=True)
    normals = grad_interp / (norm + 1e-8)

    return grad_interp, normals, corner_indices, weights


def compute_analytical_gradient(
    grid: torch.Tensor,
    continuous_coords: torch.Tensor,
    voxel_size: torch.Tensor,
    grid_res: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute analytical gradient by directly differentiating trilinear interpolation.
    Provenance: [PAPER] (Eq. 4 of VoxNeuS paper) - Baseline control for ablation E1.
    grad^{(a)}(x) = sum_{k=1}^8 grad_x w_k(x) * f[c_k(x)]

    Demonstrates discontinuous step jumps across grid cell boundaries.
    """
    res_x, res_y, res_z = int(grid_res[0]), int(grid_res[1]), int(grid_res[2])
    data = grid.squeeze(0).squeeze(0)
    eps_x, eps_y, eps_z = voxel_size[0], voxel_size[1], voxel_size[2]

    max_bound = torch.tensor([res_x - 1.0001, res_y - 1.0001, res_z - 1.0001], device=continuous_coords.device)
    min_bound = torch.zeros_like(max_bound)
    coords_clamped = torch.clamp(continuous_coords, min=min_bound, max=max_bound)

    c0 = torch.floor(coords_clamped).to(torch.int64)
    c1 = c0 + 1

    c0[..., 0] = torch.clamp(c0[..., 0], 0, res_x - 1)
    c0[..., 1] = torch.clamp(c0[..., 1], 0, res_y - 1)
    c0[..., 2] = torch.clamp(c0[..., 2], 0, res_z - 1)

    c1[..., 0] = torch.clamp(c1[..., 0], 0, res_x - 1)
    c1[..., 1] = torch.clamp(c1[..., 1], 0, res_y - 1)
    c1[..., 2] = torch.clamp(c1[..., 2], 0, res_z - 1)

    d = coords_clamped - c0.to(coords_clamped.dtype)
    dx = d[..., 0:1]
    dy = d[..., 1:2]
    dz = d[..., 2:3]

    om_dx = 1.0 - dx
    om_dy = 1.0 - dy
    om_dz = 1.0 - dz

    # Derivatives with respect to continuous grid indices (scaled by 1 / voxel_size for world space)
    # dw / dx_world = (dw / dx_grid) / eps_x
    dw_dx = torch.cat([
        -om_dy * om_dz, -om_dy * dz, -dy * om_dz, -dy * dz,
        om_dy * om_dz, om_dy * dz, dy * om_dz, dy * dz
    ], dim=-1) / eps_x

    dw_dy = torch.cat([
        -om_dx * om_dz, -om_dx * dz, om_dx * om_dz, om_dx * dz,
        -dx * om_dz, -dx * dz, dx * om_dz, dx * dz
    ], dim=-1) / eps_y

    dw_dz = torch.cat([
        -om_dx * om_dy, om_dx * om_dy, -om_dx * dy, om_dx * dy,
        -dx * om_dy, dx * om_dy, -dx * dy, dx * dy
    ], dim=-1) / eps_z

    c000 = torch.stack([c0[..., 0], c0[..., 1], c0[..., 2]], dim=-1)
    c001 = torch.stack([c0[..., 0], c0[..., 1], c1[..., 2]], dim=-1)
    c010 = torch.stack([c0[..., 0], c1[..., 1], c0[..., 2]], dim=-1)
    c011 = torch.stack([c0[..., 0], c1[..., 1], c1[..., 2]], dim=-1)
    c100 = torch.stack([c1[..., 0], c0[..., 1], c0[..., 2]], dim=-1)
    c101 = torch.stack([c1[..., 0], c0[..., 1], c1[..., 2]], dim=-1)
    c110 = torch.stack([c1[..., 0], c1[..., 1], c0[..., 2]], dim=-1)
    c111 = torch.stack([c1[..., 0], c1[..., 1], c1[..., 2]], dim=-1)

    corners = torch.stack([c000, c001, c010, c011, c100, c101, c110, c111], dim=-2)
    # Query 8 corner SDF values
    values = data[corners[..., 0], corners[..., 1], corners[..., 2]]  # (..., 8)

    grad_x = torch.sum(dw_dx * values, dim=-1, keepdim=True)
    grad_y = torch.sum(dw_dy * values, dim=-1, keepdim=True)
    grad_z = torch.sum(dw_dz * values, dim=-1, keepdim=True)

    grad_analytical = torch.cat([grad_x, grad_y, grad_z], dim=-1)
    norm = torch.linalg.norm(grad_analytical, dim=-1, keepdim=True)
    normals = grad_analytical / (norm + 1e-8)

    return grad_analytical, normals
