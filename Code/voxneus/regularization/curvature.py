"""Discrete Laplacian Curvature Regularization.
Provenance: [PAPER] (Sec. 4.2 & Eq. 12-14 of VoxNeuS paper)
L_curv = 1 / |V_R| * sum_{x in V_R} ||nabla^2 f[x]||_2^2
Calculates closed-form discrete Laplacian curvature and its exact gradient G_curv.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from voxneus.cuda import cuda_curvature_grad


def compute_curvature_loss_and_gradient(
    grid_data: torch.Tensor,
    active_vertices: torch.Tensor,
    voxel_size: torch.Tensor,
    grid_res: torch.Tensor,
    accumulate_grad: Optional[torch.Tensor] = None,
    weight: float = 1.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute discrete Laplacian curvature loss and its exact closed-form gradient on active vertices V_R.
    Provenance: [PAPER] (Eq. 12, Eq. 13, Eq. 14) / [DERIVED] chain rule.

    Args:
        grid_data: (1, 1, Res_x, Res_y, Res_z) or (Res_x, Res_y, Res_z) SDF values.
        active_vertices: (V, 3) Coordinates of active vertices V_R.
        voxel_size: (3,) Physical spacing epsilon = [eps_x, eps_y, eps_z].
        grid_res: (3,) Resolution along X, Y, Z.
        accumulate_grad: Optional target tensor (e.g. grid.grad) to accumulate directly in CUDA via atomicAdd.
        weight: Regularization weight w_curv for scaled accumulation.

    Returns:
        loss_val: Scalar curvature loss tensor.
        grad_sdf: (1, 1, Res_x, Res_y, Res_z) Closed-form gradient tensor G_curv, or None if accumulated in-place.
    """
    if grid_data.ndim == 5:
        data = grid_data.squeeze(0).squeeze(0)
    else:
        data = grid_data

    res_x, res_y, res_z = int(grid_res[0]), int(grid_res[1]), int(grid_res[2])
    num_active = active_vertices.shape[0]

    if num_active == 0:
        zero_grad = None if accumulate_grad is not None else torch.zeros([1, 1, res_x, res_y, res_z], dtype=data.dtype, device=data.device)
        return torch.tensor(0.0, device=data.device), zero_grad

    # Direct in-place CUDA accumulation if accumulate_grad is provided
    if data.is_cuda and accumulate_grad is not None:
        target_tensor = accumulate_grad.squeeze(0).squeeze(0) if accumulate_grad.ndim == 5 else accumulate_grad
        if cuda_curvature_grad(data, active_vertices, target_tensor, voxel_size, weight=weight):
            with torch.no_grad():
                ix_a = active_vertices[:, 0]
                iy_a = active_vertices[:, 1]
                iz_a = active_vertices[:, 2]
                vmask = (
                    (ix_a > 0) & (ix_a < res_x - 1) &
                    (iy_a > 0) & (iy_a < res_y - 1) &
                    (iz_a > 0) & (iz_a < res_z - 1)
                )
                if vmask.any():
                    vix = ix_a[vmask]; viy = iy_a[vmask]; viz = iz_a[vmask]
                    ex, ey, ez = voxel_size[0], voxel_size[1], voxel_size[2]
                    fc = data[vix, viy, viz]
                    Lx = (data[vix+1, viy, viz] + data[vix-1, viy, viz] - 2.0*fc) / (ex*ex)
                    Ly = (data[vix, viy+1, viz] + data[vix, viy-1, viz] - 2.0*fc) / (ey*ey)
                    Lz = (data[vix, viy, viz+1] + data[vix, viy, viz-1] - 2.0*fc) / (ez*ez)
                    loss_val = torch.mean(Lx**2 + Ly**2 + Lz**2)
                else:
                    loss_val = torch.tensor(0.0, device=data.device)
            return loss_val, None

    grad_sdf = torch.zeros([res_x, res_y, res_z], dtype=data.dtype, device=data.device)

    # Fast CUDA path if compiled backend is present (separate grad tensor)
    if data.is_cuda and cuda_curvature_grad(data, active_vertices, grad_sdf, voxel_size, weight=1.0):
        with torch.no_grad():
            ix_a = active_vertices[:, 0]
            iy_a = active_vertices[:, 1]
            iz_a = active_vertices[:, 2]
            vmask = (
                (ix_a > 0) & (ix_a < res_x - 1) &
                (iy_a > 0) & (iy_a < res_y - 1) &
                (iz_a > 0) & (iz_a < res_z - 1)
            )
            if vmask.any():
                vix = ix_a[vmask]; viy = iy_a[vmask]; viz = iz_a[vmask]
                ex, ey, ez = voxel_size[0], voxel_size[1], voxel_size[2]
                fc = data[vix, viy, viz]
                Lx = (data[vix+1, viy, viz] + data[vix-1, viy, viz] - 2.0*fc) / (ex*ex)
                Ly = (data[vix, viy+1, viz] + data[vix, viy-1, viz] - 2.0*fc) / (ey*ey)
                Lz = (data[vix, viy, viz+1] + data[vix, viy, viz-1] - 2.0*fc) / (ez*ez)
                loss_val = torch.mean(Lx**2 + Ly**2 + Lz**2)
            else:
                loss_val = torch.tensor(0.0, device=data.device)
        return loss_val, grad_sdf.unsqueeze(0).unsqueeze(0)

    eps_x, eps_y, eps_z = voxel_size[0], voxel_size[1], voxel_size[2]
    eps_x_sq = eps_x * eps_x
    eps_y_sq = eps_y * eps_y
    eps_z_sq = eps_z * eps_z

    ix = active_vertices[:, 0]
    iy = active_vertices[:, 1]
    iz = active_vertices[:, 2]

    # Filter to interior active vertices to ensure exact 6-neighborhood for discrete Laplacian (Eq. 12)
    valid_mask = (
        (ix > 0) & (ix < res_x - 1) &
        (iy > 0) & (iy < res_y - 1) &
        (iz > 0) & (iz < res_z - 1)
    )

    if not valid_mask.any():
        return torch.tensor(0.0, device=data.device), grad_sdf.unsqueeze(0).unsqueeze(0)

    ix = ix[valid_mask]
    iy = iy[valid_mask]
    iz = iz[valid_mask]

    ix_p1 = ix + 1
    ix_m1 = ix - 1

    iy_p1 = iy + 1
    iy_m1 = iy - 1

    iz_p1 = iz + 1
    iz_m1 = iz - 1

    f_center = data[ix, iy, iz]

    # Discrete Laplacian components (Eq. 12)
    # L_x = (f[x+1] + f[x-1] - 2*f[x]) / eps_x^2
    Lx = (data[ix_p1, iy, iz] + data[ix_m1, iy, iz] - 2.0 * f_center) / eps_x_sq
    Ly = (data[ix, iy_p1, iz] + data[ix, iy_m1, iz] - 2.0 * f_center) / eps_y_sq
    Lz = (data[ix, iy, iz_p1] + data[ix, iy, iz_m1] - 2.0 * f_center) / eps_z_sq

    # 1. Curvature Loss: 1 / |V_R| * sum (Lx^2 + Ly^2 + Lz^2) (Eq. 13)
    loss_val = torch.mean(Lx ** 2 + Ly ** 2 + Lz ** 2)

    # 2. Closed-form gradient (Eq. 14): G_curv = 2 / |V_R| * sum nabla^2 f * d(nabla^2 f) / df
    scale = 2.0 / num_active

    grad_center = scale * (-2.0 * Lx / eps_x_sq - 2.0 * Ly / eps_y_sq - 2.0 * Lz / eps_z_sq)
    grad_xp = scale * (Lx / eps_x_sq)
    grad_xm = scale * (Lx / eps_x_sq)

    grad_yp = scale * (Ly / eps_y_sq)
    grad_ym = scale * (Ly / eps_y_sq)

    grad_zp = scale * (Lz / eps_z_sq)
    grad_zm = scale * (Lz / eps_z_sq)

    # Accumulate onto grid vertices
    grad_sdf.index_put_((ix, iy, iz), grad_center, accumulate=True)
    grad_sdf.index_put_((ix_p1, iy, iz), grad_xp, accumulate=True)
    grad_sdf.index_put_((ix_m1, iy, iz), grad_xm, accumulate=True)

    grad_sdf.index_put_((ix, iy_p1, iz), grad_yp, accumulate=True)
    grad_sdf.index_put_((ix, iy_m1, iz), grad_ym, accumulate=True)

    grad_sdf.index_put_((ix, iy, iz_p1), grad_zp, accumulate=True)
    grad_sdf.index_put_((ix, iy, iz_m1), grad_zm, accumulate=True)

    return loss_val, grad_sdf.unsqueeze(0).unsqueeze(0)

