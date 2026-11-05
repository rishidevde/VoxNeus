"""Closed-Form Explicit Eikonal Regularization.
Provenance: [PAPER] (Sec. 4.2 & Eq. 10-11 of VoxNeuS paper)
L_eik = 1 / |V_R| * sum_{x in V_R} (||n[x]||_2 - 1)^2
Computes closed-form vector gradient dL_eik / df directly without autograd computation graph.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from voxneus.cuda import cuda_eikonal_grad


def compute_eikonal_loss_and_gradient(
    grid_data: torch.Tensor,
    active_vertices: torch.Tensor,
    voxel_size: torch.Tensor,
    grid_res: torch.Tensor,
    eps: float = 1e-8,
    accumulate_grad: Optional[torch.Tensor] = None,
    weight: float = 1.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute Eikonal loss and its exact closed-form gradient on active vertices V_R.
    Provenance: [PAPER] (Eq. 10 & Eq. 11) / [DERIVED] closed-form vector chain rule.

    Args:
        grid_data: (1, 1, Res_x, Res_y, Res_z) or (Res_x, Res_y, Res_z) SDF grid values.
        active_vertices: (V, 3) Int64 coordinates of unique active vertices V_R.
        voxel_size: (3,) Physical spacing epsilon = [eps_x, eps_y, eps_z].
        grid_res: (3,) Resolution along X, Y, Z.
        eps: Small constant to prevent zero-norm division.
        accumulate_grad: Optional target tensor (e.g. grid.grad) to accumulate directly in CUDA via atomicAdd.
        weight: Regularization weight w_eik for scaled accumulation.

    Returns:
        loss_val: Scalar Eikonal loss tensor (for monitoring).
        grad_sdf: (1, 1, Res_x, Res_y, Res_z) Closed-form gradient tensor G_eik, or None if accumulated in-place.
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
        if cuda_eikonal_grad(data, active_vertices, target_tensor, voxel_size, weight=weight):
            with torch.no_grad():
                ix_all = active_vertices[:, 0]
                iy_all = active_vertices[:, 1]
                iz_all = active_vertices[:, 2]
                vmask = (
                    (ix_all > 0) & (ix_all < res_x - 1) &
                    (iy_all > 0) & (iy_all < res_y - 1) &
                    (iz_all > 0) & (iz_all < res_z - 1)
                )
                if vmask.any():
                    vix = ix_all[vmask]; viy = iy_all[vmask]; viz = iz_all[vmask]
                    eps_x, eps_y, eps_z = voxel_size[0], voxel_size[1], voxel_size[2]
                    nx = (data[vix + 1, viy, viz] - data[vix - 1, viy, viz]) / (2.0 * eps_x)
                    ny = (data[vix, viy + 1, viz] - data[vix, viy - 1, viz]) / (2.0 * eps_y)
                    nz = (data[vix, viy, viz + 1] - data[vix, viy, viz - 1]) / (2.0 * eps_z)
                    n_norm = torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2 + eps ** 2)
                    loss_val = torch.mean((n_norm - 1.0) ** 2)
                else:
                    loss_val = torch.tensor(0.0, device=data.device)
            return loss_val, None

    grad_sdf = torch.zeros([res_x, res_y, res_z], dtype=data.dtype, device=data.device)

    # Fast CUDA path if compiled backend is present (separate grad tensor)
    if data.is_cuda and cuda_eikonal_grad(data, active_vertices, grad_sdf, voxel_size, weight=1.0):
        with torch.no_grad():
            ix_all = active_vertices[:, 0]
            iy_all = active_vertices[:, 1]
            iz_all = active_vertices[:, 2]
            vmask = (
                (ix_all > 0) & (ix_all < res_x - 1) &
                (iy_all > 0) & (iy_all < res_y - 1) &
                (iz_all > 0) & (iz_all < res_z - 1)
            )
            if vmask.any():
                vix = ix_all[vmask]; viy = iy_all[vmask]; viz = iz_all[vmask]
                eps_x, eps_y, eps_z = voxel_size[0], voxel_size[1], voxel_size[2]
                nx = (data[vix + 1, viy, viz] - data[vix - 1, viy, viz]) / (2.0 * eps_x)
                ny = (data[vix, viy + 1, viz] - data[vix, viy - 1, viz]) / (2.0 * eps_y)
                nz = (data[vix, viy, viz + 1] - data[vix, viy, viz - 1]) / (2.0 * eps_z)
                n_norm = torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2 + eps ** 2)
                loss_val = torch.mean((n_norm - 1.0) ** 2)
            else:
                loss_val = torch.tensor(0.0, device=data.device)
        return loss_val, grad_sdf.unsqueeze(0).unsqueeze(0)

    eps_x, eps_y, eps_z = voxel_size[0], voxel_size[1], voxel_size[2]
    ix = active_vertices[:, 0]
    iy = active_vertices[:, 1]
    iz = active_vertices[:, 2]

    # Filter to interior active vertices for exact central finite differences (Eq. 6 / Eq. 10)
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
    denom_x = 2.0 * eps_x

    iy_p1 = iy + 1
    iy_m1 = iy - 1
    denom_y = 2.0 * eps_y

    iz_p1 = iz + 1
    iz_m1 = iz - 1
    denom_z = 2.0 * eps_z

    # Central finite differences at active vertices
    fx_p = data[ix_p1, iy, iz]
    fx_m = data[ix_m1, iy, iz]
    nx = (fx_p - fx_m) / denom_x

    fy_p = data[ix, iy_p1, iz]
    fy_m = data[ix, iy_m1, iz]
    ny = (fy_p - fy_m) / denom_y

    fz_p = data[ix, iy, iz_p1]
    fz_m = data[ix, iy, iz_m1]
    nz = (fz_p - fz_m) / denom_z

    n_vec = torch.stack([nx, ny, nz], dim=-1)  # (V, 3)
    n_norm = torch.linalg.norm(n_vec, dim=-1, keepdim=True)  # (V, 1)

    # 1. Eikonal Loss: 1 / |V_R| * sum (||n|| - 1)^2
    loss_val = torch.mean((n_norm - 1.0) ** 2)

    # 2. Closed-form derivative with respect to normal vector:
    # dL_eik / dn = 2 / |V_R| * (1 - 1 / ||n||) * n
    dL_dn = (2.0 / num_active) * (1.0 - 1.0 / (n_norm + eps)) * n_vec  # (V, 3)

    vx = dL_dn[:, 0]
    vy = dL_dn[:, 1]
    vz = dL_dn[:, 2]

    # 3. Propagate to neighbors via finite-difference Jacobian
    # dn_x / df[x+1] = +1 / denom_x, dn_x / df[x-1] = -1 / denom_x
    # Accumulate into grad_sdf using index_put_ with accumulate=True
    grad_sdf.index_put_((ix_p1, iy, iz), vx / denom_x, accumulate=True)
    grad_sdf.index_put_((ix_m1, iy, iz), -vx / denom_x, accumulate=True)

    grad_sdf.index_put_((ix, iy_p1, iz), vy / denom_y, accumulate=True)
    grad_sdf.index_put_((ix, iy_m1, iz), -vy / denom_y, accumulate=True)

    grad_sdf.index_put_((ix, iy, iz_p1), vz / denom_z, accumulate=True)
    grad_sdf.index_put_((ix, iy, iz_m1), -vz / denom_z, accumulate=True)


    return loss_val, grad_sdf.unsqueeze(0).unsqueeze(0)
