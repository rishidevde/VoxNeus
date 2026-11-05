"""SDF Grid Total Variation (TV) Regularization.
Provenance: [PAPER] / [REFERENCE IMPLEMENTATION] (Voxurf grid.py & voxurf_womask_fine.py)
Computes 3D Total Variation loss over the dense SDF voxel grid:
L_TV = mean(|f[x+1,y,z] - f[x,y,z]| + |f[x,y+1,z] - f[x,y,z]| + |f[x,y,z+1] - f[x,y,z]|)

Without TV regularization on non-sampled voxels, only 0.02% of grid voxels directly hit by
sampled rays receive gradients, leaving 99.98% of the grid frozen as the initial sphere.
TV loss smoothly propagates SDF deformations from ray-sampled voxels to adjacent grid volume.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sdf_tv_loss(
    grid: torch.Tensor,
    mask: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute 3D Total Variation (TV) loss on dense SDF volume.

    Args:
        grid: (1, 1, Res_x, Res_y, Res_z) Dense SDF grid tensor.
        mask: Optional (1, 1, Res_x, Res_y, Res_z) boolean/float mask for non-empty region.
        eps: Small constant for numerical stability under square root.

    Returns:
        tv_loss: Scalar 3D Total Variation loss tensor.
    """
    if grid.ndim == 5:
        data = grid.squeeze(0).squeeze(0)
    else:
        data = grid

    diff_x = torch.abs(data[1:, :, :] - data[:-1, :, :])
    diff_y = torch.abs(data[:, 1:, :] - data[:, :-1, :])
    diff_z = torch.abs(data[:, :, 1:] - data[:, :, :-1])

    if mask is not None:
        if mask.ndim == 5:
            m = mask.squeeze(0).squeeze(0).float()
        else:
            m = mask.float()
        mask_x = m[1:, :, :] * m[:-1, :, :]
        mask_y = m[:, 1:, :] * m[:, :-1, :]
        mask_z = m[:, :, 1:] * m[:, :, :-1]

        tv_x = torch.sum(diff_x * mask_x) / (torch.sum(mask_x) + 1e-5)
        tv_y = torch.sum(diff_y * mask_y) / (torch.sum(mask_y) + 1e-5)
        tv_z = torch.sum(diff_z * mask_z) / (torch.sum(mask_z) + 1e-5)
    else:
        tv_x = torch.mean(diff_x)
        tv_y = torch.mean(diff_y)
        tv_z = torch.mean(diff_z)

    return tv_x + tv_y + tv_z
