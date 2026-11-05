"""Progressive Grid Super-Resolution Engine.
Provenance: [PAPER] (Sec. 4.4 & 5.1) / [REFERENCE IMPLEMENTATION] (Voxurf grid.py scale_volume_grid)
Handles trilinear upsampling across resolution transitions: 96^3 -> 160^3 -> 320^3.
"""

from typing import Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample_sdf_grid(
    grid_tensor: torch.Tensor,
    new_res: Union[List[int], Tuple[int, int, int], int],
) -> torch.Tensor:
    """Upsample dense 3D SDF volume to a higher target resolution using trilinear interpolation.
    Provenance: [PAPER] / [REFERENCE IMPLEMENTATION] (Voxurf scale_volume_grid)

    Args:
        grid_tensor: (1, 1, Res_x, Res_y, Res_z) Current dense SDF parameter tensor.
        new_res: Target resolution tuple (New_x, New_y, New_z) or integer for cubic.

    Returns:
        upsampled_grid: (1, 1, New_x, New_y, New_z) Interpolated SDF grid.
    """
    if isinstance(new_res, int):
        new_shape = (new_res, new_res, new_res)
    else:
        new_shape = tuple(new_res)

    with torch.no_grad():
        upsampled = F.interpolate(
            grid_tensor.data,
            size=new_shape,
            mode="trilinear",
            align_corners=True,
        )

    return upsampled


def get_resolution_for_step(
    step: int,
    milestones: Tuple[int, int] = (10000, 30000),
    resolutions: Tuple[int, int, int] = (96, 160, 320),
) -> int:
    """Return the paper-specified dense SDF grid resolution for a given training iteration.
    Provenance: [PAPER] (Sec. 5.1: 96^3 for 0-10k, 160^3 for 10k-30k, 320^3 for 30k-40k).

    Args:
        step: Current training iteration (0 to 40,000).
        milestones: Steps at which resolution increases (10k, 30k).
        resolutions: Grid resolutions corresponding to each stage (96, 160, 320).

    Returns:
        resolution: int (96, 160, or 320).
    """
    if step < milestones[0]:
        return resolutions[0]
    elif step < milestones[1]:
        return resolutions[1]
    else:
        return resolutions[2]
