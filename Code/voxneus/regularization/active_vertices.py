"""Active Ray-Passing Grid Vertices Extraction.
Provenance: [PAPER] (Eq. 9 of VoxNeuS paper)
V_R = {c_i(x) | i=1...2^d, x in B}
Extracts the unique set of grid vertices touched by the current ray batch.
Ensures uniform regularization weighting without over-regularizing frequent near-surface points.
"""

from typing import Tuple
import torch


def extract_active_vertices(
    corner_indices: torch.Tensor,
    grid_res: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract the unique set of active grid vertices V_R traversed by the current batch.
    Provenance: [PAPER] (Eq. 9 of VoxNeuS paper)

    Args:
        corner_indices: (..., 8, 3) Integer grid coordinates of traversed corners.
        grid_res: (3,) Resolution along X, Y, Z.

    Returns:
        unique_vertices: (V, 3) Int64 coordinates of unique active vertices V_R.
        linear_indices: (V,) 1D flattened linear indices in [0, X*Y*Z - 1].
    """
    res_x, res_y, res_z = int(grid_res[0]), int(grid_res[1]), int(grid_res[2])

    # Flatten all corner indices across batch and samples
    flat_corners = corner_indices.reshape(-1, 3).to(torch.int64)

    # Clamp safely within grid bounds
    flat_corners[:, 0] = torch.clamp(flat_corners[:, 0], 0, res_x - 1)
    flat_corners[:, 1] = torch.clamp(flat_corners[:, 1], 0, res_y - 1)
    flat_corners[:, 2] = torch.clamp(flat_corners[:, 2], 0, res_z - 1)

    # Compute 1D linear indices for fast 1D deduplication: ix * (Y * Z) + iy * Z + iz
    linear = flat_corners[:, 0] * (res_y * res_z) + flat_corners[:, 1] * res_z + flat_corners[:, 2]

    # Deduplicate unique active vertices
    unique_linear = torch.unique(linear)

    # Convert back to (V, 3) 3D coordinates
    iz = unique_linear % res_z
    rem = unique_linear // res_z
    iy = rem % res_y
    ix = rem // res_y

    unique_vertices = torch.stack([ix, iy, iz], dim=-1)
    return unique_vertices, unique_linear
