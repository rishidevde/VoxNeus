"""Ray Generation and Camera Helpers.
Provenance: [REFERENCE IMPLEMENTATION]
Adapted from NeRF (Code/nerf/run_nerf_helpers.py) and NeuS (models/dataset.py).
"""

from typing import Tuple, Union, Optional
import numpy as np
import torch


def get_rays(
    H: int,
    W: int,
    K: Union[torch.Tensor, np.ndarray],
    c2w: torch.Tensor,
    normalize_dir: bool = True,
    convention: str = "opencv",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate camera rays (origins and directions) from a pinhole camera.

    Args:
        H: Image height in pixels.
        W: Image width in pixels.
        K: 3x3 intrinsic matrix or 4x4 projection matrix.
        c2w: 3x4 or 4x4 camera-to-world transformation matrix.
        normalize_dir: If True, ray directions are normalized to unit L2 norm.
        convention: "opencv" (+X right, +Y down, +Z forward) or "opengl" (+X right, +Y up, -Z forward).

    Returns:
        rays_o: (H, W, 3) or (N, 3) Ray origins in world coordinates.
        rays_d: (H, W, 3) or (N, 3) Ray directions in world coordinates.
    """
    device = c2w.device
    if not isinstance(K, torch.Tensor):
        K = torch.tensor(K, dtype=torch.float32, device=device)
    else:
        K = K.to(device=device, dtype=torch.float32)

    # Pixel coordinates (0 to W-1, 0 to H-1) centered at pixel centers (+0.5)
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device) + 0.5,
        torch.arange(H, dtype=torch.float32, device=device) + 0.5,
        indexing="xy",
    )

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    if convention.lower() == "opencv":
        # OpenCV: x right, y down, z forward
        dirs = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], dim=-1)
    else:
        # OpenGL: x right, y up, -z forward (NeRF default)
        dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], dim=-1)

    # Rotate ray directions to world frame: rays_d = dirs @ R^T
    rot = c2w[:3, :3]
    rays_d = torch.matmul(dirs, rot.T)

    if normalize_dir:
        rays_d = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True)

    # Ray origin is camera position in world coordinates
    trans = c2w[:3, 3]
    rays_o = trans.expand(rays_d.shape)

    return rays_o, rays_d


def get_rays_np(
    H: int,
    W: int,
    K: np.ndarray,
    c2w: np.ndarray,
    normalize_dir: bool = True,
    convention: str = "opencv",
) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy version of get_rays."""
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + 0.5,
        np.arange(H, dtype=np.float32) + 0.5,
        indexing="xy",
    )
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    if convention.lower() == "opencv":
        dirs = np.stack([(i - cx) / fx, (j - cy) / fy, np.ones_like(i)], axis=-1)
    else:
        dirs = np.stack([(i - cx) / fx, -(j - cy) / fy, -np.ones_like(i)], axis=-1)

    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)
    if normalize_dir:
        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)

    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape).copy()
    return rays_o, rays_d


def near_far_from_sphere(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    radius: float = 1.0,
    default_near: float = 0.05,
    default_far: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate near and far ray bounds centered on the bounding sphere.
    Provenance: [REFERENCE IMPLEMENTATION] (NeuS dataset.py lines 160-166)

    Args:
        rays_o: (..., 3) ray origins.
        rays_d: (..., 3) ray directions (unit norm).
        radius: Sphere radius (default 1.0 for normalized DTU).

    Returns:
        near: (..., 1) near clipping distance.
        far: (..., 1) far clipping distance.
    """
    a = torch.sum(rays_d**2, dim=-1, keepdim=True)
    b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = 0.5 * (-b) / a

    near = torch.clamp(mid - radius, min=default_near)
    far = torch.clamp(mid + radius, min=near + 0.1)

    return near, far


def near_far_from_aabb(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    xyz_min: torch.Tensor,
    xyz_max: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate near and far clipping distances for an Axis-Aligned Bounding Box (AABB) using Slab method.
    Provenance: [REFERENCE IMPLEMENTATION]
    """
    inv_d = 1.0 / (rays_d + torch.sign(rays_d) * eps)
    t_min = (xyz_min - rays_o) * inv_d
    t_max = (xyz_max - rays_o) * inv_d

    t1 = torch.minimum(t_min, t_max)
    t2 = torch.maximum(t_min, t_max)

    near = torch.amax(t1, dim=-1, keepdim=True)
    far = torch.amin(t2, dim=-1, keepdim=True)

    near = torch.clamp(near, min=1e-3)
    far = torch.clamp(far, min=near + 1e-3)
    return near, far
