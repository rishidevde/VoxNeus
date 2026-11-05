"""Rendering module containing camera ray generation, sampling, and NeuS volume rendering."""

from voxneus.rendering.rays import get_rays, get_rays_np, near_far_from_sphere, near_far_from_aabb
from voxneus.rendering.sampler import sample_pdf, RaySampler
from voxneus.rendering.neus_alpha import compute_neus_alpha
from voxneus.rendering.volume_render import composite_weights, volume_render_from_samples, compute_huber_color_loss
from voxneus.rendering.occupancy_grid import OccupancyGrid, logistic_density

__all__ = [
    "get_rays",
    "get_rays_np",
    "near_far_from_sphere",
    "near_far_from_aabb",
    "sample_pdf",
    "RaySampler",
    "compute_neus_alpha",
    "composite_weights",
    "volume_render_from_samples",
    "compute_huber_color_loss",
    "OccupancyGrid",
    "logistic_density",
]
