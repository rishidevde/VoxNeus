"""Geometry module: coordinate transformations, gradient interpolation, upsampling, and mesh extraction."""

from voxneus.geometry.coordinates import (
    world_to_grid_index,
    grid_index_to_world,
    world_to_normalized_device_coords,
    get_voxel_corners_and_weights,
)
from voxneus.geometry.gradients import (
    compute_vertex_finite_differences,
    query_vertex_normals_at_corners,
    interpolate_gradients,
    compute_analytical_gradient,
)
from voxneus.geometry.upsample import upsample_sdf_grid, get_resolution_for_step
from voxneus.geometry.mesh import (
    apply_gaussian_filter_to_grid,
    extract_mesh_marching_cubes,
    save_mesh_ply,
    save_mesh_obj,
)

__all__ = [
    "world_to_grid_index",
    "grid_index_to_world",
    "world_to_normalized_device_coords",
    "get_voxel_corners_and_weights",
    "compute_vertex_finite_differences",
    "query_vertex_normals_at_corners",
    "interpolate_gradients",
    "compute_analytical_gradient",
    "upsample_sdf_grid",
    "get_resolution_for_step",
    "apply_gaussian_filter_to_grid",
    "extract_mesh_marching_cubes",
    "save_mesh_ply",
    "save_mesh_obj",
]
