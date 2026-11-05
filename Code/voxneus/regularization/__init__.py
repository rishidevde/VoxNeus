"""Regularization module: active vertices V_R extraction, closed-form Eikonal and Curvature gradients."""

from voxneus.regularization.active_vertices import extract_active_vertices
from voxneus.regularization.eikonal import compute_eikonal_loss_and_gradient
from voxneus.regularization.curvature import compute_curvature_loss_and_gradient

__all__ = [
    "extract_active_vertices",
    "compute_eikonal_loss_and_gradient",
    "compute_curvature_loss_and_gradient",
]
