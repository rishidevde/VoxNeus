"""VoxNeuS CUDA Extension Interface.
Provides seamless import and dispatch for compiled CUDA C++ backend.
Falls back to pure PyTorch when compiled extension is not available.
"""

from typing import Optional
import importlib
import torch

HAS_CUDA_BACKEND = False
_backend = None

try:
    _backend = importlib.import_module("voxneus.cuda._backend")
    if _backend is not None and hasattr(_backend, "gradient_interp_forward"):
        HAS_CUDA_BACKEND = True
    else:
        HAS_CUDA_BACKEND = False
        _backend = None
except Exception:
    HAS_CUDA_BACKEND = False
    _backend = None


def is_cuda_available() -> bool:
    return HAS_CUDA_BACKEND and (_backend is not None) and torch.cuda.is_available()


class GradientInterpFunction(torch.autograd.Function):
    """Autograd Function for CUDA-accelerated Continuous Gradient Interpolation (Eq. 7).
    Forward: Fused vertex finite differences + trilinear interpolation with local memory access.
    Backward: In-place recomputation of weights and finite-difference Jacobian accumulation via atomicAdd.
    """

    @staticmethod
    def forward(ctx, grid: torch.Tensor, coords: torch.Tensor, voxel_size: torch.Tensor) -> torch.Tensor:
        grid_3d = grid.squeeze(0).squeeze(0) if grid.ndim == 5 else grid
        grid_c = grid_3d.contiguous()
        coords_c = coords.contiguous()
        voxel_size_c = voxel_size.contiguous()

        ctx.save_for_backward(coords_c, voxel_size_c)
        ctx.grid_shape = grid.shape
        ctx.grid_3d_shape = grid_3d.shape
        ctx.needs_grid_grad = grid.requires_grad

        out_grad = _backend.gradient_interp_forward(grid_c, coords_c, voxel_size_c)
        return out_grad

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        coords_c, voxel_size_c = ctx.saved_tensors
        grad_grid = None
        if ctx.needs_grid_grad and ctx.needs_input_grad[0]:
            grad_out_c = grad_out.contiguous()
            grad_grid_3d = torch.zeros(ctx.grid_3d_shape, dtype=grad_out.dtype, device=grad_out.device)
            _backend.gradient_interp_backward(grad_out_c, coords_c, grad_grid_3d, voxel_size_c)
            grad_grid = grad_grid_3d.reshape(ctx.grid_shape)
        return grad_grid, None, None


def cuda_gradient_interp(
    grid: torch.Tensor,
    coords: torch.Tensor,
    voxel_size: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Execute CUDA gradient interpolation with autograd forward and backward support."""
    if not is_cuda_available() or not grid.is_cuda:
        return None
    return GradientInterpFunction.apply(grid, coords, voxel_size)


def cuda_eikonal_grad(
    grid: torch.Tensor,
    active_verts: torch.Tensor,
    grad_grid: torch.Tensor,
    voxel_size: torch.Tensor,
    weight: float = 1.0,
) -> bool:
    """Execute CUDA Eikonal closed-form gradient accumulation if available."""
    if not is_cuda_available() or not grid.is_cuda:
        return False
    grid_c = grid.squeeze(0).squeeze(0).contiguous()
    active_c = active_verts.to(torch.int64).contiguous()
    grad_c = grad_grid.squeeze(0).squeeze(0).contiguous()
    voxel_size_c = voxel_size.contiguous()
    _backend.eikonal_grad(grid_c, active_c, grad_c, voxel_size_c, float(weight))
    return True


def cuda_curvature_grad(
    grid: torch.Tensor,
    active_verts: torch.Tensor,
    grad_grid: torch.Tensor,
    voxel_size: torch.Tensor,
    weight: float = 1.0,
) -> bool:
    """Execute CUDA Curvature closed-form gradient accumulation if available."""
    if not is_cuda_available() or not grid.is_cuda:
        return False
    grid_c = grid.squeeze(0).squeeze(0).contiguous()
    active_c = active_verts.to(torch.int64).contiguous()
    grad_c = grad_grid.squeeze(0).squeeze(0).contiguous()
    voxel_size_c = voxel_size.contiguous()
    _backend.curvature_grad(grid_c, active_c, grad_c, voxel_size_c, float(weight))
    return True
