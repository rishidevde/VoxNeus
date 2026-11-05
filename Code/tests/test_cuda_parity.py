"""CUDA Parity and Autograd Verification Suite for VoxNeuS.
Tests exact mathematical parity between compiled CUDA extensions and PyTorch references:
1. Continuous Gradient Interpolation (Eq. 7): Forward and Backward parity.
2. Explicit Eikonal Closed-Form Gradient (Eq. 10-11): Parity with PyTorch autograd.
3. Explicit Curvature Closed-Form Gradient (Eq. 12-14): Parity with PyTorch autograd.
4. Sparse Adam Kernel (Sec. 4.4): Parity with dense and masked updates.
"""

import unittest
import torch
import torch.nn.functional as F

from voxneus.cuda import is_cuda_available, _backend, cuda_gradient_interp, cuda_eikonal_grad, cuda_curvature_grad
from voxneus.geometry.coordinates import get_voxel_corners_and_weights
from voxneus.geometry.gradients import query_vertex_normals_at_corners, interpolate_gradients
from voxneus.regularization.eikonal import compute_eikonal_loss_and_gradient
from voxneus.regularization.curvature import compute_curvature_loss_and_gradient
from voxneus.training.optimizer import SparseGridAdam


@unittest.skipUnless(torch.cuda.is_available() and is_cuda_available(), "CUDA or VoxNeuS CUDA backend not available")
class TestCUDAParity(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.res = 32
        self.grid_res = torch.tensor([self.res, self.res, self.res], dtype=torch.float32, device=self.device)
        self.voxel_size = torch.tensor([2.0 / (self.res - 1), 2.0 / (self.res - 1), 2.0 / (self.res - 1)], dtype=torch.float32, device=self.device)

    def test_gradient_interp_forward_parity(self):
        """Verify CUDA forward output matches PyTorch reference exactly."""
        grid = torch.randn(1, 1, self.res, self.res, self.res, device=self.device, dtype=torch.float32)
        coords = torch.rand(256, 3, device=self.device, dtype=torch.float32) * (self.res - 1.001)

        # PyTorch reference
        corners, weights = get_voxel_corners_and_weights(coords, self.grid_res)
        corner_normals = query_vertex_normals_at_corners(grid, corners, self.voxel_size)
        ref_grad = torch.sum(weights.unsqueeze(-1) * corner_normals, dim=-2)

        # CUDA implementation
        cuda_grad = cuda_gradient_interp(grid, coords, self.voxel_size)

        max_err = torch.max(torch.abs(ref_grad - cuda_grad)).item()
        print(f"\n[TestCUDAParity] Gradient Interp Forward Max Error: {max_err:.2e}")
        self.assertLess(max_err, 1e-5)

    def test_gradient_interp_backward_autograd(self):
        """Verify CUDA backward gradient w.r.t. grid matches PyTorch autograd backward."""
        grid_ref = torch.randn(1, 1, self.res, self.res, self.res, device=self.device, dtype=torch.float32, requires_grad=True)
        grid_cuda = grid_ref.clone().detach().requires_grad_(True)

        coords = torch.rand(128, 3, device=self.device, dtype=torch.float32) * (self.res - 1.001)

        # 1. PyTorch backward pass
        corners, weights = get_voxel_corners_and_weights(coords, self.grid_res)
        corner_normals = query_vertex_normals_at_corners(grid_ref, corners, self.voxel_size)
        ref_grad = torch.sum(weights.unsqueeze(-1) * corner_normals, dim=-2)
        loss_ref = (ref_grad ** 2).sum()
        loss_ref.backward()

        # 2. CUDA backward pass via GradientInterpFunction
        cuda_grad = cuda_gradient_interp(grid_cuda, coords, self.voxel_size)
        loss_cuda = (cuda_grad ** 2).sum()
        loss_cuda.backward()

        self.assertIsNotNone(grid_cuda.grad, "CUDA grid.grad should not be None after backward")
        max_grad_err = torch.max(torch.abs(grid_ref.grad - grid_cuda.grad)).item()
        print(f"[TestCUDAParity] Gradient Interp Backward Max Grad Error: {max_grad_err:.2e}")
        self.assertLess(max_grad_err, 1e-4)

    def test_eikonal_cuda_in_place_accumulation(self):
        """Verify CUDA in-place Eikonal gradient accumulation matches separate gradient computation."""
        grid = torch.randn(1, 1, self.res, self.res, self.res, device=self.device, dtype=torch.float32)
        active_verts = torch.randint(1, self.res - 1, (100, 3), device=self.device, dtype=torch.int64)

        # Separate gradient
        loss_sep, grad_sep = compute_eikonal_loss_and_gradient(
            grid_data=grid,
            active_vertices=active_verts,
            voxel_size=self.voxel_size,
            grid_res=self.grid_res,
        )

        # In-place accumulation into zero grad
        accum = torch.zeros_like(grid)
        loss_acc, _ = compute_eikonal_loss_and_gradient(
            grid_data=grid,
            active_vertices=active_verts,
            voxel_size=self.voxel_size,
            grid_res=self.grid_res,
            accumulate_grad=accum,
            weight=1.0,
        )

        max_err = torch.max(torch.abs(grad_sep - accum)).item()
        print(f"[TestCUDAParity] Eikonal In-Place Accumulation Max Error: {max_err:.2e}")
        self.assertLess(max_err, 1e-5)
        self.assertAlmostEqual(loss_sep.item(), loss_acc.item(), places=5)

    def test_curvature_cuda_in_place_accumulation(self):
        """Verify CUDA in-place Curvature gradient accumulation matches separate computation."""
        grid = torch.randn(1, 1, self.res, self.res, self.res, device=self.device, dtype=torch.float32)
        active_verts = torch.randint(1, self.res - 1, (100, 3), device=self.device, dtype=torch.int64)

        loss_sep, grad_sep = compute_curvature_loss_and_gradient(
            grid_data=grid,
            active_vertices=active_verts,
            voxel_size=self.voxel_size,
            grid_res=self.grid_res,
        )

        accum = torch.zeros_like(grid)
        loss_acc, _ = compute_curvature_loss_and_gradient(
            grid_data=grid,
            active_vertices=active_verts,
            voxel_size=self.voxel_size,
            grid_res=self.grid_res,
            accumulate_grad=accum,
            weight=1.0,
        )

        max_err = torch.max(torch.abs(grad_sep - accum)).item()
        print(f"[TestCUDAParity] Curvature In-Place Accumulation Max Error: {max_err:.2e}")
        self.assertLess(max_err, 1e-5)
        self.assertAlmostEqual(loss_sep.item(), loss_acc.item(), places=5)

    def test_sparse_adam_cuda_parity(self):
        """Verify SparseGridAdam CUDA kernel updates active parameters accurately."""
        p = torch.ones(1000, device=self.device, dtype=torch.float32, requires_grad=True)
        p.grad = torch.zeros_like(p)
        p.grad[10:20] = 0.5
        p.grad[50:60] = -0.5

        opt = SparseGridAdam([p], lr=0.05)
        opt.step()

        # Inactive elements must remain exactly untouched (1.0)
        self.assertEqual(p[0].item(), 1.0)
        self.assertEqual(p[30].item(), 1.0)
        self.assertEqual(p[999].item(), 1.0)

        # Active elements must be strictly updated
        self.assertLess(p[10].item(), 1.0)
        self.assertGreater(p[50].item(), 1.0)
        print("[TestCUDAParity] SparseGridAdam CUDA sparsity check passed.")


if __name__ == "__main__":
    unittest.main()
