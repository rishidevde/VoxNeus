"""Autograd Parity Test for Closed-Form Regularization Gradients.
Provenance: [PAPER] (Sec. 4.2) / [DERIVED]
Validates that our explicit closed-form Eikonal and Curvature gradients
exactly match PyTorch automatic differentiation (< 1e-6 error).
"""

import unittest
import torch

from voxneus.regularization.eikonal import compute_eikonal_loss_and_gradient
from voxneus.regularization.curvature import compute_curvature_loss_and_gradient


class TestRegularizationGradients(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.res = torch.tensor([16, 16, 16], dtype=torch.float32)
        self.voxel_size = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)

        # Active vertices in the interior
        self.active_verts = torch.tensor([
            [5, 5, 5],
            [6, 5, 5],
            [7, 8, 9],
            [10, 11, 12],
            [4, 8, 6],
        ], dtype=torch.int64)

    def test_eikonal_autograd_parity(self):
        # 1. Autograd version
        grid_autograd = torch.randn(16, 16, 16, dtype=torch.float64, requires_grad=True)

        eps_x, eps_y, eps_z = self.voxel_size[0].item(), self.voxel_size[1].item(), self.voxel_size[2].item()
        ix = self.active_verts[:, 0]
        iy = self.active_verts[:, 1]
        iz = self.active_verts[:, 2]

        gx = (grid_autograd[ix + 1, iy, iz] - grid_autograd[ix - 1, iy, iz]) / (2.0 * eps_x)
        gy = (grid_autograd[ix, iy + 1, iz] - grid_autograd[ix, iy - 1, iz]) / (2.0 * eps_y)
        gz = (grid_autograd[ix, iy, iz + 1] - grid_autograd[ix, iy, iz - 1]) / (2.0 * eps_z)

        n_norm = torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-12)
        loss_autograd = torch.mean((n_norm - 1.0) ** 2)
        loss_autograd.backward()

        # 2. Closed-form version
        grid_data = grid_autograd.detach().clone()
        loss_closed_form, grad_closed_form = compute_eikonal_loss_and_gradient(
            grid_data=grid_data,
            active_vertices=self.active_verts,
            voxel_size=self.voxel_size.to(torch.float64),
            grid_res=self.res,
        )

        grad_cf_squeezed = grad_closed_form.squeeze()

        # Compare loss values
        self.assertAlmostEqual(loss_autograd.item(), loss_closed_form.item(), places=6)

        # Compare gradient tensors
        max_error = torch.max(torch.abs(grid_autograd.grad - grad_cf_squeezed)).item()
        print(f"\n[TestRegularization] Max Eikonal Gradient Error vs Autograd: {max_error:.2e}")
        self.assertLess(max_error, 1e-6)

    def test_curvature_autograd_parity(self):
        # 1. Autograd version
        grid_autograd = torch.randn(16, 16, 16, dtype=torch.float64, requires_grad=True)

        eps_x, eps_y, eps_z = self.voxel_size[0].item(), self.voxel_size[1].item(), self.voxel_size[2].item()
        eps_x_sq, eps_y_sq, eps_z_sq = eps_x**2, eps_y**2, eps_z**2

        ix = self.active_verts[:, 0]
        iy = self.active_verts[:, 1]
        iz = self.active_verts[:, 2]

        f_c = grid_autograd[ix, iy, iz]
        Lx = (grid_autograd[ix + 1, iy, iz] + grid_autograd[ix - 1, iy, iz] - 2.0 * f_c) / eps_x_sq
        Ly = (grid_autograd[ix, iy + 1, iz] + grid_autograd[ix, iy - 1, iz] - 2.0 * f_c) / eps_y_sq
        Lz = (grid_autograd[ix, iy, iz + 1] + grid_autograd[ix, iy, iz - 1] - 2.0 * f_c) / eps_z_sq

        loss_autograd = torch.mean(Lx**2 + Ly**2 + Lz**2)
        loss_autograd.backward()

        # 2. Closed-form version
        grid_data = grid_autograd.detach().clone()
        loss_closed_form, grad_closed_form = compute_curvature_loss_and_gradient(
            grid_data=grid_data,
            active_vertices=self.active_verts,
            voxel_size=self.voxel_size.to(torch.float64),
            grid_res=self.res,
        )

        grad_cf_squeezed = grad_closed_form.squeeze()

        # Compare loss values
        self.assertAlmostEqual(loss_autograd.item(), loss_closed_form.item(), places=6)

        # Compare gradient tensors
        max_error = torch.max(torch.abs(grid_autograd.grad - grad_cf_squeezed)).item()
        print(f"[TestRegularization] Max Curvature Gradient Error vs Autograd: {max_error:.2e}")
        self.assertLess(max_error, 1e-6)


if __name__ == "__main__":
    unittest.main()
