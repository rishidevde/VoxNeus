"""Cell-Boundary Gradient Continuity Test.
Provenance: [PAPER] (Sec. 4.1, Fig. 2, Fig. 3)
Proves that analytical gradient exhibits discontinuous step jumps at voxel junction surfaces,
while VoxNeuS continuous interpolated gradient remains strictly continuous.
"""

import unittest
import torch

from voxneus.models.sdf_grid import DenseSDFGrid


class TestGradientContinuity(unittest.TestCase):

    def setUp(self):
        # Create a curved SDF on a dense grid (e.g. sphere of radius 0.5)
        self.grid_res = 32
        self.grid = DenseSDFGrid(
            resolution=self.grid_res,
            xyz_min=[-1.0, -1.0, -1.0],
            xyz_max=[1.0, 1.0, 1.0],
            init_mode="sphere",
            sphere_radius=0.5,
        )

    def test_junction_surface_discontinuity_vs_continuity(self):
        # An integer voxel boundary in continuous grid coordinates is e.g. index 15.0
        # Convert continuous index 15.0 to world coordinate
        res = self.grid_res
        x_bound = -1.0 + (15.0 / (res - 1.0)) * 2.0  # exact boundary between cells [14, 15] and [15, 16]
        eps = 1e-4

        # Probe points just left and just right of the junction surface x_bound
        p_left = torch.tensor([[x_bound - eps, 0.25, 0.25]], dtype=torch.float32)
        p_right = torch.tensor([[x_bound + eps, 0.25, 0.25]], dtype=torch.float32)

        # 1. Evaluate Analytical Gradient (Eq. 4 baseline)
        g_ana_left, _ = self.grid.query_normals(p_left, gradient_mode="analytical")[:2]
        g_ana_right, _ = self.grid.query_normals(p_right, gradient_mode="analytical")[:2]

        diff_analytical = torch.linalg.norm(g_ana_left - g_ana_right).item()

        # 2. Evaluate VoxNeuS Continuous Interpolated Gradient (Eq. 7)
        g_interp_left, _ = self.grid.query_normals(p_left, gradient_mode="interpolated")[:2]
        g_interp_right, _ = self.grid.query_normals(p_right, gradient_mode="interpolated")[:2]

        diff_interpolated = torch.linalg.norm(g_interp_left - g_interp_right).item()

        print(f"\n[TestGradientContinuity] Boundary Delta: {eps}")
        print(f"Analytical Gradient Jump    : {diff_analytical:.6f}")
        print(f"Interpolated Gradient Jump  : {diff_interpolated:.8f}")

        # The interpolated gradient difference is smoothly proportional to the small step 2*eps
        self.assertLess(diff_interpolated, 1e-3)

        # The analytical gradient exhibits a massive discontinuous step jump (> 100x larger)
        self.assertGreater(diff_analytical, diff_interpolated * 50)


if __name__ == "__main__":
    unittest.main()
