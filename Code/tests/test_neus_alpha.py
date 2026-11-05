"""Unit Tests for VoxNeuS Eq. 2 Discrete Opacity Formulation.
Provenance: [PAPER] (Eq. 2 of VoxNeuS paper)
"""

import unittest
import torch
from voxneus.rendering.neus_alpha import compute_neus_alpha


class TestNeuSAlpha(unittest.TestCase):

    def test_alpha_peak_at_zero_crossing(self):
        # Ray marching towards a surface: ray_dir = (0, 0, -1), normal = (0, 0, 1)
        # cos_theta = <n, v> = -1
        ray_dirs = torch.tensor([[0.0, 0.0, -1.0]])
        normals = torch.tensor([[0.0, 0.0, 1.0]])
        dists = torch.tensor([[0.05]])
        s = torch.tensor(50.0)

        # 1. At zero-crossing: sdf = 0.0
        alpha_zero = compute_neus_alpha(
            sdf=torch.tensor([[0.0]]),
            normals=normals,
            ray_dirs=ray_dirs,
            dists=dists,
            s=s,
        )

        # 2. Far outside surface: sdf = 1.0
        alpha_far = compute_neus_alpha(
            sdf=torch.tensor([[1.0]]),
            normals=normals,
            ray_dirs=ray_dirs,
            dists=dists,
            s=s,
        )

        # Opacity should peak near zero-crossing and be near zero far away
        self.assertGreater(alpha_zero.item(), alpha_far.item())
        self.assertGreater(alpha_zero.item(), 0.5)
        self.assertLess(alpha_far.item(), 1e-4)

    def test_alpha_bounds(self):
        # Generate random values and verify alpha is strictly in [0, 1]
        torch.manual_seed(42)
        N = 1000
        sdf = torch.randn(N, 1)
        normals = torch.randn(N, 3)
        normals = normals / torch.linalg.norm(normals, dim=-1, keepdim=True)
        ray_dirs = torch.randn(N, 3)
        ray_dirs = ray_dirs / torch.linalg.norm(ray_dirs, dim=-1, keepdim=True)
        dists = torch.rand(N, 1) * 0.1
        s = torch.tensor(30.0)

        alpha = compute_neus_alpha(sdf, normals, ray_dirs, dists, s)
        self.assertTrue(torch.all(alpha >= 0.0))
        self.assertTrue(torch.all(alpha <= 1.0))
        self.assertFalse(torch.any(torch.isnan(alpha)))


if __name__ == "__main__":
    unittest.main()
