"""Unit Tests for Ray Generation and Sampling Engine.
Provenance: [REFERENCE IMPLEMENTATION]
"""

import math
import numpy as np
import torch
import unittest

from voxneus.rendering.rays import get_rays, get_rays_np, near_far_from_sphere
from voxneus.rendering.sampler import sample_pdf, RaySampler


class TestRayEngine(unittest.TestCase):

    def setUp(self):
        self.H, self.W = 100, 100
        self.K = np.array([
            [120.0, 0.0, 50.0],
            [0.0, 120.0, 50.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        # Camera at (0, 0, 3) looking at origin (0, 0, 0)
        self.c2w = torch.eye(4, dtype=torch.float32)
        self.c2w[2, 3] = 3.0

    def test_get_rays_shapes_and_norm(self):
        rays_o, rays_d = get_rays(self.H, self.W, self.K, self.c2w, normalize_dir=True)

        self.assertEqual(rays_o.shape, (self.H, self.W, 3))
        self.assertEqual(rays_d.shape, (self.H, self.W, 3))

        # Check ray origins are camera center (0, 0, 3)
        self.assertTrue(torch.allclose(rays_o[..., 2], torch.tensor(3.0)))

        # Check unit norm of ray directions
        norms = torch.linalg.norm(rays_d, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_numpy_parity(self):
        rays_o_torch, rays_d_torch = get_rays(self.H, self.W, self.K, self.c2w)
        rays_o_np, rays_d_np = get_rays_np(self.H, self.W, self.K, self.c2w.numpy())

        np.testing.assert_allclose(rays_o_torch.numpy(), rays_o_np, atol=1e-5)
        np.testing.assert_allclose(rays_d_torch.numpy(), rays_d_np, atol=1e-5)

    def test_near_far_sphere_intersection(self):
        # Ray passing straight through origin: o=(0,0,3), d=(0,0,-1)
        rays_o = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32)
        rays_d = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)

        # Unit sphere centered at origin: intersections are at z = 1 and z = -1
        # t = 2 and t = 4
        near, far = near_far_from_sphere(rays_o, rays_d, radius=1.0)
        self.assertAlmostEqual(near.item(), 2.0, places=3)
        self.assertAlmostEqual(far.item(), 4.0, places=3)

    def test_sample_pdf(self):
        bins = torch.linspace(0.0, 1.0, 10).unsqueeze(0)  # (1, 10)
        weights = torch.ones(1, 9)  # Uniform weights
        samples = sample_pdf(bins, weights, n_samples=32, det=True)

        self.assertEqual(samples.shape, (1, 32))
        self.assertTrue(torch.all(samples >= 0.0))
        self.assertTrue(torch.all(samples <= 1.0))
        # Deterministic samples should be sorted
        self.assertTrue(torch.all(samples[:, 1:] >= samples[:, :-1]))


if __name__ == "__main__":
    unittest.main()
