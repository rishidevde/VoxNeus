"""Unit Tests for Progressive Grid Super-Resolution.
Provenance: [PAPER] (Sec. 4.4 & 5.1: 96^3 -> 160^3 -> 320^3)
"""

import unittest
import torch

from voxneus.models.sdf_grid import DenseSDFGrid
from voxneus.geometry.upsample import upsample_sdf_grid, get_resolution_for_step


class TestGridUpsampling(unittest.TestCase):

    def test_progressive_schedule_milestones(self):
        self.assertEqual(get_resolution_for_step(0), 96)
        self.assertEqual(get_resolution_for_step(9999), 96)
        self.assertEqual(get_resolution_for_step(10000), 160)
        self.assertEqual(get_resolution_for_step(29999), 160)
        self.assertEqual(get_resolution_for_step(30000), 320)
        self.assertEqual(get_resolution_for_step(40000), 320)

    def test_sphere_sdf_preservation_under_upsampling(self):
        # Create sphere of radius 0.5 at 32^3
        grid = DenseSDFGrid(resolution=32, init_mode="sphere", sphere_radius=0.5)

        # Upsample from 32^3 to 64^3
        grid.scale_volume_grid(64)
        self.assertEqual(list(grid.grid.shape), [1, 1, 64, 64, 64])

        # Query points on the sphere surface: distance from origin = 0.5
        test_points = torch.tensor([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [-0.5, 0.0, 0.0],
        ], dtype=torch.float32)

        sdf_vals = grid.query_sdf(test_points)
        # Values should remain close to 0.0 (the sphere zero-level set)
        for val in sdf_vals.squeeze():
            self.assertAlmostEqual(val.item(), 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
