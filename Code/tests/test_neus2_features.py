"""Unit Tests for NeuS2 Architectural and Algorithmic Components.
Provenance: [PAPER] (NeuS2 Sec. 3.1, Sec. 3.2, Sec. 3.3)
"""

import math
import unittest
import torch

from voxneus.rendering.occupancy_grid import OccupancyGrid, logistic_density
from voxneus.rendering.volume_render import compute_huber_color_loss, composite_weights
from voxneus.models.radiance_hash import RadianceHashGrid, PyTorchHashGrid
from voxneus.training.scheduler import compute_neus2_lambda


class TestNeuS2Features(unittest.TestCase):

    def test_logistic_density_distribution(self):
        # phi_s(x) = s * sigmoid(s * x) * (1 - sigmoid(s * x))
        s = torch.tensor(50.0)
        # At zero-crossing (x=0), sigmoid(0) = 0.5, so phi_s(0) = s * 0.5 * 0.5 = 0.25 * s
        d_zero = torch.tensor([0.0])
        val_zero = logistic_density(d_zero, s)
        self.assertAlmostEqual(val_zero.item(), 0.25 * 50.0, places=4)

        # Far from surface (x=1.0), density should be near 0
        d_far = torch.tensor([1.0])
        val_far = logistic_density(d_far, s)
        self.assertLess(val_far.item(), 1e-5)

    def test_occupancy_grid_update_and_query(self):
        occ = OccupancyGrid(resolution=32, threshold=0.01)

        # Define synthetic sphere of radius 0.5: d(x) = ||x|| - 0.5
        def query_sphere(pts):
            return torch.linalg.norm(pts, dim=-1, keepdim=True) - 0.5

        # Update occupancy grid
        s_val = torch.tensor(30.0)
        for _ in range(5):
            occ.update_from_sdf(query_sphere, s_val, subsample_ratio=1.0)

        # Query points on surface (norm = 0.5)
        surf_pts = torch.tensor([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ])
        self.assertTrue(torch.all(occ.is_occupied(surf_pts)))

        # Query points far outside the sphere (norm = 0.95, far from surface)
        far_pts = torch.tensor([
            [0.95, 0.95, 0.95],
            [-0.95, -0.95, -0.95],
        ])
        self.assertFalse(torch.all(occ.is_occupied(far_pts)))

    def test_progressive_hash_lambda_mask(self):
        # 16 levels, 2 features per level
        grid = PyTorchHashGrid(n_levels=16, n_features_per_level=2)
        x = torch.rand(10, 3)

        # When lambda = 2.0, only the first 2 levels (features 0:4) should be non-zero
        feats_lambda2 = grid(x, active_levels=2.0)
        self.assertEqual(feats_lambda2.shape, (10, 32))

        # Check that levels 3..16 (indices 4..31) are strictly 0.0
        self.assertTrue(torch.all(feats_lambda2[:, 4:] == 0.0))
        # Check that first 2 levels have non-zero features
        self.assertFalse(torch.all(feats_lambda2[:, :4] == 0.0))

        # When lambda = 8.0, first 8 levels (16 features) should be non-zero, rest 0
        feats_lambda8 = grid(x, active_levels=8.0)
        self.assertTrue(torch.all(feats_lambda8[:, 16:] == 0.0))
        self.assertFalse(torch.all(feats_lambda8[:, :16] == 0.0))

    def test_huber_color_loss(self):
        # Test small errors (|diff| < delta=0.1): loss is 0.5 * diff^2 / delta
        pred_small = torch.tensor([0.5, 0.5, 0.5])
        target_small = torch.tensor([0.52, 0.52, 0.52])  # diff = 0.02 < 0.1
        loss_small = compute_huber_color_loss(pred_small, target_small, delta=0.1)
        expected_small = 0.5 * (0.02 ** 2) / 0.1
        self.assertAlmostEqual(loss_small.item(), expected_small, places=5)

        # Test large errors (|diff| > delta=0.1): loss is diff - 0.5 * delta (linear)
        pred_large = torch.tensor([0.0])
        target_large = torch.tensor([1.0])  # diff = 1.0 > 0.1
        loss_large = compute_huber_color_loss(pred_large, target_large, delta=0.1)
        expected_large = 1.0 - 0.05
        self.assertAlmostEqual(loss_large.item(), expected_large, places=5)

    def test_transmittance_early_cutoff(self):
        # Create opacities where first 2 samples saturate to total opacity
        alpha = torch.zeros(1, 10)
        alpha[0, 0] = 0.999
        alpha[0, 1] = 0.999

        weights, trans = composite_weights(alpha)
        # Beyond sample 2, transmittance should be < 1e-4 and weights zeroed out
        self.assertLess(trans[0, 3].item(), 1e-4)
        self.assertEqual(weights[0, 3].item(), 0.0)
        self.assertEqual(weights[0, 9].item(), 0.0)

    def test_lambda_schedule(self):
        # Starts at 2.0 at step 0
        self.assertEqual(compute_neus2_lambda(0, total_steps=40000), 2.0)
        # Increases by 1 every 2.5% (1000 steps)
        self.assertEqual(compute_neus2_lambda(1000, total_steps=40000), 3.0)
        self.assertEqual(compute_neus2_lambda(2000, total_steps=40000), 4.0)
        # Clamped at max_lambda (16.0)
        self.assertEqual(compute_neus2_lambda(40000, total_steps=40000), 16.0)


if __name__ == "__main__":
    unittest.main()
