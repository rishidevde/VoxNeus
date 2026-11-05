"""Multiresolution Hash Grid Encoding for Appearance.
Provenance: [PAPER] (Sec. 4.3 "radiance is parameterized with an iNGP hash grid") /
            [REFERENCE IMPLEMENTATION] (tiny-cuda-nn / Instant-NGP)
Includes native tinycudann integration when available, with a pure PyTorch vectorized fallback.
"""

from typing import Optional, List, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Check if tinycudann is installed and available
HAS_TCNN = False
try:
    import tinycudann as tcnn
    if torch.cuda.is_available():
        HAS_TCNN = True
except (ImportError, Exception):
    HAS_TCNN = False


class PyTorchHashGrid(nn.Module):
    """Pure PyTorch vectorized implementation of Instant-NGP Multiresolution Hash Grid.
    Provenance: [REFERENCE IMPLEMENTATION] (Müller et al. 2022)
    Enables execution on CPU and non-CUDA environments with exact mathematical parity.
    """

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 2048,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2 ** log2_hashmap_size
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.out_dim = n_levels * n_features_per_level

        # Calculate geometric scaling factor b = exp(ln(N_max / N_min) / (L - 1))
        self.b = math.exp((math.log(max_resolution) - math.log(base_resolution)) / max(n_levels - 1, 1))

        # Spatial hash prime constants (Müller et al. 2022)
        # Using uint32 compatible primes: 1, 2654435761, 805459861
        self.register_buffer("primes", torch.tensor([1, 2654435761, 805459861], dtype=torch.int64))

        # Embeddings for each level: (n_levels, hashmap_size, n_features_per_level)
        self.embeddings = nn.Parameter(
            torch.empty(n_levels, self.hashmap_size, n_features_per_level, dtype=torch.float32)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Uniform initialization in [-1e-4, 1e-4] following tiny-cuda-nn
        nn.init.uniform_(self.embeddings, -1e-4, 1e-4)

    def _hash(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute spatial hash for integer coordinates (..., 3)."""
        # coords: (..., 3) int64
        h = (coords[..., 0] * self.primes[0]) ^ (coords[..., 1] * self.primes[1]) ^ (coords[..., 2] * self.primes[2])
        return torch.remainder(h, self.hashmap_size)

    def forward(self, x: torch.Tensor, active_levels: Optional[float] = None) -> torch.Tensor:
        """Query multiresolution hash features with optional NeuS2 progressive bandwidth masking.

        Args:
            x: (..., 3) Normalized spatial coordinates in [0, 1]^3.
            active_levels: Optional lambda parameter in [1, n_levels] controlling active frequency bands.
                           Provenance: [PAPER] (NeuS2 Sec. 3.3 Progressive Training).

        Returns:
            features: (..., n_levels * n_features_per_level)
        """
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, 3)
        device = x.device
        n_points = x_flat.shape[0]

        outputs = []

        for level in range(self.n_levels):
            # Compute level weight from NeuS2 lambda schedule: w_i(lambda)
            if active_levels is not None:
                lvl_idx = level + 1.0
                if lvl_idx <= active_levels:
                    lvl_weight = 1.0
                elif lvl_idx - 1.0 < active_levels < lvl_idx:
                    lvl_weight = active_levels - (lvl_idx - 1.0)
                else:
                    lvl_weight = 0.0
            else:
                lvl_weight = 1.0

            if lvl_weight == 0.0:
                # Level inactive: skip computation and append zeros
                outputs.append(torch.zeros((n_points, self.n_features_per_level), device=device, dtype=torch.float32))
                continue

            # Compute resolution for this level: N_l = floor(N_min * b^level)
            res = math.floor(self.base_resolution * (self.b ** level))
            res_tensor = torch.tensor(res, dtype=torch.float32, device=device)

            # Scale coordinates into continuous voxel indices: [0, res]
            scaled_x = x_flat * res_tensor
            c0 = torch.floor(scaled_x).to(torch.int64)
            c1 = c0 + 1

            # Fractional offsets delta in [0, 1]
            d = scaled_x - c0.to(scaled_x.dtype)
            dx, dy, dz = d[:, 0:1], d[:, 1:2], d[:, 2:3]
            om_dx, om_dy, om_dz = 1.0 - dx, 1.0 - dy, 1.0 - dz

            # 8 corners
            c000 = torch.stack([c0[:, 0], c0[:, 1], c0[:, 2]], dim=-1)
            c001 = torch.stack([c0[:, 0], c0[:, 1], c1[:, 2]], dim=-1)
            c010 = torch.stack([c0[:, 0], c1[:, 1], c0[:, 2]], dim=-1)
            c011 = torch.stack([c0[:, 0], c1[:, 1], c1[:, 2]], dim=-1)
            c100 = torch.stack([c1[:, 0], c0[:, 1], c0[:, 2]], dim=-1)
            c101 = torch.stack([c1[:, 0], c0[:, 1], c1[:, 2]], dim=-1)
            c110 = torch.stack([c1[:, 0], c1[:, 1], c0[:, 2]], dim=-1)
            c111 = torch.stack([c1[:, 0], c1[:, 1], c1[:, 2]], dim=-1)

            h000 = self._hash(c000)
            h001 = self._hash(c001)
            h010 = self._hash(c010)
            h011 = self._hash(c011)
            h100 = self._hash(c100)
            h101 = self._hash(c101)
            h110 = self._hash(c110)
            h111 = self._hash(c111)

            level_embed = self.embeddings[level]

            w000 = om_dx * om_dy * om_dz
            w001 = om_dx * om_dy * dz
            w010 = om_dx * dy * om_dz
            w011 = om_dx * dy * dz
            w100 = dx * om_dy * om_dz
            w101 = dx * om_dy * dz
            w110 = dx * dy * om_dz
            w111 = dx * dy * dz

            feat = (
                w000 * level_embed[h000]
                + w001 * level_embed[h001]
                + w010 * level_embed[h010]
                + w011 * level_embed[h011]
                + w100 * level_embed[h100]
                + w101 * level_embed[h101]
                + w110 * level_embed[h110]
                + w111 * level_embed[h111]
            )
            if lvl_weight < 1.0:
                feat = feat * lvl_weight
            outputs.append(feat)

        all_feats = torch.cat(outputs, dim=-1)
        return all_feats.reshape(*orig_shape, self.out_dim)


class RadianceHashGrid(nn.Module):
    """Unified Multiresolution Hash Grid Encoder with automatic backend dispatch.
    Uses tiny-cuda-nn if available and CUDA is active; otherwise falls back to pure PyTorch.
    Supports NeuS2 progressive bandwidth frequency masking (Sec. 3.3).
    """

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 2048,
        force_pytorch: bool = False,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.out_dim = n_levels * n_features_per_level
        self.use_tcnn = HAS_TCNN and not force_pytorch

        if self.use_tcnn:
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": math.exp(
                    (math.log(max_resolution) - math.log(base_resolution)) / max(n_levels - 1, 1)
                ),
            }
            self.encoder = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config)
        else:
            self.encoder = PyTorchHashGrid(
                n_levels=n_levels,
                n_features_per_level=n_features_per_level,
                log2_hashmap_size=log2_hashmap_size,
                base_resolution=base_resolution,
                max_resolution=max_resolution,
            )

    def forward(self, norm_xyz: torch.Tensor, active_levels: Optional[float] = None) -> torch.Tensor:
        """Query features at normalized coordinates in [0, 1]^3 with optional lambda masking."""
        clamped = torch.clamp(norm_xyz, min=0.0, max=1.0)
        shape = clamped.shape[:-1]

        if self.use_tcnn:
            flat = clamped.reshape(-1, 3).float()
            feats = self.encoder(flat).reshape(*shape, self.n_levels, self.n_features_per_level)
            if active_levels is not None:
                # Apply NeuS2 lambda frequency mask to tcnn output
                weights = torch.zeros((self.n_levels,), device=feats.device, dtype=torch.float32)
                for l in range(self.n_levels):
                    lvl = l + 1.0
                    if lvl <= active_levels:
                        weights[l] = 1.0
                    elif lvl - 1.0 < active_levels < lvl:
                        weights[l] = active_levels - (lvl - 1.0)
                feats = feats * weights.view(1, 1, self.n_levels, 1) if feats.ndim == 4 else feats * weights.view(1, self.n_levels, 1)
            return feats.reshape(*shape, self.out_dim).float()
        else:
            return self.encoder(clamped, active_levels=active_levels)
