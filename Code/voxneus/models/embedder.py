"""Positional Encoding and Spherical Harmonics Embedder.
Provenance: [REFERENCE IMPLEMENTATION] (NeRF / Fourier Features / NeuS)
Used to encode viewing direction vectors for view-dependent radiance synthesis.
"""

from typing import Tuple, List, Callable
import math
import torch
import torch.nn as nn


class PositionalEmbedder(nn.Module):
    """Frequency-based positional encoding: gamma(x) = [x, sin(2^k pi x), cos(2^k pi x), ...]."""

    def __init__(
        self,
        input_dims: int = 3,
        num_freqs: int = 4,
        include_input: bool = True,
        log_sampling: bool = True,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input

        out_dim = input_dims if include_input else 0

        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        else:
            freq_bands = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs)

        self.register_buffer("freq_bands", freq_bands)
        out_dim += input_dims * num_freqs * 2
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to input tensor x."""
        outputs = [x] if self.include_input else []
        for freq in self.freq_bands:
            outputs.append(torch.sin(x * freq))
            outputs.append(torch.cos(x * freq))
        return torch.cat(outputs, dim=-1)


class SphericalHarmonicsEmbedder(nn.Module):
    """Spherical Harmonics encoding up to degree 4 for viewing directions."""

    def __init__(self, degree: int = 4):
        super().__init__()
        self.degree = degree
        self.out_dim = degree ** 2

    def forward(self, dirs: torch.Tensor) -> torch.Tensor:
        """dirs: (..., 3) unit direction vectors."""
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        results = []

        # Degree 0
        results.append(torch.full_like(x, 0.28209479177387814))

        if self.degree >= 2:
            # Degree 1
            results.append(-0.4886025119029199 * y)
            results.append(0.4886025119029199 * z)
            results.append(-0.4886025119029199 * x)

        if self.degree >= 3:
            # Degree 2
            results.append(1.0925484305920792 * x * y)
            results.append(-1.0925484305920792 * y * z)
            results.append(0.31539156525252005 * (2.0 * z * z - x * x - y * y))
            results.append(-1.0925484305920792 * x * z)
            results.append(0.5462742152960396 * (x * x - y * y))

        if self.degree >= 4:
            # Degree 3
            results.append(-0.5900435899266435 * y * (3.0 * x * x - y * y))
            results.append(2.890611442640554 * x * y * z)
            results.append(-0.4570457994644658 * y * (4.0 * z * z - x * x - y * y))
            results.append(0.3731763325901154 * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y))
            results.append(-0.4570457994644658 * x * (4.0 * z * z - x * x - y * y))
            results.append(1.445305721320277 * z * (x * x - y * y))
            results.append(-0.5900435899266435 * x * (x * x - 3.0 * y * y))

        return torch.cat(results, dim=-1)
