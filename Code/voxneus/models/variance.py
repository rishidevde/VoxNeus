"""Single Variance Network (s Parameter).
Provenance: [PAPER] (Sec. 4.4 & Eq. 16) / [REFERENCE IMPLEMENTATION] (NeuS fields.py SingleVarianceNetwork)
Parameterizes the logistic distribution variance parameter s in Phi_s(x) = sigmoid(s * x).
"""

from typing import Union
import torch
import torch.nn as nn


class SingleVarianceNetwork(nn.Module):
    """Learnable parameter s that controls the standard deviation sigma = 1/s of the logistic density.
    Provenance: [PAPER] / [REFERENCE IMPLEMENTATION] (NeuS SingleVarianceNetwork)
    """

    def __init__(self, init_val: float = 0.3, s_max: float = 300.0):
        """Initialize variance parameter.
        Default init_val = 0.3 gives initial s = exp(0.3 * 10.0) ~ 20.08.
        Provenance: [PAPER] (Sec. 4.4 Eq. 16) / [REFERENCE IMPLEMENTATION] (NeuS SingleVarianceNetwork: clip 1e-6 to 1e6)

        Args:
            init_val: Initial value for the variance parameter (float32 scalar).
            s_max: Upper bound on s (default 300.0 per NeuS paper) to prevent Dirac delta variance explosion.
        """
        super().__init__()
        self.variance = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
        self._s_max = float(s_max)

    def forward(self, x: Union[int, torch.Tensor] = 1) -> torch.Tensor:
        """Return the current scalar or broadcasted s value."""
        s = torch.clamp(torch.exp(self.variance * 10.0), min=1e-4, max=self._s_max)
        if isinstance(x, torch.Tensor):
            return torch.ones_like(x[..., :1]) * s
        elif isinstance(x, int) and x > 1:
            return torch.full((x, 1), s.item(), device=self.variance.device)
        return s

    @property
    def s(self) -> torch.Tensor:
        """Scalar value of s, clamped to [1e-4, s_max]."""
        return torch.clamp(torch.exp(self.variance * 10.0), min=1e-4, max=self._s_max)

    @property
    def std(self) -> torch.Tensor:
        """Standard deviation sigma = 1 / s."""
        return 1.0 / self.s

