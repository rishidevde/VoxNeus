"""VoxNeuS Multi-Stage Loss & Parameter Schedulers.
Provenance: [PAPER] (Sec. 4.4, Sec. 5.1, Eq. 16)
Implements:
1. s-gradient amplification (k=5 when dL/ds < 0)
2. 3-stage Eikonal weight schedule
3. 3-stage Curvature weight schedule
"""

import math
from typing import Tuple
import torch


def amplify_s_gradient(variance_param: torch.nn.Parameter, k: float = 5.0) -> bool:
    """Amplify negative gradient of s by k times (towards convergence).
    Provenance: [PAPER] (Eq. 16 of VoxNeuS paper, following BakedSDF)
    nabla_s L <- k * nabla_s L if nabla_s L < 0 else nabla_s L

    Args:
        variance_param: nn.Parameter holding variance network weights.
        k: Amplification factor (k=5 for all paper experiments).

    Returns:
        is_amplified: True if gradient was negative and amplified, False otherwise.
    """
    if variance_param.grad is None:
        return False

    with torch.no_grad():
        if variance_param.grad.item() < 0.0:
            variance_param.grad.mul_(k)
            return True
        return False


class VoxNeuSScheduler:
    """Multi-stage regularizer weight scheduler.
    Provenance: [PAPER] (Sec. 5.1 of VoxNeuS paper)
    """

    def __init__(
        self,
        total_steps: int = 40000,
        stage1_end: int = 11000,
        stage2_end: int = 21000,
        w_eik_init: float = 1e-2,
        w_eik_final: float = 1e-3,
        w_curv_init: float = 1e-8,
        w_curv_peak: float = 5e-6,
        w_curv_final: float = 5e-7,
    ):
        self.total_steps = total_steps
        self.stage1_end = stage1_end
        self.stage2_end = stage2_end

        self.w_eik_init = w_eik_init
        self.w_eik_final = w_eik_final

        self.w_curv_init = w_curv_init
        self.w_curv_peak = w_curv_peak
        self.w_curv_final = w_curv_final

    def get_w_eik(self, step: int) -> float:
        """Eikonal loss weight schedule.
        - Steps 0 to 11k: Constant 10^-2
        - Steps 11k to 21k: Linear decrease from 10^-2 to 10^-3
        - Steps 21k to 40k: Constant 10^-3
        """
        if step <= self.stage1_end:
            return self.w_eik_init
        elif step <= self.stage2_end:
            progress = (step - self.stage1_end) / (self.stage2_end - self.stage1_end)
            return self.w_eik_init + progress * (self.w_eik_final - self.w_eik_init)
        else:
            return self.w_eik_final

    def get_w_curv(self, step: int) -> float:
        """Curvature loss weight schedule.
        - Steps 0 to 11k: Constant 10^-8
        - Steps 11k to 21k: Linear increase from 10^-8 to 5 * 10^-6
        - Steps 21k to 40k: Exponential decrease from 5 * 10^-6 to 5 * 10^-7
        """
        if step <= self.stage1_end:
            return self.w_curv_init
        elif step <= self.stage2_end:
            progress = (step - self.stage1_end) / (self.stage2_end - self.stage1_end)
            return self.w_curv_init + progress * (self.w_curv_peak - self.w_curv_init)
        else:
            # Exponential decay from w_curv_peak to w_curv_final
            remaining_steps = self.total_steps - self.stage2_end
            progress = (step - self.stage2_end) / max(remaining_steps, 1)
            # w = peak * (final / peak)^progress
            ratio = self.w_curv_final / self.w_curv_peak
            return self.w_curv_peak * (ratio ** progress)

    def get_weights(self, step: int) -> Tuple[float, float]:
        """Return (w_eik, w_curv) for the current step."""
        return self.get_w_eik(step), self.get_w_curv(step)

    def get_lambda(
        self,
        step: int,
        start_lambda: float = 2.0,
        step_ratio: float = 0.025,
        max_lambda: float = 16.0,
    ) -> float:
        """NeuS2 progressive hash bandwidth schedule: lambda starts at 2 and increases by 1
        every 2.5% of total training steps up to max_lambda.
        Provenance: [PAPER] (NeuS2 Sec. 3.3)
        """
        step_interval = max(int(self.total_steps * step_ratio), 1)
        added_levels = step / float(step_interval)
        current_lambda = min(start_lambda + added_levels, max_lambda)
        return current_lambda


def compute_neus2_lambda(
    step: int,
    total_steps: int = 40000,
    start_lambda: float = 2.0,
    step_ratio: float = 0.025,
    max_lambda: float = 16.0,
) -> float:
    """Standalone helper to compute NeuS2 active lambda levels for given step."""
    step_interval = max(int(total_steps * step_ratio), 1)
    added_levels = step / float(step_interval)
    return min(start_lambda + added_levels, max_lambda)
