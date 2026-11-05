"""Sparse Adam Optimizer for Dense Voxel Grid.
Provenance: [PAPER] (Sec. 4.4 "utilize sparsity of gradient and only update SDF values whose gradient is non-zero") /
            [REFERENCE IMPLEMENTATION] (Voxurf adam_upd_kernel.cu)
Supports native CUDA extension when compiled, and vectorized pure-PyTorch fallback.
"""

from typing import Dict, Any, List, Optional
import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from voxneus.cuda import is_cuda_available, _backend


class SparseGridAdam(Optimizer):
    """Adam optimizer with sparse updates: only modifies weights where grad != 0.
    Provenance: [PAPER] (Sec. 4.4)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-1,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.last_updated_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        total_updated = 0

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SparseGridAdam does not support torch.sparse tensors")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                # Dispatch to high-performance CUDA kernel if available on GPU
                if p.is_cuda and is_cuda_available() and weight_decay == 0.0:
                    _backend.masked_adam_upd(
                        p,
                        grad,
                        exp_avg,
                        exp_avg_sq,
                        int(step),
                        float(beta1),
                        float(beta2),
                        float(lr),
                        float(eps),
                    )
                    # Periodically count non-zero grads for dashboard monitoring without full-grid overhead every step
                    if step % 50 == 0 or self.last_updated_count == 0:
                        self.last_updated_count = (grad != 0.0).sum().item()
                    total_updated += self.last_updated_count
                    continue

                # Vectorized pure-PyTorch CPU fallback
                mask = grad != 0.0
                num_active = mask.sum().item()
                total_updated += num_active

                if num_active > 0:
                    g_active = grad[mask]
                    if weight_decay != 0.0:
                        g_active = g_active + weight_decay * p[mask]

                    # Update biased first moment estimate
                    m_active = exp_avg[mask] * beta1 + (1.0 - beta1) * g_active
                    exp_avg[mask] = m_active

                    # Update biased second raw moment estimate
                    v_active = exp_avg_sq[mask] * beta2 + (1.0 - beta2) * (g_active * g_active)
                    exp_avg_sq[mask] = v_active

                    # Bias-corrected step size
                    step_size = lr / bias_correction1
                    denom = (torch.sqrt(v_active) / math.sqrt(bias_correction2)) + eps

                    # Apply parameter update
                    p[mask] -= step_size * (m_active / denom)

        self.last_updated_count = total_updated
        return loss


def create_optimizer(
    model: nn.Module,
    lr_sdf: float = 0.1,
    lr_hash: float = 1e-2,
    lr_mlp: float = 1e-3,
    lr_variance: float = 1e-2,
    use_sparse_sdf: bool = True,
) -> Dict[str, Optimizer]:
    """Create optimizer suite for VoxNeuS with dedicated learning rates per component.
    Provenance: [PAPER] (Sparse updates for SDF) / [REFERENCE IMPLEMENTATION] (learning rate values)

    Returns:
        Dictionary of optimizers: {'sdf_optimizer': ..., 'net_optimizer': ...}
    """
    # 1. SDF Grid Optimizer (Sparse Adam)
    sdf_params = [model.sdf_grid.grid]
    if getattr(model.sdf_grid, "feat_grid", None) is not None:
        sdf_params.append(model.sdf_grid.feat_grid)

    if use_sparse_sdf:
        sdf_opt = SparseGridAdam(sdf_params, lr=lr_sdf, betas=(0.9, 0.99))
    else:
        sdf_opt = torch.optim.Adam(sdf_params, lr=lr_sdf, betas=(0.9, 0.99))

    # 2. Appearance & Variance Network Optimizer (Standard Adam)
    net_params = [
        {"params": model.radiance_hash.parameters(), "lr": lr_hash},
        {"params": model.radiance_mlp.parameters(), "lr": lr_mlp},
        {"params": model.variance_net.parameters(), "lr": lr_variance},
    ]
    net_opt = torch.optim.Adam(net_params, betas=(0.9, 0.999), eps=1e-8)

    return {
        "sdf_optimizer": sdf_opt,
        "net_optimizer": net_opt,
    }
