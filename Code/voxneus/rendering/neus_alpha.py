"""VoxNeuS Discrete Opacity (Alpha) Formulation.
Provenance: [PAPER]
Direct implementation of Eq. 2 from VoxNeuS paper:
"alpha_i = max((Phi_s(f(p_i) - delta_i/2 * cos_theta_i) - Phi_s(f(p_i) + delta_i/2 * cos_theta_i)) / Phi_s(f(p_i) - delta_i/2 * cos_theta_i), 0)"
"""

from typing import Optional
import torch
import torch.nn.functional as F


def compute_neus_alpha(
    sdf: torch.Tensor,
    normals: torch.Tensor,
    ray_dirs: torch.Tensor,
    dists: torch.Tensor,
    s: torch.Tensor,
    cos_anneal_ratio: float = 0.0,
    use_iter_cos_annealing: bool = False,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute discrete opacity values along ray segments using VoxNeuS Eq. 2.

    Args:
        sdf: (..., 1) or (...) Signed distance function values at sample points f(p(t_i)).
        normals: (..., 3) Surface normals n_i = grad^{(i)} / ||grad^{(i)}||.
        ray_dirs: (..., 3) Ray direction vectors v (unit norm).
        dists: (..., 1) or (...) Segment lengths delta_i.
        s: (...) or float Variance scaling factor s in Phi_s(z) = sigmoid(s * z).
        cos_anneal_ratio: Optional annealing ratio in [0, 1] if enabled.
        use_iter_cos_annealing: Provenance [REFERENCE IMPLEMENTATION] (NeuS iter_cos).
                                Default False per VoxNeuS paper Eq. 2.
        eps: Small constant for numerical stability.

    Returns:
        alpha: (..., ) Discrete opacities in [0, 1].
    """
    if sdf.ndim < normals.ndim:
        sdf = sdf.unsqueeze(-1)
    if dists.ndim < normals.ndim:
        dists = dists.unsqueeze(-1)

    # Compute cos(theta_i) = <n_i, v>
    true_cos = torch.sum(normals * ray_dirs, dim=-1, keepdim=True)

    if use_iter_cos_annealing:
        # NeuS / Voxurf iter_cos formulation: guarantees cos_val <= 0 so that f_prev >= f_next
        # and alpha_i >= 0 for all surface samples (prevents zero-alpha gradient deadlock in early training).
        # Provenance: [REFERENCE IMPLEMENTATION] (NeuS renderer.py line 234 / Voxurf voxurf_womask_fine.py line 602)
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
            + F.relu(-true_cos) * cos_anneal_ratio
        )
        cos_val = iter_cos
    else:
        # Paper Eq. 2: direct inner product cos(theta_i) = <n_i, v>
        cos_val = true_cos

    # delta_i / 2 * cos(theta_i)
    offset = 0.5 * dists * cos_val

    # f(p_i) - delta_i/2 * cos(theta_i) and f(p_i) + delta_i/2 * cos(theta_i)
    f_prev = sdf - offset
    f_next = sdf + offset

    # Phi_s(z) = sigmoid(s * z)
    phi_prev = torch.sigmoid(s * f_prev)
    phi_next = torch.sigmoid(s * f_next)

    # Discrete opacity formula:
    # alpha_i = max( (Phi_prev - Phi_next) / (Phi_prev + eps), 0 )
    diff = phi_prev - phi_next
    denom = torch.clamp(phi_prev, min=eps)

    alpha = diff / denom
    alpha = torch.clamp(alpha, min=0.0, max=1.0)

    if alpha.shape[-1] == 1 and sdf.ndim == normals.ndim:
        alpha = alpha.squeeze(-1)

    return alpha
