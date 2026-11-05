"""Shallow Radiance MLP with Disentangled Architecture.
Provenance: [PAPER] (Sec. 4.3 "2-layer shallow MLP with normal conditioning following IDR") /
            [REFERENCE IMPLEMENTATION] (NeuS2 / Instant-NGP color MLP with configurable capacity)
"""

from typing import Optional
import torch
import torch.nn as nn


class RadianceMLP(nn.Module):
    """Predicts view-dependent RGB radiance conditioned on multiresolution hash spatial features,
    surface normals, and encoded view directions.
    Strictly disentangled: NO local SDF values enter this network unless explicitly running ablation E5.
    Supports both 2-layer default (for full backward compatibility with existing checkpoints)
    and upgraded 3-layer / 128-dim architectures for higher capacity.
    """

    def __init__(
        self,
        hash_feature_dim: int = 32,
        view_embed_dim: int = 16,
        normal_dim: int = 3,
        geom_feature_dim: int = 0,
        hidden_dim: int = 64,
        num_layers: int = 2,
        weight_norm: bool = False,
        entangled: bool = False,
    ):
        super().__init__()
        self.entangled = entangled
        self.geom_feature_dim = geom_feature_dim
        self.num_layers = num_layers
        self.weight_norm = weight_norm

        in_dim = hash_feature_dim + view_embed_dim + normal_dim + geom_feature_dim
        if entangled:
            in_dim += 1

        if num_layers == 2:
            # Exact 2-layer sequential layout matching saved checkpoints
            l0 = nn.Linear(in_dim, hidden_dim)
            l1 = nn.Linear(hidden_dim, 3)
            if weight_norm:
                l0 = nn.utils.weight_norm(l0)
                l1 = nn.utils.weight_norm(l1)

            self.net = nn.Sequential(
                l0,
                nn.ReLU(inplace=True),
                l1,
                nn.Sigmoid(),
            )
        else:
            layers = []
            curr_dim = in_dim
            for l in range(num_layers - 1):
                lin = nn.Linear(curr_dim, hidden_dim)
                if weight_norm:
                    lin = nn.utils.weight_norm(lin)
                layers.append(lin)
                layers.append(nn.ReLU(inplace=True))
                curr_dim = hidden_dim

            final_lin = nn.Linear(hidden_dim, 3)
            if weight_norm:
                final_lin = nn.utils.weight_norm(final_lin)
            layers.append(final_lin)
            layers.append(nn.Sigmoid())
            self.net = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        hash_features: torch.Tensor,
        view_embeddings: torch.Tensor,
        normals: torch.Tensor,
        geom_features: Optional[torch.Tensor] = None,
        sdf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict RGB radiance."""
        inputs = [hash_features, view_embeddings, normals]
        if self.geom_feature_dim > 0 and geom_features is not None:
            inputs.append(geom_features)
        if self.entangled:
            assert sdf is not None, "Entangled ablation requires SDF input"
            inputs.append(sdf)

        mlp_input = torch.cat(inputs, dim=-1)
        rgb = self.net(mlp_input)
        return rgb
