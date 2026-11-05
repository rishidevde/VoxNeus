"""VoxNeuS Complete Assembled Model.
Provenance: [PAPER] (Sec. 4.3 Architecture & Sec. 4.4 Optimization)
Integrates Dense SDF Grid, Disentangled Radiance Hash Grid, Shallow MLP, and Single Variance Network.
"""

from typing import Dict, Tuple, Optional, Union, List
import torch
import torch.nn as nn

from voxneus.models.sdf_grid import DenseSDFGrid
from voxneus.models.radiance_hash import RadianceHashGrid
from voxneus.models.embedder import PositionalEmbedder, SphericalHarmonicsEmbedder
from voxneus.models.radiance_mlp import RadianceMLP
from voxneus.models.variance import SingleVarianceNetwork
from voxneus.geometry.coordinates import world_to_grid_index
from voxneus.rendering.sampler import RaySampler
from voxneus.rendering.volume_render import volume_render_from_samples
from voxneus.rendering.occupancy_grid import OccupancyGrid


class VoxNeuS(nn.Module):
    """VoxNeuS: Enhancing Voxel-Based Neural Surface Reconstruction via Gradient Interpolation.
    Upgraded with NeuS2 accelerated ray marching, progressive hash bandwidth, and geometry features.
    """

    def __init__(
        self,
        grid_res: Union[int, List[int]] = 96,
        xyz_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
        xyz_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        gradient_mode: str = "interpolated",
        init_s_val: float = 0.3,
        s_max: float = 300.0,       # [PAPER SOTA] Bounded s parameter (200-300) prevents Dirac delta explosion
        hash_n_levels: int = 16,
        hash_n_features: int = 2,
        hash_log2_size: int = 19,
        geom_feature_dim: int = 0,  # Pure VoxNeuS uses 0 (disentangled SDF grid); NeuS2 uses 15
        mlp_hidden_dim: int = 64,
        view_embed_type: str = "sh",  # "sh" or "pe"
        entangled: bool = False,
        occupancy_res: int = 128,
        force_pytorch: bool = False,
    ):
        super().__init__()
        self.gradient_mode = gradient_mode
        self.entangled = entangled
        self.geom_feature_dim = geom_feature_dim

        # 1. Explicit Geometry: Dense 3D SDF Grid (with optional NeuS2 15D geometry features)
        self.sdf_grid = DenseSDFGrid(
            resolution=grid_res,
            xyz_min=xyz_min,
            xyz_max=xyz_max,
            init_mode="sphere",
            feat_dim=geom_feature_dim,
        )

        # 2. Logistic Variance Parameter s (with hard cap s_max) [FIX #6]
        self.variance_net = SingleVarianceNetwork(init_val=init_s_val, s_max=s_max)

        # 3. NeuS2 Occupancy Grid for empty-space ray marching acceleration
        self.occupancy_grid = OccupancyGrid(
            resolution=occupancy_res,
            xyz_min=xyz_min,
            xyz_max=xyz_max,
        )

        # 4. Disentangled Appearance: Multiresolution Hash Grid (with NeuS2 lambda masking)
        self.radiance_hash = RadianceHashGrid(
            n_levels=hash_n_levels,
            n_features_per_level=hash_n_features,
            log2_hashmap_size=hash_log2_size,
            force_pytorch=force_pytorch,
        )
        hash_dim = self.radiance_hash.out_dim

        # 5. View direction encoder
        if view_embed_type.lower() == "sh":
            self.view_embedder = SphericalHarmonicsEmbedder(degree=4)  # 16 features
        else:
            self.view_embedder = PositionalEmbedder(num_freqs=4, include_input=True)
        view_dim = self.view_embedder.out_dim

        # 6. Shallow Radiance MLP with Normal and Geometry Feature conditioning
        self.radiance_mlp = RadianceMLP(
            hash_feature_dim=hash_dim,
            view_embed_dim=view_dim,
            normal_dim=3,
            geom_feature_dim=geom_feature_dim,
            hidden_dim=mlp_hidden_dim,
            entangled=entangled,
        )

    @property
    def s(self) -> torch.Tensor:
        """Current variance scaling factor s."""
        return self.variance_net.s

    @property
    def std(self) -> torch.Tensor:
        """Standard deviation sigma = 1 / s."""
        return self.variance_net.std

    @torch.no_grad()
    def update_occupancy_grid(self, subsample_ratio: float = 0.25) -> None:
        """Periodically update occupancy grid using NeuS2 logistic density phi_s(d).
        Provenance: [PAPER] (NeuS2 Sec. 3.1)
        """
        self.occupancy_grid.update_from_sdf(
            query_sdf_fn=self.sdf_grid.query_sdf,
            s_val=self.s.detach(),
            subsample_ratio=subsample_ratio,
        )

    def forward_points(
        self,
        xyz: torch.Tensor,
        view_dirs: torch.Tensor,
        active_levels: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Query geometry and appearance at 3D world coordinates.

        Args:
            xyz: (..., 3) 3D world coordinates.
            view_dirs: (..., 3) Unit view directions.
            active_levels: Optional lambda parameter controlling active hash frequency bands (NeuS2 Sec. 3.3).

        Returns:
            Dictionary containing 'sdf', 'grad', 'normal', 'corner_indices', 'weights', 'rgb', 'geom_feats'.
        """
        # 1. Query SDF value f(x) via trilinear interpolation (Eq. 3)
        sdf = self.sdf_grid.query_sdf(xyz)

        # 2. Query normal and gradient via continuous gradient interpolation (Eq. 7) or analytical (Eq. 4)
        grad, normal, corner_indices, weights = self.sdf_grid.query_normals(
            xyz,
            gradient_mode=self.gradient_mode,
        )

        # 3. Query NeuS2 15D geometry features if enabled
        geom_feats = self.sdf_grid.query_features(xyz)

        # 4. Normalize coordinates to [0, 1]^3 for multiresolution hash grid
        norm_xyz = (xyz - self.sdf_grid.xyz_min) / (self.sdf_grid.xyz_max - self.sdf_grid.xyz_min + 1e-8)
        hash_feats = self.radiance_hash(norm_xyz, active_levels=active_levels)

        # 5. Encode view direction
        view_embed = self.view_embedder(view_dirs)

        # 6. Predict RGB via shallow MLP (normal + geometry feature conditioned, zero raw SDF leakage)
        rgb = self.radiance_mlp(
            hash_features=hash_feats,
            view_embeddings=view_embed,
            normals=normal,
            geom_features=geom_feats,
            sdf=sdf if self.entangled else None,
        )

        return {
            "sdf": sdf,
            "grad": grad,
            "normal": normal,
            "corner_indices": corner_indices,
            "weights": weights,
            "geom_feats": geom_feats,
            "rgb": rgb,
        }

    def render(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        sampler: RaySampler,
        near: torch.Tensor,
        far: torch.Tensor,
        background_rgb: Optional[torch.Tensor] = None,
        cos_anneal_ratio: float = 0.0,
        use_iter_cos_annealing: bool = False,
        active_levels: Optional[float] = None,
        up_sample_steps: int = 4,
        sphere_radius: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Execute end-to-end volume rendering along rays with multi-step hierarchical sampling.
        Provenance: [PAPER] (Sec. 3 & Eq. 2) / [REFERENCE IMPLEMENTATION] (NeuS renderer.py lines 318-350)

        Pass 1: Multi-step iterative SDF upsampling (no grad) to cluster samples tightly
                around the zero-crossing using progressive inv_s schedules (64 -> 128 -> 256 -> 512).
        Pass 2: Evaluate geometry, normals, hash grid, and appearance at segment MIDPOINTS
                (Paper Eq. 2: f(p(t_i)) at midpoint distances t_i).
        Pass 3: Volume rendering integration with inside-sphere foreground masking.

        Args:
            rays_o: (N_rays, 3) Ray origins.
            rays_d: (N_rays, 3) Ray directions.
            sampler: RaySampler instance with n_samples (coarse) and n_importance (fine).
            near: (N_rays, 1) Near bounds.
            far: (N_rays, 1) Far bounds.
            background_rgb: Background color.
            cos_anneal_ratio: Annealing ratio for iter_cos.
            use_iter_cos_annealing: Boolean.
            active_levels: Optional NeuS2 progressive hash bandwidth lambda.
            up_sample_steps: Number of progressive upsampling iterations (default 4).
            sphere_radius: Foreground unit sphere radius (default 1.0 per NeuS/Voxurf).

        Returns:
            Rendering dictionary with RGB, depth, normal, opacity, weights, etc.
        """
        # ===================================================================
        # PASS 1 — Multi-step hierarchical SDF upsampling (64 coarse + 64 fine)
        # ===================================================================
        def sdf_query_fn(p: torch.Tensor) -> torch.Tensor:
            return self.sdf_grid.query_sdf(p).squeeze(-1)

        z_vals, dists, mid_z_vals = sampler.sample_multistep_sdf(
            rays_o=rays_o,
            rays_d=rays_d,
            near=near,
            far=far,
            sdf_fn=sdf_query_fn,
            n_samples=sampler.n_samples,
            n_importance=sampler.n_importance,
            up_sample_steps=up_sample_steps,
            occupancy_grid=self.occupancy_grid,
        )

        # ===================================================================
        # PASS 2 — Midpoint evaluation (Paper Eq. 2: t_i are mid-point distances)
        # ===================================================================
        pts_mid = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * mid_z_vals.unsqueeze(-1)
        dirs_mid = rays_d.unsqueeze(1).expand_as(pts_mid)

        # Query full network with gradient graph for backprop at midpoints
        point_outputs = self.forward_points(pts_mid, dirs_mid, active_levels=active_levels)

        # ===================================================================
        # PASS 3 — Full NeuS volume rendering integration with inside-sphere masking
        # ===================================================================
        render_outputs = volume_render_from_samples(
            colors=point_outputs["rgb"],
            sdf=point_outputs["sdf"],
            normals=point_outputs["normal"],
            ray_dirs=dirs_mid,
            z_vals=mid_z_vals,
            dists=dists,
            s=self.s,
            pts=pts_mid,
            sphere_radius=sphere_radius,
            background_rgb=background_rgb,
            cos_anneal_ratio=cos_anneal_ratio,
            use_iter_cos_annealing=use_iter_cos_annealing,
        )

        # Pass along point outputs for explicit regularization
        render_outputs["pts"] = pts_mid
        render_outputs["sdf_samples"] = point_outputs["sdf"]
        render_outputs["corner_indices"] = point_outputs["corner_indices"]
        render_outputs["grad_samples"] = point_outputs["grad"]
        render_outputs["normal_samples"] = point_outputs["normal"]

        return render_outputs
