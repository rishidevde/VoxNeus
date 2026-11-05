"""VoxNeuS 40k Training Engine.
Provenance: [PAPER] (Sec. 4.4 Optimization Workflow & Sec. 5.1 Training Schedule)
Coordinates ray sampling, volume rendering, manual gradient accumulation,
s-gradient amplification, progressive grid super-resolution, and live monitoring.

Fixes applied:
- SDF LR decays at each upsampling milestone: 0.1 -> 0.05 -> 0.02 [Fix #4]
- Periodic validation PSNR on held-out test views [Fix #7]
"""

import os
import time
import math
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from voxneus.models.voxneus import VoxNeuS
from voxneus.rendering.sampler import RaySampler
from voxneus.rendering.rays import near_far_from_sphere
from voxneus.rendering.volume_render import compute_huber_color_loss
from voxneus.regularization.active_vertices import extract_active_vertices
from voxneus.regularization.eikonal import compute_eikonal_loss_and_gradient
from voxneus.regularization.curvature import compute_curvature_loss_and_gradient
from voxneus.regularization.tv_loss import compute_sdf_tv_loss
from voxneus.training.scheduler import VoxNeuSScheduler, amplify_s_gradient
from voxneus.training.optimizer import create_optimizer, SparseGridAdam
from voxneus.training.terminal_monitor import TerminalMonitor
from voxneus.training.checkpoint import save_checkpoint, load_checkpoint


class VoxNeuSTrainer:
    """End-to-end training orchestrator for VoxNeuS with NeuS2 acceleration.
    Provenance: [PAPER] (Sec. 4.4 & Sec. 5.1 / NeuS2 Sec. 3)
    """

    def __init__(
        self,
        model: VoxNeuS,
        dataset: Any,
        config: Dict[str, Any],
        output_dir: str = "logs/default_run",
        device: Optional[torch.device] = None,
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataset = dataset
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        train_cfg = config.get("training", {})
        self.total_steps = train_cfg.get("total_steps", 40000)
        self.batch_size = train_cfg.get("batch_size", 2048)
        self.use_mask = train_cfg.get("use_mask", False)
        self.mask_weight = train_cfg.get("mask_weight", 0.1)
        self.save_interval = train_cfg.get("save_interval", 5000)
        self.loss_type = train_cfg.get("loss_type", "Huber")  # NeuS2 uses Huber loss
        self.occupancy_update_interval = train_cfg.get("occupancy_update_interval", 16)
        self.progressive_hash = train_cfg.get("progressive_hash", True)

        # Sampler
        sample_cfg = config.get("sampler", {})
        self.sampler = RaySampler(
            n_samples=sample_cfg.get("n_samples", 64),
            n_importance=sample_cfg.get("n_importance", 64),
            perturb=sample_cfg.get("perturb", True),
        ).to(self.device)

        # Schedulers
        sched_cfg = config.get("scheduler", {})
        self.scheduler = VoxNeuSScheduler(
            total_steps=self.total_steps,
            stage1_end=sched_cfg.get("stage1_end", 11000),
            stage2_end=sched_cfg.get("stage2_end", 21000),
            w_eik_init=sched_cfg.get("w_eik_init", 1e-2),
            w_eik_final=sched_cfg.get("w_eik_final", 1e-3),
            w_curv_init=sched_cfg.get("w_curv_init", 1e-8),
            w_curv_peak=sched_cfg.get("w_curv_peak", 5e-6),
            w_curv_final=sched_cfg.get("w_curv_final", 5e-7),
        )

        # Optimizers
        opt_cfg = config.get("optimizer", {})
        self.lr_sdf = opt_cfg.get("lr_sdf", 0.1)
        # Stage-wise SDF LR decay — reduces after each upsampling milestone [Fix #4]
        self.lr_sdf_stages = [
            opt_cfg.get("lr_sdf", 0.1),          # Stage 1: 0-10k
            opt_cfg.get("lr_sdf_stage2", 0.05),  # Stage 2: 10k-30k
            opt_cfg.get("lr_sdf_stage3", 0.02),  # Stage 3: 30k-40k
        ]
        self.current_lr_sdf = self.lr_sdf_stages[0]  # Track actual current SDF LR
        self.lr_hash = opt_cfg.get("lr_hash", 0.01)
        self.lr_mlp = opt_cfg.get("lr_mlp", 0.001)
        self.lr_variance = opt_cfg.get("lr_variance", 0.0005)

        self.optimizers = create_optimizer(
            model=self.model,
            lr_sdf=self.lr_sdf,
            lr_hash=self.lr_hash,
            lr_mlp=self.lr_mlp,
            lr_variance=self.lr_variance,
            use_sparse_sdf=opt_cfg.get("use_sparse_sdf", True),
        )

        # Progressive super-resolution milestones (Sec. 5.1: 10k, 30k)
        self.sr_milestones = train_cfg.get("sr_milestones", [10000, 30000])
        self.sr_resolutions = train_cfg.get("sr_resolutions", [96, 160, 320])

        # s-gradient amplification config
        sched_cfg = config.get("scheduler", {})
        self.s_amplification_k = sched_cfg.get("s_amplification_k", 5.0)
        # Don't amplify s-gradient until geometry has warmed up past this step.
        # Firing at step 0 on a sphere init just makes s grow uncontrollably.
        self.s_warmup_steps = sched_cfg.get("s_warmup_steps", 5000)
        # Clamp applied to Eikonal grad norm right after upsampling (post-upsample stabilization)
        self.eik_grad_clamp = sched_cfg.get("eik_grad_clamp", 0.1)
        self._post_upsample_cooldown = 0  # Steps remaining in post-upsample cooldown

        # Validation PSNR interval [Fix #7]
        self.val_interval = train_cfg.get("val_interval", 1000)

        # Terminal monitor
        self.monitor = TerminalMonitor(
            total_steps=self.total_steps,
            dashboard_interval=config.get("logging", {}).get("dashboard_interval", 100),
            print_interval=config.get("logging", {}).get("print_interval", 10),
        )

    def check_and_apply_upsampling(self, step: int) -> None:
        """Progressive SDF grid super-resolution at 10k and 30k steps.
        Fix #4: SDF LR is reduced at each upsampling milestone for post-upsample stability.
        Also sets a 500-step cooldown during which Eikonal grads are clamped.
        """
        if step in self.sr_milestones:
            idx = self.sr_milestones.index(step) + 1
            target_res = self.sr_resolutions[idx]

            new_lr = self.lr_sdf_stages[min(idx, len(self.lr_sdf_stages) - 1)]
            self.current_lr_sdf = new_lr

            print(
                f"\n[VoxNeuS] Progressive Super-Resolution: scaling SDF grid to {target_res}³ at step {step}..."
                f" (SDF LR: {self.lr_sdf_stages[idx-1]:.4f} → {new_lr:.4f})",
                flush=True,
            )

            self.model.sdf_grid.scale_volume_grid(target_res)

            opt_cfg = self.config.get("optimizer", {})
            use_sparse = opt_cfg.get("use_sparse_sdf", True)
            sdf_params = [self.model.sdf_grid.grid]
            if getattr(self.model.sdf_grid, "feat_grid", None) is not None:
                sdf_params.append(self.model.sdf_grid.feat_grid)
            if use_sparse:
                self.optimizers["sdf_optimizer"] = SparseGridAdam(sdf_params, lr=new_lr)
            else:
                self.optimizers["sdf_optimizer"] = torch.optim.Adam(sdf_params, lr=new_lr)

            # Start post-upsample cooldown: clamp Eikonal grads for 500 steps
            # to prevent the interpolation discontinuity spike (saw Eikonal=184 in v2 log)
            self._post_upsample_cooldown = 500
            print(f"[VoxNeuS] Post-upsample cooldown: Eikonal grads clamped to {self.eik_grad_clamp} for 500 steps.", flush=True)

    def train_step(self, step: int) -> Dict[str, Any]:
        """Execute a single training iteration."""
        # 1. Progressive upsampling check
        self.check_and_apply_upsampling(step)

        # 2. Periodically update NeuS2 occupancy grid with logistic density
        if step % self.occupancy_update_interval == 0:
            self.model.update_occupancy_grid(subsample_ratio=0.25)

        # 3. Compute NeuS2 progressive hash bandwidth lambda
        active_lambda = self.scheduler.get_lambda(step) if self.progressive_hash else None

        # 4. Sample ray batch from dataset
        batch = self.dataset.sample_rays(self.batch_size)
        rays_o = batch["rays_o"].to(self.device)
        rays_d = batch["rays_d"].to(self.device)
        target_rgb = batch["rgb"].to(self.device)
        target_mask = batch.get("mask", None)
        if target_mask is not None:
            target_mask = target_mask.to(self.device)

        # 5. Ray bounds from unit sphere
        near, far = near_far_from_sphere(rays_o, rays_d, radius=1.0)

        # Cosine annealing ratio for NeuS iter_cos (ramps 0.0 -> 1.0 in first 10,000 steps)
        cos_anneal_ratio = min(float(step) / 10000.0, 1.0)

        # 6. Forward volume rendering with multi-step SDF upsampling and inside-sphere masking
        up_sample_steps = self.config.get("sampler", {}).get("up_sample_steps", 4)
        use_iter_cos = self.config.get("model", {}).get("use_iter_cos_annealing", False)
        render_outputs = self.model.render(
            rays_o=rays_o,
            rays_d=rays_d,
            sampler=self.sampler,
            near=near,
            far=far,
            cos_anneal_ratio=cos_anneal_ratio,
            use_iter_cos_annealing=use_iter_cos,
            active_levels=active_lambda,
            up_sample_steps=up_sample_steps,
            sphere_radius=1.0,
        )

        pred_rgb = render_outputs["rgb"]
        pred_opacity = render_outputs["opacity"]
        corner_indices = render_outputs["corner_indices"]

        # 7. Main supervision loss: NeuS2 Huber loss or MSE loss
        if self.use_mask and target_mask is not None:
            # Masked color loss: only penalize color on actual foreground object surface
            if self.loss_type.lower() == "huber":
                diff = torch.abs(pred_rgb - target_rgb)
                is_small = diff < 0.1
                per_ray_loss = torch.where(is_small, 0.5 * (diff ** 2) / 0.1, diff - 0.05)
            else:
                per_ray_loss = (pred_rgb - target_rgb) ** 2

            fg_weight = target_mask.expand_as(pred_rgb)
            fg_sum = fg_weight.sum()
            if fg_sum > 0:
                loss_rgb = (per_ray_loss * fg_weight).sum() / fg_sum
                fg_mse = ((pred_rgb - target_rgb) ** 2 * fg_weight).sum() / fg_sum
                psnr = -10.0 * math.log10(max(fg_mse.item(), 1e-8))
            else:
                loss_rgb = per_ray_loss.mean()
                psnr = -10.0 * math.log10(max(loss_rgb.item(), 1e-8))
        else:
            if self.loss_type.lower() == "huber":
                loss_rgb = compute_huber_color_loss(pred_rgb, target_rgb, delta=0.1)
            else:
                loss_rgb = F.mse_loss(pred_rgb, target_rgb)
            mse_val = F.mse_loss(pred_rgb, target_rgb).item()
            psnr = -10.0 * math.log10(max(mse_val, 1e-8))

        loss_total = loss_rgb

        loss_mask = None
        if self.use_mask and target_mask is not None:
            loss_mask = F.binary_cross_entropy(torch.clamp(pred_opacity, 1e-6, 1.0 - 1e-6), target_mask)
            loss_total = loss_total + self.mask_weight * loss_mask

        # 3D Total Variation (TV) loss on dense SDF voxel grid (disabled by default in paper)
        w_tv = self.config.get("scheduler", {}).get("w_tv_sdf", 0.0)
        if w_tv > 0:
            loss_tv = compute_sdf_tv_loss(self.model.sdf_grid.grid)
            loss_total = loss_total + w_tv * loss_tv

        # 8. Backward on main loss (computes RGB + Mask gradients on sdf_grid)
        for opt in self.optimizers.values():
            opt.zero_grad()

        loss_total.backward()

        # 9. Explicit Regularization on Active Vertices V_R within relaxed unit sphere (Sec. 4.2 & 4.4)
        unique_corners, _ = extract_active_vertices(corner_indices, self.model.sdf_grid.res)
        
        # Relaxed unit sphere mask: only regularize vertices within radius 1.2 (NeuS line 251)
        world_corners = self.model.sdf_grid.xyz_min + unique_corners.float() * self.model.sdf_grid.voxel_size
        inside_relax = torch.linalg.norm(world_corners, dim=-1) < 1.2
        if inside_relax.sum() > 0:
            unique_corners = unique_corners[inside_relax]

        w_eik, w_curv = self.scheduler.get_weights(step)

        # In-place CUDA accumulation optimization: accumulates directly into grid.grad
        # when out of post-upsample cooldown, avoiding redundant 32.7M-float tensor allocations.
        use_direct_cuda = (
            self.model.sdf_grid.grid.grad is not None
            and self.model.sdf_grid.grid.is_cuda
            and self._post_upsample_cooldown <= 0
        )

        if use_direct_cuda:
            loss_eik, _ = compute_eikonal_loss_and_gradient(
                grid_data=self.model.sdf_grid.grid,
                active_vertices=unique_corners,
                voxel_size=self.model.sdf_grid.voxel_size,
                grid_res=self.model.sdf_grid.res,
                accumulate_grad=self.model.sdf_grid.grid.grad if w_eik > 0 else None,
                weight=w_eik,
            )
            loss_curv, _ = compute_curvature_loss_and_gradient(
                grid_data=self.model.sdf_grid.grid,
                active_vertices=unique_corners,
                voxel_size=self.model.sdf_grid.voxel_size,
                grid_res=self.model.sdf_grid.res,
                accumulate_grad=self.model.sdf_grid.grid.grad if w_curv > 0 else None,
                weight=w_curv,
            )
        else:
            # Separate tensor path for post-upsample cooldown clamping or CPU fallback
            loss_eik, g_eik = compute_eikonal_loss_and_gradient(
                grid_data=self.model.sdf_grid.grid,
                active_vertices=unique_corners,
                voxel_size=self.model.sdf_grid.voxel_size,
                grid_res=self.model.sdf_grid.res,
            )
            loss_curv, g_curv = compute_curvature_loss_and_gradient(
                grid_data=self.model.sdf_grid.grid,
                active_vertices=unique_corners,
                voxel_size=self.model.sdf_grid.voxel_size,
                grid_res=self.model.sdf_grid.res,
            )
            if self.model.sdf_grid.grid.grad is not None:
                if w_eik > 0 and g_eik is not None:
                    g_eik_scaled = g_eik.to(self.device) * w_eik
                    if self._post_upsample_cooldown > 0:
                        eik_norm = g_eik_scaled.abs().max()
                        if eik_norm > self.eik_grad_clamp:
                            g_eik_scaled = g_eik_scaled * (self.eik_grad_clamp / eik_norm)
                        self._post_upsample_cooldown -= 1
                    self.model.sdf_grid.grid.grad.add_(g_eik_scaled)
                if w_curv > 0 and g_curv is not None:
                    self.model.sdf_grid.grid.grad.add_(g_curv.to(self.device), alpha=w_curv)

        # 10. s-Gradient Amplification: k=5 when dL/ds < 0 (Eq. 16)
        # Gate: only apply AFTER s_warmup_steps to avoid blowing up s at initialization.
        if step >= self.s_warmup_steps:
            s_amplified = amplify_s_gradient(self.model.variance_net.variance, k=self.s_amplification_k)
        else:
            s_amplified = False

        # Cosine LR decay for hash grid & MLP (NeuS / Voxurf) + s-amplification effective scaling
        lr_factor = (math.cos(math.pi * step / max(self.total_steps, 1)) + 1.0) * 0.5 * (1.0 - 0.05) + 0.05
        cur_lr_hash = self.lr_hash * lr_factor
        cur_lr_mlp = self.lr_mlp * lr_factor
        cur_lr_var = self.lr_variance * (self.s_amplification_k if s_amplified else 1.0)
        if "net_optimizer" in self.optimizers:
            net_opt = self.optimizers["net_optimizer"]
            if len(net_opt.param_groups) >= 3:
                net_opt.param_groups[0]["lr"] = cur_lr_hash
                net_opt.param_groups[1]["lr"] = cur_lr_mlp
                net_opt.param_groups[2]["lr"] = cur_lr_var

        # 11. Optimizer steps
        for opt in self.optimizers.values():
            opt.step()

        # Progressive minimum s-floor: ensures surface sharpness scales with grid super-resolution
        # Stage 2 (160^3 at 10k): s >= 80 (beta >= 0.438)
        # Stage 3 (320^3 at 30k): s >= 150 (beta >= 0.501)
        if len(self.sr_milestones) > 0 and step >= self.sr_milestones[0]:
            min_beta = 0.4382  # exp(4.382) ~ 80.0
            if len(self.sr_milestones) > 1 and step >= self.sr_milestones[1]:
                min_beta = 0.5010  # exp(5.010) ~ 150.0
            with torch.no_grad():
                self.model.variance_net.variance.data.clamp_min_(min_beta)

        # Extract sparse count if SparseGridAdam
        sparse_count = None
        sdf_opt = self.optimizers.get("sdf_optimizer", None)
        if isinstance(sdf_opt, SparseGridAdam):
            sparse_count = sdf_opt.last_updated_count

        # 10. Metrics package
        metrics = {
            "loss_total": loss_total.item(),
            "loss_rgb": loss_rgb.item(),
            "psnr": psnr,
            "loss_mask": loss_mask.item() if loss_mask is not None else None,
            "loss_eik": loss_eik.item(),
            "w_eik": w_eik,
            "loss_curv": loss_curv.item(),
            "w_curv": w_curv,
            "s": self.model.s.item(),
            "std": self.model.std.item(),
            "s_amplified": s_amplified,
            "grid_res": int(self.model.sdf_grid.res[0]),
            "lr_sdf": self.current_lr_sdf,   # Report actual current stage LR
            "lr_hash": cur_lr_hash,
            "lr_mlp": cur_lr_mlp,
            "lr_s": cur_lr_var,
            "sparse_updated": sparse_count,
            "active_lambda": active_lambda,
            "occ_pct": self.model.occupancy_grid.occupied_percentage,
            "loss_name": self.loss_type,
        }

        # 11. Update live monitor
        self.monitor.update(step, metrics)

        # 12. Periodic checkpoint
        if step > 0 and step % self.save_interval == 0:
            ckpt_path = os.path.join(self.output_dir, f"checkpoint_step_{step}.pt")
            save_checkpoint(ckpt_path, step, self.model, self.optimizers, self.config, metrics)
            print(f"[VoxNeuS] Saved checkpoint to {ckpt_path}", flush=True)

        return metrics

    @torch.no_grad()
    def _compute_validation_psnr(self) -> float:
        """Compute average PSNR over held-out test views for monitoring. [Fix #7]"""
        self.model.eval()
        psnr_vals = []
        test_views = getattr(self.dataset, "test_views", [])[:4]  # Use up to 4 test views
        if not test_views:
            self.model.train()
            return 0.0

        for view_idx in test_views:
            try:
                batch = self.dataset.get_test_rays(view_idx)
                rays_o = batch["rays_o"].to(self.device)
                rays_d = batch["rays_d"].to(self.device)
                target_rgb = batch["rgb"].to(self.device)
                H, W = batch["H"], batch["W"]

                # Render in chunks to avoid OOM
                chunk = 2048
                pred_chunks = []
                for i in range(0, rays_o.shape[0], chunk):
                    ro = rays_o[i:i+chunk]
                    rd = rays_d[i:i+chunk]
                    near, far = near_far_from_sphere(ro, rd, radius=1.0)
                    use_iter_cos = self.config.get("model", {}).get("use_iter_cos_annealing", False)
                    out = self.model.render(
                        rays_o=ro, rays_d=rd, sampler=self.sampler,
                        near=near, far=far,
                        use_iter_cos_annealing=use_iter_cos,
                    )
                    pred_chunks.append(out["rgb"].cpu())

                pred_rgb = torch.cat(pred_chunks, dim=0)
                mse = F.mse_loss(pred_rgb, target_rgb.cpu())
                psnr = -10.0 * torch.log10(mse + 1e-10).item()
                psnr_vals.append(psnr)
            except Exception:
                pass

        self.model.train()
        return float(sum(psnr_vals) / len(psnr_vals)) if psnr_vals else 0.0

    def train(self, start_step: int = 0) -> None:
        """Run full training loop up to total_steps."""
        print(f"[VoxNeuS] Commencing training: steps {start_step} to {self.total_steps} (ray batch: {self.batch_size})", flush=True)
        for step in range(start_step, self.total_steps + 1):
            self.train_step(step)

            # Periodic validation PSNR on held-out test views [Fix #7]
            if step > 0 and self.val_interval > 0 and step % self.val_interval == 0:
                val_psnr = self._compute_validation_psnr()
                print(f"[VoxNeuS] *** Validation PSNR @ step {step:,}: {val_psnr:.2f} dB ***", flush=True)

        # Final checkpoint
        final_ckpt = os.path.join(self.output_dir, "checkpoint_final.pt")
        save_checkpoint(final_ckpt, self.total_steps, self.model, self.optimizers, self.config)
        print(f"\n[VoxNeuS] Training complete! Final checkpoint saved to: {final_ckpt}\n", flush=True)
