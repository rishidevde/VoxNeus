"""Live Terminal Training Dashboard and Progress Monitor.
Provenance: [OUR ENGINEERING OPTIMIZATION]
Renders rich terminal status monitoring training progress, iteration speed (it/s),
loss breakdown, regularization schedules, variance s state, learning rates, and memory.
"""

import time
import math
from typing import Optional, Dict, Any
import torch


class TerminalMonitor:
    """Real-time console dashboard for tracking VoxNeuS 40k training progress."""

    def __init__(
        self,
        total_steps: int = 40000,
        dashboard_interval: int = 100,
        print_interval: int = 10,
    ):
        self.total_steps = total_steps
        self.dashboard_interval = dashboard_interval
        self.print_interval = print_interval

        self.start_time = time.time()
        self.last_step_time = time.time()
        self.ema_speed = None  # Exponential moving average of it/s

    def format_time(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS or MM:SS."""
        if seconds < 0 or math.isnan(seconds) or math.isinf(seconds):
            return "N/A"
        secs = int(seconds)
        mins, secs = divmod(secs, 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours:02d}h{mins:02d}m{secs:02d}s"
        return f"{mins:02d}m{secs:02d}s"

    def update(
        self,
        step: int,
        metrics: Dict[str, Any],
        force_dashboard: bool = False,
    ) -> None:
        """Update monitor with current iteration metrics and display to terminal."""
        now = time.time()
        elapsed = now - self.start_time
        dt = now - self.last_step_time
        self.last_step_time = now

        current_speed = 1.0 / max(dt, 1e-6)
        if self.ema_speed is None:
            self.ema_speed = current_speed
        else:
            self.ema_speed = 0.9 * self.ema_speed + 0.1 * current_speed

        remaining_steps = max(self.total_steps - step, 0)
        eta_seconds = remaining_steps / max(self.ema_speed, 1e-6)
        eta_str = self.format_time(eta_seconds)
        elapsed_str = self.format_time(elapsed)

        # Decide whether to render full multi-line dashboard or 1-line ticker
        is_dashboard = force_dashboard or (step % self.dashboard_interval == 0)
        is_ticker = (step % self.print_interval == 0)

        if not (is_dashboard or is_ticker):
            return

        # Extract metrics safely
        total_loss = metrics.get("loss_total", 0.0)
        rgb_loss = metrics.get("loss_rgb", 0.0)
        mask_loss = metrics.get("loss_mask", None)
        psnr = metrics.get("psnr", -10.0 * math.log10(max(rgb_loss, 1e-8)))

        eik_loss = metrics.get("loss_eik", 0.0)
        w_eik = metrics.get("w_eik", 0.0)
        curv_loss = metrics.get("loss_curv", 0.0)
        w_curv = metrics.get("w_curv", 0.0)

        s_val = metrics.get("s", 0.0)
        std_val = metrics.get("std", 1.0 / max(s_val, 1e-6))
        s_amplified = metrics.get("s_amplified", False)

        grid_res = metrics.get("grid_res", 96)
        stage_name = "1st Stage" if grid_res == 96 else ("2nd Stage" if grid_res == 160 else "Final Stage")

        lr_sdf = metrics.get("lr_sdf", 0.1)
        lr_hash = metrics.get("lr_hash", 0.01)
        lr_mlp = metrics.get("lr_mlp", 0.001)
        lr_s = metrics.get("lr_s", 0.0005)

        # Hardware memory info
        if torch.cuda.is_available():
            vram_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            hw_str = f"GPU VRAM: {vram_alloc:.2f} GB / {vram_total:.2f} GB | CUDA Kernel: Active"
        else:
            hw_str = "Device: CPU / Pure PyTorch Vectorized Mode | Sparse Update: Enabled"

        sparse_updated = metrics.get("sparse_updated", None)
        if sparse_updated is not None:
            hw_str += f" | Updated Voxels: {sparse_updated:,}"

        pct = (step / max(self.total_steps, 1)) * 100.0

        active_lambda = metrics.get("active_lambda", None)
        occ_pct = metrics.get("occ_pct", None)
        loss_name = metrics.get("loss_name", "Huber")

        if is_dashboard:
            s_status = "amplified (k=5)" if s_amplified else "neutral"
            mask_str = f"{mask_loss:.5f}" if mask_loss is not None else "N/A"
            param_str = f"s = {s_val:.2f} (std={std_val:.4f}) | s-grad: {s_status}"
            if active_lambda is not None:
                param_str += f" | NeuS2 lambda: {active_lambda:.1f}"
            if occ_pct is not None:
                param_str += f" | Occupancy: {occ_pct:.1f}%"

            dashboard = [
                "=" * 100,
                f"[VoxNeuS + NeuS2] Iter: {step:,}/{self.total_steps:,} ({pct:.1f}%) | "
                f"Speed: {self.ema_speed:.1f} it/s (ETA: {eta_str}) | Grid: {grid_res}^3 ({stage_name})",
                "-" * 100,
                f"Losses: Total: {total_loss:.5f} | RGB ({loss_name}): {rgb_loss:.5f} | Mask: {mask_str} | PSNR: {psnr:.2f} dB",
                f"Regularization: Eikonal: {eik_loss:.4f} (w={w_eik:.1e}) | Curvature: {curv_loss:.2e} (w={w_curv:.1e})",
                f"Parameters: {param_str}",
                f"Learning Rates: SDF: {lr_sdf:.1e} | Hash: {lr_hash:.1e} | Radiance MLP: {lr_mlp:.1e} | s: {lr_s:.1e}",
                f"Hardware: {hw_str}",
                "=" * 100,
            ]
            print("\n" + "\n".join(dashboard) + "\n", flush=True)
        elif is_ticker:
            print(
                f"[VoxNeuS] Step {step:,}/{self.total_steps:,} ({pct:.1f}%) | "
                f"Loss: {total_loss:.5f} (RGB: {rgb_loss:.5f}) | PSNR: {psnr:.2f} dB | "
                f"s: {s_val:.1f} | Speed: {self.ema_speed:.1f} it/s | ETA: {eta_str}",
                flush=True,
            )
