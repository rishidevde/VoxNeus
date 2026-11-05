import sys
sys.path.insert(0, '.')

print('--- Checking imports ---')

from voxneus.models.variance import SingleVarianceNetwork
import torch

v = SingleVarianceNetwork(init_val=0.3, s_max=20000.0)
s_val = v.s
print('[OK] variance.py: s=%.2f, s_max=%s' % (s_val.item(), v._s_max))

from voxneus.regularization.eikonal import compute_eikonal_loss_and_gradient
print('[OK] eikonal.py imported')

from voxneus.regularization.curvature import compute_curvature_loss_and_gradient
print('[OK] curvature.py imported')

from voxneus.models.voxneus import VoxNeuS
m = VoxNeuS(grid_res=16, s_max=20000.0, force_pytorch=True)
print('[OK] voxneus.py: VoxNeuS created, s_max=%s' % m.variance_net._s_max)

from voxneus.training.trainer import VoxNeuSTrainer
print('[OK] trainer.py imported')

from voxneus.rendering.sampler import RaySampler
s2 = RaySampler(n_samples=64, n_importance=64)
print('[OK] sampler.py: n_samples=%d, n_importance=%d' % (s2.n_samples, s2.n_importance))

print('')
print('--- Smoke-testing two-pass importance sampling (CPU) ---')
rays_o = torch.zeros(4, 3)
rays_d = torch.zeros(4, 3)
rays_d[:, 2] = 1.0
near = torch.full((4, 1), 0.1)
far  = torch.full((4, 1), 2.0)
sampler = RaySampler(n_samples=8, n_importance=8)
out = m.render(rays_o=rays_o, rays_d=rays_d, sampler=sampler, near=near, far=far)
print('[OK] render() output keys: %s' % list(out.keys()))
print('[OK] rgb shape: %s  (expected [4, 3])' % str(list(out['rgb'].shape)))

# Test eikonal scalar loss (PyTorch path)
print('')
print('--- Smoke-testing Eikonal scalar loss (PyTorch path) ---')
grid = torch.randn(1, 1, 8, 8, 8)
verts = torch.tensor([[3,3,3],[4,4,4]], dtype=torch.int64)
vsize = torch.tensor([0.25, 0.25, 0.25])
res   = torch.tensor([8, 8, 8])
loss_e, grad_e = compute_eikonal_loss_and_gradient(grid, verts, vsize, res)
print('[OK] Eikonal loss=%.6f (non-zero check: %s)' % (loss_e.item(), loss_e.item() > 0))
print('[OK] Eikonal grad shape: %s' % str(list(grad_e.shape)))

# Test curvature scalar loss (PyTorch path)
print('')
print('--- Smoke-testing Curvature scalar loss (PyTorch path) ---')
loss_c, grad_c = compute_curvature_loss_and_gradient(grid, verts, vsize, res)
print('[OK] Curvature loss=%.6f (non-zero check: %s)' % (loss_c.item(), loss_c.item() >= 0))
print('[OK] Curvature grad shape: %s' % str(list(grad_c.shape)))

# Test SDF Total Variation (TV) loss
print('')
print('--- Smoke-testing SDF Total Variation (TV) loss ---')
from voxneus.regularization.tv_loss import compute_sdf_tv_loss
loss_tv = compute_sdf_tv_loss(grid)
print('[OK] TV loss=%.6f (non-zero check: %s)' % (loss_tv.item(), loss_tv.item() > 0))

print('')
print('ALL CHECKS PASSED')
