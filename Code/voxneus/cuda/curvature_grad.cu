#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel for Closed-Form Discrete Laplacian Curvature Gradient Accumulation (Sec. 4.2 & 4.4)
// G_curv = 2 / |V_R| * sum nabla^2 f * d(nabla^2 f) / df

template <typename scalar_t>
__global__ void curvature_grad_cuda_kernel(
    const scalar_t* __restrict__ grid,        // [Res_x, Res_y, Res_z]
    const int64_t* __restrict__ active_verts,  // [V, 3] active corner indices
    scalar_t* __restrict__ grad_grid,         // [Res_x, Res_y, Res_z]
    const int res_x, const int res_y, const int res_z,
    const float eps_x, const float eps_y, const float eps_z,
    const float weight,
    const int num_active) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_active) return;

  const int ix = active_verts[idx * 3 + 0];
  const int iy = active_verts[idx * 3 + 1];
  const int iz = active_verts[idx * 3 + 2];

  // Skip boundary vertices to ensure exact 6-neighborhood for discrete Laplacian (Eq. 12)
  if (ix <= 0 || ix >= res_x - 1 || iy <= 0 || iy >= res_y - 1 || iz <= 0 || iz >= res_z - 1) return;

  const int ix_p = ix + 1;
  const int ix_m = ix - 1;

  const int iy_p = iy + 1;
  const int iy_m = iy - 1;

  const int iz_p = iz + 1;
  const int iz_m = iz - 1;

  const float eps_x_sq = eps_x * eps_x;
  const float eps_y_sq = eps_y * eps_y;
  const float eps_z_sq = eps_z * eps_z;

  const float f_center = grid[ix * (res_y * res_z) + iy * res_z + iz];

  const float Lx = (grid[ix_p * (res_y * res_z) + iy * res_z + iz] +
                    grid[ix_m * (res_y * res_z) + iy * res_z + iz] - 2.0f * f_center) / eps_x_sq;

  const float Ly = (grid[ix * (res_y * res_z) + iy_p * res_z + iz] +
                    grid[ix * (res_y * res_z) + iy_m * res_z + iz] - 2.0f * f_center) / eps_y_sq;

  const float Lz = (grid[ix * (res_y * res_z) + iy * res_z + iz_p] +
                    grid[ix * (res_y * res_z) + iy * res_z + iz_m] - 2.0f * f_center) / eps_z_sq;

  const float scale = (2.0f / (float)num_active) * weight;

  const float grad_center = scale * (-2.0f * Lx / eps_x_sq - 2.0f * Ly / eps_y_sq - 2.0f * Lz / eps_z_sq);
  const float grad_xp = scale * (Lx / eps_x_sq);
  const float grad_xm = scale * (Lx / eps_x_sq);
  const float grad_yp = scale * (Ly / eps_y_sq);
  const float grad_ym = scale * (Ly / eps_y_sq);
  const float grad_zp = scale * (Lz / eps_z_sq);
  const float grad_zm = scale * (Lz / eps_z_sq);

  atomicAdd(&grad_grid[ix * (res_y * res_z) + iy * res_z + iz], grad_center);
  atomicAdd(&grad_grid[ix_p * (res_y * res_z) + iy * res_z + iz], grad_xp);
  atomicAdd(&grad_grid[ix_m * (res_y * res_z) + iy * res_z + iz], grad_xm);

  atomicAdd(&grad_grid[ix * (res_y * res_z) + iy_p * res_z + iz], grad_yp);
  atomicAdd(&grad_grid[ix * (res_y * res_z) + iy_m * res_z + iz], grad_ym);

  atomicAdd(&grad_grid[ix * (res_y * res_z) + iy * res_z + iz_p], grad_zp);
  atomicAdd(&grad_grid[ix * (res_y * res_z) + iy * res_z + iz_m], grad_zm);
}

void curvature_grad_cuda(
    torch::Tensor grid,
    torch::Tensor active_verts,
    torch::Tensor grad_grid,
    torch::Tensor voxel_size,
    float weight) {
  const int num_active = active_verts.size(0);
  if (num_active == 0) return;

  const int res_x = grid.size(grid.dim() - 3);
  const int res_y = grid.size(grid.dim() - 2);
  const int res_z = grid.size(grid.dim() - 1);

  const float eps_x = voxel_size[0].item<float>();
  const float eps_y = voxel_size[1].item<float>();
  const float eps_z = voxel_size[2].item<float>();

  const int threads = 256;
  const int blocks = (num_active + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(grid.scalar_type(), "curvature_grad_cuda_kernel", ([&] {
    curvature_grad_cuda_kernel<scalar_t><<<blocks, threads>>>(
        grid.data_ptr<scalar_t>(),
        active_verts.data_ptr<int64_t>(),
        grad_grid.data_ptr<scalar_t>(),
        res_x, res_y, res_z,
        eps_x, eps_y, eps_z,
        weight,
        num_active);
  }));
}
