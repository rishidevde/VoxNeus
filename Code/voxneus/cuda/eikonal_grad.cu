#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel for Closed-Form Eikonal Gradient Accumulation (Sec. 4.2 & 4.4)
// G_eik = sum_{x in V_R} dL_eik/dn * dn/df

template <typename scalar_t>
__global__ void eikonal_grad_cuda_kernel(
    const scalar_t* __restrict__ grid,       // [Res_x, Res_y, Res_z]
    const int64_t* __restrict__ active_verts, // [V, 3] active corner indices
    scalar_t* __restrict__ grad_grid,        // [Res_x, Res_y, Res_z]
    const int res_x, const int res_y, const int res_z,
    const float eps_x, const float eps_y, const float eps_z,
    const float weight,
    const int num_active) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_active) return;

  const int ix = active_verts[idx * 3 + 0];
  const int iy = active_verts[idx * 3 + 1];
  const int iz = active_verts[idx * 3 + 2];

  // Skip boundary vertices for exact central finite differences (Eq. 6 / Eq. 10)
  if (ix <= 0 || ix >= res_x - 1 || iy <= 0 || iy >= res_y - 1 || iz <= 0 || iz >= res_z - 1) return;

  const int ix_p = ix + 1;
  const int ix_m = ix - 1;
  const float denom_x = 2.0f * eps_x;

  const int iy_p = iy + 1;
  const int iy_m = iy - 1;
  const float denom_y = 2.0f * eps_y;

  const int iz_p = iz + 1;
  const int iz_m = iz - 1;
  const float denom_z = 2.0f * eps_z;

  const float fx_p = grid[ix_p * (res_y * res_z) + iy * res_z + iz];
  const float fx_m = grid[ix_m * (res_y * res_z) + iy * res_z + iz];
  const float nx = (fx_p - fx_m) / denom_x;

  const float fy_p = grid[ix * (res_y * res_z) + iy_p * res_z + iz];
  const float fy_m = grid[ix * (res_y * res_z) + iy_m * res_z + iz];
  const float ny = (fy_p - fy_m) / denom_y;

  const float fz_p = grid[ix * (res_y * res_z) + iy * res_z + iz_p];
  const float fz_m = grid[ix * (res_y * res_z) + iy * res_z + iz_m];
  const float nz = (fz_p - fz_m) / denom_z;

  const float norm = sqrtf(nx * nx + ny * ny + nz * nz) + 1e-8f;

  // dL_eik / dn = 2 / |V_R| * (1 - 1 / ||n||) * n
  const float scale = (2.0f / (float)num_active) * (1.0f - 1.0f / norm) * weight;
  const float vx = scale * nx;
  const float vy = scale * ny;
  const float vz = scale * nz;

  // Accumulate directly into grid.grad
  atomicAdd(&grad_grid[ix_p * (res_y * res_z) + iy * res_z + iz], vx / denom_x);
  atomicAdd(&grad_grid[ix_m * (res_y * res_z) + iy * res_z + iz], -vx / denom_x);

  atomicAdd(&grad_grid[ix * (res_y * res_z) + iy_p * res_z + iz], vy / denom_y);
  atomicAdd(&grad_grid[ix * (res_y * res_z) + iy_m * res_z + iz], -vy / denom_y);

  atomicAdd(&grad_grid[ix * (res_y * res_z) + iy * res_z + iz_p], vz / denom_z);
  atomicAdd(&grad_grid[ix * (res_y * res_z) + iy * res_z + iz_m], -vz / denom_z);
}

void eikonal_grad_cuda(
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

  AT_DISPATCH_FLOATING_TYPES(grid.scalar_type(), "eikonal_grad_cuda_kernel", ([&] {
    eikonal_grad_cuda_kernel<scalar_t><<<blocks, threads>>>(
        grid.data_ptr<scalar_t>(),
        active_verts.data_ptr<int64_t>(),
        grad_grid.data_ptr<scalar_t>(),
        res_x, res_y, res_z,
        eps_x, eps_y, eps_z,
        weight,
        num_active);
  }));
}

