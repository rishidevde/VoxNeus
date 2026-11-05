#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Forward pass: Fused gradient interpolation with local memory access
// Backward pass: In-place recomputation of weights and finite-difference Jacobian accumulation via atomicAdd

template <typename scalar_t>
__global__ void gradient_interp_forward_kernel(
    const scalar_t* __restrict__ grid,       // [Res_x, Res_y, Res_z]
    const scalar_t* __restrict__ coords,     // [N, 3] in [0, Res - 1]
    scalar_t* __restrict__ out_grad,         // [N, 3] interpolated gradient
    const int res_x, const int res_y, const int res_z,
    const float eps_x, const float eps_y, const float eps_z,
    const int N) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  const float cx = fminf(fmaxf(coords[idx * 3 + 0], 0.0f), res_x - 1.0001f);
  const float cy = fminf(fmaxf(coords[idx * 3 + 1], 0.0f), res_y - 1.0001f);
  const float cz = fminf(fmaxf(coords[idx * 3 + 2], 0.0f), res_z - 1.0001f);

  const int ix0 = __float2int_rd(cx);
  const int iy0 = __float2int_rd(cy);
  const int iz0 = __float2int_rd(cz);

  const float dx = cx - ix0;
  const float dy = cy - iy0;
  const float dz = cz - iz0;

  float gx_total = 0.0f;
  float gy_total = 0.0f;
  float gz_total = 0.0f;

  for (int ox = 0; ox <= 1; ++ox) {
    const float wx = ox ? dx : (1.0f - dx);
    const int ix = ix0 + ox;
    const int ix_p = min(ix + 1, res_x - 1);
    const int ix_m = max(ix - 1, 0);
    const float d_x = (ix_p - ix_m) * eps_x;

    for (int oy = 0; oy <= 1; ++oy) {
      const float wy = oy ? dy : (1.0f - dy);
      const int iy = iy0 + oy;
      const int iy_p = min(iy + 1, res_y - 1);
      const int iy_m = max(iy - 1, 0);
      const float d_y = (iy_p - iy_m) * eps_y;

      for (int oz = 0; oz <= 1; ++oz) {
        const float wz = oz ? dz : (1.0f - dz);
        const int iz = iz0 + oz;
        const int iz_p = min(iz + 1, res_z - 1);
        const int iz_m = max(iz - 1, 0);
        const float d_z = (iz_p - iz_m) * eps_z;

        const float w = wx * wy * wz;

        // Fetch central finite differences
        const float fx_p = grid[ix_p * (res_y * res_z) + iy * res_z + iz];
        const float fx_m = grid[ix_m * (res_y * res_z) + iy * res_z + iz];
        const float nx = (fx_p - fx_m) / (d_x > 0 ? d_x : eps_x);

        const float fy_p = grid[ix * (res_y * res_z) + iy_p * res_z + iz];
        const float fy_m = grid[ix * (res_y * res_z) + iy_m * res_z + iz];
        const float ny = (fy_p - fy_m) / (d_y > 0 ? d_y : eps_y);

        const float fz_p = grid[ix * (res_y * res_z) + iy * res_z + iz_p];
        const float fz_m = grid[ix * (res_y * res_z) + iy * res_z + iz_m];
        const float nz = (fz_p - fz_m) / (d_z > 0 ? d_z : eps_z);

        gx_total += w * nx;
        gy_total += w * ny;
        gz_total += w * nz;
      }
    }
  }

  out_grad[idx * 3 + 0] = gx_total;
  out_grad[idx * 3 + 1] = gy_total;
  out_grad[idx * 3 + 2] = gz_total;
}

// Backward kernel recomputes weights and uses atomicAdd to propagate to grid.grad
template <typename scalar_t>
__global__ void gradient_interp_backward_kernel(
    const scalar_t* __restrict__ grad_out,   // [N, 3] dL / d(grad_interp)
    const scalar_t* __restrict__ coords,     // [N, 3]
    scalar_t* __restrict__ grad_grid,        // [Res_x, Res_y, Res_z]
    const int res_x, const int res_y, const int res_z,
    const float eps_x, const float eps_y, const float eps_z,
    const int N) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  const float cx = fminf(fmaxf(coords[idx * 3 + 0], 0.0f), res_x - 1.0001f);
  const float cy = fminf(fmaxf(coords[idx * 3 + 1], 0.0f), res_y - 1.0001f);
  const float cz = fminf(fmaxf(coords[idx * 3 + 2], 0.0f), res_z - 1.0001f);

  const int ix0 = __float2int_rd(cx);
  const int iy0 = __float2int_rd(cy);
  const int iz0 = __float2int_rd(cz);

  const float dx = cx - ix0;
  const float dy = cy - iy0;
  const float dz = cz - iz0;

  const float dL_gx = grad_out[idx * 3 + 0];
  const float dL_gy = grad_out[idx * 3 + 1];
  const float dL_gz = grad_out[idx * 3 + 2];

  for (int ox = 0; ox <= 1; ++ox) {
    const float wx = ox ? dx : (1.0f - dx);
    const int ix = ix0 + ox;
    const int ix_p = min(ix + 1, res_x - 1);
    const int ix_m = max(ix - 1, 0);
    const float d_x = (ix_p - ix_m) * eps_x;

    for (int oy = 0; oy <= 1; ++oy) {
      const float wy = oy ? dy : (1.0f - dy);
      const int iy = iy0 + oy;
      const int iy_p = min(iy + 1, res_y - 1);
      const int iy_m = max(iy - 1, 0);
      const float d_y = (iy_p - iy_m) * eps_y;

      for (int oz = 0; oz <= 1; ++oz) {
        const float wz = oz ? dz : (1.0f - dz);
        const int iz = iz0 + oz;
        const int iz_p = min(iz + 1, res_z - 1);
        const int iz_m = max(iz - 1, 0);
        const float d_z = (iz_p - iz_m) * eps_z;

        const float w = wx * wy * wz;

        const float denom_x = (d_x > 0 ? d_x : eps_x);
        const float denom_y = (d_y > 0 ? d_y : eps_y);
        const float denom_z = (d_z > 0 ? d_z : eps_z);

        atomicAdd(&grad_grid[ix_p * (res_y * res_z) + iy * res_z + iz], (w * dL_gx) / denom_x);
        atomicAdd(&grad_grid[ix_m * (res_y * res_z) + iy * res_z + iz], -(w * dL_gx) / denom_x);

        atomicAdd(&grad_grid[ix * (res_y * res_z) + iy_p * res_z + iz], (w * dL_gy) / denom_y);
        atomicAdd(&grad_grid[ix * (res_y * res_z) + iy_m * res_z + iz], -(w * dL_gy) / denom_y);

        atomicAdd(&grad_grid[ix * (res_y * res_z) + iy * res_z + iz_p], (w * dL_gz) / denom_z);
        atomicAdd(&grad_grid[ix * (res_y * res_z) + iy * res_z + iz_m], -(w * dL_gz) / denom_z);
      }
    }
  }
}

torch::Tensor gradient_interp_cuda_forward(
    torch::Tensor grid,
    torch::Tensor coords,
    torch::Tensor voxel_size) {
  const int N = coords.size(0);
  auto out_grad = torch::zeros({N, 3}, coords.options());

  const int res_x = grid.size(grid.dim() - 3);
  const int res_y = grid.size(grid.dim() - 2);
  const int res_z = grid.size(grid.dim() - 1);

  const float eps_x = voxel_size[0].item<float>();
  const float eps_y = voxel_size[1].item<float>();
  const float eps_z = voxel_size[2].item<float>();

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(grid.scalar_type(), "gradient_interp_forward_kernel", ([&] {
    gradient_interp_forward_kernel<scalar_t><<<blocks, threads>>>(
        grid.data_ptr<scalar_t>(),
        coords.data_ptr<scalar_t>(),
        out_grad.data_ptr<scalar_t>(),
        res_x, res_y, res_z,
        eps_x, eps_y, eps_z,
        N);
  }));

  return out_grad;
}

void gradient_interp_cuda_backward(
    torch::Tensor grad_out,
    torch::Tensor coords,
    torch::Tensor grad_grid,
    torch::Tensor voxel_size) {
  const int N = coords.size(0);

  const int res_x = grad_grid.size(grad_grid.dim() - 3);
  const int res_y = grad_grid.size(grad_grid.dim() - 2);
  const int res_z = grad_grid.size(grad_grid.dim() - 1);

  const float eps_x = voxel_size[0].item<float>();
  const float eps_y = voxel_size[1].item<float>();
  const float eps_z = voxel_size[2].item<float>();

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(grad_grid.scalar_type(), "gradient_interp_backward_kernel", ([&] {
    gradient_interp_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_out.data_ptr<scalar_t>(),
        coords.data_ptr<scalar_t>(),
        grad_grid.data_ptr<scalar_t>(),
        res_x, res_y, res_z,
        eps_x, eps_y, eps_z,
        N);
  }));
}

