#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
void adam_upd_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    int step, float beta1, float beta2, float lr, float eps);

void masked_adam_upd_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    int step, float beta1, float beta2, float lr, float eps);

torch::Tensor gradient_interp_cuda_forward(
    torch::Tensor grid,
    torch::Tensor coords,
    torch::Tensor voxel_size);

void gradient_interp_cuda_backward(
    torch::Tensor grad_out,
    torch::Tensor coords,
    torch::Tensor grad_grid,
    torch::Tensor voxel_size);

void eikonal_grad_cuda(
    torch::Tensor grid,
    torch::Tensor active_verts,
    torch::Tensor grad_grid,
    torch::Tensor voxel_size,
    float weight);

void curvature_grad_cuda(
    torch::Tensor grid,
    torch::Tensor active_verts,
    torch::Tensor grad_grid,
    torch::Tensor voxel_size,
    float weight);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void adam_upd(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    int step, float beta1, float beta2, float lr, float eps) {
  CHECK_INPUT(param);
  CHECK_INPUT(grad);
  CHECK_INPUT(exp_avg);
  CHECK_INPUT(exp_avg_sq);
  adam_upd_cuda(param, grad, exp_avg, exp_avg_sq, step, beta1, beta2, lr, eps);
}

void masked_adam_upd(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    int step, float beta1, float beta2, float lr, float eps) {
  CHECK_INPUT(param);
  CHECK_INPUT(grad);
  CHECK_INPUT(exp_avg);
  CHECK_INPUT(exp_avg_sq);
  masked_adam_upd_cuda(param, grad, exp_avg, exp_avg_sq, step, beta1, beta2, lr, eps);
}

torch::Tensor gradient_interp_forward(
    torch::Tensor grid,
    torch::Tensor coords,
    torch::Tensor voxel_size) {
  CHECK_INPUT(grid);
  CHECK_INPUT(coords);
  return gradient_interp_cuda_forward(grid, coords, voxel_size);
}

void gradient_interp_backward(
    torch::Tensor grad_out,
    torch::Tensor coords,
    torch::Tensor grad_grid,
    torch::Tensor voxel_size) {
  CHECK_INPUT(grad_out);
  CHECK_INPUT(coords);
  CHECK_INPUT(grad_grid);
  gradient_interp_cuda_backward(grad_out, coords, grad_grid, voxel_size);
}

void eikonal_grad(
    torch::Tensor grid,
    torch::Tensor active_verts,
    torch::Tensor grad_grid,
    torch::Tensor voxel_size,
    float weight) {
  CHECK_INPUT(grid);
  CHECK_INPUT(active_verts);
  CHECK_INPUT(grad_grid);
  eikonal_grad_cuda(grid, active_verts, grad_grid, voxel_size, weight);
}

void curvature_grad(
    torch::Tensor grid,
    torch::Tensor active_verts,
    torch::Tensor grad_grid,
    torch::Tensor voxel_size,
    float weight) {
  CHECK_INPUT(grid);
  CHECK_INPUT(active_verts);
  CHECK_INPUT(grad_grid);
  curvature_grad_cuda(grid, active_verts, grad_grid, voxel_size, weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adam_upd", &adam_upd, "Dense Adam parameter update");
  m.def("masked_adam_upd", &masked_adam_upd, "Sparse Adam parameter update (ignoring zero grad voxels)");
  m.def("gradient_interp_forward", &gradient_interp_forward, "CUDA gradient interpolation forward");
  m.def("gradient_interp_backward", &gradient_interp_backward, "CUDA gradient interpolation backward");
  m.def("eikonal_grad", &eikonal_grad, "CUDA closed-form Eikonal gradient accumulation");
  m.def("curvature_grad", &curvature_grad, "CUDA closed-form Curvature gradient accumulation");
}
