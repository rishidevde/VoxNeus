# VoxNeuS: Enhancing Voxel-Based Neural Surface Reconstruction via Gradient Interpolation

📄 **Paper:** [VoxNeuS: Enhancing Voxel-Based Neural Surface Reconstruction via Gradient Interpolation](https://arxiv.org/pdf/2406.07170)

PyTorch and CUDA reimplementation of the paper **VoxNeuS** (ACM TOG / SIGGRAPH).

---

## 💎 Updates in `mesh_quality_updated_4` (High-Fidelity Surface & Color Improvements)

### 1. DTU Multi-View Camera Intrinsics Downsample Fix (`dtu.py`)
- **Bug Fixed**: When downsampling images by $2\times$ (`reso_level=2`), only camera 0 (`self.K`) was scaled, while the 3D intrinsics array for views 1 through 48 remained at full $2\times$ resolution.
- **Resolution**: Properly scaled `self.intrinsics[:, :2, :3] /= reso_level`, ensuring pixel-accurate 2D $\leftrightarrow$ 3D projection across all 49/64 DTU views for ray sampling, mask evaluation, and texture baking.

### 2. Occlusion-Aware Photographic Texture Baking (`mesh.py`)
- **Bug Fixed**: Previous ground-truth baking lacked line-of-sight visibility checking, causing front-view cameras to project through the object (cheeks, noses) onto occluded interior cavities and rear surfaces, producing blotches and muddied colors.
- **Resolution**: Implemented rasterized Z-buffer depth testing in `bake_gt_colors_to_mesh` with local minimum filtering ($z_{proj} \le z_{min} + \text{depth}_{\text{tol}}$) and foreground mask filtering.
- **Seamless Fallback**: Deep crevices with zero camera line-of-sight now seamlessly query the trained appearance network (`model.forward_points`) instead of flat gray/brown median colors.

### 3. Robust Mesh Extraction Pipeline (`extract_mesh.py`)
- **Pipeline Reordering**: Bounding sphere cropping (`--crop_radius 0.65`) now executes before connected component isolation, cleanly severing outer bounding shells and ensuring the true primary object is isolated.
- **Non-Destructive Visual Hull**: Made visual hull filtering strictly opt-in (`--visual_hull`) with non-destructive parameters (`dilation=25`, `min_coverage=0.15`) to prevent carving away rear and top surfaces on hemisphere camera setups like DTU.

### 4. Volume Rendering & Sampling Parity (`sampler.py`, `volume_render.py`, `voxneus.py`)
- Added NeuS 4-step progressive upsampling with segment midpoint evaluation and inside-sphere foreground masking.
- Relaxed variance upper clamp $s_{max} = 10000.0$ and calibrated variance / regularizer scheduling for razor-sharp zero-crossing convergence.

---

## 🌟 Key Contributions & Mathematical Formulation

1. **Continuous Gradient Interpolation (Eq. 7):**

   $$grad^{(i)}(\mathbf{x}) = \sum_{k=1}^8 w_k(\mathbf{x}) \mathbf{n}[\mathbf{c}_k(\mathbf{x})]$$

   Replaces discontinuous analytical gradients with smooth trilinear interpolation of central finite differences:

   $$\mathbf{n}[x, y, z] = \left( \frac{f[x+1, y, z] - f[x-1, y, z]}{2\epsilon_x}, \frac{f[x, y+1, z] - f[x, y-1, z]}{2\epsilon_y}, \frac{f[x, y, z+1] - f[x, y, z-1]}{2\epsilon_z} \right)$$

2. **VoxNeuS-Specific NeuS Discrete Opacity (Eq. 2):**

   $$\alpha_i = \max\left( \frac{\Phi_s(f(\mathbf{p}(t_i)) - \frac{\delta_i}{2}\cos\theta_i) - \Phi_s(f(\mathbf{p}(t_i)) + \frac{\delta_i}{2}\cos\theta_i)}{\Phi_s(f(\mathbf{p}(t_i)) - \frac{\delta_i}{2}\cos\theta_i)}, 0 \right)$$

   Where $\cos\theta_i = \langle \mathbf{n}_i, \mathbf{v} \rangle$ uses the interpolated gradient normal vector.

3. **Explicit Closed-Form Regularization on Active Vertices $V_R$ (Eq. 9-14):**

   $$\mathcal{L}_{eik} = \frac{1}{\lvert V_R \rvert} \sum_{\mathbf{x} \in V_R} (\lVert \mathbf{n}[\mathbf{x}] \rVert_2 - 1)^2, \quad \frac{\partial \mathcal{L}_{eik}}{\partial \mathbf{n}[\mathbf{x}]} = \frac{2}{\lvert V_R \rvert}\left(1 - \frac{1}{\lVert \mathbf{n}[\mathbf{x}] \rVert_2}\right)\mathbf{n}[\mathbf{x}]$$

   $$\mathcal{L}_{curv} = \frac{1}{\lvert V_R \rvert} \sum_{\mathbf{x} \in V_R} \lVert \nabla^2 f[\mathbf{x}] \rVert_2^2, \quad \boldsymbol{\mathcal{G}}_{curv} = \frac{2}{\lvert V_R \rvert} \sum_{\mathbf{x} \in V_R} \nabla^2 f[\mathbf{x}] \cdot \frac{\partial \nabla^2 f[\mathbf{x}]}{\partial f[U(\mathbf{x})]}$$

   Directly accumulated into grid gradients without PyTorch autograd computation graph overhead.

4. **$s$-Gradient Amplification ($k=5$) (Eq. 16):**

   $$\nabla_s \mathcal{L} \leftarrow 5 \cdot \nabla_s \mathcal{L} \quad \text{if } \nabla_s \mathcal{L} < 0$$

5. **Disentangled Architecture (Sec. 4.3):**
   Dense SDF Grid parameterizes geometry, multiresolution hash grid (`tiny-cuda-nn` / pure PyTorch) parameterizes appearance, and a shallow 2-layer MLP receives hash features, view directions, and surface normals (zero raw SDF value leakage).

6. **Dual-Backend Support:**
   - **Pure PyTorch Vectorized Mode:** Fully runs on CPU or GPU without requiring CUDA compilers or `tiny-cuda-nn`.
   - **CUDA Acceleration:** Includes fused CUDA C++ extension kernels for GPU production training.

---

## 🚀 Quickstart

### 1. Running the Automated Test Suite
Verify ray generation, discrete opacity, cell-boundary gradient continuity, closed-form autograd parity, and grid upsampling:

```bash
python -m unittest discover tests
```

### 2. Training on DTU Dataset (Scan 24)
Train for the full 40,000 steps with ray batch size of 2048:

```bash
python scripts/train.py --config configs/dtu_default.yaml --data_dir /path/to/DTU/scan24 --output_dir logs/scan24
```

For quick testing or smoke test:
```bash
python scripts/train.py --config configs/dtu_default.yaml --data_dir /path/to/DTU/scan24 --total_steps 10 --batch_size 128 --output_dir logs/debug_test
```

### 3. Extracting 3D Mesh
Extract the zero-level set via Marching Cubes at $320^3$ resolution with optional Gaussian filtering:

```bash
python scripts/extract_mesh.py --checkpoint logs/scan24/checkpoint_final.pt --output logs/scan24/mesh.ply --post_filter 1.0
```

### 4. Novel View Synthesis & Image Rendering
Render novel views and test views:

```bash
python scripts/render.py --checkpoint logs/scan24/checkpoint_final.pt --data_dir /path/to/DTU/scan24 --output_dir renders/scan24
```

### 5. Evaluating DTU Chamfer Distance
Evaluate reconstructed mesh surface accuracy, completeness, and Chamfer distance against ground-truth STL point clouds:

```bash
python scripts/evaluate.py --mesh_points logs/scan24/mesh.ply --gt_points /path/to/DTU/stl/stl024_total.ply
```