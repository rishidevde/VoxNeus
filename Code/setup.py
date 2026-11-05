"""Setup and Installation script for VoxNeuS.
Supports pure PyTorch CPU/GPU installation, and optionally compiles custom CUDA extensions if CUDA is present.
"""

import os
from setuptools import setup, find_packages

# Determine whether to build CUDA extensions
try:
    import torch
    BUILD_CUDA = torch.cuda.is_available() and os.environ.get("BUILD_CUDA_EXT", "1") == "1"
except ImportError:
    torch = None
    BUILD_CUDA = False

ext_modules = []
cmdclass = {}

if BUILD_CUDA:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    cuda_dir = os.path.join(os.path.dirname(__file__), "voxneus", "cuda")
    ext_modules = [
        CUDAExtension(
            name="voxneus.cuda._backend",
            sources=[
                os.path.join(cuda_dir, "adam_upd.cpp"),
                os.path.join(cuda_dir, "adam_upd_kernel.cu"),
                os.path.join(cuda_dir, "gradient_interp.cu"),
                os.path.join(cuda_dir, "eikonal_grad.cu"),
                os.path.join(cuda_dir, "curvature_grad.cu"),
            ],
            extra_compile_args={
                "cxx": ["/O2" if os.name == "nt" else "-O3"],
                "nvcc": ["-O3"],
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="voxneus",
    version="1.0.0",
    description="VoxNeuS: Enhancing Voxel-Based Neural Surface Reconstruction via Gradient Interpolation",
    author="Anonymous Authors (ACM TOG / SIGGRAPH)",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "Pillow>=8.0.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.60.0",
    ],
)
