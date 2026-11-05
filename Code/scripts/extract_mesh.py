"""Marching Cubes Surface Extraction CLI Entrypoint with Advanced Geometry Post-Processing and Photorealistic Colorization.
Provenance: [PAPER] (Sec. 4.4 & 5.1: 320^3 Marching Cubes with Gaussian filtering) +
            [REFERENCE IMPLEMENTATION] (NeuS / Voxurf visual hull pruning, continuous normal export, and GT texture baking).
Usage:
    # 1. Smooth paper-like mesh with photorealistic GT colors:
    python scripts/extract_mesh.py --checkpoint logs/scan114_final/checkpoint_step_40000.pt \
                                   --data_dir /home/rushikesh/.cache/kagglehub/datasets/jonasamar/dtu-dataset/versions/1/DTU/scan114 \
                                   --output logs/scan114_final/mesh_scan114_gt_baked.ply \
                                   --post_filter 1.2 --clean_components --color_mode gt_bake

    # 2. Smooth paper-like mesh with camera-aligned neural radiance colors:
    python scripts/extract_mesh.py --checkpoint logs/scan114_final/checkpoint_step_40000.pt \
                                   --data_dir /home/rushikesh/.cache/kagglehub/datasets/jonasamar/dtu-dataset/versions/1/DTU/scan114 \
                                   --output logs/scan114_final/mesh_scan114_neural.ply \
                                   --post_filter 1.2 --clean_components --color_mode neural
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from voxneus.models.voxneus import VoxNeuS
from voxneus.training.checkpoint import load_checkpoint
from voxneus.datasets.dtu import DTUDataset
from voxneus.geometry.mesh import (
    extract_mesh_marching_cubes,
    compute_mesh_normals,
    prune_mesh_by_bounding_sphere,
    prune_mesh_by_z_cutoff,
    clean_mesh_components,
    prune_mesh_by_visual_hull,
    colorize_mesh_vertices,
    bake_gt_colors_to_mesh,
    save_mesh_ply,
    save_mesh_obj,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract high-fidelity 3D surface mesh from VoxNeuS")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--output", "--output_mesh", dest="output", type=str, default="extracted_mesh.ply", help="Output .ply or .obj path")
    parser.add_argument("--data_dir", type=str, default=None, help="Optional path to dataset (enables camera-aligned neural colors, GT texture baking, and visual hull pruning)")
    parser.add_argument("--level", type=float, default=0.0, help="SDF iso-surface level threshold (default 0.0)")
    parser.add_argument("--resolution", type=int, default=None, help="Mesh extraction resolution override (default: native checkpoint resolution, 320)")
    parser.add_argument("--post_filter", type=float, default=0.7, help="Gaussian filter sigma: 0.0 for pure sharp geometry, 0.7 for paper-accurate smoothness (default 0.7)")
    parser.add_argument("--color_mode", type=str, choices=["neural", "gt_bake", "none"], default="gt_bake", help="Colorization mode: 'gt_bake' (photorealistic GT sharpness), 'neural' (trained neural network), or 'none'")
    parser.add_argument("--bake_texture", action="store_true", help="Enable photorealistic GT texture baking (sets color_mode to gt_bake)")
    parser.add_argument("--crop_radius", type=float, default=1.05, help="Bounding sphere radius (default 1.05 for DTU) to safely preserve complete object")
    parser.add_argument("--table_cutoff", type=float, default=None, help="Optional tabletop planar cutoff threshold (default None)")
    parser.add_argument("--bake_reso", type=int, default=2, help="Resolution level for GT color baking: 1 for full 1600x1200 photographic sharpness, 2 for half")
    parser.add_argument("--clean_components", action="store_true", default=True, help="Remove isolated floating noise particles/islands (default True)")
    parser.add_argument("--keep_largest_only", action="store_true", default=True, help="Keep only the single largest connected component (default True)")
    parser.add_argument("--min_component_verts", type=int, default=5000, help="Minimum vertex count per connected component (default: 5000)")
    parser.add_argument("--visual_hull", action="store_true", help="Prune non-visual-hull vertices using DTU foreground masks")
    parser.add_argument("--vh_dilation", type=int, default=1, help="Visual hull mask dilation iterations (default: 1)")
    parser.add_argument("--vh_coverage", type=float, default=0.60, help="Visual hull minimum mask coverage ratio (default: 0.60)")
    parser.add_argument("--ascii", action="store_true", help="Save ASCII PLY instead of fast binary PLY")
    parser.add_argument("--device", type=str, default=None, help="Target device (default: cuda if available)")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n========================================================")
    print(f"[VoxNeuS] Loading model checkpoint: {args.checkpoint} on {device}...")
    model = VoxNeuS()
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)
    model.eval()

    # Load dataset if provided
    dataset = None
    cam_centers = None
    if args.bake_texture:
        args.color_mode = "gt_bake"
    if args.data_dir and os.path.exists(args.data_dir):
        print(f"[VoxNeuS] Loading reference dataset from: {args.data_dir} (reso_level={args.bake_reso})...")
        dataset = DTUDataset(basedir=args.data_dir, reso_level=args.bake_reso, normalize=True, use_mask=True)
        cam_centers = dataset.poses[:, :3, 3]

    if args.resolution is not None and args.resolution != model.sdf_grid.get_numpy_grid().shape[0]:
        print(f"[VoxNeuS] Upsampling grid to extraction resolution: {args.resolution}³...")
        model.sdf_grid.scale_volume_grid(args.resolution)

    grid_numpy = model.sdf_grid.get_numpy_grid()
    xyz_min = model.sdf_grid.xyz_min.cpu().numpy()
    xyz_max = model.sdf_grid.xyz_max.cpu().numpy()

    print(f"[VoxNeuS] SDF Volume resolution: {grid_numpy.shape}")
    print(f"[VoxNeuS] Running Marching Cubes (level={args.level}, 3D Gaussian post_filter={args.post_filter})...")

    verts, faces = extract_mesh_marching_cubes(
        sdf_grid=grid_numpy,
        xyz_min=xyz_min,
        xyz_max=xyz_max,
        level=args.level,
        post_filter_sigma=args.post_filter,
    )
    print(f"[VoxNeuS] Raw extracted geometry: {len(verts):,} vertices, {len(faces):,} faces.")

    # 1. Compute continuous surface normals on vertices
    print(f"[VoxNeuS] Computing continuous surface normals via SDF gradient interpolation...")
    normals = compute_mesh_normals(model, verts, batch_size=65536, device=device)

    colors = None

    # 2. Radial bounding sphere cropping first (removes far grid boundaries and camera shell)
    if args.crop_radius is not None and args.crop_radius > 0:
        print(f"[VoxNeuS] Cropping background outside sphere radius {args.crop_radius}...")
        verts, faces, normals, colors = prune_mesh_by_bounding_sphere(
            verts, faces, radius=args.crop_radius, normals=normals, colors=colors
        )
        print(f"[VoxNeuS] After sphere cropping: {len(verts):,} vertices, {len(faces):,} faces.")

    # 2b. Tabletop planar cutoff (severs the DTU turntable plane from the base of the object)
    if args.table_cutoff is not None:
        print(f"[VoxNeuS] Severing tabletop below Z={args.table_cutoff}...")
        verts, faces, normals, colors = prune_mesh_by_z_cutoff(
            verts, faces, z_min=args.table_cutoff, normals=normals, colors=colors
        )
        print(f"[VoxNeuS] After table severing: {len(verts):,} vertices, {len(faces):,} faces.")

    # 3. Visual hull mask pruning using DTU foreground masks (removes tabletop and background floaters)
    if args.visual_hull and dataset is not None and getattr(dataset, "masks", None) is not None:
        print(f"[VoxNeuS] Pruning mesh using DTU foreground mask visual hull (dilation={args.vh_dilation}, min_coverage={args.vh_coverage})...")
        verts, faces, normals, colors = prune_mesh_by_visual_hull(
            verts, faces, dataset, dilation=args.vh_dilation, min_coverage_ratio=args.vh_coverage, normals=normals, colors=colors
        )
        print(f"[VoxNeuS] After visual hull pruning: {len(verts):,} vertices, {len(faces):,} faces.")

    # 4. Connected component filtering to isolate primary object
    if args.clean_components and len(faces) > 0:
        print(f"[VoxNeuS] Isolating primary component (keep_largest_only={args.keep_largest_only})...")
        verts, faces, normals, colors = clean_mesh_components(
            verts, faces,
            min_component_verts=args.min_component_verts,
            keep_largest_only=args.keep_largest_only,
            normals=normals,
            colors=colors,
        )
        print(f"[VoxNeuS] After component cleaning: {len(verts):,} vertices, {len(faces):,} faces.")

    # 5. Colorization
    if args.color_mode == "gt_bake":
        if dataset is not None:
            print(f"[VoxNeuS] Baking occlusion-tested photographic colors from {len(dataset.images)} views (power=3.5)...")
            colors = bake_gt_colors_to_mesh(
                verts, normals, dataset, min_cos=0.05, power=3.5, depth_tol=0.02, model=model, device=device
            )
            print(f"[VoxNeuS] Ground-truth occlusion-aware texture baking complete.")
        else:
            print(f"[VoxNeuS] Warning: --data_dir not provided for 'gt_bake'. Falling back to camera-aligned neural colorization.")
            colors = colorize_mesh_vertices(model, verts, normals=normals, camera_centers=cam_centers, device=device)
    elif args.color_mode == "neural":
        print(f"[VoxNeuS] Colorizing {len(verts):,} vertices with trained neural appearance network (camera-aligned)...")
        colors = colorize_mesh_vertices(model, verts, normals=normals, camera_centers=cam_centers, device=device)
        print(f"[VoxNeuS] Neural colorization complete (RGB range: [{colors.min():.2f}, {colors.max():.2f}]).")
    else:
        colors = None

    # 6. Save output mesh
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.output.endswith(".obj"):
        save_mesh_obj(args.output, verts, faces, vertex_normals=normals, vertex_colors=colors)
    else:
        save_mesh_ply(args.output, verts, faces, vertex_normals=normals, vertex_colors=colors, binary=not args.ascii)

    color_status = f"{args.color_mode.upper()} Color" if colors is not None else "Uncolored Clay"
    fmt_status = "ASCII" if args.ascii else "Fast Binary"
    print(f"[VoxNeuS] SUCCESS: Saved {fmt_status} {color_status} 3D surface mesh to:\n          --> {args.output}")
    print(f"========================================================\n")


if __name__ == "__main__":
    main()
