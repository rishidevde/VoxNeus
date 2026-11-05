"""Quantitative Evaluation CLI (Chamfer Distance).
Provenance: [PAPER] (Tab. 1 & Tab. 2 DTU Surface Reconstruction Benchmark)
Usage:
    python scripts/evaluate.py --mesh logs/scan24/mesh.ply --gt_points /path/to/DTU/stl/stl024_total.ply
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from voxneus.evaluation.dtu_eval import compute_chamfer_distance, sample_points_from_mesh


def load_ply_points(filepath: str) -> np.ndarray:
    """Simple parser for PLY point clouds and vertices."""
    points = []
    with open(filepath, "r", encoding="latin1") as f:
        header = True
        for line in f:
            line = line.strip()
            if header:
                if line == "end_header":
                    header = False
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
    return np.array(points, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DTU Chamfer Distance")
    parser.add_argument("--mesh_points", type=str, required=True, help="Path to reconstructed mesh or point cloud PLY")
    parser.add_argument("--gt_points", type=str, required=True, help="Path to ground-truth STL scan PLY")
    parser.add_argument("--max_dist", type=float, default=20.0, help="Max distance threshold for outlier clamping (mm)")
    parser.add_argument("--crop_radius", type=float, default=None, help="Optional radius (e.g. 1.0) to crop table/outliers")
    args = parser.parse_args()

    print(f"[VoxNeuS] Loading predicted mesh points: {args.mesh_points}...")
    pred_pts = load_ply_points(args.mesh_points)
    print(f"[VoxNeuS] Loading ground-truth STL points: {args.gt_points}...")
    gt_pts = load_ply_points(args.gt_points)

    print(f"[VoxNeuS] Evaluating Chamfer Distance on {len(pred_pts):,} pred points vs {len(gt_pts):,} GT points...")
    results = compute_chamfer_distance(pred_pts, gt_pts, max_dist=args.max_dist, crop_radius=args.crop_radius)

    print("\n" + "=" * 60)
    print(f"Accuracy (pred -> GT)     : {results['accuracy']:.4f} mm")
    print(f"Completeness (GT -> pred) : {results['completeness']:.4f} mm")
    print(f"Chamfer Distance (Mean)   : {results['chamfer_distance']:.4f} mm")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
