"""DTU Surface Reconstruction Evaluation (Chamfer Distance).
Provenance: [REFERENCE IMPLEMENTATION] (Adapted from Voxurf dtu_eval.py / DTUeval-python)
Computes Chamfer Distance, Accuracy, and Completeness against STL ground-truth scans.
"""

from typing import Tuple, Dict, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree


def sample_single_tri(input_tuple):
    """Barycentric grid subdivision on a single triangle face.
    Provenance: [REFERENCE IMPLEMENTATION] (Voxurf lib/dtu_eval.py lines 19-28)
    """
    n1, n2, v1, v2, tri_vert = input_tuple
    c = np.mgrid[:n1 + 1, :n2 + 1].astype(np.float32)
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def sample_mesh_barycentric_grid(
    vertices: np.ndarray,
    triangles: np.ndarray,
    thresh: float = 0.2,
) -> np.ndarray:
    """Sample mesh surface using official DTUeval barycentric grid sampling.
    Provenance: [REFERENCE IMPLEMENTATION] (Voxurf lib/dtu_eval.py lines 70-89)
    """
    tri_vert = vertices[triangles]
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:, 0]

    l1 = l1[non_zero_area]
    l2 = l2[non_zero_area]
    area2 = area2[non_zero_area]
    v1 = v1[non_zero_area]
    v2 = v2[non_zero_area]
    tri_vert = tri_vert[non_zero_area]

    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr).astype(np.int32)
    n2 = np.floor(l2 / thr).astype(np.int32)

    pts_list = []
    for i in range(len(n1)):
        p = sample_single_tri((n1[i, 0], n2[i, 0], v1[i:i + 1], v2[i:i + 1], tri_vert[i:i + 1, 0]))
        if len(p) > 0:
            pts_list.append(p)

    new_pts = np.concatenate(pts_list, axis=0) if pts_list else np.empty((0, 3), dtype=np.float32)
    data_pcd = np.concatenate([vertices, new_pts], axis=0)
    return data_pcd.astype(np.float32)


def sample_points_from_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    num_samples: int = 100000,
) -> np.ndarray:
    """Sample points uniformly on the triangular mesh surface by face area weighting."""
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    # Compute face areas
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=-1)
    total_area = areas.sum()

    if total_area <= 0.0:
        return vertices

    probs = areas / total_area
    face_choices = np.random.choice(len(triangles), size=num_samples, p=probs)

    r1 = np.random.rand(num_samples, 1)
    r2 = np.random.rand(num_samples, 1)
    sqrt_r1 = np.sqrt(r1)

    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2

    sampled = u * v0[face_choices] + v * v1[face_choices] + w * v2[face_choices]
    return sampled.astype(np.float32)


def filter_bounding_sphere(
    points: np.ndarray,
    radius: float = 1.0,
    center: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Filter point cloud to keep only points inside bounding sphere (e.g. DTU normalized sphere)."""
    if center is None:
        center = np.zeros(3, dtype=points.dtype)
    dists = np.linalg.norm(points - center, axis=-1)
    return points[dists <= radius]


def compute_chamfer_distance(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    max_dist: float = 20.0,
    crop_radius: Optional[float] = None,
) -> Dict[str, float]:
    """Calculate Chamfer distance, Accuracy, and Completeness between prediction and ground-truth.

    Args:
        pred_points: (N, 3) Sampled points from reconstructed mesh.
        gt_points: (M, 3) Ground-truth STL scan points.
        max_dist: Outlier clamping distance threshold (default 20.0 mm).
        crop_radius: Optional sphere radius to crop background table / out-of-bounds points.

    Returns:
        Dictionary with:
            'accuracy': Mean distance from pred to nearest GT.
            'completeness': Mean distance from GT to nearest pred.
            'chamfer_distance': Mean of accuracy and completeness: (Acc + Comp) / 2.
    """
    if crop_radius is not None and crop_radius > 0:
        pred_points = filter_bounding_sphere(pred_points, radius=crop_radius)
        gt_points = filter_bounding_sphere(gt_points, radius=crop_radius)
    # 1. Accuracy: pred -> GT
    gt_tree = cKDTree(gt_points)
    dist_pred_to_gt, _ = gt_tree.query(pred_points, k=1, workers=-1)
    if max_dist > 0:
        dist_pred_to_gt = np.clip(dist_pred_to_gt, 0, max_dist)
    accuracy = float(np.mean(dist_pred_to_gt))

    # 2. Completeness: GT -> pred
    pred_tree = cKDTree(pred_points)
    dist_gt_to_pred, _ = pred_tree.query(gt_points, k=1, workers=-1)
    if max_dist > 0:
        dist_gt_to_pred = np.clip(dist_gt_to_pred, 0, max_dist)
    completeness = float(np.mean(dist_gt_to_pred))

    chamfer = (accuracy + completeness) / 2.0

    return {
        "accuracy": accuracy,
        "completeness": completeness,
        "chamfer_distance": chamfer,
    }


def evaluate_dtu_official(
    mesh_path: str,
    scan_id: int,
    dtu_dir: str,
    output_dir: Optional[str] = None,
    downsample_density: float = 0.2,
    max_dist: float = 20.0,
) -> Dict[str, float]:
    """Execute official DTUeval benchmark protocol matching Table 1 & 2 of the paper.
    Provenance: [REFERENCE IMPLEMENTATION] (Voxurf lib/dtu_eval.py / DTUeval-python)

    Args:
        mesh_path: Path to reconstructed .ply mesh.
        scan_id: Integer DTU scan ID (e.g. 114).
        dtu_dir: Path to DTU ground truth directory containing 'Points/stl' and 'ObsMask'.
        output_dir: Optional directory to save visual error point clouds.
        downsample_density: Surface sampling density threshold (default 0.2 mm).
        max_dist: Clamping threshold for outliers (default 20.0 mm).

    Returns:
        Dict with 'accuracy', 'completeness', and 'chamfer_distance'.
    """
    import os
    import trimesh
    from scipy.io import loadmat

    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # 1. Sample dense point cloud from mesh faces using official DTUeval barycentric grid subdivision
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(faces) < 500000:
        data_pcd = sample_mesh_barycentric_grid(vertices, faces, thresh=downsample_density)
    else:
        num_samples = min(len(faces) * 10, 1000000)
        data_pcd = sample_points_from_mesh(vertices, faces, num_samples=num_samples)

    # 2. Downsample point cloud with spatial KD-tree radius
    tree = cKDTree(data_pcd)
    keep_mask = np.ones(len(data_pcd), dtype=bool)
    for i, idxs in enumerate(tree.query_ball_point(data_pcd, r=downsample_density)):
        if keep_mask[i]:
            keep_mask[idxs] = False
            keep_mask[i] = True
    data_down = data_pcd[keep_mask]

    # 3. Check for official ObsMask
    obs_mask_path = os.path.join(dtu_dir, "ObsMask", f"ObsMask{scan_id}_10.mat")
    plane_path = os.path.join(dtu_dir, "ObsMask", f"Plane{scan_id}.mat")
    stl_path = os.path.join(dtu_dir, "Points", "stl", f"stl{scan_id:03d}_total.ply")

    if not os.path.exists(stl_path):
        stl_path_alt = os.path.join(dtu_dir, f"stl{scan_id:03d}_total.ply")
        if os.path.exists(stl_path_alt):
            stl_path = stl_path_alt
        else:
            raise FileNotFoundError(f"DTU STL ground truth scan not found at: {stl_path}")

    stl_mesh = trimesh.load(stl_path)
    stl_pts = np.asarray(stl_mesh.vertices if hasattr(stl_mesh, "vertices") else stl_mesh)

    data_eval = data_down
    if os.path.exists(obs_mask_path):
        obs_mat = loadmat(obs_mask_path)
        obs_mask, bb, res = obs_mat["ObsMask"], obs_mat["BB"].astype(np.float32), obs_mat["Res"]
        inbound = ((data_down >= bb[:1] - 60) & (data_down < bb[1:] + 120)).all(axis=-1)
        data_in = data_down[inbound]
        data_grid = np.around((data_in - bb[:1]) / res).astype(np.int32)
        grid_in = ((data_grid >= 0) & (data_grid < np.expand_dims(obs_mask.shape, 0))).all(axis=-1)
        data_grid_in = data_grid[grid_in]
        in_obs = obs_mask[data_grid_in[:, 0], data_grid_in[:, 1], data_grid_in[:, 2]].astype(bool)
        data_eval = data_in[grid_in][in_obs]

    # Filter GT points below tabletop plane
    if os.path.exists(plane_path):
        plane = loadmat(plane_path)["P"]
        stl_hom = np.concatenate([stl_pts, np.ones_like(stl_pts[:, :1])], axis=-1)
        above = (plane.reshape(1, 4) * stl_hom).sum(-1) > 0
        stl_pts = stl_pts[above]

    # 4. Accuracy: data -> STL
    stl_tree = cKDTree(stl_pts)
    d2s, _ = stl_tree.query(data_eval, k=1, workers=-1)
    acc = float(np.mean(d2s[d2s < max_dist])) if len(d2s) > 0 else 0.0

    # 5. Completeness: STL -> data
    data_tree = cKDTree(data_down)
    s2d, _ = data_tree.query(stl_pts, k=1, workers=-1)
    comp = float(np.mean(s2d[s2d < max_dist])) if len(s2d) > 0 else 0.0

    cd = (acc + comp) / 2.0

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"result_scan{scan_id}.txt"), "w") as f:
            f.write(f"Accuracy: {acc:.4f}\nCompleteness: {comp:.4f}\nChamfer Distance: {cd:.4f}\n")

    return {
        "accuracy": acc,
        "completeness": comp,
        "chamfer_distance": cd,
    }

