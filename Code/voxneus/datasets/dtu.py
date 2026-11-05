"""DTU Dataset Loader for Neural Surface Reconstruction.
Provenance: [REFERENCE IMPLEMENTATION] (Adapted from Voxurf load_dtu.py / IDR load_K_Rt_from_P)
Supports all 15 official DTU benchmark scans, camera sphere normalization,
and image loading via PIL (with fallback for OpenCV/ImageIO).
"""

import os
import re
from glob import glob
from typing import Dict, Tuple, List, Optional, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.linalg import rq

from voxneus.rendering.rays import get_rays_np


def decompose_projection_matrix(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose 3x4 projection matrix P = K [R | t] into intrinsic matrix K and camera pose [R^T | C].
    Uses RQ decomposition from SciPy (equivalent to cv2.decomposeProjectionMatrix without OpenCV dependency).
    Provenance: [REFERENCE IMPLEMENTATION] (IDR load_K_Rt_from_P / Voxurf load_dtu.py)

    Args:
        P: 3x4 projection matrix.

    Returns:
        intrinsics: 4x4 intrinsic matrix.
        pose: 4x4 camera-to-world (c2w) transformation matrix [R^T | C].
    """
    # M = K @ R (3x3)
    M = P[:3, :3]
    p4 = P[:3, 3]

    # RQ decomposition gives upper triangular K and orthogonal R
    K, R = rq(M)

    # Ensure diagonal of K is positive
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R

    # Ensure det(R) == +1 (proper rotation, not reflection)
    if np.linalg.det(R) < 0:
        R = -R
        K = -K

    # Normalize K so K[2, 2] == 1.0
    K = K / K[2, 2]

    # Camera center C in world coordinates: M @ C + p4 = 0 => C = - M^-1 @ p4
    C = -np.linalg.inv(M) @ p4

    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.T  # c2w rotation is R^T
    pose[:3, 3] = C     # camera origin in world frame

    return intrinsics, pose


def load_image_rgb(path: str) -> np.ndarray:
    """Load image and normalize to [0, 1] float32 RGB array."""
    with Image.open(path) as img:
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb, dtype=np.float32) / 255.0
    return arr


def load_image_mask(path: str) -> np.ndarray:
    """Load mask and normalize to [0, 1] float32 array."""
    with Image.open(path) as img:
        img_l = img.convert("L")
        arr = np.array(img_l, dtype=np.float32) / 255.0
    return arr[..., None]


class DTUDataset:
    """DTU Dataset supporting ray batch sampling, test views, and camera sphere normalization.
    Official 15 Scans: 24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122.
    """

    def __init__(
        self,
        basedir: str,
        reso_level: int = 2,
        normalize: bool = True,
        use_mask: bool = False,
        white_bg: bool = False,
        test_views: Optional[List[int]] = None,
        device: str = "cpu",
    ):
        self.basedir = basedir
        self.reso_level = reso_level
        self.normalize = normalize
        self.use_mask = use_mask
        self.white_bg = white_bg
        self.device = device

        # 1. Discover image and mask files
        rgb_paths = sorted(glob(os.path.join(basedir, "image", "*png")))
        if not rgb_paths:
            rgb_paths = sorted(glob(os.path.join(basedir, "image", "*jpg")))
        if not rgb_paths:
            rgb_paths = sorted(glob(os.path.join(basedir, "rgb", "*png")))
        if not rgb_paths:
            rgb_paths = sorted(glob(os.path.join(basedir, "images", "*png")))
        if not rgb_paths:
            rgb_paths = sorted(glob(os.path.join(basedir, "images", "*jpg")))
        if not rgb_paths:
            raise FileNotFoundError(f"No image files found in {basedir}/image, {basedir}/images, or {basedir}/rgb")

        mask_paths = sorted(glob(os.path.join(basedir, "mask", "*png")))
        if not mask_paths:
            mask_paths = sorted(glob(os.path.join(basedir, "mask", "*jpg")))
        if not mask_paths:
            mask_paths = sorted(glob(os.path.join(basedir, "masks", "*png")))

        self.num_images = len(rgb_paths)

        # 2. Load camera matrices with robust frame index matching
        cameras_npz = "cameras_sphere.npz" if normalize else "cameras_large.npz"
        cam_path = os.path.join(basedir, cameras_npz)
        if not os.path.exists(cam_path):
            cam_path_alt = os.path.join(basedir, "cameras.npz")
            if os.path.exists(cam_path_alt):
                cam_path = cam_path_alt
            else:
                raise FileNotFoundError(f"Camera file not found: {cam_path} or {cam_path_alt}")

        camera_dict = np.load(cam_path)

        # Build mask lookup map by frame number
        mask_map = {}
        if mask_paths:
            for p in mask_paths:
                m_matches = re.findall(r"\d+", os.path.basename(p))
                if m_matches:
                    mask_map[int(m_matches[-1])] = p

        all_intrinsics = []
        all_poses = []
        all_images = []
        all_masks = []

        print(f"[DTUDataset] Loading {self.num_images} images from {basedir} (reso_level={reso_level})...")

        for idx, im_path in enumerate(rgb_paths):
            # Extract true numeric frame ID from image filename (e.g. '000003.png' -> 3)
            im_matches = re.findall(r"\d+", os.path.basename(im_path))
            frame_num = int(im_matches[-1]) if im_matches else idx

            w_key = f"world_mat_{frame_num}"
            s_key = f"scale_mat_{frame_num}"

            if w_key not in camera_dict:
                # Fallback to sequential index if named key not present
                w_key = f"world_mat_{idx}"
                s_key = f"scale_mat_{idx}"

            w_mat = camera_dict[w_key].astype(np.float32)
            s_mat = camera_dict[s_key].astype(np.float32) if (normalize and s_key in camera_dict) else None

            if normalize and s_mat is not None:
                P = (w_mat @ s_mat)[:3, :4]
            else:
                P = w_mat[:3, :4]

            intrinsics, pose = decompose_projection_matrix(P)
            all_intrinsics.append(intrinsics)
            all_poses.append(pose)

            img = load_image_rgb(im_path)
            all_images.append(img)

            # Match mask by frame number
            m_path = mask_map.get(frame_num, mask_paths[idx] if idx < len(mask_paths) else None) if mask_paths else None
            if m_path is not None and os.path.exists(m_path):
                m = load_image_mask(m_path)
                all_masks.append(m)

        self.images = np.stack(all_images, axis=0)  # (N, H, W, 3)
        self.poses = np.stack(all_poses, axis=0)    # (N, 4, 4)
        self.intrinsics = np.stack(all_intrinsics, axis=0)  # (N, 4, 4)
        self.masks = np.stack(all_masks, axis=0) if len(all_masks) == len(all_images) else None

        H, W = self.images[0].shape[:2]
        self.H, self.W = H, W
        self.K = self.intrinsics[0, :3, :3]

        # Downsample resolution if reso_level > 1
        if reso_level > 1:
            H_new, W_new = int(H / reso_level), int(W / reso_level)
            img_tensor = torch.from_numpy(self.images).permute(0, 3, 1, 2)
            self.images = F.interpolate(img_tensor, size=(H_new, W_new), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).numpy()
            if self.masks is not None:
                mask_tensor = torch.from_numpy(self.masks).permute(0, 3, 1, 2)
                self.masks = F.interpolate(mask_tensor, size=(H_new, W_new), mode="nearest").permute(0, 2, 3, 1).numpy()
            self.H, self.W = H_new, W_new
            self.intrinsics[:, :2, :3] /= reso_level
            self.K = self.intrinsics[0, :3, :3]

        # Test view split from Voxurf [REFERENCE IMPLEMENTATION]
        default_test = [8, 13, 16, 21, 26, 31, 34]
        if self.num_images * 0.1 >= 8 and 56 < self.num_images:
            default_test.append(56)
        self.test_views = test_views if test_views is not None else [v for v in default_test if v < self.num_images]
        self.train_views = [i for i in range(self.num_images) if i not in self.test_views]

        print(f"[DTUDataset] Loaded: {self.H}x{self.W} resolution, {len(self.train_views)} train views, {len(self.test_views)} test views.")

    def sample_rays(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch of rays across all training images.

        Args:
            batch_size: Number of rays to sample.

        Returns:
            Dictionary with 'rays_o', 'rays_d', 'rgb', and optional 'mask'.
        """
        # Randomly choose images from train views
        img_indices = np.random.choice(self.train_views, size=batch_size)
        # Random pixel coordinates (x in [0, W), y in [0, H))
        xs = np.random.randint(0, self.W, size=batch_size)
        ys = np.random.randint(0, self.H, size=batch_size)

        # Retrieve camera rays
        poses = self.poses[img_indices]  # (B, 4, 4)

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # Camera rays in OpenCV convention (+X right, +Y down, +Z forward)
        dirs_cam = np.stack([(xs + 0.5 - cx) / fx, (ys + 0.5 - cy) / fy, np.ones_like(xs, dtype=np.float32)], axis=-1)
        # Rotate directions: dirs_world = dirs_cam @ R^T
        rot = poses[:, :3, :3]
        dirs_world = np.sum(dirs_cam[:, np.newaxis, :] * rot, axis=-1)
        dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=-1, keepdims=True)
        origins = poses[:, :3, 3]

        rgbs = self.images[img_indices, ys, xs]

        output = {
            "rays_o": torch.from_numpy(origins.astype(np.float32)),
            "rays_d": torch.from_numpy(dirs_world.astype(np.float32)),
            "rgb": torch.from_numpy(rgbs.astype(np.float32)),
        }

        if self.masks is not None:
            masks = self.masks[img_indices, ys, xs]
            output["mask"] = torch.from_numpy(masks.astype(np.float32))

        return output

    def get_test_rays(self, img_idx: int) -> Dict[str, torch.Tensor]:
        """Generate full image rays and ground-truth pixels for evaluation."""
        pose = self.poses[img_idx]
        rays_o, rays_d = get_rays_np(self.H, self.W, self.K, pose, normalize_dir=True, convention="opencv")
        target_rgb = self.images[img_idx]

        data = {
            "rays_o": torch.from_numpy(rays_o.reshape(-1, 3).astype(np.float32)),
            "rays_d": torch.from_numpy(rays_d.reshape(-1, 3).astype(np.float32)),
            "rgb": torch.from_numpy(target_rgb.reshape(-1, 3).astype(np.float32)),
            "H": self.H,
            "W": self.W,
        }
        if self.masks is not None:
            data["mask"] = torch.from_numpy(self.masks[img_idx].reshape(-1, 1).astype(np.float32))
        return data
