"""BlendedMVS Dataset Loader.
Provenance: [REFERENCE IMPLEMENTATION] (Adapted from Voxurf load_blendedmvs.py)
"""

import os
from glob import glob
from typing import Dict, Tuple, List, Optional
import numpy as np
from PIL import Image
import torch

from voxneus.datasets.dtu import load_image_rgb


class BlendedMVSDataset:
    """BlendedMVS Dataset loader for qualitative neural surface evaluation."""

    def __init__(
        self,
        basedir: str,
        test_split_ratio: float = 0.1,
    ):
        self.basedir = basedir

        pose_paths = sorted(glob(os.path.join(basedir, "pose", "*txt")))
        rgb_paths = sorted(glob(os.path.join(basedir, "rgb", "*png")))
        if not rgb_paths:
            rgb_paths = sorted(glob(os.path.join(basedir, "image", "*png")))

        if not rgb_paths:
            raise FileNotFoundError(f"No RGB images found in {basedir}")

        all_imgs = []
        all_poses = []
        for p_path, im_path in zip(pose_paths, rgb_paths):
            all_imgs.append(load_image_rgb(im_path))
            all_poses.append(np.loadtxt(p_path).astype(np.float32))

        self.images = np.stack(all_imgs, axis=0)
        self.poses = np.stack(all_poses, axis=0)
        self.num_images = len(self.images)
        self.H, self.W = self.images[0].shape[:2]

        intrinsics_path = os.path.join(basedir, "intrinsics.txt")
        if os.path.exists(intrinsics_path):
            self.K = np.loadtxt(intrinsics_path).astype(np.float32)
        else:
            focal = 0.5 * self.W / np.tan(0.5 * 0.85)  # default approx focal
            self.K = np.array([
                [focal, 0.0, 0.5 * self.W],
                [0.0, focal, 0.5 * self.H],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)

        # Train / Test split
        n_test = max(int(self.num_images * test_split_ratio), 1)
        self.test_views = list(range(0, self.num_images, max(self.num_images // n_test, 1)))
        self.train_views = [i for i in range(self.num_images) if i not in self.test_views]

    def sample_rays(self, batch_size: int) -> Dict[str, torch.Tensor]:
        img_indices = np.random.choice(self.train_views, size=batch_size)
        xs = np.random.randint(0, self.W, size=batch_size)
        ys = np.random.randint(0, self.H, size=batch_size)

        poses = self.poses[img_indices]
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        dirs_cam = np.stack([(xs + 0.5 - cx) / fx, -(ys + 0.5 - cy) / fy, -np.ones_like(xs, dtype=np.float32)], axis=-1)
        rot = poses[:, :3, :3]
        dirs_world = np.sum(dirs_cam[:, np.newaxis, :] * rot, axis=-1)
        dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=-1, keepdims=True)
        origins = poses[:, :3, 3]

        rgbs = self.images[img_indices, ys, xs]

        return {
            "rays_o": torch.from_numpy(origins.astype(np.float32)),
            "rays_d": torch.from_numpy(dirs_world.astype(np.float32)),
            "rgb": torch.from_numpy(rgbs.astype(np.float32)),
        }
