import torch
import numpy as np
import os
import glob
from tqdm import tqdm
from PIL import Image
try:
    from .color_utils import read_image, read_semantic
except:
    from color_utils import read_image, read_semantic
try:
    from .colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary
except:
    from colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary

try:
    from .ray_utils import *
except:
    from ray_utils import *
try:
    from .base import BaseDataset
except:
    from base import BaseDataset
import matplotlib.pyplot as plt
import pickle5 as pickle
import json

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (np.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = np.linalg.norm(v)
    skew_sym_mat = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return np.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s ** 2 + 1e-8))


def auto_orient_and_center_poses(
        poses, method="up", center_poses: bool = True
):
    """Orients and centers the poses. We provide two methods for orientation: pca and up.

    pca: Orient the poses so that the principal component of the points is aligned with the axes.
        This method works well when all of the cameras are in the same plane.
    up: Orient the poses so that the average up vector is aligned with the z axis.
        This method works well when images are not at arbitrary angles.


    Args:
        poses: The poses to orient.
        method: The method to use for orientation.
        center_poses: If True, the poses are centered around the origin.

    Returns:
        The oriented poses.
    """

    translation = poses[..., :3, 3]

    mean_translation = np.mean(translation, axis=0)
    translation_diff = translation - mean_translation

    if center_poses:
        translation = mean_translation
    else:
        translation = np.zeros_like(mean_translation)

    if method == "up":
        up = np.mean(poses[:, :3, 1], axis=0)
        up = up / np.linalg.norm(up)

        rotation = rotation_matrix(up, np.array([0, 0, 1]))
        transform = np.concatenate([rotation, rotation @ -translation[..., None]], axis=-1)
        oriented_poses = transform @ poses
    elif method == "none":
        transform = np.eye(4)
        transform[:3, 3] = -translation
        transform = transform[:3, :]
        oriented_poses = transform @ poses

    return oriented_poses, transform


def auto_normalize_poses(camera_to_worlds, scale_factor=1.0):
    camera_to_worlds, transform = auto_orient_and_center_poses(
        camera_to_worlds,
        method="up",
        center_poses=True,
    )

    scale_factor /= float(np.max(np.abs(camera_to_worlds[:, :3, 3])))

    camera_to_worlds[:, :3, 3] *= scale_factor

    return camera_to_worlds, scale_factor, transform


def center_poses(poses, pts3d):
    """
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3) reconstructed point cloud

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pts3d_centered: (N, 3) centered point cloud
    """

    pose_avg = average_poses(poses, pts3d)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    pose_avg_inv = np.linalg.inv(pose_avg_homo)
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = pose_avg_inv @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    pts3d_centered = pts3d @ pose_avg_inv[:, :3].T + pose_avg_inv[:, 3:].T

    return poses_centered, pts3d_centered

class NerfstudioDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, only_pose=False, **kwargs):
        super().__init__(root_dir, split, downsample)

        meta_path = root_dir
        with open(meta_path, 'r') as f_meta:
            meta = json.load(f_meta)

        image_filenames = []
        poses = []

        fx = meta["fl_x"]
        fy = meta["fl_y"]
        cx = meta["cx"]
        cy = meta["cy"]

        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])

        w, h = meta["w"], meta["h"]
        self.img_wh = (w, h)
        self.directions, self.grid = get_ray_directions(h, w, self.K, anti_aliasing_factor=kwargs.get('anti_aliasing_factor', 1.0), return_uv=True)

        for frame_meta in meta['frames']:
            image_filenames.append(frame_meta["file_path"])
            poses.append(np.array(frame_meta["transform_matrix"]).reshape(4, 4))
        poses = np.stack(poses)

        train_filenames = meta["train_filenames"]
        val_filenames = meta["val_filenames"]

        train_ids = [image_filenames.index(name) for name in train_filenames]
        val_ids = [image_filenames.index(name) for name in val_filenames]

        # pose transform
        poses[:, 2, :] *= -1
        poses = poses[:, np.array([1, 0, 2, 3]), :]
        poses[:, 0:3, 1:3] *= -1

        poses, scale_factor, transform = auto_normalize_poses(poses, scale_factor = 0.8)

        self.pos_trans = {
            'scale_factor': scale_factor,
            'transform': transform,
        }

        if split == 'train':
            poses = poses[np.array(train_ids).reshape(-1)]
            img_paths = [image_filenames[id] for id in train_ids]
        else:
            poses = poses[np.array(val_ids).reshape(-1)]
            img_paths = [image_filenames[id] for id in val_ids]

        self.poses = torch.from_numpy(poses[:, :3, :4]).float()

        print(f'Loading {len(img_paths)} {split} images ...')
        self.rays = []
        for img_path in tqdm(img_paths):
            buf = []  # buffer for ray attributes: rgb, etc

            img = read_image(img_path, self.img_wh)
            buf += [torch.FloatTensor(img)]

            self.rays += [torch.cat(buf, 1)]

        self.rays = torch.stack(self.rays)  # (N_images, hw, ?)

        # generate MVP
        self.near = n = 0.000001
        self.far = f = 30  # infinite
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        print("fx, fy, cx, cy: ", fx, fy, cx, cy)
        width = self.img_wh[0]
        height = self.img_wh[1]
        n00 = 2.0 * fx / width
        n11 = 2.0 * fy / height
        n02 = 1.0 - 2.0 * cx / width
        n12 = 2.0 * cy / height - 1.0
        n32 = -1.0
        n22 = (f + n) / (n - f)
        n23 = (2 * f * n) / (n - f)
        camera_projmat = np.array([[n00, 0, n02, 0],
                                   [0, n11, n12, 0],
                                   [0, 0, n22, n23],
                                   [0, 0, n32, 0]], dtype=np.float32)

        self.projection = torch.from_numpy(camera_projmat)
        bottom = torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).expand((self.poses.shape[0], -1, -1))
        i_pose = self.poses.clone()
        i_pose[:, :3, 1:3] = -i_pose[:, :3, 1:3]
        square_pose = torch.cat((i_pose, bottom), dim=1)
        self.mvps = self.projection.unsqueeze(0) @ torch.inverse(square_pose)
        self.H = self.img_wh[1]
        self.W = self.img_wh[0]


    def mvp_permute(self, index):
        # print("rp")
        pose = self.poses[index].clone().unsqueeze(0)

        n = self.near
        f = self.far
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        width = self.img_wh[0]
        height = self.img_wh[1]
        n00 = 2.0 * fx / width
        n11 = 2.0 * fy / height

        rpx = np.random.random() - 0.5
        rpy = np.random.random() - 0.5
        cx = cx + rpx
        cy = cy + rpy
        n02 = 1.0 - 2.0 * cx / width
        n12 = 2.0 * cy / height - 1.0
        n32 = -1.0
        n22 = (f + n) / (n - f)
        n23 = (2 * f * n) / (n - f)
        camera_projmat = np.array([[n00, 0, n02, 0],
                                   [0, n11, n12, 0],
                                   [0, 0, n22, n23],
                                   [0, 0, n32, 0]], dtype=np.float32)

        projection = torch.from_numpy(camera_projmat)

        bottom = torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).expand((pose.shape[0], -1, -1))

        pose[:, :3, 1:3] = -pose[:, :3, 1:3]
        square_pose = torch.cat((pose, bottom), dim=1)

        mvp = projection.unsqueeze(0) @ torch.inverse(square_pose)

        u, v = self.grid.unbind(-1)
        directions =  torch.stack([(u - cx + 0.5) / fx, (v - cy + 0.5) / fy, torch.ones_like(u)], -1)
        directions = directions.reshape(-1, 3)

        return mvp.reshape(4, 4), directions
