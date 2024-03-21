import torch
import numpy as np
import os
import glob
from tqdm import tqdm
from PIL import Image
from kornia import create_meshgrid
import json
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
from torchvision.transforms.functional import resize
import cv2

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
class VRNeRFDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, only_pose=False, **kwargs):
        super().__init__(root_dir, split, downsample)
        cam_info_path = os.path.join(root_dir, "intrinsic_perspective.pkl")
        with open(cam_info_path, 'rb') as f_pkl:
            cam_info = pickle.load(f_pkl)
        self.downsample = downsample
        split_path = os.path.join(root_dir, "splits.json")
        with open(split_path, 'r') as f_pkl:
            splits_info = json.load(f_pkl)

        img_names = []
        intrinsics = []
        c2ws = []
        train_ids = []
        test_ids = []
        for fi, cam_info_frame in enumerate(cam_info["KRT"]):
            original_h = h = cam_info_frame["height"]
            original_w = w = cam_info_frame["width"]
            K = cam_info_frame["K"]
            if downsample < 1:
                h = int(h*downsample)
                w = int(w*downsample)
                h = h if h % 8 == 0 else (h // 8 + 1) * 8
                w = w if w % 8 == 0 else (w // 8 + 1) * 8
                K[:2] *= downsample
            intrinsics.append(K)
            img_name = cam_info_frame["img_path"]
            if img_name[:-4] in splits_info["train"]:
                train_ids.append(fi)
            else:
                test_ids.append(fi)
            img_names.append(img_name)
            c2ws.append(np.linalg.inv(cam_info_frame["T"]))

        print("total imgs: ", len(cam_info["KRT"]))
        print("train imgs: ", len(train_ids))
        print("test imgs: ", len(test_ids))
        # assume cv convention
        c2ws = np.stack(c2ws)
        c2ws, scale_factor, transform = auto_normalize_poses(c2ws,
                     scale_factor=kwargs.get('scale_factor', 1.0))
        c2ws = c2ws[:, :3, :4]

        if split != 'test':
            os.makedirs(os.path.join(kwargs.get('workspace', './')), exist_ok=True)
            plt.scatter(c2ws[:, 0, 3], c2ws[:, 1, 3], s=0.5)
            plt.savefig(os.path.join(kwargs.get('workspace', './'), 'loop_xy_t.png'))
            plt.clf()
            plt.scatter(c2ws[:, 0, 3], c2ws[:, 2, 3], s=0.5)
            plt.savefig(os.path.join(kwargs.get('workspace', './'), 'loop_xz_t.png'))
            plt.clf()
            plt.scatter(c2ws[:, 1, 3], c2ws[:, 2, 3], s=0.5)
            plt.savefig(os.path.join(kwargs.get('workspace', './'), 'loop_yz_t.png'))
            plt.clf()

        self.pos_trans = {
            'scale_factor': scale_factor,
            'transform': transform,
        }

        if split == "train":
            if kwargs["strict_split"]:
                img_ids = train_ids
            else:
                img_ids = list(range(len(cam_info["KRT"])))
        else:
            img_ids = test_ids

        self.img_wh = (w, h)
        grid = create_meshgrid(h, w, False, device='cpu')[0]  # (H, W, 2)
        u, v = grid.unbind(-1)

        img_names = [x for i, x in enumerate(img_names) if i in img_ids]
        intrinsics = np.array([x for i, x in enumerate(intrinsics) if i in img_ids])
        c2ws = np.array([x for i, x in enumerate(c2ws) if i in img_ids])

        rays_directions = []
        rays = []
        self.near = n = 0.001
        self.far = f = 15  # infinite
        self.H = self.img_wh[1]
        self.W = self.img_wh[0]

        self.grid = grid

        mvps = []
        if split == "train" and kwargs.get('depth_mono', False):
            self.depths = []
        if split == "train" and kwargs.get('normal_mono', False):
            self.normals = []
        self.K = []
        for id in tqdm(range(len(img_ids))):
            K = intrinsics[id]
            self.K.append(torch.from_numpy(K))
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            c2w = c2ws[id]

            if kwargs.get('load_imgs', True):
                directions = torch.stack([(u-cx+0.5)/fx, (v-cy+0.5)/fy, torch.ones_like(u)], -1).reshape(-1, 3)
                rays_d = directions @ c2w[:3, :3].T
                rays_directions.append(rays_d)
                img_path = os.path.join(root_dir, img_names[id])
                img = read_image(img_path, self.img_wh)
                buf = []
                buf += [torch.FloatTensor(img)]
                rays += [torch.cat(buf, 1)]
                if split == "train" and kwargs.get('depth_mono', False):
                    depth_path = os.path.join(root_dir, 'depth', img_names[id]+".npy")
                    depth = np.load(depth_path).reshape(-1)
                    if self.downsample < 1:
                        depth = torch.from_numpy(depth)
                        depth = depth.reshape(original_h, original_w, 1).permute(2, 0, 1)
                        depth = resize(depth, (h, w))
                        depth = depth.reshape(-1).numpy()
                    self.depths.append(depth)
                if split == "train" and kwargs.get('normal_mono', False):
                    normal_path = os.path.join(root_dir, 'normal', img_names[id])
                    img = Image.open(normal_path)
                    img = np.array(img, dtype='uint8')
                    if self.downsample < 1:
                        img = img.reshape(original_h, original_w, -1)
                        img = cv2.resize(img, (w, h))
                        img = img.reshape(-1, 3)
                    img = img.astype(np.float32)
                    img = img / 255.0
                    normal = ((img - 0.5) * 2).reshape(-1, 3)
                    normal = normal / np.linalg.norm(normal, ord=2, axis=-1, keepdims=True)
                    c2w_np = np.array(c2w[:3,:3])
                    normal = normal @ c2w_np.T
                    self.normals.append(normal)

            n00 = 2.0 * fx / w
            n11 = 2.0 * fy / h
            n02 = 1.0 - 2.0 * cx / w
            n12 = 2.0 * cy / h - 1.0
            n32 = -1.0
            n22 = (f + n) / (n - f)
            n23 = (2 * f * n) / (n - f)

            camera_projmat = np.array([[n00, 0, n02, 0],
                                       [0, n11, n12, 0],
                                       [0, 0, n22, n23],
                                       [0, 0, n32, 0]], dtype=np.float32)
            projection = torch.from_numpy(camera_projmat).reshape(4, 4)
            bottom = torch.tensor([0, 0, 0, 1]).reshape(1, 4)
            gl_c2w = torch.from_numpy(c2w.copy())[:3, :4]
            gl_c2w[:3, 1:3] *= -1
            square_pose = torch.cat((gl_c2w, bottom), dim=0)
            mvps.append(projection.float() @ torch.inverse(square_pose).float())
        self.mvps = torch.stack(mvps)
        self.K = torch.stack(self.K).reshape(-1, 3, 3)
        if split == "train" and kwargs.get('depth_mono', False):
            self.depths = torch.FloatTensor(np.stack(self.depths))
        if split == "train" and kwargs.get('normal_mono', False):
            self.normals = torch.FloatTensor(np.stack(self.normals))

        if kwargs.get('load_imgs', True):
            self.rays = torch.stack(rays)
            self.rays_d = torch.stack(rays_directions)
        self.poses = torch.from_numpy(np.stack(c2ws)[:, :3, :4]).float()
        print("finished init vr nerf dataset")


    def mvp_permute(self, index):
        # print("rp")
        pose = self.poses[index].clone().unsqueeze(0)

        n = self.near
        f = self.far
        fx, fy, cx, cy = self.K[index, 0, 0], self.K[index, 1, 1], self.K[index, 0, 2], self.K[index, 1, 2]
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

        mvp = projection.unsqueeze(0).float() @ torch.inverse(square_pose).float()

        u, v = self.grid.unbind(-1)
        directions =  torch.stack([(u - cx + 0.5) / fx, (v - cy + 0.5) / fy, torch.ones_like(u)], -1)
        directions = directions.reshape(-1, 3).float()

        return mvp.reshape(4, 4), directions
