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
import torchvision

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


class ColmapDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics(**kwargs)
        
        # load camera poses and images
        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self, **kwargs):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        _h = int(camdata[1].height * self.downsample)
        _w = int(camdata[1].width * self.downsample)
        h = _h if _h % 8 == 0 else (_h // 8 + 1) * 8
        w = _w if _w % 8 == 0 else (_w // 8 + 1) * 8
        downsample = h / int(camdata[1].height)
        self.img_wh = (w, h)

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0] * downsample
            cx = camdata[1].params[1] * downsample
            cy = camdata[1].params[2] * downsample
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0] * downsample
            fy = camdata[1].params[1] * downsample
            cx = camdata[1].params[2] * downsample
            cy = camdata[1].params[3] * downsample
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])
        # self.directions = get_ray_directions(h, w, self.K, anti_aliasing_factor=kwargs.get('anti_aliasing_factor', 1.0))
        self.directions, self.grid = get_ray_directions(h, w, self.K, anti_aliasing_factor=kwargs.get('anti_aliasing_factor', 1.0), return_uv=True)

    def read_meta(self, split, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))

        img_names = [imdata[k].name for k in imdata]

        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])

        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)

        poses = np.linalg.inv(w2c_mats)[:, :3]  # (N_images, 3, 4) cam2world matrices
        self.poses = poses
        
        self.up = torch.FloatTensor(-normalize(self.poses[:, :3, 1].mean(0)))

        self.poses = np.concatenate(
            (self.poses, np.array([0, 0, 0, 1]).reshape(1, 1, 4).repeat(self.poses.shape[0], 0)), 1)

        # camera pose normalization
        if os.path.exists(os.path.join(kwargs.get('workspace', './'), 'pos_trans.pkl')):
            with open(os.path.join(kwargs.get('workspace', './'), 'pos_trans.pkl'), 'rb') as f_pkl:
                pos_trans = pickle.load(f_pkl)
            scale_factor = pos_trans['scale_factor']
            transform = pos_trans['transform']
            self.poses = transform @ self.poses
            self.poses[:, :3, 3] *= scale_factor
            self.poses = self.poses[:, :3, :4]
        else:
            self.poses, scale_factor, transform = auto_normalize_poses(self.poses, scale_factor=kwargs.get('scale_factor', 0.1))
            self.poses = self.poses[:, :3, :4]

        self.pos_trans = {
            'scale_factor': scale_factor,
            'transform': transform,
        }

        # directories for rgb, instance (semantics) and depth prior
        folder = 'images'
        if kwargs.get('use_sem', False):
            semantics = 'instance'
        if kwargs.get('depth_mono', False):
            depths_dir = 'depth'

        # read successfully reconstructed images and ignore others
        img_paths = [os.path.join(self.root_dir, folder, name)
                     for name in img_names]
        
        # read instance (semantics) 
        # values are in 0 to N - 1
        if kwargs.get('use_sem', False):
            sem_paths = []
            for name in img_names:
                sem_file_name = os.path.splitext(name)[0] + ".npy"
                sem_paths.append(os.path.join(self.root_dir, semantics, sem_file_name))
        
        # read depth prior
        if kwargs.get('depth_mono', False):
            depth_paths = []
            for name in img_names:
                depth_file_name = os.path.splitext(os.path.basename(name))[0] + ".npy"
                depth_paths.append(os.path.join(self.root_dir, depths_dir, depth_file_name))
        
        self.rays = []
        if kwargs.get('use_sem', False):
            self.labels = []
        if kwargs.get('depth_mono', False):
            self.depths = []

        # dataset split
        test_ids = np.array([i for i in range(len(img_paths)) if i % 8 == 0]).reshape(-1)
        all_ids = np.arange(len(img_paths))
        if kwargs['strict_split']:
            train_ids = np.setdiff1d(all_ids, test_ids)
        else:
            train_ids = all_ids
        train_ids = train_ids.tolist()
        test_ids = test_ids.tolist()

        if split == 'train':
            img_paths = [x for i, x in enumerate(img_paths) if i in train_ids]
            img_names = [x for i, x in enumerate(img_names) if i in train_ids]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i in train_ids] )
            self.img_paths = img_paths
            if kwargs.get('use_sem', False):
                sem_paths = [x for i, x in enumerate(sem_paths) if i in train_ids]
            if kwargs.get('depth_mono', False):
                depth_paths = [x for i, x in enumerate(depth_paths) if i in train_ids]

        elif split == 'test':

            img_paths = [x for i, x in enumerate(img_paths) if i in test_ids]
            img_names = [x for i, x in enumerate(img_names) if i in test_ids]

            if kwargs.get('use_sem', False):
                sem_paths = [x for i, x in enumerate(sem_paths) if i in test_ids]
            if kwargs.get('depth_mono', False):
                depth_paths = [x for i, x in enumerate(depth_paths) if i in test_ids]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i in test_ids])

        self.poses = torch.FloatTensor(self.poses)  # (N_images, 3, 4)

        # load images
        print(f'Loading {len(img_paths)} {split} images from {folder}...')
        for img_path in tqdm(img_paths):
            buf = []  # buffer for ray attributes: rgb, etc

            img = read_image(img_path, self.img_wh)
            buf += [torch.FloatTensor(img)]

            self.rays += [torch.cat(buf, 1)]

        self.rays = torch.stack(self.rays)  # (N_images, hw, ?)

        # loading normal, depth priors and semantics when training 
        if split != 'test':
            if kwargs.get('normal_mono', False):
                dir_normal = os.path.join(self.root_dir, 'normals')

                normal = self.read_normal_byname(dir_normal, img_names)
                self.normals = torch.FloatTensor(normal)
                print("[info] colmap dataset: normal prior loaded")

            if kwargs.get('depth_mono', False):
                for i, depth_path in enumerate(depth_paths):
                    depth = np.load(depth_path)
                    if len(depth.shape) == 2:
                        depth = depth[..., None]
                    depth = torch.from_numpy(depth)
                    depth = depth.permute(2, 0, 1)
                    depth = torchvision.transforms.functional.resize(depth, (self.img_wh[1], self.img_wh[0]))
                    depth = depth.permute(1, 2, 0)
                    depth = depth.numpy()
                    depth = depth.reshape(-1)
                    self.depths.append(depth)
                self.depths = torch.FloatTensor(np.stack(self.depths))
                print("[info] colmap dataset: depth prior loaded")

        if kwargs.get('use_sem', False):
            def read_semantic(sem_path):
                label = np.load(sem_path).astype(np.int32)
                if len(label.shape) == 2:
                    label = label[..., None]
                label = torch.from_numpy(label)
                label = label.permute(2, 0, 1)
                label = torchvision.transforms.functional.resize(label, (self.img_wh[1], self.img_wh[0]), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
                label = label.permute(1, 2, 0)
                label = label.numpy()
                label = label.reshape(-1)
                # import ipdb; ipdb.set_trace()
                return label

            for i, sem_path in enumerate(sem_paths):
                label = read_semantic(sem_path=sem_path)
                self.labels += [label]

            self.labels = torch.LongTensor(np.stack(self.labels))
            print("[info] colmap dataset: label prior loaded")

        # mvps
        # perspective projection matrix
        self.near = n = 0.001
        self.far = f = 30  # infinite
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
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

    def read_normal(self, dir_normal, frame_id):
        poses = self.poses
        normal_list = []
        for c2w, i in zip(poses, frame_id):
            path = os.path.join(dir_normal, 'frame_{:0>5d}.jpg'.format(i))
            img = Image.open(path)
            img = np.array(img) / 255.0
            normal = ((img - 0.5) * 2).reshape(-1, 3)
            c2w_np = np.array(c2w)
            normal = normal @ c2w_np[:, :3].T
            normal_list.append(normal)
        normal_list = np.stack(normal_list)
        return normal_list

    def read_normal_byname(self, dir_normal, names):
        poses = self.poses
        normal_list = []
        for c2w, name in zip(poses, names):
            path = os.path.join(dir_normal, f'{name}')
            img = Image.open(path)
            img = np.array(img) / 255.0
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
            img = torchvision.transforms.functional.resize(img, (self.img_wh[1], self.img_wh[0]))
            img = img.permute(1, 2, 0)
            img = img.numpy()
            normal = ((img - 0.5) * 2).reshape(-1, 3)
            c2w_np = np.array(c2w)
            normal = normal @ c2w_np[:, :3].T
            normal_list.append(normal)
        normal_list = np.stack(normal_list)
        return normal_list

