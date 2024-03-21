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
import cv2
from scipy.spatial.transform import Rotation as R
from torchvision.transforms.functional import resize
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

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

class ScannetDSLRDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, only_pose=False, **kwargs):
        super().__init__(root_dir, split, downsample)
        # root_dir needs to be [prefix]/dslr and saved
        self.scannetpp_downsample_factor = downsample
        self.intrinsics_path = os.path.join(self.root_dir, 'intrinsic_perspective.pkl')
        self.poses_path = os.path.join(self.root_dir, 'colmap', 'images.txt')

        self.read_intrinsics(**kwargs)

        self.read_meta(split, **kwargs)

    def read_intrinsics(self, **kwargs):
        with open(self.intrinsics_path, 'rb') as f_pkl:
            meta = pickle.load(f_pkl)
            intrinsics = np.array(meta["intrinsics"]).reshape(3, 3)
            self.img_wh = meta["img_wh"]
        self.K = torch.FloatTensor(intrinsics).reshape(3, 3)
        w, h = self.img_wh
        if self.scannetpp_downsample_factor < 1:
            self.K[0, 0] *= self.scannetpp_downsample_factor
            self.K[1, 1] *= self.scannetpp_downsample_factor
            self.K[0, 2] *= self.scannetpp_downsample_factor
            self.K[1, 2] *= self.scannetpp_downsample_factor
            w, h = int(w * self.scannetpp_downsample_factor), int(h * self.scannetpp_downsample_factor)
        self.directions, self.grid = get_ray_directions(h, w, self.K,
                                                        anti_aliasing_factor=kwargs.get('anti_aliasing_factor', 1.0),
                                                        return_uv=True)

    def read_meta(self, split, **kwargs):
        rgb_dir = os.path.join(self.root_dir, "perspective")
        img_names = []

        # get pose, opencv convention
        pose_info_dict = {}
        with open(self.poses_path, 'r') as f_poses:
            for _ in range(4):
                f_poses.readline()
            while True:
                cam_info = f_poses.readline().strip(" \r\n")
                if len(cam_info) <= 1:
                    break
                cam_info = cam_info.split(' ')
                name = cam_info[-1]
                img_names.append(name)
                pose_info = cam_info[1:8]
                pose_info = [float(item) for item in pose_info]
                # print("pose_info: ", pose_info)
                # print("name: ", name)
                pose_info_dict[name] = pose_info

                f_poses.readline()
        img_names = sorted(img_names)
        c2ws = []
        img_paths = [os.path.join(rgb_dir, name) for name in img_names]

        for name in img_names:
            # c2w = np.array(self.pose_intrinsic_imu_info[name[:-4]]["aligned_pose"]).reshape(4, 4)
            # c2w[:3, 1:3] *= -1
            pose_info = pose_info_dict[name]
            qw, qx, qy, qz, tx, ty, tz = pose_info[0], pose_info[1], pose_info[2], pose_info[3], pose_info[4], pose_info[5], pose_info[6]
            qvec = np.array([qw, qx, qy, qz])
            tvec = np.array([tx, ty, tz])
            R = qvec2rotmat(qvec)
            t = tvec
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = t
            # c2w = w2c
            c2w = np.linalg.inv(w2c)
            # c2w[:3, 1:3] *= -1
            c2ws.append(c2w)
        self.poses = np.stack(c2ws)  # [:, :3]
        self.poses, scale_factor, transform = auto_normalize_poses(self.poses,
                                                                   scale_factor=kwargs.get('scale_factor', 0.8))

        if split != 'test':
            os.makedirs(os.path.join(kwargs.get('workspace', './')), exist_ok=True)
            plt.scatter(self.poses[:, 0, 3], self.poses[:, 1, 3], s=0.5)
            plt.savefig(os.path.join(kwargs.get('workspace', './'), 'loop_xy_t.png'))
            plt.clf()
            plt.scatter(self.poses[:, 0, 3], self.poses[:, 2, 3], s=0.5)
            plt.savefig(os.path.join(kwargs.get('workspace', './'), 'loop_xz_t.png'))
            plt.clf()
            plt.scatter(self.poses[:, 1, 3], self.poses[:, 2, 3], s=0.5)
            plt.savefig(os.path.join(kwargs.get('workspace', './'), 'loop_yz_t.png'))
            plt.clf()

        self.pos_trans = {
            'scale_factor': scale_factor,
            'transform': transform,
        }

        if kwargs.get('normal_mono', False):
            normal_paths = []
            for name in img_names:
                normal_paths.append(os.path.join(self.root_dir, 'normal', name))

        if kwargs.get('depth_mono', False):
            depth_paths = []
            for name in img_names:
                depth_file_name = name + ".npy"
                depth_paths.append(os.path.join(self.root_dir, 'depth', depth_file_name))

        # test_ids = np.array([i for i in range(len(img_paths)) if i % valhold == 0]).reshape(-1)
        self.split_json = os.path.join(self.root_dir, 'train_test_lists.json')
        with open(self.split_json, 'r') as f_spl:
            spl_dict = json.load(f_spl)
            train_ids = [img_names.index(name) for name in spl_dict["train"]]
            test_ids = [img_names.index(name) for name in spl_dict["test"]]
        print("train_ids: ", train_ids)
        print("test_ids: ", test_ids)
        all_ids = np.arange(len(img_paths))
        if not kwargs['strict_split']:
            train_ids = all_ids

        if kwargs.get('gen_json', False):
            poses = torch.from_numpy(self.poses.copy())
            self.K_00 = self.K
            self.width, self.height = self.img_wh

            def write_line(file, line):
                file.write(line + '\n')

            with open(os.path.join(kwargs['workspace'], 'transforms.json'), 'w') as json_file:
                write_line(json_file, '{')
                write_line(json_file, f'    "fl_x": {self.K_00[0, 0]},')
                write_line(json_file, f'    "fl_y": {self.K_00[1, 1]},')
                write_line(json_file, f'    "cx": {self.K_00[0, 2]},')
                write_line(json_file, f'    "cy": {self.K_00[1, 2]},')
                write_line(json_file, f'    "w": {self.width},')
                write_line(json_file, f'    "h": {self.height},')
                write_line(json_file, f'    "camara_model": "OPENCV",')

                write_line(json_file, '    "frames": [')
                for i in range(len(poses)):

                    pose = poses[i].clone()
                    pose[0:3, 1:3] *= -1
                    pose = torch.cat((pose, torch.tensor([0, 0, 0, 1]).reshape(1, 4)), dim=0)
                    pose = pose[np.array([1, 0, 2, 3]), :]
                    pose[2, :] *= -1

                    write_line(json_file, '        {')
                    write_line(json_file, f'            "file_path": "{img_paths[i]}",')

                    write_line(json_file, f'            "transform_matrix": [')
                    write_line(json_file, '                [')
                    write_line(json_file, f'                    {pose[0, 0]},')
                    write_line(json_file, f'                    {pose[0, 1]},')
                    write_line(json_file, f'                    {pose[0, 2]},')
                    write_line(json_file, f'                    {pose[0, 3]}')
                    write_line(json_file, '                ],')
                    write_line(json_file, '                [')
                    write_line(json_file, f'                    {pose[1, 0]},')
                    write_line(json_file, f'                    {pose[1, 1]},')
                    write_line(json_file, f'                    {pose[1, 2]},')
                    write_line(json_file, f'                    {pose[1, 3]}')
                    write_line(json_file, '                ],')
                    write_line(json_file, '                [')
                    write_line(json_file, f'                    {pose[2, 0]},')
                    write_line(json_file, f'                    {pose[2, 1]},')
                    write_line(json_file, f'                    {pose[2, 2]},')
                    write_line(json_file, f'                    {pose[2, 3]}')
                    write_line(json_file, '                ],')
                    write_line(json_file, '                [')
                    write_line(json_file, f'                    {pose[3, 0]},')
                    write_line(json_file, f'                    {pose[3, 1]},')
                    write_line(json_file, f'                    {pose[3, 2]},')
                    write_line(json_file, f'                    {pose[3, 3]}')
                    write_line(json_file, '                ]')
                    write_line(json_file, '            ]')
                    if i != len(poses) - 1:
                        write_line(json_file, '        },')
                    else:
                        write_line(json_file, '        }')
                write_line(json_file, '    ],')
                write_line(json_file, '    "train_filenames": [')
                assert kwargs['strict_split'] == True

                for _i, i in enumerate(train_ids):
                    if _i != len(train_ids) - 1:
                        write_line(json_file, f'        "{img_paths[i]}",')
                    else:
                        write_line(json_file, f'        "{img_paths[i]}"')
                write_line(json_file, '    ],')

                write_line(json_file, '    "val_filenames": [')
                for _i, i in enumerate(test_ids):
                    if _i != len(test_ids) - 1:
                        write_line(json_file, f'        "{img_paths[i]}",')
                    else:
                        write_line(json_file, f'        "{img_paths[i]}"')
                write_line(json_file, '    ],')

                write_line(json_file, '    "test_filenames": [')
                for _i, i in enumerate(test_ids):
                    if _i != len(test_ids) - 1:
                        write_line(json_file, f'        "{img_paths[i]}",')
                    else:
                        write_line(json_file, f'        "{img_paths[i]}"')
                write_line(json_file, '    ]')
                write_line(json_file, '}')

        if split == 'train':
            img_paths = [x for i, x in enumerate(img_paths) if i in train_ids]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i in train_ids])
            self.img_paths = img_paths
            if kwargs.get('use_sem', False):
                sem_paths = [x for i, x in enumerate(sem_paths) if i in train_ids]
            if kwargs.get('depth_mono', False):
                depth_paths = [x for i, x in enumerate(depth_paths) if i in train_ids]
            if kwargs.get('normal_mono', False):
                normal_paths = [x for i, x in enumerate(normal_paths) if i in train_ids]

        elif split == 'test':
            render_c2w_f64 = torch.FloatTensor(self.poses)

            img_paths = [x for i, x in enumerate(img_paths) if i in test_ids]
            if kwargs.get('use_sem', False):
                sem_paths = [x for i, x in enumerate(sem_paths) if i in test_ids]
            if kwargs.get('depth_mono', False):
                depth_paths = [x for i, x in enumerate(depth_paths) if i in test_ids]
            if kwargs.get('normal_mono', False):
                normal_paths = [x for i, x in enumerate(normal_paths) if i in test_ids]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i in test_ids])

        self.poses = torch.FloatTensor(self.poses)

        print(f'Loading {len(img_paths)} {split} images...')
        self.rays = []
        w, h = self.img_wh
        _w, _h = int(w * self.scannetpp_downsample_factor), int(h * self.scannetpp_downsample_factor)
        if kwargs.get('load_imgs', True):
            for img_path in tqdm(img_paths):
                buf = []  # buffer for ray attributes: rgb, etc

                img = read_image(img_path, self.img_wh)
                if self.scannetpp_downsample_factor < 1:
                    img = img.reshape(h, w, -1)
                    img = cv2.resize(img, (_w, _h))
                    img = img.reshape(-1, 3)
                buf += [torch.FloatTensor(img)]

                self.rays += [torch.cat(buf, 1)]

            self.rays = torch.stack(self.rays)  # (N_images, hw, ?)

        if split != 'test':
            if kwargs.get('normal_mono', False):
                normal_list = []
                for pi, path in enumerate(normal_paths):
                    c2w = self.poses[pi][:3, :3]
                    img = Image.open(path)
                    img = np.array(img, dtype='uint8')
                    if self.scannetpp_downsample_factor < 1:
                        img = img.reshape(h, w, -1)
                        img = cv2.resize(img, (_w, _h))
                        img = img.reshape(-1, 3)
                    img = img.astype(np.float32)
                    img = img / 255.0
                    normal = ((img - 0.5) * 2).reshape(-1, 3)
                    normal = normal / np.linalg.norm(normal, ord=2, axis=-1, keepdims=True)
                    c2w_np = np.array(c2w)
                    normal = normal @ c2w_np.T

                    normal_list.append(normal)
                normal = np.stack(normal_list)
                self.normals = torch.FloatTensor(normal)

            if kwargs.get('depth_mono', False):
                depths = []
                for i, depth_path in enumerate(depth_paths):
                    depth = np.load(depth_path).reshape(-1)
                    if self.scannetpp_downsample_factor < 1:
                        depth = torch.from_numpy(depth)
                        depth = depth.reshape(h, w, 1).permute(2, 0, 1)
                        depth = resize(depth, (_h, _w))
                        depth = depth.reshape(-1).numpy()
                    depths.append(depth)
                self.depths = torch.FloatTensor(np.stack(depths))
        if self.scannetpp_downsample_factor < 1:
            self.img_wh = (_w, _h)
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
        # y = self.img_wh[1] / (2.0 * self.K[1, 1])
        # aspect = self.img_wh[0] / self.img_wh[1]
        # self.projection = np.array([[1/(y*aspect), 0, 0, 0],
        #                             [0, -1/y, 0, 0],
        #                             [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
        #                             [0, 0, -1, 0]], dtype=np.float32)

        self.projection = torch.from_numpy(camera_projmat)
        print("self.poses: ", self.poses.shape)
        bottom = torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).expand((self.poses.shape[0], -1, -1))
        print("bottom: ", bottom.shape)
        try:
            i_pose = self.poses.clone()
        except:
            self.poses = torch.tensor(self.poses).float()
            i_pose = self.poses.clone()
        i_pose[:, :3, 1:3] = -i_pose[:, :3, 1:3]
        square_pose = torch.cat((i_pose, bottom), dim=1)
        print("square_pose: ", square_pose.shape)
        self.mvps = self.projection.unsqueeze(0) @ torch.inverse(square_pose)
        self.H = self.img_wh[1]
        self.W = self.img_wh[0]