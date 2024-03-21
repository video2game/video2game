import json
import os
import cv2,torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import skimage.measure
from models.networks import NGP
from opt import get_opts
from utils import load_ckpt
from tqdm import tqdm
import pickle5 as pickle
from models.contract import contract, contract_inv

import trimesh
from datasets import dataset_dict
import nvdiffrast.torch as dr
from meshutils import *
import pymesh
from pykdtree.kdtree import KDTree



def convert_samples_to_mesh_sphere(
    pytorch_3d_tensor,
    bbox,
    level=0.5,
    roi=None,
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    assert roi is not None
    numpy_3d_tensor = pytorch_3d_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / (np.array(pytorch_3d_tensor.shape) - 1))

    print("start marching cubes")

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_tensor, level=level, spacing=voxel_size
    )
    print("finish marching cubes")
    faces = faces[...,::-1] # inverse face orientation
    faces = faces.reshape(-1, 3)

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    if hparams.center_create_ring:
        keep_x = np.logical_and(mesh_points[:, 0] > hparams.extract_mesh_ring_scale_x_min,
                                mesh_points[:, 0] < hparams.extract_mesh_ring_scale_x_max)
        keep_y = np.logical_and(mesh_points[:, 1] > hparams.extract_mesh_ring_scale_y_min,
                                mesh_points[:, 1] < hparams.extract_mesh_ring_scale_y_max)
        keep_z = np.logical_and(mesh_points[:, 2] > hparams.extract_mesh_ring_scale_z_min,
                                mesh_points[:, 2] < hparams.extract_mesh_ring_scale_z_max)
        keep = np.logical_not(np.logical_and(np.logical_and(keep_x, keep_y), keep_z))

        def filter_mesh_from_vertices(keep, mesh_points, faces):
            filter_mapping = np.arange(keep.shape[0])[keep]
            filter_unmapping = -np.ones((keep.shape[0]))
            filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
            mesh_points = mesh_points[keep]
            keep_0 = keep[faces[:, 0]]
            keep_1 = keep[faces[:, 1]]
            keep_2 = keep[faces[:, 2]]
            keep_faces = np.logical_and(keep_0, keep_1)
            keep_faces = np.logical_and(keep_faces, keep_2)
            faces = faces[keep_faces]
            faces[:, 0] = filter_unmapping[faces[:, 0]]
            faces[:, 1] = filter_unmapping[faces[:, 1]]
            faces[:, 2] = filter_unmapping[faces[:, 2]]
            return mesh_points, faces

        mesh_points, faces = filter_mesh_from_vertices(keep, mesh_points, faces)


    mesh_points = torch.tensor(mesh_points.reshape(-1, 3))
    # mesh_points = np.array(mesh_points).reshape(-1, 3)
    mesh_points = np.array(contract_inv(mesh_points, roi=roi.to(mesh_points.device), type=hparams.contraction_type)).reshape(-1, 3)

    num_verts = mesh_points.shape[0]
    num_faces = faces.shape[0]

    print("num_verts: ", num_verts)
    print("num_faces: ", num_faces)

    def filter_mesh_from_vertices(keep, mesh_points, faces):
        filter_mapping = np.arange(keep.shape[0])[keep]
        filter_unmapping = -np.ones((keep.shape[0]))
        filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
        mesh_points = mesh_points[keep]
        keep_0 = keep[faces[:, 0]]
        keep_1 = keep[faces[:, 1]]
        keep_2 = keep[faces[:, 2]]
        keep_faces = np.logical_and(keep_0, keep_1)
        keep_faces = np.logical_and(keep_faces, keep_2)
        faces = faces[keep_faces]
        faces[:, 0] = filter_unmapping[faces[:, 0]]
        faces[:, 1] = filter_unmapping[faces[:, 1]]
        faces[:, 2] = filter_unmapping[faces[:, 2]]
        return mesh_points, faces

    keep_x = np.logical_and(mesh_points[:, 0] < 8, mesh_points[:, 0] > -8)
    keep_y = np.logical_and(mesh_points[:, 1] < 8, mesh_points[:, 1] > -8)
    keep = np.logical_and(keep_x, keep_y)
    mesh_points, faces = filter_mesh_from_vertices(keep, mesh_points, faces)
    # print(f"mesh_points: zmax: {mesh_points[:, 2].max()}; zmin: {mesh_points[:, 2].min()}")

    return mesh_points, faces

@torch.no_grad()
def mark_unseen_triangles(glctx, vertices, triangles, mvps, H, W):
    # vertices: coords in world system
    # mvps: [B, 4, 4]

    if isinstance(vertices, np.ndarray):
        vertices = torch.from_numpy(vertices).contiguous().float().cuda()
    
    if isinstance(triangles, np.ndarray):
        triangles = torch.from_numpy(triangles).contiguous().int().cuda()

    mask = torch.zeros_like(triangles[:, 0]) # [M,], for face.

    for mvp in tqdm(mvps):

        vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp.cuda(), 0, 1)).float().unsqueeze(0) # [1, N, 4]

        # ENHANCE: lower resolution since we don't need that high?
        rast_p, _ = dr.rasterize(glctx, vertices_clip, triangles, (H, W)) # [1, H, W, 4]

        # collect the triangle_id (it is offseted by 1)
        rast_rs = rast_p[..., -1].long().view(-1)
        rast_rs = rast_rs[rast_rs>0]
        trig_id = rast_rs - 1
        trig_id = torch.unique(trig_id)
        # print("trig_id max: ", trig_id.max())
        # no need to accumulate, just a 0/1 mask.
        mask[trig_id] += 1 # wrong for duplicated indices, but faster.
        # mask.index_put_((trig_id,), torch.ones(trig_id.shape[0], device=device, dtype=mask.dtype), accumulate=True)

    mask = (mask == 0) # unseen faces by all cameras

    print(f'[mark unseen trigs] {mask.sum()} from {mask.shape[0]}')
    
    return mask # [N]

def fix_mesh(vertices, faces, target_len = 0.002):
    mesh = pymesh.form_mesh(vertices, faces)

    print("Target resolution: {} mm".format(target_len))

    count = 0
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
    num_vertices = mesh.num_vertices
    mesh, __ = pymesh.split_long_edges(mesh, target_len)
    print("#v after split_long_edges: {}".format(mesh.num_vertices))

    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len )
        print("#v after collapse_short_edges: {}".format(mesh.num_vertices))
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
        print("#v after remove_obtuse_triangles: {}".format(mesh.num_vertices))

        if mesh.num_vertices == num_vertices:
            break
        num_vertices = mesh.num_vertices
        count += 1
        if count >= 10: break

    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 1)
    vertices, faces = mesh.vertices, mesh.faces
    return vertices, faces


# get parameters
hparams = get_opts()

# load nerf model
os.makedirs(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}'), exist_ok=True)
rgb_act = 'Sigmoid'
ngp_params = {
    'L': hparams.ngp_L,
    'F': hparams.ngp_F,
    'log2_T': hparams.ngp_log2_T,
    'L_': hparams.ngp_L_,
    'F_': hparams.ngp_F_,
    'log2_T_': hparams.ngp_log2_T_,
    'base_res': hparams.ngp_base_res,
    'base_growth': hparams.ngp_base_growth,
    'growth_factor': hparams.ngp_growth_factor,
}
mlp_params = {
    'sigma_neurons': hparams.mlp_sigma_neurons,
    'rgb_neurons': hparams.mlp_rgb_neurons,
    'norm_neurons': hparams.mlp_norm_neurons,
    'sem_neurons': hparams.mlp_sem_neurons,
}

model = NGP(scale=hparams.scale, rgb_act=rgb_act, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len, classes=hparams.num_classes, contraction_type=hparams.contraction_type, ngp_params=ngp_params, mlp_params=mlp_params,sphere_scale=((hparams.sphere_scale_x_n, hparams.sphere_scale_y_n, hparams.sphere_scale_z_n), (hparams.sphere_scale_x_p, hparams.sphere_scale_y_p, hparams.sphere_scale_z_p)), grid_size=hparams.ngp_gridsize).cuda()

ckpt_path = hparams.ckpt_load

print(f'ckpt specified: {ckpt_path} !')
load_ckpt(model, ckpt_path, prefixes_to_ignore=['embedding_a', 'msk_model', 'density_grid', 'grid_coords'])

# specify range
x_min, x_max = hparams.extract_mesh_scale_x_min, hparams.extract_mesh_scale_x_max
y_min, y_max = hparams.extract_mesh_scale_y_min, hparams.extract_mesh_scale_y_max
z_min, z_max = hparams.extract_mesh_scale_z_min, hparams.extract_mesh_scale_z_max

# specify resolution
chunk_size = 128*128*128
print(f'bounding box: x: {x_min}, {x_max}; y: {y_min}, {y_max} z: {z_min}, {z_max}')

partition_x = hparams.extract_mesh_partition_x
partition_y = hparams.extract_mesh_partition_y
partition_z = hparams.extract_mesh_partition_z
delta_x = (x_max - x_min) / partition_x
delta_y = (y_max - y_min) / partition_y
delta_z = (z_max - z_min) / partition_z

# specify mesh
mesh_points = np.zeros((0, 3))
faces = np.zeros((0, 3), dtype=np.int32)



# voxel size
global_voxel_size = list(np.array([delta_x, delta_y, delta_z]).reshape(-1) / (np.array([hparams.extract_mesh_cropN, hparams.extract_mesh_cropN, hparams.extract_mesh_cropN]).reshape(-1) - 1))

# mesh post-process preparation
if hparams.mesh_post_process:
    
    # load camera poses
    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {'root_dir': hparams.root_dir,
                'downsample': hparams.downsample,
                'use_sem': False,
                'depth_mono': False,
                'normal_mono': False,
                'load_imgs': False,
                'scale_factor': hparams.scale_factor,
                'strict_split': False}
    
    kwargs['workspace'] = os.path.join('results', hparams.dataset_name, hparams.exp_name)

    if hparams.dataset_name == 'kitti':
        kwargs['seq_id'] = hparams.kitti_seq
        kwargs['kitti_start'] = hparams.kitti_start
        kwargs['kitti_end'] = hparams.kitti_end
        kwargs['train_frames'] = (hparams.kitti_end - hparams.kitti_start + 1)
            
        kwargs['kitti_dual_seq'] = hparams.kitti_dual_seq
        kwargs['kitti_dual_start'] = hparams.kitti_dual_start
        kwargs['kitti_dual_end'] = hparams.kitti_dual_end
        val_list = []
        for i in hparams.kitti_test_id:
            val_list.append(int(i))
        kwargs['test_id'] = val_list

    dataset = dataset(split='train', **kwargs)
    
    # mesh post-process parameter
    visibility_mask_dilation = hparams.mesh_postprocess_dilation

    # mesh seperation preparation
    if hparams.dataset_name == 'kitti' and hparams.extract_mesh_kitti_bbox_vehicle is not None:
        # kitti
        # load the vehicle bounding boxes locations (from kitti360 dataset annotation)
        with open(hparams.extract_mesh_kitti_bbox_vehicle, 'rb') as f_vehicle_pkl:
            vehicle_bboxes = pickle.load(f_vehicle_pkl)["bboxes"]
        # load camera poses normalization info
        with open(os.path.join(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}'),
                                'pos_trans.pkl'), 'rb') as f:
            pos_trans = pickle.load(f)
            center = pos_trans['center']
            scale = pos_trans['scale']
            forward = pos_trans['forward']
        forward = forward.reshape(-1, 3)
        center = center.reshape(-1, 3)

        vehicle_bboxes_roi = []
        vehicle_transforms = []
        vehicle_inside_idxs = {}
        for bbox_i in tqdm(range(len(vehicle_bboxes))):
            # transform vertices to local coordinates
            verts = vehicle_bboxes[bbox_i][0].reshape(-1, 3)
            verts = (verts - center) / scale - 0.5 * forward

            # filter: if the bounding box vertices locations are inside the central x-y range
            if np.all(np.abs(verts[..., 0:2]) <= 4):

                pts_xy = verts[:, :2]
                assert pts_xy.shape[0] == 8

                pts_xy_n = pts_xy.reshape(8, 1, 2)
                pts_xy_f = pts_xy.reshape(1, 8, 2)
                dist = np.sum((pts_xy_n - pts_xy_f) ** 2, axis=-1).reshape(8, 8)
                n_idx = np.argsort(dist, axis=-1)[:, 1].reshape(8)
                upper_idxs = []
                for i, idx in enumerate(n_idx):
                    up_idx = i if verts[i, 2] > verts[idx, 2] else idx
                    if up_idx not in upper_idxs:
                        upper_idxs.append(up_idx)

                assert len(upper_idxs) == 4
                pts_xy = verts[np.array(upper_idxs).reshape(-1).astype(np.int32), :2]
                near_idx = 1 if np.sum((pts_xy[0] - pts_xy[1]) ** 2) < np.sum(
                    (pts_xy[0] - pts_xy[2]) ** 2) else 2
                dx = pts_xy[0, 0] - pts_xy[near_idx, 0]
                dy = pts_xy[0, 1] - pts_xy[near_idx, 1]
                angle = np.arctan(dy / dx)

                c = np.cos(angle)
                s = np.sin(angle)
                R = np.float32([c, -s, s, c]).reshape(2, 2)
                rv = verts[:, :2] @ R
                x0, y0 = rv[:, 0].min(), rv[:, 1].min()
                x1, y1 = rv[:, 0].max(), rv[:, 1].max()
                rbox = np.array([
                    [x0, y0],
                    [x1, y1]
                ])

                z0, z1 = verts[:, 2].min(), verts[:, 2].max()
                vehicle_bboxes_roi.append(
                    (rbox, angle, (z0, z1))
                )
                vehicle_transforms.append(
                    (vehicle_bboxes[bbox_i][2], vehicle_bboxes[bbox_i][3])
                )
        print("vehicle_bboxes_roi: ", len(vehicle_bboxes_roi))
    elif hparams.extract_mesh_bbox_src is not None:
        with open(hparams.extract_mesh_bbox_src, 'r') as f_json:
            seperate_bbox_list = json.load(f_json)

    # nvdiffrast for post-processing
    glctx = dr.RasterizeCudaContext()

# mesh extraction: central part (default)
if hparams.center_extract_mesh:
    meshes = [] # store mesh for each partition
    boundary_list = [] # store mesh separation boundary voxel locations
    
    # iterate over each partition
    for i in (list(range(partition_x))):
        for j in (list(range(partition_y))):
            for k in (list(range(partition_z))):
                
                # center ring: to exclude a part inside the current extraction range
                if hparams.center_create_ring:
                    _x_min = x_min + i * delta_x
                    _y_min = y_min + j * delta_y
                    _z_min = z_min + k * delta_z

                    _x_max = _x_min + delta_x
                    _y_max = _y_min + delta_y
                    _z_max = _z_min + delta_z

                    keep_x = _x_min >= hparams.extract_mesh_ring_scale_x_min and _x_max <= hparams.extract_mesh_ring_scale_x_max
                    keep_y = _y_min >= hparams.extract_mesh_ring_scale_y_min and _y_max <= hparams.extract_mesh_ring_scale_y_max
                    keep_z = _z_min >= hparams.extract_mesh_ring_scale_z_min and _z_max <= hparams.extract_mesh_ring_scale_z_max

                    keep = keep_x and keep_y and keep_z

                    if keep:
                        print(f"ring: skip {i}, {j}, {k}; bounding box: x: {_x_min}, {_x_max}; y: {_y_min}, {_y_max} z: {_z_min}, {_z_max}")
                        continue
                
                # the partition range for marching cubes
                partition_offset = hparams.extract_mesh_partition_offset

                _x_min = x_min + i * delta_x - (min(partition_offset, delta_x) if i > 0 else 0)
                _y_min = y_min + j * delta_y - (min(partition_offset, delta_y) if j > 0 else 0)
                _z_min = z_min + k * delta_z - (min(partition_offset, delta_z) if k > 0 else 0)
                _x_max = _x_min + delta_x + (min(partition_offset, delta_x) if i > 0 else 0)
                _y_max = _y_min + delta_y + (min(partition_offset, delta_y) if j > 0 else 0)
                _z_max = _z_min + delta_z + (min(partition_offset, delta_z) if k > 0 else 0)

                print(f'{i}, {j}, {k}; bounding box: x: {_x_min}, {_x_max}; y: {_y_min}, {_y_max} z: {_z_min}, {_z_max}')
                xyz_min = torch.FloatTensor([[_x_min, _y_min, _z_min]])
                xyz_max = torch.FloatTensor([[_x_max, _y_max, _z_max]])

                dense_xyz = torch.stack(torch.meshgrid(
                        torch.linspace(_x_min, _x_max, hparams.extract_mesh_cropN),
                        torch.linspace(_y_min, _y_max, hparams.extract_mesh_cropN),
                        torch.linspace(_z_min, _z_max, hparams.extract_mesh_cropN),
                    ), -1)

                # a different density level choice, usually only set level is enough
                # use different density level for different location in the scene
                # center_level: the central density level
                # level: the marginal density level
                if hparams.center_level > 0:
                    if hparams.contraction_type == "Sphere":
                        dense_xyz_flatten = dense_xyz.reshape(-1, 3)
                        dense_xyz_flatten = (dense_xyz_flatten - 0.5) * 4
                        norm_sq = (dense_xyz_flatten * dense_xyz_flatten).sum(-1).reshape(-1, 1)
                        norm = torch.sqrt(norm_sq)
                    else:
                        assert  False

                # calculate density for each voxel
                samples = dense_xyz.reshape(-1, 3)

                density = []
                with torch.no_grad():
                    for _i in tqdm(range(0, samples.shape[0], chunk_size)):
                        samples_ = samples[_i:_i+chunk_size].cuda()
                        tmp = model.density(samples_, make_contract=False)
                        density.append(tmp.cpu())
                density = torch.stack(density, dim=0)

                # set marching cube level: if use dynamic marching cube level
                if hparams.center_level > 0:
                    density = density.reshape(-1)
                    norm = norm.reshape(-1)
                    decay = 1.5
                    coff_a = (hparams.center_level - hparams.level) / (1-np.exp(-2*np.sqrt(3)*decay))
                    coff_b = hparams.center_level - coff_a
                    density -= (coff_a * torch.exp(-norm*decay) + coff_b)
                    mc_level = 0
                else:
                    mc_level = hparams.level

                density = density.reshape((dense_xyz.shape[0], dense_xyz.shape[1], dense_xyz.shape[2]))
                roi = model.aabb
                
                # set density in mesh seperation
                # in kitti dataset, using annotation bbox
                if hparams.dataset_name == 'kitti' and hparams.extract_mesh_kitti_bbox_vehicle is not None:
                    dense_xyz_flatten = dense_xyz.reshape(-1, 3)
                    dense_xyz_flatten_invc = contract_inv(dense_xyz_flatten, model.aabb.to(dense_xyz_flatten.device), model.contraction_type)
                    in_bbox_grid = torch.zeros((dense_xyz_flatten.shape[0]))
                    density_flatten = density.reshape(-1)

                    offset = 0
                    for bbox_i, bbox in tqdm(enumerate(vehicle_bboxes_roi)):

                        rbox = bbox[0]
                        x0, y0, x1, y1 = rbox[0, 0], rbox[0, 1], rbox[1, 0], rbox[1, 1]
                        dx = (x1-x0)
                        dy = (y1-y0)

                        offset_x = (0.05 if dx < dy else 0.3) * dx
                        offset_y = (0.05 if dy < dx else 0.3) * dy
                        c = np.cos(bbox[1])
                        s = np.sin(bbox[1])
                        R = np.float32([c, -s, s, c]).reshape(2, 2)
                        z0, z1 = bbox[2]

                        R_dense_xyz_invc = dense_xyz_flatten_invc[:, 0:2] @ R
                        # x0, y0 = pts[:, 0].min(), pts[:, 1].min()
                        # x1, y1 = pts[:, 0].max(), pts[:, 1].max()

                        in_x = torch.logical_and(R_dense_xyz_invc[:, 0] > x0-offset_x, R_dense_xyz_invc[:, 0] < x1+offset_x)
                        in_y = torch.logical_and(R_dense_xyz_invc[:, 1] > y0-offset_y, R_dense_xyz_invc[:, 1] < y1+offset_y)

                        in_z = torch.logical_and(dense_xyz_flatten_invc[:, 2] > z0+0.008, dense_xyz_flatten_invc[:, 2] < z1+0.002)
                        in_xyz = torch.logical_and(torch.logical_and(in_x, in_y), in_z)
                        in_bbox_grid[in_xyz] = 1.

                        if torch.any(in_xyz) and torch.any(density_flatten[in_xyz] > mc_level):
                            vehicle_inside_idxs[bbox_i] = vehicle_inside_idxs.get(bbox_i, 0) + int(torch.count_nonzero(in_xyz))

                    in_bbox_grid = in_bbox_grid.reshape(dense_xyz.shape[0], dense_xyz.shape[1], dense_xyz.shape[2])
                    in_bbox_grid = in_bbox_grid > 0

                    if hparams.extract_mesh_generate_semantic_boundary:
                        density_over_mc_level = (density >= mc_level)
                        meet_semantic = in_bbox_grid if not hparams.extract_mesh_sem_filter_negative else torch.logical_not(in_bbox_grid)
                        density_over_mc_level_and_meet_semantic = torch.logical_and(density_over_mc_level, meet_semantic)
                        density_over_mc_level_and_not_meet_semantic = torch.logical_and(density_over_mc_level, torch.logical_not(meet_semantic))

                        density_below_mc_level = (density < mc_level)

                        boundary = torch.zeros((hparams.extract_mesh_cropN - 1, hparams.extract_mesh_cropN - 1, hparams.extract_mesh_cropN - 1)).float().to(density.device)

                        grid_property = torch.zeros((hparams.extract_mesh_cropN, hparams.extract_mesh_cropN, hparams.extract_mesh_cropN)).float().to(density.device)
                        grid_property[density_over_mc_level_and_meet_semantic] = 1.
                        grid_property[density_over_mc_level_and_not_meet_semantic] = 0.
                        grid_property[density_below_mc_level] = -100.
                        # print("grid: ", torch.count_nonzero(density_over_mc_level_and_meet_semantic), torch.count_nonzero(density_over_mc_level_and_not_meet_semantic), torch.count_nonzero(density_below_mc_level))
                        x_start = y_start = z_start = [0, 1]
                        gap = hparams.extract_mesh_cropN - 1
                        for _x_start in x_start:
                            for _y_start in y_start:
                                for _z_start in z_start:
                                    _x_end = _x_start + gap
                                    _y_end = _y_start + gap
                                    _z_end = _z_start + gap
                                    boundary += grid_property[_x_start:_x_end, _y_start:_y_end, _z_start:_z_end]
                        boundary = boundary / 8
                        boundary = torch.logical_and(boundary < 0.99, boundary > 0.01)

                        boundary = (dense_xyz[:-1, :-1, :-1][boundary]).reshape(-1, 3)
                        boundary_list.append(boundary)

                    if hparams.extract_mesh_sem_filter_negative:
                        density[in_bbox_grid] = 0.
                    else:
                        density[torch.logical_not(in_bbox_grid)] = 0.
                        
                # use manually set 3d bbox
                elif hparams.extract_mesh_bbox_src is not None:
                    dense_xyz_flatten = dense_xyz.reshape(-1, 3)
                    dense_xyz_flatten_invc = contract_inv(dense_xyz_flatten,
                                                            model.aabb.to(dense_xyz_flatten.device),
                                                            model.contraction_type)
                    in_bbox_grid = torch.zeros((dense_xyz_flatten.shape[0]))
                    # density_flatten = density.reshape(-1)

                    for bbox_i, bbox in tqdm(enumerate(seperate_bbox_list)):
                        bbox_T = bbox["T"] # central position of bounding box
                        bbox_HE = bbox["HE"] # half extent of bounding box

                        in_x = torch.logical_and(dense_xyz_flatten_invc[:, 0] > bbox_T[0] - bbox_HE[0],
                                                    dense_xyz_flatten_invc[:, 0] < bbox_T[0] + bbox_HE[0])
                        in_y = torch.logical_and(dense_xyz_flatten_invc[:, 1] > bbox_T[1] - bbox_HE[1],
                                                    dense_xyz_flatten_invc[:, 1] < bbox_T[1] + bbox_HE[1])
                        in_z = torch.logical_and(dense_xyz_flatten_invc[:, 2] > bbox_T[2] - bbox_HE[2],
                                                    dense_xyz_flatten_invc[:, 2] < bbox_T[2] + bbox_HE[2])
                        in_xyz = torch.logical_and(torch.logical_and(in_x, in_y), in_z)
                        in_bbox_grid[in_xyz] = 1.

                    in_bbox_grid = in_bbox_grid.reshape(dense_xyz.shape[0], dense_xyz.shape[1],
                                                        dense_xyz.shape[2])
                    in_bbox_grid = in_bbox_grid > 0

                    if hparams.extract_mesh_generate_semantic_boundary:
                        density_over_mc_level = (density >= mc_level)
                        meet_semantic = in_bbox_grid if not hparams.extract_mesh_sem_filter_negative else torch.logical_not(in_bbox_grid)
                        density_over_mc_level_and_meet_semantic = torch.logical_and(density_over_mc_level, meet_semantic)
                        density_over_mc_level_and_not_meet_semantic = torch.logical_and(density_over_mc_level, torch.logical_not(meet_semantic))

                        density_below_mc_level = (density < mc_level)

                        boundary = torch.zeros((hparams.extract_mesh_cropN - 1, hparams.extract_mesh_cropN - 1, hparams.extract_mesh_cropN - 1)).float().to(density.device)

                        grid_property = torch.zeros((hparams.extract_mesh_cropN, hparams.extract_mesh_cropN, hparams.extract_mesh_cropN)).float().to(density.device)
                        grid_property[density_over_mc_level_and_meet_semantic] = 1.
                        grid_property[density_over_mc_level_and_not_meet_semantic] = 0.
                        grid_property[density_below_mc_level] = -100.
                        # print("grid: ", torch.count_nonzero(density_over_mc_level_and_meet_semantic), torch.count_nonzero(density_over_mc_level_and_not_meet_semantic), torch.count_nonzero(density_below_mc_level))
                        x_start = y_start = z_start = [0, 1]
                        gap = hparams.extract_mesh_cropN - 1
                        for _x_start in x_start:
                            for _y_start in y_start:
                                for _z_start in z_start:
                                    _x_end = _x_start + gap
                                    _y_end = _y_start + gap
                                    _z_end = _z_start + gap
                                    boundary += grid_property[_x_start:_x_end, _y_start:_y_end, _z_start:_z_end]
                        boundary = boundary / 8
                        boundary = torch.logical_and(boundary < 0.99, boundary > 0.01)

                        boundary = (dense_xyz[:-1, :-1, :-1][boundary]).reshape(-1, 3)
                        boundary_list.append(boundary)

                    if hparams.extract_mesh_sem_filter_negative:
                        density[in_bbox_grid] = 0.
                    else:
                        density[torch.logical_not(in_bbox_grid)] = 0.

                # mesh separation using semantics
                if hparams.mesh_semantic_filter:
                    semid = hparams.mesh_semantic_filter_semid
                    assert semid >= 0 and semid < hparams.num_classes
                    semantics = []
                    with torch.no_grad():
                        for _i in tqdm(range(0, samples.shape[0], chunk_size)):
                            samples_ = samples[_i:_i+chunk_size].cuda()
                            tmp = model.semantic_pred(samples_, make_contract=False)
                            semantics.append(tmp.cpu())
                    semantics = torch.stack(semantics, dim=0)

                    semantics = semantics.reshape(dense_xyz.shape[0], dense_xyz.shape[1], dense_xyz.shape[2], hparams.num_classes).argmax(dim=-1)

                    if hparams.extract_mesh_generate_semantic_boundary:
                        density_over_mc_level = (density >= mc_level)
                        meet_semantic = (semantics != semid) if hparams.extract_mesh_sem_filter_negative else (semantics == semid)
                        density_over_mc_level_and_meet_semantic = torch.logical_and(density_over_mc_level, meet_semantic)
                        density_over_mc_level_and_not_meet_semantic = torch.logical_and(density_over_mc_level, torch.logical_not(meet_semantic))

                        density_below_mc_level = (density < mc_level)

                        boundary = torch.zeros((hparams.extract_mesh_cropN - 1, hparams.extract_mesh_cropN - 1, hparams.extract_mesh_cropN - 1)).float().to(density.device)

                        grid_property = torch.zeros((hparams.extract_mesh_cropN, hparams.extract_mesh_cropN, hparams.extract_mesh_cropN)).float().to(density.device)
                        grid_property[density_over_mc_level_and_meet_semantic] = 1.
                        grid_property[density_over_mc_level_and_not_meet_semantic] = 0.
                        grid_property[density_below_mc_level] = -100.

                        x_start = y_start = z_start = [0, 1]
                        gap = hparams.extract_mesh_cropN - 1
                        for _x_start in x_start:
                            for _y_start in y_start:
                                for _z_start in z_start:
                                    _x_end = _x_start + gap
                                    _y_end = _y_start + gap
                                    _z_end = _z_start + gap
                                    boundary += grid_property[_x_start:_x_end, _y_start:_y_end, _z_start:_z_end]
                        boundary = boundary / 8
                        boundary = torch.logical_and(boundary < 0.99, boundary > 0.01)

                        boundary = (dense_xyz[:-1, :-1, :-1][boundary]).reshape(-1, 3)
                        boundary_list.append(boundary)


                    if hparams.extract_mesh_sem_filter_negative:
                        density[semantics == semid] = 0.
                    else:
                        density[semantics!=semid] = 0.


                print("density: ", density.max(), density.min())

                bbox = torch.cat([xyz_min, xyz_max], dim=0)

                # import ipdb; ipdb.set_trace()

                if (density > mc_level).any():
                    rs = convert_samples_to_mesh_sphere(density.cpu(), bbox, level=mc_level, roi=roi)

                    _mesh_points, _faces = rs

                    if hparams.extract_mesh_partition_decimate < 1.0:
                        decimate_target = hparams.extract_mesh_partition_decimate * _faces.shape[0]
                        _mesh_points, _faces = decimate_mesh(_mesh_points, _faces, decimate_target, backend='pyfqmr', remesh=False, optimalplacement=True)

                    faces = np.concatenate((faces, _faces + mesh_points.shape[0]), axis=0)
                    mesh_points = np.concatenate((mesh_points, _mesh_points), axis=0)

                    if hparams.mesh_post_process and visibility_mask_dilation >= 0 and _mesh_points.shape[0] > 0 and _faces.shape[0] > 0 and faces.shape[0] > 5000000:
                        visibility_mask = mark_unseen_triangles(glctx, mesh_points, faces, dataset.mvps, dataset.H, dataset.W).cpu().numpy()
                        mesh_points, faces = remove_masked_trigs(mesh_points, faces, visibility_mask, dilation=visibility_mask_dilation)

num_verts = mesh_points.shape[0]
num_faces = faces.shape[0]
print("num_verts: ", num_verts)
print("num_faces: ", num_faces)

if hparams.center_extract_mesh and hparams.mesh_post_process:
    
    clean_min_f = hparams.mesh_postprocess_clean_f # mesh cleaning: minimum faces number
    clean_min_d = 0
    v_pct = hparams.mesh_postprocess_v_pct # merge edge length percetage

    if clean_min_f > 0 or clean_min_d > 0 or v_pct > 0:
        mesh_points, faces = clean_mesh(mesh_points, faces, v_pct=v_pct, min_f=clean_min_f, min_d=clean_min_d, repair=True, remesh=False)

    # remove vertices and faces if out of the large box
    if hparams.filter_box > 0:
        keep = np.logical_and(np.logical_and(np.abs(mesh_points[:, 0]) < hparams.filter_box, np.abs(mesh_points[:, 1]) < hparams.filter_box), np.abs(mesh_points[:, 2]) < hparams.filter_box)


        def filter_mesh_from_vertices(keep, mesh_points, faces):
            filter_mapping = np.arange(keep.shape[0])[keep]
            filter_unmapping = -np.ones((keep.shape[0]))
            filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
            mesh_points = mesh_points[keep]
            keep_0 = keep[faces[:, 0]]
            keep_1 = keep[faces[:, 1]]
            keep_2 = keep[faces[:, 2]]
            keep_faces = np.logical_and(keep_0, keep_1)
            keep_faces = np.logical_and(keep_faces, keep_2)
            faces = faces[keep_faces]
            faces[:, 0] = filter_unmapping[faces[:, 0]]
            faces[:, 1] = filter_unmapping[faces[:, 1]]
            faces[:, 2] = filter_unmapping[faces[:, 2]]
            return mesh_points, faces

        mesh_points, faces = filter_mesh_from_vertices(keep, mesh_points, faces)
    
    # remove vertices that cannot see from camera training views
    visibility_mask_dilation = hparams.mesh_postprocess_dilation
    if hparams.mesh_post_process and visibility_mask_dilation >= 0:
        visibility_mask = mark_unseen_triangles(glctx, mesh_points, faces, dataset.mvps, dataset.H, dataset.W).cpu().numpy()
        mesh_points, faces = remove_masked_trigs(mesh_points, faces, visibility_mask,
                                                    dilation=visibility_mask_dilation)

    # mesh decimation
    if hparams.center_decimate:
        if hparams.extract_mesh_post_contract:
            mesh_points = torch.from_numpy(mesh_points)
            mesh_points = np.array(contract(mesh_points, roi=roi.to(mesh_points.device), type=hparams.contraction_type)).reshape(-1, 3)
        decimate_target = hparams.center_decimate_target
        mesh_points, faces = decimate_mesh(mesh_points, faces, decimate_target, backend='pyfqmr', remesh=False, optimalplacement=True)
        if hparams.extract_mesh_post_contract:
            mesh_points = torch.from_numpy(mesh_points)
            mesh_points = np.array(contract_inv(mesh_points, roi=roi.to(mesh_points.device), type=hparams.contraction_type)).reshape(-1, 3)

# mesh vertices clip: prevent infinite points
mesh_points = np.clip(mesh_points, -32, 32)

# mesh post-processing using pymesh: for better texture mapping using xatlas
if hparams.pymesh_processing:
    if hparams.extract_mesh_post_contract:
        mesh_points = torch.from_numpy(mesh_points)
        mesh_points = np.array(contract(mesh_points, roi=roi.to(mesh_points.device), type=hparams.contraction_type)).reshape(-1, 3)
    mesh_points, faces = fix_mesh(mesh_points, faces, target_len=hparams.pymesh_processing_target_len)
    if hparams.extract_mesh_post_contract:
        mesh_points = torch.from_numpy(mesh_points)
        mesh_points = np.array(contract_inv(mesh_points, roi=roi.to(mesh_points.device), type=hparams.contraction_type)).reshape(-1, 3)

# for the final game in kitti loop
if hparams.dataset_name == 'kitti' and hparams.extract_mesh_kitti_bbox_vehicle is not None:

    car_transforms = []

    print("car collision models: ", len(list(vehicle_inside_idxs.keys())))

    with open('pickles/center_trans.pkl', 'rb') as f:
        pos_trans = pickle.load(f)
        center = pos_trans['center']
        scale = pos_trans['scale']
        forward = pos_trans['forward']
        print('center_trans: ', pos_trans)

    T1 = np.eye(4)
    T1[:3, 3] = -center.reshape(3)
    S0 = np.eye(4)
    S0[:3, :3] *= (1 / scale)
    T2 = np.eye(4)
    T2[:3, 3] = - 0.5 * forward.reshape(3)

    TST = np.matmul(T2, np.matmul(S0, T1))

    for idx, cnt in vehicle_inside_idxs.items():
        R, T = vehicle_transforms[idx]
        R0 = np.eye(4)
        R0[:3, :3] = R.reshape(3,3)
        T0 = np.eye(4)
        T0[:3, 3] = T.reshape(3)

        car_transforms.append({
            "M": np.matmul(TST, np.matmul(T0, R0)).tolist(),
            "cnt": cnt,
        })
    boundary_save_path = f'results/{hparams.dataset_name}/{hparams.exp_name}/{hparams.exp_name}_car_transform_{hparams.extract_mesh_boundary_label}.json'
    print("saving car bboxes transform to %s" % (boundary_save_path))

    with open(boundary_save_path, 'w') as f_json:
        json.dump(car_transforms, f_json)

# store mesh seperation boundary voxels for textured mesh
if hparams.center_extract_mesh and len(boundary_list) > 0:
    mesh_points_t = torch.from_numpy(mesh_points)
    mesh_points_contract = contract(mesh_points_t, model.aabb.to(mesh_points_t.device), hparams.contraction_type)
    # global_voxel_size
    aggr_boundary = torch.zeros((0, 3))
    for boundary in boundary_list:
        aggr_boundary = torch.cat((aggr_boundary, boundary.to(aggr_boundary.device)), 0)
    aggr_boundary = aggr_boundary.cpu().numpy()
    boundary_save_path = f'results/{hparams.dataset_name}/{hparams.exp_name}/{hparams.exp_name}_{hparams.extract_mesh_boundary_label}.pkl'
    print("saving boundary to %s" % (boundary_save_path))
    with open(boundary_save_path, 'wb') as f_pkl:
        pickle.dump({
            "boundary": aggr_boundary,
            "global_voxel_size": global_voxel_size,
        }, f_pkl)

    print("global_voxel_size: ", global_voxel_size)
    print("boundary: ", aggr_boundary.shape, aggr_boundary.max(), aggr_boundary.min())

    # label vertices that in those boundary voxels
    chunk_size = 1024
    mesh_points_contract = mesh_points_contract.reshape(-1, 3).cpu().numpy()
    voxel_length = (global_voxel_size[0]**2 + global_voxel_size[1]**2 + global_voxel_size[2]**2)**0.5
    vertex_colors = np.zeros_like(mesh_points_contract)
    print("mesh_points_contract: ", mesh_points_contract.shape, mesh_points_contract.max(), mesh_points_contract.min())
    for start_i in range(0, aggr_boundary.shape[0], chunk_size):
        end_i = min(start_i + chunk_size, aggr_boundary.shape[0])
        slice_length = end_i - start_i
        boundary_slice = aggr_boundary[start_i:end_i].reshape(-1, 3) + np.array(global_voxel_size).reshape(1, 3)

        tree = KDTree(boundary_slice)
        neighbor_dists, neighbor_indices = tree.query(mesh_points_contract, distance_upper_bound=1*voxel_length)
        # print("neighbor_indices: ", neighbor_indices.shape, neighbor_indices.max(), neighbor_indices.min())

        vertex_colors[neighbor_indices.reshape(-1) != slice_length] = 1.0

else:
    vertex_colors = None

# save extracted mesh
ply_filename_out = f'results/{hparams.dataset_name}/{hparams.exp_name}/{hparams.exp_name}_{hparams.mesh_output_label}.ply'

print("saving mesh to %s" % (ply_filename_out))
export_mesh = trimesh.Trimesh(vertices=mesh_points, faces=faces, vertex_colors=vertex_colors)
trimesh.exchange.export.export_mesh(export_mesh, ply_filename_out)
