import torch
from opt import get_opts
import os
import cv2
from models.global_var import global_var

from utils import load_ckpt
import trimesh

import warnings; warnings.filterwarnings("ignore")
import pickle5 as pickle
import plyfile

from models.tcngp import TCNGP

import nvdiffrast.torch as dr

import xatlas
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_dilation, binary_erosion

from meshutils import *
import json

from pykdtree.kdtree import KDTree

from models.contract import contract
def depth2img(depth, scale=16):
    depth = depth/scale
    depth = np.clip(depth, a_min=0., a_max=1.)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img

def mask2img(mask):
    mask_img = cv2.applyColorMap((mask*255).astype(np.uint8),
                                  cv2.COLORMAP_BONE)

    return mask_img

def semantic2img(sem_label, classes):
    level = 1/(classes-1)
    sem_color = level * sem_label
    sem_color = cv2.applyColorMap((sem_color*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return sem_color

def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]

def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]

def filter_mesh(keep, mesh_points, faces):
    filter_mapping = torch.arange(keep.shape[0]).cuda()[keep]
    filter_unmapping = -torch.ones((keep.shape[0])).cuda().long()
    filter_unmapping[filter_mapping] = torch.arange(filter_mapping.shape[0]).cuda()
    mesh_points = mesh_points[keep]
    keep_0 = keep[faces[:, 0]]
    keep_1 = keep[faces[:, 1]]
    keep_2 = keep[faces[:, 2]]
    keep_faces = torch.logical_and(keep_0, keep_1)
    keep_faces = torch.logical_and(keep_faces, keep_2)
    faces = faces[keep_faces]
    faces[:, 0] = filter_unmapping[faces[:, 0]]
    faces[:, 1] = filter_unmapping[faces[:, 1]]
    faces[:, 2] = filter_unmapping[faces[:, 2]]
    return mesh_points, faces



if __name__ == '__main__':
    torch.manual_seed(20220806)
    torch.cuda.manual_seed_all(20220806)
    np.random.seed(20220806)
    hparams = get_opts()
    global_var._init()

    # set up texture color NGP model
    tcngp_params = {
        'L': hparams.tcngp_L,
        'F': hparams.tcngp_F,
        'log2_T': hparams.tcngp_log2_T,
    }
    color_net_params = {
        'dim_hidden': hparams.tcngp_dim_hidden,
        'num_layers': hparams.tcngp_num_layers,
    }
    hparams.tcngp_contraction_type = hparams.contraction_type if hparams.tcngp_contraction_type is None else hparams.tcngp_contraction_type
    color_model = TCNGP(scale=hparams.tcngp_scale, contraction_type=hparams.tcngp_contraction_type,
                                sphere_scale=(
                                (hparams.sphere_scale_x_n, hparams.sphere_scale_y_n, hparams.sphere_scale_z_n),
                                (hparams.sphere_scale_x_p, hparams.sphere_scale_y_p, hparams.sphere_scale_z_p)),
                                ngp_params=tcngp_params, specular_dim_mlp=hparams.baking_specular_dim_mlp, per_level_scale=hparams.baking_per_level_scale,specular_dim=hparams.baking_specular_dim,color_net_params=color_net_params).cuda()

    if hparams.contraction_type == "AABB":
        ngp_aabb = torch.cat((-torch.ones(1, 3) * hparams.scale, torch.ones(1, 3) * hparams.scale), dim=0).reshape(2, 3)
    else:
        sphere_scale = ((hparams.sphere_scale_x_n, hparams.sphere_scale_y_n, hparams.sphere_scale_z_n),
                        (hparams.sphere_scale_x_p, hparams.sphere_scale_y_p, hparams.sphere_scale_z_p))
        ngp_aabb =  torch.cat((-torch.ones(1, 3) * torch.tensor(sphere_scale[0]).reshape(1, 3),  torch.ones(1, 3) * torch.tensor(sphere_scale[1]).reshape(1, 3)), dim=0).reshape(2, 3)

    # load baking ckpt
    baking_ckpt = os.path.join('ckpts', hparams.dataset_name, hparams.exp_name, 'baking', hparams.workspace, 'last.ckpt') if hparams.ckpt_load is None else hparams.ckpt_load
    
    load_ckpt(color_model, baking_ckpt, model_name='color_model', prefixes_to_ignore=["aabb", "aabb_train", "aabb_infer", "density_bitfield", "density_grid", "grid_coords", "xyz_encoder.params", "xyz_net.0.weight", "xyz_net.0.bias", "xyz_net.2.weight", "xyz_net.2.bias", "rgb_encoder.params", "dir_encoder.params", "rgb_net.params", "norm_pred_header.params", "semantic_header.params"])

    workspace = os.path.join('results', hparams.dataset_name, hparams.exp_name, 'baking', f'{"base" if hparams.workspace is None else f"{hparams.workspace}"}')

    assert os.path.exists(os.path.join(workspace, 'build_dict.pkl')), "you must assign a mesh dict to load"
    
    with open(os.path.join(workspace, 'build_dict.pkl'), 'rb') as f:
        build_dict = pickle.load(f)

    mesh_cascades = build_dict['cascades']
    v_cumsum = build_dict['v_cumsum']
    f_cumsum = build_dict['f_cumsum']
    vertices = build_dict['vertices']
    triangles = build_dict['triangles']
    ssaa = build_dict['ssaa']
    h0 = hparams.texture_size
    w0 = hparams.texture_size
    skydome = build_dict['skydome']

    vertices = torch.tensor(vertices).cuda()
    triangles = torch.tensor(triangles).cuda().int()

    glctx = dr.RasterizeCudaContext()
    
    assert hparams.baking_output is not None, "you must assign a export directory"
    export_path = os.path.join(workspace, hparams.baking_output)
    os.makedirs(export_path, exist_ok=True)
    
    
    @torch.no_grad()
    def _export_obj(v, f, h0, w0, export_path, ssaa=1, cas=0):

        v_np = v.cpu().numpy()  # [N, 3]
        f_np = f.cpu().numpy()  # [M, 3]

        # unwrap uvs
        print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')
        atlas = xatlas.Atlas()
        
        # contraction before unwarp: let central mesh faces have larger area in the final texture image
        if hparams.contraction_type == "Sphere":
            _v_np = torch.from_numpy(v_np)
            _v_np = contract(_v_np, color_model.aabb.to(_v_np.device), "Sphere").numpy()
        else:
            _v_np = v_np
        atlas.add_mesh(_v_np*10, f_np)
        
        # xatlas to unwarp
        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 0  # disable merge_chart for faster unwrap...
        pack_options = xatlas.PackOptions()
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
        _, ft_np, vt_np = atlas[0] # [N], [M, 3], [N, 2]
        print(f'[INFO] finished: xatlas unwraps UVs for mesh: v={v_np.shape} f={f_np.shape} vt={vt_np.shape} ft={ft_np.shape}')

        vt = torch.from_numpy(vt_np.astype(np.float32)).float().cuda()
        ft = torch.from_numpy(ft_np.astype(np.int64)).int().cuda()

        # padding
        uv = vt * 2.0 - 1.0 # uvs to range [-1, 1]
        uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

        if ssaa > 1:
            h = int(h0 * ssaa)
            w = int(w0 * ssaa)
        else:
            h, w = h0, w0

        # rasterize 2d texture vertices to texture image
        rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (h, w)) # [1, h, w, 4]rast

        # interpolate to get the corresponding 3D location of each pixel
        xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f) # [1, h, w, 3]
        mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f) # [1, h, w, 1]
        vt = vt.cpu()
        ft = ft.cpu()
        uv = uv.cpu()
        
        # masked query
        xyzs = xyzs.view(-1, 3).cpu()
        mask = (mask > 0).view(-1).cpu()

        # auto-completion for the boundary in mesh seperation
        if mask.any() and hparams.texture_boundary_input_label is not None and os.path.exists(hparams.texture_boundary_input_label):
            # the same contraction used in nerf training
            xyzs_mask_contract = contract(xyzs[mask], ngp_aabb.to(xyzs.device), hparams.contraction_type)
            with open(hparams.texture_boundary_input_label, 'rb') as f_bound:
                boundary_dict = pickle.load(f_bound)
                boundary = boundary_dict["boundary"]
                global_voxel_size = boundary_dict["global_voxel_size"]
            voxel_length = (global_voxel_size[0] ** 2 + global_voxel_size[1] ** 2 + global_voxel_size[2] ** 2) ** 0.5

            # our job is to find the intersection between "xyzs_mask_contract" and boundary voxels (criterion is within 5 times of voxel length)
            tree = KDTree(boundary)
            neighbor_dists, neighbor_indices = tree.query(xyzs_mask_contract.numpy(), distance_upper_bound=5 * voxel_length)
            
            # the corresponding 3D location of each pixel that is also in the boundary
            in_boundary = torch.from_numpy(neighbor_indices != boundary.shape[0])
            total_cnt = int(torch.count_nonzero(in_boundary))

            # we replace those location with the nearest 3D location that is not inside the boundary gradually
            lookup_batch = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            xyzs_mask_original = xyzs[mask].clone()
            if total_cnt > 100:
                for lookup_ratio in lookup_batch:

                    xyzs_in_boundary = xyzs_mask_original[in_boundary]
                    xyzs_to_lookup = xyzs_mask_original[torch.logical_not(in_boundary)]

                    step_cnt = xyzs_in_boundary.shape[0]

                    tree = KDTree(xyzs_to_lookup.numpy())
                    neighbor_dists, neighbor_indices = tree.query(xyzs_in_boundary.numpy())

                    sort_dist_idxs = np.argsort(neighbor_dists)
                    near_idxs = sort_dist_idxs[:int(lookup_ratio*step_cnt)]

                    mask_and_in_boundary = torch.arange(xyzs.shape[0])[mask][in_boundary][near_idxs]
                    xyzs[mask_and_in_boundary.cpu()] = xyzs[mask][torch.logical_not(in_boundary)][torch.from_numpy(neighbor_indices.astype(np.int32)[near_idxs]).long()]

                    in_boundary[np.arange(in_boundary.shape[0])[in_boundary][near_idxs]] = False

        feats = torch.zeros(h * w, 3+hparams.baking_specular_dim, dtype=torch.float32).cpu()

        if mask.any():
            with torch.no_grad():
                xyzs = xyzs[mask] # [M, 3]
                chunk_size = 160000
                # batched inference to avoid OOM
                all_feats = torch.zeros((xyzs.shape[0], 3+hparams.baking_specular_dim)).cpu()
                head = 0
                while head < xyzs.shape[0]:
                    tail = min(head + chunk_size, xyzs.shape[0])
                    with torch.cuda.amp.autocast(enabled=False):
                        slice_xyzs = xyzs[head:tail].clone().detach().cuda()
                        all_feats[head:tail] = color_model.geo_feat(slice_xyzs, make_contract=True).cpu().float()
                        slice_xyzs = slice_xyzs.cpu()
                        del slice_xyzs
                    head += chunk_size

                feats[mask] = all_feats

        feats = feats.view(h, w, -1)
        mask = mask.view(h, w)

        # we might need antialias here
        # feats = feats.unsqueeze(0)
        # mask = mask.unsqueeze(0).unsqueeze(-1)

        # feats = dr.antialias(feats.cuda().float(), rast, uv.unsqueeze(0), ft).squeeze(0).cpu()
        # mask = dr.antialias(mask.cuda().float(), rast, uv.unsqueeze(0), ft).squeeze(0).squeeze(-1).cpu().int()

        # quantize [0.0, 1.0] to [0, 255]
        feats = feats.cpu().numpy()
        feats = (feats * 255).astype(np.uint8)

        # antialiasing and padding
        mask = mask.cpu().numpy()

        if np.logical_not(mask).any():
            inpaint_region = binary_dilation(mask, iterations=hparams.texture_dilate)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=hparams.texture_erode)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
            _, indices = knn.kneighbors(inpaint_coords)

            feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]

        # ssaa
        if hparams.baking_specular_dim == 0:
            feats0 = cv2.cvtColor(feats[..., :3], cv2.COLOR_RGB2BGR)
            if ssaa > 1:
                feats0 = cv2.resize(feats0, (w0, h0), interpolation=cv2.INTER_LINEAR)
            cas = cas if hparams.texture_export_cas is None else hparams.texture_export_cas
            os.makedirs(export_path, exist_ok=True)
            if hparams.texture_compress != 0:
                png_compression_level = hparams.texture_compress
                cv2.imwrite(os.path.join(export_path, f'diffuse_{cas}.png'), feats0,
                            [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
            else:
                cv2.imwrite(os.path.join(export_path, f'diffuse_{cas}.png'), feats0)

        else:
            feats0 = cv2.cvtColor(feats[..., :3], cv2.COLOR_RGB2BGR)
            feats1 = cv2.cvtColor(feats[..., 3:], cv2.COLOR_RGB2BGR)

            if ssaa > 1:
                feats0 = cv2.resize(feats0, (w0, h0), interpolation=cv2.INTER_LINEAR)
                feats1 = cv2.resize(feats1, (w0, h0), interpolation=cv2.INTER_LINEAR)

            cas = cas if hparams.texture_export_cas is None else hparams.texture_export_cas
            os.makedirs(export_path, exist_ok=True)
            if hparams.texture_compress != 0:
                png_compression_level = hparams.texture_compress
                cv2.imwrite(os.path.join(export_path, f'feat0_{cas}.png'), feats0, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
                cv2.imwrite(os.path.join(export_path, f'feat1_{cas}.png'), feats1, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
            else:
                cv2.imwrite(os.path.join(export_path, f'feat0_{cas}.png'), feats0)
                cv2.imwrite(os.path.join(export_path, f'feat1_{cas}.png'), feats1)

        # save obj (v, vt, f /)
        mesh_file = os.path.join(export_path, f'mesh_{cas}.ply')

        # relocation for kitti loop
        if hparams.dataset_name == 'kitti' and hparams.kitti_reloc:
            with open(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'pos_trans.pkl'),  'rb') as f:
                pos_trans = pickle.load(f)
                center = pos_trans['center']
                scale = pos_trans['scale']
                forward = pos_trans['forward']
                print('local pos_trans: ', pos_trans)
            v_np = (v_np + forward * 0.5) * scale + center

            with open(hparams.center_pos_trans,  'rb') as f:
                pos_trans = pickle.load(f)
                center = pos_trans['center']
                scale = pos_trans['scale']
                forward = pos_trans['forward']
                print('center pos_trans: ', pos_trans)
            v_np = (v_np - center) / scale - 0.5 * forward
        
        print(f'[INFO] writing obj mesh to {mesh_file}')


        num_verts = v_np.shape[0]
        num_faces = f_np.shape[0]

        verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

        for i in range(0, num_verts):
            verts_tuple[i] = tuple(v_np[i, :])

        faces_building = []
        vt[:, 1] = 1 - vt[:, 1]
        for i in range(0, num_faces):
            faces_building.append(((f_np[i, :].tolist(), vt[ft[i, :]].reshape(-1).tolist())))
        faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,)), ("texcoord", "f4", (6,))])

        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

        ply_data = plyfile.PlyData([el_verts, el_faces])
        ply_filename_out = './mesh_0.ply'
        print("saving mesh to %s" % (ply_filename_out))
        ply_data.write(ply_filename_out)

        plydata = plyfile.PlyData.read('./mesh_0.ply')
        print(plydata.elements[1].name)
        print(plydata.elements[1].properties)
        print(plydata.elements[1].data[0])



    vts = []
    fts = []
    if hparams.texture_export_substitude_mesh:
        vertices = []
        triangles = []
        vts = []
        fts = []
        v_cumsum = [0]
        f_cumsum = [0]

        assert len(hparams.load_mesh_paths) > 0
        mesh_cascades = 0
        for mesh_path in hparams.load_mesh_paths:
            print(f"loading mesh {mesh_path}")
            if mesh_path[-4:] == '.ply':
                mesh = trimesh.load(mesh_path, force='mesh', skip_material=True, process=False)
                print(f"mesh {mesh_path} loaded")
                vertices.append(mesh.vertices)
                triangles.append(mesh.faces + v_cumsum[-1])
                v_cumsum.append(v_cumsum[-1] + mesh.vertices.shape[0])
                f_cumsum.append(f_cumsum[-1] + mesh.faces.shape[0])
            else:
                assert False

            mesh_cascades += 1

        vertices = torch.from_numpy(np.concatenate(vertices, axis=0)).float()
        triangles = torch.from_numpy(np.concatenate(triangles, axis=0)).int()

    v = vertices.cuda().detach()
    f = triangles.cuda().detach()

    # for kitti loop: divide the whole mesh to get the part that's fixed in the loop
    keep_conds = hparams.keep

    for cas in range(mesh_cascades):

        # for skydome, we use a smaller texture size
        if cas == mesh_cascades - 1 and hparams.skydome:
            h0 //= 2
            w0 //= 2
            
        cur_v = v[v_cumsum[cas]:v_cumsum[cas+1]]
        cur_f = f[f_cumsum[cas]:f_cumsum[cas+1]] - v_cumsum[cas]
        cur_f = cur_f.long()

        print(f"[INFO] cascade = {cas}")
        print("cur_v: ", cur_v.shape)
        print("cur_f: ", cur_f.min(), cur_f.max(), cur_f.shape)
        
        # for kitti loop: divide the whole mesh to get the part that's fixed in the loop
        # but don't act on skydome mesh
        if keep_conds != '' and (not (cas == mesh_cascades - 1 and hparams.skydome)):

            vertices = cur_v.reshape(-1, 3).clone().cpu()

            pos_trans_path = os.path.join(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}'), 'pos_trans.pkl')

            if 'kitti' in hparams.dataset_name and hparams.kitti_reloc:
                center_trans_path = hparams.center_pos_trans

                with open(pos_trans_path,  'rb') as file:
                    pos_trans = pickle.load(file)
                    center = pos_trans['center']
                    scale = pos_trans['scale']
                    forward = pos_trans['forward']
                _vertices = (vertices + 0.5 * forward.reshape(-1, 3)) * scale + center.reshape(-1, 3)

                with open(center_trans_path, 'rb') as file:
                    pos_trans = pickle.load(file)
                    center = pos_trans['center']
                    scale = pos_trans['scale']
                    forward = pos_trans['forward']
                _vertices = (_vertices - center.reshape(-1, 3)) / scale - 0.5 * forward.reshape(-1, 3)
            else:
                _vertices = vertices
            conds = keep_conds.split(':')
            print("conds: ", conds)
            keep = torch.ones((_vertices.shape[0])).bool().cuda()
            _vertices = _vertices.cuda()
            for cond in conds:
                axis = cond[0]
                dir = cond[1]
                amnt = float(cond[2:])
                if axis == 'x':
                    test_v = _vertices[:, 0]
                elif axis == 'y':
                    test_v = _vertices[:, 1]
                if dir == '<':
                    keep = torch.logical_and(keep, test_v < amnt)
                elif dir == '>':
                    keep = torch.logical_and(keep, test_v > amnt)

            cur_v, cur_f = filter_mesh(keep, cur_v, cur_f)

        _export_obj(cur_v, cur_f.int(), h0, w0, export_path, hparams.ssaa, cas)
    
    
    # save mlp as json
    if hparams.baking_specular_dim > 0:
        params = dict(color_model.specular_net.named_parameters())

        mlp = {}
        for k, p in params.items():
            p_np = p.detach().cpu().numpy().T
            print(f'[INFO] wrting MLP param {k}: {p_np.shape}')
            mlp[k] = p_np.tolist()

        mlp['bound'] = color_model.scale
        mlp['cascade'] = mesh_cascades

        mlp_file = os.path.join(export_path, f'mlp.json')
        with open(mlp_file, 'w') as fp:
            json.dump(mlp, fp, indent=2)