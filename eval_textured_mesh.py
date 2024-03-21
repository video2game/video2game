from opt import get_opts
import cv2
import time
import os
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as calculate_ssim
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import load_ckpt
import warnings; warnings.filterwarnings("ignore")
import pickle5 as pickle
from models.tcngp import TCNGP
import torch.nn.functional as F
import nvdiffrast.torch as dr
from meshutils import *
import tqdm
from PIL import Image
from uvmesh import read_obj
import json

class Timer:
    def __init__(self, cuda_sync: bool = False):
        self.cuda_sync = cuda_sync
        self.reset()

    def reset(self):
        if self.cuda_sync:
            torch.cuda.synchronize()
        self.start = time.time()

    def get_time(self, reset=True):
        if self.cuda_sync:
            torch.cuda.synchronize()
        now = time.time()
        interval = now - self.start
        if reset:
            self.reset()
        return interval

    def print_time(self, info, reset=True):
        interval = self.get_time(reset)
        print('{:.5f} | {}'.format(interval, info))


def compute_psnr(gt, pred):
    mse = torch.mean((gt - pred) ** 2)
    device = gt.device
    psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).to(device))
    psnr = psnr.cpu().item()
    return psnr


def compute_ssim(gt, pred):
    '''image size: (h, w, 3)'''
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()
    ssim = calculate_ssim(pred, gt, data_range=gt.max() - gt.min(), multichannel=True)
    return ssim


class LPIPSVal(nn.Module):
    def __init__(self):
        super().__init__()
        self.val_lpips = LearnedPerceptualImagePatchSimilarity('alex')
        for p in self.val_lpips.net.parameters():
            p.requires_grad = False

    def forward(self, pred, x):
        pred = pred.permute(2, 0, 1).unsqueeze(0)
        x = x.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            self.val_lpips(torch.clip(pred * 2 - 1, -1, 1),
                           torch.clip(x * 2 - 1, -1, 1))
            lpips = self.val_lpips.compute()
            self.val_lpips.reset()
        return lpips

val_pips = LPIPSVal().cuda()

def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db,
                          diff_attrs=None if rast_db is None else 'all')

if __name__ == '__main__':


    hparams = get_opts()

    test_dataset = dataset_dict[hparams.dataset_name]
    kwargs = {'root_dir': hparams.root_dir,
              'downsample': hparams.downsample,
              'strict_split': hparams.strict_split,
              'use_sem': hparams.render_semantic,
              'depth_mono': False,
              'normal_mono': False,
              'depth_dp_proxy': False,
              'use_mask': False,
              'classes': hparams.num_classes,
              'scale_factor': hparams.scale_factor}

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

    test_dataset = test_dataset(split='test', **kwargs)
    resolution = [test_dataset.img_wh[1], test_dataset.img_wh[0]]

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
                             ngp_params=tcngp_params, specular_dim_mlp=hparams.baking_specular_dim_mlp,
                             per_level_scale=hparams.baking_per_level_scale, specular_dim=hparams.baking_specular_dim,
                             color_net_params=color_net_params).cuda().eval()
    ckpt_path = os.path.join('ckpts', hparams.dataset_name, hparams.exp_name, 'baking', hparams.workspace,'last.ckpt') if hparams.ckpt_load is None else hparams.ckpt_load

    load_ckpt(color_model, ckpt_path, model_name='color_model',
              prefixes_to_ignore=["aabb", "aabb_train", "aabb_infer", "density_bitfield", "density_grid", "grid_coords",
                                  "xyz_encoder.params", "xyz_net.0.weight", "xyz_net.0.bias", "xyz_net.2.weight",
                                  "xyz_net.2.bias", "rgb_encoder.params", "dir_encoder.params", "rgb_net.params",
                                  "norm_pred_header.params", "semantic_header.params"])

    glctx = dr.RasterizeGLContext(output_db=False)
    mesh_root = os.path.join('results', hparams.dataset_name, hparams.exp_name, 'baking', hparams.workspace, hparams.baking_output)

    mesh_cascades = len([item for item in os.listdir(mesh_root) if item[-4:] == '.obj'])
    mesh_dict_list = []

    for cas in range(mesh_cascades):

        filename = f'{mesh_root}/mesh_{cas}.obj'
        v_np, vt_np, f_np, ft_np = read_obj(filename)

        if 'kitti' in hparams.dataset_name and hparams.kitti_reloc:
            center_trans_path = hparams.center_pos_trans
            with open(center_trans_path, 'rb') as file:
                pos_trans = pickle.load(file)
                center = pos_trans['center']
                scale = pos_trans['scale']
                forward = pos_trans['forward']

            v_np = (v_np + 0.5 * forward.reshape(-1, 3)) * scale + center.reshape(-1, 3)
            pos_trans_path = f'results/kitti/{hparams.exp_name}/pos_trans.pkl'
            with open(pos_trans_path, 'rb') as file:
                pos_trans = pickle.load(file)
                center = pos_trans['center']
                scale = pos_trans['scale']
                forward = pos_trans['forward']

            v_np = (v_np - center.reshape(-1, 3)) / scale - 0.5 * forward.reshape(-1, 3)

        tex0_path = f'{mesh_root}/feat0_{cas}.png'
        tex1_path = f'{mesh_root}/feat1_{cas}.png'
        vertices = torch.tensor(v_np.astype(np.float32)).float()
        pos_idx = torch.tensor(f_np.astype(np.int32))
        uv = torch.tensor(vt_np.astype(np.float32)).float()
        uv_idx = torch.tensor(ft_np.astype(np.int32))

        tex0 = torch.tensor(np.array(Image.open(tex0_path))).float()
        tex1 = torch.tensor(np.array(Image.open(tex1_path))).float()
        mesh_dict = {
            'vertices': F.pad(vertices, pad=(0, 1), mode='constant', value=1.0).cuda().contiguous(),
            'pos_idx': pos_idx.cuda().contiguous(),
            'uv': uv[None, ...].cuda().contiguous(),
            'uv_idx': uv_idx.cuda().contiguous(),
            'tex0': tex0[None, ...].cuda().contiguous(),
            'tex1': tex1[None, ...].cuda().contiguous(),
        }
        mesh_dict_list.append(mesh_dict)

    avg_stats = {
        'mse_color': 0,
        'psnr': 0,
        'ssim': 0,
        'lpips': 0,
        'FPS': 0,
        'time': 0
    }

    img_dir = os.path.join('results', hparams.dataset_name, hparams.exp_name, 'baking', hparams.workspace, hparams.baking_output, hparams.eval_textured_mesh_output)
    os.makedirs(img_dir, exist_ok=True)

    with torch.no_grad():

        for index in tqdm.tqdm(range(test_dataset.mvps.shape[0])):
            rgb_canvas = torch.ones((resolution[0], resolution[1], 3)).cuda()
            depth_buf = torch.ones((resolution[0], resolution[1], 3)).cuda()*1e5
            timer = Timer(cuda_sync=True)
            mvp = test_dataset.mvps[index].float().cuda()
            poses = test_dataset.poses[index].cuda()

            if hparams.dataset_name not in ["vr_nerf"]:
                directions = test_dataset.directions.cuda()
                rays_o, rays_d = get_rays(directions, poses)
            else:
                rays_d = test_dataset.rays_d[index].float().cuda()
                rays_o = test_dataset.poses[index, :3, 3].expand_as(rays_d).float().cuda()

            rays_d = rays_d.reshape(-1, 3)
            dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

            for cas in range(mesh_cascades):
                mesh_dict = mesh_dict_list[cas]
                vertices = mesh_dict['vertices']
                pos_idx = mesh_dict['pos_idx']
                uv = mesh_dict['uv']
                uv_idx = mesh_dict['uv_idx']
                tex0 = mesh_dict['tex0']
                tex1 = mesh_dict['tex1']

                vertices_clip = torch.matmul(vertices, torch.transpose(mvp, 0, 1)).float().unsqueeze(0)
                rast_out, _ = dr.rasterize(glctx, vertices_clip, pos_idx, resolution=resolution)
                rast_out = rast_out.flip([1])
                rast_out_db = None

                texc, _ = dr.interpolate(uv, rast_out, uv_idx)

                diffuse = dr.texture(tex0, texc, filter_mode='auto', boundary_mode='clamp')

                feats = dr.texture(tex1, texc, filter_mode='auto', boundary_mode='clamp')

                depth = rast_out[..., 2]
                mask = torch.clamp(rast_out[..., -1:], 0, 1)

                feats = feats.reshape(-1, hparams.baking_specular_dim) / 255.0

                mask = mask.int().bool()

                layer_mask = mask.reshape(test_dataset.img_wh[1], test_dataset.img_wh[0], 1)

                specular = color_model.specular_net(torch.cat((dirs, feats), dim=-1)).reshape(test_dataset.img_wh[1], test_dataset.img_wh[0], 3)
                specular = torch.sigmoid(specular)

                layer_color = torch.clamp(diffuse + specular * 255.0, 0, 255)
                layer_depth = depth.view(test_dataset.img_wh[1], test_dataset.img_wh[0], 1)

                cond = torch.logical_and(layer_depth < depth_buf, layer_mask)
                rgb_canvas = torch.where(cond.expand(-1, -1, 3), layer_color, rgb_canvas)
                depth_buf = torch.where(cond.expand(-1, -1, 3), layer_depth, depth_buf)                    # layer_stack[cond.cpu()] = cas+1
                    # rgb_canvas = layer_color
            _time = timer.get_time()

            _color = np.array(rgb_canvas.clamp(0., 255.).clone().cpu(), dtype=np.uint8).reshape(resolution[0], resolution[1], 3)
            _color = cv2.cvtColor(_color, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(img_dir, f"texture_test_{index}.png"), _color)
            color = rgb_canvas / 255.0
            img_gt = test_dataset.rays[index].float().cuda()
            w, h = test_dataset.img_wh
            color_gt = img_gt.reshape((h, w, 3))
            color_pred = color.reshape((h, w, 3))

            mse_color = F.mse_loss(color_pred, color_gt)

            psnr = compute_psnr(color_gt, color_pred)
            ssim = compute_ssim(color_gt, color_pred)
            lpips = val_pips(color_gt, color_pred)

            stats = {
                'mse_color': float(mse_color.detach().cpu().item()),
                'psnr': float(psnr),
                'ssim': float(ssim),
                'lpips': float(lpips),
                'FPS': float(1 / _time),
                'time': float(_time)
            }
            for k in stats.keys():
                avg_stats[k] += stats[k]

    for k in avg_stats.keys():
        avg_stats[k] = avg_stats[k] / test_dataset.poses.shape[0]

    print("avg_stats: ", avg_stats)

    with open(os.path.join(img_dir, 'eval_stat.json'), 'w') as f:
        json.dump(avg_stats, f)