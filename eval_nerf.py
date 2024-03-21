import json

import torch
from torch import nn
from opt import get_opts
import os
import numpy as np
import cv2
from einops import rearrange
from tqdm import tqdm
# data
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES

from utils import save_image

from utils import load_ckpt

import warnings; warnings.filterwarnings("ignore")
import pickle5 as pickle

import torch.nn.functional as F
from skimage.metrics import structural_similarity as calculate_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import time

def depth2img(depth):
    max_depth = depth.max()
    min_depth = depth.min()
    print("max_depth: ", max_depth)
    print("min_depth: ", min_depth)
    max_depth = 2
    min_depth = 0
    depth = (depth - min_depth) / (max_depth - min_depth)
    depth = np.clip(depth, a_min=0., a_max=1.)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_PLASMA)
    return depth_img

def compute_psnr(gt, pred):
    mse = torch.mean((gt - pred)**2)
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
            self.val_lpips(torch.clip(pred*2-1, -1, 1),
                            torch.clip(x*2-1, -1, 1))
            lpips = self.val_lpips.compute()
            self.val_lpips.reset()
        return lpips

if __name__ == '__main__':
    hparams = get_opts()

    # network parameters
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

    model = NGP(scale=hparams.scale, rgb_act=rgb_act, embed_a=hparams.embed_a,
                     embed_a_len=hparams.embed_a_len, classes=hparams.num_classes,
                     contraction_type=hparams.contraction_type, ngp_params=ngp_params, mlp_params=mlp_params,
                     sphere_scale=((hparams.sphere_scale_x_n, hparams.sphere_scale_y_n, hparams.sphere_scale_z_n),
                                   (hparams.sphere_scale_x_p, hparams.sphere_scale_y_p, hparams.sphere_scale_z_p)),
                     grid_size=hparams.ngp_gridsize,
                     cascade=hparams.cascade,
                     ).cuda().eval()

    ### setup appearance embeddings
    if hparams.embed_a:

        img_dir_name = None
        if os.path.exists(os.path.join(hparams.root_dir, 'images')):
            img_dir_name = 'images'
        elif os.path.exists(os.path.join(hparams.root_dir, 'rgb')):
            img_dir_name = 'rgb'
        elif os.path.exists(os.path.join(hparams.root_dir, 'perspective')):
            img_dir_name = 'perspective'

        if hparams.dataset_name == 'kitti':
            N_imgs = 2 * (hparams.kitti_end - hparams.kitti_start + 1 +
                               (0 if not hparams.kitti_dual_seq else (
                                       hparams.kitti_dual_end - hparams.kitti_dual_start + 1)) +
                               (0 if not hparams.strict_split else (-1 * len(hparams.kitti_test_id)))
                               ) if hparams.n_imgs is None else hparams.n_imgs
        else:
            N_imgs = len(os.listdir(os.path.join(hparams.root_dir, img_dir_name)))

        all_embedding_a = torch.nn.Embedding(N_imgs, hparams.embed_a_len)

    # density grid for ray points sampling
    G = model.grid_size
    model.register_buffer('density_grid',
        torch.zeros(model.cascades, G**3))
    model.register_buffer('grid_coords',
        create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

    ckpt_path = hparams.ckpt_load

    print(f'ckpt specified: {ckpt_path} !')
    load_ckpt(model, ckpt_path)
    if hparams.embed_a:
        load_ckpt(all_embedding_a, ckpt_path, model_name='embedding_a')


    dataset = dataset_dict[hparams.dataset_name]
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

        # double sequence in kitti360 dataset
        kwargs['kitti_dual_seq'] = hparams.kitti_dual_seq
        kwargs['kitti_dual_start'] = hparams.kitti_dual_start
        kwargs['kitti_dual_end'] = hparams.kitti_dual_end

        val_list = []
        for i in hparams.kitti_test_id:
            val_list.append(int(i))
        kwargs['test_id'] = val_list

        kwargs['lidar_proxy'] = False

    dataset = dataset(split='test', **kwargs)
    val_pips = LPIPSVal().cuda()

    avg_stats = {
        'mse_color': 0,
        'psnr': 0,
        'ssim': 0,
        'lpips': 0,
        'l1_loss': 0,
        'l2_loss': 0,
        'outlier': 0,
    }

    img_dir = os.path.join('results', hparams.dataset_name, hparams.exp_name, hparams.eval_nerf_output)
    os.makedirs(img_dir, exist_ok=True)
    w, h = dataset.img_wh
    for tid in tqdm(range(dataset.poses.shape[0])):
        batch = dataset[tid]
        for i in batch:
            if isinstance(batch[i], torch.Tensor):
                batch[i] = batch[i].cuda()
        with torch.no_grad():
            kwargs = {'test_time': True,
                  'random_bg': hparams.random_bg,
                  'render_rgb': hparams.render_rgb,
                  'render_depth': hparams.render_depth,
                  'render_normal': hparams.render_normal,
                  'render_sem': hparams.render_semantic,
                  'img_wh': dataset.img_wh,
                  'T_threshold': hparams.T_threshold,
                  'classes': hparams.num_classes}

            kwargs['exp_step_factor'] = 1/256

            if hparams.embed_a:
                embedding_a = all_embedding_a(torch.tensor([0])).cuda()
                kwargs['embedding_a'] = embedding_a

            if hparams.dataset_name not in ["vr_nerf"]:
                poses = batch['pose'].cuda()
                directions = dataset.directions.cuda()
                rays_o, rays_d = get_rays(directions, poses)
            else:
                rays_d = dataset.rays_d[batch["img_idxs"]].float().cuda()
                rays_o = dataset.poses[batch['img_idxs'], :3, 3].expand_as(rays_d).float().cuda()

            chunk_size = 131072
            all_ret = {}
            for i in range(0, rays_o.shape[0], chunk_size):
                ret = render(model, rays_o[i:i+chunk_size], rays_d[i:i+chunk_size], **kwargs)
                for k in ret:
                    if k not in all_ret:
                        all_ret[k] = []
                    all_ret[k].append(ret[k])
            for k in all_ret:
                if k in ['total_samples']:
                    continue
                all_ret[k] = torch.cat(all_ret[k], 0)
            all_ret['total_samples'] = torch.sum(torch.tensor(all_ret['total_samples']))

            results = all_ret

        rgb_pred = rearrange(results['rgb'], '(h w) c -> h w c', h=h)

        color_pred = rgb_pred.reshape(h, w, 3)
        color_gt = batch['rgb'].reshape(h, w, 3)

        mse_color = F.mse_loss(color_pred, color_gt)

        psnr = compute_psnr(color_gt, color_pred)
        ssim = compute_ssim(color_gt, color_pred)
        lpips = val_pips(color_gt, color_pred)

        # DEPTH AND NORMAL

        # print("results['depth']: ", results['depth'].shape)
        depth_raw = rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h)
        depth_pred = depth2img(depth_raw)

        accumulate = rearrange(results['opacity'].cpu().numpy(), '(h w) -> h w', h=h)
        accumulate = cv2.applyColorMap((accumulate * 255).astype(np.uint8),
                                      cv2.COLORMAP_PLASMA)

        def convert_normal(normal, pose_c2w):
            R_w2c = pose_c2w[:3, :3].T
            normal_cam = normal @ R_w2c.T
            x, y, z = normal_cam[..., 0], normal_cam[..., 1], normal_cam[..., 2]
            # normal_out = np.stack([z, x, -y], axis=-1)
            normal_out = np.stack([x, y, z], axis=-1)
            # normal_out = np.stack([y, x, z], axis=-1)
            return normal_out
        results['normal_pred'] = np.array(results['normal_pred'].cpu()).reshape(-1, 3)
        results['normal_pred'] = convert_normal(results['normal_pred'], np.array(poses.cpu()).reshape(-1, 4))

        results['normal_raw'] = np.array(results['normal_raw'].cpu()).reshape(-1, 3)
        results['normal_raw'] = convert_normal(results['normal_raw'], np.array(poses.cpu()).reshape(-1, 4))

        normal_pred = rearrange((results['normal_pred']+1)/2, '(h w) c -> h w c', h=h)
        normal_raw = rearrange((results['normal_raw']+1)/2, '(h w) c -> h w c', h=h)

        save_image(color_pred, os.path.join(img_dir, f'rgb_{tid}.png'))
        save_image(color_gt, os.path.join(img_dir, f'rgb_gt_{tid}.png'))
        cv2.imwrite(os.path.join(img_dir, f'depth_{tid}.png'), depth_pred)
        cv2.imwrite(os.path.join(img_dir, f'accu_{tid}.png'), accumulate)
        save_image(normal_pred, os.path.join(img_dir, f'normal_pred_{tid}.png'))
        save_image(normal_raw, os.path.join(img_dir, f'normal_raw_{tid}.png'))

        stats = {
            'mse_color': float(mse_color.detach().cpu().item()),
            'psnr': float(psnr),
            'ssim': float(ssim),
            'lpips': float(lpips),
        }

        print(f"psnr: {psnr}")

        for k in stats.keys():
            avg_stats[k] += stats[k]

    for k in avg_stats.keys():
        avg_stats[k] = avg_stats[k] / dataset.poses.shape[0]

    print("avg_stats: ", avg_stats)
    with open(os.path.join(img_dir, 'eval_stat.json'), 'w') as f:
        json.dump(avg_stats, f)
