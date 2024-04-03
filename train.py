import torch
from torch import nn
from opt import get_opts
import os
import imageio
import numpy as np
import cv2
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES
from models.global_var import global_var

# optimizer, losses
# from apex.optimizers import FusedAdam
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure
)

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils import slim_ckpt, load_ckpt, save_image

# render path
from utils import load_ckpt

import warnings;

warnings.filterwarnings("ignore")
import pickle
from tqdm import tqdm

from skimage.metrics import structural_similarity as calculate_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb


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


def depth2img(depth):
    max_depth = depth.max()
    min_depth = depth.min()
    max_depth = 8
    min_depth = 0
    depth = (depth - min_depth) / (max_depth - min_depth)
    depth = np.clip(depth, a_min=0., a_max=1.)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_PLASMA)
    return depth_img


def mask2img(mask):
    mask_img = cv2.applyColorMap((mask * 255).astype(np.uint8),
                                 cv2.COLORMAP_BONE)

    return mask_img


def semantic2img(sem_label, classes):
    level = 1 / (classes - 1)
    sem_color = level * sem_label
    sem_color = cv2.applyColorMap((sem_color * 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return sem_color


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        # loss function
        loss_settings = {
            # opacity loss in every ray
            # default: encourage opacity to be either 0 or 1 to avoid floater
            # solid: only encourage opacity to be 1
            'opacity_solid': hparams.nerf_opacity_solid,
            'lambda_opa': hparams.nerf_lambda_opa,
            
            # opacity loss in every sampled points
            'lambda_xyz_opa': hparams.nerf_lambda_xyz_opa,
            
            # distortion loss
            'lambda_distortion': hparams.nerf_lambda_distortion,
            
            # depth prior loss
            'lambda_depth_mono': hparams.nerf_lambda_depth_mono,
            'lambda_depth_dp': hparams.nerf_lambda_depth_dp,
            
            # normal prior loss
            'lambda_normal_mono': hparams.nerf_lambda_normal_mono,
            # skip normal loss in those semantics
            'normal_mono_skipsem': hparams.normal_mono_skipsem,
            
            # normal ref loss: encourage pred normal and analytic normal to be consistent
            'lambda_normal_ref': hparams.nerf_lambda_normal_ref,
            
            # sky loss: push the sky far away
            'lambda_sky': hparams.nerf_lambda_sky,
            # assign sky semantic label to prevent depth prior loss in those sky pixels since skyloss has already existed
            'sky_sem': hparams.sky_sem,
            
            # semantic loss
            'lambda_semantic': hparams.nerf_lambda_semantic,
            
            # sparsity loss
            'remove_sparsity': hparams.remove_sparsity,
            'lambda_sparsity': hparams.nerf_lambda_sparsity,
            
            # lidar prior loss if necessary
            'lambda_lidar_proxy': hparams.nerf_lambda_lidar_proxy,
            
            # normal smoothness loss if necessary
            'lambda_normal_sm': hparams.nerf_lambda_normal_sm,
        }
        
        self.loss = NeRFLoss(**loss_settings)

        # metrics
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('alex')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

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

        self.model = NGP(scale=hparams.scale, rgb_act=rgb_act, embed_a=hparams.embed_a,
                         embed_a_len=hparams.embed_a_len, classes=hparams.num_classes,
                         contraction_type=hparams.contraction_type, ngp_params=ngp_params, mlp_params=mlp_params,
                         sphere_scale=((hparams.sphere_scale_x_n, hparams.sphere_scale_y_n, hparams.sphere_scale_z_n),
                                       (hparams.sphere_scale_x_p, hparams.sphere_scale_y_p, hparams.sphere_scale_z_p)),
                         grid_size=hparams.ngp_gridsize,
                         cascade=hparams.cascade,
                         )

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
                self.N_imgs = 2 * (hparams.kitti_end - hparams.kitti_start + 1 +
                                   (0 if not hparams.kitti_dual_seq else (
                                               hparams.kitti_dual_end - hparams.kitti_dual_start + 1)) +
                                   (0 if not hparams.strict_split else (-1 * len(hparams.kitti_test_id)))
                                   ) if hparams.n_imgs is None else hparams.n_imgs
            else:
                self.N_imgs = len(os.listdir(os.path.join(hparams.root_dir, img_dir_name)))

            self.embedding_a = torch.nn.Embedding(self.N_imgs, hparams.embed_a_len)

        # density grid for ray points sampling
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
                                   torch.zeros(self.model.cascades, G ** 3))
        self.model.register_buffer('grid_coords',
                                   create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

    def forward(self, batch, split):

        # get ray info
        
        # for datasets that all intrinsics are the same
        if self.hparams.dataset_name not in ["vr_nerf"]:
            if split == 'train':
                poses = self.poses[batch['img_idxs']]
                directions = self.directions[batch['pix_idxs']]
            else:
                poses = batch['pose']
                directions = self.directions

            poses_ = poses
            
            # optimize extrinsics if neccessary
            if self.hparams.optimize_ext:
                dR = axisangle_to_R(self.dR[batch['img_idxs']])
                poses_ = torch.zeros_like(poses).cuda()
                poses_[..., :3] = dR @ poses[..., :3]
                dT = self.dT[batch['img_idxs']]
                poses_[..., 3] = poses[..., 3] + dT

            rays_o, rays_d = get_rays(directions, poses_)
        
        # for dataset like vr_nerf that has different intrinsics in one scene
        else:
            if split == 'train':
                selected_dataset = self.train_dataset
                rays_d = selected_dataset.rays_d[batch['img_idxs'], batch['pix_idxs']].float().cuda()
                rays_o = selected_dataset.poses[batch['img_idxs'], :3, 3].expand_as(rays_d).float().cuda()

            else:
                selected_dataset = self.test_dataset
                rays_d = selected_dataset.rays_d[batch["img_idxs"]].float().cuda()
                rays_o = selected_dataset.poses[batch['img_idxs'], :3, 3].expand_as(rays_d).float().cuda()

        # get appearance embeddings
        if self.hparams.embed_a and split == 'train':
            embedding_a = self.embedding_a(batch['img_idxs'])
        elif self.hparams.embed_a and split == 'test':
            embedding_a = self.embedding_a(torch.tensor([0], device=rays_d.device))

        # rendering parameters
        kwargs = {'test_time': split != 'train',
                  'random_bg': self.hparams.random_bg,
                  'render_rgb': hparams.render_rgb,
                  'render_depth': hparams.render_depth,
                  'render_normal': hparams.render_normal,
                  'render_sem': hparams.render_semantic,
                  'img_wh': self.img_wh,
                  'T_threshold': self.hparams.T_threshold,
                  "normal_sm_eps": self.hparams.nerf_normal_sm_eps,
                  'classes': self.hparams.num_classes,
                  'normal_sm': self.hparams.nerf_normal_sm}

        if self.hparams.embed_a:
            kwargs['embedding_a'] = embedding_a

        # rendering process
        if split == 'train':
            if self.hparams.max_samples is not None:
                kwargs['max_samples'] = self.hparams.max_samples

            return render(self.model, rays_o, rays_d, **kwargs)
        else:
            chunk_size = self.hparams.chunk_size
            all_ret = {}
            for i in range(0, rays_o.shape[0], chunk_size):
                ret = render(self.model, rays_o[i:i + chunk_size], rays_d[i:i + chunk_size], **kwargs)
                for k in ret:
                    if k not in all_ret:
                        all_ret[k] = []
                    all_ret[k].append(ret[k])
            for k in all_ret:
                if k in ['total_samples']:
                    continue
                all_ret[k] = torch.cat(all_ret[k], 0)
            all_ret['total_samples'] = torch.sum(torch.tensor(all_ret['total_samples']))
            return all_ret

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample,
                  'strict_split': self.hparams.strict_split,
                  'use_sem': self.hparams.render_semantic,
                  'depth_mono': self.hparams.depth_mono,
                  'normal_mono': self.hparams.normal_mono,
                  'depth_dp_proxy': self.hparams.depth_dp_proxy,
                  'use_mask': self.hparams.use_mask,
                  'classes': self.hparams.num_classes,
                  'scale_factor': self.hparams.scale_factor}
        
        kwargs['workspace'] = os.path.join('results', self.hparams.dataset_name, self.hparams.exp_name)

        if self.hparams.dataset_name == 'kitti':
            kwargs['seq_id'] = self.hparams.kitti_seq
            kwargs['kitti_start'] = self.hparams.kitti_start
            kwargs['kitti_end'] = self.hparams.kitti_end
            kwargs['train_frames'] = (hparams.kitti_end - hparams.kitti_start + 1)

            # doubel sequence in kitti360 dataset
            kwargs['kitti_dual_seq'] = self.hparams.kitti_dual_seq
            kwargs['kitti_dual_start'] = self.hparams.kitti_dual_start
            kwargs['kitti_dual_end'] = self.hparams.kitti_dual_end
            
            val_list = []
            for i in self.hparams.kitti_test_id:
                val_list.append(int(i))
            kwargs['test_id'] = val_list
            
            # lidar if neccessary
            kwargs['lidar_proxy'] = self.hparams.kitti_lidar_proxy

        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        if not os.path.exists(os.path.join('results', self.hparams.dataset_name, self.hparams.exp_name)):
            os.makedirs(os.path.join('results', self.hparams.dataset_name, self.hparams.exp_name))

        # camera pose normalization info
        with open(os.path.join('results', self.hparams.dataset_name, self.hparams.exp_name, 'pos_trans.pkl'),
                  'wb') as f:
            pickle.dump(self.train_dataset.pos_trans, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.test_dataset = dataset(split='test', **kwargs)

        self.img_wh = self.test_dataset.img_wh

        # define additional parameters
        if self.hparams.dataset_name not in ["vr_nerf"]:
            self.register_buffer('directions', self.train_dataset.directions.to(self.device))
            self.register_buffer('poses', self.train_dataset.poses.to(self.device))

    def configure_optimizers(self):
        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                                    nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                                    nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.ckpt_load, prefixes_to_ignore=['embedding_a'])
        if self.hparams.embed_a:
            load_ckpt(self.embedding_a, self.hparams.ckpt_load, model_name='embedding_a',
                      prefixes_to_ignore=['model'])

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]

        opts = []
        self.net_opt = Adam(net_params, self.hparams.lr, eps=1e-8)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [Adam([self.dR, self.dT], 1e-6)]

        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr / self.hparams.lr_decay)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=2,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=2,
                          batch_size=None,
                          pin_memory=True)

    def training_step(self, batch, batch_nb, *args):
        
        # update hashgrid for the efficient ray sampling
        uniform_density = None
        if self.global_step % self.update_interval == 0:
            self.model.update_density_grid(0.01 * MAX_SAMPLES / 3 ** 0.5,
                                           warmup=self.global_step < self.warmup_steps)
        
        # ray sampling and rendering, get results
        results = self(batch, split='train')

        # loss parameters
        loss_kwargs = {'dataset_name': self.hparams.dataset_name,
                       'uniform_density': uniform_density,
                       'normal_ref': self.hparams.normal_ref,
                       'semantic': self.hparams.render_semantic,
                       'depth_mono': self.hparams.depth_mono,
                       'normal_mono': self.hparams.normal_mono,
                       'normal_analytic_mono': self.hparams.normal_analytic_mono,
                       'xyz_opacity': self.hparams.nerf_xyz_opacity,
                       'lidar_proxy': self.hparams.kitti_lidar_proxy,
                       'depth_dp': self.hparams.depth_dp_proxy,
                       "normal_sm": self.hparams.nerf_normal_sm,
                       'step': self.global_step}

        # calculate loss
        loss_d = self.loss(results, batch, **loss_kwargs)
        loss = sum(lo.mean() for lo in loss_d.values())
        
        # logging
        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/s_per_ray', results['total_samples'] / len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)

        for k, v in loss_d.items():
            self.log(f'train/loss_{k}', v.mean())

        if self.global_step % self.hparams.train_log_iterations == self.hparams.train_log_iterations - 1 or self.global_step == 1000:
            with torch.no_grad():
                print('[val in training]')
                w, h = self.img_wh
                # for tid in range(1):
                mean_psnr = 0
                selected_ids = np.random.choice(len(self.test_dataset), 1, replace=False).astype(np.int32)
                for tid in selected_ids:
                # if True:
                    batch = self.test_dataset[tid]
                    for i in batch:
                        if isinstance(batch[i], torch.Tensor):
                            batch[i] = batch[i].cuda()
                    results = self(batch, split='test')
                    rgb_pred = rearrange(results['rgb'], '(h w) c -> h w c', h=h)
                    rgb_gt = rearrange(batch['rgb'], '(h w) c -> h w c', h=h)

                    mean_psnr += (compute_psnr(rgb_gt, rgb_pred) / len(selected_ids))

                    if hparams.render_semantic:
                        semantic_pred = semantic2img(
                            rearrange(results['semantic'].squeeze(-1).cpu().numpy(), '(h w) -> h w', h=h),
                            self.hparams.get('num_classes', 7))
                    depth_raw = rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h)
                    depth_pred = depth2img(depth_raw)
                    normal_pred = rearrange((results['normal_pred'] + 1) / 2, '(h w) c -> h w c', h=h)
                    normal_raw = rearrange((results['normal_raw'] + 1) / 2, '(h w) c -> h w c', h=h)

                    accumulate = rearrange(results['opacity'].cpu().numpy(), '(h w) -> h w', h=h)
                    accumulate = cv2.applyColorMap((accumulate * 255).astype(np.uint8),
                                                   cv2.COLORMAP_PLASMA)

                    img_dir = os.path.join('results', hparams.dataset_name, hparams.exp_name, 'val')
                    os.makedirs(img_dir, exist_ok=True)
                    save_image(rgb_pred, os.path.join(img_dir, f'{self.global_step:0>5d}_rgb_{tid}.png'))
                    save_image(rgb_gt, os.path.join(img_dir, f'{self.global_step:0>5d}_rgb_gt_{tid}.png'))

                    if hparams.render_semantic:
                        save_image(semantic_pred, os.path.join(img_dir, f'{self.global_step:0>5d}_semantic_{tid}.png'))
                    cv2.imwrite(os.path.join(img_dir, f'{self.global_step:0>5d}_depth_{tid}.png'), depth_pred)
                    cv2.imwrite(os.path.join(img_dir, f'{self.global_step:0>5d}_accu_{tid}.png'), accumulate)
                    save_image(normal_pred, os.path.join(img_dir, f'{self.global_step:0>5d}_normal_{tid}.png'))
                    save_image(normal_raw, os.path.join(img_dir, f'{self.global_step:0>5d}_normal_raw_{tid}.png'))

                self.log('val/psnr', mean_psnr, True)

                if self.hparams.validate_all:
                    print(f'[validate all in iteration {self.global_step}]')
                    best_val_psnr = getattr(self, 'best_val_psnr', 0)
                    val_pips = LPIPSVal().cuda()
                    psnr_list = []
                    ssim_list = []
                    lpips_list = []
                    train_cam_pos = self.train_dataset.poses[:, :3, 3].reshape(-1, 3).clone().detach().cuda()
                    for tid in tqdm(range(self.test_dataset.poses.shape[0])):
                        batch = self.test_dataset[tid]
                        for i in batch:
                            if isinstance(batch[i], torch.Tensor):
                                batch[i] = batch[i].cuda()
                        with torch.no_grad():
                            # search adjacent pose
                            if self.hparams.embed_a:
                                test_cam_pos = self.test_dataset.poses[tid][:3, 3].reshape(1, 3).cuda()
                                embed_id = torch.argsort(((test_cam_pos - train_cam_pos) ** 2).sum(-1).reshape(-1))[:4]
                                embedding_a = self.embedding_a(embed_id).cuda()
                                embedding_a = embedding_a.mean(0).reshape(1, -1)
                            kwargs = {'test_time': True,
                                      'random_bg': self.hparams.random_bg,
                                      'render_rgb': self.hparams.render_rgb,
                                      'render_depth': self.hparams.render_depth,
                                      'render_normal': self.hparams.render_normal,
                                      'render_sem': self.hparams.render_semantic,
                                      'img_wh': self.test_dataset.img_wh,
                                      'T_threshold': self.hparams.T_threshold,
                                      'classes': self.hparams.num_classes}

                            if hparams.dataset_name in ['colmap', 'nerfpp', 'tnt']:
                                kwargs['exp_step_factor'] = 1 / 256
                            if hparams.use_exposure:
                                kwargs['exposure'] = batch['exposure']
                            if hparams.embed_a:
                                kwargs['embedding_a'] = embedding_a

                            poses = batch['pose']
                            directions = self.test_dataset.directions.cuda()
                            rays_o, rays_d = get_rays(directions, poses)

                            chunk_size = 131072 // 8
                            all_ret = {}
                            for i in range(0, rays_o.shape[0], chunk_size):
                                ret = render(self.model, rays_o[i:i + chunk_size], rays_d[i:i + chunk_size], **kwargs)
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

                        color_pred = rgb_pred.reshape((h, w, 3))
                        color_gt = batch['rgb'].reshape((h, w, 3))

                        psnr = compute_psnr(color_gt, color_pred)
                        ssim = compute_ssim(color_gt, color_pred)
                        lpips = val_pips(color_gt, color_pred)

                        psnr_list.append(psnr)
                        ssim_list.append(ssim)
                        lpips_list.append(lpips)

                    mean_psnr = torch.tensor(psnr_list).reshape(-1).mean().item()
                    mean_ssim = torch.tensor(ssim_list).reshape(-1).mean().item()
                    mean_lpips = torch.tensor(lpips_list).reshape(-1).mean().item()

                    wandb.log({"val_step/psnr": mean_psnr,
                               "val_step/ssim": mean_ssim,
                               "val_step/lpips": mean_lpips})

                    val_save_ckpt_path = os.path.join('ckpts', self.hparams.dataset_name, self.hparams.exp_name, 'val',
                                                      'best.pth')
                    os.makedirs(os.path.join('ckpts', self.hparams.dataset_name, self.hparams.exp_name, 'val'),
                                exist_ok=True)
                    print("best_val_psnr: ", best_val_psnr)
                    print("mean_psnr now: ", mean_psnr)
                    if best_val_psnr < mean_psnr:
                        self.best_val_psnr = mean_psnr

                        torch.save({
                            'model': self.model,
                            'embedding_a': self.embedding_a,
                        }, val_save_ckpt_path)

        for name, params in self.model.named_parameters():
            check_nan = None
            check_inf = None
            if params.grad is not None:
                check_nan = torch.any(torch.isnan(params.grad))
                check_inf = torch.any(torch.isinf(params.grad))
            if check_inf or check_nan:
                import ipdb;
                ipdb.set_trace()

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        results = self(batch, split='test')

        logs = {}
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        if self.hparams.eval_lpips:
            self.val_lpips(torch.clip(rgb_pred * 2 - 1, -1, 1),
                           torch.clip(rgb_gt * 2 - 1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()

        if not self.hparams.no_save_test:  # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred * 255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth)

        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        print(f'test/mean_PSNR: {mean_psnr}')
        self.log('test/psnr', mean_psnr)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        print(f'test/mean_SSIM: {mean_ssim}')
        self.log('test/ssim', mean_ssim)

        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            print(f'test/mean_LPIPS: {mean_lpips}')
            self.log('test/lpips_alex', mean_lpips)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    torch.manual_seed(20220806)
    torch.cuda.manual_seed_all(20220806)
    np.random.seed(20220806)
    hparams = get_opts()
    global_var._init()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename=hparams.ckpt_save.split('.')[0],
                              save_weights_only=True,
                              every_n_epochs=1,
                              save_last=True,
                              save_on_train_epoch_end=True)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]
    os.makedirs("./.wandb", exist_ok=True)
    os.makedirs(f"./.wandb/{hparams.dataset_name}", exist_ok=True)
    wandb.init(dir="./.wandb")
    logger = WandbLogger(
        project='video2game',
        save_dir=f"./.wandb/{hparams.dataset_name}",
        name=hparams.exp_name)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=32,
                      gradient_clip_val=50)

    trainer.fit(system)

    # save slimmed ckpt for the last epoch
    ckpt_ = slim_ckpt(os.path.join(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}', 'last.ckpt'),
                      save_poses=hparams.optimize_ext)
    torch.save(ckpt_, os.path.join(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}', 'last_slim.ckpt'))


