from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from opt import get_opts
import os
import numpy as np
import cv2

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays
import trimesh

# models
from models.global_var import global_var
from models.tcngp import TCNGP

# optimizer, losses
# from apex.optimizers import FusedAdam
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
)

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger

# nvdiffrast
import nvdiffrast.torch as dr

# misc
import warnings; warnings.filterwarnings("ignore")
import pickle5 as pickle
import tqdm
from meshutils import *

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

def convert_normal(normal, pose_c2w):
    R_w2c = pose_c2w[:3, :3].T
    normal_cam = normal @ R_w2c.T
    return normal_cam
    
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

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)      

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps))

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

class BakingSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
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
        self.color_model = TCNGP(scale=hparams.tcngp_scale, contraction_type=hparams.tcngp_contraction_type,
                                 sphere_scale=(
                                 (hparams.sphere_scale_x_n, hparams.sphere_scale_y_n, hparams.sphere_scale_z_n),
                                 (hparams.sphere_scale_x_p, hparams.sphere_scale_y_p, hparams.sphere_scale_z_p)),
                                 ngp_params=tcngp_params, specular_dim_mlp=hparams.baking_specular_dim_mlp, per_level_scale=hparams.baking_per_level_scale,specular_dim=hparams.baking_specular_dim,color_net_params=color_net_params).cuda()
        self.color_model.train()
        
        # nvdiffrast rasterizer
        self.glctx = dr.RasterizeCudaContext()
        
        # dataset setup
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample,
                  'use_sem': False,
                  'depth_mono': False,
                  'normal_mono': False,
                  'use_mask': self.hparams.use_mask,
                  'strict_split': self.hparams.strict_split}
        
        kwargs['workspace'] = os.path.join('results', hparams.dataset_name, hparams.exp_name)

        if self.hparams.dataset_name == 'kitti':
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

        self.train_dataset = dataset(split='train', **kwargs)
        self.img_wh = self.train_dataset.img_wh
        
        # define additional parameters for dataset
        if self.hparams.dataset_name not in ["vr_nerf"]:
            self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))
        self.register_buffer('mvps', self.train_dataset.mvps.to(self.device))
   
        self.train_dataset.split = 'train'
        
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy
        self.train_dataset.batch_size = self.hparams.batch_size
        
        save_path = os.path.join('results', hparams.dataset_name, hparams.exp_name, 'baking', f'{"base" if hparams.workspace is None else f"{hparams.workspace}"}')
        os.makedirs(save_path, exist_ok=True)
        self.workspace = save_path
        
        # sequentially load cascaded meshes
        vertices = []
        triangles = []
        v_cumsum = [0]
        f_cumsum = [0]
    
        if len(hparams.load_mesh_paths) > 0:
            self.mesh_cascades = 0
            for mesh_path in hparams.load_mesh_paths:
                print(f"loading mesh {mesh_path}")
                mesh = trimesh.load(mesh_path, force='mesh', skip_material=True, process=False)
                print(f"mesh {mesh_path} loaded")
                vertices.append(mesh.vertices)
                triangles.append(mesh.faces + v_cumsum[-1])
                
                v_cumsum.append(v_cumsum[-1] + mesh.vertices.shape[0])
                f_cumsum.append(f_cumsum[-1] + mesh.faces.shape[0])
                self.mesh_cascades += 1
        elif hparams.load_mesh_path is not None:
            mesh = trimesh.load(hparams.load_mesh_path, force='mesh', skip_material=True, process=False)
            
            vertices.append(mesh.vertices)
            triangles.append(mesh.faces + v_cumsum[-1])
            
            v_cumsum.append(v_cumsum[-1] + mesh.vertices.shape[0])
            f_cumsum.append(f_cumsum[-1] + mesh.faces.shape[0])
            self.mesh_cascades = 1
        else:
            assert False, "need to assign a mesh path to load"

        # setup skydome if neccessary
        if hparams.skydome:
            self.mesh_cascades += 1
            if hparams.dataset_name == 'kitti':
                sky_dome = trimesh.primitives.Sphere(radius=5, center=(0, 0, 0), subdivisions=8)
                sky_dome_vert = np.array(sky_dome.vertices).reshape(-1, 3)
                sky_dome_vert[:, -1] = sky_dome.vertices[:, -1] / 5
                with open(hparams.center_pos_trans,  'rb') as f:
                    pos_trans = pickle.load(f)
                    center = pos_trans['center']
                    scale = pos_trans['scale']
                    forward = pos_trans['forward']
                    print('center pos_trans: ', pos_trans)
                sky_dome_vert = (sky_dome_vert + forward * 0.5) * scale + center
                with open(os.path.join(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}'), 'pos_trans.pkl'),  'rb') as f:
                    pos_trans = pickle.load(f)
                    center = pos_trans['center']
                    scale = pos_trans['scale']
                    forward = pos_trans['forward']
                    print('local pos_trans: ', pos_trans)
                sky_dome_vert = (sky_dome_vert - center) / scale - 0.5 * forward
                sky_dome_face = np.array(sky_dome.faces).reshape(-1, 3).astype(np.int32)

            elif hparams.dataset_name == 'colmap':
                sky_dome = trimesh.primitives.Sphere(radius=8, center=(0, 0, 0), subdivisions=8)
                sky_dome_vert = np.array(sky_dome.vertices).reshape(-1, 3)
                sky_dome_face = np.array(sky_dome.faces).reshape(-1, 3).astype(np.int32)
                
            vertices.append(sky_dome_vert)
            triangles.append(sky_dome_face + v_cumsum[-1])
            
            _vertices = np.concatenate(vertices, axis=0)
            _triangles = np.concatenate(triangles, axis=0)

            if self.train_dataset is not None and hparams.dataset_name != 'colmap':
                @torch.no_grad()
                def mark_unseen_triangles(vertices, triangles, mvps, H, W):
                    # vertices: coords in world system
                    # mvps: [B, 4, 4]
                    print("mvps: ", mvps.shape)
                    if isinstance(vertices, np.ndarray):
                        vertices = torch.from_numpy(vertices).contiguous().float().cuda()
                    
                    if isinstance(triangles, np.ndarray):
                        triangles = torch.from_numpy(triangles).contiguous().int().cuda()

                    mask = torch.zeros_like(triangles[:, 0]) # [M,], for face.

                    glctx = dr.RasterizeCudaContext()
                    
                    for i in tqdm.tqdm(range(mvps.shape[0])):
                        mvp = mvps[i]
                        vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp.cuda(), 0, 1)).float().unsqueeze(0) # [1, N, 4]

                        # ENHANCE: lower resolution since we don't need that high?
                        rast, _ = dr.rasterize(glctx, vertices_clip, triangles, (H, W)) # [1, H, W, 4]
                        
                        # collect the triangle_id (it is offseted by 1)
                        trig_id = rast[..., -1].long().view(-1) - 1
                        
                        # no need to accumulate, just a 0/1 mask.
                        mask[trig_id] += 1 # wrong for duplicated indices, but faster.
                        # mask.index_put_((trig_id,), torch.ones(trig_id.shape[0], device=device, dtype=mask.dtype), accumulate=True)
                    mask = (mask == 0) # unseen faces by all cameras

                    print(f'[mark unseen trigs] {mask.sum()} from {mask.shape[0]}')
                    
                    return mask # [N]
                # sky_dome_vert, sky_dome_face
                visibility_mask = mark_unseen_triangles(_vertices, _triangles, self.train_dataset.mvps, self.train_dataset.H, self.train_dataset.W).cpu().numpy()

                visibility_mask = visibility_mask[-sky_dome.faces.shape[0]:]
                sky_dome_vert, sky_dome_face = remove_masked_trigs(sky_dome_vert, sky_dome.faces, visibility_mask)
            
            vertices[-1] = sky_dome_vert
            triangles[-1] = sky_dome_face + v_cumsum[-1]
            
            v_cumsum.append(v_cumsum[-1] + sky_dome_vert.shape[0])
            f_cumsum.append(f_cumsum[-1] + sky_dome_face.shape[0])
            
        vertices = np.concatenate(vertices, axis=0)
        triangles = np.concatenate(triangles, axis=0)
            
        self.v_cumsum = np.array(v_cumsum)
        self.f_cumsum = np.array(f_cumsum)

        # must put to cuda manually, we don't want these things in the model as buffers...
        self.vertices = torch.from_numpy(vertices).float().cuda() # [N, 3]
        self.triangles = torch.from_numpy(triangles).int().cuda()
        
        # setup loss function
        if self.hparams.baking_rgb_loss == 'sml1':
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
        elif self.hparams.baking_rgb_loss == 'rl1':
            def rl1(pred, gt):
                drgb = gt - pred
                c = 0.2
                loss_rgb = torch.log(0.5 * ((drgb / c) ** 2) + 1).mean()
                return loss_rgb
            self.criterion = rl1
            
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        
        self.hparams.num_epochs = int(np.ceil(self.hparams.baking_iters / self.train_dataset.epoch_length_baking))

    def export_stage1(self):


        resolution = self.hparams.texture_size

        self._export_stage1(resolution, resolution)


    @torch.no_grad()
    def _export_stage1(self, h0=2048, w0=2048):
        v = self.vertices.detach()
        f = self.triangles.detach()
        
        # store info
        build_dict = {
            "cascades": self.mesh_cascades,
            "v_cumsum": self.v_cumsum,
            "f_cumsum": self.f_cumsum,
            "vertices": v,
            "triangles": f,
            "ssaa": self.hparams.ssaa,
            "h0": h0,
            "w0": w0,
            "skydome": self.hparams.skydome,
        }
        os.makedirs(self.workspace, exist_ok=True)
        with open(os.path.join(self.workspace, 'build_dict.pkl'), 'wb') as f_pkl:
            pickle.dump(build_dict, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)

    
    # phase 2
    def render_stage1(self, rays_d, mvp, h0, w0, shading='full', mesh_normals=None, pose=None, K=None, **kwargs):

        prefix = rays_d.shape[:-1]
        rays_d = rays_d.contiguous().view(-1, 3).cuda()
        device = rays_d.device
        
        # do super-sampling
        if self.hparams.ssaa > 1:
            h = int(h0 * self.hparams.ssaa)
            w = int(w0 * self.hparams.ssaa)
            # interpolate rays_d when ssaa > 1 ...
            dirs = rays_d.view(h0, w0, 3)
            dirs = scale_img_hwc(dirs, (h, w), mag='nearest').view(-1, 3).contiguous()
        else:
            h, w = h0, w0
            dirs = rays_d.contiguous()
            
        dirs = dirs.cuda()
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        
        bg_color = 1

        results = {}
        
        vertices = self.vertices.cuda()
        self.triangles = self.triangles.cuda()
        
        # vertices in clip space
        vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp, 0, 1)).float().unsqueeze(0) # [1, N, 4]

        rast, _ = dr.rasterize(self.glctx, vertices_clip, self.triangles, (h, w))
        # we need to flip in order to align with gt rgb image
        frast = rast.flip([1]).cuda()
        
        # interpolate using uv in rasterization result to get global location of each pixel
        xyzs, _ = dr.interpolate(vertices.unsqueeze(0), frast, self.triangles) # [1, H, W, 3]
        mask, _ = dr.interpolate(torch.ones_like(vertices[:, :1]).unsqueeze(0).cuda(), frast, self.triangles) # [1, H, W, 1]
        mask_flatten = (mask > 0).view(-1)
        xyzs = xyzs.view(-1, 3).cuda()

        # get rgb of each pixel calculated from NGP and MLP 
        rgbs = torch.zeros(h * w, 3, device=device, dtype=torch.float32)
        self.color_model = self.color_model.cuda()
        if mask_flatten.any():
            with torch.cuda.amp.autocast(enabled=False):
                mask_rgbs, masked_specular = self.color_model(
                    xyzs[mask_flatten].detach().cuda(), 
                    dirs[mask_flatten].cuda(), None, shading)

            rgbs[mask_flatten] = mask_rgbs.float()

        rgbs = rgbs.view(1, h, w, 3).flip([1])
        alphas = mask.float().detach().flip([1])
        
        # nvdiffrast antialias
        alphas = dr.antialias(alphas, rast, vertices_clip, self.triangles, pos_gradient_boost=self.hparams.pos_gradient_boost).flip([1]).squeeze(0).clamp(0, 1)
        rgbs = dr.antialias(rgbs, rast, vertices_clip, self.triangles, pos_gradient_boost=self.hparams.pos_gradient_boost).flip([1]).squeeze(0).clamp(0, 1)

        # alpha blending
        image = alphas * rgbs 
        T = 1 - alphas

        # ssaa
        if self.hparams.ssaa > 1:
            image = scale_img_hwc(image, (h0, w0))
            T = scale_img_hwc(T, (h0, w0))
        
        image = image + T * bg_color
        image = image.view(*prefix, 3)
            
        results['image'] = image
        results['mask'] = (mask > 0).reshape(-1)
        
        return results
        
    def forward(self, samples):
        
        loss = 0
        metric_dict = {
            'loss_rgb': 0,
            'psnr': 0,
        }

        for batch in samples:
            index = batch['index']
            poses = self.poses[index].cuda()
            if self.hparams.dataset_name not in ["vr_nerf"]:
                directions = self.directions.cuda()
                _, rays_d = get_rays(directions, poses)
            else:
                rays_d = self.train_dataset.rays_d[index].float().cuda()

            images = batch['rays'].cuda() # [N, 3/4]
            H, W = self.img_wh[1], self.img_wh[0]
            N, C = images.shape

            def srgb_to_linear(x):
                return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
            if self.hparams.color_space == 'linear':
                images[..., :3] = srgb_to_linear(images[..., :3])

            bg_color = 1
            gt_rgb = images
            
            if self.global_step < self.hparams.diffuse_step:
                shading = 'diffuse'
            else:
                shading = 'full'

            if self.hparams.baking_antialias and hasattr(self.train_dataset, 'mvp_permute'):
                mvp, directions = self.train_dataset.mvp_permute(index)
                mvp = mvp.reshape(4, 4).cuda()
                _, rays_d = get_rays(directions.cuda(), poses)
            else:
                mvp = self.mvps[index].reshape(4, 4).cuda() # [4, 4]
            
            outputs = self.render_stage1(rays_d, mvp, H, W, shading=shading, pose=self.poses[index], K=self.train_dataset.K, **vars(self.hparams))

            pred_rgb = outputs['image']
            
            if batch.get('mask', None) is not None:        
                pred_rgb = pred_rgb * batch['mask'] + (gt_rgb * ~batch['mask']).detach()
                
            if self.global_step % hparams.train_log_iterations == hparams.train_log_iterations - 1 or self.global_step == 100:
                os.makedirs(os.path.join(self.workspace, 'log'), exist_ok=True)
                with torch.no_grad():
                    render_color = np.array(torch.clip(pred_rgb.detach().clone().cpu(), 0, 1)*255).reshape(H, W, 3)
                    gt_color = np.array(torch.clip(gt_rgb.detach().clone().cpu(), 0, 1)*255).reshape(H, W, 3)

                    cv2.imwrite(os.path.join(self.workspace, 'log', f'rgb_rd_step{self.global_step}_idx{index}.jpg'), cv2.cvtColor(render_color, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(self.workspace, 'log', f'rgb_gt_step{self.global_step}_idx{index}.jpg'), cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR))
                    
            loss_rgb = self.criterion(pred_rgb, gt_rgb).mean(-1).mean() # [H, W]
            psnr = self.psnr(pred_rgb.cpu(), gt_rgb.cpu())
            metric_dict['psnr'] += psnr

            loss += loss_rgb
            metric_dict['loss_rgb'] += loss_rgb
                
        loss = loss / len(samples)   
        for k, v in metric_dict.items():
            metric_dict[k] = v / len(samples)   
    
        return loss, metric_dict

    def setup(self, stage):
        pass
        

    def configure_optimizers(self):
        opts = []

        self.net_opt = Adam([{'params': self.color_model.encoder_color.parameters(), 'initial_lr': self.hparams.lr}], self.hparams.lr, eps=1e-8)
        self.net_opt.add_param_group({
            'params': self.color_model.color_net.parameters(),
            'initial_lr': self.hparams.lr,
            'lr': self.hparams.lr,
            'weight_decay': self.hparams.weight_decay,
            'eps': 1e-8
        })
        self.net_opt.add_param_group({
            'params': self.color_model.specular_net.parameters(),
            'initial_lr': self.hparams.lr,
            'lr': self.hparams.lr,
            'weight_decay': self.hparams.weight_decay,
            'eps': 1e-8
        })
        opts += [self.net_opt]
        
        sches = []
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/self.hparams.lr_decay,
                                    last_epoch=-1)  
        sches += [net_sch]

        return opts, sches

    def train_dataloader(self):
        self.train_dataset.ray_sampling_strategy = 'single_image'
            
        return DataLoader(self.train_dataset,
                          num_workers=2,
                          batch_size=None,
                          shuffle=True,
                          pin_memory=True)

    def training_step(self, batch, batch_nb, *args):

        self.color_model = self.color_model.cuda()
            
        loss, metric_dict = self(batch)    

        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        
        for k, v in metric_dict.items():
            self.log(f'train/{k}', v)

        return loss
     
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
    
class ExportCallback(Callback):
    def __init__(self):
        pass
    def on_fit_end(self, trainer, pl_module):
        pl_module.export_stage1()

if __name__ == '__main__':
    torch.manual_seed(20220806)
    torch.cuda.manual_seed_all(20220806)
    np.random.seed(20220806)
    hparams = get_opts()
    global_var._init()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')

    os.makedirs("./wandb", exist_ok=True)
    logger = WandbLogger(
        project='video2game_baking',
        save_dir=f"./wandb",
        name=hparams.exp_name)

    system = BakingSystem(hparams)

    hparams.num_epochs = system.hparams.num_epochs

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/baking/{"base" if hparams.workspace is None else f"{hparams.workspace}"}',
                              filename=hparams.ckpt_save.split('.')[0],
                              save_weights_only=True,
                              every_n_epochs=100,
                              save_last=True,
                              save_on_train_epoch_end=True)
    
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1), ExportCallback()]

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=32,
                      gradient_clip_val=50,
                      reload_dataloaders_every_n_epochs=1)

    trainer.fit(system)

