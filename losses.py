import torch
from torch import nn
import vren
import math

def compute_scale_and_shift(prediction, target):
    dr = prediction.reshape(-1, 1)
    dr = torch.cat((dr, torch.ones_like(dr).to(dr.device)), -1).unsqueeze(-1) # (N, 2, 1)
    left_part = torch.inverse(torch.sum(dr @ dr.transpose(1, 2), dim=0)).reshape(2, 2) # (2, 2)
    right_part = torch.sum(dr*target.reshape(-1, 1, 1), dim=0).reshape(2, 1)
    rs = left_part @ right_part
    rs = rs.reshape(2)
    return rs[0], rs[1]

class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)
    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]
    Outputs:
        loss: (N_rays)
    """
    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan, ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        ws_inclusive_scan, wts_inclusive_scan, ws, deltas, ts, rays_a = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan, wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None

class OpacityAggr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ws, rays_a):
        xyz_opas = vren.opacity_aggr_fw(ws, rays_a)
        ctx.save_for_backward(rays_a)
        return xyz_opas

    @staticmethod
    def backward(ctx, dxyz_opas):
        rays_a = ctx.saved_tensors[0]
        d_ws = vren.opacity_aggr_bw(dxyz_opas, rays_a)
        return d_ws, None

class ExponentialAnnealingWeight():
    def __init__(self, max, min, k):
        super().__init__()
        # 5e-2
        self.max = max
        self.min = min
        self.k = k

    def getWeight(self, Tcur):
        return max(self.min, self.max * math.exp(-Tcur*self.k))

class NeRFLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.lambda_opa = kwargs.get('lambda_opa', 2e-4)
        self.lambda_distortion = kwargs.get('lambda_distortion', 3e-4) # default
        # self.lambda_distortion = 1e-4 # for meganerf
        self.lambda_depth_mono = kwargs.get('lambda_depth_mono', 1)
        self.lambda_depth_dp = kwargs.get('lambda_depth_dp', 1)
        self.lambda_normal_mono = kwargs.get('lambda_normal_mono', 1e-3)
        self.lambda_normal_ref = kwargs.get('lambda_normal_ref', 1e-3)
        self.lambda_sky = kwargs.get('lambda_sky', 1e-1)
        self.lambda_semantic = kwargs.get('lambda_semantic', 4e-2)
        self.lambda_sparsity = kwargs.get('lambda_sparsity', 1e-4) if not kwargs['remove_sparsity'] else 0
        self.lambda_lidar_proxy = kwargs.get('lambda_lidar_proxy', 1e-3)
        self.lambda_xyz_opa = kwargs.get('lambda_xyz_opa', 3e-4)
        self.lambda_normal_sm = kwargs.get('lambda_normal_sm', 2e-4)
        self.normal_mono_skipsem = kwargs.get('normal_mono_skipsem', [])
        self.nerf_opacity_solid = kwargs.get('opacity_solid', False)
        self.sky_sem = kwargs.get('sky_sem', 4)
        # print("self.  opacity solid: ", self.nerf_opacity_solid)
        self.Annealing = ExponentialAnnealingWeight(max = 1, min = 6e-2, k = 1e-3)
        
        self.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=256)
        self.remove_sparsity = kwargs['remove_sparsity']

    def forward(self, results, target, **kwargs):
        d = {}
        
        # rgb loss
        d['rgb'] = (results['rgb']-target['rgb'])**2

        # opacity loss
        if self.nerf_opacity_solid:
            o = results['opacity']
            d['opacity'] = self.lambda_opa*((1-o)**2)

        else:
            o = results['opacity']+1e-10
            # encourage opacity to be either 0 or 1 to avoid floater
            d['opacity'] = self.lambda_opa*(-o*torch.log(o))

        if kwargs.get('xyz_opacity', False):
            xyz_opas = OpacityAggr.apply(results["ws"], results["rays_a"])
            _o = xyz_opas + 1e-10
            d['xyzs_opacity'] = self.lambda_xyz_opa * (-_o*torch.log(_o))

        # normal smoothness loss if necessary
        if kwargs.get('normal_sm', False):
            l1_loss = torch.abs(results["normals_raw_eps"] - results["normals_raw"])  # (n, 3)
            cos_loss = -(results["normals_raw_eps"] * results["normals_raw"])  # (n, 3)
            d['normal_sm'] = self.lambda_normal_sm * (l1_loss + 0.1 * cos_loss)

        # sparsity loss
        if not self.remove_sparsity:
            d['sparsity'] = self.lambda_sparsity*(1.0-torch.exp(-0.01*results['sigma_sparse']).mean())
            d['sparsity'] = self.lambda_sparsity * (results['sigma_sparse'].mean())
        
        # distortion loss
        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion * \
            DistortionLoss.apply(results['ws'], results['deltas'],
                                    results['ts'], results['rays_a'])

        # normal ref loss
        if kwargs.get('normal_ref', False):
            d['normal_ref'] = self.lambda_normal_ref * results['Rp'] # for ref-nerf model

        # normal prediction loss: prediction normal <--> normal prior
        if kwargs.get('normal_mono', False):
            normal_pred = results['normal_pred']
            normal_gt = target['normal']
            
            if kwargs.get('semantic', False) and len(self.normal_mono_skipsem) != 0:
                sem_mask = torch.ones_like(target['label']).bool()
                for sem_i in self.normal_mono_skipsem:
                    sem_mask = torch.logical_and(sem_mask, (target['label']!=sem_i).reshape(-1))
                normal_pred = normal_pred[sem_mask]
                normal_gt = normal_gt[sem_mask]
            
            l1_loss = torch.abs(normal_pred - normal_gt) #(n, 3)
            cos_loss = -(normal_pred * normal_gt) #(n, 3)
            d['normal_mono'] = self.lambda_normal_mono * (l1_loss + 0.1 * cos_loss)
        
        # analytic normal loss: analytic normal from gradients <--> normal prior
        if kwargs.get('normal_analytic_mono', False):
            normal_pred = results['normal_raw']
            normal_gt = target['normal']

            if kwargs.get('semantic', False) and len(self.normal_mono_skipsem) != 0:
                sem_mask = torch.ones_like(target['label']).bool()
                for sem_i in self.normal_mono_skipsem:
                    sem_mask = torch.logical_and(sem_mask, (target['label'] != sem_i).reshape(-1))
                normal_pred = normal_pred[sem_mask]
                normal_gt = normal_gt[sem_mask]

            l1_loss = torch.abs(normal_pred - normal_gt)  # (n, 3)
            cos_loss = -(normal_pred * normal_gt)  # (n, 3)
            d['normal_analytic_mono'] = self.lambda_normal_mono * (l1_loss + 0.1 * cos_loss)

        # semantic loss
        if kwargs.get('semantic', False):
            d['CELoss'] = self.lambda_semantic*self.CrossEntropyLoss(results['semantic'], target['label'])
            sky_mask = torch.where(target['label']==4, 1., 0.)
            d['sky_depth'] = self.lambda_sky*sky_mask*torch.exp(-results['depth'])

        # depth prior loss
        if kwargs.get('depth_mono', False): # for kitti360 dataset
            if kwargs.get('semantic', False):
                sky_mask = (target['label']!=self.sky_sem).reshape(-1)
                pred_depth = results['depth'][sky_mask]
                gt_depth = target['depth'][sky_mask]
                w, q = compute_scale_and_shift(pred_depth, gt_depth)
                d['depth_mono'] = self.lambda_depth_mono * (((w*pred_depth+q) - gt_depth)**2).sum()

            else:
                w, q = compute_scale_and_shift(results['depth'], target['depth'] )
                d['depth_mono'] = self.lambda_depth_mono * (((w*results['depth']+q) - target['depth'])**2).sum()
        
        # binocular depth from deepruner 
        if kwargs.get('depth_dp', False): # for kitti360 dataset
            if kwargs.get('semantic', False):
                sky_mask = (target['label']!=self.sky_sem).reshape(-1)
                pred_depth = results['depth'][sky_mask]
                gt_depth = target['depth_dp'][sky_mask]
                d['depth_dp'] = self.lambda_depth_dp * torch.abs(pred_depth - gt_depth).mean()

            else:
                d['depth_dp'] = self.lambda_depth_dp * torch.abs(results['depth'] - target['depth_dp']).mean()
        
        # lidar loss if necessary    
        if kwargs.get('lidar_proxy', False):
            lidarp = target['lidarp']
            lidarp_mask = target['lidarp_mask']
            render_points = results['points']
            d['lidar_proxy'] = self.lambda_lidar_proxy * torch.abs(lidarp - render_points)[lidarp_mask].mean()
            

        flag = 0
        for (i, n) in d.items():
            if torch.any(torch.isnan(n)):
                print(f'nan in d[{i}]')
                print(f'max: {torch.max(n)}')
                flag = 1
            
        return d

 