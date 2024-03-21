import torch
import vren
from torch.cuda.amp import custom_fwd, custom_bwd
from torch_scatter import segment_csr
import torch.functional as F
from einops import rearrange
from .global_var import global_var

class RayAABBIntersector(torch.autograd.Function):
    """
    Computes the intersections of rays and axis-aligned voxels.

    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        centers: (N_voxels, 3) voxel centers
        half_sizes: (N_voxels, 3) voxel half sizes
        max_hits: maximum number of intersected voxels to keep for one ray
                  (for a cubic scene, this is at most 3*N_voxels^(1/3)-2)

    Outputs:
        hits_cnt: (N_rays) number of hits for each ray
        (followings are from near to far)
        hits_t: (N_rays, max_hits, 2) hit t's (-1 if no hit)
        hits_voxel_idx: (N_rays, max_hits) hit voxel indices (-1 if no hit)
    """
    @staticmethod
    # @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, center, half_size, max_hits):
        return vren.ray_aabb_intersect(rays_o, rays_d, center, half_size, max_hits)


class RaySphereIntersector(torch.autograd.Function):
    """
    Computes the intersections of rays and spheres.

    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        centers: (N_spheres, 3) sphere centers
        radii: (N_spheres, 3) radii
        max_hits: maximum number of intersected spheres to keep for one ray

    Outputs:
        hits_cnt: (N_rays) number of hits for each ray
        (followings are from near to far)
        hits_t: (N_rays, max_hits, 2) hit t's (-1 if no hit)
        hits_sphere_idx: (N_rays, max_hits) hit sphere indices (-1 if no hit)
    """
    @staticmethod
    # @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, center, radii, max_hits):
        return vren.ray_sphere_intersect(rays_o, rays_d, center, radii, max_hits)


class RayMarcher(torch.autograd.Function):
    """
    March the rays to get sample point positions and directions.

    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) normalized ray directions
        hits_t: (N_rays, 2) near and far bounds from aabb intersection
        density_bitfield: (C*G**3//8)
        cascades: int
        scale: float
        exp_step_factor: the exponential factor to scale the steps
        grid_size: int
        max_samples: int
        mean_samples: int, mean total samples per batch

    Outputs:
        rays_a: (N_rays) ray_idx, start_idx, N_samples
        xyzs: (N, 3) sample positions
        dirs: (N, 3) sample view directions
        deltas: (N) dt for integration
        ts: (N) sample ts
    """
    @staticmethod
    # @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, hits_t,
                density_bitfield, cascades, scale, exp_step_factor,
                grid_size, max_samples):
        # noise to perturb the first sample of each ray
        noise = torch.rand_like(rays_o[:, 0])

        rays_a, xyzs, dirs, deltas, ts, counter = \
            vren.raymarching_train(
                rays_o, rays_d, hits_t,
                density_bitfield, cascades, scale,
                exp_step_factor, noise, grid_size, max_samples)

        total_samples = counter[0] # total samples for all rays
        # remove redundant output
        xyzs = xyzs[:total_samples]
        dirs = dirs[:total_samples]
        deltas = deltas[:total_samples]
        ts = ts[:total_samples]

        ctx.save_for_backward(rays_a, ts)

        return rays_a, xyzs, dirs, deltas, ts, total_samples

    @staticmethod
    # @custom_bwd
    def backward(ctx, dL_drays_a, dL_dxyzs, dL_ddirs,
                 dL_ddeltas, dL_dts, dL_dtotal_samples):
        rays_a, ts = ctx.saved_tensors
        segments = torch.cat([rays_a[:, 1], rays_a[-1:, 1]+rays_a[-1:, 2]])
        dL_drays_o = segment_csr(dL_dxyzs, segments)
        dL_drays_d = \
            segment_csr(dL_dxyzs*rearrange(ts, 'n -> n 1')+dL_ddirs, segments)

        return dL_drays_o, dL_drays_d, None, None, None, None, None, None, None


class VolumeRenderer(torch.autograd.Function):
    """
    Volume rendering with different number of samples per ray
    Used in training only

    Inputs:
        sigmas: (N)
        rgbs: (N, 3)
        normals_pred: (N, 3)
        sems: (N, classes)
        deltas: (N)
        ts: (N)
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]
        T_threshold: float, stop the ray if the transmittance is below it

    Outputs:
        opacity: (N_rays)
        depth: (N_rays)
        rgb: (N_rays, 3)
        normal_pred: (N_rays, 3)
        sem: (N_rays, classes)
    """
    @staticmethod
    # @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, normals_pred, sems, deltas, ts, rays_a, T_threshold, classes):
        total_samples, opacity, depth, rgb, normal_pred, sem, ws = \
            vren.composite_train_fw(sigmas, rgbs, normals_pred, sems, deltas, ts,
                                    rays_a, T_threshold, classes)
        ctx.save_for_backward(sigmas, rgbs, normals_pred, deltas, ts, rays_a,
                              opacity, depth, rgb, normal_pred, ws)
        ctx.T_threshold = T_threshold
        ctx.classes = classes
        return total_samples.sum(), opacity, depth, rgb, normal_pred, sem, ws

    @staticmethod
    # @custom_bwd
    def backward(ctx, dL_dtotal_samples, dL_dopacity, dL_ddepth, dL_drgb, dL_dnormal_pred, dL_dsem, dL_dws):
        sigmas, rgbs, normals_pred, deltas, ts, rays_a, \
        opacity, depth, rgb, normal_pred, ws = ctx.saved_tensors
        dL_dsigmas, dL_drgbs, dL_dnormals_pred, dL_dsems = \
            vren.composite_train_bw(dL_dopacity, dL_ddepth,
                                    dL_drgb, dL_dnormal_pred, dL_dsem, dL_dws,
                                    sigmas, rgbs, normals_pred, ws, deltas, ts,
                                    rays_a,
                                    opacity, depth, rgb, normal_pred,
                                    ctx.T_threshold, ctx.classes)
        
        return dL_dsigmas, dL_drgbs, dL_dnormals_pred, dL_dsems, None, None, None, None, None


class VolumeRendererDN(torch.autograd.Function):
    """
    Volume rendering with different number of samples per ray
    Used in training only

    Inputs:
        sigmas: (N)
        rgbs: (N, 3)
        normals_raw: (N, 3)
        normals_pred: (N, 3)
        sems: (N, classes)
        deltas: (N)
        ts: (N)
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]
        T_threshold: float, stop the ray if the transmittance is below it

    Outputs:
        opacity: (N_rays)
        depth: (N_rays)
        rgb: (N_rays, 3)
        normal_raw: (N_rays, 3)
        normal_pred: (N_rays, 3)
        sem: (N_rays, classes)
    """

    @staticmethod
    # @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, normals_raw, normals_pred, sems, deltas, ts, rays_a, T_threshold, classes):
        total_samples, opacity, depth, rgb, normal_raw, normal_pred, sem, ws = \
            vren.composite_train_dn_fw(sigmas, rgbs, normals_raw, normals_pred, sems, deltas, ts,
                                    rays_a, T_threshold, classes)
        ctx.save_for_backward(sigmas, rgbs, normals_raw, normals_pred, deltas, ts, rays_a,
                              opacity, depth, rgb, normal_raw, normal_pred, ws)
        ctx.T_threshold = T_threshold
        ctx.classes = classes
        return total_samples.sum(), opacity, depth, rgb, normal_raw, normal_pred, sem, ws

    @staticmethod
    # @custom_bwd
    def backward(ctx, dL_dtotal_samples, dL_dopacity, dL_ddepth, dL_drgb, dL_dnormal_raw, dL_dnormal_pred, dL_dsem, dL_dws):
        sigmas, rgbs, normals_raw, normals_pred, deltas, ts, rays_a, \
        opacity, depth, rgb, normal_raw, normal_pred, ws = ctx.saved_tensors
        dL_dsigmas, dL_drgbs, dL_dnormals_raw, dL_dnormals_pred, dL_dsems = \
            vren.composite_train_dn_bw(dL_dopacity, dL_ddepth,
                                    dL_drgb, dL_dnormal_raw, dL_dnormal_pred, dL_dsem, dL_dws,
                                    sigmas, rgbs, normals_raw, normals_pred, ws, deltas, ts,
                                    rays_a,
                                    opacity, depth, rgb, normal_raw, normal_pred,
                                    ctx.T_threshold, ctx.classes)

        return dL_dsigmas, dL_drgbs, dL_dnormals_raw, dL_dnormals_pred, dL_dsems, None, None, None, None, None


class RefLoss(torch.autograd.Function):
    @staticmethod
    # @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, normals_diff, normals_ori, deltas, ts, rays_a, T_threshold):
        loss_o, loss_p = \
            vren.composite_refloss_fw(sigmas, normals_diff, normals_ori, deltas, ts,
                                    rays_a, T_threshold)
        ctx.save_for_backward(sigmas, normals_diff, normals_ori, deltas, ts, rays_a,
                              loss_o, loss_p)
        ctx.T_threshold = T_threshold
        return loss_o, loss_p

    @staticmethod
    # @custom_bwd
    def backward(ctx, dL_dloss_o, dL_dloss_p):
        
        sigmas, normals_diff, normals_ori, deltas, ts, rays_a, \
        loss_o, loss_p = ctx.saved_tensors
        dL_dsigmas, dL_dnormals_diff, dL_dnormals_ori = \
            vren.composite_refloss_bw(dL_dloss_o, dL_dloss_p,
                                    sigmas, normals_diff, normals_ori, deltas, ts,
                                    rays_a,
                                    loss_o, loss_p,
                                    ctx.T_threshold)
        if torch.any(torch.isnan(dL_dsigmas)) or torch.any(torch.isinf(dL_dsigmas)):
            print('dL_dsigmas contains nan or inf')
        if torch.any(torch.isnan(dL_dnormals_diff)) or torch.any(torch.isinf(dL_dnormals_diff)):
            print('dL_dnormals_diff contains nan or inf')
        if torch.any(torch.isnan(dL_dnormals_ori)) or torch.any(torch.isinf(dL_dnormals_ori)):
            print('dL_dnormals_ori contains nan or inf')
        # global_var.set_value('log_dL_dsigmas', dL_dsigmas)
        # global_var.set_value('log_dL_dnormals_diff', dL_dnormals_diff)
        # global_var.set_value('log_dL_dnormals_ori', dL_dnormals_ori)
        return None, dL_dnormals_diff, dL_dnormals_ori, None, None, None, None

class TruncExp(torch.autograd.Function):
    @staticmethod
    # @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    # @custom_bwd
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-7, 7))
    
class ReLU(torch.autograd.Function):
    @staticmethod
    # @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        res = torch.zeros_like(x).cuda()
        mask = x>0
        res[mask] = x[mask]
        ctx.save_for_backward(x, mask)
        return res

    @staticmethod
    # @custom_bwd
    def backward(ctx, dL_dout):
        x, mask = ctx.saved_tensors
        dL_dx = torch.zeros_like(x).cuda()+1e-6
        dL_dx[mask] = dL_dout[mask]
        return dL_dx

class TruncTanh(torch.autograd.Function):
    @staticmethod
    # @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        func = torch.nn.Tanh()
        return func(x)

    @staticmethod
    # @custom_bwd
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        func = torch.nn.Tanh()
        return dL_dout * (1-func(x.clamp(-15, 15))**2)
    

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]).cuda(), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples).cuda()
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).cuda()

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, classes=7):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, n]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists: 1.-torch.exp(-raw*dists)
    
    sigmas = raw[..., 0]
    rgbs = raw[..., 1:4]
    normals_raw = raw[..., 4:7]
    normals_pred = raw[..., 7:10]
    sems = raw[..., 10:]
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).cuda()], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(sigmas.shape) * raw_noise_std

    alpha = raw2alpha(sigmas + noise, dists)  # [N_rays, N_samples]
    
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.-alpha + 1e-10], -1), -1)[:, :-1] # [N_rays, N_samples]
    opacity = torch.sum(weights, -1) # [N_rays]
    rgb_map = torch.sum(weights[...,None] * rgbs, -2)  # [N_rays, 3]
    normal_raw = torch.sum(weights[...,None] * normals_raw, -2)  # [N_rays, 3]
    normal_pred = torch.sum(weights[...,None] * normals_pred, -2)  # [N_rays, 3]
    sem = torch.sum(weights[...,None] * sems, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    # disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))

    return opacity, rgb_map, normal_raw, normal_pred, sem, weights, depth_map