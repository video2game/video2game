import torch
import torch.nn.functional as F
from models.custom_functions import \
    RayAABBIntersector, RayMarcher, RefLoss, VolumeRendererDN
from einops import rearrange
import vren
from models.volume_render import volume_render
from models.const import NEAR_DISTANCE, MAX_SAMPLES

def render(model, rays_o, rays_d, **kwargs):
    """
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)

    Inputs:
        model: NGP
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions

    Outputs:
        result: dictionary containing final rgb and depth
    """
    
    rays_o = rays_o.contiguous()
    rays_d = rays_d.contiguous()
    
    # calculate the intersection of rays with hashgrid AABB
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE

    # rendering
    if kwargs.get('test_time', False):
        render_func = __render_rays_test
    else:
        render_func = __render_rays_train

    results = render_func(model, rays_o, rays_d, hits_t, **kwargs)
    for k, v in results.items():
        if kwargs.get('to_cpu', False):
            v = v.cpu()
            if kwargs.get('to_numpy', False):
                v = v.numpy()
        results[k] = v
        
    return results


def __render_rays_train(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by
    1. March the rays along their directions, querying @density_bitfield
       to skip empty space, and get the effective sample points (where
       there is object)
    2. Infer the NN at these positions and view directions to get properties
       (currently sigmas and rgbs)
    3. Use volume rendering to combine the result (front to back compositing
       and early stop the ray if its transmittance is below a threshold)
    """
    
    # ray marching to sample points
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}

    max_samples = kwargs.get('max_samples', MAX_SAMPLES)
    with torch.no_grad():
        rays_a, xyzs, dirs, results['deltas'], results['ts'], total_samples = \
            RayMarcher.apply(
                rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
                model.cascades, model.scale,
                exp_step_factor, model.grid_size, max_samples)
    results['total_samples'] = total_samples
    
    for k, v in kwargs.items(): # supply additional inputs, repeated per ray
        if isinstance(v, torch.Tensor):
            kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)
    
    # calculate for each sampled point
    sigmas, rgbs, normals_raw, normals_pred, sems, _ = model(xyzs, dirs, **kwargs)
    
    results['sigma'] = sigmas
    results['xyzs'] = xyzs
    results['rays_a'] = rays_a
    
    # sample points for sparsity loss
    xyz_sparse = (torch.rand_like(xyzs) - 0.5) * 2 * model.scale 
    sigma_sparse = model.density(xyz_sparse)
    results['sigma_sparse'] = sigma_sparse

    # sample points for normal smoothness loss
    if kwargs.get('normal_sm', False):
        eps_sm_range = kwargs["normal_sm_eps"]
        eps_sm = eps_sm_range * (torch.rand_like(normals_raw).to(normals_raw.device) * 2 - 1)
        _, _, grads_eps, _ = model.grad(xyzs + eps_sm)
        results["normals_raw_eps"] = -F.normalize(grads_eps, p=2, dim=-1, eps=1e-6)
        results["normals_raw"] = normals_raw

    # volume rendering for each ray
    results['vr_samples'], results['opacity'], results['depth'], results['rgb'], results['normal_raw'], results['normal_pred'], results['semantic'], results['ws'] = \
        VolumeRendererDN.apply(sigmas.contiguous(), rgbs.contiguous(), normals_raw.contiguous(), normals_pred.contiguous(),
                                        sems.contiguous(), results['deltas'], results['ts'],
                                        rays_a, kwargs.get('T_threshold', 1e-4), kwargs.get('classes', 7))
    
    # calculate ref loss
    normals_diff = (normals_raw - normals_pred.detach())**2
    dirs = F.normalize(dirs, p=2, dim=-1, eps=1e-6)
    normals_ori = torch.clamp(torch.sum(normals_pred*dirs, dim=-1), min=0.)**2 # don't keep dim!
    
    results['Ro'], results['Rp'] = \
        RefLoss.apply(sigmas.detach().contiguous(), normals_diff.contiguous(), normals_ori.contiguous(), results['deltas'], results['ts'],
                            rays_a, kwargs.get('T_threshold', 1e-4))
    
    # rgb output
    if exp_step_factor==0: # synthetic
        rgb_bg = torch.zeros(3, device=rays_o.device)
    else: # real
        if kwargs.get('random_bg', False):
            rgb_bg = torch.rand(3, device=rays_o.device)
        else:
            rgb_bg = torch.zeros(3, device=rays_o.device)
    
    results['rgb'] = results['rgb'] + \
                rgb_bg*rearrange(1-results['opacity'], 'n -> n 1')

    results['points'] = rays_o + rays_d * results['depth'].reshape(-1, 1)
    
    for (i, n) in results.items():
        if torch.any(torch.isnan(n)):
            print(f'nan in results[{i}]')
        if torch.any(torch.isinf(n)):
            print(f'inf in results[{i}]')

    return results


def __render_rays_test(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Input:
        rays_o: [h*w, 3] rays origin
        rays_d: [h*w, 3] rays direction

    Render rays by

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples 
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    # sampling parameter
    hits_t = hits_t[:,0,:]
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    
    # output tensors to be filled in
    classes = kwargs.get('classes', 7)
    N_rays = len(rays_o) # h*w
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)
    normal_pred = torch.zeros(N_rays, 3, device=device)
    normal_raw = torch.zeros(N_rays, 3, device=device)
    sem = torch.zeros(N_rays, classes, device=device)
    
    # Perform volume rendering
    total_samples = \
        volume_render(
            model, rays_o, rays_d, hits_t,
            opacity, depth, rgb, normal_pred, normal_raw, sem,
            **kwargs
        )
    
    results = {}
    results['opacity'] = opacity # (h*w)
    results['depth'] = depth # (h*w)
    results['rgb'] = rgb # (h*w, 3)
    results['normal_pred'] = normal_pred
    results['normal_raw'] = normal_raw
    results['semantic'] = torch.argmax(sem, dim=-1, keepdim=True)
    results['total_samples'] = total_samples # total samples for all rays
    results['points'] = rays_o + rays_d * depth.unsqueeze(-1)
    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    
    if exp_step_factor==0: # synthetic
        rgb_bg = torch.zeros(3, device=device)

    return results