import torch
from einops import rearrange
import vren
from models.const import MAX_SAMPLES

def volume_render(
    model, 
    rays_o,
    rays_d,
    hits_t,
    # Image properties to be updated
    opacity,
    depth,
    rgb,
    normal_pred,
    normal_raw,
    sem,
    # Other parameters
    **kwargs
):
    N_rays = len(rays_o)
    device = rays_o.device
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    if isinstance(kwargs.get('embedding_a', None), torch.Tensor):
        embedding_a = kwargs['embedding_a']

    classes = kwargs.get('classes', 7)
    samples = 0
    total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)

    min_samples = kwargs.get('min_samples', 4)

    while samples < kwargs.get('max_samples', MAX_SAMPLES):

        N_alive = len(alive_indices)
        if N_alive==0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays//N_alive, 64), min_samples)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t, alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, kwargs.get('max_samples', MAX_SAMPLES), N_samples)
        total_samples += N_eff_samples.sum()
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs==0, dim=1)
        if valid_mask.sum()==0: break

        ## Shapes
        # xyzs: (N_alive*N_samples, 3)
        # dirs: (N_alive*N_samples, 3)
        # deltas: (N_alive, N_samples) intervals between samples (with previous ones)
        # ts: (N_alive, N_samples) ray length for each samples
        # N_eff_samples: (N_alive) #samples along each ray <= N_smaples

        sigmas = torch.zeros(len(xyzs), device=device)
        rgbs = torch.zeros(len(xyzs), 3, device=device)
        normals_pred = torch.zeros(len(xyzs), 3, device=device)
        normals_raw = torch.zeros(len(xyzs), 3, device=device)
        sems = torch.zeros(len(xyzs), classes, device=device)
        if isinstance(kwargs.get('embedding_a', None), torch.Tensor):
            kwargs['embedding_a'] = torch.repeat_interleave(embedding_a, len(xyzs), 0)[valid_mask]
       
        _sigmas, _rgbs, _normals_pred, _sems, _normals_raw = model.forward_test(xyzs[valid_mask], dirs[valid_mask], **kwargs)

        if kwargs.get('render_rgb', False) or kwargs.get('render_depth', False):
            sigmas[valid_mask], rgbs[valid_mask] = _sigmas.detach().float(), _rgbs.detach().float()
        if kwargs.get('render_sem', False):
            sems[valid_mask] = _sems.float()
        if kwargs.get('render_normal', False):
            normals_pred[valid_mask] = _normals_pred.float()
            normals_raw[valid_mask] = _normals_raw.float()

        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        normals_pred = rearrange(normals_pred, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        normals_raw = rearrange(normals_raw, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        sems = rearrange(sems, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        vren.composite_test_fw(
            sigmas, rgbs, normals_pred, normals_raw, sems, deltas, ts,
            hits_t, alive_indices, kwargs.get('T_threshold', 1e-4), classes,
            N_eff_samples, opacity, depth, rgb, normal_pred, normal_raw, sem)
        alive_indices = alive_indices[alive_indices>=0]

    rgb_bg = torch.zeros(3, device=device)
    rgb += rgb_bg*rearrange(1 - opacity, 'n -> n 1')

    return total_samples