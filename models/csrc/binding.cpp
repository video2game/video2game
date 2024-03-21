#include "utils.h"


std::vector<torch::Tensor> ray_aabb_intersect(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor half_sizes,
    const int max_hits
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(centers);
    CHECK_INPUT(half_sizes);
    return ray_aabb_intersect_cu(rays_o, rays_d, centers, half_sizes, max_hits);
}


std::vector<torch::Tensor> ray_sphere_intersect(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor radii,
    const int max_hits
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(centers);
    CHECK_INPUT(radii);
    return ray_sphere_intersect_cu(rays_o, rays_d, centers, radii, max_hits);
}


void packbits(
    torch::Tensor density_grid,
    const float density_threshold,
    torch::Tensor density_bitfield
){
    CHECK_INPUT(density_grid);
    CHECK_INPUT(density_bitfield);
    
    return packbits_cu(density_grid, density_threshold, density_bitfield);
}


torch::Tensor morton3D(const torch::Tensor coords){
    CHECK_INPUT(coords);

    return morton3D_cu(coords);
}


torch::Tensor morton3D_invert(const torch::Tensor indices){
    CHECK_INPUT(indices);

    return morton3D_invert_cu(indices);
}


std::vector<torch::Tensor> raymarching_train(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor hits_t,
    const torch::Tensor density_bitfield,
    const int cascades,
    const float scale,
    const float exp_step_factor,
    const torch::Tensor noise,
    const int grid_size,
    const int max_samples
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(density_bitfield);
    CHECK_INPUT(noise);

    return raymarching_train_cu(
        rays_o, rays_d, hits_t, density_bitfield, cascades,
        scale, exp_step_factor, noise, grid_size, max_samples);
}


std::vector<torch::Tensor> raymarching_test(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    torch::Tensor hits_t,
    const torch::Tensor alive_indices,
    const torch::Tensor density_bitfield,
    const int cascades,
    const float scale,
    const float exp_step_factor,
    const int grid_size,
    const int max_samples,
    const int N_samples
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(alive_indices);
    CHECK_INPUT(density_bitfield);

    return raymarching_test_cu(
        rays_o, rays_d, hits_t, alive_indices, density_bitfield, cascades,
        scale, exp_step_factor, grid_size, max_samples, N_samples);
}

std::vector<torch::Tensor> composite_alpha_fw(
    const torch::Tensor sigmas,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const float opacity_threshold
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(deltas);
    CHECK_INPUT(rays_a);

    return composite_alpha_fw_cu(sigmas, deltas, rays_a, opacity_threshold);
}

std::vector<torch::Tensor> composite_train_fw(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor normals_pred,
    const torch::Tensor sems,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const float opacity_threshold,
    const int classes
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(normals_pred);
    CHECK_INPUT(sems);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);

    return composite_train_fw_cu(
                sigmas, rgbs, normals_pred, sems, deltas, ts,
                rays_a, opacity_threshold, classes);
}

std::vector<torch::Tensor> composite_train_dn_fw(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor normals_raw,
    const torch::Tensor normals_pred,
    const torch::Tensor sems,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const float opacity_threshold,
    const int classes
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(normals_raw);
    CHECK_INPUT(normals_pred);
    CHECK_INPUT(sems);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);

    return composite_train_dn_fw_cu(
                sigmas, rgbs, normals_raw, normals_pred, sems, deltas, ts,
                rays_a, opacity_threshold, classes);
}

std::vector<torch::Tensor> composite_discrete_train_fw(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor normals_pred,
    const torch::Tensor sems,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const float opacity_threshold,
    const int classes
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(normals_pred);
    CHECK_INPUT(sems);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);

    return composite_discrete_train_fw_cu(
                sigmas, rgbs, normals_pred, sems, deltas, ts,
                rays_a, opacity_threshold, classes);
}

std::vector<torch::Tensor> composite_train_fw_alltmt(
    const torch::Tensor sigmas,
    const torch::Tensor normals_pred,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const float opacity_threshold,
    const float low_threshold,
    const float high_threshold,
    const float mid_threshold
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);

    return composite_train_fw_alltmt_cu(
                sigmas, normals_pred, deltas, ts,
                rays_a, opacity_threshold,
                low_threshold, high_threshold, mid_threshold);
}

std::vector<torch::Tensor> composite_train_bw(
    const torch::Tensor dL_dopacity,
    const torch::Tensor dL_ddepth,
    const torch::Tensor dL_drgb,
    const torch::Tensor dL_dnormal_pred,
    const torch::Tensor dL_dsem,
    const torch::Tensor dL_dws,
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor normals_pred,
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor opacity,
    const torch::Tensor depth,
    const torch::Tensor rgb,
    const torch::Tensor normal_pred,
    const float opacity_threshold,
    const int classes
){
    CHECK_INPUT(dL_dopacity);
    CHECK_INPUT(dL_ddepth);
    CHECK_INPUT(dL_drgb);
    CHECK_INPUT(dL_dnormal_pred);
    CHECK_INPUT(dL_dsem);
    CHECK_INPUT(dL_dws);
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(normals_pred);
    CHECK_INPUT(ws);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);
    CHECK_INPUT(opacity);
    CHECK_INPUT(depth);
    CHECK_INPUT(rgb);
    CHECK_INPUT(normal_pred);

    return composite_train_bw_cu(
                dL_dopacity, dL_ddepth, dL_drgb, dL_dnormal_pred, dL_dsem, dL_dws,
                sigmas, rgbs, normals_pred, ws, deltas, ts, rays_a,
                opacity, depth, rgb, normal_pred, opacity_threshold, classes);
}

std::vector<torch::Tensor> composite_train_dn_bw(
    const torch::Tensor dL_dopacity,
    const torch::Tensor dL_ddepth,
    const torch::Tensor dL_drgb,
    const torch::Tensor dL_dnormal_raw,
    const torch::Tensor dL_dnormal_pred,
    const torch::Tensor dL_dsem,
    const torch::Tensor dL_dws,
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor normals_raw,
    const torch::Tensor normals_pred,
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor opacity,
    const torch::Tensor depth,
    const torch::Tensor rgb,
    const torch::Tensor normal_raw,
    const torch::Tensor normal_pred,
    const float opacity_threshold,
    const int classes
){
    CHECK_INPUT(dL_dopacity);
    CHECK_INPUT(dL_ddepth);
    CHECK_INPUT(dL_drgb);
    CHECK_INPUT(dL_dnormal_raw);
    CHECK_INPUT(dL_dnormal_pred);
    CHECK_INPUT(dL_dsem);
    CHECK_INPUT(dL_dws);
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(normals_raw);
    CHECK_INPUT(normals_pred);
    CHECK_INPUT(ws);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);
    CHECK_INPUT(opacity);
    CHECK_INPUT(depth);
    CHECK_INPUT(rgb);
    CHECK_INPUT(normal_raw);
    CHECK_INPUT(normal_pred);

    return composite_train_dn_bw_cu(
                dL_dopacity, dL_ddepth, dL_drgb, dL_dnormal_raw, dL_dnormal_pred, dL_dsem, dL_dws,
                sigmas, rgbs, normals_raw, normals_pred, ws, deltas, ts, rays_a,
                opacity, depth, rgb, normal_raw, normal_pred, opacity_threshold, classes);
}
////////////////////////////////////////////////
std::vector<torch::Tensor> composite_refloss_fw(
    const torch::Tensor sigmas,
    const torch::Tensor normals_diff,
    const torch::Tensor normals_ori,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const float opacity_threshold
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(normals_diff);
    CHECK_INPUT(normals_ori);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);

    return composite_refloss_fw_cu(
                sigmas, normals_diff, normals_ori, deltas, ts,
                rays_a, opacity_threshold);
}


std::vector<torch::Tensor> composite_refloss_bw(
    const torch::Tensor dL_dloss_o,
    const torch::Tensor dL_dloss_p,
    const torch::Tensor sigmas,
    const torch::Tensor normals_diff,
    const torch::Tensor normals_ori,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor loss_o,
    const torch::Tensor loss_p,
    const float opacity_threshold
){
    CHECK_INPUT(dL_dloss_o);
    CHECK_INPUT(dL_dloss_p);
    CHECK_INPUT(sigmas);
    CHECK_INPUT(normals_diff);
    CHECK_INPUT(normals_ori);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);
    CHECK_INPUT(loss_o);
    CHECK_INPUT(loss_p);

    return composite_refloss_bw_cu(
                dL_dloss_o, dL_dloss_p,
                sigmas, normals_diff, normals_ori, deltas, ts, rays_a,
                loss_o, loss_p, opacity_threshold);
}
////////////////////////////////////////////////
void composite_test_fw(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor normals,
    const torch::Tensor normals_raw,
    const torch::Tensor sems,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor hits_t,
    const torch::Tensor alive_indices,
    const float T_threshold,
    const int classes,
    const torch::Tensor N_eff_samples,
    torch::Tensor opacity,
    torch::Tensor depth,
    torch::Tensor rgb,
    torch::Tensor normal,
    torch::Tensor normal_raw,
    torch::Tensor sem
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(normals);
    CHECK_INPUT(normals_raw);
    CHECK_INPUT(sems);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(alive_indices);
    CHECK_INPUT(N_eff_samples);
    CHECK_INPUT(opacity);
    CHECK_INPUT(depth);
    CHECK_INPUT(rgb);
    CHECK_INPUT(normal);
    CHECK_INPUT(normal_raw);
    CHECK_INPUT(sem);

    composite_test_fw_cu(
        sigmas, rgbs, normals, normals_raw, sems, 
        deltas, ts, hits_t, alive_indices,
        T_threshold, classes, N_eff_samples,
        opacity, depth, rgb, normal, normal_raw, sem);
}


std::vector<torch::Tensor> distortion_loss_fw(
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
){
    CHECK_INPUT(ws);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);

    return distortion_loss_fw_cu(ws, deltas, ts, rays_a);
}


torch::Tensor distortion_loss_bw(
    const torch::Tensor dL_dloss,
    const torch::Tensor ws_inclusive_scan,
    const torch::Tensor wts_inclusive_scan,
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
){
    CHECK_INPUT(dL_dloss);
    CHECK_INPUT(ws_inclusive_scan);
    CHECK_INPUT(wts_inclusive_scan);
    CHECK_INPUT(ws);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);

    return distortion_loss_bw_cu(dL_dloss, ws_inclusive_scan, wts_inclusive_scan,
                                 ws, deltas, ts, rays_a);
}

torch::Tensor opacity_aggr_fw(
    const torch::Tensor ws,
    const torch::Tensor rays_a
){
    CHECK_INPUT(ws);
    CHECK_INPUT(rays_a);

    return opacity_aggr_fw_cu(ws, rays_a);
}


torch::Tensor opacity_aggr_bw(
    const torch::Tensor dxyz_opas,
    const torch::Tensor rays_a
){
    CHECK_INPUT(dxyz_opas);
    CHECK_INPUT(rays_a);

    return opacity_aggr_bw_cu(dxyz_opas, rays_a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("ray_aabb_intersect", &ray_aabb_intersect);
    m.def("ray_sphere_intersect", &ray_sphere_intersect);

    m.def("morton3D", &morton3D);
    m.def("morton3D_invert", &morton3D_invert);
    m.def("packbits", &packbits);

    m.def("raymarching_train", &raymarching_train);
    m.def("raymarching_test", &raymarching_test);
    m.def("composite_alpha_fw", &composite_alpha_fw);
    m.def("composite_train_fw", &composite_train_fw);
    m.def("composite_train_dn_fw", &composite_train_dn_fw);
    m.def("composite_discrete_train_fw", &composite_discrete_train_fw);
    m.def("composite_train_fw_alltmt", &composite_train_fw_alltmt);
    m.def("composite_train_bw", &composite_train_bw);
    m.def("composite_train_dn_bw", &composite_train_dn_bw);
    m.def("composite_refloss_fw", &composite_refloss_fw);
    m.def("composite_refloss_bw", &composite_refloss_bw);
    m.def("composite_test_fw", &composite_test_fw);

    m.def("distortion_loss_fw", &distortion_loss_fw);
    m.def("distortion_loss_bw", &distortion_loss_bw);

    m.def("opacity_aggr_fw", &opacity_aggr_fw);
    m.def("opacity_aggr_bw", &opacity_aggr_bw);
}