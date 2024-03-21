#include "utils.h"
#include <thrust/scan.h>


template <typename scalar_t>
__global__ void composite_alpha_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const scalar_t T_threshold,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> alphas,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ws
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // front to back compositing
    int samples = 0; scalar_t T = 1.0f;

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        alphas[s] = a;
        ws[s] = w;
        T *= 1.0f-a;

        if (T <= T_threshold) break; // ray has enough opacity
        samples++;
    }
}


std::vector<torch::Tensor> composite_alpha_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const float T_threshold
){
    const int N = sigmas.size(0), N_rays = rays_a.size(0);

    auto alphas = torch::zeros({N}, sigmas.options());
    auto ws = torch::zeros({N}, sigmas.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_alpha_fw_cu", 
    ([&] {
            composite_alpha_fw_kernel<scalar_t><<<blocks, threads>>>(
            sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            T_threshold,
            alphas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {alphas, ws};
}

template <typename scalar_t>
__global__ void composite_train_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normals_pred,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sems,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const scalar_t T_threshold,
    const int64_t classes, 
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> total_samples,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> opacity,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normal_pred,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sem,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ws
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= opacity.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // front to back compositing
    int samples = 0; scalar_t T = 1.0f;

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        rgb[ray_idx][0] += w*rgbs[s][0];
        rgb[ray_idx][1] += w*rgbs[s][1];
        rgb[ray_idx][2] += w*rgbs[s][2];
        normal_pred[ray_idx][0] += w*normals_pred[s][0];
        normal_pred[ray_idx][1] += w*normals_pred[s][1];
        normal_pred[ray_idx][2] += w*normals_pred[s][2];
        depth[ray_idx] += w*ts[s];
        for (int i=0;i<classes;i++) {
            sem[ray_idx][i] += w*sems[s][i];
        }
        opacity[ray_idx] += w;
        ws[s] = w;
        T *= 1.0f-a;

        if (T <= T_threshold) break; // ray has enough opacity
        samples++;
    }
    total_samples[ray_idx] = samples;
}


template <typename scalar_t>
__global__ void composite_train_dn_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normals_raw,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normals_pred,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sems,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const scalar_t T_threshold,
    const int64_t classes,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> total_samples,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> opacity,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normal_raw,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normal_pred,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sem,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ws
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= opacity.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // front to back compositing
    int samples = 0; scalar_t T = 1.0f;

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        rgb[ray_idx][0] += w*rgbs[s][0];
        rgb[ray_idx][1] += w*rgbs[s][1];
        rgb[ray_idx][2] += w*rgbs[s][2];
        normal_raw[ray_idx][0] += w*normals_raw[s][0];
        normal_raw[ray_idx][1] += w*normals_raw[s][1];
        normal_raw[ray_idx][2] += w*normals_raw[s][2];
        normal_pred[ray_idx][0] += w*normals_pred[s][0];
        normal_pred[ray_idx][1] += w*normals_pred[s][1];
        normal_pred[ray_idx][2] += w*normals_pred[s][2];
        depth[ray_idx] += w*ts[s];
        for (int i=0;i<classes;i++) {
            sem[ray_idx][i] += w*sems[s][i];
        }
        opacity[ray_idx] += w;
        ws[s] = w;
        T *= 1.0f-a;

        if (T <= T_threshold) break; // ray has enough opacity
        samples++;
    }
    total_samples[ray_idx] = samples;
}


std::vector<torch::Tensor> composite_train_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor normals_pred,
    const torch::Tensor sems,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const float T_threshold,
    const int classes
){
    const int N = sigmas.size(0), N_rays = rays_a.size(0);

    auto opacity = torch::zeros({N_rays}, sigmas.options());
    auto depth = torch::zeros({N_rays}, sigmas.options());
    auto rgb = torch::zeros({N_rays, 3}, sigmas.options());
    auto normal_pred = torch::zeros({N_rays, 3}, sigmas.options());\
    auto sem = torch::zeros({N_rays, classes}, sigmas.options());
    auto ws = torch::zeros({N}, sigmas.options());
    auto total_samples = torch::zeros({N_rays}, torch::dtype(torch::kLong).device(sigmas.device()));

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_train_fw_cu", 
    ([&] {
        composite_train_fw_kernel<scalar_t><<<blocks, threads>>>(
            sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normals_pred.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sems.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            T_threshold,
            classes,
            total_samples.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normal_pred.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sem.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            ws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {total_samples, opacity, depth, rgb, normal_pred, sem, ws};
}

std::vector<torch::Tensor> composite_train_dn_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor normals_raw,
    const torch::Tensor normals_pred,
    const torch::Tensor sems,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const float T_threshold,
    const int classes
){
    const int N = sigmas.size(0), N_rays = rays_a.size(0);

    auto opacity = torch::zeros({N_rays}, sigmas.options());
    auto depth = torch::zeros({N_rays}, sigmas.options());
    auto rgb = torch::zeros({N_rays, 3}, sigmas.options());
    auto normal_raw = torch::zeros({N_rays, 3}, sigmas.options());
    auto normal_pred = torch::zeros({N_rays, 3}, sigmas.options());
    auto sem = torch::zeros({N_rays, classes}, sigmas.options());
    auto ws = torch::zeros({N}, sigmas.options());
    auto total_samples = torch::zeros({N_rays}, torch::dtype(torch::kLong).device(sigmas.device()));

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_train_dn_fw_cu",
    ([&] {
        composite_train_dn_fw_kernel<scalar_t><<<blocks, threads>>>(
            sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normals_raw.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normals_pred.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sems.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            T_threshold,
            classes,
            total_samples.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normal_raw.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normal_pred.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sem.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            ws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {total_samples, opacity, depth, rgb, normal_raw, normal_pred, sem, ws};
}


template <typename scalar_t>
__global__ void composite_train_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dopacity,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_ddepth,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_drgb,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dnormal_pred,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dsem,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dws,
    scalar_t* __restrict__ dL_dws_times_ws,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normals_pred,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> opacity,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normal_pred,
    const scalar_t T_threshold,
    const int64_t classes,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dsigmas,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_drgbs,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dnormals_pred,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dsems
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= opacity.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // front to back compositing
    int samples = 0;
    scalar_t R = rgb[ray_idx][0], G = rgb[ray_idx][1], B = rgb[ray_idx][2];
    scalar_t O = opacity[ray_idx], D = depth[ray_idx];
    scalar_t T = 1.0f, r = 0.0f, g = 0.0f, b = 0.0f, d = 0.0f;

    // compute prefix sum of dL_dws * ws
    // [a0, a1, a2, a3, ...] -> [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, ...]
    thrust::inclusive_scan(thrust::device,
                           dL_dws_times_ws+start_idx,
                           dL_dws_times_ws+start_idx+N_samples,
                           dL_dws_times_ws+start_idx);
    scalar_t dL_dws_times_ws_sum = dL_dws_times_ws[start_idx+N_samples-1];

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        r += w*rgbs[s][0]; g += w*rgbs[s][1]; b += w*rgbs[s][2];
        d += w*ts[s];
        T *= 1.0f-a;

        // compute gradients by math...
        dL_drgbs[s][0] = dL_drgb[ray_idx][0]*w;
        dL_drgbs[s][1] = dL_drgb[ray_idx][1]*w;
        dL_drgbs[s][2] = dL_drgb[ray_idx][2]*w;
        // compute gradients by math...
        dL_dnormals_pred[s][0] = dL_dnormal_pred[ray_idx][0]*w;
        dL_dnormals_pred[s][1] = dL_dnormal_pred[ray_idx][1]*w;
        dL_dnormals_pred[s][2] = dL_dnormal_pred[ray_idx][2]*w;

        for (int i=0;i<classes;i++){
            dL_dsems[s][i] = dL_dsem[ray_idx][i]*w;
        }

        dL_dsigmas[s] = deltas[s] * (
            dL_drgb[ray_idx][0]*(rgbs[s][0]*T-(R-r)) + 
            dL_drgb[ray_idx][1]*(rgbs[s][1]*T-(G-g)) + 
            dL_drgb[ray_idx][2]*(rgbs[s][2]*T-(B-b)) + 
            dL_dopacity[ray_idx]*(1-O) + 
            dL_ddepth[ray_idx]*(ts[s]*T-(D-d)) + 
            T*dL_dws[s]-(dL_dws_times_ws_sum-dL_dws_times_ws[s])
        );

        if (T <= T_threshold) break; // ray has enough opacity
        samples++;
    }
}

template <typename scalar_t>
__global__ void composite_train_dn_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dopacity,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_ddepth,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_drgb,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dnormal_raw,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dnormal_pred,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dsem,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dws,
    scalar_t* __restrict__ dL_dws_times_ws,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normals_raw,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normals_pred,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> opacity,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normal_raw,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normal_pred,
    const scalar_t T_threshold,
    const int64_t classes,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dsigmas,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_drgbs,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dnormals_raw,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dnormals_pred,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dsems
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= opacity.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // front to back compositing
    int samples = 0;
    scalar_t R = rgb[ray_idx][0], G = rgb[ray_idx][1], B = rgb[ray_idx][2];
    scalar_t O = opacity[ray_idx], D = depth[ray_idx];
    scalar_t T = 1.0f, r = 0.0f, g = 0.0f, b = 0.0f, d = 0.0f;

    // compute prefix sum of dL_dws * ws
    // [a0, a1, a2, a3, ...] -> [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, ...]
    thrust::inclusive_scan(thrust::device,
                           dL_dws_times_ws+start_idx,
                           dL_dws_times_ws+start_idx+N_samples,
                           dL_dws_times_ws+start_idx);
    scalar_t dL_dws_times_ws_sum = dL_dws_times_ws[start_idx+N_samples-1];

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        r += w*rgbs[s][0]; g += w*rgbs[s][1]; b += w*rgbs[s][2];
        d += w*ts[s];
        T *= 1.0f-a;

        // compute gradients by math...
        dL_drgbs[s][0] = dL_drgb[ray_idx][0]*w;
        dL_drgbs[s][1] = dL_drgb[ray_idx][1]*w;
        dL_drgbs[s][2] = dL_drgb[ray_idx][2]*w;
        // compute gradients by math...
        dL_dnormals_raw[s][0] = dL_dnormal_raw[ray_idx][0]*w;
        dL_dnormals_raw[s][1] = dL_dnormal_raw[ray_idx][1]*w;
        dL_dnormals_raw[s][2] = dL_dnormal_raw[ray_idx][2]*w;
        dL_dnormals_pred[s][0] = dL_dnormal_pred[ray_idx][0]*w;
        dL_dnormals_pred[s][1] = dL_dnormal_pred[ray_idx][1]*w;
        dL_dnormals_pred[s][2] = dL_dnormal_pred[ray_idx][2]*w;

        for (int i=0;i<classes;i++){
            dL_dsems[s][i] = dL_dsem[ray_idx][i]*w;
        }

        dL_dsigmas[s] = deltas[s] * (
            dL_drgb[ray_idx][0]*(rgbs[s][0]*T-(R-r)) +
            dL_drgb[ray_idx][1]*(rgbs[s][1]*T-(G-g)) +
            dL_drgb[ray_idx][2]*(rgbs[s][2]*T-(B-b)) +
            dL_dopacity[ray_idx]*(1-O) +
            dL_ddepth[ray_idx]*(ts[s]*T-(D-d)) +
            T*dL_dws[s]-(dL_dws_times_ws_sum-dL_dws_times_ws[s])
        );

        if (T <= T_threshold) break; // ray has enough opacity
        samples++;
    }
}

std::vector<torch::Tensor> composite_train_bw_cu(
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
    const float T_threshold,
    const int classes
){
    const int N = sigmas.size(0), N_rays = rays_a.size(0);

    auto dL_dsigmas = torch::zeros({N}, sigmas.options());
    auto dL_drgbs = torch::zeros({N, 3}, sigmas.options());
    auto dL_dnormals_pred = torch::zeros({N, 3}, sigmas.options());
    auto dL_dsems = torch::zeros({N, classes}, sigmas.options());

    auto dL_dws_times_ws = dL_dws * ws; // auxiliary input

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_train_bw_cu", 
    ([&] {
        composite_train_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dopacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_ddepth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_drgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dnormal_pred.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dsem.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_dws_times_ws.data_ptr<scalar_t>(),
            sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normals_pred.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normal_pred.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            T_threshold,
            classes,
            dL_dsigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_drgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dnormals_pred.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dsems.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {dL_dsigmas, dL_drgbs, dL_dnormals_pred, dL_dsems};
}


std::vector<torch::Tensor> composite_train_dn_bw_cu(
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
    const float T_threshold,
    const int classes
){
    const int N = sigmas.size(0), N_rays = rays_a.size(0);

    auto dL_dsigmas = torch::zeros({N}, sigmas.options());
    auto dL_drgbs = torch::zeros({N, 3}, sigmas.options());
    auto dL_dnormals_raw = torch::zeros({N, 3}, sigmas.options());
    auto dL_dnormals_pred = torch::zeros({N, 3}, sigmas.options());
    auto dL_dsems = torch::zeros({N, classes}, sigmas.options());

    auto dL_dws_times_ws = dL_dws * ws; // auxiliary input

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_train_dn_bw_cu",
    ([&] {
        composite_train_dn_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dopacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_ddepth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_drgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dnormal_raw.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dnormal_pred.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dsem.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_dws_times_ws.data_ptr<scalar_t>(),
            sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normals_raw.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normals_pred.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normal_raw.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normal_pred.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            T_threshold,
            classes,
            dL_dsigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_drgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dnormals_raw.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dnormals_pred.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dsems.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {dL_dsigmas, dL_drgbs, dL_dnormals_raw, dL_dnormals_pred, dL_dsems};
}

template <typename scalar_t>
__global__ void composite_test_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> normals,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> normals_raw,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> sems,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> hits_t,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> alive_indices,
    const scalar_t T_threshold,
    const int64_t classes,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> N_eff_samples,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> opacity,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normal,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normal_raw,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sem
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= alive_indices.size(0)) return;

    if (N_eff_samples[n]==0){ // no hit
        alive_indices[n] = -1;
        return;
    }

    const size_t r = alive_indices[n]; // ray index

    // front to back compositing
    int s = 0; scalar_t T = 1-opacity[r];

    while (s < N_eff_samples[n]) {
        const scalar_t a = 1.0f - __expf(-sigmas[n][s]*deltas[n][s]);
        const scalar_t w = a * T;

        rgb[r][0] += w*rgbs[n][s][0];
        rgb[r][1] += w*rgbs[n][s][1];
        rgb[r][2] += w*rgbs[n][s][2];
        depth[r] += w*ts[n][s];
        opacity[r] += w;
        normal[r][0] += w*normals[n][s][0];
        normal[r][1] += w*normals[n][s][1];
        normal[r][2] += w*normals[n][s][2];
        normal_raw[r][0] += w*normals_raw[n][s][0];
        normal_raw[r][1] += w*normals_raw[n][s][1];
        normal_raw[r][2] += w*normals_raw[n][s][2];
        for(int i=0;i<classes;i++){
            sem[r][i] += w*sems[n][s][i];
        }
        T *= 1.0f-a;

        if (T <= T_threshold){ // ray has enough opacity
            alive_indices[n] = -1;
            break;
        }
        s++;
    }
}

void composite_test_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor normals,
    const torch::Tensor normals_raw,
    const torch::Tensor sems,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor hits_t,
    torch::Tensor alive_indices,
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
    const int N_rays = alive_indices.size(0);

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_test_fw_cu", 
    ([&] {
        composite_test_fw_kernel<scalar_t><<<blocks, threads>>>(
            sigmas.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            rgbs.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            normals.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            normals_raw.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            sems.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            hits_t.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            alive_indices.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            T_threshold,
            classes,
            N_eff_samples.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normal.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normal_raw.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sem.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));
}




template <typename scalar_t>
__global__ void composite_train_fw_alltmt_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normals_pred,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const scalar_t T_threshold,
    const scalar_t low_threshold,
    const scalar_t high_threshold,
    const scalar_t mid_threshold,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normal_pred,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth_t_low_slice,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth_t_high_slice,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth_t_mid_slice
    
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= depth_t_low_slice.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // front to back compositing
    int samples = 0;
    scalar_t T = 1.0f;
    scalar_t depth = 0.0f; 
    bool low_t_marked = false;
    bool high_t_marked = false;
    bool mid_t_marked = false;

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        depth += w*ts[s];
        T *= 1.0f-a;

        normal_pred[ray_idx][0] += w*normals_pred[s][0];
        normal_pred[ray_idx][1] += w*normals_pred[s][1];
        normal_pred[ray_idx][2] += w*normals_pred[s][2];
        
        if (T <= low_threshold && !low_t_marked){
            depth_t_low_slice[ray_idx] = depth;
            low_t_marked = true;
        } 
        if (T <= high_threshold && !high_t_marked){
            depth_t_high_slice[ray_idx] = depth;
            high_t_marked = true;
        } 
        if (T <= mid_threshold && !mid_t_marked){
            depth_t_mid_slice[ray_idx] = depth;
            mid_t_marked = true;
        }
        if (T <= T_threshold) break; // ray has enough opacity
	    samples++;
    }
    if (!low_t_marked) depth_t_low_slice[ray_idx] = depth;
    if (!high_t_marked) depth_t_high_slice[ray_idx] = depth;
    if (!mid_t_marked) depth_t_mid_slice[ray_idx] = depth;
}



std::vector<torch::Tensor> composite_train_fw_alltmt_cu(
    const torch::Tensor sigmas,
    const torch::Tensor normals_pred,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const float T_threshold,
    const float low_threshold,
    const float high_threshold,
    const float mid_threshold
){
    const int N = sigmas.size(0), N_rays = rays_a.size(0);

    auto normal_pred = torch::zeros({N_rays, 3}, sigmas.options());
    auto depth_t_low_slice = torch::zeros({N_rays}, sigmas.options());
    auto depth_t_high_slice = torch::zeros({N_rays}, sigmas.options());
    auto depth_t_mid_slice = torch::zeros({N_rays}, sigmas.options());
    
    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_train_fw_alltmt_cu", 
    ([&] {
        composite_train_fw_alltmt_kernel<scalar_t><<<blocks, threads>>>(
            sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            normals_pred.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            T_threshold,
            low_threshold,
            high_threshold,
            mid_threshold,
            normal_pred.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            depth_t_low_slice.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth_t_high_slice.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth_t_mid_slice.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {normal_pred, depth_t_low_slice, depth_t_high_slice, depth_t_mid_slice};
}



template <typename scalar_t>
__global__ void composite_discrete_train_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normals_pred,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sems,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const scalar_t T_threshold,
    const int64_t classes, 
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> total_samples,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> opacity,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> xyzs_opacity,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normal_pred,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sem,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ws
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= opacity.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // front to back compositing
    int samples = 0; scalar_t T = 1.0f;

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        rgb[ray_idx][0] += w*rgbs[s][0];
        rgb[ray_idx][1] += w*rgbs[s][1];
        rgb[ray_idx][2] += w*rgbs[s][2];
        normal_pred[ray_idx][0] += w*normals_pred[s][0];
        normal_pred[ray_idx][1] += w*normals_pred[s][1];
        normal_pred[ray_idx][2] += w*normals_pred[s][2];
        depth[ray_idx] += w*ts[s];
        for (int i=0;i<classes;i++) {
            sem[ray_idx][i] += w*sems[s][i];
        }
        opacity[ray_idx] += w;
        ws[s] = w;
        T *= 1.0f-a;
        xyzs_opacity[s] = opacity[ray_idx];
        if (T <= T_threshold) break; // ray has enough opacity
        samples++;
    }
    total_samples[ray_idx] = samples;
}


std::vector<torch::Tensor> composite_discrete_train_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor normals_pred,
    const torch::Tensor sems,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const float T_threshold,
    const int classes
){
    const int N = sigmas.size(0), N_rays = rays_a.size(0);

    auto opacity = torch::zeros({N_rays}, sigmas.options());
    auto xyzs_opacity = torch::zeros({N}, sigmas.options());
    auto depth = torch::zeros({N_rays}, sigmas.options());
    auto rgb = torch::zeros({N_rays, 3}, sigmas.options());
    auto normal_pred = torch::zeros({N_rays, 3}, sigmas.options());\
    auto sem = torch::zeros({N_rays, classes}, sigmas.options());
    auto ws = torch::zeros({N}, sigmas.options());
    auto total_samples = torch::zeros({N_rays}, torch::dtype(torch::kLong).device(sigmas.device()));

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_discrete_train_fw_cu", 
    ([&] {
        composite_discrete_train_fw_kernel<scalar_t><<<blocks, threads>>>(
            sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normals_pred.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sems.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            T_threshold,
            classes,
            total_samples.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            xyzs_opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normal_pred.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sem.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            ws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {total_samples, opacity, xyzs_opacity, depth, rgb, normal_pred, sem, ws};
}






