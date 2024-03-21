#include "utils.h"


template <typename scalar_t>
__global__ void composite_refloss_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normals_diff,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> normals_ori,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const scalar_t T_threshold,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> loss_o,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> loss_p
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= loss_o.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // front to back compositing
    int samples = 0; scalar_t T = 1.0f;

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        loss_p[ray_idx][0] += w*normals_diff[s][0];
        loss_p[ray_idx][1] += w*normals_diff[s][1];
        loss_p[ray_idx][2] += w*normals_diff[s][2];
        loss_o[ray_idx] += w*normals_ori[s];
        T *= 1.0f-a;

        if (T <= T_threshold) break; // ray has enough opacity
        samples++;
    }
}


std::vector<torch::Tensor> composite_refloss_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor normals_diff,
    const torch::Tensor normals_ori,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const float T_threshold
){
    const int N_rays = rays_a.size(0);

    auto loss_o = torch::zeros({N_rays}, sigmas.options());
    auto loss_p = torch::zeros({N_rays, 3}, sigmas.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_refloss_fw_cu", 
    ([&] {
        composite_refloss_fw_kernel<scalar_t><<<blocks, threads>>>(
            sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            normals_diff.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normals_ori.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            T_threshold,
            loss_o.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            loss_p.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {loss_o, loss_p};
}


template <typename scalar_t>
__global__ void composite_refloss_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dloss_o,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dloss_p,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> normals_diff,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> normals_ori,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> loss_o,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> loss_p,
    const scalar_t T_threshold,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dsigmas,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dnormals_diff,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dnormals_ori
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= loss_o.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // front to back compositing
    int samples = 0;
    scalar_t X = loss_p[ray_idx][0], Y = loss_p[ray_idx][1], Z = loss_p[ray_idx][2];
    scalar_t O = loss_o[ray_idx];
    scalar_t T = 1.0f, x = 0.0f, y = 0.0f, z = 0.0f, o = 0.0f;

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        x += w*normals_diff[s][0]; y += w*normals_diff[s][1]; z += w*normals_diff[s][2];
        o += w*normals_ori[s];
        T *= 1.0f-a;

        // compute gradients by math...
        dL_dnormals_diff[s][0] = dL_dloss_p[ray_idx][0]*w;
        dL_dnormals_diff[s][1] = dL_dloss_p[ray_idx][1]*w;
        dL_dnormals_diff[s][2] = dL_dloss_p[ray_idx][2]*w;
        // compute gradients by math...
        dL_dnormals_ori[s] = dL_dloss_o[ray_idx]*w;

        dL_dsigmas[s] = deltas[s] * (
            dL_dloss_p[ray_idx][0]*(normals_diff[s][0]*T-(X-x)) + 
            dL_dloss_p[ray_idx][1]*(normals_diff[s][1]*T-(Y-y)) + 
            dL_dloss_p[ray_idx][2]*(normals_diff[s][2]*T-(Z-z)) + 
            dL_dloss_o[ray_idx]*(normals_ori[s]*T-(O-o))
        );

        if (T <= T_threshold) break; // ray has enough opacity
        samples++;
    }
}


std::vector<torch::Tensor> composite_refloss_bw_cu(
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
    const float T_threshold
){
    const int N = sigmas.size(0), N_rays = rays_a.size(0);

    auto dL_dsigmas = torch::zeros({N}, sigmas.options());
    auto dL_dnormals_diff = torch::zeros({N, 3}, sigmas.options());
    auto dL_dnormals_ori = torch::zeros({N}, sigmas.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_refloss_bw_cu", 
    ([&] {
        composite_refloss_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dloss_o.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_dloss_p.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            normals_diff.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            normals_ori.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            loss_o.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            loss_p.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            T_threshold,
            dL_dsigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_dnormals_diff.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dnormals_ori.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {dL_dsigmas, dL_dnormals_diff, dL_dnormals_ori};
}