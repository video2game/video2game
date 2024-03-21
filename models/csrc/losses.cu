#include "utils.h"
#include <thrust/scan.h>
#include <thrust/reverse.h>
#include <thrust/execution_policy.h>

// for details of the formulae, please see https://arxiv.org/pdf/2206.05085.pdf

template <typename scalar_t>
__global__ void prefix_sums_kernel(
    const scalar_t* __restrict__ ws,
    const scalar_t* __restrict__ wts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    scalar_t* __restrict__ ws_inclusive_scan,
    scalar_t* __restrict__ ws_exclusive_scan,
    scalar_t* __restrict__ wts_inclusive_scan,
    scalar_t* __restrict__ wts_exclusive_scan
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // compute prefix sum of ws and ws*ts
    // [a0, a1, a2, a3, ...] -> [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, ...]
    thrust::inclusive_scan(thrust::device,
                           ws+start_idx,
                           ws+start_idx+N_samples,
                           ws_inclusive_scan+start_idx);
    thrust::inclusive_scan(thrust::device,
                           wts+start_idx,
                           wts+start_idx+N_samples,
                           wts_inclusive_scan+start_idx);
    // [a0, a1, a2, a3, ...] -> [0, a0, a0+a1, a0+a1+a2, ...]
    thrust::exclusive_scan(thrust::device,
                           ws+start_idx,
                           ws+start_idx+N_samples,
                           ws_exclusive_scan+start_idx);
    thrust::exclusive_scan(thrust::device,
                           wts+start_idx,
                           wts+start_idx+N_samples,
                           wts_exclusive_scan+start_idx);
}


template <typename scalar_t>
__global__ void distortion_loss_fw_kernel(
    const scalar_t* __restrict__ _loss,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> loss
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    loss[ray_idx] = thrust::reduce(thrust::device, 
                                   _loss+start_idx,
                                   _loss+start_idx+N_samples,
                                   (scalar_t)0);
}


std::vector<torch::Tensor> distortion_loss_fw_cu(
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
){
    const int N_rays = rays_a.size(0), N = ws.size(0);

    auto wts = ws * ts;

    auto ws_inclusive_scan = torch::zeros({N}, ws.options());
    auto ws_exclusive_scan = torch::zeros({N}, ws.options());
    auto wts_inclusive_scan = torch::zeros({N}, ws.options());
    auto wts_exclusive_scan = torch::zeros({N}, ws.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(ws.type(), "distortion_loss_fw_cu_prefix_sums", 
    ([&] {
        prefix_sums_kernel<scalar_t><<<blocks, threads>>>(
            ws.data_ptr<scalar_t>(),
            wts.data_ptr<scalar_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            ws_inclusive_scan.data_ptr<scalar_t>(),
            ws_exclusive_scan.data_ptr<scalar_t>(),
            wts_inclusive_scan.data_ptr<scalar_t>(),
            wts_exclusive_scan.data_ptr<scalar_t>()
        );
    }));

    auto _loss = 2*(wts_inclusive_scan*ws_exclusive_scan-
                    ws_inclusive_scan*wts_exclusive_scan) + 1.0f/3*ws*ws*deltas;

    auto loss = torch::zeros({N_rays}, ws.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(ws.type(), "distortion_loss_fw_cu", 
    ([&] {
        distortion_loss_fw_kernel<scalar_t><<<blocks, threads>>>(
            _loss.data_ptr<scalar_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            loss.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {loss, ws_inclusive_scan, wts_inclusive_scan};
}


template <typename scalar_t>
__global__ void distortion_loss_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dloss,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ws_inclusive_scan,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> wts_inclusive_scan,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ws,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dws
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];
    const int end_idx = start_idx+N_samples-1;

    const scalar_t ws_sum = ws_inclusive_scan[end_idx];
    const scalar_t wts_sum = wts_inclusive_scan[end_idx];
    // fill in dL_dws from start_idx to end_idx
    for (int s=start_idx; s<=end_idx; s++){
        dL_dws[s] = dL_dloss[ray_idx] * 2 * (
            (s==start_idx?
                (scalar_t)0:
                (ts[s]*ws_inclusive_scan[s-1]-wts_inclusive_scan[s-1])
            ) + 
            (wts_sum-wts_inclusive_scan[s]-ts[s]*(ws_sum-ws_inclusive_scan[s]))
        );
        dL_dws[s] += dL_dloss[ray_idx] * (scalar_t)2/3*ws[s]*deltas[s];
    }
}


torch::Tensor distortion_loss_bw_cu(
    const torch::Tensor dL_dloss,
    const torch::Tensor ws_inclusive_scan,
    const torch::Tensor wts_inclusive_scan,
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
){
    const int N_rays = rays_a.size(0), N = ws.size(0);

    auto dL_dws = torch::zeros({N}, dL_dloss.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(ws.type(), "distortion_loss_bw_cu", 
    ([&] {
        distortion_loss_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dloss.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ws_inclusive_scan.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            wts_inclusive_scan.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            dL_dws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return dL_dws;
}



template <typename scalar_t>
__global__ void opacity_aggr_fw_kernel(
    const scalar_t* __restrict__ ws,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    scalar_t* __restrict__ xyz_opas
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // [a0, a1, a2, a3, ...] -> [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, ...]
    thrust::inclusive_scan(thrust::device,
                           ws+start_idx,
                           ws+start_idx+N_samples,
                           xyz_opas+start_idx);
}


torch::Tensor opacity_aggr_fw_cu(
    const torch::Tensor ws,
    const torch::Tensor rays_a
){
    const int N_rays = rays_a.size(0), N = ws.size(0);

    auto xyz_opas = torch::zeros({N}, ws.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(ws.type(), "opacity_aggr_fw_cu",
    ([&] {
        opacity_aggr_fw_kernel<scalar_t><<<blocks, threads>>>(
            ws.data_ptr<scalar_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            xyz_opas.data_ptr<scalar_t>()
        );
    }));

    return xyz_opas;
}


template <typename scalar_t>
__global__ void opacity_aggr_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dxyz_opas,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> d_ws
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int start_idx = rays_a[n][1], N_samples = rays_a[n][2];
    const int end_idx = start_idx+N_samples-1;

    auto acc_w = 0.0;

    for (int s=end_idx; s>=start_idx; s--){
        acc_w += dxyz_opas[s];
        d_ws[s] = acc_w;
    }
}

torch::Tensor opacity_aggr_bw_cu(
    const torch::Tensor dxyz_opas,
    const torch::Tensor rays_a
){
    const int N_rays = rays_a.size(0), N = dxyz_opas.size(0);

    auto d_ws = torch::zeros({N}, dxyz_opas.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dxyz_opas.type(), "opacity_aggr_bw_cu",
    ([&] {
        opacity_aggr_bw_kernel<scalar_t><<<blocks, threads>>>(
            dxyz_opas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            d_ws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return d_ws;
}