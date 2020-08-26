#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>


__global__ void anchor_scatter_kernel(const float *anchors, float *output, 
					const int ph, const int pw, const int stride, const int an,
					const float to_move)
{
	int phw_i = blockIdx.x*512 + threadIdx.x;
	if (phw_i >= ph*pw) return;
	int ph_i = phw_i / pw;
	int pw_i = phw_i % pw;
	int an_i = blockIdx.y;
	int b_i = blockIdx.z;
	float center_y = ph_i * stride + to_move;
	float center_x = pw_i * stride + to_move;
	int offset = b_i*ph*pw*an*4 + ph_i*pw*an*4 + pw_i*an*4 + an_i*4;
	float h_2 = anchors[an_i*2+0]/2.0;
	float w_2 = anchors[an_i*2+1]/2.0;
	output[offset+0] = center_y - h_2;
	output[offset+1] = center_x - w_2;
	output[offset+2] = center_y + h_2;
	output[offset+3] = center_x + w_2;
}


at::Tensor anchor_scatter_cuda(const at::Tensor &anchors,  const int batch_size,
                              const int ph, const int pw, const int stride, const float to_move)
{
	// anchors: F(an, 2) h,w
	// ->       F(b, hwan, 4) yxyx
	const int an = anchors.size(0);
	auto output = at::zeros({batch_size, ph*pw*an, 4}, anchors.options());
	if (output.numel() == 0) {
		THCudaCheck(cudaGetLastError());
		return output;
	}
	dim3 grid(ph*pw/512+1, an, batch_size), block(512);
	anchor_scatter_kernel<<<grid, block>>>(
		anchors.contiguous().data<float>(),
    	output.contiguous().data<float>(),
    	ph, pw, stride, an, to_move);
	THCudaCheck(cudaGetLastError());
	return output;
}


__global__ void center_scatter_kernel(const float *anchors, float *output, 
		const int ph, const int pw, const int stride, const int n, const float to_move)
{
	int phw_i = blockIdx.x*512 + threadIdx.x;
	if (phw_i >= ph*pw) return;
	int ph_i = phw_i / pw;
	int pw_i = phw_i % pw;
	int b_i = blockIdx.y;
	float center_y = ph_i * stride + to_move;
	float center_x = pw_i * stride + to_move;
	int offset = b_i*ph*pw*(2+n) + ph_i*pw*(2+n) + pw_i*(2+n);
	output[offset+0] = center_y;
	output[offset+1] = center_x;
	for(int i=0; i<n; i++) {
		output[offset+2+i] = anchors[i];
	}
}


at::Tensor center_scatter_cuda(const at::Tensor &anchors,  const int batch_size,
			const int ph, const int pw, const int stride, const float to_move)
{
	// anchors: F(n)
	// ->       F(b, hw, 2+n) # 2+n: cy, cx, ...
	const int n = anchors.size(0);
	auto output = at::zeros({batch_size, ph*pw, 2+n}, anchors.options());
	if (output.numel() == 0) {
		THCudaCheck(cudaGetLastError());
		return output;
	}
	dim3 grid(ph*pw/512+1, batch_size), block(512);
	center_scatter_kernel<<<grid, block>>>(
		anchors.contiguous().data<float>(),
    	output.contiguous().data<float>(),
    	ph, pw, stride, n, to_move);
	THCudaCheck(cudaGetLastError());
	return output;
}
