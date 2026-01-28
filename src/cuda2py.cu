/**
 * @File		cuda2py.cu
 * @brief		single thread, multi thread and SIMT GPU
 * @author		Jeong Hoon (Sian) Choi
 * @version 	1.0.0
 * @date		2026-01-27
 */

/* Copyright (C)
 * 2026 - Jeong Hoon (Sian) Choi
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// void custom_terminate_fnct(void) {
//	exit(1);
// }

static const int block_size = 32;

template <typename T>
__global__ void optimize_cuda(const T* a, const T* b, T* c, const int n, const int m, const int k) {
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int local_col = threadIdx.x;
	int local_row = threadIdx.y;
	
	__shared__ T partial_a[block_size][block_size];
	__shared__ T partial_b[block_size][block_size];

	T value = 0;
	
	for (int blk = 0; blk < std::ceil(static_cast<float>(k) / block_size); ++blk) {
		int stride = blk * block_size;

	    if (row >= n || stride + local_col >= k)
			partial_a[local_row][local_col] = 0;
		else
			partial_a[local_row][local_col] = a[row * k + (stride + local_col)];

		if (col >= m || stride + local_row >= k)
			partial_b[local_col][local_row] = 0;	// transpose (bank-confilic minimize)
		else
			partial_b[local_col][local_row] = b[(stride + local_row) * m + col];	// transpose
			
		__syncthreads();

		for (int i = 0; i < block_size; ++i) {
			value += partial_a[local_row][i] * partial_b[local_col][i];	// partial_b transpose
		}
		
		__syncthreads();
	}
	if (row >= n || col >= m) return;
	
	c[m * row + col] = value;
}

#ifdef __cplusplus
extern "C"  {
#endif
void cuda2py(const float* a, float* b, float* c, const int n, const int m, const int k) {
	dim3 grid_dim(std::ceil(static_cast<float>(m) / block_size),
				  std::ceil(static_cast<float>(n) / block_size));
	dim3 block_dim(block_size, block_size);

	float* d_a;
	float* d_b;
	float* d_c;

	cudaMalloc(&d_a, sizeof(float) * n * k);
	cudaMalloc(&d_b, sizeof(float) * k * m);
	cudaMalloc(&d_c, sizeof(float) * n * m);

	cudaMemcpy(d_a, a, sizeof(float) * n * k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(float) * k * m, cudaMemcpyHostToDevice);
	cudaMemset(d_c, 0, sizeof(float) * n * m);

	optimize_cuda<float><<<grid_dim, block_dim>>>(d_a, d_b, d_c, n, m, k);

	cudaMemcpy(c, d_c, sizeof(float) * n * m, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

#ifdef __cplusplus
}
#endif
