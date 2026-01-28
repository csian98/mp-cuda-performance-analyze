/**
 * @File		naive_cuda.cu
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

#include <iostream>
#include <utility>
#include <algorithm>
#include <numeric>

#include <random>

#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>

#include "timer.h"

#if _TARGET_OS == OS_WINDOWS

#elif _TARGET_OS == OS_LINUX

// void custom_terminate_fnct(void) {
//	exit(1);
// }

static const int block_size = 32;

template <typename T>
__global__ void naive_cuda(const T* a, const T* b, T* c, const int n, const int m, const int k) {
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (row < n && col < m) {
		T value = 0;
		for (int l = 0; l < k; ++l) {
			value += a[k * row + l] * b[m * l + col];
		}
		c[row * m + col] = value;
	}
}

int main(int argc, char* argv[]) {
//	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	sian::Timer timer(1);

	std::random_device rd;
	std::mt19937 engine(rd());
	std::uniform_real_distribution<double> distribution(-1.0, 1.0);
	
	const int n = (argc > 1) ? std::stoi(argv[1]) : 1024;
	const int k = (argc > 1) ? std::stoi(argv[1]) : 1024;
	const int m = (argc > 1) ? std::stoi(argv[1]) : 1024;
	
	double* a = new double[n * k];
	double* b = new double[k * m];
	double* c = new double[n * m];

	for (int i = 0; i < n * k; ++i) a[i] = distribution(engine);
	for (int i = 0; i < k * m; ++i) b[i] = distribution(engine);

    std::cout << "####\nNaive CUDA Matrix Multiplication\n####\n" << std::endl;
	std::cout << "n: " << n << ", k: " << k << ", m: " << m << std::endl;
	
	timer[0].set_name("navie cuda");
	timer[0].start();

	dim3 grid_dim(std::ceil(static_cast<float>(m) / block_size),
				  std::ceil(static_cast<float>(n) / block_size));
	dim3 block_dim(block_size, block_size);
	double* d_a;
	double* d_b;
	double* d_c;
	cudaMalloc(&d_a, sizeof(double) * n * k);
	cudaMalloc(&d_b, sizeof(double) * k * m);
	cudaMalloc(&d_c, sizeof(double) * n * m);

	cudaMemcpy(d_a, a, sizeof(double) * n * k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(double) * k * m, cudaMemcpyHostToDevice);
	cudaMemset(d_c, 0, sizeof(double) * n * k);

	naive_cuda<double><<<grid_dim, block_dim>>>(d_a, d_b, d_c, n, m ,k);

	cudaMemcpy(c, d_c, sizeof(double) * n * m, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaDeviceSynchronize();
	
	timer[0].stop();

	std::cout << timer;
	
	delete[] c;
	delete[] b;
	delete[] a;

	return 0;
}

#endif // OS dependency
