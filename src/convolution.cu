/**
 * @File		convolution.cu
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
bool check_matrix(const T* a, const T* b, const int n, const int m, const double tolerance = 1e-5) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			int index = i * m + j;
			if (std::fabs(a[index] - b[index]) > tolerance) {
				std::cout << a[index] << " : " << b[index] << "[" << index << "]" <<std::endl;
				return false;
			}
		}
	}
	return true;
}

// n: input, m: kernel, k: output
template <typename T>
void cpu_convolution(const T* a, const T* b, T* c,
					 const int n, const int m, const int k) {
	for (int i = 0; i < k; ++i) {
		for (int j = 0; j < k; ++j) {
			T value = 0;
			for (int ki = 0; ki < m; ++ki) {
				for (int kj = 0; kj < m; ++kj) {
					int ir = i + ki - (m / 2);
					int ic = j + kj - (m / 2);
					if (ir >= 0 && ir < n && ic >= 0 && ic < n)
						value += a[n * ir + ic] * b[m * ki + kj];
				}
			}
			c[k * i + j] = value;
		}
	}
}

template <typename T>
__global__ void naive_convolution(const T* a, const T* b, T* c,
								  const int n, const int m, const int k) {
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (row < k && col < k) {
		T value = 0;
		for (int ki = 0; ki < m; ++ki) {
			for (int kj = 0; kj < m; ++kj) {
				int ir = row + ki - (m / 2);
				int ic = col + kj - (m / 2);
				if (ir >= 0 && ir < n && ic >= 0 && ic < n)
					value += a[n * ir + ic] * b[m * ki + kj];
			}
		}
		c[k * row + col] = value;
	}
}

int main(int argc, char* argv[]) {
//	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	sian::Timer timer(3);

	std::random_device rd;
	std::mt19937 engine(rd());
	std::uniform_int_distribution<unsigned int> distribution(0, 255);
	
	const int n = 512;
	const int m = 7;
	const int k = n + m - 1;
	
	unsigned int* a = new unsigned int[n * n];
	unsigned int* b = new unsigned int[m * m];
	unsigned int* c1 = new unsigned int[k * k];
	unsigned int* c2 = new unsigned int[k * k];
	unsigned int* c3 = new unsigned int[k * k];

	for (int i = 0; i < n * n; ++i) a[i] = distribution(engine);
	for (int i = 0; i < m * m; ++i) b[i] = distribution(engine);

	memset(c1, 0, sizeof(k * k));
	memset(c2, 0, sizeof(k * k));
	memset(c3, 0, sizeof(k * k));
	
	timer[0].set_name("cpu convolution");
	timer[0].start();

	cpu_convolution(a, b, c1, n, m, k);

	timer[0].stop();

	timer[1].set_name("naive convolution");
	timer[1].start();
	
	dim3 grid_dim(std::ceil(static_cast<float>(k) / block_size),
				  std::ceil(static_cast<float>(k) / block_size));
	dim3 block_dim(block_size, block_size);
	unsigned int* d_a;
	unsigned int* d_b;
	unsigned int* d_c;
	cudaMalloc(&d_a, sizeof(unsigned int) * n * n);
	cudaMalloc(&d_b, sizeof(unsigned int) * m * m);
	cudaMalloc(&d_c, sizeof(unsigned int) * k * k);

	cudaMemcpy(d_a, a, sizeof(unsigned int) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(unsigned int) * m * m, cudaMemcpyHostToDevice);
	cudaMemset(d_c, 0, sizeof(unsigned int) * k * k);

	naive_convolution<unsigned int><<<grid_dim, block_dim>>>(d_a, d_b, d_c, n, m ,k);

	cudaMemcpy(c2, d_c, sizeof(unsigned int) * k * k, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaDeviceSynchronize();
	
	timer[1].stop();
	std::cout << "c2 is correct : " << std::boolalpha << check_matrix(c1, c2, k, k) << std::endl;

	std::cout << timer;
	
	delete[] c3;
	delete[] c2;
	delete[] c1;
	delete[] b;
	delete[] a;

	return 0;
}

#endif // OS dependency
