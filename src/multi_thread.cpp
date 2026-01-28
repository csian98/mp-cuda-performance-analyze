/**
 * @File		multi_thread.cpp
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

template <typename T>
void multi_thread(const T* a, const T* b, T* c, const int n, const int m, const int k,
				  const int thread_index, const int thread_num) {
	const int tasks = std::ceil(static_cast<float>(m) / thread_num);
	for (int i = 0; i < n; ++i) {
		for (int j = thread_index * tasks; j < (thread_index + 1) * tasks; ++j) {
			T value = 0;
			for (int l = 0; l < k; ++l) {
				if (j < m)
					value += a[i * k + l] * b[l * m + j];
			}
			c[i * m + j] += value;
		}
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

	auto thread_num = std::thread::hardware_concurrency();

    std::cout << "####\nMulti Thread Matrix Multiplication\n####\n" << std::endl;
	std::cout << "n: " << n << ", k: " << k << ", m: " << m << std::endl;
	std::cout << "number of threads: # " << thread_num << std::endl;
	
	timer[0].set_name("multi thread");
	timer[0].start();

	std::vector<std::thread> threads;
	
	for (int i = 0; i < thread_num; ++i)
		threads.emplace_back(&multi_thread<double>, a, b, c, n, m, k, i, thread_num);

	for (auto& thread: threads) thread.join();
	
	timer[0].stop();

	std::cout << timer;
	
	delete[] c;
	delete[] b;
	delete[] a;

	return 0;
}

#endif // OS dependency
