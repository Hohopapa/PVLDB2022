#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
//#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "kernels.cuh"
#include "bfs.h"


void CUDA_BFS(int *V, const int V_size, int *E, bool *F, bool *L, int *d, bool *done) {
	const int blocks = V_size / ELEMENTS_PER_BLOCK;
    if (V_size % ELEMENTS_PER_BLOCK == 0)
    	CUDA_BFS_KERNEL <<<blocks, THREADS_PER_BLOCK >>>(V, V_size, E, F, L, d, done);
    else
    	CUDA_BFS_KERNEL <<<blocks+1, THREADS_PER_BLOCK >>>(V, V_size, E, F, L, d, done);
}

void CUDA_SET_MAXDIST (bool* _L_dev, int *_d_dev, const int V_size) {
	const int blocks = V_size / ELEMENTS_PER_BLOCK;
	if (V_size % ELEMENTS_PER_BLOCK == 0)
		CUDA_SET_MAXDIST_KERNEL <<<blocks, THREADS_PER_BLOCK>>> (_L_dev, _d_dev, V_size);
	else
		CUDA_SET_MAXDIST_KERNEL <<<blocks+1, THREADS_PER_BLOCK>>> (_L_dev, _d_dev, V_size);
}

void run_bfs (
		const int source, int* _V_dev, const int V_size, int* _E_dev,  /* input */
		int *_d_dev, bool *_L_dev,  /* output */
		bool *_F_dev) {


	/*
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	if (!deviceProp.cooperativeLaunch) {
		printf("\nSelected GPU (%d) does not support Cooperative Kernel Launch, Waiving the run\n", 0);
		exit(1);
	}
	// Statistics about the GPU device
	printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n", deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	const int blocks = V_size / ELEMENTS_PER_BLOCK;
    if (V_size % ELEMENTS_PER_BLOCK == 0)
		k_bfs<<<blocks, THREADS_PER_BLOCK >>>(
				source, _V_dev, V_size, _E_dev,
				_d_dev, _L_dev, _F_dev, _done_dev);
    else
    	k_bfs<<<blocks+1, THREADS_PER_BLOCK >>>(
				source, _V_dev, V_size, _E_dev,
				_d_dev, _L_dev, _F_dev, _done_dev);
				*/

	// run with a single block
	k_bfs <<<1, THREADS_PER_BLOCK>>> (
			source, _V_dev, V_size, _E_dev,
			_d_dev, _L_dev, _F_dev);
}

void cudaBFS(
		const int source, int* _V_dev, const int V_size, int* _E_dev,  /* input */
		int *_d_dev, bool *_L_dev,  /* output */
		bool *_F_dev,  /* frontier */
		bool *_done_dev) { /* no frontier any more? */

	int count = 0;

	// init d and L
	cudaMemset(_d_dev, 0, sizeof(int) * V_size);
	cudaMemset(_L_dev, 0, sizeof(bool) * V_size);

	// put source into frontier
	cudaMemset(_F_dev, 0, sizeof(bool) * V_size);
	bool T = true;
	cudaMemcpy(&(_F_dev[source]), &T, sizeof(bool), cudaMemcpyHostToDevice);

	//bool *tmp = new bool [V_size];
	//cudaMemcpy(tmp, L_dev, sizeof(bool) * V_size, cudaMemcpyDeviceToHost);
	//std::cout << "check L_dev source [" << source << "]:" << std::endl;
	//for (int i=0; i<V_size; i++) std::cout << " " << tmp[i];
	//std::cout << std::endl;

	//printf("Order: \n\n");
	bool done;
	do {
		count++;
		//done = true;
		//cudaMemcpy(_done_dev, &done, sizeof(bool), cudaMemcpyHostToDevice);
		CUDA_BFS (_V_dev, V_size, _E_dev, _F_dev, _L_dev, _d_dev, _done_dev);
		cudaMemcpy(&done, _done_dev, sizeof(bool), cudaMemcpyDeviceToHost);
	} while (!done);

	CUDA_SET_MAXDIST (_L_dev, _d_dev, V_size);
	//printf("Number of times the kernel is called : %d \n", count);
}
