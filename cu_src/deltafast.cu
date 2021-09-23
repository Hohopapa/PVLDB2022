#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "constants.h"
#include "kernels.cuh"
#include "utils.h"
#include "deltafast.h"

void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool bcao);
void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length, bool bcao);
void scanSmallDeviceArray(int *d_out, int *d_in, int length, bool bcao);

void compute_alpha (
		bool *Lv, int *ds, int *dv, int ds_v, int V_size,
		int *alpha) {

	const int blocks = V_size / ELEMENTS_PER_BLOCK;
	if (V_size % ELEMENTS_PER_BLOCK == 0)
		k_compute_alpha <<<blocks, 
						THREADS_PER_BLOCK>>> (
//						sizeof(int)*V_size*2 + sizeof(bool)*V_size>>> (
				Lv, ds, dv, ds_v, V_size, 
				alpha);
	else
		k_compute_alpha <<<blocks+1,
						THREADS_PER_BLOCK>>> (
//						sizeof(int)*V_size*2 + sizeof(bool)*V_size>>> (
				Lv, ds, dv, ds_v, V_size, 
				alpha);
}

void compute_1moment_alpha (int *alpha, int max_exclusive) {
	const int blocks = max_exclusive / ELEMENTS_PER_BLOCK;

	/*
	if (V_size % ELEMENTS_PER_BLOCK == 0)
		k_compute_1moment_alpha <<<blocks, 
								THREADS_PER_BLOCK,
								sizeof(int)*V_size>>> (alpha, ds, v);
	else
		k_compute_1moment_alpha <<<blocks+1, 
								THREADS_PER_BLOCK,
								sizeof(int)*V_size>>> (alpha, ds, v);
								*/
	if (max_exclusive % ELEMENTS_PER_BLOCK == 0)
		k_compute_1moment_alpha <<<blocks, 
								THREADS_PER_BLOCK>>> (alpha, max_exclusive);
	else
		k_compute_1moment_alpha <<<blocks+1, 
								THREADS_PER_BLOCK>>> (alpha, max_exclusive);
}

void increase_M (
		int *beta, int *gamma, int V_size, bool *Ls, int *ds, int ds_v,
		int *M_v_idx) {
	const int blocks = V_size / ELEMENTS_PER_BLOCK;

	if (V_size % ELEMENTS_PER_BLOCK == 0)
		k_increase_M <<<blocks, 
					 THREADS_PER_BLOCK>>> (
//					 sizeof(int)*V_size*3+sizeof(bool)*V_size>>> (
				beta, gamma, V_size, Ls, ds, ds_v, M_v_idx);
	else
		k_increase_M <<<blocks+1,
					 THREADS_PER_BLOCK>>> (
//					 sizeof(int)*V_size*3+sizeof(bool)*V_size>>> (
				beta, gamma, V_size, Ls, ds, ds_v, M_v_idx);
}

static int *d_sums, *d_incr;

void init_kernel (int length) {
	const int blocks = length / ELEMENTS_PER_BLOCK;
	cudaMalloc((void **)&d_sums, blocks * sizeof(int));
	cudaMalloc((void **)&d_incr, blocks * sizeof(int));
}

void fin_kernel () {
	cudaFree (d_sums);
	cudaFree (d_incr);
}

/* call for example, 
       scan (beta, alpha, ds[v], true);
	   scan (gamma, alpha, ds[v], true);
*/
void scan(int *d_out, int *d_in, int length, bool bcao) {
	if (length > ELEMENTS_PER_BLOCK) {
		scanLargeDeviceArray(d_out, d_in, length, bcao);
	}
	else {
		scanSmallDeviceArray(d_out, d_in, length, bcao);
	}
}


void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
	int remainder = length % (ELEMENTS_PER_BLOCK);
	if (remainder == 0) {
		scanLargeEvenDeviceArray(d_out, d_in, length, bcao);
	}
	else {
		// perform a large scan on a compatible multiple of elements
		int lengthMultiple = length - remainder;
		scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, bcao);

		// scan the remaining elements and add the (inclusive) last element of the large scan to this
		int *startOfOutputArray = &(d_out[lengthMultiple]);
		scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder, bcao);

		add<<<1, remainder>>>(startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
	}
}

void scanSmallDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
	int powerOfTwo = nextPowerOfTwo(length);

	if (bcao) {
		prescan_arbitrary << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> >(d_out, d_in, length, powerOfTwo);
	}
	else {
		prescan_arbitrary_unoptimized<< <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> >(d_out, d_in, length, powerOfTwo);
	}
}

void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
	const int blocks = length / ELEMENTS_PER_BLOCK;
	const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

	if (bcao) {
		prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
	}
	else {
		prescan_large_unoptimized<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
	}

	const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
		// perform a large scan on the sums arr
		scanLargeDeviceArray(d_incr, d_sums, blocks, bcao);
	}
	else {
		// only need one block to scan sums arr so can use small scan
		scanSmallDeviceArray(d_incr, d_sums, blocks, bcao);
	}

	add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);
}



/* 
 * misc. utils used in TestScan 
 */
void sequential_scan(int* output, int* input, int length) {
	output[0] = 0; // since this is a prescan, not a scan
	for (int j = 1; j < length; ++j)
	{
		output[j] = input[j - 1] + output[j - 1];
	}
}

