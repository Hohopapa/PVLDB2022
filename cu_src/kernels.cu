#include <stdio.h>
#include <cooperative_groups.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <device_functions.h>

#include "constants.h"
#include "kernels.cuh"

#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

__global__ void CUDA_SET_MAXDIST_KERNEL(bool *L, int *d, int V_size) {

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < V_size) {
		if (! L[id]) d[id] = V_size - 1;
	}
}

__global__ void CUDA_BFS_KERNEL(
		int *Va, 
		const int V_size,
		int *Ea, 
		bool *Fa, /* frontier */
		bool *Xa, /* visisted */
		int *Ca,  /* distance */
		bool *done) {

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id == 0) *done = true;
	__syncthreads();
	//if (id > NUM_NODES)
	//	*done = false;

	if (id < V_size && Fa[id] == true && Xa[id] == false) {
		//printf("%d ", id); //This printf gives the order of vertices in BFS	
		Fa[id] = false;
		Xa[id] = true;
		__syncthreads();
		//int k = 0;
		//int i;

		int start = Va[id];
		//int start = Va[id].start;
		int end = Va[id+1]; // XXX: exclusive end
		//int end = start + Va[id].length;
        //printf("%d, %d\n", start, end);
		for (int i = start; i < end; i++) {
			int nid = Ea[i];
	//		printf ("\t%d", nid);

			if (Xa[nid] == false) {
				Ca[nid] = Ca[id] + 1;
				Fa[nid] = true;
                //printf("%d frontier set true\n", nid);
				*done = false;
			}
		}
	}
}

__global__ void k_bfs (
		const int source,
		int *V, 
		const int V_size,
		int *E, 
		int *d,
		bool *L, /* visisted */
		bool *F) { /* frontier */
		
	__shared__ bool done;
	__shared__ int _d;
	//const int id = threadIdx.x + blockIdx.x * blockDim.x;
	const int tid = threadIdx.x;
	const int blocksize = blockDim.x;

	// this thread is in charge of all entries of index
	// (blocksize * k + tid) with k = 0, 1, 2, ...
	for (int id=tid ; id < V_size; id += blocksize) {
		if (id == source) {
			F[id] = true;
			d[id] = 0;
		}
		else {
			F[id] = false;
			d[id] = -1;
		}

		L[id] = false;
	}

	if (tid == 0) _d = 1;
	__syncthreads();

	int expand[256];
	do {
		if (tid == 0) done = true;
		//bool expanded = false;

		int n_expand = 0;
		for (int id = tid; id < V_size; id += blocksize) {
			if (id < V_size && F[id] == true) { // && L[id] == false) {
				//printf("visited: %d\n", id); //This printf gives the order of vertices in BFS	
				F[id] = false;
				//L[id] = true;
				//expanded = true;
				expand[n_expand++] = id;
			}
		}

		__syncthreads();
//		g.sync();
//cooperative_groups::sync(g);

		for (int i=0; i<n_expand; i++) {
			//int __d = d[expand[i]];

//if (id < V_size) {
//		if (expanded) {
			int start = V[expand[i]];
			int end = V[expand[i]+1]; // XXX: exclusive end

			for (int j = start; j < end; j++) {
				int nid = E[j]; // neighbor's id

				//if (L[nid] == false) {
				if (d[nid] < 0) {
					d[nid] = _d;
					//atomicAdd(done, 1);
					F[nid] = true;
					done = false;
				}
			}
			//atomicAdd(done, -1);
		}

		__syncthreads();

		if (tid == 0) _d += 1;
		__syncthreads();

//		g.sync();
//printf("[%d] in: %d\n", cnt, id);
//g.sync();
//printf("[%d] out: %d\n", cnt++, id);
//printf("cnt: %d\n", *done);
	}
	while (! done);

	//free(expand);

	for (int id=tid ; id < V_size; id += blocksize) {
		if (d[id] < 0) d[id] = V_size - 1;
		else L[id] = true;
		//if (! L[id]) d[id] = V_size - 1;
	}
}

__global__ void k_compute_alpha (
		bool *Lv,
		int *ds,
		int *dv,
		int ds_v,
		int V_size,
		int *alpha) {

	int t = threadIdx.x + blockIdx.x * blockDim.x;

	// JIWON XXX
	//extern __shared__ int sharedarr[];

	//int *fast_ds = sharedarr;
	//int *fast_dv = &(sharedarr[V_size]);
	//bool *fast_Lv = (bool*) &(sharedarr[V_size*2]);

	if (t < V_size) {
		//fast_dv[t] = dv[t];
		//fast_ds[t] = ds[t];
		//fast_Lv[t] = Lv[t];
		alpha[t] = 0;

		__syncthreads();
	}

	if (t < V_size && Lv[t]) {
		int dd = ds_v + dv[t] - ds[t];
		if (dd < V_size) {
			atomicAdd( &(alpha[dd]), 1 );
		}
	}
}

__global__ void k_compute_1moment_alpha (int *alpha, int max_exclusive) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// JIWON XXX
	/*
	extern __shared__ int fast_alpha[];
	__shared__ int max_exclusive;
	if (threadIdx.x == 0) max_exclusive = ds[v];
	__syncthreads();

	if (i < max_exclusive) fast_alpha[i] = alpha[i];
	__syncthreads();
	
	if (i < max_exclusive) {
		fast_alpha[i] *= i;
	}
	__syncthreads();

	if (i < max_exclusive) {
		alpha[i] = fast_alpha[i];
	}
	*/

	if (i < max_exclusive) {
		alpha[i] *= i;
	}
}

__global__ void k_increase_M (
		int *beta, int *gamma, int V_size, bool *Ls, int *ds, int ds_v,
		int *M_v_idx) {

	// JIWON XXX
	//extern __shared__ int sharedarr[];

	int u = threadIdx.x + blockIdx.x * blockDim.x;

	/*
	int *fast_ds = sharedarr;
	int *fast_beta = &(sharedarr[V_size]);
	int *fast_gamma = &(sharedarr[V_size*2]);
	bool *fast_Ls = (bool*) &(sharedarr[V_size*3]);

	if (u < V_size) {
		fast_Ls[u] = Ls[u];
		fast_gamma[u] = gamma[u];
		fast_beta[u] = beta[u];
		fast_ds[u] = ds[u];

		__syncthreads();
	}
	*/

	if (u < V_size && Ls[u]) {
		int delta = ds_v - ds[u] - 1;
		if (delta > 0) {
			// fill sM_v_idx
			M_v_idx[u] += delta * beta[delta] - gamma[delta];
		}
	}

	// syncthreads()
	// M_v_idx[v] = sM_v_idx[];
}






/*
 * prefix sum
 */

__global__ void prescan_arbitrary(int *output, int *input, int n, int powerOfTwo)
{
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	
	if (threadID < n) {
		temp[ai + bankOffsetA] = input[ai];
		temp[bi + bankOffsetB] = input[bi];
	}
	else {
		temp[ai + bankOffsetA] = 0;
		temp[bi + bankOffsetB] = 0;
	}
	

	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) {
		temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0; // clear the last element
	}

	for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadID < n) {
		output[ai] = temp[ai + bankOffsetA];
		output[bi] = temp[bi + bankOffsetB];
	}
}

__global__ void prescan_arbitrary_unoptimized(int *output, int *input, int n, int powerOfTwo) {
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	if (threadID < n) {
		temp[2 * threadID] = input[2 * threadID]; // load input into shared memory
		temp[2 * threadID + 1] = input[2 * threadID + 1];
	}
	else {
		temp[2 * threadID] = 0;
		temp[2 * threadID + 1] = 0;
	}


	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) { temp[powerOfTwo - 1] = 0; } // clear the last element

	for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadID < n) {
		output[2 * threadID] = temp[2 * threadID]; // write results to device memory
		output[2 * threadID + 1] = temp[2 * threadID + 1];
	}
}


__global__ void prescan_large(int *output, int *input, int n, int *sums) {
	extern __shared__ int temp[];

	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;
	
	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = input[blockOffset + ai];
	temp[bi + bankOffsetB] = input[blockOffset + bi];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (threadID == 0) { 
		sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	} 
	
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[blockOffset + ai] = temp[ai + bankOffsetA];
	output[blockOffset + bi] = temp[bi + bankOffsetB];
}

__global__ void prescan_large_unoptimized(int *output, int *input, int n, int *sums) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;

	extern __shared__ int temp[];
	temp[2 * threadID] = input[blockOffset + (2 * threadID)];
	temp[2 * threadID + 1] = input[blockOffset + (2 * threadID) + 1];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (threadID == 0) {
		sums[blockID] = temp[n - 1];
		temp[n - 1] = 0;
	}

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[blockOffset + (2 * threadID)] = temp[2 * threadID];
	output[blockOffset + (2 * threadID) + 1] = temp[2 * threadID + 1];
}


__global__ void add(int *output, int length, int *n) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n[blockID];
}

__global__ void add(int *output, int length, int *n1, int *n2) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}
