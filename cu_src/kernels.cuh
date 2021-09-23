#ifndef KERNELS_CUH
#define KERNELS_CUH
/*
 * BFS kernel functions
 */
__global__ void CUDA_SET_MAXDIST_KERNEL(bool *L, int *d, int V_size);
__global__ void CUDA_BFS_KERNEL( int *Va, const int V_size, int *Ea, bool *Fa, bool *Xa, int *Ca, bool *done);
__global__ void k_bfs (const int source, int *V, const int V_size, int *E, int *d, bool *L, bool *F);

/*
 * DeltaFast kernel functions
 */
__global__ void k_compute_alpha ( bool *Lv, int *ds, int *dv, int ds_v, int V_size, int *alpha);
__global__ void k_compute_1moment_alpha (int *alpha, int max_exclusive);
__global__ void k_increase_M (
		int *beta, int *gamma, int V_size, bool *Ls, int *ds, int ds_v,
		int *M_v_idx);

/*
 * Prefix scan kernel functions
 */
__global__ void prescan_arbitrary(int *g_odata, int *g_idata, int n, int powerOfTwo);
__global__ void prescan_arbitrary_unoptimized(int *g_odata, int *g_idata, int n, int powerOfTwo);

__global__ void prescan_large(int *g_odata, int *g_idata, int n, int* sums);
__global__ void prescan_large_unoptimized(int *output, int *input, int n, int *sums);

__global__ void add(int *output, int length, int *n1);
__global__ void add(int *output, int length, int *n1, int *n2);
#endif
