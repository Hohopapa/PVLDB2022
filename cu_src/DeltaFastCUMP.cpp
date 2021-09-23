#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "bfs.h"
#include "deltafast.h"
#include "graph.h"
#include "utils.h"

#include <iostream>
#include <sstream>
#include <string>

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define M_host(i,j)     __M_host[(i) * V_size + (j)]
#define M_global(i,j)   __M_global[(i) * V_size + (j)]
#define M_dev(i,j)     	__M_dev[(i) * V_size + (j)]
#define L_dev(i,j)      __L_dev[(i) * V_size + (j)]
#define d_dev(i,j)      __d_dev[(i) * V_size + (j)]
#define L_host(i,j)     __L[(i) * V_size + (j)]
#define d_host(i,j)     __d[(i) * V_size + (j)]

void print_debug_d(int* __d, const int V_size, int *A, const int A_size) {
#define d(i,j)      __d[(i) * V_size + (j)]

	std::cout << "**** d: ****" << std::endl;
	for (int i=0; i<A_size; i++) {
		std::cout << "A[" << i << "]=" << A[i] << ":";
		for (int j=0; j<V_size; j++) {
			std::cout << "\t" << d(i, j);
		}
		std::cout << std::endl;
	}
}

void print_debug(int* __M, const int V_size, int *A, const int A_size) {
#define M(i,j)      __M[(i) * V_size + (j)]

	std::cout << "**** M: ****" << std::endl;
	for (int i=0; i<A_size; i++) {
		std::cout << "A[" << i << "]=" << A[i] << ":";
		for (int j=0; j<V_size; j++) {
			std::cout << "\t" << M(i, j);
		}
		std::cout << std::endl;
	}
}

// set M[u][v] for all (u, v) in A x V
// V, V_size: graph structure; starting indices in E of all vertices
// E, E_size: graph structure; neighboring vertices
// A, A_size: active vertices; a vertex is set true if it is active
static void build (int pid, int nproc, int *V, const int V_size, int *E, const int E_size, int* A, const int A_size) {
	MPI_Aint size;


	/*
	 * computing bfs: without gpus
	 */

	// shared
	MPI_Win __L_win, __d_win;
	bool *__L;
	int *__d;

	// select device
	cudaSetDevice(pid);
	
	// L and d
	//     __L (handler = __L_win), __d (handler = __d_win) : shared memory in host
	//     __L_dev, __d_dev : in each gpu device (inevitable)
	if (pid == 0) {
		// similarly, the first index is that of A --> shared memory
		size = A_size * V_size * sizeof(bool);
		MPI_Win_allocate_shared(
				size, /* size of array */
				sizeof(bool), /* the shared memory will be access as a int array */
				MPI_INFO_NULL,
				MPI_COMM_WORLD, 
				(void**) &__L, 
				&__L_win); /* window object returned by the call (handle) */

		size = A_size * V_size * sizeof(int);
		MPI_Win_allocate_shared(
				size, /* size of array */
				sizeof(int), /* the shared memory will be access as a int array */
				MPI_INFO_NULL,
				MPI_COMM_WORLD, 
				(void**) &__d, 
				&__d_win); /* window object returned by the call (handle) */
	}
	else {
		int disp_unit;

		MPI_Win_allocate_shared(0, sizeof(bool), MPI_INFO_NULL,
				MPI_COMM_WORLD, (void**)&__L, &__L_win);
		MPI_Win_shared_query(__L_win, 0, &size, &disp_unit, (void**)&__L);

		MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL,
				MPI_COMM_WORLD, (void**)&__d, &__d_win);
		MPI_Win_shared_query(__d_win, 0, &size, &disp_unit, (void**)&__d);
	}

	// graph resides in device memry
	int* V_dev;
	cudaMalloc((void**) &V_dev, sizeof(int) * (V_size + 1));
	cudaMemcpy(V_dev, V, sizeof(int) * (V_size + 1), cudaMemcpyHostToDevice);

	int* E_dev;
	cudaMalloc((void**) &E_dev, sizeof(int) * E_size);
	cudaMemcpy(E_dev, E, sizeof(int) * E_size, cudaMemcpyHostToDevice);

	bool *__L_dev;
	cudaMalloc((void**) &__L_dev, sizeof(bool) * V_size * A_size);
	int  *__d_dev;
	cudaMalloc((void**) &__d_dev, sizeof(int) * V_size * A_size);
	bool* F_dev;
	cudaMalloc((void**) &F_dev,  sizeof(bool) *  V_size);
	
	bool *done_dev;
	cudaMalloc((void**) &done_dev, sizeof(bool));

    // A_map: active vertices (vertex id -> A idx)
    int* A_map = new int[V_size];
    for (int i=0; i<V_size; i++) A_map[i] = -1;
    int* _F = new int[V_size];

#if PRINT_PROGRESS == 1
	char progress_prefix[256];
	sprintf(progress_prefix, "[pid: %d] ", pid);
    std::cout << "1) " << progress_prefix << "computing BFS:" << std::endl;
	init_progress();
    int cnt = 0;
#endif

	init_kernel(V_size);

    for (int i=pid; i<A_size; i+=nproc) {
		// note that the first order index of d and L is that of A
		cudaBFS(A[i], V_dev, V_size, E_dev, &(d_dev(i, 0)), &(L_dev(i, 0)), F_dev, done_dev);
		cudaMemcpy(&(L_host(i, 0)), &(L_dev(i, 0)), sizeof(bool) * V_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(&(d_host(i, 0)), &(d_dev(i, 0)), sizeof(int) * V_size, cudaMemcpyDeviceToHost);

#if 0
#if DEBUG == 1
		// print result for debugging
		int *tmp_d = new int [V_size];
		cudaMemcpy(tmp_d, &(d_dev(i, 0)), sizeof(int) * V_size, cudaMemcpyDeviceToHost);
		std::cout << "d[" << A[i] << "]:";
		for (int j=0; j<V_size; j++) {
			std::cout << " " << tmp_d[j];
		}
		std::cout << std::endl;
		delete[] tmp_d;
#endif
#endif

#if PRINT_PROGRESS == 1
        print_progress(progress_prefix, cnt++, A_size / nproc);
#endif
	}

	MPI_Barrier (MPI_COMM_WORLD);

//#if DEBUG == 1
//	if (pid == 0) print_debug_d (__d, V_size, A, A_size);
//#endif

	// copy from shared memory over multiporcesses to device
	cudaMemcpy(__L_dev, __L, sizeof(bool) * V_size * A_size, cudaMemcpyHostToDevice); 
	cudaMemcpy(__d_dev, __d, sizeof(int) * V_size * A_size, cudaMemcpyHostToDevice);

	/*
	 * computing all-paired expected distance decreases with gpus
	 */

	// in device
	int *__M_dev;
	cudaMalloc((void**) &__M_dev, sizeof(int) * V_size * A_size);
	cudaMemset(__M_dev, 0x0, sizeof(int) * V_size * A_size);

	bool* Ls_dev;
	cudaMalloc((void**) &Ls_dev, sizeof(bool) * V_size);
	int* ds_dev;
	cudaMalloc((void**) &ds_dev, sizeof(int) *  V_size);

	int *alpha, *beta, *gamma;
	cudaMalloc((void**) &alpha, sizeof(int) * (V_size + 1));
	cudaMalloc((void**) &beta,  sizeof(int) * (V_size + 1)); // exclusive prefix sum
	cudaMalloc((void**) &gamma, sizeof(int) * (V_size + 1));

#if PRINT_PROGRESS == 1
    std::cout << "2) " << progress_prefix << "computing expected distance decreases:" << std::endl;
	init_progress ();
	cnt = 0;
#endif

	// distribute jobs with a stride of nproc
    for (int s=pid; s<V_size; s+=nproc) {
#if PRINT_PROGRESS == 1
        print_progress (progress_prefix, cnt++, V_size / nproc);
#endif

		// memory in device
		int *ds;
		bool *Ls;

        // if s is in A, ds and Ls are already computed
        if (A_map[s] >= 0) {
            ds =  &(d_dev(A_map[s], 0));
            Ls =  &(L_dev(A_map[s], 0));
        }
        else {
            // compute single source distance with s
            cudaBFS(s, V_dev, V_size, E_dev, ds_dev, Ls_dev, F_dev, done_dev);
            ds = ds_dev;
            Ls = Ls_dev;
        }

		for (int v_idx=0; v_idx<A_size; v_idx++) {
			int v = A[v_idx];

			// if s is identical to v, skip
			if (v == s) continue;

			// for an active vertex v, its distances and reachable node list
			// are already calculated
            int*  dv = &(d_dev(v_idx, 0));
            bool* Lv = &(L_dev(v_idx, 0));

			int ds_v;
			cudaMemcpy(&ds_v, (void*)&(ds[v]), sizeof(int), cudaMemcpyDeviceToHost);

			// initialize alpha
			cudaMemset(alpha, 0x0, sizeof(int) * (ds_v + 1));

			// with reachable vertex t, do count sorting
			compute_alpha(Lv, ds, dv, v, V_size, alpha);

#if 0
			int* _alpha = new int [ds_v];
			cudaMemcpy(_alpha, alpha, sizeof(int) * ds_v, cudaMemcpyDeviceToHost);
			std::cout << "alpha (ds[v] = " << ds_v << "):";
			for (int i=0; i<ds_v; i++) {
				std::cout << " " <<  _alpha[i];
			}
			std::cout << std::endl;
			delete[] _alpha;
#endif

			// prefix sum
			// in: alpha -> out: beta
			// in: alpha * d -> out: gamma
			// XXX: compute beta and gamma with separate calls of kernel
			//      function

			//cudaMemset (beta, 0, sizeof(int) * (ds_v + 1));
			//cudaMemset (gamma, 0, sizeof(int) * (ds_v + 1));

			scan (beta, alpha, ds_v + 1, true);
			compute_1moment_alpha (alpha, ds_v);
			scan (gamma, alpha, ds_v + 1, true);

			/*
            beta[0] = alpha[0];
            gamma[0] = 0; // 0 * alpha[0]
            for (int i=1; i<ds[v]; i++) {
                beta[i] = alpha[i] + beta[i-1];
                gamma[i] = i * alpha[i] + gamma[i-1];
            }
			*/
			
            // compute M_dev in the device
			increase_M (&(beta[1]), &(gamma[1]), V_size, Ls, ds, ds_v, &(M_dev(v_idx, 0)));
			/*
            for (int u=0; u<V_size; u++) {
                if (! Ls[u]) continue;

                if (ds[u] <= ds[v] - 2) {
                    int delta = ds[v] - ds[u] - 1;
					// compute M[v][u]
                    M(v_idx, u) += delta * beta[delta] - gamma[delta];
                }
            }
			*/
		}
	}

	// copy M from device to host
	int *__M_host = new int [A_size * V_size];
	cudaMemcpy(__M_host, __M_dev, sizeof(int) * A_size * V_size, cudaMemcpyDeviceToHost);

	int *__M_global = NULL;
	if (pid == 0) __M_global = new int [A_size * V_size];

	MPI_Reduce(
			(void*)__M_host, /* sendbuf */
			(void*)__M_global, /* recvbuf */
			A_size * V_size, /* number of element to send from sendbuf to recvbuf */
			MPI_INT, /* data type of element in buf */
			MPI_SUM,
			0, /* root */
			MPI_COMM_WORLD);

#if DEBUG == 1
	if (pid == 0) {
		print_debug(__M_global, V_size, A, A_size);
	}
#endif

	fin_kernel ();

	if (pid == 0) delete[] __M_global;
	delete[] __M_host;
	delete[] A_map;

	cudaFree (V_dev);
	cudaFree (E_dev);
	cudaFree (__M_dev);
	cudaFree (__L_dev);
	cudaFree (__d_dev);
	cudaFree (Ls_dev);
	cudaFree (ds_dev);
	cudaFree (F_dev);
	cudaFree (alpha);
	cudaFree (beta);
	cudaFree (gamma);
	cudaFree (done_dev);

	MPI_Win_free(&__L_win);
	MPI_Win_free(&__d_win);
}


int main (int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	int rank, nproc;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc); 

    if (argc != 7 && argc != 5) {
        std::cerr << "usage: <V: start index file> <size of V> <E: edge file> <size of E> [<A: active vertex set> <size of A>]" << std::endl;
        exit (1);
    }

	int ngpus;
	cudaGetDeviceCount(&ngpus);

	if (nproc != ngpus) {
		std::cerr << "error: there are " << ngpus << " gpus available" << std::endl;
		exit(1);
	}

    const char *V_file = argv[1];
    const int V_size = atoi(argv[2]);
    const char *E_file = argv[3];
    const int E_size = atoi(argv[4]);
    char *A_file;
    int A_size;
	if (argc == 7) {
		A_file = argv[5];
		A_size = atoi(argv[6]);
	}
	else {
		A_file = NULL;
		A_size = V_size;
	}

    int* V = new int [V_size + 1]; // use the last entry for the (exclusive) ending index of array
    int* E = new int [E_size];
    int* A = new int [A_size];

    read_graph (V_file, V_size, E_file, E_size, V, E);
    read_active_vertex (A_file, V_size, A, A_size);
    build(rank, nproc, V, V_size, E, E_size, A, A_size);

	MPI_Finalize();

	delete[] V;
	delete[] E;
	delete[] A;

	return 0;
}
