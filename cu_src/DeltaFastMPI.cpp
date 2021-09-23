#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include <iostream>
#include <sstream>
#include <string>
#include "graph.h"
#include "utils.h"

#define M(i,j)      __M[(i) * V_size + (j)]
#define M_global(i,j)      __M_global[(i) * V_size + (j)]
#define L(i,j)      __L[(i) * V_size + (j)]
#define d(i,j)      __d[(i) * V_size + (j)]

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
	for (int i=0; i<A_size; i++) {
		std::cout << "A[" << i << "]=" << A[i] << ":";
		for (int j=0; j<V_size; j++) {
			std::cout << "\t" << M(A[i], j);
		}
		std::cout << std::endl;
	}
}

// set M[u][v] for all (u, v) in A x V
// V, V_size: graph structure; starting indices in E of all vertices
// E, E_size: graph structure; neighboring vertices
// A, A_size: list of active vertices
static void build (int pid, int nproc, int *V, const int V_size, int *E, const int E_size, int* A, const int A_size) {
	MPI_Aint size;


	/*
	 * memory allocation
	 */

	// shared
	MPI_Win __L_win, __d_win;
	bool *__L;
	int *__d;
	if (pid == 0) {
		// similarly, the first index is that of A
		// --> shared memory
		//__L = new bool[A_size * V_size];
		//__d = new int[A_size * V_size];

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

	// local --> for reduce in the last of algorithm
    // i is not for all vertices but those in candidate nodes
    // note that the first index is not the vertex id but the index in A for active node
    int *__M = new int [A_size * V_size];
    for (int i=0; i<A_size; i++)
        for (int j=0; j<V_size; j++)
            M(i, j) = 0;

	// local --> to use temporary in the inner most loop
    bool* _Ls = new bool[V_size];
    int* _ds = new int[V_size];
    int* _F = new int[V_size];

    int* alpha = new int [V_size];
    int* beta  = new int [V_size];
    int* gamma = new int [V_size];

    // local --> A_map: active vertices (vertex id -> A idx)
    int* A_map = new int[V_size];
    for (int i=0; i<V_size; i++) A_map[i] = -1;
    for (int i=0; i<V_size; i++) A_map[A[i]] = i;




	/*
	 * running bfs for computing all-sourced shorted path
	 */

#if PRINT_PROGRESS == 1
	char progress_prefix[256];
	sprintf(progress_prefix, "[pid: %d] ", pid);
    std::cout << "1) " << progress_prefix << "computing BFS:" << std::endl;
    init_progress();
    int cnt = 0;
#endif

	// distribute jobs with a stride of nproc
    for (int i=pid; i<A_size; i+=nproc) {
        // note that the first order index of d and L is that of A
        sequentialBFS(A[i], V, V_size, E, &(d(i, 0)), &(L(i, 0)), _F);
        // later, we need to access the index of A with a vertex id

#if 0
#if DEBUG == 1
		std::cout << A[i] << ":";
		for (int j=0; j<V_size; j++) {
			std::cout << " " << d(i, j);
		}
		std::cout << std::endl;
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



	/*
	 * computing all-paired expected distance decreases
	 */
	
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

        int *ds;
        bool *Ls;

        // if s is in A, ds and Ls are already computed
        if (A_map[s] >= 0) {
            ds =  &(d(A_map[s], 0));
            Ls =  &(L(A_map[s], 0));
        }
        else {
            // compute single source distance with s
            sequentialBFS(s, V, V_size, E, _ds, _Ls, _F);
            ds = _ds;
            Ls = _Ls;
        }

        for (int v_idx=0; v_idx<A_size; v_idx++) {
            int v = A[v_idx];

            // if s is identical to v, skip
            if (v == s) continue;

            // for an active vertex v, its distances and reachable node list
            // are already calculated
            int*  dv = &(d(v_idx, 0));
            bool* Lv = &(L(v_idx, 0));

            for (int i=0; i<V_size; i++) alpha[i] = 0;

            // with reachable vertex t, do count sorting
            for (int t=0; t<V_size; t++) {
                if (! Lv[t]) continue;

                int dd = ds[v] + dv[t] - ds[t];
                if (dd < V_size) {
                    alpha[dd]++;
                }
            }

            // prefix sum
            beta[0] = alpha[0];
            gamma[0] = 0; // 0 * alpha[0]
            for (int i=1; i<ds[v]; i++) {
                beta[i] = alpha[i] + beta[i-1];
                gamma[i] = i * alpha[i] + gamma[i-1];
            }
			
            // compute M_dev in the device
            for (int u=0; u<V_size; u++) {
                if (A_map[u] < 0 || ! Ls[u]) continue;

                if (ds[u] <= ds[v] - 2) {
                    int delta = ds[v] - ds[u] - 1;
					// compute M[v][u]
                    M(v_idx, u) += delta * beta[delta] - gamma[delta];
                }
            }
        }
    }


	/*
	 * reducing for completing all-paired expected distance decrease
	 */
	
	// reduce M
	int *__M_global = NULL;
	if (pid == 0) {
		__M_global = new int [A_size * V_size];
		//for (int i=0; i<A_size; i++)
		//	for (int j=0; j<V_size; j++)
		//		M_global(i, j) = 0;
	}

	MPI_Reduce(
			(void*)__M, /* sendbuf */
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

	delete[] alpha;
	delete[] beta;
	delete[] gamma;
	delete[] _Ls;
	delete[] _ds;
	delete[] _F;
	delete[] A_map;

	delete[] __M;
	if (pid == 0) delete[] __M_global;

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
}
