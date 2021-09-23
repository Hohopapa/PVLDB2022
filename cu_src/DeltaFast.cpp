#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <sstream>
#include <string>
#include "graph.h"
#include "utils.h"
#include <sys/time.h>

#define M(i,j)      __M[(i) * V_size + (j)]
#define L(i,j)      __L[(i) * V_size + (j)]
#define d(i,j)      __d[(i) * V_size + (j)]

void print_debug(int* __M, const int V_size, int *A, const int A_size) {
	for (int i=0; i<A_size; i++) {
		std::cout << "A[" << i << "]=" << A[i] << ":";
		for (int j=0; j<V_size; j++) {
			std::cout << "\t" << M(i, j);
		}
		std::cout << std::endl;
	}
}


long getCurrentTime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}


// set M[u][v] for all (u, v) in A x V
// V, V_size: graph structure; starting indices in E of all vertices
// E, E_size: graph structure; neighboring vertices
// A, A_size: list of active vertices
static void build (int *V, const int V_size, int *E, const int E_size, int* A, const int A_size) {
    // i is not for all vertices but those in candidate nodes
    // note that the first index is not the vertex id but the index in A for active node
    int *__M = new int [A_size * V_size];
    for (int i=0; i<A_size; i++)
        for (int j=0; j<V_size; j++)
            M(i, j) = 0;

    // similarly, the first index is that of A
    bool *__L = new bool[A_size * V_size];
    int *__d = new int[A_size * V_size];

    bool* _Ls = new bool[V_size];
    int* _ds = new int[V_size];
    int* _F = new int[V_size];

    int* alpha = new int [V_size];
    int* beta  = new int [V_size];
    int* gamma = new int [V_size];

    // A_map: active vertices (vertex id -> A idx)
    int* A_map = new int[V_size];
    for (int i=0; i<V_size; i++) A_map[i] = -1;

#if PRINT_PROGRESS == 1
    std::cout << "1) computing BFS:" << std::endl;
    init_progress();
#endif

#if ACCTIME == 1
	long total_start = 0;
	long time_total = 0;
	long start = 0;

	long time_bfs = 0;
	long time_M = 0;
	long time_alpha = 0;
	long time_beta = 0;
	long time_gamma = 0;

	total_start = getCurrentTime();
	start = getCurrentTime();
#endif

    int cnt = 0;
    for (int i=0; i<A_size; i++) {
        // note that the first order index of d and L is that of A
        sequentialBFS(A[i], V, V_size, E, &(d(i, 0)), &(L(i, 0)), _F);
        // later, we need to access the index of A with a vertex id
        A_map[A[i]] = i;

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
        print_progress(cnt++, A_size);
#endif
    }

#if ACCTIME == 1
	time_bfs += getCurrentTime() - start;
#endif
	
#if PRINT_PROGRESS == 1
    std::cout << "2) computing expected distance decreases:" << std::endl;
    init_progress ();
#endif
	
    for (int s=0; s<V_size; s++) {
#if PRINT_PROGRESS == 1
        print_progress (s, V_size);
#endif

        int *ds;
        bool *Ls;

        // if s is in A, ds and Ls are already computed
        if (A_map[s] >= 0) {
            ds =  &(d(A_map[s], 0));
            Ls =  &(L(A_map[s], 0));
        }
        else {
#if ACCTIME == 1
			start = getCurrentTime();
#endif
            // compute single source distance with s
            sequentialBFS(s, V, V_size, E, _ds, _Ls, _F);
            ds = _ds;
            Ls = _Ls;
#if ACCTIME == 1
			time_bfs += getCurrentTime() - start;
#endif
        }

#if 0
		printf("ds: s = %d\n", s);
		for (int i=0; i<V_size; i++) {
			std::cout << " " << ds[i];
		}
		std::cout << std::endl;
		getchar();
#endif

        for (int v_idx=0; v_idx<A_size; v_idx++) {
            int v = A[v_idx];

            // if s is identical to v, skip
            if (v == s) continue;

            // for an active vertex v, its distances and reachable node list
            // are already calculated
            int*  dv = &(d(v_idx, 0));
            bool* Lv = &(L(v_idx, 0));

#if ACCTIME == 1
		start = getCurrentTime();
#endif
            for (int i=0; i<V_size; i++) alpha[i] = 0;

            // with reachable vertex t, do count sorting
            for (int t=0; t<V_size; t++) {
                if (! Lv[t]) continue;

                int dd = ds[v] + dv[t] - ds[t];
                if (dd < V_size) {
                    alpha[dd]++;
                }
            }
#if ACCTIME == 1
		time_alpha += getCurrentTime() - start;
#endif

#if 0
			printf("s = %d, v = %d\n", s, v);
			std::cout << "alpha (ds[v] = " << ds[v] << "):";
			for (int i=0; i<V_size; i++) {
				if (alpha[i] != 0) std::cout << " " <<  i << ":" << alpha[i];
			}
			std::cout << std::endl;
			getchar();
#endif

            // prefix sum
            beta[0] = alpha[0];
            gamma[0] = 0; // 0 * alpha[0]

#if ACCTIME == 1
		start = getCurrentTime();
#endif
            for (int i=1; i<ds[v]; i++) {
                beta[i] = alpha[i] + beta[i-1];
	    }

#if ACCTIME == 1
		time_beta += getCurrentTime() - start;
#endif

#if ACCTIME == 1
		start = getCurrentTime();
#endif
            for (int i=1; i<ds[v]; i++) {
                gamma[i] = i * alpha[i] + gamma[i-1];
            }
#if ACCTIME == 1
		time_gamma += getCurrentTime() - start;
#endif
			
#if ACCTIME == 1
		start = getCurrentTime();
#endif
            // compute M_dev in the device
            for (int u=0; u<V_size; u++) {
                if (! Ls[u]) continue;

                if (ds[u] <= ds[v] - 2) {
                    int delta = ds[v] - ds[u] - 1;
					// compute M[v][u]
                    M(v_idx, u) += delta * beta[delta] - gamma[delta];
                }
            }
#if ACCTIME == 1
		time_M += getCurrentTime() - start;
#endif
        }
    }

#if ACCTIME == 1
	printf("%ld\t%ld\t%ld\t%ld\t%ld\t%ld\n", time_M, time_gamma, time_beta, time_alpha, time_bfs, (getCurrentTime() - total_start));
#endif

#if DEBUG == 1
	print_debug(__M, V_size, A, A_size);
#endif

	delete[] __M;
	delete[] __L;
	delete[] __d;
	delete[] _Ls;
	delete[] _ds;
	delete[] _F;
	delete[] alpha;
	delete[] beta;
	delete[] gamma;
	delete[] A_map;
}


int main (int argc, char *argv[])
{
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
    build(V, V_size, E, E_size, A, A_size);

	delete[] V;
	delete[] E;
	delete[] A;
}
