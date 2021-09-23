#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <errno.h>
#include <error.h>

#include <sys/shm.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>

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
//#include "device_functions.h"

#define M_host(i,j)     __M_host[(i) * V_size + (j)]
#define M_dev(i,j)     	__M_dev[(i) * V_size + (j)]
#define L_dev(i,j)      __L_dev[(i) * V_size + (j)]
#define d_dev(i,j)      __d_dev[(i) * V_size + (j)]

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>


#define NGPUCARDS		4

int nproc = 0;
int *shm = NULL;
int *M_shm = NULL;
int Cid;     /* queue segment id */
int Mid;	 /* M segmenmt id */
int base_pid, status, pid;
int my_id = -1;

// semaphore
struct sembuf mlock[2] = { 0, 0, 0, 0, 1, SEM_UNDO };
struct sembuf M_mlock[2] = { 0, 0, 0, 0, 1, SEM_UNDO };
struct sembuf munlock[2] = { 0, -1, (IPC_NOWAIT | SEM_UNDO) };
struct sembuf M_munlock[2] = { 0, -1, (IPC_NOWAIT | SEM_UNDO) };

int semid = -1;
int M_semid = -1;
void init_lock() {
	if (semid < 0) {
		if ( (semid = semget(IPC_PRIVATE, 1, (IPC_CREAT | 06660))) < 0 ) {
			perror("shit! semget returned error\n");
			exit(-1);
		}
	}
}

void init_M_lock() {
	if (M_semid < 0) {
		if ( (M_semid = semget(IPC_PRIVATE, 1, (IPC_CREAT | 06660))) < 0 ) {
			perror("shit! semget returned error\n");
			exit(-1);
		}
	}
}

int lock() {
	if ( semop(semid, mlock, 2) < 0 ) {
		perror("fuck! semop returned error: lock\n");
		return -1;
	}
	return 0;
}

int lock_M() {
	if ( semop(M_semid, M_mlock, 2) < 0 ) {
		perror("fuck! semop returned error: lock\n");
		return -1;
	}
	return 0;
}

int unlock() {
	if ( semop(semid, munlock, 1) < 0 ) {
		perror("fuck! semop returned error: unlock\n");
		return -1;
	}
	return 0;
}

int unlock_M() {
	if ( semop(M_semid, M_munlock, 1) < 0 ) {
		perror("fuck! semop returned error: unlock\n");
		return -1;
	}
	return 0;
}

// XXX: running in parent
void create_shm_queue() {
	init_lock();

    /*
    *
    * STEP 1: create the shared segment
    */
    if ( (Cid = shmget(IPC_PRIVATE, sizeof(int), IPC_CREAT | 0666)) < 0 )
    {
        perror("fuck! smget returned -1\n");
        error(-1, errno, " ");
        exit(-1);
    }

	// initialize in parent
    if ( (shm = (int*) shmat(Cid, NULL, 0)) == (int*) -1 ){
        perror("shit! process shmat returned NULL\n");
        error(-1, errno, " ");
		exit(-1);
    }

	// parent initializes the memory
	shm[0] = 0;

	// detach
    if (shmdt(shm) == -1){
        perror("fuck! shmdt returned -1\n");
        error(-1, errno, " ");
		exit(-1);
    }
}

void create_shm_M(int V_size, int A_size) {
	init_M_lock();

    /*
    *
    * STEP 1: create the shared segment
    */
    if ( (Mid = shmget(IPC_PRIVATE, sizeof(int) * V_size * A_size, IPC_CREAT | 0666)) < 0 )
    {
        perror("fuck! smget returned -1\n");
        error(-1, errno, " ");
        exit(-1);
    }

	// initialize in parent
    if ( (M_shm = (int*) shmat(Mid, NULL, 0)) == (int*) -1 ){
        perror("shit! process shmat returned NULL\n");
        error(-1, errno, " ");
		exit(-1);
    }

	// parent initializes the memory
	for (int i=0; i<V_size * A_size; i++) M_shm[i] = 0;

	// detach
    if (shmdt(M_shm) == -1){
        perror("fuck! shmdt returned -1\n");
        error(-1, errno, " ");
		exit(-1);
    }
}

void delete_shm_queue() {
    if (shmctl(Cid,IPC_RMID,NULL) == -1){
        perror("shmctl returned -1\n");
        error(-1, errno, " ");
    }
}

void delete_shm_M() {
    if (shmctl(Mid,IPC_RMID,NULL) == -1){
        perror("shmctl returned -1\n");
        error(-1, errno, " ");
    }
}

// XXX: running in child
void attach_shm_queue() {
    if ( (shm = (int*) shmat(Cid, NULL, 0)) == (int*) -1 ){
        perror("shit! process shmat returned NULL\n");
        error(-1, errno, " ");
		exit(-1);
    }
}

void detach_shm_queue() {
    if (shmdt(shm) == -1){
        perror("fuck! shmdt returned -1\n");
        error(-1, errno, " ");
		exit(-1);
    }
}

void attach_shm_M() {
    if ( (M_shm = (int*) shmat(Mid, NULL, 0)) == (int*) -1 ){
        perror("shit! process shmat returned NULL\n");
        error(-1, errno, " ");
		exit(-1);
    }
}

void detach_shm_M() {
    if (shmdt(M_shm) == -1){
        perror("fuck! shmdt returned -1\n");
        error(-1, errno, " ");
		exit(-1);
    }
}

// return -1 if the queue top number is already maxnum
int queue_get(int maxnum) {
	int ret;

	lock();
	if (shm[0] == maxnum) {
		ret = -1;
	}
	else {
		ret = shm[0]++;
	}
	unlock();

	return ret;
}

long getCurrentTime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}

// set M[u][v] for all (u, v) in A x V
// V, V_size: graph structure; starting indices in E of all vertices
// E, E_size: graph structure; neighboring vertices
// A, A_size: active vertices; a vertex is set true if it is active
static void build (int *V, int V_size, int *E, int E_size, int *A, int A_size) {
	// create shared memory space for reducing the results from multi gpus
	create_shm_M(V_size, A_size);

	// get node number one by one in each process
	create_shm_queue();

	// here we fork child processes
    base_pid = getpid();
	my_id = -1;
    for(int i=0; i<nproc; i++) {
        pid = fork();

        if (getpid()==base_pid) { // parent
			continue;
		}
        else { // child
			my_id = i;
			//printf("[child] my id is %d\n", my_id);
            break;
		}
    }
	
	if (getpid() == base_pid) {
		// parent => ???
	}
	else{
	// ---------------------------- for child processes   -----------------------

	// attach shm queue
	attach_shm_queue();

	// set device if child
	cudaSetDevice(my_id % NGPUCARDS);

	// graph resides in device memry
	int* V_dev;
	cudaMalloc((void**) &V_dev, sizeof(int) * (V_size + 1));
	cudaMemcpy(V_dev, V, sizeof(int) * (V_size + 1), cudaMemcpyHostToDevice);

	int* E_dev;
	cudaMalloc((void**) &E_dev, sizeof(int) * E_size);
	cudaMemcpy(E_dev, E, sizeof(int) * E_size, cudaMemcpyHostToDevice);

	int *__M_dev;
	cudaMalloc((void**) &__M_dev, sizeof(int) * V_size * A_size);
	cudaMemset(__M_dev, 0x0, sizeof(int) * V_size * A_size);

	bool *__L_dev;
	cudaMalloc((void**) &__L_dev, sizeof(bool) * V_size * A_size);
	int  *__d_dev;
	cudaMalloc((void**) &__d_dev, sizeof(int) * V_size * A_size);
	
    //bool* _Ls = new bool[V_size];
	bool* Ls_dev;
	cudaMalloc((void**) &Ls_dev, sizeof(bool) * V_size);
    //int* _ds = new int[V_size];
	int* ds_dev;
	cudaMalloc((void**) &ds_dev, sizeof(int) *  V_size);
    //int* _F = new int[V_size];
	bool* F_dev;
	cudaMalloc((void**) &F_dev,  sizeof(bool) *  V_size);
	//cudaMemset(F_dev, 0, sizeof(bool) * V_size);

	int *alpha, *beta, *gamma;
	cudaMalloc((void**) &alpha, sizeof(int) * (V_size + 1));
	cudaMalloc((void**) &beta,  sizeof(int) * (V_size + 1)); // exclusive prefix sum
	cudaMalloc((void**) &gamma, sizeof(int) * (V_size + 1));

	bool *done_dev;
	cudaMalloc((void**) &done_dev, sizeof(bool));

    // A_map: active vertices (vertex id -> A idx)
    int* A_map = new int[V_size];
    for (int i=0; i<V_size; i++) A_map[i] = -1;

	init_kernel(V_size);

	// we keep BFS result with vertices in A only, not all in V
	// XXX: for now, we assume that the BFS result including distances
	//      to all the other vertices with every active node can be
	//      fit into the memory in a gpu device (11G)
	// XXX: we access L_dev and L_dev (A x V matrics) as if a single dimensional
	//      array (i.e., to access an element La_dev[i][j], refer to
	//      L_dev[i * V_size + j])
	long start, end;

	for (int i=0; i<A_size; i++) {
		// note that the first order index of d and L is that of A
		cudaBFS(A[i], V_dev, V_size, E_dev, &(d_dev(i, 0)), &(L_dev(i, 0)), F_dev, done_dev);
		//run_bfs(A[i], V_dev, V_size, E_dev, &(d_dev(i, 0)), &(L_dev(i, 0)), F_dev);
		// later, we need to access the index of A with a vertex id
		A_map[A[i]] = i;

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
	}

	int *ds_local = new int [V_size];

	// XXX: get s node id from shm_queue
	//for (int s=0; s<V_size; s++) {
	for ( ; ; ) {
		int s = queue_get(V_size);
		//printf("child = %d got s = %d\n", my_id, s);
		if (s < 0) break;

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

		//start = get_nanos();
		cudaMemcpy((void*)ds_local, (void*)ds, sizeof(int) * V_size, cudaMemcpyDeviceToHost);
		//end = get_nanos();
		//fprintf(stderr, "Algo Line 6 (cudaMemcpy for ds): %ld us (%ld ms)\n", 
		//		(end-start)/1000, (end-start)/1000/1000);

#if 0
		printf("ds: s = %d\n", s);
		for (int i=0; i<V_size; i++) {
			std::cout << " " << ds_local[i];
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
            int*  dv = &(d_dev(v_idx, 0));
            bool* Lv = &(L_dev(v_idx, 0));

			int ds_v = ds_local[v];
			//fprintf(stderr, "ds[s: %d, v: %d] = %d\n", s, v, ds_v);

			/*
			start = get_nanos();
			cudaMemcpy(&ds_v, (void*)&(ds[v]), sizeof(int), cudaMemcpyDeviceToHost);
			end = get_nanos();
			fprintf(stderr, "Algo Line 6 (cudaMemcpy for ds[v] = %d): %ld us (%ld ms)\n", 
					ds_v,
					(end-start)/1000, (end-start)/1000/1000);
			*/

			//start = get_nanos();
			// with reachable vertex t, do count sorting
			compute_alpha(Lv, ds, dv, ds_v, V_size, alpha);
			//compute_alpha(Lv, ds, dv, v, V_size, alpha);
			//end = get_nanos();
			//fprintf(stderr, "Algo Line 6 (alpha computation): %ld us (%ld ms)\n", 
			//		(end-start)/1000, (end-start)/1000/1000);


#if 0
			int* _alpha = new int [V_size];
			cudaMemcpy(_alpha, alpha, sizeof(int) * V_size, cudaMemcpyDeviceToHost);
			printf("s = %d, v = %d\n", s, v);
			std::cout << "alpha (ds[v] = " << ds_v << "):";
			for (int i=0; i<V_size; i++) {
				if (_alpha[i] != 0) std::cout << " " <<  i << ":" << _alpha[i];
			}
			std::cout << std::endl;
			delete[] _alpha;
			getchar();
#endif

			// prefix sum
			// in: alpha -> out: beta
			// in: alpha * d -> out: gamma
			// XXX: compute beta and gamma with separate calls of kernel
			//      function

			//start = get_nanos();
			scan (beta, alpha, ds_v + 1, true);
			//scan (beta, alpha, V_size, true);
			//end = get_nanos();
			//fprintf(stderr, "Algo Line 8,9 (beta computaion): %ld us (%ld ms)\n", 
			//		(end-start)/1000, (end-start)/1000/1000);

			//start = get_nanos();
			compute_1moment_alpha (alpha, ds_v);
			//compute_1moment_alpha (alpha, V_size, ds, v);
			//end = get_nanos();
			//fprintf(stderr, "Algo Line 8,9 (1 moment of alpha computaion): %ld us (%ld ms)\n", 
			//		(end-start)/1000, (end-start)/1000/1000);

			//start = get_nanos();
			scan (gamma, alpha, ds_v + 1, true);
			//scan (gamma, alpha, V_size, true);
			//end = get_nanos();
			//fprintf(stderr, "Algo Line 8,9 (gamma computaion): %ld us (%ld ms)\n", 
			//		(end-start)/1000, (end-start)/1000/1000);
			//scan (gamma, alpha, ds_v + 1, true);


#if 0
	int *__gamma = new int [V_size];
	int *__beta = new int [V_size];

	cudaMemcpy(__gamma, gamma, sizeof(int) * V_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(__beta, beta, sizeof(int) * V_size, cudaMemcpyDeviceToHost);

	std::cout << "ds_v = " << ds_v << std::endl;
	std::cout << "beta: ";
	for (int j=0; j<V_size; j++) {
		std::cout << "\t" << __beta[j];
	}
	std::cout << std::endl;
	std::cout << "gamma: ";
	for (int j=0; j<V_size; j++) {
		std::cout << "\t" << __gamma[j];
	}
	std::cout << std::endl;

	delete[] __gamma;
	delete[] __beta;
	getchar();
#endif


			/*
            beta[0] = alpha[0];
            gamma[0] = 0; // 0 * alpha[0]
            for (int i=1; i<ds[v]; i++) {
                beta[i] = alpha[i] + beta[i-1];
                gamma[i] = i * alpha[i] + gamma[i-1];
            }
			*/
			
            // compute M_dev in the device
			//start = get_nanos();
			increase_M (&(beta[1]), &(gamma[1]), V_size, Ls, ds, ds_v, &(M_dev(v_idx, 0)));
			//increase_M (&(beta[1]), &(gamma[1]), V_size, Ls, ds, ds_v, &(M_dev(v_idx, 0)));
			//end = get_nanos();
			//fprintf(stderr, "Algo Line 10-12 (M_v computaion): %ld us (%ld ms)\n\n", 
			//		(end-start)/1000, (end-start)/1000/1000);

#if 0
	// host 
	int *__M_host = new int [A_size * V_size];

	cudaMemcpy(__M_host, __M_dev, sizeof(int) * A_size * V_size, cudaMemcpyDeviceToHost);
	for (int i=0; i<A_size; i++) {
		std::cout << "A[" << i << "]=" << A[i] << ":";
		for (int j=0; j<V_size; j++) {
			std::cout << "\t" << M_host(i, j);
		}
		std::cout << std::endl;
	}

	delete[] __M_host;
	getchar();
#endif
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

	detach_shm_queue();

	// host : need to fetch result from cuda
	int *__M_host = new int [A_size * V_size];

	cudaMemcpy(__M_host, __M_dev, sizeof(int) * A_size * V_size, cudaMemcpyDeviceToHost);

	// XXX: obtain shared memory for M & add
	attach_shm_M();

	// lock
	lock_M();

	// add
	for (int i=0; i<V_size * A_size; i++) M_shm[i] += __M_host[i];

	// unlock
	unlock_M();
	detach_shm_M();

	delete[] __M_host;

	fin_kernel ();

	delete[] A_map;
	delete[] ds_local;

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

	}
	// -------------------------- end for child -------------------------
	
	// wait until all children exit
    if (getpid()==base_pid)
        for(int i=0; i<nproc; i++) {
            wait(NULL);
        }
    else
        exit(0);

	// XXX: print reduced M in shared memory
#if DEBUG == 1
	attach_shm_M();

	for (int i=0; i<A_size; i++) {
		std::cout << A[i] << ":";
		for (int j=0; j<V_size; j++) {
			std::cout << "\t" << M_shm[i * V_size +  j];
		}
		std::cout << std::endl;
	}

	detach_shm_M();
#endif

	delete_shm_queue();
	delete_shm_M();

}


int main (int argc, char *argv[])
{
	// simply, for active nodes, we increase id from 0 to active set size
    if (argc != 8 && argc != 6) {
        std::cerr << "usage: <V: start index file> <size of V> <E: edge file> <size of E> <nproc> [<A: active file> <size of A>]" << std::endl;
        exit (1);
    }


    const char *V_file = argv[1];
    const int V_size = atoi(argv[2]);
    const char *E_file = argv[3];
    const int E_size = atoi(argv[4]);

	nproc = atoi(argv[5]);

    char *A_file;
    int A_size;
	if (argc == 8) {
		A_file = argv[6];
		A_size = atoi(argv[7]);
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

#if ACCTIME == 1
	long total_start = 0;
	total_start = getCurrentTime();
#endif
    build(V, V_size, E, E_size, A, A_size);
#if ACCTIME == 1
	printf("%ld\n", (getCurrentTime() - total_start));
#endif

	delete[] V;
	delete[] E;
	delete[] A;
}
