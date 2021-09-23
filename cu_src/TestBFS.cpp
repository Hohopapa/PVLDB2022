#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
//#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>

#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bfs.h"
#include "graph.h"
#include "utils.h"

long getCurrentTime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}


// The BFS frontier corresponds to all the nodes being processed at the current level.
int main(int argc, char *argv[]) {
    if (argc != 6) {
        std::cerr << "usage: <V file> <V size> <E file> <E size> <start vertex idx>" << std::endl;
        exit(1);
    }

    int NUM_NODES = atoi(argv[2]);
    int NUM_EDGES = atoi(argv[4]);
    int source = atoi(argv[5]);
	int nodes[NUM_NODES + 1]; // use the last entry for the (exclusive) ending index of array
	int edges[NUM_EDGES];

    printf("vertex (start index): %d\n", source);

    read_graph (argv[1], NUM_NODES, argv[3], NUM_EDGES, nodes, edges);

	int* Va;
	cudaMalloc((void**) &Va, sizeof(int) * (NUM_NODES + 1));
	cudaMemcpy(Va, nodes, sizeof(int) * (NUM_NODES + 1), cudaMemcpyHostToDevice);

	int* Ea;
	cudaMalloc((void**) &Ea, sizeof(int) * NUM_EDGES);
	cudaMemcpy(Ea, edges, sizeof(int) * NUM_EDGES, cudaMemcpyHostToDevice);

    // frontier
	bool* Fa;
	cudaMalloc((void**) &Fa, sizeof(bool) * NUM_NODES);

    // visited
	bool* Xa;
	cudaMalloc((void**) &Xa, sizeof(bool) * NUM_NODES);

    // cost
	int* Ca;
	cudaMalloc((void**) &Ca, sizeof(int) * NUM_NODES);

	// done flag
	bool* d_done;
	cudaMalloc((void**) &d_done, sizeof(bool));

	/*
	printf("\nCost: ");
	for (int i = 0; i < NUM_NODES; i++)
		printf("%d    ", cost[i]);
	printf("\n");
	*/

	long start, end;
	long s;

	int* cost_seq  = new int [NUM_NODES];
	bool* L_seq = new bool [NUM_NODES];
	int* F_seq = new int [NUM_NODES];

	start = get_nanos();
	s = getCurrentTime();
	sequentialBFS(source, nodes, NUM_NODES, edges, cost_seq, L_seq, F_seq);
	end = get_nanos();
	fprintf(stderr, "BFS w/o CUDA: %ld us (%ld ms)\n", (end-start)/1000, (end-start)/1000/1000);
	fprintf(stderr, "BFS w/o CUDA: %ld ms\n", (getCurrentTime()-s));

	/*
	printf("\nCost: ");
	for (int i = 0; i < NUM_NODES; i++)
		printf("%d    ", cost_seq[i]);
	printf("\n");
	*/

	//cudaEvent_t start_e, stop_e;
	//cudaEventCreate(&start_e);
	//cudaEventCreate(&stop_e);
	//cudaEventRecord(start_e);

	// run BFS
	start = get_nanos();
	s = getCurrentTime();
	//run_bfs (source, Va, NUM_NODES, Ea, Ca, Xa, Fa);
	cudaBFS (source, Va, NUM_NODES, Ea, Ca, Xa, Fa, d_done);
	cudaDeviceSynchronize();

	//cudaEventRecord(stop_e);
	//cudaEventSynchronize(stop_e);
	//float elapsedTime = 0;
	//cudaEventElapsedTime(&elapsedTime, start_e, stop_e); // milli secs
	end = get_nanos();
	fprintf(stderr, "BFS w/ CUDA: %ld us (%ld ms)\n", 
			(end-start)/1000, (end-start)/1000/1000);
	fprintf(stderr, "BFS w/ CUDA: %ld ms\n", (getCurrentTime()-s));

	//cudaEventDestroy (start_e);
	//cudaEventDestroy (stop_e);

	int* cost = new int [NUM_NODES];
	cudaMemcpy(cost, Ca, sizeof(int) * NUM_NODES, cudaMemcpyDeviceToHost);

	for (int i=0; i<NUM_NODES; i++) {
		if (cost[i] != cost_seq[i]) {
			fprintf(stderr, "error: distance of index = %d mismatches!!\n", i);
			exit(1);
		}
	}
	fprintf(stderr, "\n");

	delete[] cost;
	delete[] cost_seq;
	delete[] L_seq;
	delete[] F_seq;
	cudaFree(Va);
	cudaFree(Ea);
	cudaFree(Ca);
	cudaFree(Xa);
	cudaFree(Va);
	cudaFree(d_done);
}
