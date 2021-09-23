#include <stdio.h>
#include "process.h"
#include <sys/time.h>
#include <time.h>

#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>

#include "utils.h"
#include <iostream>

void _checkCudaError(const char *message, cudaError_t err, const char *caller) {
	if (err != cudaSuccess) {
		fprintf(stderr, "Error in: %s\n", caller);
		fprintf(stderr, "%s", message);
		fprintf(stderr, ": %s\n", cudaGetErrorString(err));
		exit(0);
	}
}

void printResult(const char* prefix, int result, long nanoseconds) {
	printf("  ");
	printf("%s", prefix);
	printf(" : %d in %ld ns \n", result, nanoseconds);
}

void printResult(const char* prefix, int result, float milliseconds) {
	printf("  ");
	printf("%s", prefix);
	printf(" : %d in %f ms \n", result, milliseconds);
}


// from https://stackoverflow.com/a/3638454
bool isPowerOfTwo(int x) {
	return x && !(x & (x - 1));
}

// from https://stackoverflow.com/a/12506181
int nextPowerOfTwo(int x) {
	int power = 1;
	while (power < x) {
		power *= 2;
	}
	return power;
}


// from https://stackoverflow.com/a/36095407
long get_nanos() {
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

static int prev_progress = -1;
static struct timeval ft;

void init_progress () {
    prev_progress = -1;
}


void print_progress (int index, int datasize) {
	print_progress("", index, datasize);
}

void print_progress (const char *prefix, int index, int datasize) {
    int progress = (int) ((double)index / (double) datasize * 100.0);
	if (prev_progress < 0) {
    	gettimeofday (&ft, NULL);
        std::cout << "\t... " << prefix << progress << " 0%" << std::endl;
	}
	else if (prev_progress >= 0 && progress != prev_progress) {
		struct timeval st = ft;
		gettimeofday (&ft, NULL);

		int gap = (ft.tv_sec - st.tv_sec) * 1000  + (ft.tv_usec - st.tv_usec) / 1000;
        std::cout << "\t... " << prefix << progress << " %" << " (" << gap << " ms)" << std::endl;
    }
    prev_progress = progress;
}

