#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"

#include <time.h>

#include "deltafast.h"
#include "utils.h"

void test_sequential_scan (int *in, const int N) {
	// sequential scan on CPU
	float elapsedTime = 0;
	long start_time, end_time;

	int *out = new int[N];

	//cudaEvent_t start, stop;

	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);

	//cudaEventRecord(start);
	start_time = get_nanos();
	sequential_scan(out, in, N);
	end_time = get_nanos();
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&elapsedTime, start, stop);
	//printResult("host    ", out[N - 1], elapsedTime);
	printResult("host    ", out[N - 1], end_time - start_time);

	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);

	delete[] out;
}

void test_scan_wo_bcao (int *d_in, const int N) {
	const int arraySize = N * sizeof(int);
	// full scan
	float elapsedTime = 0;
	long start_time, end_time;

	int *out = new int[N];
	int *d_out;
	cudaMalloc((void **)&d_out, arraySize);

	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);

	//cudaEventRecord(start);
	cudaMemset(d_out, 0, arraySize);
	start_time = get_nanos();
	scan(d_out, d_in, N, false);
	end_time = get_nanos();
	cudaMemcpy(out, d_out, arraySize, cudaMemcpyDeviceToHost);
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&elapsedTime, start, stop);
	//printResult("gpu     ", out[N - 1], elapsedTime);
	printResult("gpu     ", out[N - 1], end_time - start_time);

	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);

	cudaFree(d_out);
	delete[] out;
}

void test_scan_w_bcao (int *d_in, const int N) {
	const int arraySize = N * sizeof(int);
	// full scan with BCAO
	float elapsedTime = 0;
	long start_time, end_time;

	int *out = new int[N];
	int *d_out;
	cudaMalloc((void **)&d_out, arraySize);

	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);

	//cudaEventRecord(start);
	cudaMemset(d_out, 0, arraySize);
	start_time = get_nanos();
	scan(d_out, d_in, N, true);
	end_time = get_nanos();
	cudaMemcpy(out, d_out, arraySize, cudaMemcpyDeviceToHost);
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&elapsedTime, start, stop);
	//printResult("gpu bcao", out[N - 1], elapsedTime);
	printResult("gpu bcao", out[N - 1], end_time - start_time);

	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);

	cudaFree(d_out);
	delete[] out;
}


void test(int N) {
	time_t t;
	srand((unsigned)time(&t));
	int *in = new int[N];
	for (int i = 0; i < N; i++) {
		in[i] = rand() % 10;
	}
	printf("%i Elements \n", N);

	int *d_in;
	const int arraySize = N * sizeof(int);
	cudaMalloc((void **)&d_in, arraySize);
	cudaMemcpy(d_in, in, arraySize, cudaMemcpyHostToDevice);

	init_kernel (N);
	test_sequential_scan (in, N);
	test_scan_wo_bcao (d_in, N);
	test_scan_w_bcao (d_in, N);
	fin_kernel();

	/* 
	 * NOTE:
	 *    block scan (with small number of elements) is always worse than
	 *    sequantial scan 
	 */

#if 0
	if (canBeBlockscanned) {
		// basic level 1 block scan
		int *out_1block = new int[N]();
		float time_1block = blockscan(out_1block, in, N, false);
		printResult("level 1 ", out_1block[N - 1], time_1block);

		// level 1 block scan with BCAO
		int *out_1block_bcao = new int[N]();
		float time_1block_bcao = blockscan(out_1block_bcao, in, N, true);
		printResult("l1 bcao ", out_1block_bcao[N - 1], time_1block_bcao);

		delete[] out_1block;
		delete[] out_1block_bcao;
	}

	printf("\n");
#endif

	cudaFree(d_in);
	delete[] in;
}

int main()
{
	int TEN_MILLION = 10001941;
	int ONE_MILLION = 1001031;
	int TEN_THOUSAND = 10117;

	int elements[] = {
		TEN_MILLION * 4,
		TEN_MILLION * 2,
		TEN_MILLION,
		ONE_MILLION,
		TEN_THOUSAND,
		5001,
		4096,
		2048,
		2013,
		1111,
		501,
		103,
		64,
		8,
		5
	};

	int numElements = sizeof(elements) / sizeof(elements[0]);

	for (int i = 0; i < numElements; i++) {
		test(elements[i]);
	}

	return 0;
}
