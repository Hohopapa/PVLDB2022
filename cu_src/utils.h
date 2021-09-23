#ifndef UTILS_H
#define UTILS_H
#include "cuda_runtime.h"

void _checkCudaError(const char *message, cudaError_t err, const char *caller);
void printResult(const char* prefix, int result, long nanoseconds);
void printResult(const char* prefix, int result, float milliseconds);

bool isPowerOfTwo(int x);
int nextPowerOfTwo(int x);

long get_nanos();

void init_progress ();
void print_progress (int index, int datasize);
void print_progress (const char *prefx, int index, int datasize);
#endif
