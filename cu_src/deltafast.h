#ifndef DELTA_H
#define DELTA_H
void init_kernel (int V_size);
void fin_kernel ();

void compute_alpha (
		bool *Lv, int *ds, int *dv, int ds_v, int V_size,
		int *alpha);
void compute_1moment_alpha (int *alpha, int max_exclusive);
void increase_M (
		int *beta, int *gamma, int V_size, bool *Ls, int *ds, int ds_v,
		int *M_v_idx);
void scan(int *output, int *input, int length, bool bcao);
void sequential_scan(int* output, int* input, int length);
#endif
