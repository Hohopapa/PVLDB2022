#ifndef BFS_H
#define BFS_H
void CUDA_BFS(int *Va, const int Va_size, int *Ea, bool *Fa, bool *Xa, int *Ca, bool *done);
void CUDA_SET_MAXDIST (bool* _L_dev, int *_d_dev, const int V_size);
void cudaBFS(
		const int source, int* _V_dev, const int V_size, int* _E_dev,  /* input */
		int *_d_dev, bool *_L_dev,  /* output */
		bool *_F_dev,  /* frontier */
		bool *_done_dev); /* no frontier any more? */
void run_bfs (
		const int source, int* _V_dev, const int V_size, int* _E_dev,  /* input */
		int *_d_dev, bool *_L_dev,  /* output */
		bool *_F_dev);
#endif
