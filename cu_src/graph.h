#ifndef GRAPH_H
#define GRAPH_H
void read_graph (const char *filename1, int V_size, const char *filename2, int E_size, int V[], int E[]);
void read_active_vertex (const char *filename, const int V_size, int A[], int A_size);
void sequentialBFS(int source, int* V, int V_size, int* E, int *d, bool *L, int *F);
#endif
