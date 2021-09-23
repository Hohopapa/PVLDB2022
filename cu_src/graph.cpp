#include "graph.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// filename1: file containing vertices' offset for adjacent nodes (actually,
//            includes V_size + elements)
// filename2: file containing adjacent nodes
// V: output; size should be V_size + 1
// E: output
void read_graph (const char *filename1, const int V_size, const char *filename2, const int E_size, int V[], int E[]) {
	std::ifstream vertices_f(filename1);

    //printf("V (%s):\n", filename1);
	if (vertices_f.is_open()) {
        int idx = 0;
		std::string line;
		while (std::getline(vertices_f, line)) {
			std::istringstream iss(line);
            int a;
            if (!(iss >> a)) { break; } // error

            //std::cout << "\t" << idx << ":" << a;
            V[idx++] = a;
		}
        //std::cout << std::endl;
		vertices_f.close();

        // file sanity check (line number should be V_size + 1)
        assert (idx == V_size + 1);
        // content sanity check (the last element should be the same as E_size)
        assert (V[V_size] == E_size);
	}

	std::ifstream edges_f(filename2);

    //printf("E (%s):\n", filename2);
	if (edges_f.is_open()) {
		std::string line;
        int idx = 0;
		while (std::getline(edges_f, line)) {
			std::istringstream iss(line);
            int a;
            if (!(iss >> a)) { break; } // error

            //std::cout << "\t" << idx << ":" << a;
            E[idx++] = a;
		}
        //std::cout << std::endl;
		edges_f.close();

        assert (idx == E_size);
	}
}

// filename: file containing 
// act: output
void read_active_vertex (const char *filename, const int V_size, int A[], int A_size) {
	if (filename == NULL) {
		for (int i=0; i<V_size; i++) A[i] = i;
	}
	else {
		std::ifstream file(filename);
		int idx = 0;

		if (file.is_open()) {
			std::string line;

			while (std::getline(file, line)) {
				std::istringstream iss (line);

				int a;
				if (! (iss >> a)) { break; }

				A[idx++] = a;
				//std::cout << a << std::endl;
			}
		}
		file.close();

		//printf("Active vertices:\n");
		//for (int i=0; i<A_size; i++) std::cout << " " << A[i];
		//std::cout << std::endl;
		assert (idx == A_size);
	}
}

void sequentialBFS(int source, int* V, const int V_size, int* E, int *d, bool *L, int *F) {
    // init d (distance or cost) and L (visited)
    for (int i=0; i<V_size; i++) {
        d[i] = -1;
        L[i] = false;
    }

    d[source] = 0;

    // frontier
    int Q_first = -1;
    int Q_last = -1;

    F[++Q_last] = source;

    while (Q_last > Q_first) {
        // get a frontier & set its visited flag to true
        int u = F[++Q_first];
        L[u] = true;

        // for each neighboring nodes,
        for (int i=V[u]; i<V[u+1]; i++) {
            int v = E[i];

            // if neither it is visited nor its distance is computed yet,
            if (d[v] < 0) {
                F[++Q_last] = v;
                d[v] = d[u] + 1;
            }
        }
    }

    for (int i=0; i<V_size; i++) {
        if (d[i] < 0){
            d[i] = V_size - 1;
        }
    }
}
