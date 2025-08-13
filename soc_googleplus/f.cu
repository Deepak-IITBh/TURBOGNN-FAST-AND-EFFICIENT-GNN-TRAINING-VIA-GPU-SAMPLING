#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define MAX_LINE_LENGTH 1024
#define INF 1000000000

// Kernel to calculate degree centrality
__global__ void degreeCentralityKernel(int *row_ptr, int num_nodes, float *degree_centrality) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_nodes) {
        degree_centrality[idx] = (float)(row_ptr[idx+1] - row_ptr[idx]);
    }
}

// Kernel for single-source BFS (will be called for each node)
__global__ void bfsKernel(int *row_ptr, int *col_idx, int num_nodes, int source_node, int *distance) {
    // Initialize distances
    if (threadIdx.x == 0) {
        for (int i = 0; i < num_nodes; i++) {
            distance[i] = INF;
        }
        distance[source_node] = 0;
    }
    __syncthreads();

    // Simple parallel BFS implementation
    bool changed = true;
    while (changed) {
        changed = false;
        __syncthreads();
        
        for (int u = threadIdx.x; u < num_nodes; u += blockDim.x) {
            if (distance[u] < INF) {
                for (int i = row_ptr[u]; i < row_ptr[u+1]; i++) {
                    int v = col_idx[i];
                    if (distance[v] > distance[u] + 1) {
                        atomicMin(&distance[v], distance[u] + 1);
                        changed = true;
                    }
                }
            }
        }
        __syncthreads();
    }
}

void readCSRFile(const char *filename, int **row_ptr, int **col_idx, int *num_nodes, int *num_edges) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }

    // Read number of nodes and edges from first line
    char line[MAX_LINE_LENGTH];
    fgets(line, MAX_LINE_LENGTH, file);
    sscanf(line, "%d %d", num_nodes, num_edges);

    // Allocate memory for row pointers and column indices
    *row_ptr = (int *)malloc((*num_nodes + 1) * sizeof(int));
    *col_idx = (int *)malloc(*num_edges * sizeof(int));

    // Read row pointers
    for (int i = 0; i <= *num_nodes; i++) {
        fgets(line, MAX_LINE_LENGTH, file);
        (*row_ptr)[i] = atoi(line);
    }

    // Read column indices
    for (int i = 0; i < *num_edges; i++) {
        fgets(line, MAX_LINE_LENGTH, file);
        (*col_idx)[i] = atoi(line);
    }

    fclose(file);
}

void writeResultsToFile(const char *filename, float *values, int num_nodes) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return;
    }
    
    for (int i = 0; i < num_nodes; i++) {
        fprintf(file, "%d %.6f\n", i, values[i]);
    }
    
    fclose(file);
}

int main() {
    const char *csr_filename = "soc_google-plus_output.csr";
    int *h_row_ptr, *h_col_idx;
    int num_nodes, num_edges;
    
    // Read CSR file
    printf("Reading CSR file %s...\n", csr_filename);
    readCSRFile(csr_filename, &h_row_ptr, &h_col_idx, &num_nodes, &num_edges);
    printf("Graph loaded with %d nodes and %d edges\n", num_nodes, num_edges);
    
    // Allocate host memory for results
    float *h_degree_centrality = (float *)malloc(num_nodes * sizeof(float));
    float *h_closeness_centrality = (float *)malloc(num_nodes * sizeof(float));
    
    // Allocate device memory
    int *d_row_ptr, *d_col_idx;
    float *d_degree_centrality;
    cudaMalloc((void **)&d_row_ptr, (num_nodes + 1) * sizeof(int));
    cudaMalloc((void **)&d_col_idx, num_edges * sizeof(int));
    cudaMalloc((void **)&d_degree_centrality, num_nodes * sizeof(float));
    
    // Copy data to device
    printf("Copying data to GPU...\n");
    cudaMemcpy(d_row_ptr, h_row_ptr, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    
    // Calculate grid size
    int blocks = (num_nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Calculate degree centrality
    printf("Calculating degree centrality...\n");
    degreeCentralityKernel<<<blocks, THREADS_PER_BLOCK>>>(d_row_ptr, num_nodes, d_degree_centrality);
    cudaDeviceSynchronize();
    
    // Copy degree results back to host
    cudaMemcpy(h_degree_centrality, d_degree_centrality, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate closeness centrality (one node at a time due to memory constraints)
    printf("Calculating closeness centrality (this will take a while)...\n");
    
    // Allocate device memory for BFS
    int *d_distance;
    cudaMalloc((void **)&d_distance, num_nodes * sizeof(int));
    
    for (int src = 0; src < num_nodes; src++) {
        if (src % 1000 == 0) {
            printf("Processing node %d of %d\n", src, num_nodes);
        }
        
        // Run BFS for this source node
        bfsKernel<<<1, THREADS_PER_BLOCK>>>(d_row_ptr, d_col_idx, num_nodes, src, d_distance);
        cudaDeviceSynchronize();
        
        // Copy distances back to host
        int *h_distance = (int *)malloc(num_nodes * sizeof(int));
        cudaMemcpy(h_distance, d_distance, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Calculate closeness for this node
        float total_distance = 0.0f;
        int reachable_nodes = 0;
        
        for (int i = 0; i < num_nodes; i++) {
            if (i != src && h_distance[i] < INF) {
                total_distance += h_distance[i];
                reachable_nodes++;
            }
        }
        
        if (reachable_nodes > 0 && total_distance > 0) {
            h_closeness_centrality[src] = (float)reachable_nodes / total_distance;
        } else {
            h_closeness_centrality[src] = 0.0f;
        }
        
        free(h_distance);
    }
    
    // Free device memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_degree_centrality);
    cudaFree(d_distance);
    
    // Verify some results
    printf("Sample results:\n");
    for (int i = 0; i < 10; i++) {
        printf("Node %d: Degree=%.2f, Closeness=%.6f\n", 
               i, h_degree_centrality[i], h_closeness_centrality[i]);
    }
    
    // Write results to files
    printf("Writing results to files...\n");
    writeResultsToFile("degreea2d.txt", h_degree_centrality, num_nodes);
    writeResultsToFile("closenessa2d.txt", h_closeness_centrality, num_nodes);
    
    // Free host memory
    free(h_row_ptr);
    free(h_col_idx);
    free(h_degree_centrality);
    free(h_closeness_centrality);
    
    printf("Centrality calculations completed. Results saved to degree.txt and closeness.txt\n");
    
    return 0;
}
