#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "include/knncuda.h"


/**
 * Initializes randomly the reference and query points.
 *
 * @param ref        refence points
 * @param ref_nb     number of reference points
 * @param query      query points
 * @param query_nb   number of query points
 * @param dim        dimension of points
 */
void initialize_data(float * ref,
                     int     ref_nb,
                     float * query,
                     int     query_nb,
                     int     dim) {

    // Initialize random number generator
    srand(time(NULL));

    // Generate random reference points
    for (int i=0; i<ref_nb*dim; ++i) {
        ref[i] = 10. * (float)(rand() / (double)RAND_MAX);
    }

    // Generate random query points
    for (int i=0; i<query_nb*dim; ++i) {
        query[i] = 10. * (float)(rand() / (double)RAND_MAX);
    }
}

struct DatasetConfig {
    int ref_nb;
    int query_nb;
    int dim;
    const char* label;
};

DatasetConfig configs[] = {
    {1000, 200, 32,  "Small"},
    {10000, 1000, 64, "Medium"},
    {100000, 10000, 128, "Large"}
};

/**
 * Computes the Euclidean distance between a reference point and a query point.
 *
 * @param ref          refence points
 * @param ref_nb       number of reference points
 * @param query        query points
 * @param query_nb     number of query points
 * @param dim          dimension of points
 * @param ref_index    index to the reference point to consider
 * @param query_index  index to the query point to consider
 * @return computed distance
 */
float compute_distance(const float * ref,
                       int           ref_nb,
                       const float * query,
                       int           query_nb,
                       int           dim,
                       int           ref_index,
                       int           query_index) {
    float sum = 0.f;
    for (int d=0; d<dim; ++d) {
        const float diff = ref[d * ref_nb + ref_index] - query[d * query_nb + query_index];
        sum += diff * diff;
    }
    return sqrtf(sum);
}


/**
 * Gathers at the beginning of the `dist` array the k smallest values and their
 * respective index (in the initial array) in the `index` array. After this call,
 * only the k-smallest distances are available. All other distances might be lost.
 *
 * Since we only need to locate the k smallest distances, sorting the entire array
 * would not be very efficient if k is relatively small. Instead, we perform a
 * simple insertion sort by eventually inserting a given distance in the first
 * k values.
 *
 * @param dist    array containing the `length` distances
 * @param index   array containing the index of the k smallest distances
 * @param length  total number of distances
 * @param k       number of smallest distances to locate
 */
void  modified_insertion_sort(float *dist, int *index, int length, int k){

    // Initialise the first index
    index[0] = 0;

    // Go through all points
    for (int i=1; i<length; ++i) {

        // Store current distance and associated index
        float curr_dist  = dist[i];
        int   curr_index = i;

        // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
        if (i >= k && curr_dist >= dist[k-1]) {
            continue;
        }

        // Shift values (and indexes) higher that the current distance to the right
        int j = std::min(i, k-1);
        while (j > 0 && dist[j-1] > curr_dist) {
            dist[j]  = dist[j-1];
            index[j] = index[j-1];
            --j;
        }

        // Write the current distance and index at their position
        dist[j]  = curr_dist;
        index[j] = curr_index; 
    }
}


/*
 * For each input query point, locates the k-NN (indexes and distances) among the reference points.
 *
 * @param ref        refence points
 * @param ref_nb     number of reference points
 * @param query      query points
 * @param query_nb   number of query points
 * @param dim        dimension of points
 * @param k          number of neighbors to consider
 * @param knn_dist   output array containing the query_nb x k distances
 * @param knn_index  output array containing the query_nb x k indexes
 */
bool knn_c(const float * ref,
           int           ref_nb,
           const float * query,
           int           query_nb,
           int           dim,
           int           k,
           float *       knn_dist,
           int *         knn_index) {

    // Allocate local array to store all the distances / indexes for a given query point 
    float * dist  = (float *) malloc(ref_nb * sizeof(float));
    int *   index = (int *)   malloc(ref_nb * sizeof(int));

    // Allocation checks
    if (!dist || !index) {
        printf("Memory allocation error\n");
        free(dist);
        free(index);
        return false;
    }

    // Process one query point at the time
    for (int i=0; i<query_nb; ++i) {

        // Compute all distances / indexes
        for (int j=0; j<ref_nb; ++j) {
            dist[j]  = compute_distance(ref, ref_nb, query, query_nb, dim, j, i);
            index[j] = j;
        }

        // Sort distances / indexes
        modified_insertion_sort(dist, index, ref_nb, k);

        // Copy k smallest distances and their associated index
        for (int j=0; j<k; ++j) {
            knn_dist[j * query_nb + i]  = dist[j];
            knn_index[j * query_nb + i] = index[j];
        }
    }

    // Memory clean-up
    free(dist);
    free(index);

    return true;

}


/**
 * Test an input k-NN function implementation by verifying that its output
 * results (distances and corresponding indexes) are similar to the expected
 * results (ground truth).
 *
 * Since the k-NN computation might end-up in slightly different results
 * compared to the expected one depending on the considered implementation,
 * the verification consists in making sure that the accuracy is high enough.
 *
 * The tested function is ran several times in order to have a better estimate
 * of the processing time.
 *
 * @param ref            reference points
 * @param ref_nb         number of reference points
 * @param query          query points
 * @param query_nb       number of query points
 * @param dim            dimension of reference and query points
 * @param k              number of neighbors to consider
 * @param gt_knn_dist    ground truth distances
 * @param gt_knn_index   ground truth indexes
 * @param knn            function to test
 * @param name           name of the function to test (for display purpose)
 * @param nb_iterations  number of iterations
 * return false in case of problem, true otherwise
 */
bool test(const float * ref,
          int           ref_nb,
          const float * query,
          int           query_nb,
          int           dim,
          int           k,
          float *       gt_knn_dist,
          int *         gt_knn_index,
          bool (*knn)(const float *, int, const float *, int, int, int, float *, int *),
          const char *  name,
          int           nb_iterations) {

    // Parameters
    const float precision    = 0.001f; // distance error max
    const float min_accuracy = 0.999f; // percentage of correct values required

    // Display k-NN function name
    printf("- %-17s : ", name);

    // Allocate memory for computed k-NN neighbors
    float * test_knn_dist  = (float*) malloc(query_nb * k * sizeof(float));
    int   * test_knn_index = (int*)   malloc(query_nb * k * sizeof(int));

    // Allocation check
    if (!test_knn_dist || !test_knn_index) {
        printf("ALLOCATION ERROR\n");
        free(test_knn_dist);
        free(test_knn_index);
        return false;
    }

    // Start timer
    struct timeval tic;
    gettimeofday(&tic, NULL);

    // Compute k-NN several times
    for (int i=0; i<nb_iterations; ++i) {
        if (!knn(ref, ref_nb, query, query_nb, dim, k, test_knn_dist, test_knn_index)) {
            free(test_knn_dist);
            free(test_knn_index);
            return false;
        }
    }

    // Stop timer
    struct timeval toc;
    gettimeofday(&toc, NULL);

    // Elapsed time in ms
    double elapsed_time = toc.tv_sec - tic.tv_sec;
    elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;

    // Verify both precisions and indexes of the k-NN values
    int nb_correct_precisions = 0;
    int nb_correct_indexes    = 0;
    for (int i=0; i<query_nb*k; ++i) {
        if (fabs(test_knn_dist[i] - gt_knn_dist[i]) <= precision) {
            nb_correct_precisions++;
        }
        if (test_knn_index[i] == gt_knn_index[i]) {
            nb_correct_indexes++;
        }
    }

    // Compute accuracy
    float precision_accuracy = nb_correct_precisions / ((float) query_nb * k);
    float index_accuracy     = nb_correct_indexes    / ((float) query_nb * k);

    // Display report
    if (precision_accuracy >= min_accuracy && index_accuracy >= min_accuracy ) {
        printf("PASSED in %8.5f seconds (averaged over %3d iterations)\n", elapsed_time / nb_iterations, nb_iterations);
    }
    else {
        printf("FAILED\n");
    }

    // Free memory
    free(test_knn_dist);
    free(test_knn_index);

    return true;
}

// Wrapper function for knn_cuda_global with EUCLIDEAN metric
bool knn_cuda_global_euclidean(const float* ref, int ref_nb,
    const float* query, int query_nb,
    int dim, int k,
    float* knn_dist, int* knn_index) {
return knn_cuda_global(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, EUCLIDEAN);
}

bool knn_cuda_global_manhattan(const float* ref, int ref_nb,
    const float* query, int query_nb,
    int dim, int k,
    float* knn_dist, int* knn_index) {
    return knn_cuda_global(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, MANHATTAN);
}

/**
 * 1. Create the synthetic data (reference and query points).
 * 2. Compute the ground truth.
 * 3. Test the different implementation of the k-NN algorithm.
 */
int main(void) {

    // Parameters
    const int ref_nb   = 16384;
    const int query_nb = 4096;
    const int dim      = 128;
    const int k        = 16;

    // Parameters for visualization
    const int ref_nb_vis   = 100;
    const int query_nb_vis = 3;
    const int dim_vis      = 2;
    const int k_vis        = 5;

    FILE* out = fopen("results/benchmark.csv", "w");
    if (!out) {
        printf("Error: Unable to create results/benchmark.csv\n");
        return EXIT_FAILURE;
    }
    fprintf(out, "label,ref_nb,query_nb,dim,time_sec\n");

    // Display
    printf("PARAMETERS\n");
    printf("- Number reference points : %d\n",   ref_nb);
    printf("- Number query points     : %d\n",   query_nb);
    printf("- Dimension of points     : %d\n",   dim);
    printf("- Number of neighbors     : %d\n\n", k);

    // Sanity check
    if (ref_nb<k) {
        printf("Error: k value is larger that the number of reference points\n");
        return EXIT_FAILURE;
    }

    // Allocate input points and output k-NN distances / indexes
    float * ref        = (float*) malloc(ref_nb   * dim * sizeof(float));
    float * query      = (float*) malloc(query_nb * dim * sizeof(float));
    float * knn_dist   = (float*) malloc(query_nb * k   * sizeof(float));
    int   * knn_index  = (int*)   malloc(query_nb * k   * sizeof(int));

    // Allocation checks
    if (!ref || !query || !knn_dist || !knn_index) {
        printf("Error: Memory allocation error\n"); 
        free(ref);
	    free(query);
	    free(knn_dist);
	    free(knn_index);
        return EXIT_FAILURE;
    }

    // Initialize reference and query points with random values
    initialize_data(ref, ref_nb, query, query_nb, dim);

    // Compute the ground truth k-NN distances and indexes for each query point
    printf("Ground truth computation in progress...\n\n");
    if (!knn_c(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index)) {
        free(ref);
	    free(query);
	    free(knn_dist);
	    free(knn_index);
        return EXIT_FAILURE;
    }

    // Test all k-NN functions
    printf("TESTS\n");
    test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_c,            "knn_c",              2);
    test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index,
        &knn_cuda_global_euclidean, "knn_cuda_global", 100);
    test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_cuda_texture, "knn_cuda_texture", 100); 
    test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_cublas,       "knn_cublas",       100); 

    for (auto& config : configs) {
        int ref_nb   = config.ref_nb;
        int query_nb = config.query_nb;
        int dim      = config.dim;

        printf("\nBENCHMARKING: %s Dataset\n", config.label);
        printf("- Reference Points: %d\n", ref_nb);
        printf("- Query Points    : %d\n", query_nb);
        printf("- Dimensions      : %d\n", dim);
        printf("- k               : %d\n\n", k);
    
        // Allocate memory
        float *ref       = (float*) malloc(ref_nb * dim * sizeof(float));
        float *query     = (float*) malloc(query_nb * dim * sizeof(float));
        float *knn_dist  = (float*) malloc(query_nb * k * sizeof(float));
        int   *knn_index = (int*)   malloc(query_nb * k * sizeof(int));
    
        initialize_data(ref, ref_nb, query, query_nb, dim);
    
        // Benchmark CUDA Global - EUCLIDEAN
        struct timeval tic, toc;
        gettimeofday(&tic, NULL);
        knn_cuda_global(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, EUCLIDEAN);
        gettimeofday(&toc, NULL);
        double time_euclidean = (toc.tv_sec - tic.tv_sec) + (toc.tv_usec - tic.tv_usec) / 1e6;
    
        printf("[CUDA GLOBAL][EUCLIDEAN] Time: %.5f seconds\n", time_euclidean);
        fprintf(out, "%s_EUCLIDEAN,%d,%d,%d,%.5f\n", config.label, ref_nb, query_nb, dim, time_euclidean);
    
        // Benchmark CUDA Global - MANHATTAN
        gettimeofday(&tic, NULL);
        knn_cuda_global(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, MANHATTAN);
        gettimeofday(&toc, NULL);
        double time_manhattan = (toc.tv_sec - tic.tv_sec) + (toc.tv_usec - tic.tv_usec) / 1e6;
    
        printf("[CUDA GLOBAL][MANHATTAN] Time: %.5f seconds\n", time_manhattan);
        fprintf(out, "%s_MANHATTAN,%d,%d,%d,%.5f\n", config.label, ref_nb, query_nb, dim, time_manhattan);
    
        // Benchmark CUDA Global - COSINE
        gettimeofday(&tic, NULL);
        knn_cuda_global(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, COSINE);
        gettimeofday(&toc, NULL);
        double time_cosine = (toc.tv_sec - tic.tv_sec) + (toc.tv_usec - tic.tv_usec) / 1e6;
    
        printf("[CUDA GLOBAL][COSINE] Time: %.5f seconds\n", time_cosine);
        fprintf(out, "%s_COSINE,%d,%d,%d,%.5f\n", config.label, ref_nb, query_nb, dim, time_cosine);
    
        free(ref);
        free(query);
        free(knn_dist);
        free(knn_index);
    }

    // Deallocate memory 
    free(ref);
    free(query);
    free(knn_dist);
    free(knn_index);

    fclose(out);

    // Visualization output (2D scatter plot + animation)
    FILE* vis_file = fopen("results/visual_knn.csv", "w");
    if (!vis_file) {
        printf("Error: Unable to create results/visual_knn.csv\n");
        return EXIT_FAILURE;
    }

    // Generate small 2D dataset for plotting
    float* ref_vis   = (float*) malloc(ref_nb_vis * dim_vis * sizeof(float));
    float* query_vis = (float*) malloc(query_nb_vis * dim_vis * sizeof(float));
    float* dist_vis  = (float*) malloc(query_nb_vis * k_vis * sizeof(float));
    int*   idx_vis   = (int*)   malloc(query_nb_vis * k_vis * sizeof(int));

    initialize_data(ref_vis, ref_nb_vis, query_vis, query_nb_vis, dim_vis);

    if (!knn_cuda_global(ref_vis, ref_nb_vis, query_vis, query_nb_vis, dim_vis, k_vis, dist_vis, idx_vis, EUCLIDEAN)) {
        printf("Error: kNN visualization computation failed\n");
        return EXIT_FAILURE;
    }

    // Write CSV headers
    fprintf(vis_file, "type,x,y,query_id\n");
    for (int i = 0; i < ref_nb_vis; ++i)
        fprintf(vis_file, "ref,%f,%f,\n", ref_vis[i * dim_vis + 0], ref_vis[i * dim_vis + 1]);
    for (int i = 0; i < query_nb_vis; ++i)
        fprintf(vis_file, "query,%f,%f,%d\n", query_vis[i * dim_vis + 0], query_vis[i * dim_vis + 1], i);
    for (int i = 0; i < query_nb_vis; ++i) {
        for (int j = 0; j < k_vis; ++j) {
            int ref_i = idx_vis[j * query_nb_vis + i];
            fprintf(vis_file, "neighbor,%f,%f,%d\n", ref_vis[ref_i * dim_vis + 0], ref_vis[ref_i * dim_vis + 1], i);
        }
    }

    fclose(vis_file);
    free(ref_vis);
    free(query_vis);
    free(dist_vis);
    free(idx_vis);

    return EXIT_SUCCESS;
}
