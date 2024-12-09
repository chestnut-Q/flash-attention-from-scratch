#include "mpi.h"  
#include "omp.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

const char* version_name = "Optimized implementation.";

extern void custom_sgemm (int, int, int, float*, float*, float*, float);
extern void reference_sgemm (int, int, int, float*, float*, float*, int, int, int);
/*
transpose: 由 row-major 转为 column-major
*/
static void padding_and_transform(const float* src, float* dst, size_t m, size_t padded_m, size_t n, size_t padded_n, int transpose) {
    if (!transpose) {
        #pragma omp parallel for
        for (size_t i = 0; i < m; ++i) {
            memcpy(dst + i * padded_n, src + i * n, n * sizeof(float));
        }
    } else {
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                dst[j * padded_m + i] = src[i * n + j];
            }
        }
    }
}

static void copy_result_and_transform(const float* src, float* dst, size_t m, size_t padded_m, size_t n, size_t padded_n, int transpose) {
    if (!transpose) {
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                dst[i * n + j] += src[i * padded_n + j];
            }
        }
    } else {
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                dst[i * n + j] += src[j * padded_m + i];
            }
        }
    }
}

static inline void partition_m_dim(const int rank, const int size, const int m, int *offset, int *length)
{
    int band = m / size;
    int tail = m - (size - 1) * band;
    if (tail > (band + 1)){
        band++;
    }
    tail = m - (size - 1) * band;
    if (rank < (size - 1)){
        *length = band;
    }else{
        *length = tail;
    }
    *offset = rank * band;
    if (*offset >= m) {
        *length = 0;
        *offset = 0;
    }else if ((*offset + *length) > m) {
        *length = m - *offset;
    }
}

static void padded_attention_naive(float* Q, float* K, float* V, float* Y, float *S, int m, const int padded_m, const int n, const int padded_n, const float softmax_scale) {
    const int kHeadDim = padded_n;
    // 计算 S^T = K * Q^T
    reference_sgemm(padded_n, kHeadDim, padded_m, K, Q, S, 1, 1, 1);
    
    // 归一化 S^T
    for (int i = 0; i < padded_m; i++) {
        for (int j = 0; j < padded_n; j++) {
            if (j >= n) {
                S[i * padded_n + j] = -INFINITY;
            } else {
                S[i * padded_n + j] *= softmax_scale;
            }
        }
    }

    // P^T = softmax(S^T)
    for (int i = 0; i < padded_m; i++) {
        float max_val = -INFINITY;
        for (int j = 0; j < padded_n; j++) {
            if (S[i * padded_n + j] > max_val) {
                max_val = S[i * padded_n + j];
            }
        }

        float sum_exp = 0;
        // 计算Softmax的分母
        for (int j = 0; j < padded_n; j++) {
            S[i * padded_n + j] = expf(S[i * padded_n + j] - max_val);  // 减去最大值来避免溢出
            sum_exp += S[i * padded_n + j];
        }

        // 计算Softmax结果
        for (int j = 0; j < padded_n; j++) {
            S[i * padded_n + j] /= sum_exp;
        }
    }

    // 计算 Y^T = V^T * P^T
    reference_sgemm(kHeadDim, padded_n, padded_m, V, S, Y, 1, 1, 1);
}

static void padded_attention(float* Q, float* K, float* V, float* Y, float *S, int m, const int padded_m, const int n, const int padded_n, const float softmax_scale) {
    const int kHeadDim = padded_n;
    // 计算 S^T = K * Q^T
    // reference_sgemm(padded_n, kHeadDim, padded_m, K, Q, S, 1, 1, 1);
    custom_sgemm(padded_n, kHeadDim, padded_m, K, Q, S, 1.0);
    
    // 归一化 S^T
    for (int i = 0; i < padded_m; i++) {
        for (int j = 0; j < padded_n; j++) {
            if (j >= n) {
                S[i * padded_n + j] = -INFINITY;
            } else {
                S[i * padded_n + j] *= softmax_scale;
            }
        }
    }

    // P^T = softmax(S^T)
    for (int i = 0; i < padded_m; i++) {
        float max_val = -INFINITY;
        for (int j = 0; j < padded_n; j++) {
            if (S[i * padded_n + j] > max_val) {
                max_val = S[i * padded_n + j];
            }
        }

        float sum_exp = 0;
        // 计算Softmax的分母
        for (int j = 0; j < padded_n; j++) {
            S[i * padded_n + j] = expf(S[i * padded_n + j] - max_val);  // 减去最大值来避免溢出
            sum_exp += S[i * padded_n + j];
        }

        // 计算Softmax结果
        for (int j = 0; j < padded_n; j++) {
            S[i * padded_n + j] /= sum_exp;
        }
    }

    // 计算 Y^T = V^T * P^T
    // reference_sgemm(kHeadDim, padded_n, padded_m, V, S, Y, 1, 1, 1);
    custom_sgemm(kHeadDim, padded_n, padded_m, V, S, Y, 1.0);
}

void square_attention (int n, float* Q, float* K, float* V, float* Y, int rank, int size)
{
    int m_offset = 0, m_length = 0;
    partition_m_dim(rank, size, n, &m_offset, &m_length);

    // scatter Q
    float *Q_global = NULL;
    if (rank == 0) {
        Q_global = Q;
    }
    
    // Use MPI_Scatterv to distribute Q
    int *sendcounts_Q = NULL;
    int *displs_Q = NULL;
    if (rank == 0) {
        sendcounts_Q = (int*)malloc(size * sizeof(int));
        displs_Q = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            int offset, length;
            partition_m_dim(i, size, n, &offset, &length);
            sendcounts_Q[i] = length * n;
            displs_Q[i] = offset * n;
        }
    }
    
    MPI_Scatterv(Q_global, sendcounts_Q, displs_Q, MPI_FLOAT, Q, m_length * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(K, n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(V, n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // local attention
    const int ALIGN = 64;
    const float softmax_scale = 1.0 / sqrt(n);
    float *QT_l, *K_g, *VT_g, *YT_l, *ST_l, *padded_memory;
    int padded_n = ((n + ALIGN - 1) / ALIGN) * ALIGN;
    int padded_m = ((m_length + ALIGN - 1) / ALIGN) * ALIGN;
    padded_memory = (float*)calloc(3 * padded_n * padded_m + 2 * padded_n * padded_n, sizeof(float));
    if (!padded_memory) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    QT_l = padded_memory; // padded_n * padded_m
    K_g = padded_memory + padded_n * padded_m; // padded_n * padded_n
    VT_g = padded_memory + padded_n * padded_m + padded_n * padded_n; // padded_n * padded_n
    YT_l = padded_memory + padded_n * padded_m + 2 * padded_n * padded_n; // padded_n * padded_m
    ST_l = padded_memory + 2 * padded_n * padded_m + 2 * padded_n * padded_n; // padded_n * padded_m

    padding_and_transform(Q, QT_l, m_length, padded_m, n, padded_n, 0);
    padding_and_transform(K, K_g, n, padded_n, n, padded_n, 1);
    padding_and_transform(V, VT_g, n, padded_n, n, padded_n, 0);
    
    padded_attention(QT_l, K_g, VT_g, YT_l, ST_l, m_length, padded_m, n, padded_n, softmax_scale);

    copy_result_and_transform(YT_l, Y, m_length, padded_m, n, padded_n, 0);

    free(padded_memory);
    
    // gather output
    float *Y_global = NULL;
    if (rank == 0) {
        Y_global = Y;
    }
    
    // Use MPI_Gatherv to collect Y from all ranks
    int *recvcounts_Y = NULL;
    int *displs_Y = NULL;
    if (rank == 0) {
        recvcounts_Y = (int*)malloc(size * sizeof(int));
        displs_Y = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            int offset, length;
            partition_m_dim(i, size, n, &offset, &length);
            recvcounts_Y[i] = length * n;
            displs_Y[i] = offset * n;
        }
    }
    MPI_Gatherv(Y, m_length * n, MPI_FLOAT, Y_global, recvcounts_Y, displs_Y, MPI_FLOAT, 0, MPI_COMM_WORLD);
}
