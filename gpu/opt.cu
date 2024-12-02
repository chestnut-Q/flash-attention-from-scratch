#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using index_t = int64_t;

const char *version_name = "Optimized implementation.";

__global__ void fwd_kernel(const float *Q, const float *K, const float *V, float* O, const int N, 
                            const int kBlockM, const int kBlockN, const int kHeadDim,
                            const int nBlockM, const int nBlockN, const float softmax_scale) {
    // TODO: 处理非对齐情况
    const int tidx = threadIdx.x;
    const int bdim = blockDim.x;
    const int bidx = blockIdx.x;

    extern __shared__ float smem[];

    int tile_size = max(kBlockM, kBlockN) * kHeadDim;
    float* sQ = smem;
    float* sK = &smem[tile_size];
    float* sV = &smem[tile_size * 2];
    float* sS = &smem[tile_size * 3];
    float* sO = &smem[tile_size * 4];
    float* sm = &smem[tile_size * 5];
    float* sl = &smem[tile_size * 5 + kBlockM];
    
    // Copy gQ -> sQ
    for (int i = tidx; i < kBlockM; i += bdim) {
        for (int j = 0; j < kHeadDim; ++j) {
            sQ[i * kHeadDim + j] = Q[(bidx * kBlockM + i) * kHeadDim + j];
        }
    }
    // Initialize sO, sm, sl
    for (int i = tidx; i < kBlockM; i += bdim) {
        for (int j = 0; j < kHeadDim; ++j) {
            sO[i * kHeadDim + j] = 0.0;
        }
        sm[i] = -INFINITY;
        sl[i] = 0.0;
    }

    for (int i = 0; i < nBlockN; ++i) { // 全局遍历 KV
        // copy gK -> sK, gV, -> sV
        for (int j = tidx; j < kBlockN; j += bdim) {
            auto mask = i * kBlockN + j >= N;
            for (int k = 0; k < kHeadDim; ++k) {
                sK[j * kHeadDim + k] = mask ? -INFINITY : K[(i * kBlockN + j) * kHeadDim + k];
                sV[j * kHeadDim + k] = mask ? 0.0 : V[(i * kBlockN + j) * kHeadDim + k];
            }
        }
        __syncthreads();
        
        for (int j = tidx; j < kBlockM; j += bdim) { // 遍历该 tidx 对应的 Q
            float row_m_prev = sm[j];
            float row_l_prev = sl[j];
            // S = QK^T, row_m = rowmax(S)
            // row_m_new = max(row_m_prev, row_m)
            float row_m = -INFINITY;
            for (int k = 0; k < kBlockN; ++k) {
                float sum = 0.0;
                for (int m = 0; m < kHeadDim; ++m) {
                    sum += sQ[j * kHeadDim + m] * sK[k * kHeadDim + m];
                }
                sum *= softmax_scale;
                sS[j * kBlockN + k] = sum;
                if (sum > row_m)
                    row_m = sum;
            }
            float row_m_new = max(row_m_prev, row_m);

            // P = exp(S - row_m_new), row_l = rowsum(P)
            // row_l_new = exp(row_m_prev - row_m_new) * row_l_prev + row_l
            float row_l = 0.0;
            for (int k = 0; k < kBlockN; ++k) {
                sS[j * kBlockN + k] = __expf(sS[j * kBlockN + k] - row_m_new);
                row_l += sS[j * kBlockN + k];
            }
            float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev + row_l;

            // O = O_old * l * exp(row_m_prev - row_m_new) + PV
            for (int k = 0; k < kHeadDim; ++k) {
                float pv = 0.0;  // Pij * Vj
                for (int m = 0; m < kBlockN; ++m) {
                    pv += sS[j * kBlockN + m] * sV[m * kHeadDim + k];
                }
                sO[j * kHeadDim + k] = (__expf(row_m_prev - row_m_new) * sO[j * kHeadDim + k]) + pv;
            }
            sm[j] = row_m_new;
            sl[j] = row_l_new;
            __syncthreads();
        }
    }

    // rescale O and copy sO -> gO
    for (int i = tidx; i < kBlockM && bidx * kBlockM + i < N; i += bdim) {
        for (int j = 0; j < kHeadDim; ++j) {
            O[(bidx * kBlockM + i) * kHeadDim + j] = sO[i * kHeadDim + j] / sl[i];
        }
    }
}

void square_attention(int n, float *gpu_Q, float *gpu_K, float *gpu_V, float *gpu_Y) {
    const int kBlockM = 32, kBlockN = 32, kHeadDim = n;
    const int nBlockM = (n + kBlockM - 1) / kBlockM;
    const int nBlockN = (n + kBlockN - 1) / kBlockN;
    const float softmax_scale = 1.0 / sqrt(kHeadDim);
    dim3 grid(nBlockM);   // Q parallel
    dim3 block(kBlockM);

    const int sram_size = (5 * max(kBlockM, kBlockN) * kHeadDim * sizeof(float)) + (2 * kBlockM * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    // printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    fwd_kernel<<<grid, block, sram_size>>>(
        gpu_Q, gpu_K, gpu_V, gpu_Y,
        n, kBlockM, kBlockN, kHeadDim, nBlockM, nBlockN,
        softmax_scale
    );

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "fwd_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching fwd_kernel: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
    }
}
