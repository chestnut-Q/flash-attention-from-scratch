#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define CFLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

using index_t = int64_t;

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (error code: %d) at %s:%d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

const char *version_name = "Optimized implementation.";

__global__ void fwd_kernel_naive(const float *Q, const float *K, const float *V, float* O, const int N, 
                            const int kBlockM, const int kBlockN, const int kBlockK, const int kHeadDim,
                            const int nBlockM, const int nBlockN, const float softmax_scale) {
    // TODO: 处理非对齐情况
    const int tidx = threadIdx.x;
    const int bdim = blockDim.x;
    const int bidx = blockIdx.x;

    // kBlockM = 64;
    // kBlockN = 64;
    // kBlockK = 8;
    // blockDim = 64;

    extern __shared__ float smem[];

    float* sS = smem; // kBlockM * kBlockN 
    float* sm = &smem[kBlockM * kBlockN ]; // kBlockM
    float* sl = &smem[kBlockM * kBlockN + kBlockM]; // kBlockM
    
    // Initialize sO, sm, sl
    for (int i = tidx; i < kBlockM; i += bdim) {
        sm[i] = -INFINITY;
        sl[i] = 0.0;
    }

    for (int i = 0; i < nBlockN; ++i) { // 全局遍历 KV
        for (int j = tidx; j < kBlockM; j += bdim) { // 遍历该 tidx 对应的 Q
            float row_m_prev = sm[j];
            float row_l_prev = sl[j];
            // S = QK^T, row_m = rowmax(S)
            // row_m_new = max(row_m_prev, row_m)
            float row_m = -INFINITY;
            for (int k = 0; k < kBlockN; ++k) {
                float sum = 0.0;
                for (int m = 0; m < kHeadDim; ++m) {
                    sum += Q[(bidx * kBlockM + j) * kHeadDim + m] * K[(i * kBlockN + k) * kHeadDim + m];
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

            // O = O_old * exp(row_m_prev - row_m_new) + PV
            for (int k = 0; k < kHeadDim; ++k) {
                float pv = 0.0;  // Pij * Vj
                for (int m = 0; m < kBlockN; ++m) {
                    pv += sS[j * kBlockN + m] * V[(i * kBlockN + m) * kHeadDim + k];
                }
                O[(bidx * kBlockM + j) * kHeadDim + k] = (__expf(row_m_prev - row_m_new) * O[(bidx * kBlockM + j) * kHeadDim + k]) + pv;
            }
            sm[j] = row_m_new;
            sl[j] = row_l_new;
            __syncthreads();
        }
    }

    // rescale O and copy sO -> gO
    for (int i = tidx; i < kBlockM && bidx * kBlockM + i < N; i += bdim) {
        for (int j = 0; j < kHeadDim; ++j) {
            O[(bidx * kBlockM + i) * kHeadDim + j] = O[(bidx * kBlockM + i) * kHeadDim + j] / sl[i];
        }
    }
}

__global__ void fwd_kernel(const float *Q, const float *K, const float *V, float* O, const int N, const int real_n,
                            const int kBlockM, const int kBlockN, const int kBlockK, const int kHeadDim,
                            const int nBlockM, const int nBlockN, const float softmax_scale) {
    // kBlockM = 64;
    // kBlockN = 64;
    // kBlockK = 8;
    // blockDim = 64;
    const int tBlockM = 8;
    const int tBlockN = 8;

    const int tx = threadIdx.x; // 0-7
    const int ty = threadIdx.y; // 0-7
    const int tid = ty * blockDim.x + tx; // 0-63
    const int bidx = blockIdx.x;

    const int rsize = tBlockM * tBlockN; // 寄存器大小限制
    const int qid = tid; // 每个 thread 负责一个 q

    extern __shared__ float smem[];

    float* sQ = smem; // kBlockK * kBlockM, shared with sO
    float* sK = &smem[kBlockK * kBlockM]; // kBlockK * kBlockN
    float* sV = &smem[kBlockK * (kBlockM + kBlockN)]; // kBlockK * kBlockN
    float* sS = &smem[kBlockK * (kBlockM + 2 * kBlockN)]; // kBlockM * kBlockN 
    float* sl = &smem[kBlockK * (kBlockM + 2 * kBlockN) + kBlockM * kBlockN ]; // kBlockM
    float* s_rescale = &smem[kBlockK * (kBlockM + 2 * kBlockN) + kBlockM * kBlockN + kBlockM]; // kBlockM

    float rS[tBlockM * tBlockN] = {0.0}; // tBlockM * tBlockN
    float row_m_prev = -INFINITY, row_m_new = -INFINITY, row_l_prev = 0.0, row_l_new = 0.0;
    
    for (int i = 0; i < nBlockN; ++i) { // 全局遍历 KV
        for (int j = 0; j < tBlockM * tBlockN; ++j) {
            rS[j] = 0.0;
        }
        // S = QK^T
        for (int j = 0; j < (kHeadDim + kBlockK - 1) / kBlockK; ++j) {
            // copy gQ -> sQ, gK -> sK
            index_t offset_gqx = bidx * kBlockM + tid; index_t offset_gqy = j * kBlockK;
            index_t offset_gkx = i * kBlockN + tid; index_t offset_gky = j * kBlockK;

            FLOAT4(sQ[OFFSET(tid, 0, kBlockK)]) = CFLOAT4(Q[OFFSET(offset_gqx, offset_gqy, kHeadDim)]);
            FLOAT4(sQ[OFFSET(tid, 4, kBlockK)]) = CFLOAT4(Q[OFFSET(offset_gqx, offset_gqy + 4, kHeadDim)]);
            FLOAT4(sK[OFFSET(tid, 0, kBlockK)]) = CFLOAT4(K[OFFSET(offset_gkx, offset_gky, kHeadDim)]);
            FLOAT4(sK[OFFSET(tid, 4, kBlockK)]) = CFLOAT4(K[OFFSET(offset_gkx, offset_gky + 4, kHeadDim)]);
            __syncthreads();

            // rS = sQ * sK^T
            #pragma unroll
            for (int k = 0; k < kBlockK; ++k) {
                #pragma unroll
                for (int ii = 0; ii < tBlockM; ++ii) {
                    #pragma unroll
                    for (int jj = 0; jj < tBlockN; ++jj) {
                        rS[OFFSET(ii, jj, tBlockN)] += sQ[OFFSET(ty * tBlockM + ii, k, kBlockK)] * sK[OFFSET(tx * tBlockN + jj, k, kBlockK)];
                    }
                }
            }
            __syncthreads();
        }

        // rS -> sS
        for (int ii = 0; ii < tBlockM; ++ii) {
            for (int jj = 0; jj < tBlockN; jj += 4) {
                FLOAT4(sS[OFFSET(ty * tBlockM + ii, tx * tBlockN + jj, kBlockN)]) = FLOAT4(rS[OFFSET(ii, jj, tBlockN)]);
            }
        }
        __syncthreads();

        // S = S * softmax_scale
        // row_m_new = max(row_m_prev, rowmax(S))
        // P = exp(S - row_m_new), row_l = rowsum(P)
        // row_l_new = exp(row_m_prev - row_m_new) * row_l_prev + row_l
        float row_m = -INFINITY;
        for (int k = 0; k < rsize; k += 4) {
            FLOAT4(rS[k]) = FLOAT4(sS[OFFSET(qid, k, kBlockN)]);
        }
        for (int k = 0; k < rsize; ++k) {
            if (i * kBlockN + k < real_n) {
                rS[k] *= softmax_scale;
            } else {
                rS[k] = -INFINITY;
            }
            
            if (rS[k] > row_m) row_m = rS[k];
        }
        row_m_new = max(row_m_prev, row_m);

        // P = exp(S - row_m_new), row_l = rowsum(P)
        // row_l_new = exp(row_m_prev - row_m_new) * row_l_prev + row_l
        float row_l = 0.0;
        for (int k = 0; k < rsize; ++k) {
            rS[k] = __expf(rS[k] - row_m_new);
            row_l += rS[k];
        }
        row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev + row_l;
        for (int k = 0; k < rsize; k += 4) {
            FLOAT4(sS[OFFSET(qid, k, kBlockN)]) = FLOAT4(rS[k]);
        }
        s_rescale[qid] = __expf(row_m_prev - row_m_new); // 用于计算 O
        sl[qid] = row_l_new; // 用于计算 O
        row_m_prev = row_m_new;
        row_l_prev = row_l_new;
        __syncthreads();

        // compute O
        for (int j = 0; j < (kHeadDim + kBlockN - 1) / kBlockN; ++j) { // 在 kHeadDim 维度遍历
            // copy gO -> rO, rO = rO * exp(row_m_prev - row_m_new)
            if (i > 0) {
                for (int ii = 0; ii < tBlockM; ++ii) {
                    index_t offset_gox = bidx * kBlockM + ty * tBlockM + ii;
                    for (int jj = 0; jj < tBlockN; jj += 4) {
                        FLOAT4(rS[OFFSET(ii, jj, tBlockN)]) = FLOAT4(O[OFFSET(offset_gox, j * kBlockN + tx * tBlockN + jj, kHeadDim)]);
                    }
                }
                for (int ii = 0; ii < tBlockM; ++ii) {
                    float rescale_o = s_rescale[ty * tBlockM + ii];
                    for (int jj = 0; jj < tBlockN; ++jj) {
                        rS[OFFSET(ii, jj, tBlockN)] *= rescale_o;
                    }
                }
            } else {
                for (int ii = 0; ii < tBlockM; ++ii) {
                    for (int jj = 0; jj < tBlockN; ++jj) {
                        rS[OFFSET(ii, jj, tBlockN)] = 0.0;
                    }
                }
            }
            __syncthreads();

            // rO += sP * gV, copy rO -> sO
            for (int k = 0; k < (kBlockN + kBlockK - 1) / kBlockK; ++k) {
                // copy gO -> sO, gV -> sV, sS 已经在 smem 内了
                index_t offset_sx = tid >> 3; index_t offset_sy = (tid & 7) << 3;
                index_t offset_gvx = i * kBlockN + k * kBlockK + offset_sx;
                index_t offset_gvy = j * kBlockN + offset_sy;

                FLOAT4(sV[OFFSET(offset_sx, offset_sy, kBlockN)]) = CFLOAT4(V[OFFSET(offset_gvx, offset_gvy, kHeadDim)]);
                FLOAT4(sV[OFFSET(offset_sx, offset_sy + 4, kBlockN)]) = CFLOAT4(V[OFFSET(offset_gvx, offset_gvy + 4, kHeadDim)]);

                __syncthreads();

                #pragma unroll
                for (int kk = 0; kk < kBlockK; ++kk) {
                    #pragma unroll
                    for (int ii = 0; ii < tBlockM; ++ii) {
                        #pragma unroll
                        for (int jj = 0; jj < tBlockN; ++jj) {
                            rS[OFFSET(ii, jj, tBlockN)] += sS[OFFSET(ty * tBlockM + ii, k * kBlockK + kk, kBlockN)] * sV[OFFSET(kk, tx * tBlockN + jj, kBlockN)];
                        }
                    }
                }
                __syncthreads();
            }

            // copy rO -> gO
            if (i < nBlockN - 1) {
                for (int ii = 0; ii < tBlockM; ++ii) {
                    index_t offset_gox = bidx * kBlockM + ty * tBlockM + ii;
                    #pragma unroll
                    for (int jj = 0; jj < tBlockN; jj += 4) {
                        FLOAT4(O[OFFSET(offset_gox, j * kBlockN + tx * tBlockN + jj, kHeadDim)]) = FLOAT4(rS[OFFSET(ii, jj, tBlockN)]);
                    }
                }
            } else {
                // rescale O
                for (int ii = 0; ii < tBlockM; ++ii) {
                    float rescale_o = sl[ty * tBlockM + ii];
                    index_t offset_gox = bidx * kBlockM + ty * tBlockM + ii;
                    #pragma unroll
                    for (int jj = 0; jj < tBlockN; ++jj) {
                        O[OFFSET(offset_gox, j * kBlockN + tx * tBlockN + jj, kHeadDim)] = rS[OFFSET(ii, jj, tBlockN)] / rescale_o;
                    }
                }
            }
            __syncthreads();
        }
    }
}

void square_attention(int n, float *gpu_Q, float *gpu_K, float *gpu_V, float *gpu_Y) {
    // 预处理
    const int ALIGN = 64;
    float *new_gpu_Q, *new_gpu_K, *new_gpu_V, *new_gpu_Y, *padded_memory;
    int padded_n = ((n + ALIGN - 1) / ALIGN) * ALIGN;
    if (padded_n == n) {
        new_gpu_Q = gpu_Q;
        new_gpu_K = gpu_K;
        new_gpu_V = gpu_V;
        new_gpu_Y = gpu_Y;
    } else {
        // 处理非对齐矩阵
        size_t matrix_size = padded_n * padded_n * sizeof(float);
        CUDA_CHECK(cudaMalloc(&padded_memory, 4 * matrix_size));
        CUDA_CHECK(cudaMemset(padded_memory, 0, 4 * matrix_size));
        new_gpu_Q = padded_memory;
        new_gpu_K = padded_memory + matrix_size / sizeof(float);
        new_gpu_V = padded_memory + 2 * matrix_size / sizeof(float);
        new_gpu_Y = padded_memory + 3 * matrix_size / sizeof(float);
        
        size_t pitch = n * sizeof(float);
        // Use cudaMemcpy2D to handle padding
        CUDA_CHECK(cudaMemcpy2D(new_gpu_Q, padded_n * sizeof(float), gpu_Q, pitch,
                        n * sizeof(float), n, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy2D(new_gpu_K, padded_n * sizeof(float), gpu_K, pitch,
                        n * sizeof(float), n, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy2D(new_gpu_V, padded_n * sizeof(float), gpu_V, pitch,
                        n * sizeof(float), n, cudaMemcpyDeviceToDevice));
    }

    // kernel 参数
    const int kBlockM = 64, kBlockN = 64, kHeadDim = padded_n, kBlockK = 8;
    const int tBlockM = 8, tBlockN = 8;
    const int nBlockM = padded_n / kBlockM;
    const int nBlockN = padded_n / kBlockN;
    const float softmax_scale = 1.0 / sqrt(n);
    dim3 grid(nBlockM); // parallel Q
    dim3 block(kBlockN / tBlockN, kBlockM / tBlockM); // 等于 kBlockM，即每个线程负责一个 q

    const int sram_size = (kBlockK * (kBlockM + 2 * kBlockN) + kBlockM * kBlockN + 2 * kBlockM) * sizeof(float);

    fwd_kernel<<<grid, block, sram_size>>>(
        new_gpu_Q, new_gpu_K, new_gpu_V, new_gpu_Y,
        padded_n, n, kBlockM, kBlockN, kBlockK, kHeadDim, nBlockM, nBlockN,
        softmax_scale
    );

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "fwd_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    // 拷贝回原数组
    if (padded_n != n) {
        CUDA_CHECK(cudaMemcpy2D(gpu_Y, n * sizeof(float), new_gpu_Y, padded_n * sizeof(float),
                        n * sizeof(float), n, cudaMemcpyDeviceToDevice));
        cudaFree(padded_memory);
    }
}
