#include "immintrin.h"
#include "omp.h"
#include <stdio.h>
#include <assert.h>

const char* sgemm_desc = "Final sgemm.";

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

static void packing_a_32(float alpha, float *src, float *dst, int lda, int M0, int K0, int M_padding) {
  float *src_ptr, *dst_ptr;
  dst_ptr = dst;
  int count_first, count_second, count_sub = M0;
  __m512 valpha=_mm512_set1_ps(alpha);
  int i;
  for (count_first = 0; count_sub > 31; count_first += 32, count_sub -= 32) {
    src_ptr = src + count_first;
    for (count_second = 0; count_second < K0; count_second++) {
      _mm512_store_ps(dst_ptr, _mm512_mul_ps(_mm512_loadu_ps(src_ptr), valpha));
      _mm512_store_ps(dst_ptr + 16, _mm512_mul_ps(_mm512_loadu_ps(src_ptr + 16), valpha));  
      src_ptr += lda;
      dst_ptr += 32;
    }
  }

  if (M_padding > 0 && count_sub + M_padding == 32) {
    src_ptr = src + count_first;
    for (count_second = 0; count_second < K0; count_second++) {
      for (i = 0; i < count_sub; i++) {
          *(dst_ptr + i) = *(src_ptr + i) * alpha;
      }
      src_ptr += lda;
      dst_ptr += 32;
    }
    count_first += 32;
    count_sub = 0;
  }

  for (; count_sub > 0; count_first++, count_sub--) {
    src_ptr = src + count_first;
    for (count_second = 0; count_second < K0; count_second++) {
      *dst_ptr = *src_ptr * alpha;
      src_ptr += lda;
      dst_ptr++;
    }
  }
}

static void packing_a_64(float alpha, float *src, float *dst, int lda, int M0, int K0, int M_padding) {
  float *src_ptr, *dst_ptr;
  dst_ptr = dst;
  int count_first, count_second, count_sub = M0;
  __m512 valpha=_mm512_set1_ps(alpha);
  int i;
  for (count_first = 0; count_sub > 63; count_first += 64, count_sub -= 64) {
    src_ptr = src + count_first;
    for (count_second = 0; count_second < K0; count_second++) {
      _mm512_store_ps(dst_ptr, _mm512_mul_ps(_mm512_loadu_ps(src_ptr), valpha));
      _mm512_store_ps(dst_ptr + 16, _mm512_mul_ps(_mm512_loadu_ps(src_ptr + 16), valpha));  
      _mm512_store_ps(dst_ptr + 32, _mm512_mul_ps(_mm512_loadu_ps(src_ptr + 32), valpha));  
      _mm512_store_ps(dst_ptr + 48, _mm512_mul_ps(_mm512_loadu_ps(src_ptr + 48), valpha));      
      src_ptr += lda;
      dst_ptr += 64;
    }
  }

  // pad to 64
  if (M_padding > 0 && count_sub + M_padding == 64) {
    src_ptr = src + count_first;
    for (count_second = 0; count_second < K0; count_second++) {
      for (i = 0; i < count_sub; i++) {
          *(dst_ptr + i) = *(src_ptr + i) * alpha;
      }
      src_ptr += lda;
      dst_ptr += 64;
    }
    count_first += 64;
    count_sub = 0;
  }

  for (; count_sub > 31; count_first += 32, count_sub -= 32) {
    src_ptr = src + count_first;
    for (count_second = 0; count_second < K0; count_second++) {
      _mm512_store_ps(dst_ptr, _mm512_mul_ps(_mm512_loadu_ps(src_ptr), valpha));
      _mm512_store_ps(dst_ptr + 16, _mm512_mul_ps(_mm512_loadu_ps(src_ptr + 16), valpha));      
      src_ptr += lda;
      dst_ptr += 32;
    }
  }

  // pad to 32
  if (M_padding > 0 && count_sub + M_padding == 32) {
    src_ptr = src + count_first;
    for (count_second = 0; count_second < K0; count_second++) {
      for (i = 0; i < count_sub; i++) {
          *(dst_ptr + i) = *(src_ptr + i) * alpha;
      }
      src_ptr += lda;
      dst_ptr += 32;
    }
    count_first += 32;
    count_sub = 0;
  }

  for (; count_sub > 0; count_first++, count_sub--) {
    src_ptr = src + count_first;
    for (count_second = 0; count_second < K0; count_second++) {
      *dst_ptr = *src_ptr * alpha;
      src_ptr += lda;
      dst_ptr++;
    }
  }
}

static void packing_b_4(float *src, float *dst, int ldb, int K0, int N0) {
  float *src0_ptr, *src1_ptr, *src2_ptr, *src3_ptr, *dst_ptr;
  dst_ptr = dst;
  int count_first, count_second, count_sub = N0;
  for (count_second = 0; count_sub > 3; count_second += 4, count_sub -= 4) {
    src0_ptr = src + count_second * ldb; src1_ptr = src0_ptr + ldb;
    src2_ptr = src1_ptr + ldb; src3_ptr = src2_ptr + ldb;
    for (count_first = 0; count_first < K0; count_first++) {
      *dst_ptr = *src0_ptr; dst_ptr++; src0_ptr++;
      *dst_ptr = *src1_ptr; dst_ptr++; src1_ptr++;
      *dst_ptr = *src2_ptr; dst_ptr++; src2_ptr++;
      *dst_ptr = *src3_ptr; dst_ptr++; src3_ptr++;
    }
  }
  for (; count_sub > 0; count_second++, count_sub--) {
    src0_ptr = src + count_second * ldb;
    for (count_first = 0; count_first < K0; count_first++) {
      *dst_ptr = *src0_ptr;
      dst_ptr++;
      src0_ptr++;
    }
  }
}

static void packing_b_8(float *src, float *dst, int ldb, int K0, int N0) {
  float *src0_ptr, *src1_ptr, *src2_ptr, *src3_ptr, *src4_ptr, *src5_ptr, *src6_ptr, *src7_ptr, \
        *dst_ptr;
  dst_ptr = dst;
  int count_first, count_second, count_sub = N0;
  for (count_second = 0; count_sub > 7; count_second += 8, count_sub -= 8) {
    src0_ptr = src + count_second * ldb; src1_ptr = src0_ptr + ldb;
    src2_ptr = src1_ptr + ldb; src3_ptr = src2_ptr + ldb;
    src4_ptr = src3_ptr + ldb; src5_ptr = src4_ptr + ldb;
    src6_ptr = src5_ptr + ldb; src7_ptr = src6_ptr + ldb;

    for (count_first = 0; count_first < K0; count_first++) {
      *dst_ptr = *src0_ptr; dst_ptr++; src0_ptr++;
      *dst_ptr = *src1_ptr; dst_ptr++; src1_ptr++;
      *dst_ptr = *src2_ptr; dst_ptr++; src2_ptr++;
      *dst_ptr = *src3_ptr; dst_ptr++; src3_ptr++;
      *dst_ptr = *src4_ptr; dst_ptr++; src4_ptr++;
      *dst_ptr = *src5_ptr; dst_ptr++; src5_ptr++;
      *dst_ptr = *src6_ptr; dst_ptr++; src6_ptr++;
      *dst_ptr = *src7_ptr; dst_ptr++; src7_ptr++;
    }
  }
  for (; count_sub > 0; count_second++, count_sub--) {
    src0_ptr = src + count_second * ldb;
    for (count_first = 0; count_first < K0; count_first++) {
      *dst_ptr = *src0_ptr;
      dst_ptr++;
      src0_ptr++;
    }
  }
}

static void nontrans_packing_b_8(float *src, float *dst, int ldb, int K0, int N0) {
  float *src0_ptr, *src1_ptr, *src2_ptr, *src3_ptr, *src4_ptr, *src5_ptr, *src6_ptr, *src7_ptr, *dst_ptr;
  dst_ptr = dst;
  int count_first, count_second, count_sub = N0;
  for (count_second = 0; count_sub > 7; count_second += 8, count_sub -= 8) {
    src0_ptr = src + count_second * ldb; src1_ptr = src0_ptr + ldb;
    src2_ptr = src1_ptr + ldb; src3_ptr = src2_ptr + ldb;
    src4_ptr = src3_ptr + ldb; src5_ptr = src4_ptr + ldb;
    src6_ptr = src5_ptr + ldb; src7_ptr = src6_ptr + ldb;

    for (count_first = 0; count_first < K0; count_first++) {
      *dst_ptr = *src0_ptr; dst_ptr++; src0_ptr++;
      *dst_ptr = *src1_ptr; dst_ptr++; src1_ptr++;
      *dst_ptr = *src2_ptr; dst_ptr++; src2_ptr++;
      *dst_ptr = *src3_ptr; dst_ptr++; src3_ptr++;
      *dst_ptr = *src4_ptr; dst_ptr++; src4_ptr++;
      *dst_ptr = *src5_ptr; dst_ptr++; src5_ptr++;
      *dst_ptr = *src6_ptr; dst_ptr++; src6_ptr++;
      *dst_ptr = *src7_ptr; dst_ptr++; src7_ptr++;
    }
  }
  for (; count_sub > 0; count_second++, count_sub--) {
    src0_ptr = src + count_second * ldb;
    for (count_first = 0; count_first < K0; count_first++) {
      *dst_ptr = *src0_ptr;
      dst_ptr++;
      src0_ptr++;
    }
  }
}

#define KERNEL_K1_32x4_avx512\
  a0 = _mm512_load_ps(ptr_a);\
  a1 = _mm512_load_ps(ptr_a + 16);\
  b0 = _mm512_set1_ps(*ptr_b);\
  b1 = _mm512_set1_ps(*(ptr_b + 1));\
  b2 = _mm512_set1_ps(*(ptr_b + 2));\
  b3 = _mm512_set1_ps(*(ptr_b + 3));\
  c00 = _mm512_fmadd_ps(a0, b0, c00);\
  c01 = _mm512_fmadd_ps(a0, b1, c01);\
  c02 = _mm512_fmadd_ps(a0, b2, c02);\
  c03 = _mm512_fmadd_ps(a0, b3, c03);\
  c10 = _mm512_fmadd_ps(a1, b0, c10);\
  c11 = _mm512_fmadd_ps(a1, b1, c11);\
  c12 = _mm512_fmadd_ps(a1, b2, c12);\
  c13 = _mm512_fmadd_ps(a1, b3, c13);\
  ptr_a += 32;\
  ptr_b += 4;

#define KERNEL_K1_32x8_avx512\
  a0 = _mm512_load_ps(ptr_a);\
  a1 = _mm512_load_ps(ptr_a + 16);\
  b0 = _mm512_set1_ps(*ptr_b);\
  b1 = _mm512_set1_ps(*(ptr_b + 1));\
  b2 = _mm512_set1_ps(*(ptr_b + 2));\
  b3 = _mm512_set1_ps(*(ptr_b + 3));\
  c00 = _mm512_fmadd_ps(a0, b0, c00);\
  c01 = _mm512_fmadd_ps(a0, b1, c01);\
  c02 = _mm512_fmadd_ps(a0, b2, c02);\
  c03 = _mm512_fmadd_ps(a0, b3, c03);\
  c10 = _mm512_fmadd_ps(a1, b0, c10);\
  c11 = _mm512_fmadd_ps(a1, b1, c11);\
  c12 = _mm512_fmadd_ps(a1, b2, c12);\
  c13 = _mm512_fmadd_ps(a1, b3, c13);\
  b0 = _mm512_set1_ps(*(ptr_b + 4));\
  b1 = _mm512_set1_ps(*(ptr_b + 5));\
  b2 = _mm512_set1_ps(*(ptr_b + 6));\
  b3 = _mm512_set1_ps(*(ptr_b + 7));\
  c04 = _mm512_fmadd_ps(a0, b0, c04);\
  c05 = _mm512_fmadd_ps(a0, b1, c05);\
  c06 = _mm512_fmadd_ps(a0, b2, c06);\
  c07 = _mm512_fmadd_ps(a0, b3, c07);\
  c14 = _mm512_fmadd_ps(a1, b0, c14);\
  c15 = _mm512_fmadd_ps(a1, b1, c15);\
  c16 = _mm512_fmadd_ps(a1, b2, c16);\
  c17 = _mm512_fmadd_ps(a1, b3, c17);\
  ptr_a += 32;\
  ptr_b += 8;

#define KERNEL_K1_nopacking_32x8_avx512\
  a0 = _mm512_load_ps(ptr_a);\
  a1 = _mm512_load_ps(ptr_a + 16);\
  b0 = _mm512_set1_ps(*ptr_b);\
  b1 = _mm512_set1_ps(*(ptr_b + ldb));\
  b2 = _mm512_set1_ps(*(ptr_b + 2 * ldb));\
  b3 = _mm512_set1_ps(*(ptr_b + 3 * ldb));\
  c00 = _mm512_fmadd_ps(a0, b0, c00);\
  c01 = _mm512_fmadd_ps(a0, b1, c01);\
  c02 = _mm512_fmadd_ps(a0, b2, c02);\
  c03 = _mm512_fmadd_ps(a0, b3, c03);\
  c10 = _mm512_fmadd_ps(a1, b0, c10);\
  c11 = _mm512_fmadd_ps(a1, b1, c11);\
  c12 = _mm512_fmadd_ps(a1, b2, c12);\
  c13 = _mm512_fmadd_ps(a1, b3, c13);\
  b0 = _mm512_set1_ps(*(ptr_b + 4 * ldb));\
  b1 = _mm512_set1_ps(*(ptr_b + 5 * ldb));\
  b2 = _mm512_set1_ps(*(ptr_b + 6 * ldb));\
  b3 = _mm512_set1_ps(*(ptr_b + 7 * ldb));\
  c04 = _mm512_fmadd_ps(a0, b0, c04);\
  c05 = _mm512_fmadd_ps(a0, b1, c05);\
  c06 = _mm512_fmadd_ps(a0, b2, c06);\
  c07 = _mm512_fmadd_ps(a0, b3, c07);\
  c14 = _mm512_fmadd_ps(a1, b0, c14);\
  c15 = _mm512_fmadd_ps(a1, b1, c15);\
  c16 = _mm512_fmadd_ps(a1, b2, c16);\
  c17 = _mm512_fmadd_ps(a1, b3, c17);\
  ptr_a += lda;\
  ptr_b++;

#define KERNEL_K1_64x4_avx512\
  a0 = _mm512_load_ps(ptr_a);\
  a1 = _mm512_load_ps(ptr_a + 16);\
  a2 = _mm512_load_ps(ptr_a + 32);\
  a3 = _mm512_load_ps(ptr_a + 48);\
  b0 = _mm512_set1_ps(*ptr_b);\
  b1 = _mm512_set1_ps(*(ptr_b + 1));\
  b2 = _mm512_set1_ps(*(ptr_b + 2));\
  b3 = _mm512_set1_ps(*(ptr_b + 3));\
  c00 = _mm512_fmadd_ps(a0, b0, c00);\
  c01 = _mm512_fmadd_ps(a0, b1, c01);\
  c02 = _mm512_fmadd_ps(a0, b2, c02);\
  c03 = _mm512_fmadd_ps(a0, b3, c03);\
  c10 = _mm512_fmadd_ps(a1, b0, c10);\
  c11 = _mm512_fmadd_ps(a1, b1, c11);\
  c12 = _mm512_fmadd_ps(a1, b2, c12);\
  c13 = _mm512_fmadd_ps(a1, b3, c13);\
  c20 = _mm512_fmadd_ps(a2, b0, c20);\
  c21 = _mm512_fmadd_ps(a2, b1, c21);\
  c22 = _mm512_fmadd_ps(a2, b2, c22);\
  c23 = _mm512_fmadd_ps(a2, b3, c23);\
  c30 = _mm512_fmadd_ps(a3, b0, c30);\
  c31 = _mm512_fmadd_ps(a3, b1, c31);\
  c32 = _mm512_fmadd_ps(a3, b2, c32);\
  c33 = _mm512_fmadd_ps(a3, b3, c33);\
  ptr_a += 64;\
  ptr_b += 4;

__m512 a0, a1, a2, a3, b0, b1, b2, b3, b4, b5, b6, b7,\
      c00, c01, c02, c03, c04, c05, c06, c07,\
      c10, c11, c12, c13, c14, c15, c16, c17,\
      c20, c21, c22, c23, c30, c31, c32, c33;
float sa, sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7, sb0, sb1, sb2, sb3, sb4, sb5, sb6, sb7;
int MICRO_M_BLOCK_SIZE, MICRO_N_BLOCK_SIZE;

static void block_sgemm_kernel_64xkx4_avx512(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, int K, int ldc, __mmask16 mask) {
  float *ptr_a, *ptr_b;
  ptr_a = A;
  ptr_b = B;
  c00 = _mm512_setzero_ps(); c01 = _mm512_setzero_ps();
  c02 = _mm512_setzero_ps(); c03 = _mm512_setzero_ps();
  c10 = _mm512_setzero_ps(); c11 = _mm512_setzero_ps();
  c12 = _mm512_setzero_ps(); c13 = _mm512_setzero_ps();
  c20 = _mm512_setzero_ps(); c21 = _mm512_setzero_ps();
  c22 = _mm512_setzero_ps(); c23 = _mm512_setzero_ps();
  c30 = _mm512_setzero_ps(); c31 = _mm512_setzero_ps();
  c32 = _mm512_setzero_ps(); c33 = _mm512_setzero_ps();
  for (int k = 0; k < K; k++) {
    KERNEL_K1_64x4_avx512
  }
  _mm512_storeu_ps(C, _mm512_add_ps(c00, _mm512_loadu_ps(C)));
  _mm512_storeu_ps(C + ldc, _mm512_add_ps(c01, _mm512_loadu_ps(C + ldc)));
  _mm512_storeu_ps(C + ldc * 2, _mm512_add_ps(c02, _mm512_loadu_ps(C + ldc * 2)));
  _mm512_storeu_ps(C + ldc * 3, _mm512_add_ps(c03, _mm512_loadu_ps(C + ldc * 3)));
  _mm512_storeu_ps(C + 16, _mm512_add_ps(c10, _mm512_loadu_ps(C + 16)));
  _mm512_storeu_ps(C + 16 + ldc, _mm512_add_ps(c11, _mm512_loadu_ps(C + 16 + ldc)));
  _mm512_storeu_ps(C + 16 + ldc * 2, _mm512_add_ps(c12, _mm512_loadu_ps(C + 16 + ldc * 2)));
  _mm512_storeu_ps(C + 16 + ldc * 3, _mm512_add_ps(c13, _mm512_loadu_ps(C + 16 + ldc * 3)));
  _mm512_storeu_ps(C + 32, _mm512_add_ps(c20, _mm512_loadu_ps(C + 32)));
  _mm512_storeu_ps(C + 32 + ldc, _mm512_add_ps(c21, _mm512_loadu_ps(C + 32 + ldc)));
  _mm512_storeu_ps(C + 32 + ldc * 2, _mm512_add_ps(c22, _mm512_loadu_ps(C + 32 + ldc * 2)));
  _mm512_storeu_ps(C + 32 + ldc * 3, _mm512_add_ps(c23, _mm512_loadu_ps(C + 32 + ldc * 3)));
  if (mask != 0xFFFF) {
    _mm512_mask_storeu_ps(C + 48, mask, _mm512_add_ps(c30, _mm512_maskz_loadu_ps(mask, C + 48)));
    _mm512_mask_storeu_ps(C + 48 + ldc, mask, _mm512_add_ps(c31, _mm512_maskz_loadu_ps(mask, C + 48 + ldc)));
    _mm512_mask_storeu_ps(C + 48 + ldc * 2, mask, _mm512_add_ps(c32, _mm512_maskz_loadu_ps(mask, C + 48 + ldc * 2)));
    _mm512_mask_storeu_ps(C + 48 + ldc * 3, mask, _mm512_add_ps(c33, _mm512_maskz_loadu_ps(mask, C + 48 + ldc * 3)));
  } else {
    _mm512_storeu_ps(C + 48, _mm512_add_ps(c30, _mm512_loadu_ps(C + 48)));
    _mm512_storeu_ps(C + 48 + ldc, _mm512_add_ps(c31, _mm512_loadu_ps(C + 48 + ldc)));
    _mm512_storeu_ps(C + 48 + ldc * 2, _mm512_add_ps(c32, _mm512_loadu_ps(C + 48 + ldc * 2)));
    _mm512_storeu_ps(C + 48 + ldc * 3, _mm512_add_ps(c33, _mm512_loadu_ps(C + 48 + ldc * 3)));
  }
}

static void block_sgemm_kernel_32xkx8_avx512(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, int K, int ldc, __mmask16 mask) {
  float *ptr_a, *ptr_b;
  ptr_a = A;
  ptr_b = B;
  c00 = _mm512_setzero_ps(); c01 = _mm512_setzero_ps();
  c02 = _mm512_setzero_ps(); c03 = _mm512_setzero_ps();
  c04 = _mm512_setzero_ps(); c05 = _mm512_setzero_ps();
  c06 = _mm512_setzero_ps(); c07 = _mm512_setzero_ps();
  c10 = _mm512_setzero_ps(); c11 = _mm512_setzero_ps();
  c12 = _mm512_setzero_ps(); c13 = _mm512_setzero_ps();
  c14 = _mm512_setzero_ps(); c15 = _mm512_setzero_ps();
  c16 = _mm512_setzero_ps(); c17 = _mm512_setzero_ps();
  for (int k = 0; k < K; k++) {
    KERNEL_K1_32x8_avx512
  }

  _mm512_storeu_ps(C, _mm512_add_ps(c00, _mm512_loadu_ps(C)));
  _mm512_storeu_ps(C + ldc, _mm512_add_ps(c01, _mm512_loadu_ps(C + ldc)));
  _mm512_storeu_ps(C + ldc * 2, _mm512_add_ps(c02, _mm512_loadu_ps(C + ldc * 2)));
  _mm512_storeu_ps(C + ldc * 3, _mm512_add_ps(c03, _mm512_loadu_ps(C + ldc * 3)));
  _mm512_storeu_ps(C + ldc * 4, _mm512_add_ps(c04, _mm512_loadu_ps(C + ldc * 4)));
  _mm512_storeu_ps(C + ldc * 5, _mm512_add_ps(c05, _mm512_loadu_ps(C + ldc * 5)));
  _mm512_storeu_ps(C + ldc * 6, _mm512_add_ps(c06, _mm512_loadu_ps(C + ldc * 6)));
  _mm512_storeu_ps(C + ldc * 7, _mm512_add_ps(c07, _mm512_loadu_ps(C + ldc * 7)));
  if (mask != 0xFFFF) {
    _mm512_mask_storeu_ps(C + 16, mask, _mm512_add_ps(c10, _mm512_loadu_ps(C + 16)));
    _mm512_mask_storeu_ps(C + 16 + ldc, mask, _mm512_add_ps(c11, _mm512_loadu_ps(C + 16 + ldc)));
    _mm512_mask_storeu_ps(C + 16 + ldc * 2, mask, _mm512_add_ps(c12, _mm512_loadu_ps(C + 16 + ldc * 2)));
    _mm512_mask_storeu_ps(C + 16 + ldc * 3, mask, _mm512_add_ps(c13, _mm512_loadu_ps(C + 16 + ldc * 3)));
    _mm512_mask_storeu_ps(C + 16 + ldc * 4, mask, _mm512_add_ps(c14, _mm512_loadu_ps(C + 16 + ldc * 4)));
    _mm512_mask_storeu_ps(C + 16 + ldc * 5, mask, _mm512_add_ps(c15, _mm512_loadu_ps(C + 16 + ldc * 5)));
    _mm512_mask_storeu_ps(C + 16 + ldc * 6, mask, _mm512_add_ps(c16, _mm512_loadu_ps(C + 16 + ldc * 6)));
    _mm512_mask_storeu_ps(C + 16 + ldc * 7, mask, _mm512_add_ps(c17, _mm512_loadu_ps(C + 16 + ldc * 7)));
  } else {
    _mm512_storeu_ps(C + 16, _mm512_add_ps(c10, _mm512_loadu_ps(C + 16)));
    _mm512_storeu_ps(C + 16 + ldc, _mm512_add_ps(c11, _mm512_loadu_ps(C + 16 + ldc)));
    _mm512_storeu_ps(C + 16 + ldc * 2, _mm512_add_ps(c12, _mm512_loadu_ps(C + 16 + ldc * 2)));
    _mm512_storeu_ps(C + 16 + ldc * 3, _mm512_add_ps(c13, _mm512_loadu_ps(C + 16 + ldc * 3)));
    _mm512_storeu_ps(C + 16 + ldc * 4, _mm512_add_ps(c14, _mm512_loadu_ps(C + 16 + ldc * 4)));
    _mm512_storeu_ps(C + 16 + ldc * 5, _mm512_add_ps(c15, _mm512_loadu_ps(C + 16 + ldc * 5)));
    _mm512_storeu_ps(C + 16 + ldc * 6, _mm512_add_ps(c16, _mm512_loadu_ps(C + 16 + ldc * 6)));
    _mm512_storeu_ps(C + 16 + ldc * 7, _mm512_add_ps(c17, _mm512_loadu_ps(C + 16 + ldc * 7)));
  }
}

static void block_sgemm_kernel_nopacking_32xkx8_avx512(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, int M, int K, int lda, int ldb, int ldc) {
  float *ptr_a, *ptr_b;
  ptr_a = A;
  ptr_b = B;
  c00 = _mm512_setzero_ps(); c01 = _mm512_setzero_ps();
  c02 = _mm512_setzero_ps(); c03 = _mm512_setzero_ps();
  c04 = _mm512_setzero_ps(); c05 = _mm512_setzero_ps();
  c06 = _mm512_setzero_ps(); c07 = _mm512_setzero_ps();
  c10 = _mm512_setzero_ps(); c11 = _mm512_setzero_ps();
  c12 = _mm512_setzero_ps(); c13 = _mm512_setzero_ps();
  c14 = _mm512_setzero_ps(); c15 = _mm512_setzero_ps();
  c16 = _mm512_setzero_ps(); c17 = _mm512_setzero_ps();
  for (int k = 0; k < K; k++) {
    KERNEL_K1_nopacking_32x8_avx512
  }
  _mm512_storeu_ps(C, _mm512_add_ps(c00, _mm512_loadu_ps(C)));
  _mm512_storeu_ps(C + ldc, _mm512_add_ps(c01, _mm512_loadu_ps(C + ldc)));
  _mm512_storeu_ps(C + ldc * 2, _mm512_add_ps(c02, _mm512_loadu_ps(C + ldc * 2)));
  _mm512_storeu_ps(C + ldc * 3, _mm512_add_ps(c03, _mm512_loadu_ps(C + ldc * 3)));
  _mm512_storeu_ps(C + ldc * 4, _mm512_add_ps(c04, _mm512_loadu_ps(C + ldc * 4)));
  _mm512_storeu_ps(C + ldc * 5, _mm512_add_ps(c05, _mm512_loadu_ps(C + ldc * 5)));
  _mm512_storeu_ps(C + ldc * 6, _mm512_add_ps(c06, _mm512_loadu_ps(C + ldc * 6)));
  _mm512_storeu_ps(C + ldc * 7, _mm512_add_ps(c07, _mm512_loadu_ps(C + ldc * 7)));
  _mm512_storeu_ps(C + 16, _mm512_add_ps(c10, _mm512_loadu_ps(C + 16)));
  _mm512_storeu_ps(C + 16 + ldc, _mm512_add_ps(c11, _mm512_loadu_ps(C + 16 + ldc)));
  _mm512_storeu_ps(C + 16 + ldc * 2, _mm512_add_ps(c12, _mm512_loadu_ps(C + 16 + ldc * 2)));
  _mm512_storeu_ps(C + 16 + ldc * 3, _mm512_add_ps(c13, _mm512_loadu_ps(C + 16 + ldc * 3)));
  _mm512_storeu_ps(C + 16 + ldc * 4, _mm512_add_ps(c14, _mm512_loadu_ps(C + 16 + ldc * 4)));
  _mm512_storeu_ps(C + 16 + ldc * 5, _mm512_add_ps(c15, _mm512_loadu_ps(C + 16 + ldc * 5)));
  _mm512_storeu_ps(C + 16 + ldc * 6, _mm512_add_ps(c16, _mm512_loadu_ps(C + 16 + ldc * 6)));
  _mm512_storeu_ps(C + 16 + ldc * 7, _mm512_add_ps(c17, _mm512_loadu_ps(C + 16 + ldc * 7)));
}

static void block_sgemm_kernel_32xkx4_avx512(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, int K, int ldc, __mmask16 mask) {
  float *ptr_a, *ptr_b;
  ptr_a = A;
  ptr_b = B;
  c00 = _mm512_setzero_ps(); c01 = _mm512_setzero_ps();
  c02 = _mm512_setzero_ps(); c03 = _mm512_setzero_ps();
  c10 = _mm512_setzero_ps(); c11 = _mm512_setzero_ps();
  c12 = _mm512_setzero_ps(); c13 = _mm512_setzero_ps();
  for (int k = 0; k < K; k++) {
    KERNEL_K1_32x4_avx512
  }
  _mm512_storeu_ps(C, _mm512_add_ps(c00, _mm512_loadu_ps(C)));
  _mm512_storeu_ps(C + ldc, _mm512_add_ps(c01, _mm512_loadu_ps(C + ldc)));
  _mm512_storeu_ps(C + ldc * 2, _mm512_add_ps(c02, _mm512_loadu_ps(C + ldc * 2)));
  _mm512_storeu_ps(C + ldc * 3, _mm512_add_ps(c03, _mm512_loadu_ps(C + ldc * 3)));
  if (mask != 0xFFFF) {
    _mm512_mask_storeu_ps(C + 16, mask, _mm512_add_ps(c10, _mm512_loadu_ps(C + 16)));
    _mm512_mask_storeu_ps(C + 16 + ldc, mask, _mm512_add_ps(c11, _mm512_loadu_ps(C + 16 + ldc)));
    _mm512_mask_storeu_ps(C + 16 + ldc * 2, mask, _mm512_add_ps(c12, _mm512_loadu_ps(C + 16 + ldc * 2)));
    _mm512_mask_storeu_ps(C + 16 + ldc * 3, mask, _mm512_add_ps(c13, _mm512_loadu_ps(C + 16 + ldc * 3)));
  } else {
    _mm512_storeu_ps(C + 16, _mm512_add_ps(c10, _mm512_loadu_ps(C + 16)));
    _mm512_storeu_ps(C + 16 + ldc, _mm512_add_ps(c11, _mm512_loadu_ps(C + 16 + ldc)));
    _mm512_storeu_ps(C + 16 + ldc * 2, _mm512_add_ps(c12, _mm512_loadu_ps(C + 16 + ldc * 2)));
    _mm512_storeu_ps(C + 16 + ldc * 3, _mm512_add_ps(c13, _mm512_loadu_ps(C + 16 + ldc * 3)));
  }
}

static void kernel_n_1(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, int M, int K, int ldc) {
  int m, m_sub;
  int k;
  float *ptr_packing_a,*ptr_packing_b;
  for (m_sub = M, m = 0; m_sub > MICRO_M_BLOCK_SIZE - 1; m_sub -= MICRO_M_BLOCK_SIZE, m += MICRO_M_BLOCK_SIZE) {
    ptr_packing_a = A + m * K;
    ptr_packing_b = B;
    
    if (MICRO_M_BLOCK_SIZE == 16) {
      c00 = _mm512_setzero_ps();
      for (k = 0; k < K; k++) {
        a0 = _mm512_load_ps(ptr_packing_a);
        b0 = _mm512_set1_ps(*ptr_packing_b);
        c00 = _mm512_fmadd_ps(a0, b0, c00);
        ptr_packing_a += 16;
        ptr_packing_b++;
      }
      _mm512_storeu_ps(C + m, _mm512_add_ps(c00, _mm512_loadu_ps(C + m)));
    } else if (MICRO_M_BLOCK_SIZE == 32) {
      c00 = _mm512_setzero_ps(); c10 = _mm512_setzero_ps();
      for (k = 0; k < K; k++) {
        a0 = _mm512_load_ps(ptr_packing_a);
        a1 = _mm512_load_ps(ptr_packing_a + 16);
        b0 = _mm512_set1_ps(*ptr_packing_b);
        c00 = _mm512_fmadd_ps(a0, b0, c00);
        c10 = _mm512_fmadd_ps(a1, b0, c10);
        ptr_packing_a += 32;
        ptr_packing_b++;
      }
      _mm512_storeu_ps(C + m, _mm512_add_ps(c00, _mm512_loadu_ps(C + m)));
      _mm512_storeu_ps(C + m + 16, _mm512_add_ps(c10, _mm512_loadu_ps(C + m + 16)));
    } else if (MICRO_M_BLOCK_SIZE == 64) {
      c00 = _mm512_setzero_ps(); c10 = _mm512_setzero_ps();
      c20 = _mm512_setzero_ps(); c30 = _mm512_setzero_ps();
      for (k = 0; k < K; k++) {
        a0 = _mm512_load_ps(ptr_packing_a);
        a1 = _mm512_load_ps(ptr_packing_a + 16);
        a2 = _mm512_load_ps(ptr_packing_a + 32);
        a3 = _mm512_load_ps(ptr_packing_a + 48);
        b0 = _mm512_set1_ps(*ptr_packing_b);
        c00 = _mm512_fmadd_ps(a0, b0, c00);
        c10 = _mm512_fmadd_ps(a1, b0, c10);
        c20 = _mm512_fmadd_ps(a2, b0, c20);
        c30 = _mm512_fmadd_ps(a3, b0, c30);
        ptr_packing_a += 64;
        ptr_packing_b++;
      }
      _mm512_storeu_ps(C + m, _mm512_add_ps(c00, _mm512_loadu_ps(C + m)));
      _mm512_storeu_ps(C + m + 16, _mm512_add_ps(c10, _mm512_loadu_ps(C + m + 16)));
      _mm512_storeu_ps(C + m + 32, _mm512_add_ps(c20, _mm512_loadu_ps(C + m + 32)));
      _mm512_storeu_ps(C + m + 48, _mm512_add_ps(c30, _mm512_loadu_ps(C + m + 48)));
    }
  }
  
  for (; m_sub > 0; m_sub--, m++) {
    sc0 = 0;
    ptr_packing_a = A + m * K;
    ptr_packing_b = B;
    for (k = 0; k < K; k++) {
      sa = *ptr_packing_a;
      sb0 = *ptr_packing_b;
      sc0 += sa * sb0;
      ptr_packing_a++;
      ptr_packing_b++;
    }
    C[m] += sc0;
  }
}

static void kernel_n_4(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, int M, int K, int ldc, int M_padding) {
  int m, m_sub;
  int k;
  float *ptr_packing_a, *ptr_packing_b;

  for (m_sub = M, m = 0; m_sub > 63; m_sub -= 64, m += 64) {
    block_sgemm_kernel_64xkx4_avx512(A + m * K, B, C + m, K, ldc, 0xFFFF);
  }

  // padding sgemm
  if (M_padding > 0 && m_sub + M_padding == 64) {
    __mmask16 mask = M_padding >= 16 ? 0x0000 : (0xFFFF >> M_padding);
    block_sgemm_kernel_64xkx4_avx512(A + m * K, B, C + m, K, ldc, mask);
    return;
  }

  for (; m_sub > 31; m_sub -= 32, m += 32) {
    block_sgemm_kernel_32xkx4_avx512(A + m * K, B, C + m, K, ldc, 0xFFFF);
  }

  // padding sgemm
  if (M_padding > 0 && m_sub + M_padding == 32) {
    __mmask16 mask = M_padding >= 16 ? 0x0000 : (0xFFFF >> M_padding);
    block_sgemm_kernel_32xkx4_avx512(A + m * K, B, C + m, K, ldc, mask);
    return;
  }

  for (; m_sub > 0; m_sub--, m++) {
    ptr_packing_a = A + m * K;
    ptr_packing_b = B;
    sc0 = sc1 = sc2 = sc3 = 0.;
    for (k = 0; k < K; k++) {
      sa = *ptr_packing_a;
      sb0 = *ptr_packing_b; sb1 = *(ptr_packing_b + 1); sb2 = *(ptr_packing_b + 2); sb3 = *(ptr_packing_b + 3);
      sc0 += sa * sb0; sc1 += sa * sb1; sc2 += sa * sb2; sc3 += sa * sb3;
      ptr_packing_a++; ptr_packing_b += 4;
    }
    C[m] += sc0; C[m + ldc] += sc1; C[m + ldc * 2] += sc2; C[m + ldc * 3] += sc3;
  }
}

static void kernel_n_8(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, int M, int K, int ldc, int M_padding) {
  int m, m_sub, k;
  float *ptr_packing_a, *ptr_packing_b;

  // block sgemm
  for (m_sub = M, m = 0; m_sub > MICRO_M_BLOCK_SIZE - 1; m_sub -= MICRO_M_BLOCK_SIZE, m += MICRO_M_BLOCK_SIZE) {
    block_sgemm_kernel_32xkx8_avx512(A + m * K, B, C + m, K, ldc, 0xFFFF);
  }

  // padding sgemm
  if (M_padding > 0 && m_sub + M_padding == MICRO_M_BLOCK_SIZE) {
    __mmask16 mask = M_padding >= 16 ? 0x0000 : (0xFFFF >> M_padding);
    block_sgemm_kernel_32xkx8_avx512(A + m * K, B, C + m, K, ldc, mask);
    return;
  }

  // edge case
  for (; m_sub > 0; m_sub--, m++) {
    ptr_packing_a = A + m * K;
    ptr_packing_b = B;
    sc0 = sc1 = sc2 = sc3 = sc4 = sc5 = sc6 = sc7 = 0.;
    for (k = 0; k < K; k++) {
      sa = *ptr_packing_a;
      sb0 = *ptr_packing_b; sb1 = *(ptr_packing_b + 1); sb2 = *(ptr_packing_b + 2); sb3 = *(ptr_packing_b + 3);
      sb4 = *(ptr_packing_b + 4); sb5 = *(ptr_packing_b + 5); sb6 = *(ptr_packing_b + 6); sb7 = *(ptr_packing_b + 7);
      sc0 += sa * sb0; sc1 += sa * sb1; sc2 += sa * sb2; sc3 += sa * sb3;
      sc4 += sa * sb4; sc5 += sa * sb5; sc6 += sa * sb6; sc7 += sa * sb7;
      ptr_packing_a++; ptr_packing_b += 8;
    }
    C[m] += sc0; C[m + ldc] += sc1; C[m + ldc * 2] += sc2; C[m + ldc * 3] += sc3;
    C[m + ldc * 4] += sc4; C[m + ldc * 5] += sc5; C[m + ldc * 6] += sc6; C[m + ldc * 7] += sc7;
  }
}

static void kernel_n_8_nopacking(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, int M, int K, int lda, int ldb, int ldc) {
  int m, m_sub;
  int k;
  float *ptr_packing_a, *ptr_packing_b;

  // block sgemm
  for (m = 0, m_sub = M; m_sub > 31; m_sub -= 32, m += 32) {
    block_sgemm_kernel_nopacking_32xkx8_avx512(A + m, B, C + m, M, K, lda, ldb, ldc);
  }

  // edge case
  for (; m_sub > 0; m_sub--, m++) {
    ptr_packing_a = A + m;
    ptr_packing_b = B;
    sc0 = sc1 = sc2 = sc3 = sc4 = sc5 = sc6 = sc7 = 0.;
    for (k = 0; k < K; k++) {
      sa = *ptr_packing_a;
      sb0 = *ptr_packing_b; sb1 = *(ptr_packing_b + ldb); sb2 = *(ptr_packing_b + 2 * ldb); sb3 = *(ptr_packing_b + 3 * ldb);
      sb4 = *(ptr_packing_b + 4 * ldb); sb5 = *(ptr_packing_b + 5 * ldb); sb6 = *(ptr_packing_b + 6 * ldb); sb7 = *(ptr_packing_b + 7 * ldb);
      sc0 += sa * sb0; sc1 += sa * sb1; sc2 += sa * sb2; sc3 += sa * sb3;
      sc4 += sa * sb4; sc5 += sa * sb5; sc6 += sa * sb6; sc7 += sa * sb7;
      ptr_packing_a += lda; ptr_packing_b++;
    }
    C[m] += sc0; C[m + ldc] += sc1; C[m + ldc * 2] += sc2; C[m + ldc * 3] += sc3;
    C[m + ldc * 4] += sc4; C[m + ldc * 5] += sc5; C[m + ldc * 6] += sc6; C[m + ldc * 7] += sc7;
  }
}

static void macro_kernel_small(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, int M, int K, int N, int ldc, int M_padding) {
  int n, n_sub;
  MICRO_M_BLOCK_SIZE = 32, MICRO_N_BLOCK_SIZE = 8;
  for (n = 0, n_sub = N; n_sub > MICRO_N_BLOCK_SIZE - 1; n_sub -= MICRO_N_BLOCK_SIZE, n += MICRO_N_BLOCK_SIZE) {
    kernel_n_8(A, B + n * K, C + n * ldc, M, K, ldc, M_padding);
  }
  for (; n_sub > 0; n_sub--, n++) {
    kernel_n_1(A, B + n * K, C + n * ldc, M, K, ldc);
  }
}

static void macro_kernel_large(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, int M, int K, int N, int ldc, int M_padding) {
  int n, n_sub;
  MICRO_M_BLOCK_SIZE = 64, MICRO_N_BLOCK_SIZE = 4;
  for (n = 0, n_sub = N; n_sub > 3; n_sub -= 4, n += 4) {
    kernel_n_4(A, B + n * K, C + n * ldc, M, K, ldc, M_padding);
  }
  for (; n_sub > 0; n_sub--, n++) {
    kernel_n_1(A, B + n * K, C + n * ldc, M, K, ldc);
  }
}

static void macro_kernel_nopacking(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, int M, int K, int N, int lda, int ldb, int ldc) {
  int n, n_sub;
  for (n = 0, n_sub = N; n_sub > 7; n_sub -= 8, n += 8) {
    kernel_n_8_nopacking(A, B + n * ldb, C + n * ldc, M, K, lda, ldb, ldc);
  }
}

/*
 * parallel functions
 * 
 */

static inline void partition_m_dim(const int ithr, const int nthrs, const int m, int *offset, int *length)
{
  assert(m % 32 == 0);
  int base = (m / nthrs) / 64;
  int remain = m - base * nthrs * 64;
  int addi_thrs = remain / 64;
  remain = remain - addi_thrs * 64;
  if (ithr < addi_thrs) {
    *length = (base + 1) * 64;
    *offset = (base + 1) * 64 * ithr;
  } else if (ithr == addi_thrs) {
    *length = base * 64 + remain;
    *offset = (base + 1) * 64 * ithr;
  } else {
    *length = base * 64;
    *offset = (base + 1) * 64 * addi_thrs + remain + base * 64 * (ithr - addi_thrs);
  }
}

static inline void partition_n_dim(const int ithr, const int nthrs, const int m, int *offset, int *length, int size)
{
  assert(m % size == 0);
  int base = (m / nthrs) / size;
  int addi_thrs = m / size;
  if (ithr < addi_thrs) {
    *length = (base + 1) * size;
    *offset = (base + 1) * size * ithr;
  } else {
    *length = base * size;
    *offset = (base + 1) * size * addi_thrs + base * size * (ithr - addi_thrs);
  }
}

static inline int div_up(int a, int b){
    return ((a + b - 1) / b);
}

static inline int rnd_up(int a, int b) {
    return (div_up(a, b) * b);
}


static inline int get_n_padd(int n, int un, int bn)
{
    return rnd_up(min(max(n, un), bn), un);
}

static inline int get_n_padd_parallel_a(int n, int nthr, int N_BLOCKING)
{
    int n_padd = get_n_padd(n, 8, N_BLOCKING);
    return n_padd;
}

void custom_sgemm(int M, int K, int N, float* A, float* B, float* C, float alpha) {
  float *b_buffer_global = NULL;

  #pragma omp parallel
  {
    int m, n, k, M0, N0, K0, M_padding = 0;
    int lda = M, ldb = K, ldc = M;
    int M_BLOCK_SIZE, N_BLOCK_SIZE, K_BLOCK_SIZE, ALIGNMENT_SIZE;
    M_BLOCK_SIZE = 512, N_BLOCK_SIZE = 2048, K_BLOCK_SIZE = 512, ALIGNMENT_SIZE = 512;

    int nthr = omp_get_num_threads();
    int ithr = omp_get_thread_num();
    
    int m_offset = 0, m_length = 0;
    partition_m_dim(ithr, nthr, M, &m_offset, &m_length);

    float *a_buffer_local = NULL;
    float *b_buffer_local = NULL;
    if (ithr == 0) {
      b_buffer_global = (float *) aligned_alloc(ALIGNMENT_SIZE, K_BLOCK_SIZE * N_BLOCK_SIZE * sizeof(float));
    }
    #pragma omp barrier
    b_buffer_local = b_buffer_global;

    if (lda < 64) { // small matrix: 32xkx8, packing
      for (k = 0; k < K; k += K_BLOCK_SIZE) {
        K0 = min(K_BLOCK_SIZE, K - k);
        for (n = 0; n < N; n += N_BLOCK_SIZE) {
          N0 = min(N_BLOCK_SIZE, N - n);
          if (ithr == 0) {
            packing_b_8(B + k + n * ldb, b_buffer_local, ldb, K0, N0);
          }
          #pragma omp barrier
          if (m_length > 0 && !a_buffer_local) {
            a_buffer_local = (float *) aligned_alloc(ALIGNMENT_SIZE, K_BLOCK_SIZE * M_BLOCK_SIZE * sizeof(float));
          }
          for (m = 0; m < m_length; m += M_BLOCK_SIZE) {
            M0 = min(M_BLOCK_SIZE, m_length - m);
            M_padding = M0 % 32 > 16 ? 32 - M0 % 32 : 0;
            packing_a_32(alpha, A + (m_offset + m) + k * lda, a_buffer_local, lda, M0, K0, M_padding); // pad to 32
            macro_kernel_small(a_buffer_local, b_buffer_local, C + (m_offset + m) + n * ldc, M0, K0, N0, ldc, M_padding);
          }
          #pragma omp barrier
        }
      }
    } else { // large matrix: 64xkx4 and 32xkx4, packing
      for (k = 0; k < K; k += K_BLOCK_SIZE) {
        K0 = min(K_BLOCK_SIZE, K - k);
        for (n = 0; n < N; n += N_BLOCK_SIZE) {
          N0 = min(N_BLOCK_SIZE, N - n);
          if (ithr == 0) {
            packing_b_4(B + k + n * ldb, b_buffer_local, ldb, K0, N0);
          }
          #pragma omp barrier
          if (m_length > 0 && !a_buffer_local) {
            a_buffer_local = (float *) aligned_alloc(ALIGNMENT_SIZE, K_BLOCK_SIZE * M_BLOCK_SIZE * sizeof(float));
          }
          for (m = 0; m < m_length; m += M_BLOCK_SIZE) {
            M0 = min(M_BLOCK_SIZE, m_length - m);
            M_padding = M0 % 32 > 16 ? 32 - M0 % 32 : 0;
            packing_a_64(alpha, A + (m_offset + m) + k * lda, a_buffer_local, lda, M0, K0, M_padding); // pad to 64 or 32
            macro_kernel_large(a_buffer_local, b_buffer_local, C + (m_offset + m) + n * ldc, M0, K0, N0, ldc, M_padding);
          }
          #pragma omp barrier
        }
      }
    }
    if (a_buffer_local) { free(a_buffer_local); }
  }
  if (b_buffer_global) { free(b_buffer_global); }
}

void reference_sgemm(int M, int K, int N, float* A, float* B, float* C, int TRANSA, int TRANSB, int TRANSC) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) 
    {
      float cij = TRANSC ? C[i+j*M] : C[i*N+j]; 
      for( int k = 0; k < K; k++ ) {
        cij += (TRANSA ? A[i+k*M] : A[i*K+k]) * (TRANSB ? B[k+j*K] : B[k*N+j]);
      }
      if (TRANSC)
        C[i+j*M] = cij;
      else
        C[i*N+j] = cij;
    }
}