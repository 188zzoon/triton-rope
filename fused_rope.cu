/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/fused_rope.h>

#include "../common.h"
#include "../util/logging.h"
#include "../utils.cuh"

namespace transformer_engine {

/**
 * template <typename scalar_t>
 * template parameter
 *  CUDA 코드에서 GPU에서 실행되는 코드를 제네릭하게 만들기 위해서입니다.
 * C++의 템플릿을 사용하여 함수를 정의함으로써 여러 종류의 데이터 타입에 대해 동일한 함수 코드를 사용할 수 있다
*/
template <typename scalar_t>
__device__ void fused_rope_block_forward(
    const scalar_t *src, const float *freqs, scalar_t *dst,
    const int offset_block, const int offset_block_dst, const int h,
    const int d, const int d2, const int stride_h, const int stride_d,
    const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x;
/**
 * #pragma unroll
 * hints to the compiler that loops should be unrolled for efficiency.
 * 컴파일러에게 다음 for 루프를 풀어서 최적화하도록 지시
*/

// 각 스레드는 d2 만큼의 작업을 처리
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    float v_cos, v_sin;
    /**
     * https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html
     * __device__​ void sincosf ( float  x, float* sptr, float* cptr )
     * Calculate the sine and cosine of the first input argument x (measured in radians). 
     * The results for sine and cosine are written into the second argument, sptr, and, respectively, third argument, cptr.
    */
    // freqs 배열에서 해당 위치의 각도값에 대한 sine과 cosine을 계산
    sincosf(freqs[s_id * d2 + d_id], &v_sin, &v_cos);
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      // 입력(src)과 출력(dst)의 오프셋 계산
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      // 현재 위치의 입력 값 및 회전에 사용할 값 계산
      float v_src = src[offset_src];
      /**
       * static_cast<float> 컴파일 타임에서 캐스팅
       * v_ : vector
       * (d_id + d2 / 2 < d2) 이면, 양수로 처리하고 그렇지 않으면 음수로 처리
      */
      float v_src_rotate = (d_id + d2 / 2 < d2)
                                  ? -static_cast<float>(src[offset_src + (d2 / 2) * stride_d])
                                  : static_cast<float>(src[offset_src + (d2 / 2 - d2) * stride_d]);
      dst[offset_dst] =
          v_src * v_cos + v_src_rotate * v_sin;
    }
  }
/**
 * h_id = threadIdx.y : 현재 스레드의 y 인데스
 * h_id += blockDim.y : warp의 크기인 blockDim.y만큼씩 반복
*/
  // copy the rest
  if (d > d2) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
        dst[offset_head_dst + d_id * o_stride_d] =
            src[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename scalar_t>
__device__ void fused_rope_block_backward(
    const scalar_t *src, const float *freqs, scalar_t *dst,
    const int offset_block, const int offset_block_dst, const int h,
    const int d, const int d2, const int stride_h, const int stride_d,
    const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x;
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    float v_cos = cosf(freqs[s_id * d2 + d_id]);
    float v_sin = (d_id + d2 / 2 < d2)
                         ? sinf(freqs[s_id * d2 + d_id + d2 / 2])
                         : -sinf(freqs[s_id * d2 + d_id + d2 / 2 - d2]);
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = src[offset_src];
      float v_src_rotate = (d_id + d2 / 2 < d2)
                                  ? src[offset_src + (d2 / 2) * stride_d]
                                  : src[offset_src + (d2 / 2 - d2) * stride_d];
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // handle the tail
  if (d > d2) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
        dst[offset_head_dst + d_id * o_stride_d] =
            src[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename scalar_t>
__global__ void fused_rope_forward_kernel(
    const scalar_t *src, const float *freqs, scalar_t *dst, const int h,
    const int d, const int d2, const int stride_s, const int stride_b,
    const int stride_h, const int stride_d, const int o_stride_s,
    const int o_stride_b, const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
  fused_rope_block_forward(src, freqs, dst, offset_block, offset_block_dst, h,
                           d, d2, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t>
__global__ void fused_rope_backward_kernel(
    const scalar_t *src, const float *freqs, scalar_t *dst, const int h,
    const int d, const int d2, const int stride_s, const int stride_b,
    const int stride_h, const int stride_d, const int o_stride_s,
    const int o_stride_b, const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
  fused_rope_block_backward(src, freqs, dst, offset_block, offset_block_dst, h,
                            d, d2, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t>
void fused_rope_forward_launcher(const scalar_t *input, const float *freqs,
                                 scalar_t *output, const int s, const int b,
                                 const int h, const int d, const int d2,
                                 const int stride_s, const int stride_b,
                                 const int stride_h, const int stride_d,
                                 const int o_stride_s, const int o_stride_b,
                                 const int o_stride_h, const int o_stride_d,
                                 cudaStream_t stream) {
  /**
   * Warp는 CUDA에서 한 번에 처리되는 스레드 그룹
   * dim3는 CUDA에서 3차원 그리드와 블록 크기를 나타내는 구조체
   * blocks는 그리드의 크기
   * threads는 블록의 크기
   * THREADS_PER_WARP는 warp 당 스레드 수
  **/
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);

  /**
   * cuda kernel<<#blocks, #threads, 동적 공유메모리 크기, cudaStream_t>>호출
  */
  fused_rope_forward_kernel<<<blocks, threads, 0, stream>>>(
      input, freqs, output, h, d, d2, stride_s, stride_b, stride_h, stride_d,
      o_stride_s, o_stride_b, o_stride_h, o_stride_d);
/**
 * CUDA 커널 호출 후 발생한 오류를 확인
 * cudaGetLastError() 함수는 이전 CUDA 호출에서 발생한 오류를 확인
 * NVTE_CHECK_CUDA 매크로는 CUDA 호출의 반환 값을 확인하고 오류가 발생하면 예외를 throw 처리
*/
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void fused_rope_backward_launcher(const scalar_t *output_grads,
                                  const float *freqs, scalar_t *input_grads,
                                  const int s, const int b, const int h,
                                  const int d, const int d2, const int stride_s,
                                  const int stride_b, const int stride_h,
                                  const int stride_d, const int o_stride_s,
                                  const int o_stride_b, const int o_stride_h,
                                  const int o_stride_d, cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);

  fused_rope_backward_kernel<<<blocks, threads, 0, stream>>>(
      output_grads, freqs, input_grads, h, d, d2, stride_s, stride_b, stride_h,
      stride_d, o_stride_s, o_stride_b, o_stride_h, o_stride_d);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

/**
 * TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT 자료형 변경
*/

void fused_rope_forward(const Tensor &input, const Tensor &freqs,
                        Tensor *output, const int s, const int b, const int h,
                        const int d, const int d2, const int stride_s,
                        const int stride_b, const int stride_h,
                        const int stride_d, const int o_stride_s,
                        const int o_stride_b, const int o_stride_h,
                        const int o_stride_d, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, scalar_t,
      fused_rope_forward_launcher(
          reinterpret_cast<const scalar_t *>(input.data.dptr),
          reinterpret_cast<const float *>(freqs.data.dptr),
          reinterpret_cast<scalar_t *>(output->data.dptr), s, b, h, d, d2,
          stride_s, stride_b, stride_h, stride_d, o_stride_s, o_stride_b,
          o_stride_h, o_stride_d, stream););
}

void fused_rope_backward(const Tensor &output_grads, const Tensor &freqs,
                         Tensor *input_grads, const int s, const int b,
                         const int h, const int d, const int d2,
                         const int stride_s, const int stride_b,
                         const int stride_h, const int stride_d,
                         const int o_stride_s, const int o_stride_b,
                         const int o_stride_h, const int o_stride_d,
                         cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      output_grads.data.dtype, scalar_t,
      fused_rope_backward_launcher(
          reinterpret_cast<const scalar_t *>(output_grads.data.dptr),
          reinterpret_cast<const float *>(freqs.data.dptr),
          reinterpret_cast<scalar_t *>(input_grads->data.dptr), s, b, h, d, d2,
          stride_s, stride_b, stride_h, stride_d, o_stride_s, o_stride_b,
          o_stride_h, o_stride_d, stream););
}
}  // end namespace transformer_engine

