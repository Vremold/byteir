/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fastertransformerv4 {
enum class OperationType { FP32, HALF };

template <OperationType OpType> class Traits;

template <> class Traits<OperationType::FP32> {
public:
  typedef float DataType;
  // cuBLAS parameters
  static cudaDataType_t const computeType = CUDA_R_32F;
  static cudaDataType_t const AType = CUDA_R_32F;
  static cudaDataType_t const BType = CUDA_R_32F;
  static cudaDataType_t const CType = CUDA_R_32F;
  static const int algo = -1;
};

template <> class Traits<OperationType::HALF> {
public:
  typedef __half DataType;
  // cuBLAS parameters
  static cudaDataType_t const computeType = CUDA_R_16F;
  static cudaDataType_t const AType = CUDA_R_16F;
  static cudaDataType_t const BType = CUDA_R_16F;
  static cudaDataType_t const CType = CUDA_R_16F;
  static const int algo = 99;
};

enum ActType { Relu, Sigmoid, SoftPlus, No };

template <ActType act, typename T> __inline__ __device__ T act_fun(T val) {
  if (act == ActType::Relu)
    return (val <= (T)0.0f) ? (T)0.0f : val;
  else if (act == ActType::SoftPlus)
    return logf(__expf((float)val) + 1.0f);
  else if (act == ActType::Sigmoid)
    return 1.0f / (1.0f + __expf(-1.0f * (float)val));
  else
    return val;
}

enum transposeType { TRANSPOSE0213, TRANSPOSE1203, TRANSPOSE2013 };

template <transposeType transpose_type>
__inline__ __device__ int transpose3d(const int dim0, const int dim1,
                                      const int dim2, int d0, int d1, int d2) {
  if (transpose_type == TRANSPOSE0213)
    return (d0 * dim2 + d2) * dim1 + d1;
  else if (transpose_type == TRANSPOSE1203)
    return (d1 * dim2 + d2) * dim0 + d0;
  else if (transpose_type == TRANSPOSE2013)
    return (d2 * dim0 + d0) * dim1 + d1;
  else
    return (d0 * dim2 + d2) * dim1 + d1;
}

#define PRINT_FUNC_NAME_()                                                     \
  do {                                                                         \
    std::cout << "[FT][CALL] " << __FUNCTION__ << " " << std::endl;            \
  } while (0)

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorString(error);
}

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";

  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";

  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

template <typename T>
void check(T result, char const *const, const char *const file,
           int const line) {
  if (result)
    throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") +
                             (_cudaGetErrorEnum(result)) + " " + file + ":" +
                             std::to_string(line) + " \n");
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)
} // namespace fastertransformerv4
