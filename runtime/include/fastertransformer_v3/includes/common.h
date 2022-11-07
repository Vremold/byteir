/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
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

namespace fastertransformerv3 {
template <typename T>
void print_vec(const T *data, const char *str, const int size) {
  printf("print %s\n", str);
  T *tmp = (T *)malloc(sizeof(T) * size);
  cudaMemcpy(tmp, data, sizeof(T) * size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < size; ++i)
    printf("%d %f\n", i, (float)tmp[i]);
  free(tmp);
  printf("\n");
}

template <typename T> void result_check(const T *ptr, char *file, int size) {
  T *h_ptr = (T *)malloc(sizeof(T) * size);
  cudaMemcpy(h_ptr, ptr, sizeof(T) * size, cudaMemcpyDeviceToHost);

  float max_diff = -1e20f;
  FILE *fd = fopen(file, "r");
  if (fd == NULL) {
    printf("FILE %s does not exist.\n", file);
    exit(0);
    return;
  }
  for (int i = 0; i < size; ++i) {
    float tmp;
    fscanf(fd, "%f", &tmp);
    float diff = fabs(tmp - (float)h_ptr[i]);
    if (diff > max_diff) {
      max_diff = diff;
      printf("%d CUDA %f TF %f diff %f\n", i, (float)h_ptr[i], (float)tmp,
             (float)diff);
    }
  }
  free(h_ptr);
  fclose(fd);

  printf("\033[0;32m");
  printf("Check with %s max_diff %f\n", file, max_diff);
  printf("\033[0m");
}

enum class OperationType { FP32, HALF };
enum ActType { Relu, Sigmoid, SoftPlus, No };

template <OperationType OpType_> class Traits;

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
  if (result) {
    throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") +
                             (_cudaGetErrorEnum(result)) + " " + file + ":" +
                             std::to_string(line) + " \n");
  }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

} // namespace fastertransformerv3
