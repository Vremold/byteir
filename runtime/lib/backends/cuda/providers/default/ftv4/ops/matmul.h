#pragma once

#include "./common.h"

#include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"
#include "fastertransformer_v4/includes/matmul.h"

namespace brt {
namespace cuda {
namespace ftv4 {
template <OperationType OpType, typename T> class MatMulImplBase {
  using Kernel = fastertransformerv4::MatMul<OpType>;

public:
  MatMulImplBase(const OpAccessor &accessor, size_t A_shape_provider_idx,
                 size_t B_shape_provider_idx)
      : kernel_(std::make_unique<Kernel>()) {
    auto A_shape = accessor.GetArgShape(A_shape_provider_idx);
    auto B_shape = accessor.GetArgShape(B_shape_provider_idx);
    auto A_ndim = A_shape.size(), B_ndim = B_shape.size();

    A_T = accessor.GetAttrAsBool("transpose_a");
    B_T = accessor.GetAttrAsBool("transpose_b");

    batch_count = 1;
    for (size_t i = 0; i < A_ndim - 2; ++i)
      batch_count *= A_shape[i];
    M = A_shape[A_ndim - 2];
    K = A_shape[A_ndim - 1];
    if (A_T) {
      std::swap(M, K);
    }
    N = B_T ? B_shape[B_ndim - 2] : B_shape[B_ndim - 1];

    scale = accessor.GetAttrAsFloat("scale");
  }

protected:
  int M, K, N;
  int batch_count;
  float scale;
  bool A_T, B_T;
  std::unique_ptr<Kernel> kernel_;
};

template <OperationType OpType, typename T>
class MatMulForwardImpl : public MatMulImplBase<OpType, T> {
  using ForwardParam = fastertransformerv4::MatMulForwardParam<T>;

public:
  MatMulForwardImpl(const OpAccessor &accessor)
      : MatMulImplBase<OpType, T>(accessor, 0, 1) {}

  void Execute(const T *A, const T *B, T *C, cublasHandle_t handle,
               cudaStream_t stream) {
    ForwardParam param{.input_A = A,
                       .input_B = B,
                       .output = C,
                       .M = this->M,
                       .K = this->K,
                       .N = this->N,
                       .batch_count = this->batch_count,
                       .scale = this->scale,
                       .A_T = this->A_T,
                       .B_T = this->B_T,
                       .cublas_handle = handle,
                       .stream = stream};
    this->kernel_->forward(param);
  }
};

template <OperationType OpType, typename T>
class MatMulBackwardImpl : public MatMulImplBase<OpType, T> {
  using BackwardParam = fastertransformerv4::MatMulBackwardParam<T>;

public:
  MatMulBackwardImpl(const OpAccessor &accessor)
      : MatMulImplBase<OpType, T>(accessor, 1, 2) {}

  void Execute(const T *grad_C, const T *A, const T *B, T *grad_A, T *grad_B,
               cublasHandle_t handle, cudaStream_t stream) {
    BackwardParam param{.grad_out = grad_C,
                        .input_A = A,
                        .input_B = B,
                        .grad_A = grad_A,
                        .grad_B = grad_B,
                        .M = this->M,
                        .K = this->K,
                        .N = this->N,
                        .batch_count = this->batch_count,
                        .scale = this->scale,
                        .A_T = this->A_T,
                        .B_T = this->B_T,
                        .cublas_handle = handle,
                        .stream = stream};
    this->kernel_->backward(param);
  }
};

template <OperationType OpType,
          typename T = typename fastertransformerv4::Traits<OpType>::DataType>
using MatMulForward = CublasOpKernel<MatMulForwardImpl<OpType, T>,
                                     TypedOperand<const T *, 0>, // A
                                     TypedOperand<const T *, 1>, // B
                                     TypedOperand<T *, 2>        // C
                                     >;

template <OperationType OpType,
          typename T = typename fastertransformerv4::Traits<OpType>::DataType>
using MatMulBackward = CublasOpKernel<MatMulBackwardImpl<OpType, T>,
                                      TypedOperand<const T *, 0>, // grad_C
                                      TypedOperand<const T *, 1>, // A
                                      TypedOperand<const T *, 2>, // B
                                      TypedOperand<T *, 3>,       // grad_A
                                      TypedOperand<T *, 4>        // grad_B
                                      >;

} // namespace ftv4
} // namespace cuda
} // namespace brt
