#pragma once

#include "./common.h"

#include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"
#include "fastertransformer_v4/includes/linear.h"

namespace brt {
namespace cuda {
namespace ftv4 {
template <OperationType OpType, typename T> class LinearImplBase {
  using Kernel = fastertransformerv4::Linear<OpType>;
  using Param = fastertransformerv4::LinearParam<OpType>;

public:
  LinearImplBase(const OpAccessor &accessor, size_t tensor_shape_provider_idx,
                 size_t weight_shape_provider_idx, bool act_gelu = false,
                 float dropout_rate = 0) {
    auto tensor_shape = accessor.GetArgShape(tensor_shape_provider_idx);
    auto rows =
        accessor.GetNumElementsOfShape(tensor_shape) / tensor_shape.back();
    auto weight_shape = accessor.GetArgShape(weight_shape_provider_idx);

    // transposed_weight is always set to true followed pytorch extension so
    // weight_shape is (N, K)
    bool transposed_weight = true;
    int K = weight_shape[1], N = weight_shape[0];

    kernel_ = std::make_unique<Kernel>(rows, K, N, transposed_weight, act_gelu,
                                       dropout_rate);
    rows_ = rows;
  }

  void Initialize(const T *weight, const T *bias) {
    Param param;
    param.weight = weight;
    param.bias = bias;
    kernel_->initialize(param);
  }

protected:
  int rows_;
  std::unique_ptr<Kernel> kernel_;
};

template <OperationType OpType, typename T>
class LinearGeluDropoutImplBase : public LinearImplBase<OpType, T> {
public:
  LinearGeluDropoutImplBase(const OpAccessor &accessor,
                            size_t tensor_shape_provider_idx,
                            size_t weight_shape_provider_idx)
      : LinearImplBase<OpType, T>(accessor, tensor_shape_provider_idx,
                                  weight_shape_provider_idx,
                                  accessor.GetAttrAsBool("act_gelu"),
                                  accessor.GetAttrAsFloat("dropout_rate")) {}
};

template <OperationType OpType, typename T,
          template <OperationType, typename> class BaseT>
class LinearForwardImpl : public BaseT<OpType, T> {
  using ForwardParam = fastertransformerv4::LinearForwardParam<T>;
  using Base = BaseT<OpType, T>;

public:
  LinearForwardImpl(const OpAccessor &accessor) : Base(accessor, 0, 1) {}

  void Execute(const T *input, const T *weight, const T *bias, T *output,
               T *bias_out, uint8_t *dropout_mask, cublasHandle_t cublas_handle,
               cudaStream_t stream) {
    Base::Initialize(weight, bias);
    ForwardParam param{.input = input,
                       .output = output,
                       .rows = this->rows_,
                       .cublas_handle = cublas_handle,
                       .stream = stream,
                       .bias_out = bias_out,
                       .dropout_mask = dropout_mask};
    this->kernel_->forward(param);
  }
};

template <OperationType OpType, typename T,
          template <OperationType, typename> class BaseT>
class LinearBackwardImpl : public BaseT<OpType, T> {
  using BackwardParam = fastertransformerv4::LinearBackwardParam<T>;
  using Base = BaseT<OpType, T>;

public:
  LinearBackwardImpl(const OpAccessor &accessor) : Base(accessor, 0, 2) {}

  void Execute(const T *grad_out, const T *input, const T *weight, T *grad_in,
               T *grad_weight, T *grad_bias, T *bias_out, uint8_t *dropout_mask,
               void *buf, cublasHandle_t cublas_handle, cudaStream_t stream) {
    Base::Initialize(weight, nullptr);
    BackwardParam param{.grad_out = grad_out,
                        .input = input,
                        .grad_in = grad_in,
                        .grad_weight = grad_weight,
                        .grad_bias = grad_bias,
                        .buf = buf,
                        .rows = this->rows_,
                        .cublas_handle = cublas_handle,
                        .stream = stream,
                        .bias_out = bias_out,
                        .dropout_mask = dropout_mask};
    this->kernel_->backward(param);
  }

  unsigned long long GetWorkspaceSize(const ExecutionContext & /*ctx*/) const {
    return this->kernel_->cal_bw_bufsize();
  }
};

template <OperationType OpType,
          typename T = typename fastertransformerv4::Traits<OpType>::DataType>
using LinearForward =
    CublasOpKernel<LinearForwardImpl<OpType, T, LinearImplBase>,
                   TypedOperand<const T *, 0>, // input
                   TypedOperand<const T *, 1>, // weight
                   TypedOperand<const T *, 2>, // bias
                   TypedOperand<T *, 3>,       // output
                   NoneArg,                    // bias_out
                   NoneArg                     // dropout_mask
                   >;

template <OperationType OpType,
          typename T = typename fastertransformerv4::Traits<OpType>::DataType>
using LinearGeluDropoutForward =
    CublasOpKernel<LinearForwardImpl<OpType, T, LinearGeluDropoutImplBase>,
                   TypedOperand<const T *, 0>,     // input
                   TypedOperand<const T *, 1>,     // weight
                   TypedOperand<const T *, 2>,     // bias
                   TypedOperand<T *, 3>,           // output
                   TypedOperand<T *, 4>,           // bias_out
                   TypedOperand<std::uint8_t *, 5> // dropout_mask
                   >;

template <OperationType OpType,
          typename T = typename fastertransformerv4::Traits<OpType>::DataType>
using LinearBackward =
    CublasOpKernelWithWorkspace<LinearBackwardImpl<OpType, T, LinearImplBase>,
                                TypedOperand<const T *, 0>, // grad_out
                                TypedOperand<const T *, 1>, // input
                                TypedOperand<const T *, 2>, // weight
                                TypedOperand<T *, 3>,       // grad_in
                                TypedOperand<T *, 4>,       // grad_weight
                                TypedOperand<T *, 5>,       // grad_bias
                                NoneArg,                    // bias_out
                                NoneArg                     // dropout_mask
                                >;

template <OperationType OpType,
          typename T = typename fastertransformerv4::Traits<OpType>::DataType>
using LinearGeluDropoutBackward = CublasOpKernelWithWorkspace<
    LinearBackwardImpl<OpType, T, LinearGeluDropoutImplBase>,
    TypedOperand<const T *, 0>,     // grad_out
    TypedOperand<const T *, 1>,     // input
    TypedOperand<const T *, 2>,     // weight
    TypedOperand<T *, 5>,           // grad_in
    TypedOperand<T *, 6>,           // grad_weight
    TypedOperand<T *, 7>,           // grad_bias
    TypedOperand<T *, 3>,           // bias_out
    TypedOperand<std::uint8_t *, 4> // dropout_mask
    >;

} // namespace ftv4
} // namespace cuda
} // namespace brt