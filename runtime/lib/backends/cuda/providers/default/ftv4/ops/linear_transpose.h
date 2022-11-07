#pragma once

#include "./common.h"

#include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"
#include "fastertransformer_v4/includes/linear_transpose.h"

namespace brt {
namespace cuda {
namespace ftv4 {
template <OperationType OpType, typename T> class LinearTransposeImplBase {
  using Kernel = fastertransformerv4::LinearTranspose<OpType>;
  using Param = fastertransformerv4::LinearTransposeParam<OpType>;

public:
  LinearTransposeImplBase(const OpAccessor &accessor,
                          size_t tensor_shape_provider_idx,
                          size_t weight_shape_provider_idx) {
    auto tensor_shape = accessor.GetArgShape(tensor_shape_provider_idx);
    auto weight_shape = accessor.GetArgShape(weight_shape_provider_idx);
    int batch_size = tensor_shape[0];
    int seq_len = tensor_shape[1];

    int from_hidden_dim = weight_shape[1];
    int to_hidden_dim = weight_shape[0];
    bool transposed_weight = true; // follow the pytorch extension

    int head_num = accessor.GetAttrAsInt("head_num");
    kernel_ =
        std::make_unique<Kernel>(batch_size, seq_len, from_hidden_dim,
                                 to_hidden_dim, head_num, transposed_weight);
    batch_size_ = batch_size;
    transpose_type_ = ConvertTransposeType(
        accessor.GetAttrAsString("forward_transpose_type"));
  }

  void Initialize(const T *weight, const T *bias) {
    Param param;
    param.weight = weight;
    param.bias = bias;
    kernel_->initialize(param);
  }

protected:
  int batch_size_;
  TransposeType transpose_type_;
  std::unique_ptr<Kernel> kernel_;
};

template <OperationType OpType, typename T>
class LinearTransposeForwardImpl : public LinearTransposeImplBase<OpType, T> {
  using ForwardParam = fastertransformerv4::LinearTransposeForwardParam<T>;
  using Base = LinearTransposeImplBase<OpType, T>;

public:
  LinearTransposeForwardImpl(const OpAccessor &accessor)
      : Base(accessor, 0, 1) {}

  void Execute(const T *input, const T *weight, const T *bias, T *output,
               void *buf, cublasHandle_t handle, cudaStream_t stream) {
    Base::Initialize(weight, bias);
    ForwardParam param{.input = input,
                       .output = output,
                       .buf = buf,
                       .batch_size = this->batch_size_,
                       .cublas_handle = handle,
                       .stream = stream,
                       .transpose_type = this->transpose_type_};
    this->kernel_->forward(param);
  }

  size_t GetWorkspaceSize(const ExecutionContext & /*ctx*/) const {
    return this->kernel_->cal_fw_bufsize();
  }
};

template <OperationType OpType, typename T>
class LinearTransposeBackwardImpl : public LinearTransposeImplBase<OpType, T> {
  using BackwardParam = fastertransformerv4::LinearTransposeBackwardParam<T>;
  using Base = LinearTransposeImplBase<OpType, T>;

public:
  LinearTransposeBackwardImpl(const OpAccessor &accessor)
      : Base(accessor, 1, 2) {}

  void Execute(const T *grad_out, const T *input, const T *weight, T *grad_in,
               T *grad_weight, T *grad_bias, void *buf, cublasHandle_t handle,
               cudaStream_t stream) {
    Base::Initialize(weight, nullptr);
    BackwardParam param{.grad_out = grad_out,
                        .input = input,
                        .grad_in = grad_in,
                        .grad_weight = grad_weight,
                        .grad_bias = grad_bias,
                        .buf = buf,
                        .batch_size = this->batch_size_,
                        .cublas_handle = handle,
                        .stream = stream,
                        .transpose_type = this->transpose_type_};
    this->kernel_->backward(param);
  }

  size_t GetWorkspaceSize(const ExecutionContext & /*ctx*/) const {
    return this->kernel_->cal_bw_bufsize();
  }
};

template <OperationType OpType,
          typename T = typename fastertransformerv4::Traits<OpType>::DataType>
using LinearTransposeForward =
    CublasOpKernelWithWorkspace<LinearTransposeForwardImpl<OpType, T>,
                                TypedOperand<const T *, 0>, // input
                                TypedOperand<const T *, 1>, // weight
                                TypedOperand<const T *, 2>, // bias
                                TypedOperand<T *, 3>        // output
                                >;

template <OperationType OpType,
          typename T = typename fastertransformerv4::Traits<OpType>::DataType>
using LinearTransposeBackward =
    CublasOpKernelWithWorkspace<LinearTransposeBackwardImpl<OpType, T>,
                                TypedOperand<const T *, 0>, // grad_out
                                TypedOperand<const T *, 1>, // input
                                TypedOperand<const T *, 2>, // weight
                                TypedOperand<T *, 3>,       // grad_in
                                TypedOperand<T *, 4>,       // grad_weight
                                TypedOperand<T *, 5>        // grad_bias
                                >;

} // namespace ftv4
} // namespace cuda
} // namespace brt
