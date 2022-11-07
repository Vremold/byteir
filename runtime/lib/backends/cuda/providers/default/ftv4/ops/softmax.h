#pragma once

#include "./common.h"

#include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"
#include "fastertransformer_v4/includes/softmax.h"

namespace brt {
namespace cuda {
namespace ftv4 {

template <OperationType OpType, typename T> class SoftmaxForwardImpl {
public:
  SoftmaxForwardImpl(const OpAccessor &accessor) {
    auto shape = accessor.GetArgShape(0);
    param_.rows = accessor.GetNumElementsOfShape(shape) / shape.back();
    param_.cols = shape.back();

    param_.head_num = accessor.GetAttrAsInt("head_num");
    param_.dropout_rate = accessor.GetAttrAsFloat("dropout_rate");
    param_.batch_first = accessor.GetAttrAsBool("batch_first");

    param_.add_mask = true;

    param_.apply_dropout = true;
    if (param_.dropout_rate < 1e-3f)
      param_.apply_dropout = false;
  }

  void Execute(const T *input, T *mask, T *softmax_output,
               T *softmax_dropout_output, uint8_t *dropout_mask,
               cudaStream_t stream) {
    fastertransformerv4::SoftmaxForwardParam<T> param{
        .input = input,
        .softmax_output = softmax_output,
        .buf = nullptr,
        .rows = param_.rows,
        .cols = param_.cols,
        .stream = stream,
        .add_mask = param_.add_mask,
        .mask = mask,
        .head_num = param_.head_num,
        .apply_dropout = param_.apply_dropout,
        .dropout_rate = param_.dropout_rate,
        .dropout_mask = dropout_mask,
        .softmax_dropout_output = softmax_dropout_output,
        .batch_first = param_.batch_first};
    fastertransformerv4::Softmax<OpType>::forward(param);
  }

private:
  struct ParametersPack {
    int rows;
    int cols;
    int head_num;
    float dropout_rate;
    bool batch_first;
    bool add_mask;
    bool apply_dropout;
  } param_;
};

template <OperationType OpType, typename T> class SoftmaxBackwardImpl {
public:
  SoftmaxBackwardImpl(const OpAccessor &accessor) {
    auto shape = accessor.GetArgShape(0);
    param_.rows = accessor.GetNumElementsOfShape(shape) / shape.back();
    param_.cols = shape.back();

    param_.dropout_rate = accessor.GetAttrAsFloat("dropout_rate");

    param_.apply_dropout = true;
    if (param_.dropout_rate < 1e-3f)
      param_.apply_dropout = false;
  }

  void Execute(const T *grad_out, const T *out, T *grad_in,
               uint8_t *dropout_mask, cudaStream_t stream) {
    fastertransformerv4::SoftmaxBackwardParam<T> param{
        .grad_out = grad_out,
        .out = out,
        .grad_in = grad_in,
        .buf = nullptr,
        .rows = param_.rows,
        .cols = param_.cols,
        .stream = stream,
        .apply_dropout = param_.apply_dropout,
        .dropout_rate = param_.dropout_rate,
        .dropout_mask = dropout_mask};
    fastertransformerv4::Softmax<OpType>::backward(param);
  }

private:
  struct ParametersPack {
    int rows;
    int cols;
    bool apply_dropout;
    float dropout_rate;
  } param_;
};

template <OperationType OpType,
          typename T = typename fastertransformerv4::Traits<OpType>::DataType>
using SoftmaxForward =
    CudaOpKernel<SoftmaxForwardImpl<OpType, T>, //
                 TypedOperand<const T *, 0>,    // input
                 TypedOperand<T *, 1>,          // mask
                 TypedOperand<T *, 2>,          // softmax_output
                 TypedOperand<T *, 3>,          // softmax_dropout_output
                 TypedOperand<uint8_t *, 4>     // dropout_mask
                 >;

template <OperationType OpType,
          typename T = typename fastertransformerv4::Traits<OpType>::DataType>
using SoftmaxBackward = CudaOpKernel<SoftmaxBackwardImpl<OpType, T>, //
                                     TypedOperand<const T *, 0>,     // grad_out
                                     TypedOperand<const T *, 1>,     // out
                                     TypedOperand<T *, 3>,           // grad_in
                                     TypedOperand<uint8_t *, 2> // dropout_mask
                                     >;
} // namespace ftv4
} // namespace cuda
} // namespace brt