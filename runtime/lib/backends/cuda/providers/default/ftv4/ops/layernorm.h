#pragma once

#include "./common.h"

#include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"
#include "fastertransformer_v4/includes/layernorm.h"

namespace brt {
namespace cuda {
namespace ftv4 {
template <OperationType OpType, typename T> class LayerNormImplBase {
  using Kernel = fastertransformerv4::LayerNorm<OpType>;
  using Param = fastertransformerv4::LayerNormParam<OpType>;

public:
  LayerNormImplBase(const OpAccessor &accessor, size_t shape_provider_idx) {
    auto shape = accessor.GetArgShape(shape_provider_idx);
    auto max_rows = accessor.GetNumElementsOfShape(shape) / shape.back();
    auto hidden_dims = shape.back();

    bool use_fp32 = false;
    if constexpr (OpType == OperationType::HALF) {
      // FIXME: set use_fp32 to true if weight is of ScalarType::Float
    } else {
      use_fp32 = true;
    }

    kernel_ = std::make_unique<Kernel>(max_rows, hidden_dims, use_fp32);
    rows_ = max_rows;
  }

  void Initialize(const T *gamma, const T *beta) {
    Param param;
    param.gamma = gamma;
    param.beta = beta;
    kernel_->initialize(param);
  }

protected:
  int rows_;
  std::unique_ptr<Kernel> kernel_;
};

template <OperationType OpType, typename T>
class LayerNormForwardImpl : public LayerNormImplBase<OpType, T> {
  using ForwardParam = fastertransformerv4::LayerNormForwardParam<T>;
  using Base = LayerNormImplBase<OpType, T>;

public:
  LayerNormForwardImpl(const OpAccessor &accessor) : Base(accessor, 0) {}

  void Execute(const T *input, const T *gamma, const T *beta, T *output,
               T *mean, T *var_rsqrt, const T *residual, T *input_add_residual,
               cudaStream_t stream) {
    Base::Initialize(gamma, beta);

    ForwardParam param{.input = input,
                       .mean = mean,
                       .var_rsqrt = var_rsqrt,
                       .layernorm_out = output,
                       .rows = this->rows_,
                       .stream = stream,
                       .residual = residual,
                       .input_add_residual = input_add_residual};
    this->kernel_->forward(param);
  }
};

template <OperationType OpType, typename T>
class LayerNormBackwardImpl : public LayerNormImplBase<OpType, T> {
  using BackwardParam = fastertransformerv4::LayerNormBackwardParam<T>;
  using Base = LayerNormImplBase<OpType, T>;

public:
  LayerNormBackwardImpl(const OpAccessor &accessor) : Base(accessor, 0) {}

  void Execute(const T *grad_out, const T *input_add_residual, const T *gamma,
               const T *mean, const T *var_rsqrt, T *grad_in, T *grad_gamma,
               T *grad_beta, T *grad_residual, void *buf, cudaStream_t stream) {
    Base::Initialize(gamma, nullptr);
    BackwardParam param{.grad_out = grad_out,
                        .input_add_residual = input_add_residual,
                        .mean = mean,
                        .var_rsqrt = var_rsqrt,
                        .grad_in = grad_in,
                        .grad_gamma = grad_gamma,
                        .grad_beta = grad_beta,
                        .buf = buf,
                        .rows = this->rows_,
                        .stream = stream,
                        .grad_residual = grad_residual};
    this->kernel_->backward(param);
  }

  unsigned long long GetWorkspaceSize(const ExecutionContext & /*ctx*/) const {
    return this->kernel_->cal_bw_bufsize();
  }
};

template <OperationType OpType,
          typename T = typename fastertransformerv4::Traits<OpType>::DataType>
using LayerNormForward = CudaOpKernel<LayerNormForwardImpl<OpType, T>,
                                      TypedOperand<const T *, 0>, // input
                                      TypedOperand<const T *, 1>, // gamma
                                      TypedOperand<const T *, 2>, // beta
                                      TypedOperand<T *, 3>,       // output
                                      TypedOperand<T *, 4>,       // mean
                                      TypedOperand<T *, 5>,       // var_rsqrt
                                      NoneArg,                    // residual
                                      NoneArg // input_add_residual
                                      >;

template <OperationType OpType,
          typename T = typename fastertransformerv4::Traits<OpType>::DataType>
using LayerNormResidualForward =
    CudaOpKernel<LayerNormForwardImpl<OpType, T>,
                 TypedOperand<const T *, 0>, // input
                 TypedOperand<const T *, 1>, // gamma
                 TypedOperand<const T *, 2>, // beta
                 TypedOperand<T *, 4>,       // output
                 TypedOperand<T *, 5>,       // mean
                 TypedOperand<T *, 6>,       // var_rsqrt
                 TypedOperand<const T *, 3>, // residual
                 TypedOperand<T *, 7>        // input_add_residual
                 >;

template <OperationType OpType,
          typename T = typename fastertransformerv4::Traits<OpType>::DataType>
using LayerNormBackward =
    CudaOpKernelWithWorkspace<LayerNormBackwardImpl<OpType, T>,
                              TypedOperand<const T *, 0>, // grad_out
                              TypedOperand<const T *, 1>, // input_add_residual
                              TypedOperand<const T *, 2>, // gamma
                              TypedOperand<const T *, 3>, // mean
                              TypedOperand<const T *, 4>, // var_rsqrt
                              TypedOperand<T *, 5>,       // grad_in
                              TypedOperand<T *, 6>,       // grad_gamma
                              TypedOperand<T *, 7>,       // grad_beta
                              NoneArg                     // grad_residual
                              >;

template <OperationType OpType,
          typename T = typename fastertransformerv4::Traits<OpType>::DataType>
using LayerNormResidualBackward =
    CudaOpKernelWithWorkspace<LayerNormBackwardImpl<OpType, T>,
                              TypedOperand<const T *, 0>, // grad_out
                              TypedOperand<const T *, 1>, // input_add_residual
                              TypedOperand<const T *, 2>, // gamma
                              TypedOperand<const T *, 3>, // mean
                              TypedOperand<const T *, 4>, // var_rsqrt
                              TypedOperand<T *, 5>,       // grad_in
                              TypedOperand<T *, 6>,       // grad_gamma
                              TypedOperand<T *, 7>,       // grad_beta
                              TypedOperand<T *, 8>        // grad_residual
                              >;

} // namespace ftv4
} // namespace cuda
} // namespace brt
