#pragma once

#include "./common.h"

#include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"
#include "fastertransformer_v4/includes/transpose.h"

namespace brt {
namespace cuda {
namespace ftv4 {
template <OperationType OpType, typename T, bool IsReverse>
class Transpose4DImpl {
  using Kernel = fastertransformerv4::Transpose<OpType>;
  using Param = fastertransformerv4::TransposeParam<T>;

public:
  Transpose4DImpl(const OpAccessor &accessor) {
    auto tensor_shape = accessor.GetArgShape(0);
    dim_1_ = tensor_shape[0];
    dim_2_ = tensor_shape[1];
    dim_3_ = tensor_shape[2];
    dim_4_ = tensor_shape[3];

    auto forward_transpose_type = ConvertTransposeType(
        accessor.GetAttrAsString("forward_transpose_type"));
    if constexpr (IsReverse) {
      switch (forward_transpose_type) {
      case TransposeType::TRANSPOSE0213:
        transpose_type_ = TransposeType::TRANSPOSE0213;
        break;
      case TransposeType::TRANSPOSE1203:
        transpose_type_ = TransposeType::TRANSPOSE2013;
        break;
      case TransposeType::TRANSPOSE2013:
        transpose_type_ = TransposeType::TRANSPOSE1203;
        break;
      default:
        BRT_THROW("unknown transpose type");
      }
    } else {
      transpose_type_ = forward_transpose_type;
    }
  }

  void Execute(const T *input, T *output, cudaStream_t stream) {
    Param param{.input = input,
                .output = output,
                .buf = nullptr,
                .dim_1 = dim_1_,
                .dim_2 = dim_2_,
                .dim_3 = dim_3_,
                .dim_4 = dim_4_,
                .stream = stream,
                .transpose_type = transpose_type_};
    Kernel::forward(param);
  }

private:
  int dim_1_, dim_2_, dim_3_, dim_4_;
  TransposeType transpose_type_;
};

template <OperationType OpType, typename T, bool IsReverse>
using Transpose4DBase = CudaOpKernel<Transpose4DImpl<OpType, T, IsReverse>,
                                     TypedOperand<const T *, 0>, // input
                                     TypedOperand<T *, 1>        // output
                                     >;

template <OperationType OpType,
          typename T = typename fastertransformerv4::Traits<OpType>::DataType>
using Transpose4DForward = Transpose4DBase<OpType, T, false>;

template <OperationType OpType,
          typename T = typename fastertransformerv4::Traits<OpType>::DataType>
using Transpose4DBackward = Transpose4DBase<OpType, T, true>;

} // namespace ftv4
} // namespace cuda
} // namespace brt
