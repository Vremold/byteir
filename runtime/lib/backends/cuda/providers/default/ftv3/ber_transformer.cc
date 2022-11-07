
#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/common/util.h"
#include "brt/backends/cuda/providers/default/ftv3/bert_transformer.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/context/work_queue.h"
#include "brt/core/framework/allocator.h"
#include "brt/core/framework/kernel_registry.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "fastertransformer_v3/includes/bert_transformer.h"
#include "fastertransformer_v3/includes/common.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

using namespace brt;
using namespace brt::common;
using namespace brt::cuda;
using namespace brt::ir;
using namespace fastertransformerv3;
using namespace llvm;
using namespace mlir;
using namespace mlir::byre;

namespace brt {
namespace cuda {

template <OperationType OpType> struct BertTransformerImpl {
  std::string op_uid;
  int batch_size;
  int seq_len;
  int head_num;
  int size_per_head;
  bool is_prenorm;
  bool fused_attention;
  bool remove_padding;

  BertTransformerParam<OpType> init_param;
  std::unique_ptr<BertTransformer<OpType>> transformer_layer;
};

template <OperationType OpType, typename T>
static void SetBertTransformerImpl(BertTransformerImpl<OpType> *impl,
                                   const OpKernelInfo &info) {

  auto byre_op = cast<byre::ByreOp>(info.GetOperation());
  impl->op_uid = ByREHandle::GetOpUID(byre_op);

  auto input_tensor = GetMLIRValueFromOpArgIndex(info, 0);
  impl->batch_size = brt::ir::GetStaticShape(input_tensor).value().front();

  auto atten_mask = GetMLIRValueFromOpArgIndex(info, 1);
  impl->seq_len = brt::ir::GetStaticShape(atten_mask).value().back();

  bool ln_use_fp32 = false;
  auto attr_output_layernorm_beta = GetMLIRValueFromOpArgIndex(info, 10);
  ln_use_fp32 = IsElementType<Float32Type>(attr_output_layernorm_beta);

  if (auto attr = info.GetOperation()->getAttrOfType<IntegerAttr>("head_num")) {
    impl->head_num = attr.getInt();
  } else {
    BRT_THROW("Attribute head_num is not set");
  }

  if (auto attr =
          info.GetOperation()->getAttrOfType<IntegerAttr>("size_per_head")) {
    impl->size_per_head = attr.getInt();
  } else {
    BRT_THROW("Attribute size_per_head is not set");
  }

  if (auto attr = info.GetOperation()->getAttrOfType<BoolAttr>("is_prenorm")) {
    impl->is_prenorm = attr.getValue();
  } else {
    impl->is_prenorm = false;
  }

  if (auto attr =
          info.GetOperation()->getAttrOfType<BoolAttr>("fused_attention")) {
    impl->fused_attention = attr.getValue();
  } else {
    impl->fused_attention = false;
  }

  if (auto attr =
          info.GetOperation()->getAttrOfType<BoolAttr>("remove_padding")) {
    impl->remove_padding = attr.getValue();
  } else {
    impl->remove_padding = false;
  }

  impl->transformer_layer =
      std::unique_ptr<BertTransformer<OpType>>(new BertTransformer<OpType>(
          impl->batch_size, impl->head_num, impl->size_per_head, impl->seq_len,
          impl->fused_attention, impl->remove_padding, ln_use_fp32,
          impl->is_prenorm, false));

  // binding weights buffer
  impl->init_param.attr_kernel_Q =
      static_cast<T *>(GetWeightFromOpArgIndex(info, 2));
  impl->init_param.attr_kernel_K =
      static_cast<T *>(GetWeightFromOpArgIndex(info, 3));
  impl->init_param.attr_kernel_V =
      static_cast<T *>(GetWeightFromOpArgIndex(info, 4));
  impl->init_param.attention_param.attr_bias_Q =
      static_cast<T *>(GetWeightFromOpArgIndex(info, 5));
  impl->init_param.attention_param.attr_bias_K =
      static_cast<T *>(GetWeightFromOpArgIndex(info, 6));
  impl->init_param.attention_param.attr_bias_V =
      static_cast<T *>(GetWeightFromOpArgIndex(info, 7));
  impl->init_param.attr_output_kernel =
      static_cast<T *>(GetWeightFromOpArgIndex(info, 8));
  impl->init_param.attr_output_bias =
      static_cast<T *>(GetWeightFromOpArgIndex(info, 9));

  impl->init_param.inter_kernel =
      static_cast<T *>(GetWeightFromOpArgIndex(info, 12));
  impl->init_param.inter_bias =
      static_cast<T *>(GetWeightFromOpArgIndex(info, 13));
  impl->init_param.output_kernel =
      static_cast<T *>(GetWeightFromOpArgIndex(info, 14));
  impl->init_param.output_bias =
      static_cast<T *>(GetWeightFromOpArgIndex(info, 15));

  impl->init_param.attr_output_layernorm_beta =
      static_cast<void *>(GetWeightFromOpArgIndex(info, 10));
  impl->init_param.attr_output_layernorm_gamma =
      static_cast<void *>(GetWeightFromOpArgIndex(info, 11));
  impl->init_param.output_layernorm_beta =
      static_cast<void *>(GetWeightFromOpArgIndex(info, 16));
  impl->init_param.output_layernorm_gamma =
      static_cast<void *>(GetWeightFromOpArgIndex(info, 17));
}

template <typename T, OperationType OpType>
BertTransformerOp<T, OpType>::BertTransformerOp(const OpKernelInfo &info)
    : OpKernel(info, false, false, true, true),
      impl_(new BertTransformerImpl<OpType>) {}

template <>
BertTransformerOp<float, OperationType::FP32>::BertTransformerOp(
    const OpKernelInfo &info)
    : OpKernel(info, false, false, true, true),
      impl_(new BertTransformerImpl<OperationType::FP32>) {
  SetBertTransformerImpl<OperationType::FP32, float>(impl_, info_);
}

template <>
BertTransformerOp<half, OperationType::HALF>::BertTransformerOp(
    const OpKernelInfo &info)
    : OpKernel(info, false, false, true, true),
      impl_(new BertTransformerImpl<OperationType::HALF>) {
  SetBertTransformerImpl<OperationType::HALF, half>(impl_, info_);
}

template <typename T, OperationType OpType>
BertTransformerOp<T, OpType>::~BertTransformerOp() {
  delete impl_;
}

template <typename T, OperationType OpType>
common::Status
BertTransformerOp<T, OpType>::RunImpl(const ExecutionContext &ctx) {
  ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  size_t handle_offset = state_info.GetStateOffset(BRT_CUBLAS_HANDLE_NAME);
  size_t buf_offset = state_info.GetStateOffset(impl_->op_uid);

  // create BertTransformerInferParam
  BertTransformerInferParam<T> infer_param;
  infer_param.cublas_handle =
      static_cast<cublasHandle_t>(ctx.exec_frame->GetState(handle_offset));
  infer_param.buf =
      static_cast<cublasHandle_t>(ctx.exec_frame->GetState(buf_offset));

  infer_param.batch_size = impl_->batch_size;
  infer_param.seq_len = impl_->seq_len;

  infer_param.stream = nullptr; // for testing now

  auto input_tensor_id = GetTensorIndexFromOpArgIndex(info_, 0);
  auto input_tensor = ctx.exec_frame->GetAsyncValueRef(input_tensor_id);
  infer_param.input_tensor = (T *)input_tensor;

  auto atten_mask_id = GetTensorIndexFromOpArgIndex(info_, 1);
  auto atten_mask = ctx.exec_frame->GetAsyncValueRef(atten_mask_id);
  infer_param.atten_mask = (T *)atten_mask;

  auto output_id = GetTensorIndexFromOpArgIndex(info_, 18);
  auto output = ctx.exec_frame->GetAsyncValueRef(output_id);
  infer_param.transformer_output = (T *)output;

  impl_->transformer_layer->infer(infer_param);

  return Status::OK();
}

template <typename T, OperationType OpType>
common::Status
BertTransformerOp<T, OpType>::ProloguePerFrame(const ExecutionContext &ctx) {

  // create handle if needed
  common::Status status_handle = CreateCuBlasHandle(ctx);
  if (!status_handle.IsOK()) {
    return status_handle;
  }

  // TODO try to move to constructor
  impl_->transformer_layer->initialize(impl_->init_param);

  // Create temp buffer for a frame
  unsigned long long buf_size = impl_->transformer_layer->cal_bufsize();

  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  return state_info.CreateStateIfNotExist(impl_->op_uid, ctx.exec_frame, [&]() {
    void *buf = info_.GetAllocator("cuda")->Alloc(buf_size);
    return buf;
  });
}

template <typename T, OperationType OpType>
common::Status
BertTransformerOp<T, OpType>::EpiloguePerFrame(const ExecutionContext &ctx) {
  common::Status status_handle = DeleteCuBlasHandle(ctx);

  if (!status_handle.IsOK()) {
    return status_handle;
  }

  // free temp buffer for a frame
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  size_t offset = state_info.GetStateOffset(impl_->op_uid);
  void *ptr = ctx.exec_frame->GetAndResetState(offset);
  if (ptr != nullptr) {
    info_.GetAllocator("cuda")->Free(ptr);
  }
  return brt::common::Status::OK();
}

// instantiate
template class BertTransformerOp<float, OperationType::FP32>;
template class BertTransformerOp<half, OperationType::HALF>;

void RegisterBertTransformerOp(KernelRegistry *registry) {
  registry->Register(
      "BertTransformerOp",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<brt::OpKernel>(
            new cuda::BertTransformerOp<float, OperationType::FP32>(info));
        return kernel;
      });
}

} // namespace cuda
} // namespace brt
