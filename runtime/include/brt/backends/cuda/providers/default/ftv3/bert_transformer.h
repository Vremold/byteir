
#pragma once

#include "brt/core/framework/op_kernel.h"

namespace fastertransformerv3 {
enum class OperationType;
}

namespace brt {

class KernelRegistry;

namespace cuda {

template <fastertransformerv3::OperationType OpType> struct BertTransformerImpl;

/**
 * BertTransformer Ops
 */
template <typename T, fastertransformerv3::OperationType OpType>
class BertTransformerOp final : public OpKernel {

public:
  explicit BertTransformerOp(const OpKernelInfo &info);

  ~BertTransformerOp();

  common::Status RunImpl(const ExecutionContext &) override;

  common::Status ProloguePerFrame(const ExecutionContext &) override;

  common::Status EpiloguePerFrame(const ExecutionContext &) override;

private:
  BertTransformerImpl<OpType> *impl_;
};

void RegisterBertTransformerOp(KernelRegistry *registry);

} // namespace cuda
} // namespace brt
