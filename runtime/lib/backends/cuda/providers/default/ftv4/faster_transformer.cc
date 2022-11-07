#include "./ops/layernorm.h"
#include "./ops/linear.h"
#include "./ops/linear_transpose.h"
#include "./ops/matmul.h"
#include "./ops/softmax.h"
#include "./ops/transpose.h"

#include "brt/backends/cuda/providers/default/ftv4/faster_transformer.h"
#include "brt/core/framework/kernel_registry.h"

using namespace brt::cuda::ftv4;

namespace brt {
namespace cuda {

void RegisterFasterTransformerOps(KernelRegistry *registry) {
  registry->Register(
      "ftv4.layernorm",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<LayerNormForward<OperationType::FP32>>(info);
      });
  registry->Register(
      "ftv4.layernorm_backward",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<LayerNormBackward<OperationType::FP32>>(info);
      });
  registry->Register(
      "ftv4.layernorm_residual",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<LayerNormResidualForward<OperationType::FP32>>(
            info);
      });
  registry->Register(
      "ftv4.layernorm_backward_residual",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<LayerNormResidualBackward<OperationType::FP32>>(
            info);
      });
  registry->Register(
      "ftv4.softmax",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<SoftmaxForward<OperationType::FP32>>(info);
      });
  registry->Register(
      "ftv4.softmax_backward",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<SoftmaxBackward<OperationType::FP32>>(info);
      });
  registry->Register(
      "ftv4.linear",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<LinearForward<OperationType::FP32>>(info);
      });
  registry->Register(
      "ftv4.linear_backward",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<LinearBackward<OperationType::FP32>>(info);
      });
  registry->Register(
      "ftv4.linear_gelu_dropout",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<LinearGeluDropoutForward<OperationType::FP32>>(
            info);
      });
  registry->Register(
      "ftv4.linear_gelu_dropout_backward",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<LinearGeluDropoutBackward<OperationType::FP32>>(
            info);
      });
  registry->Register(
      "ftv4.linear_transpose",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<LinearTransposeForward<OperationType::FP32>>(
            info);
      });
  registry->Register(
      "ftv4.linear_transpose_backward",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<LinearTransposeBackward<OperationType::FP32>>(
            info);
      });
  registry->Register(
      "ftv4.transpose4d",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<Transpose4DForward<OperationType::FP32>>(info);
      });
  registry->Register(
      "ftv4.transpose4d_backward",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<Transpose4DBackward<OperationType::FP32>>(info);
      });
  registry->Register(
      "ftv4.matmul",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<MatMulForward<OperationType::FP32>>(info);
      });
  registry->Register(
      "ftv4.matmul_backward",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<MatMulBackward<OperationType::FP32>>(info);
      });
}

} // namespace cuda
} // namespace brt
