/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v4/includes/gemm.h"
#include "fastertransformer_v4/includes/matmul.h"
using namespace std;

namespace fastertransformerv4 {
template <OperationType OpType>
void MatMul<OpType>::forward(MatMulForwardParam param) {
  cublasOperation_t trans_A = param.A_T ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t trans_B = param.B_T ? CUBLAS_OP_T : CUBLAS_OP_N;

  // if(param.batch_count > 1)
  // {
  cublas_Gemm_Strided_Batched(param.input_A, param.input_B, param.output,
                              param.M, param.K, param.N, param.batch_count,
                              trans_A, trans_B, (DataType_)param.scale,
                              (DataType_)0.0f, param.cublas_handle);
  // }
  // else
  // {
  //     dense_layer_kernel_launcher(
  //         param.input_A, param.input_B, param.output,
  //         param.M, param.K, param.N, trans_A, trans_B,
  //         (DataType_)param.scale, (DataType_)0.0f, param.cublas_handle);
  // }
}

template <OperationType OpType>
void MatMul<OpType>::backward(MatMulBackwardParam param) {
  // cublasOperation_t trans_A, trans_B;

  // if(param.batch_count > 1)
  // {
  if (param.A_T == false && param.B_T == false) {
    cublas_Gemm_Strided_Batched(
        param.grad_out, param.input_B, param.grad_A, param.M, param.N, param.K,
        param.batch_count, CUBLAS_OP_N, CUBLAS_OP_T, (DataType_)param.scale,
        (DataType_)0.0f, param.cublas_handle);

    cublas_Gemm_Strided_Batched(
        param.input_A, param.grad_out, param.grad_B, param.K, param.M, param.N,
        param.batch_count, CUBLAS_OP_T, CUBLAS_OP_N, (DataType_)param.scale,
        (DataType_)0.0f, param.cublas_handle);
  } else if (param.A_T == false && param.B_T == true) {
    cublas_Gemm_Strided_Batched(
        param.grad_out, param.input_B, param.grad_A, param.M, param.N, param.K,
        param.batch_count, CUBLAS_OP_N, CUBLAS_OP_N, (DataType_)param.scale,
        (DataType_)0.0f, param.cublas_handle);

    cublas_Gemm_Strided_Batched(
        param.grad_out, param.input_A, param.grad_B, param.N, param.M, param.K,
        param.batch_count, CUBLAS_OP_T, CUBLAS_OP_N, (DataType_)param.scale,
        (DataType_)0.0f, param.cublas_handle);
  } else if (param.A_T == true && param.B_T == false) {
    cublas_Gemm_Strided_Batched(
        param.input_B, param.grad_out, param.grad_A, param.K, param.N, param.M,
        param.batch_count, CUBLAS_OP_N, CUBLAS_OP_T, (DataType_)param.scale,
        (DataType_)0.0f, param.cublas_handle);

    cublas_Gemm_Strided_Batched(
        param.input_A, param.grad_out, param.grad_B, param.K, param.M, param.N,
        param.batch_count, CUBLAS_OP_N, CUBLAS_OP_N, (DataType_)param.scale,
        (DataType_)0.0f, param.cublas_handle);
  } else if (param.A_T == true && param.B_T == true) {
    cublas_Gemm_Strided_Batched(
        param.input_B, param.grad_out, param.grad_A, param.K, param.N, param.M,
        param.batch_count, CUBLAS_OP_T, CUBLAS_OP_T, (DataType_)param.scale,
        (DataType_)0.0f, param.cublas_handle);

    cublas_Gemm_Strided_Batched(
        param.grad_out, param.input_A, param.grad_B, param.N, param.M, param.K,
        param.batch_count, CUBLAS_OP_T, CUBLAS_OP_T, (DataType_)param.scale,
        (DataType_)0.0f, param.cublas_handle);
  }
  // }
  // else
  // {

  // }
}

template void MatMul<OperationType::FP32>::forward(MatMulForwardParam param);
template void MatMul<OperationType::HALF>::forward(MatMulForwardParam param);

template void MatMul<OperationType::FP32>::backward(MatMulBackwardParam param);
template void MatMul<OperationType::HALF>::backward(MatMulBackwardParam param);
} // namespace fastertransformerv4