
#include "backends/cuda/providers/default/ftv4/models.h"
#include "brt/core/ir/builder.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"

using namespace brt;
using namespace brt::ir;
using namespace brt::test;
using namespace mlir;
using namespace mlir::byre;

using AT = EntryFuncArgType;

namespace brt {
namespace test {
namespace cuda {
namespace ftv4 {
const void *CreateLayerNorm(brt::ir::ByREBuilder &byre_builder) {

  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  int hidden_dim = 768, rows = 32768;
  auto space_attr = StringAttr::get(ctx, "cuda");
  auto inout = MemRefType::get({rows, hidden_dim}, op_builder.getF32Type(),
                               MemRefLayoutAttrInterface{}, space_attr);
  auto parameter = MemRefType::get({hidden_dim}, op_builder.getF32Type(),
                                   MemRefLayoutAttrInterface{}, space_attr);
  auto statistic = MemRefType::get({rows}, op_builder.getF32Type(),
                                   MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{parameter, AT::Weight, "gamma"},
               {parameter, AT::Weight, "beta"},
               {inout, AT::Input, "input"},
               {inout, AT::Output, "output"},
               {statistic, AT::Output, "mean"},
               {statistic, AT::Output, "var_rsqrt"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert Ops
  SmallVector<Value, 4> op_inputs{entry_block->getArgument(2),
                                  entry_block->getArgument(0),
                                  entry_block->getArgument(1)};
  SmallVector<Value, 4> op_outputs{entry_block->getArgument(3),
                                   entry_block->getArgument(4),
                                   entry_block->getArgument(5)};
  op_builder.create<byre::ComputeOp>(UnknownLoc::get(ctx), "ftv4.layernorm",
                                     op_inputs, op_outputs);

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));
  return module_op.getAsOpaquePointer();
}

const void *CreateLayerNormBackward(brt::ir::ByREBuilder &byre_builder) {

  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  int hidden_dim = 768, rows = 32768;
  auto space_attr = StringAttr::get(ctx, "cuda");
  auto inout = MemRefType::get({rows, hidden_dim}, op_builder.getF32Type(),
                               MemRefLayoutAttrInterface{}, space_attr);
  auto parameter = MemRefType::get({hidden_dim}, op_builder.getF32Type(),
                                   MemRefLayoutAttrInterface{}, space_attr);
  auto statistic = MemRefType::get({rows}, op_builder.getF32Type(),
                                   MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{parameter, AT::Weight, "gamma"},
               {inout, AT::Input, "grad_out"},
               {inout, AT::Input, "input"},
               {statistic, AT::Input, "mean"},
               {statistic, AT::Input, "var_rsqrt"},
               {inout, AT::Output, "grad_in"},
               {parameter, AT::Output, "grad_gamma"},
               {parameter, AT::Output, "grad_beta"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert Ops
  SmallVector<Value> op_inputs = {
      entry_block->getArgument(1), entry_block->getArgument(2),
      entry_block->getArgument(0), entry_block->getArgument(3),
      entry_block->getArgument(4)};
  SmallVector<Value> op_outputs{entry_block->getArgument(5),
                                entry_block->getArgument(6),
                                entry_block->getArgument(7)};
  op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "ftv4.layernorm_backward", op_inputs, op_outputs);

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}

const void *CreateSoftmax(brt::ir::ByREBuilder &byre_builder) {

  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  int bs = 128, seq_len = 256, head_num = 12;
  auto space_attr = StringAttr::get(ctx, "cuda");
  auto inout =
      MemRefType::get({bs, head_num, seq_len, seq_len}, op_builder.getF32Type(),
                      MemRefLayoutAttrInterface{}, space_attr);
  auto mask =
      MemRefType::get({bs, head_num, seq_len, seq_len}, op_builder.getI8Type(),
                      MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{inout, AT::Input, "input"},
               {mask, AT::Input, "mask"},
               {inout, AT::Output, "softmax_output"},
               {inout, AT::Output, "softmax_dropout_output"},
               {inout, AT::Output, "dropout_mask"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert Ops
  SmallVector<Value> op_inputs{entry_block->getArgument(0),
                               entry_block->getArgument(1)};
  SmallVector<Value> op_outputs{entry_block->getArgument(2),
                                entry_block->getArgument(3),
                                entry_block->getArgument(4)};
  auto op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "ftv4.softmax", op_inputs, op_outputs);

  op->setAttr("head_num", op_builder.getI64IntegerAttr(head_num));
  op->setAttr("dropout_rate", op_builder.getF32FloatAttr(0.1f));
  op->setAttr("batch_first", op_builder.getBoolAttr(true));

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}

const void *CreateSoftmaxBackward(brt::ir::ByREBuilder &byre_builder) {

  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  int bs = 128, seq_len = 256, head_num = 12;
  auto space_attr = StringAttr::get(ctx, "cuda");
  auto inout =
      MemRefType::get({bs, head_num, seq_len, seq_len}, op_builder.getF32Type(),
                      MemRefLayoutAttrInterface{}, space_attr);
  auto mask =
      MemRefType::get({bs, head_num, seq_len, seq_len}, op_builder.getI8Type(),
                      MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{inout, AT::Input, "grad_out"},
               {inout, AT::Input, "out"},
               {mask, AT::Input, "dropout_mask"},
               {inout, AT::Output, "grad_in"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert Ops
  SmallVector<Value> op_inputs{entry_block->getArgument(0),
                               entry_block->getArgument(1),
                               entry_block->getArgument(2)};
  SmallVector<Value> op_outputs{entry_block->getArgument(3)};
  auto op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "ftv4.softmax_backward", op_inputs, op_outputs);

  op->setAttr("dropout_rate", op_builder.getF32FloatAttr(0.1f));

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}

const void *CreateLinearGeluDropout(brt::ir::ByREBuilder &byre_builder) {

  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  int64_t rows = 1024, K = 128, N = 64;
  auto space_attr = StringAttr::get(ctx, "cuda");
  auto input = MemRefType::get({rows, K}, op_builder.getF32Type(),
                               MemRefLayoutAttrInterface{}, space_attr);
  auto weight = MemRefType::get({N, K}, op_builder.getF32Type(),
                                MemRefLayoutAttrInterface{}, space_attr);
  auto bias = MemRefType::get({N}, op_builder.getF32Type(),
                              MemRefLayoutAttrInterface{}, space_attr);
  auto output = MemRefType::get({rows, N}, op_builder.getF32Type(),
                                MemRefLayoutAttrInterface{}, space_attr);
  auto mask = MemRefType::get({rows, N}, op_builder.getI8Type(),
                              MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{weight, AT::Weight, "weight"},
               {bias, AT::Weight, "bias"},
               {input, AT::Input, "input"},
               {output, AT::Output, "output"},
               {output, AT::Output, "bias_out"},
               {mask, AT::Output, "dropout_mask"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert Ops
  SmallVector<Value> op_inputs{entry_block->getArgument(2),
                               entry_block->getArgument(0),
                               entry_block->getArgument(1)};
  SmallVector<Value> op_outputs{entry_block->getArgument(3),
                                entry_block->getArgument(4),
                                entry_block->getArgument(5)};
  auto op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "ftv4.linear_gelu_dropout", op_inputs, op_outputs);

  op->setAttr("dropout_rate", op_builder.getF32FloatAttr(0.1f));
  op->setAttr("act_gelu", op_builder.getBoolAttr(true));

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}

const void *
CreateLinearGeluDropoutBackward(brt::ir::ByREBuilder &byre_builder) {

  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  int64_t rows = 1024, K = 128, N = 64;
  auto space_attr = StringAttr::get(ctx, "cuda");
  auto input = MemRefType::get({rows, K}, op_builder.getF32Type(),
                               MemRefLayoutAttrInterface{}, space_attr);
  auto weight = MemRefType::get({N, K}, op_builder.getF32Type(),
                                MemRefLayoutAttrInterface{}, space_attr);
  auto bias = MemRefType::get({N}, op_builder.getF32Type(),
                              MemRefLayoutAttrInterface{}, space_attr);
  auto output = MemRefType::get({rows, N}, op_builder.getF32Type(),
                                MemRefLayoutAttrInterface{}, space_attr);
  auto mask = MemRefType::get({rows, N}, op_builder.getI8Type(),
                              MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{weight, AT::Weight, "weight"},
               {output, AT::Input, "grad_output"},
               {input, AT::Input, "input"},
               {output, AT::Input, "bias_out"},
               {mask, AT::Input, "dropout_mask"},
               {input, AT::Output, "grad_in"},
               {weight, AT::Output, "grad_weight"},
               {bias, AT::Output, "grad_bias"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert Ops
  SmallVector<Value> op_inputs{
      entry_block->getArgument(1), entry_block->getArgument(2),
      entry_block->getArgument(0), entry_block->getArgument(3),
      entry_block->getArgument(4)};
  SmallVector<Value> op_outputs{entry_block->getArgument(5),
                                entry_block->getArgument(6),
                                entry_block->getArgument(7)};
  auto op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "ftv4.linear_gelu_dropout_backward", op_inputs,
      op_outputs);

  op->setAttr("dropout_rate", op_builder.getF32FloatAttr(0.1f));
  op->setAttr("act_gelu", op_builder.getBoolAttr(true));

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}

const void *CreateLinearTranspose0213(brt::ir::ByREBuilder &byre_builder) {

  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  int from_feature = 384, to_feature = 768, batch_size = 256, seq_len = 128,
      head_num = 12, head_size = to_feature / head_num;
  auto space_attr = StringAttr::get(ctx, "cuda");
  auto input = MemRefType::get({batch_size, seq_len, from_feature},
                               op_builder.getF32Type(),
                               MemRefLayoutAttrInterface{}, space_attr);
  auto weight =
      MemRefType::get({to_feature, from_feature}, op_builder.getF32Type(),
                      MemRefLayoutAttrInterface{}, space_attr);
  auto bias = MemRefType::get({to_feature}, op_builder.getF32Type(),
                              MemRefLayoutAttrInterface{}, space_attr);
  auto output = MemRefType::get({batch_size, head_num, seq_len, head_size},
                                op_builder.getF32Type(),
                                MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{weight, AT::Weight, "weight"},
               {bias, AT::Weight, "bias"},
               {input, AT::Input, "input"},
               {output, AT::Output, "output"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert Ops
  SmallVector<Value> op_inputs{entry_block->getArgument(2),
                               entry_block->getArgument(0),
                               entry_block->getArgument(1)};
  SmallVector<Value> op_outputs{entry_block->getArgument(3)};
  auto op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "ftv4.linear_transpose", op_inputs, op_outputs);

  op->setAttr("head_num", op_builder.getI32IntegerAttr(head_num));
  op->setAttr("forward_transpose_type",
              op_builder.getStringAttr("TRANSPOSE0213"));

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}

const void *
CreateLinearTranspose0213Backward(brt::ir::ByREBuilder &byre_builder) {

  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  int from_feature = 384, to_feature = 768, batch_size = 256, seq_len = 128,
      head_num = 12, head_size = to_feature / head_num;
  auto space_attr = StringAttr::get(ctx, "cuda");
  auto input = MemRefType::get({batch_size, seq_len, from_feature},
                               op_builder.getF32Type(),
                               MemRefLayoutAttrInterface{}, space_attr);
  auto weight =
      MemRefType::get({to_feature, from_feature}, op_builder.getF32Type(),
                      MemRefLayoutAttrInterface{}, space_attr);
  auto bias = MemRefType::get({to_feature}, op_builder.getF32Type(),
                              MemRefLayoutAttrInterface{}, space_attr);
  auto output = MemRefType::get({batch_size, head_num, seq_len, head_size},
                                op_builder.getF32Type(),
                                MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{weight, AT::Weight, "weight"},
               {output, AT::Input, "grad_out"},
               {input, AT::Input, "input"},
               {input, AT::Output, "grad_in"},
               {weight, AT::Output, "grad_weight"},
               {bias, AT::Output, "grad_bias"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert Ops
  SmallVector<Value> op_inputs{entry_block->getArgument(1),
                               entry_block->getArgument(2),
                               entry_block->getArgument(0)};
  SmallVector<Value> op_outputs{entry_block->getArgument(3),
                                entry_block->getArgument(4),
                                entry_block->getArgument(5)};
  auto op = op_builder.create<byre::ComputeOp>(UnknownLoc::get(ctx),
                                               "ftv4.linear_transpose_backward",
                                               op_inputs, op_outputs);

  op->setAttr("head_num", op_builder.getI32IntegerAttr(head_num));
  op->setAttr("forward_transpose_type",
              op_builder.getStringAttr("TRANSPOSE0213"));

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}

const void *CreateTranspose4d2013(brt::ir::ByREBuilder &byre_builder) {

  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  int d0 = 3, d1 = 4, d2 = 5, d3 = 6;
  auto space_attr = StringAttr::get(ctx, "cuda");
  auto input = MemRefType::get({d0, d1, d2, d3}, op_builder.getF32Type(),
                               MemRefLayoutAttrInterface{}, space_attr);
  auto output = MemRefType::get({d0, d2, d1, d3}, op_builder.getF32Type(),
                                MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{input, AT::Input, "input"},
               {output, AT::Input, "grad_out"},
               {output, AT::Output, "output"},
               {input, AT::Output, "grad_in"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert Ops
  auto op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "ftv4.transpose4d",
      ValueRange{entry_block->getArgument(0)},
      ValueRange{entry_block->getArgument(2)});
  op->setAttr("forward_transpose_type",
              op_builder.getStringAttr("TRANSPOSE0213"));
  op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "ftv4.transpose4d_backward",
      ValueRange{entry_block->getArgument(1)},
      ValueRange{entry_block->getArgument(3)});
  op->setAttr("forward_transpose_type",
              op_builder.getStringAttr("TRANSPOSE0213"));

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}

const void *CreateMatmul(brt::ir::ByREBuilder &byre_builder) {

  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  int M = 32, N = 64, K = 128;
  auto space_attr = StringAttr::get(ctx, "cuda");
  auto A = MemRefType::get({M, K}, op_builder.getF32Type(),
                           MemRefLayoutAttrInterface{}, space_attr);
  auto B = MemRefType::get({N, K}, op_builder.getF32Type(),
                           MemRefLayoutAttrInterface{}, space_attr);
  auto C = MemRefType::get({M, N}, op_builder.getF32Type(),
                           MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{A, AT::Input, "input_A"},
               {B, AT::Input, "input_B"},
               {C, AT::Input, "grad_C"},
               {C, AT::Output, "output_C"},
               {A, AT::Output, "grad_A"},
               {B, AT::Output, "grad_B"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert Ops
  auto op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "ftv4.matmul",
      ValueRange{entry_block->getArgument(0), entry_block->getArgument(1)},
      ValueRange{entry_block->getArgument(3)});
  op->setAttr("transpose_a", op_builder.getBoolAttr(false));
  op->setAttr("transpose_b", op_builder.getBoolAttr(true));
  op->setAttr("scale", op_builder.getF32FloatAttr(1.0f));
  op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "ftv4.matmul_backward",
      ValueRange{entry_block->getArgument(2), entry_block->getArgument(0),
                 entry_block->getArgument(1)},
      ValueRange{entry_block->getArgument(4), entry_block->getArgument(5)});
  op->setAttr("transpose_a", op_builder.getBoolAttr(false));
  op->setAttr("transpose_b", op_builder.getBoolAttr(true));
  op->setAttr("scale", op_builder.getF32FloatAttr(1.0f));

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}
} // namespace ftv4
} // namespace cuda
} // namespace test
} // namespace brt
