# ONNX-Frontend

ONNX-Frontend is a project to build customized onnx graph --> onnx dialect --> mhlo dialect pipeline.

## How to Add ONNX-2-MHLO conversion

### In onnx-mlir repo
- Before you start

  - Build onnx-mlir, see [this](https://code.byted.org/byteir/onnx-mlir/blob/main/docs/BuildOnLinuxOSX.md).
  - Learn the basic knowledge of MLIR, especially the [Pattern Rewritten doc](https://mlir.llvm.org/docs/PatternRewriter/) and the [DDR doc](https://mlir.llvm.org/docs/DeclarativeRewrites/).

- Workflow
  - Sync the [internal gitlab repo's](https://code.byted.org/byteir/onnx-mlir) main branch with the upstream's
  - Create an PR in the internal gitlab repo for security review.
  - After the review approval, push the created branch to your own GitHub forked onnx-mlir repo.
  - Create an PR in GitHub to merge to the upstream onnx-mlir repo.

- Steps to add a onnx-2-mhlo convertion pattern, You can find a op conversion example in [this MR](https://code.byted.org/byteir/onnx-mlir/merge_requests/5).

  - Take a look at the [definition of the onnx op you want to convert](https://code.byted.org/byteir/onnx-mlir/blob/main/src/Dialect/ONNX/ONNXOps.td.inc), and onnx is more like a coarse-grained op collection.
  - Take a look at the [corresponding mhlo op definition](https://github.com/tensorflow/mlir-hlo/blob/master/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td), and mhlo is more like a fine-grained op collection.
  - Implement the pattern under `onnx-mlir/src/Conversion/` directory, note that if there's no 1:1 mapping from onnx to mhlo, converting the coarse-grained onnx op to fine-grained onnx ops is preferred before converting from onnx dialect to mhlo dialect. In this way, the fine-grained onnx ops to mhlo ops' conversion could be reused.
  - Populate the implemented pattern into convert-onnx-to-mhlo pass
  - Add lit test under `onnx-mlir/test/mlir/conversion/onnx_to_mhlo` directory

- Developing tips
  
  - Reference how [tensorflow convert coarse-grained ops to fine-grained ops](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.cc). 
  - Reference how [tensorflow lower tf dialect to mhlo dialect](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/xla/transforms/legalize_tf.cc).


### In onnx-frontend repo

 It submodules onnx-mlir and mhlo-tools and is used to test numerical accuracy. It doesn't plan to open source currently so it is a standalone repo.

- Steps to add numerical accuracy test
  - Add a onnx protobuf file, it could be an op or a model.
  - Convert the onnx pb file to mlir file of mhlo dialect by using onnx-mlir repo.
  - Calculate the onnx pb result by using onnx python package and the mhlo file result by using mhlo-tools, note that they should using the same input data. And then compare their results.
- To be confirmed
  - Could we compare the results of an onnx dialect file and a mhlo file?


