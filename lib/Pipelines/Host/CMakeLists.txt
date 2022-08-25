add_mlir_library(ByteIRHostPipelines
  HostOpt.cpp
  ToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Pipelines/Host

  DEPENDS
  ByteIRHostPipelinesPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR

  MLIRArithmeticDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect

  ByteIRToLLVM
  MLIRArithmeticTransforms
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRTensorTransforms
  MLIRTransforms
)