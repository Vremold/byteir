add_mlir_library(ByteIRGPUPipelines
  GPUOpt.cpp
  LinalgMemrefGPU.cpp
  NVVMCodegen.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Pipelines/GPU

  DEPENDS
  ByteIRTransforms
  ByteIRUtils
  MhloDialect

  LINK_LIBS PUBLIC
  ByteIRLinalgPasses
  ByteIRUtils
  ByteIRSCFPasses
  ByteIRToPTX
  MLIRIR
  MLIRTransforms
)