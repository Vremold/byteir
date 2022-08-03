add_subdirectory(GPU)

add_mlir_library(ByteIRPipelines
  AffineOpt.cpp
  AllOpt.cpp
  ByreHost.cpp
  ByreOpt.cpp
  HloOpt.cpp
  LinalgTensorOpt.cpp
  SCFOpt.cpp
  ShapeOpt.cpp
  TotalBufferize.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Pipelines

  DEPENDS
  ByteIRGPUPipelines
  ByteIRMhloPasses
  ByteIRPipelinesPassIncGen
  ByteIRShapePasses
  ByteIRLinalgPasses
  ByteIRTransforms
  ByteIRUtils

  LINK_LIBS PUBLIC
  ByteIRGPUPipelines
  ByteIRUtils
  MLIRIR
  MLIRAffineDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTransforms
  ByteIRByrePasses
  ByteIRHloToLhlo
  ByteIRLmhloToLace
  ByteIRMhloPasses
  ByteIRShapePasses
  ByteIRLinalgPasses
  ByteIRToByre
)