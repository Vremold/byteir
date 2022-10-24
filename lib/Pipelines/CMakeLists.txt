add_subdirectory(GPU)
add_subdirectory(Host)

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
  ByteIRHostPipelines
  ByteIRMhloPasses
  ByteIRShapePasses
  ByteIRLinalgPasses
  ByteIRTransforms
  ByteIRUtils

  LINK_LIBS PUBLIC
  ByteIRGPUPipelines
  ByteIRHostPipelines
  ByteIRUtils
  MLIRIR
  MLIRTransforms
  ByteIRAcePasses
  ByteIRByrePasses
  ByteIRHloToLhlo
  ByteIRLmhloToLace
  ByteIRMhloPasses
  ByteIRShapePasses
  ByteIRLinalgPasses
  ByteIRToByre
  ByteIRToGPU
  ByteIRHloToLinalg
)