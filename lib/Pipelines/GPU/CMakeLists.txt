add_mlir_library(ByteIRGPUPipelines
  GPUOpt.cpp
  LinalgMemrefGPU.cpp
  NVVMCodegen.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Pipelines/GPU

  DEPENDS
  ByteIRGPUPipelinesPassIncGen
  ByteIRTransforms
  ByteIRUtils

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRAffine
  MLIRLinalg
  MLIRMemRef
  MLIRSCF
  MLIRTransforms
)