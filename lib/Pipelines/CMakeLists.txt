add_mlir_library(ByteIRPipelines
  AffineOpt.cpp
  ByreOpt.cpp
  Common.cpp
  GPUOpt.cpp
  HloOpt.cpp
  LinalgOpt.cpp
  NVVMCodegen.cpp
  TotalBufferize.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Pipelines

  DEPENDS
  ByteIRPipelinesPassIncGen
  ByteIRTransforms

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRAffine
  MLIRLinalg
  MLIRMemRef
  MLIRSCF
  MLIRTransforms
)