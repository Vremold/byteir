add_mlir_library(ByteIRPipelines
  AffineOpt.cpp
  AllOpt.cpp
  ByreHost.cpp
  ByreOpt.cpp
  GPUOpt.cpp
  HloOpt.cpp
  LinalgOpt.cpp
  NVVMCodegen.cpp
  SCFOpt.cpp
  TotalBufferize.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Pipelines

  DEPENDS
  ByteIRPipelinesPassIncGen
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