add_mlir_library(ByteIRTransforms
  CMAE.cpp
  CollectFunc.cpp
  FuncTag.cpp
  GraphClusteringByDevice.cpp
  LoopUnroll.cpp
  RewriteOpToStdCall.cpp
  SetArgShape.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Transforms

  DEPENDS
  ByteIRTransformsPassIncGen
  ByteIRUtils

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRAffine
  MLIRMemRef
  MLIRSCF
  MLIRTransforms
)