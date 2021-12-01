add_mlir_library(ByteIRTransforms
  CMAE.cpp
  GraphClusteringByDevice.cpp
  RewriteOpToStdCall.cpp
  SetArgShape.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Transforms

  DEPENDS
  ByteIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRef
  MLIRTransforms
)