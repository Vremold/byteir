add_mlir_library(ByteIRAnalysis
  OpDependence.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Analysis

  LINK_LIBS PUBLIC
  MLIRIR
)