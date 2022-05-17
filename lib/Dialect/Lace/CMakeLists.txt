add_mlir_dialect_library(MLIRLace
  IR/LaceDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/mlir/Dialect/Lace

  DEPENDS
  MLIRLaceOpsIncGen
  MLIRLaceOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRViewLikeInterface
)