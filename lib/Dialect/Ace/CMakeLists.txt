add_mlir_dialect_library(MLIRAce
  IR/AceDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/mlir/Dialect/Ace

  DEPENDS
  MLIRAceOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  )