add_mlir_dialect_library(MLIRByre
  IR/ByreDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Byre

  DEPENDS
  MLIRByreOpsIncGen
  MLIRByreOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRDLTI
  MLIRIR
  MLIRMemRef
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRLLVMIR
  )

add_mlir_dialect_library(ByteIRByrePasses
  transforms/Fold.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Byre

  DEPENDS
  ByteIRByrePassIncGen
  MLIRByre

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  )
