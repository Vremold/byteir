add_mlir_dialect_library(ByteIRLinalgPasses
  transforms/LinalgScopeTiling.cpp
  transforms/TilingUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Linalg
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Linalg/transforms

  DEPENDS
  ByteIRLinalgPassIncGen
  ByteIRUtils
  MLIRLinalg

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRAffine
  MLIRTensor
  MLIRSCF
  MLIRLinalg
  MLIRMemRef
  MLIRSupport
  )
