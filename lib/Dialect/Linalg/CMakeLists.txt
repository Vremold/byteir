add_mlir_dialect_library(ByteIRLinalgPasses
  Transforms/LinalgDataPlace.cpp
  Transforms/LinalgPrefetch.cpp
  Transforms/LinalgScopeTiling.cpp
  Transforms/TilingUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Linalg
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Linalg/Transforms

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
