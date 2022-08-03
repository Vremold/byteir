add_mlir_dialect_library(ByteIRLinalgPasses
  Transforms/LinalgDataPlace.cpp
  Transforms/LinalgFuseReshape.cpp
  Transforms/LinalgPrefetch.cpp
  Transforms/LinalgScopeTiling.cpp
  Transforms/TilingUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Linalg
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Linalg/Transforms

  DEPENDS
  ByteIRLinalgPassIncGen
  ByteIRUtils
  MLIRLinalgDialect

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRAffineDialect
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRSupport
  )
