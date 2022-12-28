add_mlir_dialect_library(ByteIRLinalgPasses
  FuseElementwise.cpp
  LinalgDataPlace.cpp
  LinalgPrefetch.cpp
  Tiling.cpp
  TransformInsertion.cpp
  ScopeTiling.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Linalg
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Linalg/Transforms

  DEPENDS
  ByteIRLinalgPassIncGen
  MLIRLinalgExtTransformOps
  ByteIRUtils
  MLIRDestinationStyleOpInterface
  MLIRLinalgExt
  MLIRLinalgDialect

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRAffineDialect
  MLIRDestinationStyleOpInterface
  MLIRLinalgExtTransformOps
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRLinalgDialect
  MLIRLinalgExt
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRSupport
  )
