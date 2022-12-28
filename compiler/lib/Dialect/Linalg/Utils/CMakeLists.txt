add_mlir_library(ByteIRLinalgUtils
  TilingUtils.cpp
  Transforms.cpp

  DEPENDS
  ByteIRUtils
  MLIRLinalgExtTransformOpsIncGen
  MLIRLinalgExt
  MLIRLinalgDialect

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRAffineDialect
  MLIRLinalgExt
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRDestinationStyleOpInterface
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRSupport
  )