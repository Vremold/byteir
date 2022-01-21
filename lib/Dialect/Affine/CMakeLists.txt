add_mlir_dialect_library(ByteIRAffinePasses
  Transforms/AffineLoopFusionEx.cpp
  Transforms/RewriteAffineToMemref.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Affine

  DEPENDS
  ByteIRAffinePassIncGen
  ByteIRUtils
  MLIRAffine
  MLIRMemRef

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRAffine
  MLIRMemRef
  MLIRSideEffectInterfaces
  MLIRSupport
  )
