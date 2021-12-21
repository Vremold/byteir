add_mlir_dialect_library(ByteIRAffinePasses
  transforms/AffineLoopFusionEx.cpp
  transforms/RewriteAffineToMemref.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Affine

  DEPENDS
  ByteIRAffinePassIncGen
  MLIRAffine
  MLIRMemRef

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffine
  MLIRMemRef
  MLIRSideEffectInterfaces
  MLIRSupport
  )
