add_mlir_dialect_library(MLIRShapeExt
  IR/ShapeExtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/mlir/Dialect/Shape

  DEPENDS
  MLIRShapeExtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

add_mlir_dialect_library(ByteIRShapePasses
  Transforms/InsertTieShape.cpp
  Transforms/ResolveShapeConstraint.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Shape
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Shape/Transforms

  DEPENDS
  ByteIRShapePassIncGen
  ByteIRUtils
  MLIRShapeExt

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRShape
  MLIRShapeExt
  MLIRSupport
  )
