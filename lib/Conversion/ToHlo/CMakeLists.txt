set(LLVM_TARGET_DEFINITIONS ArithToMhloPattern.td)
mlir_tablegen(ArithToMhloPattern.inc -gen-rewriters)
add_public_tablegen_target(ArithToMhloPatternIncGen)

add_byteir_conversion_library(ByteIRToMhlo
  ArithToMhlo.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/ToMhlo

  DEPENDS
  ByteIRConversionPassIncGen
  ArithToMhloPatternIncGen

  LINK_LIBS PUBLIC
  MhloDialect
  MLIRArithmetic
  MLIRIR
  )
