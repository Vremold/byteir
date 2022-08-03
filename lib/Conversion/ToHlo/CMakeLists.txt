set(LLVM_TARGET_DEFINITIONS ArithToMhloPattern.td)
mlir_tablegen(ArithToMhloPattern.inc -gen-rewriters)
add_public_tablegen_target(ArithToMhloPatternIncGen)

add_byteir_conversion_library(ByteIRToMhlo
  ArithToMhlo.cpp

  DEPENDS
  ByteIRConversionPassIncGen
  ArithToMhloPatternIncGen

  LINK_LIBS PUBLIC
  MhloDialect
  MLIRArithmeticDialect
  MLIRIR
  )

target_include_directories(ByteIRToMhlo PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
