set(LLVM_TARGET_DEFINITIONS MhloToAceActivationPattern.td)
mlir_tablegen(MhloToAceActivationPattern.inc -gen-rewriters)
add_public_tablegen_target(MhloToAceActivationPatternIncGen)

add_byteir_conversion_library(ByteIRToAce
  MhloToAce.cpp
  UnregisteredToAce.cpp

  DEPENDS
  ByteIRConversionPassIncGen
  ByteIRConversionCommon
  MhloToAceActivationPatternIncGen

  LINK_LIBS PUBLIC
  ByteIRConversionCommon
  ByteIRUtils
  MhloDialect
  MLIRIR
  )

target_include_directories(ByteIRToAce PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
