set(LLVM_TARGET_DEFINITIONS MhloToAceActivationPattern.td)
mlir_tablegen(MhloToAceActivationPattern.inc -gen-rewriters)
add_public_tablegen_target(MhloToAceActivationPatternIncGen)

add_byteir_conversion_library(ByteIRToAce
  MhloToAce.cpp
  UnregisteredToAce.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/ToAce

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
