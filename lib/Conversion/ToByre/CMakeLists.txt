add_byteir_conversion_library(ByteIRToByre
  Common.cpp
  ToByre.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/ToByre

  DEPENDS
  MLIRAce
  MLIRByre
  MLIRLace
  ByteIRConversionPassIncGen
  ByteIRConversionCommon
  ByteIRMhloUtils

  LINK_LIBS PUBLIC
  MLIRAce
  MLIRByre
  MLIRIR
  MLIRLace
  MLIRMemRefDialect
  MLIRTransforms
  ByteIRConversionCommon
  ByteIRMhloUtils
  )
