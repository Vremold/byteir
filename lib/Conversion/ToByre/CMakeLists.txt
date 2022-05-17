add_byteir_conversion_library(ByteIRToByre
  Common.cpp
  ToByre.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/ToByre

  DEPENDS
  MLIRByre
  MLIRLace
  ByteIRConversionPassIncGen
  ByteIRConversionCommon
  ByteIRMhloUtils

  LINK_LIBS PUBLIC
  MLIRByre
  MLIRIR
  MLIRLace
  MLIRMemRef
  MLIRTransforms
  ByteIRConversionCommon
  ByteIRMhloUtils
  )
