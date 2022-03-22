add_byteir_conversion_library(ByteIRToByre
  Common.cpp
  ToByre.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/ToByre

  DEPENDS
  MLIRByre
  ByteIRConversionPassIncGen
  ByteIRConversionCommon
  ByteIRMhlo

  LINK_LIBS PUBLIC
  MLIRByre
  MLIRIR
  MLIRMemRef
  MLIRTransforms
  ByteIRConversionCommon
  ByteIRMhlo
  )
