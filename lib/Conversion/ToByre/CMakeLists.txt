add_byteir_conversion_library(ByteIRToByre
  Common.cpp
  ToByre.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/ToByre

  DEPENDS
  ByteIRConversionPassIncGen
  ByteIRConversionCommon
  ByteIRMhlo

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRef
  MLIRTransforms
  ByteIRConversionCommon
  ByteIRMhlo
  )
