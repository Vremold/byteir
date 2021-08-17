add_byteir_conversion_library(ByteIRAffineToGPU
  CoalescedForToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/AffineToGPU

  DEPENDS
  ByteIRConversionPassIncGen
  ByteIRConversionCommon

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffine
  MLIRGPUOps
  MLIRMemRef
  MLIRTransforms
  ByteIRConversionCommon
  )
