add_byteir_conversion_library(ByteIRGPUToNVVM
  GPUToNVVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/GPUToNVVM

  DEPENDS
  ByteIRConversionPassIncGen
  ByteIRConversionCommon

  LINK_LIBS PUBLIC
  MLIRGPUOps
  MLIRGPUToGPURuntimeTransforms
  MLIRGPUToNVVMTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMIR
  MLIRMemRef
  MLIRMemRefToLLVM
  MLIRNVVMIR
  MLIRPass
  MLIRStandardToLLVM
  MLIRTransformUtils
  )
