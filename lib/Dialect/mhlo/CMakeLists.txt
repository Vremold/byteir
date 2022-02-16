add_mlir_dialect_library(ByteIRMhlo
  Util/Util.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo/Util

  DEPENDS
  MhloDialect
  ByteIRUtils

  LINK_LIBS PUBLIC
  MLIRIR
  MhloDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  ByteIRUtils
  )

add_mlir_dialect_library(ByteIRMhloAnalysis
  Analysis/DimFromBroadcast.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo/Analysis

  DEPENDS
  ByteIRAnalysis
  MhloDialect
  

  LINK_LIBS PUBLIC
  ByteIRAnalysis
  MLIRIR
  MLIRSupport
  MhloDialect
)

add_mlir_dialect_library(ByteIRMhloPasses
  Transforms/ConvBiasActFusion.cpp
  Transforms/DotTransposeFusion.cpp
  Transforms/ElementFusion.cpp
  Transforms/FusionOutlining.cpp
  Transforms/FusionUtil.cpp
  Transforms/HloFolder.cpp
  Transforms/HloTransposeDotToDotGeneral.cpp
  Transforms/ReduceFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo

  DEPENDS
  ByteIRMhloPassIncGen
  ByteIRAnalysis
  MhloDialect
  ByteIRMhlo
  ByteIRUtils
  ByteIRMhloAnalysis

  LINK_LIBS PUBLIC
  MLIRIR
  MhloDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  ByteIRAnalysis
  ByteIRMhlo
  ByteIRUtils
  ByteIRMhloAnalysis
  )
