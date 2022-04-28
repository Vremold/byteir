add_mlir_dialect_library(ByteIRMhloUtils
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
  LmhloDialect
  

  LINK_LIBS PUBLIC
  ByteIRAnalysis
  MLIRIR
  MLIRSupport
  MhloDialect
  LmhloDialect
)

add_mlir_dialect_library(ByteIRMhloPasses
  Transforms/CanonicalExt.cpp
  Transforms/ClusterConstraint.cpp
  Transforms/ConvBackwardFusion.cpp
  Transforms/ConvBiasActFusion.cpp
  Transforms/DotTransposeFusion.cpp
  Transforms/ElementFusion.cpp
  Transforms/FusionOutlining.cpp
  Transforms/FusionUtil.cpp
  Transforms/HloFolder.cpp
  Transforms/HloMoveDown.cpp
  Transforms/HloMoveUp.cpp
  Transforms/HloTransposeDotToDotGeneral.cpp
  Transforms/IOConvertFusion.cpp
  Transforms/LayoutTransformation.cpp
  Transforms/ReduceFusion.cpp
  Transforms/RewriteWithConstraint.cpp
  Transforms/TrivialFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo

  DEPENDS
  ByteIRMhloPassIncGen
  ByteIRAnalysis
  MhloDialect
  ByteIRMhloUtils
  ByteIRUtils
  ByteIRMhloAnalysis

  LINK_LIBS PUBLIC
  MLIRIR
  MhloDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  ByteIRAnalysis
  ByteIRMhloUtils
  ByteIRUtils
  ByteIRMhloAnalysis
  )
