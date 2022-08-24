add_mlir_dialect_library(ByteIRMhloUtils
  Util/FusionUtil.cpp
  Util/ShapeInferUtil.cpp
  Util/Util.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo/Util

  DEPENDS
  MhloDialect
  ByteIRUtils

  LINK_LIBS PUBLIC
  LmhloDialect
  MLIRIR
  MhloDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  ByteIRAnalysis
  ByteIRUtils
  )


add_mlir_dialect_library(ByteIRMhloDynamicShapeOpRegister
  DynamicShapeOpRegister/Convolution.cpp
  DynamicShapeOpRegister/Einsum.cpp
  DynamicShapeOpRegister/DotLike.cpp 
  DynamicShapeOpRegister/NonZero.cpp
  DynamicShapeOpRegister/DynamicBroadcastInDim.cpp
  DynamicShapeOpRegister/RealDynamicSlice.cpp
  DynamicShapeOpRegister/DynamicPartition.cpp
  DynamicShapeOpRegister/Reduce.cpp
  DynamicShapeOpRegister/DynamicReshape.cpp
  DynamicShapeOpRegister/TorchIndexSelect.cpp
  DynamicShapeOpRegister/DynamicStitchLike.cpp
  DynamicShapeOpRegister/Where.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo/BoundedShapes
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo/ReifyShapes
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo/ShapeConstraints

  DEPENDS
  MhloDialect
  ByteIRMhloUtils
  ByteIRUtils

  LINK_LIBS PUBLIC
  MLIRIR
  MhloDialect
  MLIRSupport
  ByteIRUtils
)

add_mlir_dialect_library(ByteIRMhloAnalysis
  Analysis/DimFromBroadcast.cpp
  Analysis/BoundedShapeAnalysis.cpp

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
  Transforms/BoundedShapeInference.cpp
  Transforms/CanonicalExt.cpp
  Transforms/ClusterConstraint.cpp
  Transforms/ConvBackwardFusion.cpp
  Transforms/ConvForwardFusion.cpp
  Transforms/ConvertFuncToCustomCall.cpp
  Transforms/ConvertFuncToCustomCall/TorchXLA.cpp
  Transforms/ConvertInsertion.cpp
  Transforms/DotTransposeFusion.cpp
  Transforms/DynamicShapeClustering.cpp
  Transforms/FuncArgRearrangement.cpp
  Transforms/FusionOutlining.cpp
  Transforms/GenericFusion.cpp
  Transforms/HloFolder.cpp
  Transforms/HloMoveDown.cpp
  Transforms/HloMoveUp.cpp
  Transforms/HloTransposeDotToDotGeneral.cpp
  Transforms/InsertShapeConstraint.cpp
  Transforms/IOConvertFusion.cpp
  Transforms/LayoutTransformation.cpp
  Transforms/ReduceFusion.cpp
  Transforms/RewriteWithConstraint.cpp
  Transforms/ShapeReification.cpp
  Transforms/StaticShapeInference.cpp
  Transforms/TrivialFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo/Transforms

  DEPENDS
  ByteIRMhloPassIncGen
  ByteIRAnalysis
  MhloDialect
  ByteIRMhloUtils
  ByteIRUtils
  ByteIRMhloAnalysis
  ByteIRMhloDynamicShapeOpRegister
  MLIRAceDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MhloDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  ByteIRAnalysis
  ByteIRMhloUtils
  ByteIRUtils
  ByteIRMhloAnalysis
  ByteIRMhloDynamicShapeOpRegister
  MLIRAceDialect
  MLIRQuantDialect # FIXME: this should link at MhloDialect
  )
