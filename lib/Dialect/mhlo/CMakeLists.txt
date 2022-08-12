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

add_mlir_dialect_library(ByteIRMhloReifyShapes
  ReifyShapes/DotRegister.cpp
  ReifyShapes/DotGeneralRegister.cpp
  ReifyShapes/DynamicStitchRegister.cpp
  ReifyShapes/DynamicMaskStitchRegister.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo/ReifyShapes

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

add_mlir_dialect_library(ByteIRMhloShapeConstraints
  ShapeConstraints/DynamicPartitionRegister.cpp
  ShapeConstraints/DynamicReshapeRegister.cpp
  ShapeConstraints/DotGeneralRegister.cpp
  ShapeConstraints/EinsumRegister.cpp

  ADDITIONAL_HEADER_DIRS
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

add_mlir_dialect_library(ByteIRMhloBoundedShapes
  BoundedShapes/NonZeroRegister.cpp
  BoundedShapes/WhereRegister.cpp
  BoundedShapes/DynamicReshapeRegister.cpp
  BoundedShapes/DynamicPartitionRegister.cpp
  BoundedShapes/DynamicBroadcastInDimRegister.cpp
  BoundedShapes/ReduceRegister.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo/BoundedShapes

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
  ByteIRMhloBoundedShapes
  ByteIRMhloPassIncGen
  ByteIRAnalysis
  MhloDialect
  ByteIRMhloUtils
  ByteIRUtils
  ByteIRMhloAnalysis
  ByteIRMhloShapeConstraints
  ByteIRMhloReifyShapes

  LINK_LIBS PUBLIC
  MLIRIR
  MhloDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  ByteIRAnalysis
  ByteIRMhloBoundedShapes
  ByteIRMhloUtils
  ByteIRUtils
  ByteIRMhloAnalysis
  ByteIRMhloShapeConstraints
  ByteIRMhloReifyShapes
  )
