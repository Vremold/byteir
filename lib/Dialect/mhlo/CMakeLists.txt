set(LLVM_TARGET_DEFINITIONS transforms/ArithOptimizePattern.td)
mlir_tablegen(transforms/ArithOptimizePattern.inc -gen-rewriters)
add_public_tablegen_target(MhloArithOptPatternIncGen)

add_mlir_dialect_library(ByteIRMhloPasses
  transforms/ArithOptimize.cpp
  transforms/ConvBiasActFusion.cpp
  transforms/DotTransposeFusion.cpp
  transforms/ElementFusion.cpp
  transforms/FusionOutlining.cpp
  transforms/FusionUtil.cpp
  transforms/HloFolder.cpp
  transforms/HloTransposeDotToDotGeneral.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo

  DEPENDS
  ByteIRMhloPassIncGen
  ByteIRAnalysis
  MhloDialect
  MhloArithOptPatternIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MhloDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  ByteIRAnalysis
  )
