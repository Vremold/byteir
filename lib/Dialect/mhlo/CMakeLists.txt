add_mlir_dialect_library(ByteIRMhloPasses
  transforms/ConvBiasActFusion.cpp
  transforms/ElementFusion.cpp
  transforms/FusionOutlining.cpp
  transforms/FusionUtil.cpp
  transforms/TransposeDotFusion.cpp
  transforms/TransposeFolder.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/mhlo

  DEPENDS
  ByteIRMhloPassIncGen
  ByteIRAnalysis
  MhloDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MhloDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  ByteIRAnalysis
  )
