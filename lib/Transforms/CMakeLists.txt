add_mlir_library(ByteIRTransforms
  AnchoredFuncPipeline.cpp
  CMAE.cpp
  CondCanonicalize.cpp
  CollectFunc.cpp
  FuncTag.cpp
  GenericDeviceConfig.cpp
  GraphClusteringByDevice.cpp
  LoopTag.cpp
  LoopUnroll.cpp
  MemoryPlanning.cpp
  RemoveFuncBody.cpp
  RewriteOpToStdCall.cpp
  SetArgShape.cpp
  SetSpace.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Transforms

  DEPENDS
  ByteIRAnalysis
  ByteIRTransformsPassIncGen
  ByteIRUtils

  LINK_LIBS PUBLIC
  ByteIRAnalysis
  ByteIRUtils
  MLIRIR
  MLIRAffine
  MLIRMemRef
  MLIRSCF
  MLIRTransforms
)