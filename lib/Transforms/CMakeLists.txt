add_mlir_library(ByteIRTransforms
  AnchoredFuncPipeline.cpp
  CMAE.cpp
  CondCanonicalize.cpp
  CollectFunc.cpp
  FuncTag.cpp
  GraphClusteringByDevice.cpp
  LoopTag.cpp
  LoopUnroll.cpp
  MemoryPlanning.cpp
  RewriteOpToStdCall.cpp
  SetArgShape.cpp
  SetArgSpace.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Transforms

  DEPENDS
  ByteIRTransformsPassIncGen
  ByteIRUtils

  LINK_LIBS PUBLIC
  ByteIRUtils
  MLIRIR
  MLIRAffine
  MLIRMemRef
  MLIRSCF
  MLIRTransforms
)