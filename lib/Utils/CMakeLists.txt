add_mlir_library(ByteIRUtils
  AttrUtils.cpp
  Hoist.cpp
  IRRewrite.cpp
  LoopUtils.cpp
  MemUtils.cpp
  PipelineUtils.cpp
  TypeUtils.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Utils

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineDialect
  MLIRArithmeticDialect
  MLIRMemRefDialect
  MLIRSCFDialect
)