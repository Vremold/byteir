add_mlir_library(ByteIRUtils
  Hoist.cpp
  IRRewrite.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Utils

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRef
)