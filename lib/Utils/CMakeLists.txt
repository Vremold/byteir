add_mlir_library(ByteIRUtils
  IRRewrite.cpp
  Utils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRef
)