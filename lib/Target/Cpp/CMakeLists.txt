add_byteir_translation_library(ByteIRTargetCpp
  TranslateRegistration.cpp
  TranslateToCpp.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Target/Cpp

  LINK_LIBS PUBLIC
  MLIREmitC
  MLIRIR
  MLIRSCF
  MLIRStandard
  MLIRMemRef
  MLIRSupport
  MLIRTranslation
  )
