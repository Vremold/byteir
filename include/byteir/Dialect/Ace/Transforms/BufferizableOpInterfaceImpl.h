//===- BufferizableOpInterfaceImpl.h --------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_ACE_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
#define BYTEIR_DIALECT_ACE_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
class DialectRegistry;

namespace ace {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace ace
} // namespace mlir

#endif // BYTEIR_DIALECT_ACE_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
