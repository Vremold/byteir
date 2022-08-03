//===- GraphClusteringByDevice.h ------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_GRAPH_CLUSTERING_BY_DEVICE_H
#define BYTEIR_TRANSFORMS_GRAPH_CLUSTERING_BY_DEVICE_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;

constexpr StringRef getHostAnchorName() { return "__byteir_host_device__"; }

// Currently the usage of the pass is limited and it may not work correctly in
// non-tensor level dialects. Before this pass, user need to add `device = host`
// attribute to those operations that could only be run on host. Then this pass
// will cluster the host ops and their recursive producers into a host function,
// the other ops will be clustered into a device function.
std::unique_ptr<OperationPass<ModuleOp>> createGraphClusteringByDevicePass(
    std::string attrName = "device", std::string device = "test",
    std::string deviceAnchorName = "__byteir_test_device__",
    bool dupNonSplat = false);

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_GRAPH_CLUSTERING_BY_DEVICE_H