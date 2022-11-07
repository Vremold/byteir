//===- ir_test.cc ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/core/common/status.h"
#include "brt/core/ir/ir.h"
#include "brt/test/common/models.h"
#include "brt/test/common/util.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "gtest/gtest.h"
#include <string>

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;
using namespace mlir;

TEST(IRTest, IterateNode) {
  ByREHandle hdl;
  auto status_init = hdl.Initialize();
  BRT_TEST_CHECK_STATUS(status_init);

  ByREBuilder byre_builder;
  auto status_load =
      hdl.LoadFromMemory(CreateAddOp2(byre_builder, "cpu"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  Status status_iterate_internal;

  auto status_iterate_final = hdl.IterateNode([&](Operation *op) {
    if (auto byre_op = dyn_cast<byre::ByreOp>(op)) {
      auto key = ByREHandle::GetKey(byre_op);
      if (key != "AddOpf32f32f32") {
        status_iterate_internal =
            Status(BRT, FAIL, "Expect get AddOpf32f32f32 but get " + key);
        return WalkResult::interrupt();
      }
      for (auto opArg : byre_op->getOperands()) {
        if (opArg.getAsOpaquePointer() == nullptr) {
          status_iterate_internal =
              Status(BRT, FAIL, "IterateNode get a null arg");
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  BRT_TEST_CHECK_STATUS(status_iterate_final);
  BRT_TEST_CHECK_STATUS(status_iterate_internal);
}

TEST(IRTest, IterateNodeWithInterrupt) {
  ByREHandle hdl;
  auto status_init = hdl.Initialize();
  BRT_TEST_CHECK_STATUS(status_init);

  ByREBuilder byre_builder;
  auto status_load =
      hdl.LoadFromMemory(CreateUnknown(byre_builder, "cpu"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  Status status_iterate_internal;

  auto status_iterate_final = hdl.IterateNode([&](Operation *op) {
    if (auto byre_op = dyn_cast<byre::ByreOp>(op)) {
      const std::string key = ByREHandle::GetKey(byre_op);
      if (key == "UnknownOp") {
        status_iterate_internal =
            Status(BRT, FAIL, "IterateNode gets UnknownOp");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  EXPECT_FALSE(status_iterate_final.IsOK());
  EXPECT_FALSE(status_iterate_internal.IsOK());
}

TEST(IRTest, IterateEntryFuncArg) {
  ByREHandle hdl;
  auto status_init = hdl.Initialize();
  BRT_TEST_CHECK_STATUS(status_init);

  ByREBuilder byre_builder;
  auto status_load =
      hdl.LoadFromMemory(CreateAddOp2(byre_builder, "cpu"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  Status status_iterate_internal;
  auto status_iterate =
      hdl.IterateEntryFuncArg([&](mlir::BlockArgument block_arg) {
        if (block_arg.getAsOpaquePointer() == nullptr) {
          return Status(BRT, FAIL, "IterateNode get a null arg");
        }
        return Status::OK();
      });
  BRT_TEST_CHECK_STATUS(status_iterate);
}
