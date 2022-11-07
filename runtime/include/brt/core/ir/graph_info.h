//===- graph_info.h -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace brt {
namespace ir {

/**
 * GraphInfo holds the graph's weights, io, and basic meta for runtime
 * It is created by IRHandle.
 *
 * Note GraphInfo's lifetime extends in the entire evaluation,
 * compared to IRHandle's lifetime that might be destroyed after planning.
 *
 */

struct GraphInfo {
  // Note IR ptr is used as an unique key
  // In MLIR, it is mlir::Value's raw ptr.
  using Key = void *;

  // map a IR ptr to into a unique tensor id
  // including weights/inputs/outputs and intermediates
  std::unordered_map<Key, size_t> tensor_to_id;

  // store all IR ptrs of tensors,
  // including weights/inputs/outputs and intermediates
  std::vector<Key> tensors;

  // map a IR ptr to into a unique scalar id
  // TODO: only intermediates was supported
  std::unordered_map<Key, size_t> scalar_to_id;

  // store all IR ptrs of scalars,
  // TODO: only intermediates was supported
  std::vector<Key> scalars;

  // count of weights
  size_t weight_count;

  // count count of input and output, not including weight
  size_t io_count;

  // arg alias handling
  std::vector<std::pair<size_t, size_t>> arg_alias_to_id_and_offset;

  // map a name, as a string, to an arg offset
  std::unordered_map<std::string, size_t> name_to_arg_offset;

  // meta data for names
  std::vector<std::string> weight_names;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  // meta data for arg offsets
  std::vector<size_t> weight_arg_offsets;
  std::vector<size_t> input_arg_offsets;
  std::vector<size_t> output_arg_offsets;

  inline int GetGraphArgOffset(const std::string &name) {
    auto found = name_to_arg_offset.find(name);
    if (found == name_to_arg_offset.end()) {
      return -1;
    }
    return static_cast<int>(found->second);
  }

  inline size_t GetArgNum() { return weight_count + io_count; }
};

} // namespace ir
} // namespace brt
