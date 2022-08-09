//===- ShapeAnalysis.h ----------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_ANALYSIS_SHAPEANALYSIS_H
#define BYTEIR_ANALYSIS_SHAPEANALYSIS_H

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace shape_analysis {
/// Statically known information for a particular Value.
///
/// This struct currently tracks only information relevant for tensor/array-like
/// shaped types. It is fine to associate a `ValueKnowledge` with a non-shaped
/// type as long as it is in the default "no knowledge" state returned by
/// `getPessimisticValueState`. The important invariant is that we cannot
/// claim to know something about a value which is false.
///
/// This class could also be called "dataflow facts", "lattice value", etc.
struct ValueKnowledge {
  ValueKnowledge() = delete;
  ValueKnowledge(bool hasRank, llvm::ArrayRef<int64_t> newSizes, Type dtype)
      : hasError(false), hasRank(hasRank), dtype(dtype) {
    sizes.reserve(newSizes.size());
    for (auto size : newSizes)
      sizes.push_back(size);
  }

  operator bool() const { return !hasError; }

  // Get the static knowledge intrinsic to `type`.
  static ValueKnowledge getKnowledgeFromType(Type type);

  // Return a pessimistic/conservative value state without assuming any knowlege
  // about the IR.
  static ValueKnowledge getPessimisticValueState();

  static ValueKnowledge getPessimisticValueState(Value value);

  ShapedTypeComponents getShapedTypeComponents() const {
    return hasRank ? ShapedTypeComponents(sizes) : ShapedTypeComponents();
  }

  Type getType() const {
    if (hasRank)
      return RankedTensorType::get(llvm::makeArrayRef(sizes), dtype);
    return UnrankedTensorType::get(dtype);
  }

  bool operator==(const ValueKnowledge &rhs) const {
    return hasRank == rhs.hasRank && sizes == rhs.sizes && dtype == rhs.dtype;
  }

  static ValueKnowledge join(const ValueKnowledge &lhs,
                             const ValueKnowledge &rhs);

  static ValueKnowledge meet(const ValueKnowledge &lhs,
                             const ValueKnowledge &rhs);

  void print(raw_ostream &os) const;

  // Whether the value information has an error.
  bool hasError;
  // Whether the value has known rank.
  bool hasRank;
  // If `hasRank`, the sizes along each rank. Unknown sizes are represented as
  // `ShapedType::kDynamicSize`.
  llvm::SmallVector<int64_t> sizes;
  // The dtype of a tensor.
  // This is equal to nullptr if we don't know that it is a specific concrete
  // type.
  Type dtype;
};

struct ValueTypeModificatoinRAII {
  ~ValueTypeModificatoinRAII() {
    for (auto &&pi : toRestore) {
      std::get<0>(pi).setType(std::get<1>(pi));
    }
  }
  void Push(Value value, Type type) {
    Type originType = value.getType();
    value.setType(type);
    toRestore.emplace_back(value, originType);
  }
  SmallVector<std::pair<Value, Type>> toRestore;
};
} // namespace shape_analysis

using ShapeLattice = dataflow::Lattice<shape_analysis::ValueKnowledge>;

class ShapeAnalysis : public dataflow::SparseDataFlowAnalysis<ShapeLattice> {
public:
  using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

  void visitOperation(Operation *op, ArrayRef<const ShapeLattice *> operands,
                      ArrayRef<ShapeLattice *> results) override;

protected:
  using ShapeKnowledges = function_ref<Type(Value)>;
  using ShapeValueKnowledges = function_ref<Attribute(Value)>;

  virtual LogicalResult inferResultShapesWithKnowledges(
      Operation *op, ShapeKnowledges shapeKnowledges,
      ShapeValueKnowledges shapeValueKnowledges,
      llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &results);
};

// FIXME: strictly speaking ShapeValueLattice should be a 1-d tensor which could
// be inferred partially, here we use the same Lattice as the CPA did, so once
// the state of the Lattice is mutated the subscribed CPA would be triggered
using ShapeValueLattice = dataflow::Lattice<dataflow::ConstantValue>;

// derived from SCP but override for some operation which could be folded with
// operand type instead of operand value, for such operation it should not mark
// them as pessimistic fixpoint when fold failed with given operand value, it
// might be updated once operand type is inferred
class ShapeValueAnalysis : public dataflow::SparseConstantPropagation {
public:
  using SparseConstantPropagation::SparseConstantPropagation;

  void visitOperation(Operation *op,
                      ArrayRef<const ShapeValueLattice *> operands,
                      ArrayRef<ShapeValueLattice *> results) override;

protected:
  // very similar to SparseConstantPropagation but fold \p op with given
  // inferred operand shape which is stored in \p ShapeLattices
  virtual void visitOperation(Operation *op,
                              ArrayRef<const ShapeValueLattice *> operands,
                              ArrayRef<const ShapeLattice *> ShapeLattices,
                              ArrayRef<ShapeValueLattice *> results);
};

} // namespace mlir

#endif // BYTEIR_ANALYSIS_SHAPEANALYSIS_H
