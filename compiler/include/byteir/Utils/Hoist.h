//===- Hoist.h ------------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_UTILS_HOIST_H
#define BYTEIR_UTILS_HOIST_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include <functional>

namespace mlir {
class Value;
class Block;
class DominanceInfo;
class FunctionOpInterface;
class Operation;
class PostDominanceInfo;

/// return least ProperlyDominant use or def.
/// aka return the last def or use berfore refOp
/// Note: val must be one of refOp's operands
/// Case 1
///```mlir
///  val = def
///  refOp(val)
/// ```
///  return def
///
/// Case 2
///```mlir
///  val = def
///  anotherUser1(val)
///  anotherUser2(val)
///  refOp(val)
///  anotherUser3(val)
/// ```
///  return anotherUser2
Operation *leastProperlyDominantUseOrDef(Value val, DominanceInfo &domInfo,
                                         Operation *refOp);

/// return least ProperlyPostDominant use
/// aka return the last use after refOp
/// Note: val must be one of refOp's operands or results
/// Case 1
///```mlir
///  val = refOp(...)
///  user1(val)
///  user2(val)
/// ```
///  return user1
Operation *leastProperlyPostDominantUse(Value val,
                                        PostDominanceInfo &postDomInfo,
                                        Operation *refOp);

/// return least ProperlyDominant among a set of Operations
/// aka return the first-defined op in ops
/// Case 1
///```mlir
///  op1
///  op2
///  op3
/// ```
///  ops = {op3, op1, op2}
///  return op1
Operation *leastProperlyDominantOp(ArrayRef<Operation *> ops,
                                   DominanceInfo &domInfo);

/// return least ProperlyPostDominant among a set of Operations
/// aka return the last-defined op in ops
/// Case 1
///```mlir
///  op1
///  op2
///  op3
/// ```
///  ops = {op3, op1, op2}
///  return op3
Operation *leastProperlyPostDominantOp(ArrayRef<Operation *> ops,
                                       PostDominanceInfo &postDomInfo);

// return Operation Hoist Up within a Block of op
Operation *findHoistUpInBlock(Operation *op, DominanceInfo &domInfo);

// return Operation Hoist Down within a Block of op
Operation *findHoistDownInBlock(Operation *op, PostDominanceInfo &postDomInfo);

// hoist up an op in its Block
void hoistUpOpInBlock(Operation *op, DominanceInfo &domInfo);

// hoist down an op in its Block
void hoistDownOpInBlock(Operation *op, PostDominanceInfo &postDomInfo);

// hoist up ops in a given Block
void hoistUpOpsInBlock(Block *block, DominanceInfo &domInfo,
                       std::function<bool(Operation *)> checkFunc);

// hoist up ops in a given FunctionOpInterface
// Note it performs DominanceInfo internally
void hoistUpOpsInFuncLike(FunctionOpInterface funclike,
                          std::function<bool(Operation *)> checkFunc);

// hoist down ops in a given Block
void hoistDownOpsInBlock(Block *block, PostDominanceInfo &postDomInfo,
                         std::function<bool(Operation *)> checkFunc);

// hoist down ops in a given FunctionOpInterface
// Note it performs DominanceInfo internally
void hoistDownOpsInFuncLike(FunctionOpInterface funclike,
                            std::function<bool(Operation *)> checkFunc);

/// hoist up op and responding defs to target
/// ```mlir
/// otherOp
/// target
/// val1 = def1()
/// val2 = def2(val1)
/// val3 = def3()
/// op(val2, val3)
/// ```
///
/// into
///
/// ```mlir
/// otherOp
/// val1 = def1()
/// val2 = def2(val1)
/// val3 = def3()
/// op(val2, val3)
/// target
/// ```
void hoistUpOpAndDefs(Operation *op, Operation *target, DominanceInfo &domInfo);

/// hoist down op and responding users to target
/// ```mlir
/// val1 = op(...)
/// val2 = user1(val1)
/// val3 = user2(val2)
/// target
/// val3 = user2(val1)
/// ```
///
/// into
///
/// ```mlir
/// target
/// val1 = op(...)
/// val2 = user1(val1)
/// val3 = user2(val2)
/// val3 = user2(val1)
/// ```
void hoistDownOpAndUsers(Operation *op, Operation *target,
                         PostDominanceInfo &postDomInfo);

void hoistDownDescendantUsers(Value val, PostDominanceInfo &postDomInfo);

} // namespace mlir

#endif // BYTEIR_UTILS_HOIST_H
