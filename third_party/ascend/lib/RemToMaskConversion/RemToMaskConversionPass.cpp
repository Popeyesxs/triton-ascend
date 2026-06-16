/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/RemToMaskConversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_REMTOMASKCONVERSION
#include "ascend/include/RemToMaskConversion/Passes.h.inc"
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "rem-to-mask-conversion"

using namespace mlir;
using namespace triton;

// Check if a Value is derived from tt.make_range (possibly through
// expand_dims, broadcast, splat, extsi chains).
static bool isDerivedFromMakeRange(Value val) {
  if (!val)
    return false;
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return false;

  if (isa<triton::MakeRangeOp>(defOp))
    return true;

  if (isa<arith::ExtSIOp, arith::ExtUIOp, triton::ExpandDimsOp,
          triton::BroadcastOp, triton::SplatOp>(defOp)) {
    return isDerivedFromMakeRange(defOp->getOperand(0));
  }

  return false;
}

// Check if a Value is a scalar (possibly splatted to tensor).
static bool isScalarOrSplat(Value val) {
  if (!val)
    return false;
  if (!isa<RankedTensorType>(val.getType()))
    return true;
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return false;
  if (isa<triton::SplatOp>(defOp))
    return true;
  if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(defOp))
    return isScalarOrSplat(broadcastOp.getSrc());
  return false;
}

// Check if a RemSIOp matches the pattern:
//   (scalar_offset + make_range(0, BLOCK_SIZE)) % Bound
// where scalar_offset is typically pid * BLOCK_SIZE (splatted).
// Returns true if the pattern is matched, and sets `lhsAddOp` and `bound`.
static bool matchRemPattern(arith::RemSIOp remOp, Value &unwrappedOffset,
                            Value &bound) {
  Value lhs = remOp.getLhs();
  Value rhs = remOp.getRhs();

  // RHS must be a scalar or splatted scalar (the Bound: M, N, etc.)
  if (!isScalarOrSplat(rhs))
    return false;

  bound = rhs;

  // LHS should be an add of (scalar_splat + range) or just a range
  if (auto addOp = lhs.getDefiningOp<arith::AddIOp>()) {
    Value addLhs = addOp.getLhs();
    Value addRhs = addOp.getRhs();
    // One side should be derived from make_range, the other from a scalar
    if ((isDerivedFromMakeRange(addLhs) && isScalarOrSplat(addRhs)) ||
        (isDerivedFromMakeRange(addRhs) && isScalarOrSplat(addLhs))) {
      unwrappedOffset = lhs;
      return true;
    }
  }

  // LHS could also be just a make_range (when pid_offset is 0)
  if (isDerivedFromMakeRange(lhs)) {
    unwrappedOffset = lhs;
    return true;
  }

  return false;
}

// Information about how a remsi result reaches a specific load/store,
// capturing the expand_dims axes encountered along the path.
struct MemOpExpansionInfo {
  SmallVector<int> expandAxes;
};

// DFS helper: find a path from `current` to `target` op through the use chain,
// collecting expand_dims axes. Returns true if a path is found.
static bool findPathToOp(Value current, Operation *target,
                         SmallVector<int> &expandAxes,
                         llvm::SmallDenseSet<Value, 16> &visited) {
  if (!visited.insert(current).second)
    return false;

  for (Operation *user : current.getUsers()) {
    if (user == target)
      return true;

    if (auto expandOp = dyn_cast<triton::ExpandDimsOp>(user)) {
      int axis = expandOp.getAxis();
      expandAxes.push_back(axis);
      for (Value result : expandOp->getResults()) {
        if (findPathToOp(result, target, expandAxes, visited))
          return true;
      }
      expandAxes.pop_back();
    } else if (isa<triton::AddPtrOp, arith::AddIOp, arith::MulIOp,
                   arith::ExtSIOp, arith::ExtUIOp, triton::BroadcastOp,
                   triton::SplatOp>(user)) {
      for (Value result : user->getResults()) {
        if (findPathToOp(result, target, expandAxes, visited))
          return true;
      }
    }
  }

  visited.erase(current);
  return false;
}

// Trace from a remsi result to all tt.load / tt.store operations that use it,
// recording the expand_dims axes for each path.
static void traceToMemOps(
    Value remResult,
    SmallVectorImpl<std::pair<triton::LoadOp, MemOpExpansionInfo>> &loads,
    SmallVectorImpl<std::pair<triton::StoreOp, MemOpExpansionInfo>> &stores) {
  // First, collect all reachable loads and stores via simple BFS
  SmallVector<Operation *, 16> targetOps;
  {
    SmallVector<Value, 16> worklist;
    llvm::SmallDenseSet<Value, 16> visited;
    worklist.push_back(remResult);

    while (!worklist.empty()) {
      Value curr = worklist.pop_back_val();
      if (!visited.insert(curr).second)
        continue;

      for (Operation *user : curr.getUsers()) {
        if (isa<triton::LoadOp, triton::StoreOp>(user)) {
          targetOps.push_back(user);
        } else if (isa<triton::AddPtrOp, arith::AddIOp, arith::MulIOp,
                       arith::ExtSIOp, arith::ExtUIOp, triton::ExpandDimsOp,
                       triton::BroadcastOp, triton::SplatOp>(user)) {
          for (Value result : user->getResults())
            worklist.push_back(result);
        }
      }
    }
  }

  // For each target, trace the expand_dims path
  for (Operation *target : targetOps) {
    MemOpExpansionInfo info;
    llvm::SmallDenseSet<Value, 16> visited;
    findPathToOp(remResult, target, info.expandAxes, visited);

    if (auto loadOp = dyn_cast<triton::LoadOp>(target)) {
      loads.push_back({loadOp, info});
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(target)) {
      stores.push_back({storeOp, info});
    }
  }
}

// Check if the remsi result is ONLY used in load/store pointer chains
// (not in computation that requires true wrap-around semantics).
static bool isUsedOnlyInMemoryAccess(arith::RemSIOp remOp) {
  Value remResult = remOp.getResult();
  SmallVector<Value, 16> worklist;
  llvm::SmallDenseSet<Value, 16> visited;
  worklist.push_back(remResult);

  while (!worklist.empty()) {
    Value curr = worklist.pop_back_val();
    if (!visited.insert(curr).second)
      continue;

    for (Operation *user : curr.getUsers()) {
      if (isa<triton::LoadOp, triton::StoreOp>(user)) {
        continue;
      } else if (isa<triton::AddPtrOp, arith::MulIOp, arith::ExtSIOp,
                     arith::ExtUIOp, triton::ExpandDimsOp,
                     triton::BroadcastOp, triton::SplatOp>(user)) {
        for (Value result : user->getResults())
          worklist.push_back(result);
      } else if (auto addOp = dyn_cast<arith::AddIOp>(user)) {
        // AddIOp is allowed only if it feeds into addptr chain
        for (Value result : addOp->getResults())
          worklist.push_back(result);
      } else {
        // Used in non-memory computation - not safe to transform
        LLVM_DEBUG({
          llvm::dbgs() << "RemToMask: remsi result used in non-memory op: "
                       << *user << "\n";
        });
        return false;
      }
    }
  }
  return true;
}

// Generate a comparison mask: offset < Bound
// Handles tensor and scalar types, and broadcasts as needed.
static Value generateBoundaryMask(OpBuilder &builder, Location loc,
                                  Value offset, Value bound) {
  auto offsetType = dyn_cast<RankedTensorType>(offset.getType());
  if (!offsetType) {
    // Scalar case
    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                         offset, bound);
  }

  // Ensure bound is the same tensor type as offset for comparison
  Value boundTensor = bound;
  if (!isa<RankedTensorType>(bound.getType())) {
    // Scalar bound - splat it to match offset shape
    boundTensor =
        builder.create<triton::SplatOp>(loc, offsetType, bound).getResult();
  } else {
    auto boundType = cast<RankedTensorType>(bound.getType());
    if (boundType.getShape() != offsetType.getShape()) {
      // Broadcast bound to match offset shape
      auto targetType = RankedTensorType::get(offsetType.getShape(),
                                              boundType.getElementType());
      boundTensor =
          builder.create<triton::BroadcastOp>(loc, targetType, bound)
              .getResult();
    }
  }

  // Generate integer comparison: offset < bound
  return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, offset,
                                       boundTensor);
}

// Expand a mask to match a load pointer shape using the traced expand_dims
// axes from the original IR. This ensures correct dimension correspondence.
//
// Example: offs_am (1D, dim M) → expand_dims axis=1 → broadcast [M,K]
//   mask should follow the same expansion: [M] → [M,1] → [M,K]
//
// Example: offs_bn (1D, dim N) → expand_dims axis=0 → broadcast [K,N]
//   mask should follow: [N] → [1,N] → [K,N]
static Value expandMaskToMatchPtr(OpBuilder &builder, Location loc,
                                  Value mask, Value ptr,
                                  const MemOpExpansionInfo &expansionInfo) {
  auto maskType = dyn_cast<RankedTensorType>(mask.getType());
  auto ptrType = dyn_cast<RankedTensorType>(ptr.getType());

  if (!maskType || !ptrType)
    return mask;

  if (maskType.getShape() == ptrType.getShape())
    return mask;

  Value result = mask;

  // Apply the same expand_dims sequence as the original offset path
  for (int axis : expansionInfo.expandAxes) {
    result = builder.create<triton::ExpandDimsOp>(loc, result, axis);
  }

  // After expand_dims, broadcast to match the full ptr shape
  auto resultType = cast<RankedTensorType>(result.getType());
  auto ptrShape = ptrType.getShape();

  if (resultType.getRank() == static_cast<int64_t>(ptrShape.size())) {
    SmallVector<int64_t> broadcastShape(ptrShape);
    auto broadcastType =
        RankedTensorType::get(broadcastShape, builder.getI1Type());
    if (resultType.getShape() != broadcastType.getShape()) {
      result = builder.create<triton::BroadcastOp>(loc, broadcastType, result);
    }
  }

  return result;
}

// Combine a new mask with an existing mask (if any) using AND.
static Value combineMasks(OpBuilder &builder, Location loc, Value existingMask,
                          Value newMask) {
  if (!existingMask)
    return newMask;
  if (!newMask)
    return existingMask;

  // Ensure shapes match
  auto existType = dyn_cast<RankedTensorType>(existingMask.getType());
  auto newType = dyn_cast<RankedTensorType>(newMask.getType());
  if (existType && newType && existType.getShape() != newType.getShape()) {
    // Broadcast newMask to match existingMask shape
    newMask = builder.create<triton::BroadcastOp>(loc, existType, newMask)
                  .getResult();
  }

  return builder.create<arith::AndIOp>(loc, existingMask, newMask).getResult();
}

// Rewrite a tt.load to add boundary mask and use unwrapped offset.
static LogicalResult rewriteLoadWithMask(triton::LoadOp loadOp,
                                         Value unwrappedOffset, Value bound,
                                         const MemOpExpansionInfo &info,
                                         IRRewriter &rewriter) {
  // Guard: verify the load op is still alive (not erased by earlier rewrite)
  if (!loadOp->getParentOp())
    return failure();

  auto loc = loadOp.getLoc();
  auto ptr = loadOp.getPtr();
  auto existingMask = loadOp.getMask();
  auto other = loadOp.getOther();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(loadOp);

  Value mask = generateBoundaryMask(rewriter, loc, unwrappedOffset, bound);
  mask = expandMaskToMatchPtr(rewriter, loc, mask, ptr, info);
  Value finalMask = combineMasks(rewriter, loc, existingMask, mask);

  // If no 'other' value provided, use zero (masked-out elements return 0)
  if (!other) {
    auto ptrTensorType = dyn_cast<RankedTensorType>(ptr.getType());
    if (ptrTensorType) {
      auto ptrElemType =
          cast<triton::PointerType>(ptrTensorType.getElementType())
              .getPointeeType();
      auto resultType =
          RankedTensorType::get(ptrTensorType.getShape(), ptrElemType);
      if (isa<FloatType>(ptrElemType)) {
        auto zeroAttr = rewriter.getFloatAttr(ptrElemType, 0.0);
        auto denseZero = DenseElementsAttr::get(resultType, zeroAttr);
        other = rewriter.create<arith::ConstantOp>(loc, denseZero);
      } else if (isa<IntegerType>(ptrElemType)) {
        auto zeroAttr = rewriter.getIntegerAttr(ptrElemType, 0);
        auto denseZero = DenseElementsAttr::get(resultType, zeroAttr);
        other = rewriter.create<arith::ConstantOp>(loc, denseZero);
      }
    }
  }

  auto newLoad = rewriter.create<triton::LoadOp>(
      loc, ptr, finalMask, other, loadOp.getCache(), loadOp.getEvict(),
      loadOp.getIsVolatile());

  rewriter.replaceOp(loadOp, newLoad.getResult());
  return success();
}

// Rewrite a tt.store to add boundary mask.
static LogicalResult rewriteStoreWithMask(triton::StoreOp storeOp,
                                          Value unwrappedOffset, Value bound,
                                          const MemOpExpansionInfo &info,
                                          IRRewriter &rewriter) {
  // Guard: verify the store op is still alive (not erased by earlier rewrite)
  if (!storeOp->getParentOp())
    return failure();

  auto loc = storeOp.getLoc();
  auto ptr = storeOp.getPtr();
  auto existingMask = storeOp.getMask();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(storeOp);

  Value mask = generateBoundaryMask(rewriter, loc, unwrappedOffset, bound);
  mask = expandMaskToMatchPtr(rewriter, loc, mask, ptr, info);
  Value finalMask = combineMasks(rewriter, loc, existingMask, mask);

  auto newStore = rewriter.create<triton::StoreOp>(
      loc, ptr, storeOp.getValue(), finalMask, storeOp.getCache(),
      storeOp.getEvict());

  rewriter.replaceOp(storeOp, newStore);
  return success();
}

void RemToMaskConversionPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<arith::ArithDialect, triton::TritonDialect>();
}

void RemToMaskConversionPass::runOnOperation() {
  auto moduleOp = getOperation();
  bool changed = false;

  // Collect all RemSIOp candidates upfront to avoid iterator invalidation
  SmallVector<arith::RemSIOp> remOps;
  moduleOp.walk([&](arith::RemSIOp remOp) { remOps.push_back(remOp); });

  for (auto remOp : remOps) {
    // Skip if this remOp was already erased by an earlier iteration
    if (!remOp->getParentOp())
      continue;

    Value unwrappedOffset;
    Value bound;

    // Step 1: Pattern matching — is this (scalar + range) % Bound?
    if (!matchRemPattern(remOp, unwrappedOffset, bound))
      continue;

    // Step 2: Safety check — is remsi result only used in memory access?
    if (!isUsedOnlyInMemoryAccess(remOp))
      continue;

    LLVM_DEBUG({
      llvm::dbgs() << "RemToMask: Found candidate remsi: " << remOp << "\n";
      llvm::dbgs() << "  unwrappedOffset: " << unwrappedOffset << "\n";
      llvm::dbgs() << "  bound: " << bound << "\n";
    });

    // Step 3: Trace paths to all load/store ops, collecting expand_dims info.
    // MUST be done BEFORE replaceAllUsesWith, because it traces the
    // original data flow graph from remsi result.
    SmallVector<std::pair<triton::LoadOp, MemOpExpansionInfo>> loads;
    SmallVector<std::pair<triton::StoreOp, MemOpExpansionInfo>> stores;
    traceToMemOps(remOp.getResult(), loads, stores);

    if (loads.empty() && stores.empty())
      continue;

    // Step 4: Replace remsi uses with the unwrapped offset in all pointer
    // chains. After this, addptr operations use the raw (pid*BLOCK + arange)
    // instead of the wrapped (... % M) value.
    remOp.getResult().replaceAllUsesWith(unwrappedOffset);

    // Step 5: Rewrite each load/store to add the boundary mask.
    // The expansion info ensures the mask is expanded along the correct axes.
    IRRewriter rewriter(moduleOp.getContext());

    for (auto &[loadOp, info] : loads) {
      if (succeeded(
              rewriteLoadWithMask(loadOp, unwrappedOffset, bound, info,
                                  rewriter)))
        changed = true;
    }

    for (auto &[storeOp, info] : stores) {
      if (succeeded(
              rewriteStoreWithMask(storeOp, unwrappedOffset, bound, info,
                                   rewriter)))
        changed = true;
    }

    // Step 6: Erase the now-unused remsi operation
    if (remOp.getResult().use_empty())
      rewriter.eraseOp(remOp);

    changed = true;
  }

  if (changed) {
    // Run CSE and canonicalization to clean up redundant ops
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      moduleOp->emitWarning("RemToMaskConversion: cleanup failed");
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "==============================================\n";
    llvm::dbgs() << "After RemToMaskConversionPass:\n" << moduleOp;
    llvm::dbgs() << "\n==============================================\n";
  });
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createRemToMaskConversionPass() {
  return std::make_unique<RemToMaskConversionPass>();
}
