// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_REWRITE_FINALIZER_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_REWRITE_FINALIZER_H
#include "TritonToGraph/BlockBatching/BatchAccessAnalysis.h"
#include "TritonToGraph/BlockBatching/BatchRewriteResult.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
namespace mlir::triton::cfg {
llvm::SmallVector<BatchEffectSource, 0>
captureBatchEffectSources(const BatchSourceIndex &sources,
                          scf::ForOp groupedLoop);
void markBatchEffects(ArrayRef<Operation *> effects);
void copyBatchEffectIdentity(Operation *source, Operation *target);
class BatchRewriteFinalizer {
public:
  static std::optional<BatchAppliedRewrite>
  programLanes(ModuleOp module, const BatchPlan &plan,
               ArrayRef<BatchEffectSource> sources, unsigned originalArgCount,
               ArrayRef<BatchLayoutResult> layouts);
  static std::optional<BatchAppliedRewrite>
  sequentialIterations(ModuleOp module, const BatchPlan &plan,
                       ArrayRef<BatchEffectSource> sources,
                       unsigned originalArgCount);

  // The existing matrix lowerers establish their own structured legality.
  // Validate their common output ABI without inventing effect provenance.
  static std::optional<BatchAppliedRewrite>
  matrixLanes(ModuleOp module, const BatchPlan &plan, unsigned originalArgCount,
              ArrayRef<BatchLayoutResult> layouts = {});

private:
  static std::optional<BatchAppliedRewrite>
  finalize(ModuleOp module, const BatchPlan &plan,
           ArrayRef<BatchEffectSource> sources, unsigned originalArgCount,
           ArrayRef<BatchLayoutResult> layouts, bool sequential);
};
} // namespace mlir::triton::cfg
#endif
