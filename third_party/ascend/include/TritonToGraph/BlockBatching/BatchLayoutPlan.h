// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_LAYOUT_PLAN_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_LAYOUT_PLAN_H
#include "TritonToGraph/BlockBatching/BatchAccessAnalysis.h"
#include "TritonToGraph/BlockBatching/BatchLayoutFacts.h"
namespace mlir::triton::cfg {
SmallVector<BatchLayoutResult, 0>
initialBatchLayoutResults(ArrayRef<BatchLayoutPlan> plans);
std::string serializeBatchLayoutResults(ArrayRef<BatchLayoutResult> results);

SmallVector<BatchLayoutPlan, 0>
planBatchLayouts(const BatchAccessReport &accesses, int axis, int64_t factor);
// Read-only proof for automatic axis selection. Uses the same closed-region
// check as the memory rewrite and requires unit-stride column accesses.
bool hasBatchColumnPointwiseRegion(triton::FuncOp function,
                                   const BatchAccessReport &accesses, int axis,
                                   int64_t factor);

std::string serializeBatchLayouts(ArrayRef<BatchLayoutPlan> plans);
LogicalResult
rewriteBatchMemoryViews(ModuleOp module, const BatchAccessReport &accesses,
                        ArrayRef<BatchLayoutPlan> plans,
                        SmallVectorImpl<BatchLayoutResult> &results,
                        ArrayRef<Operation *> semanticAccesses, Value group,
                        unsigned originalArgCount, int axis, int64_t factor);
} // namespace mlir::triton::cfg
#endif
