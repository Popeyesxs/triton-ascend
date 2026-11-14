// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_REWRITE_WORKSPACE_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_REWRITE_WORKSPACE_H
#include "TritonToGraph/BlockBatching/BatchAccessAnalysis.h"
#include "TritonToGraph/BlockBatching/BatchPlan.h"
#include "TritonToGraph/BlockBatching/BatchValueMapping.h"
#include "TritonToGraph/BlockBatching/ControlFlowModels.h"

namespace mlir::triton::cfg {
// Transaction-local state only. Shared layout analyzers consume analysis as
// a compatibility view; it is never stored inside the accepted VV plan.
struct BatchRewriteWorkspace {
  // Pending owned values also support legacy refusal diagnostics. Successful
  // execution contracts are encoded only from the finalized accepted plan.
  std::shared_ptr<const BatchFacts> facts;
  std::optional<BatchProofObligations> pendingProof;
  int proofAxis = -1;
  int64_t proofFactor = 1;
  BatchAccessReport analysis;
  BatchControlFlowPlan variation;
  llvm::DenseSet<Operation *> replicatedLoads;
  BatchValueMapping values;
  llvm::DenseMap<Operation *, unsigned> accessIDs;
  SmallVector<Operation *> semanticAccesses;

  void invalidateSource() {
    analysis.sources.clear();
    analysis.operation = nullptr;
    variation.varying.clear();
    variation.reason.clear();
    variation.operation = nullptr;
    replicatedLoads.clear();
    accessIDs.clear();
    values = BatchValueMapping{};
  }
  void invalidateGenerated() { semanticAccesses.clear(); }
};
} // namespace mlir::triton::cfg
#endif
