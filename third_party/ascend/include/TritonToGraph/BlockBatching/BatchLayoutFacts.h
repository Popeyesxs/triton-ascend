// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_LAYOUT_FACTS_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_LAYOUT_FACTS_H
#include "TritonToGraph/BlockBatching/BatchExpr.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
namespace mlir::triton::cfg {
using llvm::SmallVector;
inline constexpr unsigned kBatchLayoutPlannerVersion = 2;
struct BatchLayoutTerm {
  BatchExpr::Kind kind;
  int64_t id;
  BatchExprRef coefficient;
  int64_t divisor = 0;
};
struct BatchLayoutPlan {
  enum Action { Keep, MemoryView, PointwiseRegion };
  Action action = Keep;
  unsigned accessID = 0;
  int gridAxis = -1;
  SmallVector<int64_t> localAxes, shape;
  int64_t tileExtent = 0, origin = 0;
  // Uniform integer expressions, independent of all program/local coordinates.
  // Owned DAGs survive the original IR, unlike an MLIR Value captured here.
  BatchExprRef elementOrigin, elementStride;
  bool symbolic = false;
  bool linearMemoryView = false;
  SmallVector<BatchLayoutTerm> restTerms;
  SmallVector<BatchExprRef> residualMasks;
  std::string reason = "DIRECTION_UNPROVEN", maskClass = "unknown";
  std::string location;
  BatchExprRef coordinate, upper;
  bool controlFlowIndependent = false;
};
// An owned outcome, never an IR handle. Layout attempts may downgrade this
// copy to Keep without changing the accepted source plan.
struct BatchLayoutResult {
  BatchLayoutPlan layout;
  bool rewritten = false;
  std::string runtimeGuardJSON;
};
} // namespace mlir::triton::cfg
#endif
