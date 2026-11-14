// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_PLAN_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_PLAN_H
#include "TritonToGraph/BlockBatching/BatchFacts.h"
#include "TritonToGraph/BlockBatching/BatchLayoutFacts.h"
#include <memory>
#include <optional>

namespace mlir::triton::cfg {
enum class BatchSchedule { ProgramLanes, SequentialIterations };
// Strategy describes lowering semantics, not a second class hierarchy.
enum class BatchStrategy {
  VectorLanes,
  SequentialLoop,
  MatrixStaticLanes,
  MatrixSpatial,
  MatrixBatch
};

enum class BatchEffectRule {
  ReplicateEffect,
  PreserveVolatile,
  ShareInvariantLoad,
  SupportedEffect
};
inline const char *batchEffectRuleName(BatchEffectRule rule) {
  switch (rule) {
  case BatchEffectRule::ReplicateEffect:
    return "ReplicateEffect";
  case BatchEffectRule::PreserveVolatile:
    return "PreserveVolatile";
  case BatchEffectRule::ShareInvariantLoad:
    return "ShareInvariantLoad";
  case BatchEffectRule::SupportedEffect:
    return "SupportedEffect";
  }
  return nullptr;
}
enum class BatchEffectProof { ByteDomain, LaneCorrespondence };
struct BatchEffectObligation {
  unsigned accessID;
  BatchEffectRule rule;
  BatchEffectProof proof;
};
// Owned planning output. These are obligations, not proofs of runtime inputs.
struct BatchProofObligations {
  llvm::SmallVector<BatchEffectObligation, 0> effects;
  llvm::SmallVector<unsigned, 0> terminalReads;
  bool validFor(const BatchFacts &facts) const {
    if (effects.size() != facts.accesses().size())
      return false;
    for (unsigned id = 0; id < effects.size(); ++id)
      if (effects[id].accessID != id ||
          !batchEffectRuleName(effects[id].rule) ||
          (effects[id].proof != BatchEffectProof::ByteDomain &&
           effects[id].proof != BatchEffectProof::LaneCorrespondence))
        return false;
    for (unsigned id : terminalReads)
      if (id >= effects.size())
        return false;
    return true;
  }
};

// An accepted decision owns immutable original-domain facts and layout plans.
// Grid and hidden ABI are derived from schedule, never independently set flags.
class BatchPlan {
public:
  static std::optional<BatchPlan>
  programLanes(std::shared_ptr<const BatchFacts> facts, int axis,
               int64_t factor, llvm::SmallVector<BatchLayoutPlan, 0> layouts,
               BatchProofObligations proof) {
    if (!validRequest(facts, axis, factor) ||
        layouts.size() != facts->accesses().size() || !proof.validFor(*facts))
      return std::nullopt;
    for (unsigned id = 0; id < layouts.size(); ++id)
      if (layouts[id].accessID != id)
        return std::nullopt;
    return BatchPlan(std::move(facts), axis, factor,
                     BatchSchedule::ProgramLanes, BatchStrategy::VectorLanes,
                     {}, std::move(layouts), std::move(proof));
  }
  static std::optional<BatchPlan>
  sequentialIterations(std::shared_ptr<const BatchFacts> facts, int axis,
                       int64_t factor, unsigned loopID,
                       BatchProofObligations proof) {
    if (!validRequest(facts, axis, factor) || !facts->findLoop(loopID) ||
        !proof.validFor(*facts))
      return std::nullopt;
    return BatchPlan(
        std::move(facts), axis, factor, BatchSchedule::SequentialIterations,
        BatchStrategy::SequentialLoop, loopID, {}, std::move(proof));
  }
  static std::optional<BatchPlan>
  matrixLanes(std::shared_ptr<const BatchFacts> facts, int axis, int64_t factor,
              BatchStrategy strategy,
              llvm::SmallVector<BatchLayoutPlan, 0> layouts,
              BatchProofObligations proof) {
    if (!validRequest(facts, axis, factor) || !proof.validFor(*facts) ||
        (strategy != BatchStrategy::MatrixStaticLanes &&
         strategy != BatchStrategy::MatrixSpatial &&
         strategy != BatchStrategy::MatrixBatch))
      return std::nullopt;
    if (strategy == BatchStrategy::MatrixBatch) {
      if (layouts.size() != facts->accesses().size())
        return std::nullopt;
      for (unsigned id = 0; id < layouts.size(); ++id)
        if (layouts[id].accessID != id)
          return std::nullopt;
    } else if (!layouts.empty())
      return std::nullopt;
    return BatchPlan(std::move(facts), axis, factor,
                     BatchSchedule::ProgramLanes, strategy, {},
                     std::move(layouts), std::move(proof));
  }
  BatchStrategy strategy() const { return selectedStrategy; }
  bool isMatrix() const {
    return selectedStrategy == BatchStrategy::MatrixStaticLanes ||
           selectedStrategy == BatchStrategy::MatrixSpatial ||
           selectedStrategy == BatchStrategy::MatrixBatch;
  }
  const BatchFacts &facts() const { return *sourceFacts; }
  int axis() const { return selectedAxis; }
  int64_t factor() const { return laneWidth; }
  BatchSchedule schedule() const { return selectedSchedule; }
  bool preservesGrid() const {
    return selectedSchedule == BatchSchedule::SequentialIterations;
  }
  unsigned hiddenExtentCount() const { return preservesGrid() ? 0 : 3; }
  const BatchLoopFact *groupedLoop() const {
    return loopID ? sourceFacts->findLoop(*loopID) : nullptr;
  }
  llvm::ArrayRef<BatchLayoutPlan> layouts() const { return layoutPlans; }
  const BatchProofObligations &proof() const { return proofObligations; }

private:
  static bool validRequest(const std::shared_ptr<const BatchFacts> &facts,
                           int axis, int64_t factor) {
    return facts && axis >= 0 && axis < 3 && factor > 1 && factor <= INT32_MAX;
  }
  BatchPlan(std::shared_ptr<const BatchFacts> facts, int axis, int64_t factor,
            BatchSchedule schedule, BatchStrategy strategy,
            std::optional<unsigned> loop,
            llvm::SmallVector<BatchLayoutPlan, 0> layouts,
            BatchProofObligations proof)
      : sourceFacts(std::move(facts)), selectedAxis(axis), laneWidth(factor),
        selectedSchedule(schedule), selectedStrategy(strategy), loopID(loop),
        layoutPlans(std::move(layouts)), proofObligations(std::move(proof)) {}
  std::shared_ptr<const BatchFacts> sourceFacts;
  int selectedAxis;
  int64_t laneWidth;
  BatchSchedule selectedSchedule;
  BatchStrategy selectedStrategy;
  std::optional<unsigned> loopID;
  llvm::SmallVector<BatchLayoutPlan, 0> layoutPlans;
  BatchProofObligations proofObligations;
};
} // namespace mlir::triton::cfg
#endif
