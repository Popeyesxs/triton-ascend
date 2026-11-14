// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_REWRITE_RESULT_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_REWRITE_RESULT_H
#include "TritonToGraph/BlockBatching/BatchPlan.h"
namespace mlir::triton::cfg {
struct BatchEffectSource {
  bool write, isVolatile, grouped;
  int64_t cache, evict;
};
struct BatchMappedEffect {
  unsigned accessID;
  llvm::SmallVector<unsigned, 0> generatedIDs;
};
class BatchRewriteFinalizer;
// Record exactly which validation was performed. Matrix lowering retains its
// existing structured proof; it does not claim per-effect provenance checking.
enum class BatchVerification { EffectCorrespondence, StructuredMatrix };
// Only the IR finalizer can create a completed result. No IR handle survives.
class BatchAppliedRewrite {
public:
  const BatchPlan &plan() const { return acceptedPlan; }
  BatchVerification verification() const { return validation; }
  llvm::ArrayRef<BatchMappedEffect> effects() const { return effectMappings; }
  llvm::ArrayRef<BatchLayoutResult> layouts() const { return layoutResults; }

private:
  friend class BatchRewriteFinalizer;
  BatchAppliedRewrite(
      const BatchPlan &plan, llvm::SmallVector<BatchMappedEffect, 0> effects,
      llvm::ArrayRef<BatchLayoutResult> layouts,
      BatchVerification verification = BatchVerification::EffectCorrespondence)
      : acceptedPlan(plan), validation(verification),
        effectMappings(std::move(effects)),
        layoutResults(layouts.begin(), layouts.end()) {}
  BatchPlan acceptedPlan;
  BatchVerification validation;
  llvm::SmallVector<BatchMappedEffect, 0> effectMappings;
  llvm::SmallVector<BatchLayoutResult, 0> layoutResults;
};

// A guard is publishable only after the strategy-specific validation and ABI
// checks recorded by the finalizer. It owns values and the immutable plan,
// never a mutable report.
class BatchGuardSpec {
public:
  static std::optional<BatchGuardSpec>
  fromApplied(const BatchAppliedRewrite &applied) {
    const auto &plan = applied.plan();
    const auto &obligations = plan.proof().effects;
    if (!plan.proof().validFor(plan.facts()))
      return std::nullopt;
    if (plan.isMatrix()) {
      if (applied.verification() != BatchVerification::StructuredMatrix ||
          !applied.effects().empty())
        return std::nullopt;
    } else {
      if (applied.verification() != BatchVerification::EffectCorrespondence ||
          obligations.size() != applied.effects().size())
        return std::nullopt;
      for (unsigned id = 0; id < obligations.size(); ++id)
        if (applied.effects()[id].accessID != id ||
            applied.effects()[id].generatedIDs.empty())
          return std::nullopt;
    }
    return BatchGuardSpec(applied);
  }
  const BatchPlan &plan() const { return result.plan(); }
  const BatchAppliedRewrite &applied() const { return result; }
  llvm::ArrayRef<BatchEffectObligation> obligations() const {
    return result.plan().proof().effects;
  }
  llvm::ArrayRef<unsigned> terminalReads() const {
    return result.plan().proof().terminalReads;
  }

private:
  explicit BatchGuardSpec(const BatchAppliedRewrite &applied)
      : result(applied) {}
  BatchAppliedRewrite result;
};
std::optional<std::string> encodeBatchGuard(const BatchGuardSpec &spec);
} // namespace mlir::triton::cfg
#endif
