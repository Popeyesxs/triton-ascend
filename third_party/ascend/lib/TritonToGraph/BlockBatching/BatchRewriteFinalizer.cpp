// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchRewriteFinalizer.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "mlir/IR/Verifier.h"
namespace mlir::triton::cfg {
namespace {
constexpr llvm::StringLiteral effectID = "__vv_effect_id";
BatchEffectSource policy(Operation *op, bool grouped) {
  if (auto load = dyn_cast<triton::LoadOp>(op))
    return {false, load.getIsVolatile(), grouped,
            static_cast<int64_t>(load.getCache()),
            static_cast<int64_t>(load.getEvict())};
  auto store = cast<triton::StoreOp>(op);
  return {true, false, grouped, static_cast<int64_t>(store.getCache()),
          static_cast<int64_t>(store.getEvict())};
}
} // namespace
llvm::SmallVector<BatchEffectSource, 0>
captureBatchEffectSources(const BatchSourceIndex &sources,
                          scf::ForOp groupedLoop) {
  llvm::SmallVector<BatchEffectSource, 0> result;
  for (Operation *op : sources.accesses)
    result.push_back(
        policy(op, groupedLoop && groupedLoop->isProperAncestor(op)));
  return result;
}
void markBatchEffects(ArrayRef<Operation *> effects) {
  for (auto [id, op] : llvm::enumerate(effects)) {
    assert(op && "missing generated VV effect");
    op->setAttr(effectID, Builder(op->getContext()).getI64IntegerAttr(id));
  }
}
void copyBatchEffectIdentity(Operation *source, Operation *target) {
  if (auto identity = source->getAttr(effectID))
    target->setAttr(effectID, identity);
}
std::optional<BatchAppliedRewrite> BatchRewriteFinalizer::programLanes(
    ModuleOp module, const BatchPlan &plan, ArrayRef<BatchEffectSource> sources,
    unsigned originalArgCount, ArrayRef<BatchLayoutResult> layouts) {
  if (plan.isMatrix() || plan.schedule() != BatchSchedule::ProgramLanes)
    return std::nullopt;
  return finalize(module, plan, sources, originalArgCount, layouts, false);
}
std::optional<BatchAppliedRewrite> BatchRewriteFinalizer::sequentialIterations(
    ModuleOp module, const BatchPlan &plan, ArrayRef<BatchEffectSource> sources,
    unsigned originalArgCount) {
  if (plan.schedule() != BatchSchedule::SequentialIterations)
    return std::nullopt;
  return finalize(module, plan, sources, originalArgCount, {}, true);
}
std::optional<BatchAppliedRewrite> BatchRewriteFinalizer::finalize(
    ModuleOp module, const BatchPlan &plan, ArrayRef<BatchEffectSource> sources,
    unsigned originalArgCount, ArrayRef<BatchLayoutResult> layouts,
    bool sequential) {
  auto fail = [&]() -> std::optional<BatchAppliedRewrite> {
    module.emitError("invalid finalized VV effect or ABI correspondence");
    return std::nullopt;
  };
  triton::FuncOp function;
  for (auto candidate : module.getOps<triton::FuncOp>())
    if (candidate.isPublic()) {
      if (function)
        return fail();
      function = candidate;
    }
  if (!function || sources.size() != plan.facts().accesses().size() ||
      function.getNumArguments() != originalArgCount + plan.hiddenExtentCount())
    return fail();
  auto grid = module->getAttr(kProgramGridTransformsAttr);
  if (sequential) {
    if (grid || !layouts.empty())
      return fail();
  } else {
    if (!hasProgramGridHiddenExtentArguments(function, 3) || !grid ||
        layouts.size() != sources.size())
      return fail();
    auto contract = parseProgramGridTransformContract(grid);
    if (failed(contract) || contract->version != 3 ||
        contract->transforms.size() != 1)
      return fail();
    const auto &transform = contract->transforms.front();
    if (transform.axis != plan.axis() || transform.factor != plan.factor() ||
        transform.order != 0 || transform.persistentCoverage ||
        transform.gridStrideAbiVerified)
      return fail();
    for (unsigned id = 0; id < layouts.size(); ++id)
      if (layouts[id].layout.accessID != id)
        return fail();
  }
  llvm::SmallVector<BatchMappedEffect, 0> mappings;
  for (unsigned id = 0; id < sources.size(); ++id)
    mappings.push_back({id, {}});
  bool valid = true;
  unsigned targetID = 0;
  function.walk([&](Operation *op) {
    if (!isa<triton::LoadOp, triton::StoreOp>(op))
      return;
    auto id = op->getAttrOfType<IntegerAttr>(effectID);
    op->removeAttr(effectID);
    if (!id || id.getInt() < 0 ||
        static_cast<size_t>(id.getInt()) >= sources.size()) {
      valid = false;
      return;
    }
    auto actual = policy(op, false);
    const auto &source = sources[id.getInt()];
    if (actual.write != source.write ||
        actual.isVolatile != source.isVolatile ||
        actual.cache != source.cache || actual.evict != source.evict) {
      valid = false;
      return;
    }
    mappings[id.getInt()].generatedIDs.push_back(targetID++);
  });
  if (!valid)
    return fail();
  for (unsigned id = 0; id < mappings.size(); ++id) {
    size_t count = mappings[id].generatedIDs.size();
    // The unrolled main body contains F ordered copies. A residual loop adds
    // one static copy; effects outside the grouped loop remain singletons.
    if (sequential && sources[id].grouped) {
      if (count != static_cast<size_t>(plan.factor()) &&
          count != static_cast<size_t>(plan.factor() + 1))
        return fail();
    } else if (count != 1)
      return fail();
  }
  if (failed(verify(module)))
    return fail();
  return BatchAppliedRewrite(plan, std::move(mappings), layouts);
}
std::optional<BatchAppliedRewrite>
BatchRewriteFinalizer::matrixLanes(ModuleOp module, const BatchPlan &plan,
                                   unsigned originalArgCount,
                                   ArrayRef<BatchLayoutResult> layouts) {
  if (!plan.isMatrix() || plan.preservesGrid() ||
      layouts.size() != plan.layouts().size())
    return std::nullopt;
  triton::FuncOp function;
  for (auto entry : module.getOps<triton::FuncOp>())
    if (entry.isPublic()) {
      if (function)
        return std::nullopt;
      function = entry;
    }
  if (!function || function.getNumArguments() != originalArgCount + 3 ||
      !hasProgramGridHiddenExtentArguments(function, 3))
    return std::nullopt;
  auto contract = parseProgramGridTransformContract(
      module->getAttr(kProgramGridTransformsAttr));
  if (failed(contract) || contract->version != 3 ||
      contract->transforms.size() != 1)
    return std::nullopt;
  const auto &transform = contract->transforms.front();
  if (transform.axis != plan.axis() || transform.factor != plan.factor() ||
      transform.order != 0 || transform.persistentCoverage ||
      transform.gridStrideAbiVerified)
    return std::nullopt;
  for (unsigned id = 0; id < layouts.size(); ++id)
    if (layouts[id].layout.accessID != id)
      return std::nullopt;
  if (failed(verify(module)))
    return std::nullopt;
  return BatchAppliedRewrite(plan, {}, layouts,
                             BatchVerification::StructuredMatrix);
}
} // namespace mlir::triton::cfg
