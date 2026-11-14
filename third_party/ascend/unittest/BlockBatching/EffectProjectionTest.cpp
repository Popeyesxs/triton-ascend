// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Compile and link only the owned facts and production codec, without MLIR.
#include "TritonToGraph/BlockBatching/BatchFactsJSON.h"
#include "TritonToGraph/BlockBatching/BatchPlan.h"
#include "TritonToGraph/BlockBatching/BatchRewriteResult.h"
#include <cstdlib>
#include <iostream>
#include <optional>
#include <type_traits>
#include <utility>

using namespace mlir::triton::cfg;

template <typename T, typename = void> struct HasOperation : std::false_type {};
template <typename T>
struct HasOperation<T, std::void_t<decltype(std::declval<T>().operation)>>
    : std::true_type {};
static_assert(!HasOperation<BatchAccessFact>::value,
              "owned access facts must not retain an IR operation");

static void check(bool ok, const char *message) {
  if (!ok) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

static_assert(!std::is_default_constructible_v<BatchAppliedRewrite>);
static_assert(!std::is_default_constructible_v<BatchGuardSpec>);

int main() {
  BatchEffectProjection projection;
  {
    BatchAccessFact source;
    source.root = 2;
    source.bytes = 4;
    source.write = true;
    source.shape = {8};
    source.byteOffset = BatchExpr::leaf(BatchExpr::Argument, 3, 64);
    source.pointerChecks = {source.byteOffset};
    // The wire order is innermost first, not numeric LoopId order.
    BatchLoopFact inner, outer;
    inner.id = 7;
    outer.id = 2;
    inner.bits = outer.bits = 32;
    inner.lower = outer.lower = BatchExpr::leaf(BatchExpr::Constant, 0, 32);
    inner.upper = outer.upper = BatchExpr::leaf(BatchExpr::Argument, 4, 32);
    inner.step = outer.step = BatchExpr::leaf(BatchExpr::Constant, 1, 32);
    source.loops = {inner, outer};
    projection.accesses = {source, source};
    projection.accesses[1].mask = BatchExpr::leaf(BatchExpr::False);
    check(!source.mask, "effect normalization must not overwrite source facts");
  }
  // The producer's objects have died; expressions and slots remain owned.
  auto wire = batchAccessesJSON(projection);
  auto *object = wire.getAsObject();
  auto *accesses = object->getArray("accesses");
  check(object->getInteger("version") == 2 && accesses->size() == 2,
        "legacy access protocol and member slots must remain intact");
  auto *covered = (*accesses)[1].getAsObject();
  check(covered->getInteger("root") == 2 &&
            covered->getBoolean("write") == true,
        "covered member must preserve its effect identity");
  check(covered->getArray("mask")->front().getAsString() == "false",
        "covered member suppresses only its redundant alias hull");
  check(covered->getArray("pointer_checks")->size() == 1,
        "covered member must still retain original pointer arithmetic");
  auto *loops = covered->getArray("loops");
  check((*loops)[0].getAsObject()->getInteger("id") == 7 &&
            (*loops)[1].getAsObject()->getInteger("id") == 2,
        "inner-to-outer loop order must not change");
  check(!(*loops)[0].getAsObject()->get("carriesValues"),
        "compiler-only facts must not leak into the runtime ABI");
  // Freeze VV facts, then destroy the producer and normalized projection.
  std::optional<BatchFacts> vv;
  {
    auto original = projection.accesses;
    original[1].mask = {};
    llvm::SmallVector<BatchLoopFact, 0> table{original[0].loops[1],
                                              original[0].loops[0]};
    vv.emplace(original, projection.accesses, table);
  }
  projection.accesses.clear();
  check(vv->loops().size() == 2 && vv->accesses().size() == 2,
        "one loop table must serve all original accesses and effects");
  check(vv->accesses()[1].loopIDs[0] == 7 &&
            vv->accesses()[1].loopIDs[1] == 2 &&
            vv->findLoop(7)->upper->value == 4,
        "stable sparse IDs must survive source destruction");
  check(!vv->accesses()[1].mask &&
            vv->effects()[1].mask->kind == BatchExpr::False,
        "guard normalization must not modify original facts");
  check(batchEffectsJSON(*vv) == wire,
        "VV loop-ID facts must retain the exact legacy wire contract");
  auto facts = std::make_shared<const BatchFacts>(*vv);
  llvm::SmallVector<BatchLayoutPlan, 0> layouts(2);
  layouts[0].accessID = 0;
  layouts[1].accessID = 1;
  BatchProofObligations proof;
  proof.effects = {
      {0, BatchEffectRule::SupportedEffect, BatchEffectProof::ByteDomain},
      {1, BatchEffectRule::SupportedEffect, BatchEffectProof::ByteDomain}};
  auto program = BatchPlan::programLanes(facts, 1, 4, layouts, proof);
  auto sequential = BatchPlan::sequentialIterations(facts, 1, 4, 7, proof);
  check(program && sequential,
        "both accepted VV schedules must be representable");
  check(!program->preservesGrid() && program->hiddenExtentCount() == 3 &&
            sequential->preservesGrid() && sequential->hiddenExtentCount() == 0,
        "grid and hidden ABI must be derived from the selected schedule");
  check(!BatchPlan::programLanes(facts, 3, 4, layouts, proof) &&
            !BatchPlan::programLanes(facts, 0, 1, layouts, proof) &&
            !BatchPlan::programLanes(facts, 0, 4, {}, proof) &&
            !BatchPlan::sequentialIterations(facts, 0, 4, 99, proof),
        "invalid axes, factors, layout mappings and loop IDs must be rejected");
  auto wrongProof = proof;
  wrongProof.effects[1].accessID = 0;
  check(!BatchPlan::programLanes(facts, 0, 4, layouts, wrongProof),
        "a plan must reject obligations from a mismatched access mapping");
  wrongProof = proof;
  wrongProof.terminalReads = {2};
  check(!BatchPlan::sequentialIterations(facts, 0, 4, 7, wrongProof),
        "a plan must reject an out-of-range terminal read");
  auto matrix = BatchPlan::matrixLanes(facts, 0, 4, BatchStrategy::MatrixBatch,
                                       layouts, proof);
  auto spatial = BatchPlan::matrixLanes(
      facts, 0, 4, BatchStrategy::MatrixSpatial, {}, proof);
  auto cloned = BatchPlan::matrixLanes(
      facts, 0, 4, BatchStrategy::MatrixStaticLanes, {}, proof);
  check(matrix && spatial && cloned && matrix->isMatrix() &&
            !program->isMatrix() && !spatial->preservesGrid() &&
            cloned->hiddenExtentCount() == 3,
        "matrix and vector decisions must own the same plan and ABI model");
  check(!BatchPlan::matrixLanes(facts, 0, 4, BatchStrategy::SequentialLoop, {},
                                proof) &&
            !BatchPlan::matrixLanes(facts, 0, 4, BatchStrategy::MatrixBatch, {},
                                    proof) &&
            !BatchPlan::matrixLanes(facts, 0, 4, BatchStrategy::MatrixSpatial,
                                    layouts, proof) &&
            !BatchPlan::matrixLanes(
                facts, 0, 4, BatchStrategy::MatrixStaticLanes, {}, wrongProof),
        "strategy-mismatched layouts or proof must not form matrix plans");
  proof.effects.clear();
  check(program->proof().effects.size() == 2 &&
            sequential->proof().effects.size() == 2,
        "plans must own obligations after the planning producer is modified");
  proof = program->proof();
  layouts[1].accessID = 0;
  check(!BatchPlan::programLanes(facts, 0, 4, layouts, proof),
        "duplicate layout IDs must not become an accepted plan");
  vv.reset();
  facts.reset();
  check(program->facts().accesses().size() == 2 &&
            sequential->groupedLoop()->id == 7,
        "accepted plans must own facts independently of the analysis producer");
  std::cout << "owned effect projection regressions passed\n";
}
