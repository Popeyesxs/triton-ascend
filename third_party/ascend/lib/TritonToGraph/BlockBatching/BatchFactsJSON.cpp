// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchFactsJSON.h"
#include "TritonToGraph/BlockBatching/BatchRewriteResult.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir::triton::cfg {
namespace {
llvm::json::Value exprJSON(const BatchExpr &self);
} // namespace

llvm::json::Value batchAccessesJSON(const BatchEffectProjection &effects) {
  llvm::json::Array accesses;
  using A = llvm::json::Array;
  for (const auto &fact : effects.accesses) {
    A checks, shape, loops;
    for (const auto &step : fact.pointerChecks)
      checks.push_back(batchExprJSON(step));
    for (int64_t extent : fact.shape)
      shape.push_back(extent);
    for (const auto &loop : fact.loops)
      loops.push_back(batchLoopJSON(loop));
    accesses.push_back(
        llvm::json::Object{{"root", fact.root},
                           {"offset", batchExprJSON(fact.byteOffset)},
                           {"shape", std::move(shape)},
                           {"mask", batchExprJSON(fact.mask)},
                           {"write", fact.write},
                           {"bytes", fact.bytes},
                           {"pointer_checks", std::move(checks)},
                           {"loops", std::move(loops)},
                           {"volatile", fact.isVolatile},
                           {"device_address", fact.deviceAddress}});
  }
  return llvm::json::Object{{"version", 2}, {"accesses", std::move(accesses)}};
}

// Expand loop references only at the legacy wire boundary. Keep the same
// encoder so expression widths, member slots and joint-store checks cannot
// drift.
llvm::json::Value batchEffectsJSON(const BatchFacts &facts) {
  BatchEffectProjection projection;
  for (const auto &fact : facts.effects()) {
    BatchAccessFact value{fact.root,
                          fact.bytes,
                          fact.write,
                          fact.shape,
                          fact.byteOffset,
                          fact.mask,
                          fact.pointerChecks,
                          {},
                          fact.isVolatile,
                          fact.deviceAddress,
                          fact.normalizedPointer};
    for (unsigned id : fact.loopIDs) {
      const auto *loop = facts.findLoop(id);
      assert(loop && "unknown VV guard loop ID");
      value.loops.push_back(*loop);
    }
    projection.accesses.push_back(std::move(value));
  }
  return batchAccessesJSON(projection);
}

namespace {
llvm::json::Value batchProofJSON(const BatchFacts &facts,
                                 const BatchProofObligations &proof, int axis,
                                 int64_t factor) {
  auto wire = batchEffectsJSON(facts);
  auto &object = *wire.getAsObject();
  object["version"] = 3;
  object["plan_kind"] = "unified";
  object["semantic_contract_version"] = 4;
  object["memory_policy"] = "disjoint_or_terminal_store";
  object["factor"] = factor;
  object["axis"] = axis;
  llvm::json::Array obligations, terminalReads;
  for (const auto &obligation : proof.effects)
    obligations.push_back(llvm::json::Object{
        {"access_id", static_cast<int64_t>(obligation.accessID)},
        {"rule", batchEffectRuleName(obligation.rule)},
        {"proof", obligation.proof == BatchEffectProof::ByteDomain
                      ? "byte_domain"
                      : "lane_correspondence"}});
  for (unsigned id : proof.terminalReads)
    terminalReads.push_back(id);
  object["terminal_read_ids"] = std::move(terminalReads);
  object["obligations"] = std::move(obligations);
  return wire;
}
} // namespace

// Refusal diagnostics are encoded on demand by the outer transaction. They
// are never accepted as an execution contract or stored in a planning object.
std::optional<std::string>
encodeBatchPendingGuard(const BatchFacts &facts,
                        const BatchProofObligations &proof, int axis,
                        int64_t factor) {
  if (!proof.validFor(facts))
    return std::nullopt;
  return llvm::formatv("{0}", batchProofJSON(facts, proof, axis, factor)).str();
}

std::optional<std::string> encodeBatchGuard(const BatchGuardSpec &spec) {
  const auto &plan = spec.plan();
  auto wire =
      batchProofJSON(plan.facts(), plan.proof(), plan.axis(), plan.factor());
  auto &object = *wire.getAsObject();
  auto &obligations = *object.getArray("obligations");
  llvm::json::Array layoutChecks;
  for (const auto &layout : spec.applied().layouts())
    if (layout.rewritten)
      (*obligations[layout.layout.accessID].getAsObject())["proof"] =
          "byte_domain";
  if (plan.preservesGrid())
    object["loop_schedule"] =
        llvm::json::Object{{"version", 1},
                           {"kind", "sequential_iterations_v1"},
                           {"grid_policy", "preserve"},
                           {"factor", plan.factor()},
                           {"loop", batchLoopJSON(*plan.groupedLoop())}};
  // The legacy layout lowerer owns its byte-domain proof. Decode that wire
  // only here at the compatibility boundary, never inside planning or dispatch.
  for (const auto &layout : spec.applied().layouts())
    if (layout.rewritten && !layout.runtimeGuardJSON.empty()) {
      auto parsed = llvm::json::parse(layout.runtimeGuardJSON);
      if (!parsed) {
        llvm::consumeError(parsed.takeError());
        return std::nullopt;
      }
      if (!parsed->getAsObject())
        return std::nullopt;
      layoutChecks.push_back(std::move(*parsed));
    }
  if (!layoutChecks.empty())
    object["layout_checks"] = std::move(layoutChecks);
  return llvm::formatv("{0}", wire).str();
}

llvm::json::Value batchExprJSON(const BatchExprRef &expr) {
  return expr ? exprJSON(*expr) : llvm::json::Value(nullptr);
}
llvm::json::Value batchLoopJSON(const BatchLoopFact &loop) {
  return llvm::json::Object{{"id", loop.id},
                            {"bits", loop.bits},
                            {"lower", batchExprJSON(loop.lower)},
                            {"upper", batchExprJSON(loop.upper)},
                            {"step", batchExprJSON(loop.step)}};
}

namespace {
llvm::json::Value exprJSON(const BatchExpr &self) {
  using A = llvm::json::Array;
  switch (self.kind) {
  case BatchExpr::Constant:
    return A{"const", self.value, self.bits};
  case BatchExpr::Argument:
    return A{"arg", self.value, self.bits};
  case BatchExpr::Program:
    return A{"pid", self.value, self.bits};
  case BatchExpr::ProgramQuotient:
  case BatchExpr::ProgramRemainder:
    return A{self.kind == BatchExpr::ProgramQuotient ? "pid_div" : "pid_rem",
             self.value, self.extent, self.bits};
  case BatchExpr::Grid:
    return A{"grid", self.value, self.bits};
  case BatchExpr::Coordinate:
    return A{"coord", self.value, self.extent, self.bits};
  case BatchExpr::Induction:
    return A{"iv", self.value, self.bits};
  case BatchExpr::True:
    return A{"true"};
  case BatchExpr::False:
    return A{"false"};
  case BatchExpr::Unknown:
    return A{"unknown"};
  case BatchExpr::And:
    return A{"and", batchExprJSON(self.lhs), batchExprJSON(self.rhs)};
  case BatchExpr::SLT:
  case BatchExpr::ULT:
  case BatchExpr::EQ:
    return A{self.kind == BatchExpr::SLT   ? "slt"
             : self.kind == BatchExpr::ULT ? "ult"
                                           : "eq",
             batchExprJSON(self.lhs), batchExprJSON(self.rhs), self.bits};
  case BatchExpr::ExtSI:
  case BatchExpr::ExtUI:
  case BatchExpr::Trunc:
    return A{self.kind == BatchExpr::ExtSI   ? "extsi"
             : self.kind == BatchExpr::ExtUI ? "extui"
                                             : "trunci",
             self.bits, batchExprJSON(self.lhs), self.extent};
  case BatchExpr::MinSI:
  case BatchExpr::MaxSI:
    return A{self.kind == BatchExpr::MinSI ? "minsi" : "maxsi", self.bits,
             batchExprJSON(self.lhs), batchExprJSON(self.rhs)};
  case BatchExpr::Add:
  case BatchExpr::Sub:
  case BatchExpr::Mul:
    return A{self.kind == BatchExpr::Add   ? "addi"
             : self.kind == BatchExpr::Sub ? "subi"
                                           : "muli",
             self.bits, batchExprJSON(self.lhs), batchExprJSON(self.rhs)};
  }
  llvm_unreachable("unhandled batch coordinate expression");
}
} // namespace

} // namespace mlir::triton::cfg
