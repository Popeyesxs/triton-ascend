// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_FACTS_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_FACTS_H
#include "TritonToGraph/BlockBatching/BatchAccessFact.h"
#include "llvm/ADT/ArrayRef.h"
#include <cassert>

namespace mlir::triton::cfg {
// Stable source traversal IDs. Original accesses and projected guard effects
// have independent storage: a joint store must not change a layout fact.
struct BatchSourceAccess {
  int root;
  unsigned bytes;
  bool write;
  llvm::SmallVector<int64_t> shape;
  BatchExprRef byteOffset, mask;
  llvm::SmallVector<BatchExprRef> pointerChecks;
  llvm::SmallVector<unsigned, 0> loopIDs;
  bool isVolatile, deviceAddress, normalizedPointer;
};

// Owned values only. No diagnostic, JSON, Operation, Value or applied state.
// The legacy shared analyzers retain their temporary compatibility view;
// all long-lived batching facts refer to this single loop table by stable ID.
class BatchFacts {
public:
  BatchFacts(llvm::ArrayRef<BatchAccessFact> accesses,
             llvm::ArrayRef<BatchAccessFact> effects,
             llvm::ArrayRef<BatchLoopFact> loops)
      : loopFacts(loops.begin(), loops.end()) {
    auto copy = [&](llvm::ArrayRef<BatchAccessFact> input,
                    llvm::SmallVectorImpl<BatchSourceAccess> &output) {
      for (const auto &fact : input) {
        BatchSourceAccess value{fact.root,
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
        for (const auto &loop : fact.loops) {
          assert(findLoop(loop.id) && "unknown source loop ID");
          value.loopIDs.push_back(loop.id);
        }
        output.push_back(std::move(value));
      }
    };
    copy(accesses, accessFacts);
    copy(effects, effectFacts);
  }
  llvm::ArrayRef<BatchSourceAccess> accesses() const { return accessFacts; }
  llvm::ArrayRef<BatchSourceAccess> effects() const { return effectFacts; }
  llvm::ArrayRef<BatchLoopFact> loops() const { return loopFacts; }
  const BatchLoopFact *findLoop(unsigned id) const {
    for (const auto &loop : loopFacts)
      if (loop.id == id)
        return &loop;
    return nullptr;
  }

private:
  llvm::SmallVector<BatchSourceAccess, 0> accessFacts, effectFacts;
  llvm::SmallVector<BatchLoopFact, 0> loopFacts;
};
} // namespace mlir::triton::cfg
#endif
