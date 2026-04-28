#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_COMPUTE_BLOCK_ID_MANAGER_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_COMPUTE_BLOCK_ID_MANAGER_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <mutex>

namespace mlir {
namespace CVPipeline {

namespace attr {
inline constexpr llvm::StringLiteral kBlockId = "ssbuffer.block_id";
} // namespace attr

/**
    * the class is to promise CUBEID and VECTORID are unified.
 */
class ComputeBlockIdManager {
public:
    static ComputeBlockIdManager &getInstance()
    {
        static ComputeBlockIdManager instance;
        return instance;
    }

    unsigned int getNextId();

    bool isSameBlock(Operation *a, Operation *b);
    bool isWholeCubeReady(Operation *seedOp, llvm::DenseMap<Operation *, int> &indegree);

    int lookupOpBlockId(Operation *op);
    void markOpBlockId(Operation *op, int blockId);

    llvm::SmallVector<Operation *> getOpsByBlockId(int blockId);
    int getBlockIdByOp(Operation *op);
    void markOpsWithNewId(llvm::SmallVectorImpl<Operation *> &ops);
    void reset();

private:
    ComputeBlockIdManager() : cntComputeBlockId(0) {}
    void record(Operation *op, int blockId);

    unsigned int cntComputeBlockId;
    llvm::DenseMap<int, llvm::SmallVector<Operation *>> blockIdToOps;
    llvm::DenseMap<Operation *, int> opToBlockId;
    mutable std::mutex managerMutex;
};

} // namespace CVPipeline
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_COMPUTE_BLOCK_ID_MANAGER_H
