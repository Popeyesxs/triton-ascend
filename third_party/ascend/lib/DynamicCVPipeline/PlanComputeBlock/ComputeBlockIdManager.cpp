#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace CVPipeline {

bool ComputeBlockIdManager::isWholeCubeReady(Operation *seedOp, llvm::DenseMap<Operation *, int> &indegree)
{
    auto id = lookupOpBlockId(seedOp);
    if (id == -1) {
        return (indegree[seedOp] == 0);
    }
    auto cubeBlock = getOpsByBlockId(id);
    for (auto op : cubeBlock) {
        if (!indegree.contains(op)) {
            continue;
        }
        if (indegree[op] != 0) {
            return false;
        }
    }
    return true;
}

bool ComputeBlockIdManager::isSameBlock(Operation *a, Operation *b)
{
    if (getBlockIdByOp(a) == getBlockIdByOp(b) && getBlockIdByOp(a) != -1) {
        return true;
    }
    return false;
}

unsigned int ComputeBlockIdManager::getNextId()
{
    std::lock_guard<std::mutex> lock(managerMutex);
    return cntComputeBlockId++;
}

int ComputeBlockIdManager::lookupOpBlockId(Operation *op)
{
    if (!op) {
        return -1;
    }
    if (auto a = op->getAttrOfType<IntegerAttr>(attr::kBlockId)) {
        return a.getInt();
    }
    return -1;
}

void ComputeBlockIdManager::markOpBlockId(Operation *op, int blockId)
{
    if (!op) {
        return;
    }
    MLIRContext *ctx = op->getContext();
    op->setAttr(attr::kBlockId, IntegerAttr::get(IntegerType::get(ctx, 32), blockId));
    record(op, blockId);
}

llvm::SmallVector<Operation *> ComputeBlockIdManager::getOpsByBlockId(int blockId)
{
    if (blockId == -1) {
        return {};
    }
    std::lock_guard<std::mutex> lock(managerMutex);
    auto it = blockIdToOps.find(blockId);
    if (it == blockIdToOps.end()) {
        return {};
    }
    return llvm::SmallVector<Operation *>(it->second.begin(), it->second.end());
}

int ComputeBlockIdManager::getBlockIdByOp(Operation *op)
{
    if (!op) {
        return -1;
    }
    {
        std::lock_guard<std::mutex> lock(managerMutex);
        auto it = opToBlockId.find(op);
        if (it != opToBlockId.end()) {
            return it->second;
        }
    }
    int id = lookupOpBlockId(op);
    if (id >= 0) {
        record(op, id);
    }
    return id;
}

void ComputeBlockIdManager::markOpsWithNewId(llvm::SmallVectorImpl<Operation *> &ops)
{
    if (ops.empty()) {
        return;
    }
    int id = static_cast<int>(getNextId());
    for (Operation *op : ops) {
        markOpBlockId(op, id);
    }
}

void ComputeBlockIdManager::reset()
{
    std::lock_guard<std::mutex> lock(managerMutex);
    cntComputeBlockId = 0;
    blockIdToOps.clear();
    opToBlockId.clear();
}

void ComputeBlockIdManager::record(Operation *op, int blockId)
{
    if (!op || blockId < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(managerMutex);
    auto itOld = opToBlockId.find(op);
    if (itOld != opToBlockId.end()) {
        // return llvm::failure("Operation already recorded in ComputeBlockIdManager");
    }

    opToBlockId[op] = blockId;
    auto &vec = blockIdToOps[blockId];
    for (Operation *existing : vec) {
        if (existing == op) {
            return;
        }
    }
    vec.push_back(op);
}

} // namespace CVPipeline
} // namespace mlir
