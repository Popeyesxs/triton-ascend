#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
namespace mlir {
namespace CVPipeline {

const char *literalCoreType(CoreType ct)
{
    switch (ct) {
        case VECTOR_ONLY:
            return "VECTOR_ONLY";
        case CUBE_ONLY:
            return "CUBE_ONLY";
        case CUBE_AND_VECTOR:
            return "CUBE_AND_VECTOR";
        case UNDETERMINED:
            return "UNDETERMINED";
    }
    return "UNKNOWN";
}

CoreType lookupOpCoreType(Operation *op)
{
    if (!op) {
        return UNDETERMINED;
    }
    if (auto a = op->getAttrOfType<StringAttr>(attr::kCoreType)) {
        return fromStrCoreType(a.getValue());
    }
    return UNDETERMINED;
}

} // namespace CVPipeline
} // namespace mlir
