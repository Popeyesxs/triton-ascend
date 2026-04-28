#ifndef ADD_AUTO_SCHEDULING_COMMON_UTILS_H
#define ADD_AUTO_SCHEDULING_COMMON_UTILS_H
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include <string_view>

namespace mlir {
namespace CVPipeline {

// Attribute names for block ID management
namespace attr {
inline constexpr llvm::StringLiteral kCoreType = "ssbuffer.core_type";
} // namespace attr

enum CoreType {
    UNDETERMINED = 0,
    VECTOR_ONLY = 1 << 0,
    CUBE_ONLY = 1 << 1,
    CUBE_AND_VECTOR = VECTOR_ONLY | CUBE_ONLY,
};

inline constexpr std::string_view toStrCoreType(CoreType a)
{
    switch (a) {
        case CoreType::VECTOR_ONLY:
            return "VECTOR";
        case CoreType::CUBE_ONLY:
            return "CUBE";
        case CoreType::CUBE_AND_VECTOR:
            return "CUBE_AND_VECTOR";
        case CoreType::UNDETERMINED:
        default:
            return "UNDETERMINED";
    }
}
inline constexpr CoreType fromStrCoreType(std::string_view s)
{
    if (s == "VECTOR") {
        return CoreType::VECTOR_ONLY;
    }
    if (s == "CUBE") {
        return CoreType::CUBE_ONLY;
    }

    return CoreType::UNDETERMINED;
}

// Functions for managing core types
CoreType lookupOpCoreType(Operation *op);
const char *literalCoreType(CoreType ct);
} // namespace CVPipeline
} // namespace mlir

#endif
