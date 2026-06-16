/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef REM_TO_MASK_CONVERSION_PASSES
#define REM_TO_MASK_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def RemToMaskConversion : Pass<"rem-to-mask-conversion", "mlir::ModuleOp"> {
  let summary = "Convert modulo-based boundary handling to mask operations for load/store";
  let description = [{
    This pass identifies patterns where modulo operations are used to prevent
    out-of-bounds memory access (e.g., `(pid * BLOCK_SIZE + arange) % Bound`),
    and replaces them with mask-based boundary checks (`offs < Bound`).

    This optimization is beneficial on Ascend NPU where:
    1. Modulo operations (arith.remsi) are computationally expensive
    2. Masked load/store operations are natively supported with minimal overhead
    3. The "zero on OOB" semantics (via mask+other=0) preserves correctness
       for accumulation-based computations (e.g., matrix multiplication)

    The pass is safe when the modulo is used purely for boundary protection
    (not for true cyclic/wrap-around data access patterns).
  }];
  let constructor = "triton::createRemToMaskConversionPass()";
}

#endif // REM_TO_MASK_CONVERSION_PASSES
