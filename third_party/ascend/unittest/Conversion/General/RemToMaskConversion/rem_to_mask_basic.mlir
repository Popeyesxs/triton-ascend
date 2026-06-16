// RUN: triton-opt --rem-to-mask-conversion --split-input-file %s | FileCheck %s

// Test: Basic matmul-style offs_am = (pid_m * BLOCK_M + arange(0, BLOCK_M)) % M
// The remsi should be eliminated and replaced with a mask on the load.

tt.func public @matmul_rem_to_mask(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32) attributes {noinline = false} {
  %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %c64_i32 = arith.constant 64 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c64_i32 : i32
  %2 = tt.splat %1 : i32 -> tensor<64xi32>
  %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %4 = arith.addi %2, %3 : tensor<64xi32>
  // This remsi should be removed and replaced with a boundary mask
  %5 = tt.splat %arg1 : i32 -> tensor<64xi32>
  %6 = arith.remsi %4, %5 : tensor<64xi32>
  %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %8 = tt.addptr %7, %6 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  %9 = tt.load %8 : tensor<64x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL: tt.func public @matmul_rem_to_mask(
// CHECK-NOT: arith.remsi
// CHECK: %[[OFFS:.*]] = arith.addi
// CHECK: %[[BOUND:.*]] = tt.splat %arg1
// CHECK: %[[MASK:.*]] = arith.cmpi slt, %[[OFFS]], %[[BOUND]]
// CHECK: tt.load {{.*}}, %[[MASK]]


// -----

// Test: remsi with existing mask on load - masks should be combined with AND

tt.func public @rem_to_mask_with_existing_mask(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32) attributes {noinline = false} {
  %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %c64_i32 = arith.constant 64 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c64_i32 : i32
  %2 = tt.splat %1 : i32 -> tensor<64xi32>
  %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %4 = arith.addi %2, %3 : tensor<64xi32>
  %5 = tt.splat %arg1 : i32 -> tensor<64xi32>
  %6 = arith.remsi %4, %5 : tensor<64xi32>
  %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %8 = tt.addptr %7, %6 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  // Existing mask: offs < K
  %9 = tt.splat %arg2 : i32 -> tensor<64xi32>
  %10 = arith.cmpi slt, %4, %9 : tensor<64xi32>
  %11 = tt.load %8, %10, %cst : tensor<64x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL: tt.func public @rem_to_mask_with_existing_mask(
// CHECK-NOT: arith.remsi
// CHECK: %[[BOUND_MASK:.*]] = arith.cmpi slt
// CHECK: %[[EXISTING:.*]] = arith.cmpi slt
// CHECK: %[[COMBINED:.*]] = arith.andi %[[EXISTING]], %[[BOUND_MASK]]
// CHECK: tt.load {{.*}}, %[[COMBINED]]


// -----

// Test: remsi used in non-memory computation - should NOT be transformed

tt.func public @rem_not_in_memory(%arg0: i32) -> tensor<64xi32> attributes {noinline = false} {
  %c64_i32 = arith.constant 64 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c64_i32 : i32
  %2 = tt.splat %1 : i32 -> tensor<64xi32>
  %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %4 = arith.addi %2, %3 : tensor<64xi32>
  %5 = tt.splat %arg0 : i32 -> tensor<64xi32>
  // This remsi is used in computation (return), not memory access
  %6 = arith.remsi %4, %5 : tensor<64xi32>
  tt.return %6 : tensor<64xi32>
}

// CHECK-LABEL: tt.func public @rem_not_in_memory(
// CHECK: arith.remsi


// -----

// Test: 2D matmul pattern with remsi on both M and N dimensions

tt.func public @matmul_2d_rem_to_mask(
    %a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %M: i32, %K: i32,
    %stride_am: i32, %stride_ak: i32
) attributes {noinline = false} {
  %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32>
  %c64_i32 = arith.constant 64 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c64_i32 : i32
  %2 = tt.splat %1 : i32 -> tensor<64xi32>
  %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %4 = arith.addi %2, %3 : tensor<64xi32>
  // offs_am = (pid_m * 64 + arange(0, 64)) % M
  %5 = tt.splat %M : i32 -> tensor<64xi32>
  %6 = arith.remsi %4, %5 : tensor<64xi32>
  // expand and compute pointers: offs_am[:, None] * stride_am
  %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %8 = tt.splat %stride_am : i32 -> tensor<64x1xi32>
  %9 = arith.muli %7, %8 : tensor<64x1xi32>
  %10 = tt.broadcast %9 : tensor<64x1xi32> -> tensor<64x32xi32>
  // offs_k = arange(0, 32)
  %11 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %12 = tt.expand_dims %11 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %13 = tt.splat %stride_ak : i32 -> tensor<1x32xi32>
  %14 = arith.muli %12, %13 : tensor<1x32xi32>
  %15 = tt.broadcast %14 : tensor<1x32xi32> -> tensor<64x32xi32>
  %16 = arith.addi %10, %15 : tensor<64x32xi32>
  %17 = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>>
  %18 = tt.addptr %17, %16 : tensor<64x32x!tt.ptr<f32>>, tensor<64x32xi32>
  %19 = tt.load %18 : tensor<64x32x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL: tt.func public @matmul_2d_rem_to_mask(
// CHECK-NOT: arith.remsi
// CHECK: arith.cmpi slt
// CHECK: tt.load {{.*}}, %{{.*}}
