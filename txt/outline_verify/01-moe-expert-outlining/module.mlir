module @jit_mixtral_moe attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @expert_slot_1(%arg0: tensor<512x4096xf32>, %arg1: tensor<4096x14336xf32>, %arg2: tensor<4096x14336xf32>, %arg3: tensor<14336x4096xf32>) -> tensor<512x4096xf32> attributes {moe.num_experts = 8 : i32, moe.slot_id = 1 : i32, moe.token_bucket = "large", moe.tokens_per_slot = 512 : i32} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [] : (tensor<512x4096xf32>, tensor<4096x14336xf32>) -> tensor<512x14336xf32>
    %1 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [1] x [0], precision = [] : (tensor<512x4096xf32>, tensor<4096x14336xf32>) -> tensor<512x14336xf32>
    %2 = stablehlo.negate %0 : tensor<512x14336xf32>
    %3 = stablehlo.exponential %2 : tensor<512x14336xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14336xf32>
    %5 = stablehlo.add %4, %3 : tensor<512x14336xf32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14336xf32>
    %7 = stablehlo.divide %6, %5 : tensor<512x14336xf32>
    %8 = stablehlo.multiply %0, %7 : tensor<512x14336xf32>
    %9 = stablehlo.multiply %8, %1 : tensor<512x14336xf32>
    %10 = stablehlo.dot_general %9, %arg3, contracting_dims = [1] x [0], precision = [] : (tensor<512x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    return %10 : tensor<512x4096xf32>
  }
  func.func @expert_slot_0(%arg0: tensor<512x4096xf32>, %arg1: tensor<4096x14336xf32>, %arg2: tensor<4096x14336xf32>, %arg3: tensor<14336x4096xf32>) -> tensor<512x4096xf32> attributes {moe.num_experts = 8 : i32, moe.slot_id = 0 : i32, moe.token_bucket = "large", moe.tokens_per_slot = 512 : i32} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [] : (tensor<512x4096xf32>, tensor<4096x14336xf32>) -> tensor<512x14336xf32>
    %1 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [1] x [0], precision = [] : (tensor<512x4096xf32>, tensor<4096x14336xf32>) -> tensor<512x14336xf32>
    %2 = stablehlo.negate %0 : tensor<512x14336xf32>
    %3 = stablehlo.exponential %2 : tensor<512x14336xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14336xf32>
    %5 = stablehlo.add %4, %3 : tensor<512x14336xf32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14336xf32>
    %7 = stablehlo.divide %6, %5 : tensor<512x14336xf32>
    %8 = stablehlo.multiply %0, %7 : tensor<512x14336xf32>
    %9 = stablehlo.multiply %8, %1 : tensor<512x14336xf32>
    %10 = stablehlo.dot_general %9, %arg3, contracting_dims = [1] x [0], precision = [] : (tensor<512x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    return %10 : tensor<512x4096xf32>
  }
  func.func public @main(%arg0: tensor<512x4096xf32>, %arg1: tensor<4096x8xf32>, %arg2: tensor<8x4096x14336xf32>, %arg3: tensor<8x4096x14336xf32>, %arg4: tensor<8x14336x4096xf32>) -> (tensor<512x4096xf32> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<512x4096xf32>, tensor<4096x8xf32>) -> tensor<512x8xf32>
    %1:2 = stablehlo.custom_call @mhlo.topk(%0) {mhlo.attributes = {k = 2 : i64}, mhlo.version = 1 : i64} : (tensor<512x8xf32>) -> (tensor<512x2xf32>, tensor<512x2xi32>)
    %2 = stablehlo.reduce(%1#0 init: %cst_1) applies stablehlo.maximum across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.maximum %3, %2 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<512xf32>) -> tensor<512x1xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<512x1xf32>) -> tensor<512x2xf32>
    %7 = stablehlo.subtract %1#0, %6 : tensor<512x2xf32>
    %8 = stablehlo.exponential %7 : tensor<512x2xf32>
    %9 = stablehlo.reduce(%8 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<512xf32>) -> tensor<512x1xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<512x1xf32>) -> tensor<512x2xf32>
    %12 = stablehlo.divide %8, %11 : tensor<512x2xf32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x4096xf32>
    %14 = stablehlo.slice %1#1 [0:512, 0:1] : (tensor<512x2xi32>) -> tensor<512x1xi32>
    %15 = stablehlo.reshape %14 : (tensor<512x1xi32>) -> tensor<512xi32>
    %16 = call @_one_hot(%15) : (tensor<512xi32>) -> tensor<512x8xf32>
    %17 = stablehlo.slice %12 [0:512, 0:1] : (tensor<512x2xf32>) -> tensor<512x1xf32>
    %18 = stablehlo.dot_general %arg0, %16, batching_dims = [0] x [0], contracting_dims = [] x [] : (tensor<512x4096xf32>, tensor<512x8xf32>) -> tensor<512x4096x8xf32>
    %19 = stablehlo.transpose %18, dims = [2, 0, 1] : (tensor<512x4096x8xf32>) -> tensor<8x512x4096xf32>
    %20 = stablehlo.slice %19 [0:1, 0:512, 0:4096] : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>
    %21 = stablehlo.reshape %20 : (tensor<1x512x4096xf32>) -> tensor<512x4096xf32>
    %22 = stablehlo.slice %arg2 [0:1, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %23 = stablehlo.reshape %22 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %24 = stablehlo.slice %arg3 [0:1, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %25 = stablehlo.reshape %24 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %26 = stablehlo.slice %arg4 [0:1, 0:14336, 0:4096] : (tensor<8x14336x4096xf32>) -> tensor<1x14336x4096xf32>
    %27 = stablehlo.reshape %26 : (tensor<1x14336x4096xf32>) -> tensor<14336x4096xf32>
    %28 = call @expert_slot_0(%21, %23, %25, %27) : (tensor<512x4096xf32>, tensor<4096x14336xf32>, tensor<4096x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    %29 = stablehlo.reshape %28 : (tensor<512x4096xf32>) -> tensor<1x512x4096xf32>
    %30 = stablehlo.slice %19 [1:2, 0:512, 0:4096] : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>
    %31 = stablehlo.reshape %30 : (tensor<1x512x4096xf32>) -> tensor<512x4096xf32>
    %32 = stablehlo.slice %arg2 [1:2, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %33 = stablehlo.reshape %32 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %34 = stablehlo.slice %arg3 [1:2, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %35 = stablehlo.reshape %34 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %36 = stablehlo.slice %arg4 [1:2, 0:14336, 0:4096] : (tensor<8x14336x4096xf32>) -> tensor<1x14336x4096xf32>
    %37 = stablehlo.reshape %36 : (tensor<1x14336x4096xf32>) -> tensor<14336x4096xf32>
    %38 = call @expert_slot_0(%31, %33, %35, %37) : (tensor<512x4096xf32>, tensor<4096x14336xf32>, tensor<4096x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    %39 = stablehlo.reshape %38 : (tensor<512x4096xf32>) -> tensor<1x512x4096xf32>
    %40 = stablehlo.slice %19 [2:3, 0:512, 0:4096] : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>
    %41 = stablehlo.reshape %40 : (tensor<1x512x4096xf32>) -> tensor<512x4096xf32>
    %42 = stablehlo.slice %arg2 [2:3, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %43 = stablehlo.reshape %42 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %44 = stablehlo.slice %arg3 [2:3, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %45 = stablehlo.reshape %44 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %46 = stablehlo.slice %arg4 [2:3, 0:14336, 0:4096] : (tensor<8x14336x4096xf32>) -> tensor<1x14336x4096xf32>
    %47 = stablehlo.reshape %46 : (tensor<1x14336x4096xf32>) -> tensor<14336x4096xf32>
    %48 = call @expert_slot_0(%41, %43, %45, %47) : (tensor<512x4096xf32>, tensor<4096x14336xf32>, tensor<4096x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    %49 = stablehlo.reshape %48 : (tensor<512x4096xf32>) -> tensor<1x512x4096xf32>
    %50 = stablehlo.slice %19 [3:4, 0:512, 0:4096] : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>
    %51 = stablehlo.reshape %50 : (tensor<1x512x4096xf32>) -> tensor<512x4096xf32>
    %52 = stablehlo.slice %arg2 [3:4, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %53 = stablehlo.reshape %52 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %54 = stablehlo.slice %arg3 [3:4, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %55 = stablehlo.reshape %54 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %56 = stablehlo.slice %arg4 [3:4, 0:14336, 0:4096] : (tensor<8x14336x4096xf32>) -> tensor<1x14336x4096xf32>
    %57 = stablehlo.reshape %56 : (tensor<1x14336x4096xf32>) -> tensor<14336x4096xf32>
    %58 = call @expert_slot_0(%51, %53, %55, %57) : (tensor<512x4096xf32>, tensor<4096x14336xf32>, tensor<4096x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    %59 = stablehlo.reshape %58 : (tensor<512x4096xf32>) -> tensor<1x512x4096xf32>
    %60 = stablehlo.slice %19 [4:5, 0:512, 0:4096] : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>
    %61 = stablehlo.reshape %60 : (tensor<1x512x4096xf32>) -> tensor<512x4096xf32>
    %62 = stablehlo.slice %arg2 [4:5, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %63 = stablehlo.reshape %62 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %64 = stablehlo.slice %arg3 [4:5, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %65 = stablehlo.reshape %64 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %66 = stablehlo.slice %arg4 [4:5, 0:14336, 0:4096] : (tensor<8x14336x4096xf32>) -> tensor<1x14336x4096xf32>
    %67 = stablehlo.reshape %66 : (tensor<1x14336x4096xf32>) -> tensor<14336x4096xf32>
    %68 = call @expert_slot_0(%61, %63, %65, %67) : (tensor<512x4096xf32>, tensor<4096x14336xf32>, tensor<4096x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    %69 = stablehlo.reshape %68 : (tensor<512x4096xf32>) -> tensor<1x512x4096xf32>
    %70 = stablehlo.slice %19 [5:6, 0:512, 0:4096] : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>
    %71 = stablehlo.reshape %70 : (tensor<1x512x4096xf32>) -> tensor<512x4096xf32>
    %72 = stablehlo.slice %arg2 [5:6, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %73 = stablehlo.reshape %72 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %74 = stablehlo.slice %arg3 [5:6, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %75 = stablehlo.reshape %74 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %76 = stablehlo.slice %arg4 [5:6, 0:14336, 0:4096] : (tensor<8x14336x4096xf32>) -> tensor<1x14336x4096xf32>
    %77 = stablehlo.reshape %76 : (tensor<1x14336x4096xf32>) -> tensor<14336x4096xf32>
    %78 = call @expert_slot_0(%71, %73, %75, %77) : (tensor<512x4096xf32>, tensor<4096x14336xf32>, tensor<4096x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    %79 = stablehlo.reshape %78 : (tensor<512x4096xf32>) -> tensor<1x512x4096xf32>
    %80 = stablehlo.slice %19 [6:7, 0:512, 0:4096] : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>
    %81 = stablehlo.reshape %80 : (tensor<1x512x4096xf32>) -> tensor<512x4096xf32>
    %82 = stablehlo.slice %arg2 [6:7, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %83 = stablehlo.reshape %82 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %84 = stablehlo.slice %arg3 [6:7, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %85 = stablehlo.reshape %84 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %86 = stablehlo.slice %arg4 [6:7, 0:14336, 0:4096] : (tensor<8x14336x4096xf32>) -> tensor<1x14336x4096xf32>
    %87 = stablehlo.reshape %86 : (tensor<1x14336x4096xf32>) -> tensor<14336x4096xf32>
    %88 = call @expert_slot_0(%81, %83, %85, %87) : (tensor<512x4096xf32>, tensor<4096x14336xf32>, tensor<4096x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    %89 = stablehlo.reshape %88 : (tensor<512x4096xf32>) -> tensor<1x512x4096xf32>
    %90 = stablehlo.slice %19 [7:8, 0:512, 0:4096] : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>
    %91 = stablehlo.reshape %90 : (tensor<1x512x4096xf32>) -> tensor<512x4096xf32>
    %92 = stablehlo.slice %arg2 [7:8, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %93 = stablehlo.reshape %92 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %94 = stablehlo.slice %arg3 [7:8, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %95 = stablehlo.reshape %94 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %96 = stablehlo.slice %arg4 [7:8, 0:14336, 0:4096] : (tensor<8x14336x4096xf32>) -> tensor<1x14336x4096xf32>
    %97 = stablehlo.reshape %96 : (tensor<1x14336x4096xf32>) -> tensor<14336x4096xf32>
    %98 = call @expert_slot_0(%91, %93, %95, %97) : (tensor<512x4096xf32>, tensor<4096x14336xf32>, tensor<4096x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    %99 = stablehlo.reshape %98 : (tensor<512x4096xf32>) -> tensor<1x512x4096xf32>
    %100 = stablehlo.concatenate %29, %39, %49, %59, %69, %79, %89, %99, dim = 0 : (tensor<1x512x4096xf32>, tensor<1x512x4096xf32>, tensor<1x512x4096xf32>, tensor<1x512x4096xf32>, tensor<1x512x4096xf32>, tensor<1x512x4096xf32>, tensor<1x512x4096xf32>, tensor<1x512x4096xf32>) -> tensor<8x512x4096xf32>
    %101 = stablehlo.dot_general %16, %100, batching_dims = [0] x [1], contracting_dims = [1] x [0] : (tensor<512x8xf32>, tensor<8x512x4096xf32>) -> tensor<512x4096xf32>
    %102 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<512x1xf32>) -> tensor<512x4096xf32>
    %103 = stablehlo.multiply %102, %101 : tensor<512x4096xf32>
    %104 = stablehlo.add %13, %103 : tensor<512x4096xf32>
    %105 = stablehlo.slice %1#1 [0:512, 1:2] : (tensor<512x2xi32>) -> tensor<512x1xi32>
    %106 = stablehlo.reshape %105 : (tensor<512x1xi32>) -> tensor<512xi32>
    %107 = call @_one_hot(%106) : (tensor<512xi32>) -> tensor<512x8xf32>
    %108 = stablehlo.slice %12 [0:512, 1:2] : (tensor<512x2xf32>) -> tensor<512x1xf32>
    %109 = stablehlo.dot_general %arg0, %107, batching_dims = [0] x [0], contracting_dims = [] x [] : (tensor<512x4096xf32>, tensor<512x8xf32>) -> tensor<512x4096x8xf32>
    %110 = stablehlo.transpose %109, dims = [2, 0, 1] : (tensor<512x4096x8xf32>) -> tensor<8x512x4096xf32>
    %111 = stablehlo.slice %110 [0:1, 0:512, 0:4096] : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>
    %112 = stablehlo.reshape %111 : (tensor<1x512x4096xf32>) -> tensor<512x4096xf32>
    %113 = stablehlo.slice %arg2 [0:1, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %114 = stablehlo.reshape %113 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %115 = stablehlo.slice %arg3 [0:1, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %116 = stablehlo.reshape %115 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %117 = stablehlo.slice %arg4 [0:1, 0:14336, 0:4096] : (tensor<8x14336x4096xf32>) -> tensor<1x14336x4096xf32>
    %118 = stablehlo.reshape %117 : (tensor<1x14336x4096xf32>) -> tensor<14336x4096xf32>
    %119 = call @expert_slot_1(%112, %114, %116, %118) : (tensor<512x4096xf32>, tensor<4096x14336xf32>, tensor<4096x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    %120 = stablehlo.reshape %119 : (tensor<512x4096xf32>) -> tensor<1x512x4096xf32>
    %121 = stablehlo.slice %110 [1:2, 0:512, 0:4096] : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>
    %122 = stablehlo.reshape %121 : (tensor<1x512x4096xf32>) -> tensor<512x4096xf32>
    %123 = stablehlo.slice %arg2 [1:2, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %124 = stablehlo.reshape %123 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %125 = stablehlo.slice %arg3 [1:2, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %126 = stablehlo.reshape %125 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %127 = stablehlo.slice %arg4 [1:2, 0:14336, 0:4096] : (tensor<8x14336x4096xf32>) -> tensor<1x14336x4096xf32>
    %128 = stablehlo.reshape %127 : (tensor<1x14336x4096xf32>) -> tensor<14336x4096xf32>
    %129 = call @expert_slot_1(%122, %124, %126, %128) : (tensor<512x4096xf32>, tensor<4096x14336xf32>, tensor<4096x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    %130 = stablehlo.reshape %129 : (tensor<512x4096xf32>) -> tensor<1x512x4096xf32>
    %131 = stablehlo.slice %110 [2:3, 0:512, 0:4096] : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>
    %132 = stablehlo.reshape %131 : (tensor<1x512x4096xf32>) -> tensor<512x4096xf32>
    %133 = stablehlo.slice %arg2 [2:3, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %134 = stablehlo.reshape %133 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %135 = stablehlo.slice %arg3 [2:3, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %136 = stablehlo.reshape %135 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %137 = stablehlo.slice %arg4 [2:3, 0:14336, 0:4096] : (tensor<8x14336x4096xf32>) -> tensor<1x14336x4096xf32>
    %138 = stablehlo.reshape %137 : (tensor<1x14336x4096xf32>) -> tensor<14336x4096xf32>
    %139 = call @expert_slot_1(%132, %134, %136, %138) : (tensor<512x4096xf32>, tensor<4096x14336xf32>, tensor<4096x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    %140 = stablehlo.reshape %139 : (tensor<512x4096xf32>) -> tensor<1x512x4096xf32>
    %141 = stablehlo.slice %110 [3:4, 0:512, 0:4096] : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>
    %142 = stablehlo.reshape %141 : (tensor<1x512x4096xf32>) -> tensor<512x4096xf32>
    %143 = stablehlo.slice %arg2 [3:4, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %144 = stablehlo.reshape %143 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %145 = stablehlo.slice %arg3 [3:4, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %146 = stablehlo.reshape %145 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %147 = stablehlo.slice %arg4 [3:4, 0:14336, 0:4096] : (tensor<8x14336x4096xf32>) -> tensor<1x14336x4096xf32>
    %148 = stablehlo.reshape %147 : (tensor<1x14336x4096xf32>) -> tensor<14336x4096xf32>
    %149 = call @expert_slot_1(%142, %144, %146, %148) : (tensor<512x4096xf32>, tensor<4096x14336xf32>, tensor<4096x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    %150 = stablehlo.reshape %149 : (tensor<512x4096xf32>) -> tensor<1x512x4096xf32>
    %151 = stablehlo.slice %110 [4:5, 0:512, 0:4096] : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>
    %152 = stablehlo.reshape %151 : (tensor<1x512x4096xf32>) -> tensor<512x4096xf32>
    %153 = stablehlo.slice %arg2 [4:5, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %154 = stablehlo.reshape %153 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %155 = stablehlo.slice %arg3 [4:5, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %156 = stablehlo.reshape %155 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %157 = stablehlo.slice %arg4 [4:5, 0:14336, 0:4096] : (tensor<8x14336x4096xf32>) -> tensor<1x14336x4096xf32>
    %158 = stablehlo.reshape %157 : (tensor<1x14336x4096xf32>) -> tensor<14336x4096xf32>
    %159 = call @expert_slot_1(%152, %154, %156, %158) : (tensor<512x4096xf32>, tensor<4096x14336xf32>, tensor<4096x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    %160 = stablehlo.reshape %159 : (tensor<512x4096xf32>) -> tensor<1x512x4096xf32>
    %161 = stablehlo.slice %110 [5:6, 0:512, 0:4096] : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>
    %162 = stablehlo.reshape %161 : (tensor<1x512x4096xf32>) -> tensor<512x4096xf32>
    %163 = stablehlo.slice %arg2 [5:6, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %164 = stablehlo.reshape %163 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %165 = stablehlo.slice %arg3 [5:6, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %166 = stablehlo.reshape %165 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %167 = stablehlo.slice %arg4 [5:6, 0:14336, 0:4096] : (tensor<8x14336x4096xf32>) -> tensor<1x14336x4096xf32>
    %168 = stablehlo.reshape %167 : (tensor<1x14336x4096xf32>) -> tensor<14336x4096xf32>
    %169 = call @expert_slot_1(%162, %164, %166, %168) : (tensor<512x4096xf32>, tensor<4096x14336xf32>, tensor<4096x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    %170 = stablehlo.reshape %169 : (tensor<512x4096xf32>) -> tensor<1x512x4096xf32>
    %171 = stablehlo.slice %110 [6:7, 0:512, 0:4096] : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>
    %172 = stablehlo.reshape %171 : (tensor<1x512x4096xf32>) -> tensor<512x4096xf32>
    %173 = stablehlo.slice %arg2 [6:7, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %174 = stablehlo.reshape %173 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %175 = stablehlo.slice %arg3 [6:7, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %176 = stablehlo.reshape %175 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %177 = stablehlo.slice %arg4 [6:7, 0:14336, 0:4096] : (tensor<8x14336x4096xf32>) -> tensor<1x14336x4096xf32>
    %178 = stablehlo.reshape %177 : (tensor<1x14336x4096xf32>) -> tensor<14336x4096xf32>
    %179 = call @expert_slot_1(%172, %174, %176, %178) : (tensor<512x4096xf32>, tensor<4096x14336xf32>, tensor<4096x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    %180 = stablehlo.reshape %179 : (tensor<512x4096xf32>) -> tensor<1x512x4096xf32>
    %181 = stablehlo.slice %110 [7:8, 0:512, 0:4096] : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>
    %182 = stablehlo.reshape %181 : (tensor<1x512x4096xf32>) -> tensor<512x4096xf32>
    %183 = stablehlo.slice %arg2 [7:8, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %184 = stablehlo.reshape %183 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %185 = stablehlo.slice %arg3 [7:8, 0:4096, 0:14336] : (tensor<8x4096x14336xf32>) -> tensor<1x4096x14336xf32>
    %186 = stablehlo.reshape %185 : (tensor<1x4096x14336xf32>) -> tensor<4096x14336xf32>
    %187 = stablehlo.slice %arg4 [7:8, 0:14336, 0:4096] : (tensor<8x14336x4096xf32>) -> tensor<1x14336x4096xf32>
    %188 = stablehlo.reshape %187 : (tensor<1x14336x4096xf32>) -> tensor<14336x4096xf32>
    %189 = call @expert_slot_1(%182, %184, %186, %188) : (tensor<512x4096xf32>, tensor<4096x14336xf32>, tensor<4096x14336xf32>, tensor<14336x4096xf32>) -> tensor<512x4096xf32>
    %190 = stablehlo.reshape %189 : (tensor<512x4096xf32>) -> tensor<1x512x4096xf32>
    %191 = stablehlo.concatenate %120, %130, %140, %150, %160, %170, %180, %190, dim = 0 : (tensor<1x512x4096xf32>, tensor<1x512x4096xf32>, tensor<1x512x4096xf32>, tensor<1x512x4096xf32>, tensor<1x512x4096xf32>, tensor<1x512x4096xf32>, tensor<1x512x4096xf32>, tensor<1x512x4096xf32>) -> tensor<8x512x4096xf32>
    %192 = stablehlo.dot_general %107, %191, batching_dims = [0] x [1], contracting_dims = [1] x [0] : (tensor<512x8xf32>, tensor<8x512x4096xf32>) -> tensor<512x4096xf32>
    %193 = stablehlo.broadcast_in_dim %108, dims = [0, 1] : (tensor<512x1xf32>) -> tensor<512x4096xf32>
    %194 = stablehlo.multiply %193, %192 : tensor<512x4096xf32>
    %195 = stablehlo.add %104, %194 : tensor<512x4096xf32>
    return %195 : tensor<512x4096xf32>
  }
  func.func private @_one_hot(%arg0: tensor<512xi32>) -> tensor<512x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<512xi32>) -> tensor<512x1xi32>
    %1 = stablehlo.iota dim = 1 : tensor<1x8xi32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<512x1xi32>) -> tensor<512x8xi32>
    %3 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<512x8xi32>
    %4 = stablehlo.compare EQ, %2, %3, SIGNED : (tensor<512x8xi32>, tensor<512x8xi32>) -> tensor<512x8xi1>
    %5 = stablehlo.convert %4 : (tensor<512x8xi1>) -> tensor<512x8xf32>
    return %5 : tensor<512x8xf32>
  }
}

