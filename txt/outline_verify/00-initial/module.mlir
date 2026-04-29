module @jit_mixtral_moe attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
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
    %20 = stablehlo.dot_general %19, %arg2, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x512x4096xf32>, tensor<8x4096x14336xf32>) -> tensor<8x512x14336xf32>
    %21 = stablehlo.dot_general %19, %arg3, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x512x4096xf32>, tensor<8x4096x14336xf32>) -> tensor<8x512x14336xf32>
    %22 = stablehlo.negate %20 : tensor<8x512x14336xf32>
    %23 = stablehlo.exponential %22 : tensor<8x512x14336xf32>
    %24 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x512x14336xf32>
    %25 = stablehlo.add %24, %23 : tensor<8x512x14336xf32>
    %26 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x512x14336xf32>
    %27 = stablehlo.divide %26, %25 : tensor<8x512x14336xf32>
    %28 = stablehlo.multiply %20, %27 : tensor<8x512x14336xf32>
    %29 = stablehlo.multiply %28, %21 : tensor<8x512x14336xf32>
    %30 = stablehlo.dot_general %29, %arg4, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x512x14336xf32>, tensor<8x14336x4096xf32>) -> tensor<8x512x4096xf32>
    %31 = stablehlo.dot_general %16, %30, batching_dims = [0] x [1], contracting_dims = [1] x [0] : (tensor<512x8xf32>, tensor<8x512x4096xf32>) -> tensor<512x4096xf32>
    %32 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<512x1xf32>) -> tensor<512x4096xf32>
    %33 = stablehlo.multiply %32, %31 : tensor<512x4096xf32>
    %34 = stablehlo.add %13, %33 : tensor<512x4096xf32>
    %35 = stablehlo.slice %1#1 [0:512, 1:2] : (tensor<512x2xi32>) -> tensor<512x1xi32>
    %36 = stablehlo.reshape %35 : (tensor<512x1xi32>) -> tensor<512xi32>
    %37 = call @_one_hot(%36) : (tensor<512xi32>) -> tensor<512x8xf32>
    %38 = stablehlo.slice %12 [0:512, 1:2] : (tensor<512x2xf32>) -> tensor<512x1xf32>
    %39 = stablehlo.dot_general %arg0, %37, batching_dims = [0] x [0], contracting_dims = [] x [] : (tensor<512x4096xf32>, tensor<512x8xf32>) -> tensor<512x4096x8xf32>
    %40 = stablehlo.transpose %39, dims = [2, 0, 1] : (tensor<512x4096x8xf32>) -> tensor<8x512x4096xf32>
    %41 = stablehlo.dot_general %40, %arg2, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x512x4096xf32>, tensor<8x4096x14336xf32>) -> tensor<8x512x14336xf32>
    %42 = stablehlo.dot_general %40, %arg3, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x512x4096xf32>, tensor<8x4096x14336xf32>) -> tensor<8x512x14336xf32>
    %43 = stablehlo.negate %41 : tensor<8x512x14336xf32>
    %44 = stablehlo.exponential %43 : tensor<8x512x14336xf32>
    %45 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x512x14336xf32>
    %46 = stablehlo.add %45, %44 : tensor<8x512x14336xf32>
    %47 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x512x14336xf32>
    %48 = stablehlo.divide %47, %46 : tensor<8x512x14336xf32>
    %49 = stablehlo.multiply %41, %48 : tensor<8x512x14336xf32>
    %50 = stablehlo.multiply %49, %42 : tensor<8x512x14336xf32>
    %51 = stablehlo.dot_general %50, %arg4, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x512x14336xf32>, tensor<8x14336x4096xf32>) -> tensor<8x512x4096xf32>
    %52 = stablehlo.dot_general %37, %51, batching_dims = [0] x [1], contracting_dims = [1] x [0] : (tensor<512x8xf32>, tensor<8x512x4096xf32>) -> tensor<512x4096xf32>
    %53 = stablehlo.broadcast_in_dim %38, dims = [0, 1] : (tensor<512x1xf32>) -> tensor<512x4096xf32>
    %54 = stablehlo.multiply %53, %52 : tensor<512x4096xf32>
    %55 = stablehlo.add %34, %54 : tensor<512x4096xf32>
    return %55 : tensor<512x4096xf32>
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

