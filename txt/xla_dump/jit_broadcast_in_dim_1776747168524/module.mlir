#loc1 = loc("args[0]")
module @jit_broadcast_in_dim attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<f32> loc("args[0]")) -> (tensor<8x32xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<8x32xf32> loc(#loc4)
    return %0 : tensor<8x32xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/Users/seanrck/github/remora/scripts/export/dump_xla_hlo.py":50:4 to :21)
#loc3 = loc("<module>"(#loc2))
#loc4 = loc("jit(broadcast_in_dim)/broadcast_in_dim"(#loc3))
