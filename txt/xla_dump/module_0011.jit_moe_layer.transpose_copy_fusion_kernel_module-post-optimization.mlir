module @transpose_copy_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @transpose_copy_fusion(%arg0: tensor<512xf32> {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<512xf32> {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, xla.slice_index = 1 : index}) -> tensor<512xf32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = scf.for %arg2 = %c0 to %c8 step %c1 iter_args(%arg3 = %arg1) -> (tensor<512xf32>) {
      %1 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg3) -> (tensor<512xf32>) {
        %2 = scf.for %arg6 = %c0 to %c32 step %c1 iter_args(%arg7 = %arg5) -> (tensor<512xf32>) {
          %3 = xla.apply_indexing #xla.indexing_map<"(d0, d1, d2) -> (d0 * 256 + d1 * 32 + d2), domain: d0 in [0, 1], d1 in [0, 7], d2 in [0, 31]">(%arg4, %arg2, %arg6)
          %extracted = tensor.extract %arg0[%3] : tensor<512xf32>
          %4 = xla.apply_indexing #xla.indexing_map<"(d0, d1, d2) -> (d0 * 64 + d1 * 32 + d2), domain: d0 in [0, 7], d1 in [0, 1], d2 in [0, 31]">(%arg2, %arg4, %arg6)
          %inserted = tensor.insert %extracted into %arg7[%4] : tensor<512xf32>
          scf.yield %inserted : tensor<512xf32>
        }
        scf.yield %2 : tensor<512xf32>
      } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
      scf.yield %1 : tensor<512xf32>
    } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
    return %0 : tensor<512xf32>
  }
}