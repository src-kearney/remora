module @wrapped_broadcast_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @wrapped_broadcast(%arg0: tensor<f32> {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<256xf32> {llvm.align = 64 : index, llvm.dereferenceable = 1024 : index, xla.slice_index = 1 : index}) -> tensor<256xf32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %extracted = tensor.extract %arg0[] : tensor<f32>
    %0 = scf.for %arg2 = %c0 to %c8 step %c1 iter_args(%arg3 = %arg1) -> (tensor<256xf32>) {
      %1 = scf.for %arg4 = %c0 to %c32 step %c1 iter_args(%arg5 = %arg3) -> (tensor<256xf32>) {
        %2 = xla.apply_indexing #xla.indexing_map<"(d0, d1) -> (d0 * 32 + d1), domain: d0 in [0, 7], d1 in [0, 31]">(%arg2, %arg4)
        %inserted = tensor.insert %extracted into %arg5[%2] : tensor<256xf32>
        scf.yield %inserted : tensor<256xf32>
      }
      scf.yield %1 : tensor<256xf32>
    } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
    return %0 : tensor<256xf32>
  }
}