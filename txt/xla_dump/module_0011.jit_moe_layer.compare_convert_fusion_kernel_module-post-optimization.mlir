module @compare_convert_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @compare_convert_fusion(%arg0: tensor<8xi32> {llvm.align = 64 : index, llvm.dereferenceable = 32 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<16xf32> {llvm.align = 64 : index, llvm.dereferenceable = 64 : index, xla.slice_index = 1 : index}) -> tensor<16xf32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %0 = scf.for %arg2 = %c0 to %c8 step %c1 iter_args(%arg3 = %arg1) -> (tensor<16xf32>) {
      %extracted = tensor.extract %arg0[%arg2] : tensor<8xi32>
      %1 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg3) -> (tensor<16xf32>) {
        %2 = arith.index_castui %arg4 : index to i64
        %3 = arith.trunci %2 : i64 to i32
        %4 = arith.cmpi eq, %extracted, %3 : i32
        %5 = arith.extui %4 : i1 to i8
        %6 = arith.sitofp %5 : i8 to f32
        %7 = xla.apply_indexing #xla.indexing_map<"(d0, d1) -> (d0 * 2 + d1), domain: d0 in [0, 7], d1 in [0, 1]">(%arg2, %arg4)
        %inserted = tensor.insert %6 into %arg5[%7] : tensor<16xf32>
        scf.yield %inserted : tensor<16xf32>
      }
      scf.yield %1 : tensor<16xf32>
    } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
    return %0 : tensor<16xf32>
  }
}