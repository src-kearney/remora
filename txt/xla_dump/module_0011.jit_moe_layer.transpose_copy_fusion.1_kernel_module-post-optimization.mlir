module @transpose_copy_fusion.1_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @transpose_copy_fusion.1(%arg0: tensor<256xf32> {llvm.align = 64 : index, llvm.dereferenceable = 1024 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<8xi32> {llvm.align = 64 : index, llvm.dereferenceable = 32 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<512xf32> {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, xla.slice_index = 2 : index}) -> tensor<512xf32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %arg2) -> (tensor<512xf32>) {
      %1 = arith.index_castui %arg3 : index to i64
      %2 = arith.trunci %1 : i64 to i32
      %3 = scf.for %arg5 = %c0 to %c8 step %c1 iter_args(%arg6 = %arg4) -> (tensor<512xf32>) {
        %extracted = tensor.extract %arg1[%arg5] : tensor<8xi32>
        %4 = arith.cmpi eq, %extracted, %2 : i32
        %5 = arith.extui %4 : i1 to i8
        %6 = arith.sitofp %5 : i8 to f32
        %7 = scf.for %arg7 = %c0 to %c32 step %c1 iter_args(%arg8 = %arg6) -> (tensor<512xf32>) {
          %8 = xla.apply_indexing #xla.indexing_map<"(d0, d1) -> (d0 * 32 + d1), domain: d0 in [0, 7], d1 in [0, 31]">(%arg5, %arg7)
          %extracted_0 = tensor.extract %arg0[%8] : tensor<256xf32>
          %9 = arith.mulf %extracted_0, %6 : f32
          %10 = xla.apply_indexing #xla.indexing_map<"(d0, d1, d2) -> (d0 * 256 + d1 * 32 + d2), domain: d0 in [0, 1], d1 in [0, 7], d2 in [0, 31]">(%arg3, %arg5, %arg7)
          %inserted = tensor.insert %9 into %arg8[%10] : tensor<512xf32>
          scf.yield %inserted : tensor<512xf32>
        }
        scf.yield %7 : tensor<512xf32>
      } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
      scf.yield %3 : tensor<512xf32>
    } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
    return %0 : tensor<512xf32>
  }
}