module @multiply_multiply_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @multiply_multiply_fusion(%arg0: tensor<1024xf32> {llvm.align = 64 : index, llvm.dereferenceable = 4096 : index, xla.slice_index = 0 : index}, %arg1: tensor<1024xf32> {llvm.align = 64 : index, llvm.dereferenceable = 4096 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<1024xf32> {llvm.align = 64 : index, llvm.dereferenceable = 4096 : index, xla.slice_index = 0 : index}) -> tensor<1024xf32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f32
    %0 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %arg2) -> (tensor<1024xf32>) {
      %1 = scf.for %arg5 = %c0 to %c8 step %c1 iter_args(%arg6 = %arg4) -> (tensor<1024xf32>) {
        %2 = scf.for %arg7 = %c0 to %c64 step %c1 iter_args(%arg8 = %arg6) -> (tensor<1024xf32>) {
          %3 = xla.apply_indexing #xla.indexing_map<"(d0, d1, d2) -> (d0 * 512 + d1 * 64 + d2), domain: d0 in [0, 1], d1 in [0, 7], d2 in [0, 63]">(%arg3, %arg5, %arg7)
          %extracted = tensor.extract %arg1[%3] : tensor<1024xf32>
          %4 = arith.negf %extracted : f32
          %5 = math.exp %4 : f32
          %6 = arith.addf %5, %cst : f32
          %7 = arith.divf %cst, %6 : f32
          %8 = arith.mulf %extracted, %7 : f32
          %extracted_0 = tensor.extract %arg0[%3] : tensor<1024xf32>
          %9 = arith.mulf %8, %extracted_0 : f32
          %inserted = tensor.insert %9 into %arg8[%3] : tensor<1024xf32>
          scf.yield %inserted : tensor<1024xf32>
        }
        scf.yield %2 : tensor<1024xf32>
      } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
      scf.yield %1 : tensor<1024xf32>
    } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
    return %0 : tensor<1024xf32>
  }
}