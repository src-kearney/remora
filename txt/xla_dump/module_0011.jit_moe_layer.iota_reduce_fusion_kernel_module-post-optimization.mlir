module @iota_reduce_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @iota_reduce_fusion(%arg0: tensor<16xf32> {llvm.align = 64 : index, llvm.dereferenceable = 64 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<8xf32> {llvm.align = 64 : index, llvm.dereferenceable = 32 : index, xla.slice_index = 1 : index}, %arg2: tensor<8xi32> {llvm.align = 64 : index, llvm.dereferenceable = 32 : index, xla.slice_index = 2 : index}) -> (tensor<8xf32>, tensor<8xi32>) attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0xFF800000 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %0:2 = scf.for %arg3 = %c0 to %c8 step %c1 iter_args(%arg4 = %arg1, %arg5 = %arg2) -> (tensor<8xf32>, tensor<8xi32>) {
      %1:2 = scf.for %arg6 = %c0 to %c2 step %c1 iter_args(%arg7 = %cst, %arg8 = %c0_i32) -> (f32, i32) {
        %2 = xla.apply_indexing #xla.indexing_map<"(d0, d1) -> (d0 * 2 + d1), domain: d0 in [0, 7], d1 in [0, 1]">(%arg3, %arg6)
        %extracted = tensor.extract %arg0[%2] : tensor<16xf32>
        %3 = arith.index_castui %arg6 : index to i64
        %4 = arith.trunci %3 : i64 to i32
        %5 = arith.cmpf ogt, %arg7, %extracted : f32
        %6 = arith.cmpf une, %arg7, %arg7 : f32
        %7 = arith.ori %5, %6 : i1
        %8 = arith.select %7, %arg7, %extracted : f32
        %9 = arith.cmpf oeq, %arg7, %extracted : f32
        %10 = arith.cmpi slt, %arg8, %4 : i32
        %11 = arith.andi %9, %10 : i1
        %12 = arith.ori %7, %11 : i1
        %13 = arith.select %12, %arg8, %4 : i32
        scf.yield %8, %13 : f32, i32
      }
      %inserted = tensor.insert %1#0 into %arg4[%arg3] : tensor<8xf32>
      %inserted_0 = tensor.insert %1#1 into %arg5[%arg3] : tensor<8xi32>
      scf.yield %inserted, %inserted_0 : tensor<8xf32>, tensor<8xi32>
    } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
    return %0#0, %0#1 : tensor<8xf32>, tensor<8xi32>
  }
}