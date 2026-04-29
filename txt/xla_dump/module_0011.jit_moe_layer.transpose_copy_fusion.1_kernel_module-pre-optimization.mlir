module @transpose_copy_fusion.1_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @transpose_copy_fusion.1(%arg0: tensor<8x32xf32> {llvm.align = 64 : index, llvm.dereferenceable = 1024 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<8xi32> {llvm.align = 64 : index, llvm.dereferenceable = 32 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<2x8x32xf32> {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, xla.slice_index = 2 : index}) -> tensor<2x8x32xf32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %0 = xla.workgroup_id  x {xla.range = [0 : index, 0 : index]}
    %1 = xla.workgroup_id  y {xla.range = [0 : index, 0 : index]}
    %2 = xla.workgroup_id  z {xla.range = [0 : index, 0 : index]}
    %3 = scf.forall (%arg3, %arg4, %arg5) in (1, 1, 1) shared_outs(%arg6 = %arg2) -> (tensor<2x8x32xf32>) {
      %xla_loop = xla.loop (%arg3, %arg4, %arg5, %0, %1, %2)[%i, %j, %k] -> (%ra, %rb, %rc) in #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z)[s0, s1, s2] -> (s0, s1, s2), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0], s0 in [0, 1], s1 in [0, 7], s2 in [0, 31]"> iter_args(%iter = %arg6) -> (tensor<2x8x32xf32>) {
        %pure_call = xla.pure_call @fused_computation_2_copy_4(%arg0, %arg1, %ra, %rb, %rc) : (tensor<8x32xf32>, tensor<8xi32>, index, index, index) -> f32
        %inserted = tensor.insert %pure_call into %iter[%ra, %rb, %rc] : tensor<2x8x32xf32>
        xla.yield %inserted : tensor<2x8x32xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %xla_loop into %arg6[0, 0, 0] [2, 8, 32] [1, 1, 1] : tensor<2x8x32xf32> into tensor<2x8x32xf32>
      }
    }
    return %3 : tensor<2x8x32xf32>
  }
  func.func private @fused_computation_2_copy_4(%arg0: tensor<8x32xf32>, %arg1: tensor<8xi32>, %arg2: index {xla.range = [0 : index, 1 : index]}, %arg3: index {xla.range = [0 : index, 7 : index]}, %arg4: index {xla.range = [0 : index, 31 : index]}) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %extracted = tensor.extract %arg0[%arg3, %arg4] : tensor<8x32xf32>
    %extracted_0 = tensor.extract %arg1[%arg3] : tensor<8xi32>
    %0 = arith.index_castui %arg2 : index to i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.cmpi eq, %extracted_0, %1 : i32
    %3 = arith.extui %2 : i1 to i8
    %4 = arith.sitofp %3 : i8 to f32
    %5 = arith.mulf %extracted, %4 : f32
    return %5 : f32
  }
}