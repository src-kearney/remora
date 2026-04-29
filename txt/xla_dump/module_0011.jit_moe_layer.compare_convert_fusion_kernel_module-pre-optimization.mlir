module @compare_convert_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @compare_convert_fusion(%arg0: tensor<8xi32> {llvm.align = 64 : index, llvm.dereferenceable = 32 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<8x2xf32> {llvm.align = 64 : index, llvm.dereferenceable = 64 : index, xla.slice_index = 1 : index}) -> tensor<8x2xf32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %0 = xla.workgroup_id  x {xla.range = [0 : index, 0 : index]}
    %1 = xla.workgroup_id  y {xla.range = [0 : index, 0 : index]}
    %2 = xla.workgroup_id  z {xla.range = [0 : index, 0 : index]}
    %3 = scf.forall (%arg2, %arg3, %arg4) in (1, 1, 1) shared_outs(%arg5 = %arg1) -> (tensor<8x2xf32>) {
      %xla_loop = xla.loop (%arg2, %arg3, %arg4, %0, %1, %2)[%i, %j] -> (%ra, %rb) in #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z)[s0, s1] -> (s0, s1), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0], s0 in [0, 7], s1 in [0, 1]"> iter_args(%iter = %arg5) -> (tensor<8x2xf32>) {
        %pure_call = xla.pure_call @fused_computation_4_convert_element_type_4(%arg0, %ra, %rb) : (tensor<8xi32>, index, index) -> f32
        %inserted = tensor.insert %pure_call into %iter[%ra, %rb] : tensor<8x2xf32>
        xla.yield %inserted : tensor<8x2xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %xla_loop into %arg5[0, 0] [8, 2] [1, 1] : tensor<8x2xf32> into tensor<8x2xf32>
      }
    }
    return %3 : tensor<8x2xf32>
  }
  func.func private @fused_computation_4_convert_element_type_4(%arg0: tensor<8xi32>, %arg1: index {xla.range = [0 : index, 7 : index]}, %arg2: index {xla.range = [0 : index, 1 : index]}) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %extracted = tensor.extract %arg0[%arg1] : tensor<8xi32>
    %0 = arith.index_castui %arg2 : index to i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.cmpi eq, %extracted, %1 : i32
    %3 = arith.extui %2 : i1 to i8
    %4 = arith.sitofp %3 : i8 to f32
    return %4 : f32
  }
}