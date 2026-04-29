module @multiply_multiply_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @multiply_multiply_fusion(%arg0: tensor<2x8x64xf32> {llvm.align = 64 : index, llvm.dereferenceable = 4096 : index, xla.slice_index = 0 : index}, %arg1: tensor<2x8x64xf32> {llvm.align = 64 : index, llvm.dereferenceable = 4096 : index, xla.invariant, xla.slice_index = 1 : index}, %arg2: tensor<2x8x64xf32> {llvm.align = 64 : index, llvm.dereferenceable = 4096 : index, xla.slice_index = 0 : index}) -> tensor<2x8x64xf32> attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %0 = xla.workgroup_id  x {xla.range = [0 : index, 0 : index]}
    %1 = xla.workgroup_id  y {xla.range = [0 : index, 0 : index]}
    %2 = xla.workgroup_id  z {xla.range = [0 : index, 0 : index]}
    %3 = scf.forall (%arg3, %arg4, %arg5) in (1, 1, 1) shared_outs(%arg6 = %arg2) -> (tensor<2x8x64xf32>) {
      %xla_loop = xla.loop (%arg3, %arg4, %arg5, %0, %1, %2)[%i, %j, %k] -> (%ra, %rb, %rc) in #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z)[s0, s1, s2] -> (s0, s1, s2), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0], s0 in [0, 1], s1 in [0, 7], s2 in [0, 63]"> iter_args(%iter = %arg6) -> (tensor<2x8x64xf32>) {
        %pure_call = xla.pure_call @fused_computation_1_mul_0(%arg0, %arg1, %ra, %rb, %rc) : (tensor<2x8x64xf32>, tensor<2x8x64xf32>, index, index, index) -> f32
        %inserted = tensor.insert %pure_call into %iter[%ra, %rb, %rc] : tensor<2x8x64xf32>
        xla.yield %inserted : tensor<2x8x64xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %xla_loop into %arg6[0, 0, 0] [2, 8, 64] [1, 1, 1] : tensor<2x8x64xf32> into tensor<2x8x64xf32>
      }
    }
    return %3 : tensor<2x8x64xf32>
  }
  func.func private @fused_computation_1_mul_0(%arg0: tensor<2x8x64xf32>, %arg1: tensor<2x8x64xf32>, %arg2: index {xla.range = [0 : index, 1 : index]}, %arg3: index {xla.range = [0 : index, 7 : index]}, %arg4: index {xla.range = [0 : index, 63 : index]}) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %extracted = tensor.extract %arg1[%arg2, %arg3, %arg4] : tensor<2x8x64xf32>
    %0 = arith.negf %extracted : f32
    %1 = math.exp %0 : f32
    %cst = arith.constant 1.000000e+00 : f32
    %2 = arith.addf %1, %cst : f32
    %extracted_0 = tensor.extract %arg1[%arg2, %arg3, %arg4] : tensor<2x8x64xf32>
    %3 = arith.divf %cst, %2 : f32
    %4 = arith.mulf %extracted_0, %3 : f32
    %extracted_1 = tensor.extract %arg0[%arg2, %arg3, %arg4] : tensor<2x8x64xf32>
    %5 = arith.mulf %4, %extracted_1 : f32
    return %5 : f32
  }
}