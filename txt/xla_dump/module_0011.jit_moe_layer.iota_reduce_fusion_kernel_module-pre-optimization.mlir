module @iota_reduce_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  func.func @iota_reduce_fusion(%arg0: tensor<8x2xf32> {llvm.align = 64 : index, llvm.dereferenceable = 64 : index, xla.invariant, xla.slice_index = 0 : index}, %arg1: tensor<8xf32> {llvm.align = 64 : index, llvm.dereferenceable = 32 : index, xla.slice_index = 1 : index}, %arg2: tensor<8xi32> {llvm.align = 64 : index, llvm.dereferenceable = 32 : index, xla.slice_index = 2 : index}) -> (tensor<8xf32>, tensor<8xi32>) attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
    %0 = xla.workgroup_id  x {xla.range = [0 : index, 0 : index]}
    %1 = xla.workgroup_id  y {xla.range = [0 : index, 0 : index]}
    %2 = xla.workgroup_id  z {xla.range = [0 : index, 0 : index]}
    %3:2 = scf.forall (%arg3, %arg4, %arg5) in (1, 1, 1) shared_outs(%arg6 = %arg1, %arg7 = %arg2) -> (tensor<8xf32>, tensor<8xi32>) {
      %xla_loop, %xla_loop_0 = xla.loop (%arg3, %arg4, %arg5, %0, %1, %2)[%i] -> (%ra) in #xla.indexing_map<"(th_x, th_y, th_z, bl_x, bl_y, bl_z)[s0] -> (s0), domain: th_x in [0, 0], th_y in [0, 0], th_z in [0, 0], bl_x in [0, 0], bl_y in [0, 0], bl_z in [0, 0], s0 in [0, 7]"> iter_args(%iter = %arg6, %iter_1 = %arg7) -> (tensor<8xf32>, tensor<8xi32>) {
        %pure_call, %pure_call_2 = xla.pure_call @fused_computation_5_reduce_2(%arg0, %ra) : (tensor<8x2xf32>, index) -> (f32, i32)
        %inserted = tensor.insert %pure_call into %iter[%ra] : tensor<8xf32>
        %inserted_3 = tensor.insert %pure_call_2 into %iter_1[%ra] : tensor<8xi32>
        xla.yield %inserted, %inserted_3 : tensor<8xf32>, tensor<8xi32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %xla_loop into %arg6[0] [8] [1] : tensor<8xf32> into tensor<8xf32>
        tensor.parallel_insert_slice %xla_loop_0 into %arg7[0] [8] [1] : tensor<8xi32> into tensor<8xi32>
      }
    }
    return %3#0, %3#1 : tensor<8xf32>, tensor<8xi32>
  }
  func.func private @fused_computation_5_reduce_2(%arg0: tensor<8x2xf32>, %arg1: index {xla.range = [0 : index, 7 : index]}) -> (f32, i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 0xFF800000 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %0:2 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %cst, %arg4 = %c0_i32) -> (f32, i32) {
      %true = arith.constant true
      %c0_0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %1 = arith.cmpi sge, %arg1, %c0_0 : index
      %2 = arith.cmpi sle, %arg1, %c7 : index
      %3 = arith.andi %1, %2 : i1
      %4 = arith.andi %true, %3 : i1
      %5:2 = scf.if %4 -> (f32, i32) {
        %extracted = tensor.extract %arg0[%arg1, %arg2] : tensor<8x2xf32>
        %6 = arith.index_castui %arg2 : index to i64
        %7 = arith.trunci %6 : i64 to i32
        %8:2 = func.call @region_0_1_tuple_1(%arg3, %arg4, %extracted, %7) {xla.is_reduction} : (f32, i32, f32, i32) -> (f32, i32)
        scf.yield %8#0, %8#1 : f32, i32
      } else {
        scf.yield %arg3, %arg4 : f32, i32
      }
      scf.yield %5#0, %5#1 : f32, i32
    }
    return %0#0, %0#1 : f32, i32
  }
  func.func private @region_0_1_tuple_1(%arg0: f32, %arg1: i32, %arg2: f32, %arg3: i32) -> (f32, i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = arith.cmpf ogt, %arg0, %arg2 : f32
    %1 = arith.extui %0 : i1 to i8
    %2 = arith.cmpf une, %arg0, %arg0 : f32
    %3 = arith.extui %2 : i1 to i8
    %4 = arith.ori %1, %3 : i8
    %5 = arith.trunci %4 : i8 to i1
    %6 = arith.select %5, %arg0, %arg2 : f32
    %7 = arith.cmpf oeq, %arg0, %arg2 : f32
    %8 = arith.extui %7 : i1 to i8
    %9 = arith.cmpi slt, %arg1, %arg3 : i32
    %10 = arith.extui %9 : i1 to i8
    %11 = arith.andi %8, %10 : i8
    %12 = arith.ori %4, %11 : i8
    %13 = arith.trunci %12 : i8 to i1
    %14 = arith.select %13, %arg1, %arg3 : i32
    return %6, %14 : f32, i32
  }
}