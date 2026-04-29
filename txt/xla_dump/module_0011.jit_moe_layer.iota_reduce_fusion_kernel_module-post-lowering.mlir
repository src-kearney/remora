module @iota_reduce_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @iota_reduce_fusion(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 64> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 32> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %2[2, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %8 = llvm.load %7 invariant dereferenceable<bytes = 32> : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %10 = llvm.load %9 : !llvm.ptr -> !llvm.ptr
    %11 = llvm.getelementptr inbounds %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %12 = llvm.load %11 invariant : !llvm.ptr -> i64
    %13 = llvm.getelementptr inbounds %10[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %14 = llvm.load %13 invariant : !llvm.ptr -> i64
    %15 = llvm.getelementptr inbounds %10[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %16 = llvm.load %15 invariant : !llvm.ptr -> i64
    llvm.call @iota_reduce_fusion_wrapped(%4, %6, %8, %12, %14, %16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @iota_reduce_fusion_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 64 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 32 : index, llvm.noalias}, %arg2: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 32 : index, llvm.noalias}, %arg3: i64, %arg4: i64, %arg5: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(2 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(0xFF800000 : f32) : f32
    %5 = llvm.mlir.constant(8 : index) : i64
    llvm.br ^bb1(%1 : i64)
  ^bb1(%6: i64):  // 2 preds: ^bb0, ^bb5
    %7 = llvm.icmp "slt" %6, %5 : i64
    llvm.cond_br %7, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %8 = llvm.mul %6, %0 overflow<nsw> : i64
    llvm.br ^bb3(%1, %4, %3 : i64, f32, i32)
  ^bb3(%9: i64, %10: f32, %11: i32):  // 2 preds: ^bb2, ^bb4
    %12 = llvm.icmp "slt" %9, %0 : i64
    llvm.cond_br %12, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %13 = llvm.add %8, %9 overflow<nsw> : i64
    %14 = llvm.getelementptr inbounds %arg0[0, %13] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<16 x f32>
    %15 = llvm.load %14 invariant : !llvm.ptr -> f32
    %16 = llvm.trunc %9 : i64 to i32
    %17 = llvm.fcmp "ogt" %10, %15 : f32
    %18 = llvm.fcmp "une" %10, %10 : f32
    %19 = llvm.or %17, %18 : i1
    %20 = llvm.select %19, %10, %15 : i1, f32
    %21 = llvm.fcmp "oeq" %10, %15 : f32
    %22 = llvm.icmp "slt" %11, %16 : i32
    %23 = llvm.and %21, %22 : i1
    %24 = llvm.or %19, %23 : i1
    %25 = llvm.select %24, %11, %16 : i1, i32
    %26 = llvm.add %9, %2 : i64
    llvm.br ^bb3(%26, %20, %25 : i64, f32, i32)
  ^bb5:  // pred: ^bb3
    %27 = llvm.getelementptr inbounds %arg1[0, %6] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x f32>
    llvm.store %10, %27 : f32, !llvm.ptr
    %28 = llvm.getelementptr inbounds %arg2[0, %6] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i32>
    llvm.store %11, %28 : i32, !llvm.ptr
    %29 = llvm.add %6, %2 : i64
    llvm.br ^bb1(%29 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}