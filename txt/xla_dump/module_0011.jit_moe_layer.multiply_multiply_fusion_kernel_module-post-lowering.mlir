module @multiply_multiply_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @multiply_multiply_fusion(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 4096> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 4096> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %2[2, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %8 = llvm.load %7 invariant dereferenceable<bytes = 4096> : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %10 = llvm.load %9 : !llvm.ptr -> !llvm.ptr
    %11 = llvm.getelementptr inbounds %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %12 = llvm.load %11 invariant : !llvm.ptr -> i64
    %13 = llvm.getelementptr inbounds %10[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %14 = llvm.load %13 invariant : !llvm.ptr -> i64
    %15 = llvm.getelementptr inbounds %10[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %16 = llvm.load %15 invariant : !llvm.ptr -> i64
    llvm.call @multiply_multiply_fusion_wrapped(%4, %6, %8, %12, %14, %16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @multiply_multiply_fusion_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4096 : index, llvm.noalias}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4096 : index, llvm.noalias, xla.invariant}, %arg2: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4096 : index, llvm.noalias}, %arg3: i64, %arg4: i64, %arg5: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(512 : index) : i64
    %1 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(2 : index) : i64
    %5 = llvm.mlir.constant(8 : index) : i64
    %6 = llvm.mlir.constant(64 : index) : i64
    llvm.br ^bb1(%3 : i64)
  ^bb1(%7: i64):  // 2 preds: ^bb0, ^bb8
    %8 = llvm.icmp "slt" %7, %4 : i64
    llvm.cond_br %8, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %9 = llvm.mul %7, %0 overflow<nsw> : i64
    llvm.br ^bb3(%3 : i64)
  ^bb3(%10: i64):  // 2 preds: ^bb2, ^bb7
    %11 = llvm.icmp "slt" %10, %5 : i64
    llvm.cond_br %11, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    %12 = llvm.mul %10, %6 overflow<nsw> : i64
    %13 = llvm.add %9, %12 overflow<nsw> : i64
    llvm.br ^bb5(%3 : i64)
  ^bb5(%14: i64):  // 2 preds: ^bb4, ^bb6
    %15 = llvm.icmp "slt" %14, %6 : i64
    llvm.cond_br %15, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %16 = llvm.add %13, %14 overflow<nsw> : i64
    %17 = llvm.getelementptr inbounds %arg1[0, %16] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1024 x f32>
    %18 = llvm.load %17 invariant : !llvm.ptr -> f32
    %19 = llvm.fneg %18 : f32
    %20 = llvm.intr.exp(%19) : (f32) -> f32
    %21 = llvm.fadd %20, %1 : f32
    %22 = llvm.fdiv %1, %21 : f32
    %23 = llvm.fmul %18, %22 : f32
    %24 = llvm.getelementptr inbounds %arg0[0, %16] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1024 x f32>
    %25 = llvm.load %24 : !llvm.ptr -> f32
    %26 = llvm.fmul %23, %25 : f32
    llvm.store %26, %24 : f32, !llvm.ptr
    %27 = llvm.add %14, %2 : i64
    llvm.br ^bb5(%27 : i64)
  ^bb7:  // pred: ^bb5
    %28 = llvm.add %10, %2 : i64
    llvm.br ^bb3(%28 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb8:  // pred: ^bb3
    %29 = llvm.add %7, %2 : i64
    llvm.br ^bb1(%29 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb9:  // pred: ^bb1
    llvm.return
  }
}