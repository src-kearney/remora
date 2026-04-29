module @transpose_copy_fusion.1_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @transpose_copy_fusion.1(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 1024> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 32> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %2[2, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %8 = llvm.load %7 invariant dereferenceable<bytes = 2048> : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %10 = llvm.load %9 : !llvm.ptr -> !llvm.ptr
    %11 = llvm.getelementptr inbounds %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %12 = llvm.load %11 invariant : !llvm.ptr -> i64
    %13 = llvm.getelementptr inbounds %10[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %14 = llvm.load %13 invariant : !llvm.ptr -> i64
    %15 = llvm.getelementptr inbounds %10[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %16 = llvm.load %15 invariant : !llvm.ptr -> i64
    llvm.call @transpose_copy_fusion.1_wrapped(%4, %6, %8, %12, %14, %16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @transpose_copy_fusion.1_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 1024 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 32 : index, llvm.noalias, xla.invariant}, %arg2: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, llvm.noalias}, %arg3: i64, %arg4: i64, %arg5: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(256 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(0 : index) : i64
    %3 = llvm.mlir.constant(2 : index) : i64
    %4 = llvm.mlir.constant(8 : index) : i64
    %5 = llvm.mlir.constant(32 : index) : i64
    llvm.br ^bb1(%2 : i64)
  ^bb1(%6: i64):  // 2 preds: ^bb0, ^bb8
    %7 = llvm.icmp "slt" %6, %3 : i64
    llvm.cond_br %7, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %8 = llvm.trunc %6 : i64 to i32
    %9 = llvm.mul %6, %0 overflow<nsw> : i64
    llvm.br ^bb3(%2 : i64)
  ^bb3(%10: i64):  // 2 preds: ^bb2, ^bb7
    %11 = llvm.icmp "slt" %10, %4 : i64
    llvm.cond_br %11, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    %12 = llvm.getelementptr inbounds %arg1[0, %10] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i32>
    %13 = llvm.load %12 invariant : !llvm.ptr -> i32
    %14 = llvm.icmp "eq" %13, %8 : i32
    %15 = llvm.zext %14 : i1 to i8
    %16 = llvm.sitofp %15 : i8 to f32
    %17 = llvm.mul %10, %5 overflow<nsw> : i64
    %18 = llvm.add %9, %17 overflow<nsw> : i64
    llvm.br ^bb5(%2 : i64)
  ^bb5(%19: i64):  // 2 preds: ^bb4, ^bb6
    %20 = llvm.icmp "slt" %19, %5 : i64
    llvm.cond_br %20, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %21 = llvm.add %17, %19 overflow<nsw> : i64
    %22 = llvm.getelementptr inbounds %arg0[0, %21] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<256 x f32>
    %23 = llvm.load %22 invariant : !llvm.ptr -> f32
    %24 = llvm.fmul %23, %16 : f32
    %25 = llvm.add %18, %19 overflow<nsw> : i64
    %26 = llvm.getelementptr inbounds %arg2[0, %25] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x f32>
    llvm.store %24, %26 : f32, !llvm.ptr
    %27 = llvm.add %19, %1 : i64
    llvm.br ^bb5(%27 : i64)
  ^bb7:  // pred: ^bb5
    %28 = llvm.add %10, %1 : i64
    llvm.br ^bb3(%28 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb8:  // pred: ^bb3
    %29 = llvm.add %6, %1 : i64
    llvm.br ^bb1(%29 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb9:  // pred: ^bb1
    llvm.return
  }
}