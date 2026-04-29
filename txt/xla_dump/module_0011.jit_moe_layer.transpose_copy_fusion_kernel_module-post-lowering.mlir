module @transpose_copy_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @transpose_copy_fusion(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 2048> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 2048> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %8 = llvm.load %7 : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %10 = llvm.load %9 invariant : !llvm.ptr -> i64
    %11 = llvm.getelementptr inbounds %8[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %12 = llvm.load %11 invariant : !llvm.ptr -> i64
    %13 = llvm.getelementptr inbounds %8[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %14 = llvm.load %13 invariant : !llvm.ptr -> i64
    llvm.call @transpose_copy_fusion_wrapped(%4, %6, %10, %12, %14) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @transpose_copy_fusion_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 2048 : index, llvm.noalias}, %arg2: i64, %arg3: i64, %arg4: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(64 : index) : i64
    %1 = llvm.mlir.constant(256 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(8 : index) : i64
    %5 = llvm.mlir.constant(2 : index) : i64
    %6 = llvm.mlir.constant(32 : index) : i64
    llvm.br ^bb1(%3 : i64)
  ^bb1(%7: i64):  // 2 preds: ^bb0, ^bb8
    %8 = llvm.icmp "slt" %7, %4 : i64
    llvm.cond_br %8, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %9 = llvm.mul %7, %6 overflow<nsw> : i64
    %10 = llvm.mul %7, %0 overflow<nsw> : i64
    llvm.br ^bb3(%3 : i64)
  ^bb3(%11: i64):  // 2 preds: ^bb2, ^bb7
    %12 = llvm.icmp "slt" %11, %5 : i64
    llvm.cond_br %12, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    %13 = llvm.mul %11, %1 overflow<nsw> : i64
    %14 = llvm.add %9, %13 overflow<nsw> : i64
    %15 = llvm.mul %11, %6 overflow<nsw> : i64
    %16 = llvm.add %10, %15 overflow<nsw> : i64
    llvm.br ^bb5(%3 : i64)
  ^bb5(%17: i64):  // 2 preds: ^bb4, ^bb6
    %18 = llvm.icmp "slt" %17, %6 : i64
    llvm.cond_br %18, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %19 = llvm.add %14, %17 overflow<nsw> : i64
    %20 = llvm.getelementptr inbounds %arg0[0, %19] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x f32>
    %21 = llvm.load %20 invariant : !llvm.ptr -> f32
    %22 = llvm.add %16, %17 overflow<nsw> : i64
    %23 = llvm.getelementptr inbounds %arg1[0, %22] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x f32>
    llvm.store %21, %23 : f32, !llvm.ptr
    %24 = llvm.add %17, %2 : i64
    llvm.br ^bb5(%24 : i64)
  ^bb7:  // pred: ^bb5
    %25 = llvm.add %11, %2 : i64
    llvm.br ^bb3(%25 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb8:  // pred: ^bb3
    %26 = llvm.add %7, %2 : i64
    llvm.br ^bb1(%26 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb9:  // pred: ^bb1
    llvm.return
  }
}