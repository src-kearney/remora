module @wrapped_broadcast_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @wrapped_broadcast(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 4> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 16384> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %8 = llvm.load %7 : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %10 = llvm.load %9 invariant : !llvm.ptr -> i64
    %11 = llvm.getelementptr inbounds %8[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %12 = llvm.load %11 invariant : !llvm.ptr -> i64
    %13 = llvm.getelementptr inbounds %8[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %14 = llvm.load %13 invariant : !llvm.ptr -> i64
    llvm.call @wrapped_broadcast_wrapped(%4, %6, %10, %12, %14) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @wrapped_broadcast_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 4 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, llvm.noalias}, %arg2: i64, %arg3: i64, %arg4: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(2048 : index) : i64
    %1 = llvm.mlir.constant(32 : index) : i64
    %2 = llvm.mlir.constant(64 : index) : i64
    %3 = llvm.mlir.constant(2 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x f32>
    %7 = llvm.load %6 invariant : !llvm.ptr -> f32
    llvm.br ^bb1(%4 : i64)
  ^bb1(%8: i64):  // 2 preds: ^bb0, ^bb8
    %9 = llvm.icmp "slt" %8, %3 : i64
    llvm.cond_br %9, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %10 = llvm.mul %8, %0 overflow<nsw> : i64
    llvm.br ^bb3(%4 : i64)
  ^bb3(%11: i64):  // 2 preds: ^bb2, ^bb7
    %12 = llvm.icmp "slt" %11, %2 : i64
    llvm.cond_br %12, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    %13 = llvm.mul %11, %1 overflow<nsw> : i64
    %14 = llvm.add %10, %13 overflow<nsw> : i64
    llvm.br ^bb5(%4 : i64)
  ^bb5(%15: i64):  // 2 preds: ^bb4, ^bb6
    %16 = llvm.icmp "slt" %15, %1 : i64
    llvm.cond_br %16, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %17 = llvm.add %14, %15 overflow<nsw> : i64
    %18 = llvm.getelementptr inbounds %arg1[0, %17] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x f32>
    llvm.store %7, %18 : f32, !llvm.ptr
    %19 = llvm.add %15, %5 : i64
    llvm.br ^bb5(%19 : i64)
  ^bb7:  // pred: ^bb5
    %20 = llvm.add %11, %5 : i64
    llvm.br ^bb3(%20 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb8:  // pred: ^bb3
    %21 = llvm.add %8, %5 : i64
    llvm.br ^bb1(%21 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb9:  // pred: ^bb1
    llvm.return
  }
}