module @compare_convert_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @compare_convert_fusion(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 32> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 64> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %8 = llvm.load %7 : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %10 = llvm.load %9 invariant : !llvm.ptr -> i64
    %11 = llvm.getelementptr inbounds %8[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %12 = llvm.load %11 invariant : !llvm.ptr -> i64
    %13 = llvm.getelementptr inbounds %8[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %14 = llvm.load %13 invariant : !llvm.ptr -> i64
    llvm.call @compare_convert_fusion_wrapped(%4, %6, %10, %12, %14) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @compare_convert_fusion_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 32 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 64 : index, llvm.noalias}, %arg2: i64, %arg3: i64, %arg4: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(2 : index) : i64
    %1 = llvm.mlir.constant(8 : index) : i64
    %2 = llvm.mlir.constant(0 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%2 : i64)
  ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb5
    %5 = llvm.icmp "slt" %4, %1 : i64
    llvm.cond_br %5, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %6 = llvm.getelementptr inbounds %arg0[0, %4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i32>
    %7 = llvm.load %6 invariant : !llvm.ptr -> i32
    %8 = llvm.mul %4, %0 overflow<nsw> : i64
    llvm.br ^bb3(%2 : i64)
  ^bb3(%9: i64):  // 2 preds: ^bb2, ^bb4
    %10 = llvm.icmp "slt" %9, %0 : i64
    llvm.cond_br %10, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %11 = llvm.trunc %9 : i64 to i32
    %12 = llvm.icmp "eq" %7, %11 : i32
    %13 = llvm.zext %12 : i1 to i8
    %14 = llvm.sitofp %13 : i8 to f32
    %15 = llvm.add %8, %9 overflow<nsw> : i64
    %16 = llvm.getelementptr inbounds %arg1[0, %15] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<16 x f32>
    llvm.store %14, %16 : f32, !llvm.ptr
    %17 = llvm.add %9, %3 : i64
    llvm.br ^bb3(%17 : i64)
  ^bb5:  // pred: ^bb3
    %18 = llvm.add %4, %3 : i64
    llvm.br ^bb1(%18 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}