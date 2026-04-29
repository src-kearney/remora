; ModuleID = '__compute_module_transpose_copy_fusion.1_kernel_module'
source_filename = "__compute_module_transpose_copy_fusion.1_kernel_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin23.5.0"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

; Function Attrs: uwtable
define ptr @transpose_copy_fusion.1(ptr %0) #0 {
  %2 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !4
  %6 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 1, i32 0
  %7 = load ptr, ptr %6, align 8, !invariant.load !3, !dereferenceable !5
  %8 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 2, i32 0
  %9 = load ptr, ptr %8, align 8, !invariant.load !3, !dereferenceable !6
  %10 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %11 = load ptr, ptr %10, align 8
  %12 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 0
  %13 = load i64, ptr %12, align 4, !invariant.load !3
  %14 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 1
  %15 = load i64, ptr %14, align 4, !invariant.load !3
  %16 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 2
  %17 = load i64, ptr %16, align 4, !invariant.load !3
  call void @transpose_copy_fusion.1_wrapped(ptr %5, ptr %7, ptr %9, i64 %13, i64 %15, i64 %17)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @transpose_copy_fusion.1_wrapped(ptr noalias align 64 dereferenceable(1024) %0, ptr noalias align 64 dereferenceable(32) %1, ptr noalias align 64 dereferenceable(2048) %2, i64 %3, i64 %4, i64 %5) #1 {
  br label %7

7:                                                ; preds = %37, %6
  %8 = phi i64 [ %38, %37 ], [ 0, %6 ]
  %9 = icmp slt i64 %8, 2
  br i1 %9, label %10, label %39

10:                                               ; preds = %7
  %11 = trunc i64 %8 to i32
  %12 = mul nsw i64 %8, 256
  br label %13

13:                                               ; preds = %35, %10
  %14 = phi i64 [ %36, %35 ], [ 0, %10 ]
  %15 = icmp slt i64 %14, 8
  br i1 %15, label %16, label %37

16:                                               ; preds = %13
  %17 = getelementptr inbounds [8 x i32], ptr %1, i32 0, i64 %14
  %18 = load i32, ptr %17, align 4, !invariant.load !3
  %19 = icmp eq i32 %18, %11
  %20 = zext i1 %19 to i8
  %21 = sitofp i8 %20 to float
  %22 = mul nsw i64 %14, 32
  %23 = add nsw i64 %12, %22
  br label %24

24:                                               ; preds = %27, %16
  %25 = phi i64 [ %34, %27 ], [ 0, %16 ]
  %26 = icmp slt i64 %25, 32
  br i1 %26, label %27, label %35

27:                                               ; preds = %24
  %28 = add nsw i64 %22, %25
  %29 = getelementptr inbounds [256 x float], ptr %0, i32 0, i64 %28
  %30 = load float, ptr %29, align 4, !invariant.load !3
  %31 = fmul float %30, %21
  %32 = add nsw i64 %23, %25
  %33 = getelementptr inbounds [512 x float], ptr %2, i32 0, i64 %32
  store float %31, ptr %33, align 4
  %34 = add i64 %25, 1
  br label %24

35:                                               ; preds = %24
  %36 = add i64 %14, 1
  br label %13, !llvm.loop !7

37:                                               ; preds = %13
  %38 = add i64 %8, 1
  br label %7, !llvm.loop !7

39:                                               ; preds = %7
  ret void
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 5}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 1024}
!5 = !{i64 32}
!6 = !{i64 2048}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.unroll.disable"}
