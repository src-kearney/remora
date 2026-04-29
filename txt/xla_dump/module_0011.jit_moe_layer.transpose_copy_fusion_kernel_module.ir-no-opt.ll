; ModuleID = '__compute_module_transpose_copy_fusion_kernel_module'
source_filename = "__compute_module_transpose_copy_fusion_kernel_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin23.5.0"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

; Function Attrs: uwtable
define ptr @transpose_copy_fusion(ptr %0) #0 {
  %2 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !4
  %6 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 1, i32 0
  %7 = load ptr, ptr %6, align 8, !invariant.load !3, !dereferenceable !4
  %8 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %9 = load ptr, ptr %8, align 8
  %10 = getelementptr inbounds %kernel_dim3, ptr %9, i32 0, i32 0
  %11 = load i64, ptr %10, align 4, !invariant.load !3
  %12 = getelementptr inbounds %kernel_dim3, ptr %9, i32 0, i32 1
  %13 = load i64, ptr %12, align 4, !invariant.load !3
  %14 = getelementptr inbounds %kernel_dim3, ptr %9, i32 0, i32 2
  %15 = load i64, ptr %14, align 4, !invariant.load !3
  call void @transpose_copy_fusion_wrapped(ptr %5, ptr %7, i64 %11, i64 %13, i64 %15)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @transpose_copy_fusion_wrapped(ptr noalias align 64 dereferenceable(2048) %0, ptr noalias align 64 dereferenceable(2048) %1, i64 %2, i64 %3, i64 %4) #1 {
  br label %6

6:                                                ; preds = %32, %5
  %7 = phi i64 [ %33, %32 ], [ 0, %5 ]
  %8 = icmp slt i64 %7, 8
  br i1 %8, label %9, label %34

9:                                                ; preds = %6
  %10 = mul nsw i64 %7, 32
  %11 = mul nsw i64 %7, 64
  br label %12

12:                                               ; preds = %30, %9
  %13 = phi i64 [ %31, %30 ], [ 0, %9 ]
  %14 = icmp slt i64 %13, 2
  br i1 %14, label %15, label %32

15:                                               ; preds = %12
  %16 = mul nsw i64 %13, 256
  %17 = add nsw i64 %10, %16
  %18 = mul nsw i64 %13, 32
  %19 = add nsw i64 %11, %18
  br label %20

20:                                               ; preds = %23, %15
  %21 = phi i64 [ %29, %23 ], [ 0, %15 ]
  %22 = icmp slt i64 %21, 32
  br i1 %22, label %23, label %30

23:                                               ; preds = %20
  %24 = add nsw i64 %17, %21
  %25 = getelementptr inbounds [512 x float], ptr %0, i32 0, i64 %24
  %26 = load float, ptr %25, align 4, !invariant.load !3
  %27 = add nsw i64 %19, %21
  %28 = getelementptr inbounds [512 x float], ptr %1, i32 0, i64 %27
  store float %26, ptr %28, align 4
  %29 = add i64 %21, 1
  br label %20

30:                                               ; preds = %20
  %31 = add i64 %13, 1
  br label %12, !llvm.loop !5

32:                                               ; preds = %12
  %33 = add i64 %7, 1
  br label %6, !llvm.loop !5

34:                                               ; preds = %6
  ret void
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 4}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 2048}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.unroll.disable"}
