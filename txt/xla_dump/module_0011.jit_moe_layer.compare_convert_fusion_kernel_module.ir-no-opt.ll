; ModuleID = '__compute_module_compare_convert_fusion_kernel_module'
source_filename = "__compute_module_compare_convert_fusion_kernel_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin23.5.0"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

; Function Attrs: uwtable
define ptr @compare_convert_fusion(ptr %0) #0 {
  %2 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !4
  %6 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 1, i32 0
  %7 = load ptr, ptr %6, align 8, !invariant.load !3, !dereferenceable !5
  %8 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %9 = load ptr, ptr %8, align 8
  %10 = getelementptr inbounds %kernel_dim3, ptr %9, i32 0, i32 0
  %11 = load i64, ptr %10, align 4, !invariant.load !3
  %12 = getelementptr inbounds %kernel_dim3, ptr %9, i32 0, i32 1
  %13 = load i64, ptr %12, align 4, !invariant.load !3
  %14 = getelementptr inbounds %kernel_dim3, ptr %9, i32 0, i32 2
  %15 = load i64, ptr %14, align 4, !invariant.load !3
  call void @compare_convert_fusion_wrapped(ptr %5, ptr %7, i64 %11, i64 %13, i64 %15)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @compare_convert_fusion_wrapped(ptr noalias align 64 dereferenceable(32) %0, ptr noalias align 64 dereferenceable(64) %1, i64 %2, i64 %3, i64 %4) #1 {
  br label %6

6:                                                ; preds = %24, %5
  %7 = phi i64 [ %25, %24 ], [ 0, %5 ]
  %8 = icmp slt i64 %7, 8
  br i1 %8, label %9, label %26

9:                                                ; preds = %6
  %10 = getelementptr inbounds [8 x i32], ptr %0, i32 0, i64 %7
  %11 = load i32, ptr %10, align 4, !invariant.load !3
  %12 = mul nsw i64 %7, 2
  br label %13

13:                                               ; preds = %16, %9
  %14 = phi i64 [ %23, %16 ], [ 0, %9 ]
  %15 = icmp slt i64 %14, 2
  br i1 %15, label %16, label %24

16:                                               ; preds = %13
  %17 = trunc i64 %14 to i32
  %18 = icmp eq i32 %11, %17
  %19 = zext i1 %18 to i8
  %20 = sitofp i8 %19 to float
  %21 = add nsw i64 %12, %14
  %22 = getelementptr inbounds [16 x float], ptr %1, i32 0, i64 %21
  store float %20, ptr %22, align 4
  %23 = add i64 %14, 1
  br label %13

24:                                               ; preds = %13
  %25 = add i64 %7, 1
  br label %6, !llvm.loop !6

26:                                               ; preds = %6
  ret void
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 1}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 32}
!5 = !{i64 64}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.unroll.disable"}
