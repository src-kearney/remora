; ModuleID = '__compute_module_iota_reduce_fusion_kernel_module'
source_filename = "__compute_module_iota_reduce_fusion_kernel_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin23.5.0"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

; Function Attrs: uwtable
define ptr @iota_reduce_fusion(ptr %0) #0 {
  %2 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !4
  %6 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 1, i32 0
  %7 = load ptr, ptr %6, align 8, !invariant.load !3, !dereferenceable !5
  %8 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 2, i32 0
  %9 = load ptr, ptr %8, align 8, !invariant.load !3, !dereferenceable !5
  %10 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %11 = load ptr, ptr %10, align 8
  %12 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 0
  %13 = load i64, ptr %12, align 4, !invariant.load !3
  %14 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 1
  %15 = load i64, ptr %14, align 4, !invariant.load !3
  %16 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 2
  %17 = load i64, ptr %16, align 4, !invariant.load !3
  call void @iota_reduce_fusion_wrapped(ptr %5, ptr %7, ptr %9, i64 %13, i64 %15, i64 %17)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @iota_reduce_fusion_wrapped(ptr noalias align 64 dereferenceable(64) %0, ptr noalias align 64 dereferenceable(32) %1, ptr noalias align 64 dereferenceable(32) %2, i64 %3, i64 %4, i64 %5) #1 {
  br label %7

7:                                                ; preds = %32, %6
  %8 = phi i64 [ %35, %32 ], [ 0, %6 ]
  %9 = icmp slt i64 %8, 8
  br i1 %9, label %10, label %36

10:                                               ; preds = %7
  %11 = mul nsw i64 %8, 2
  br label %12

12:                                               ; preds = %17, %10
  %13 = phi i64 [ %31, %17 ], [ 0, %10 ]
  %14 = phi float [ %25, %17 ], [ 0xFFF0000000000000, %10 ]
  %15 = phi i32 [ %30, %17 ], [ 0, %10 ]
  %16 = icmp slt i64 %13, 2
  br i1 %16, label %17, label %32

17:                                               ; preds = %12
  %18 = add nsw i64 %11, %13
  %19 = getelementptr inbounds [16 x float], ptr %0, i32 0, i64 %18
  %20 = load float, ptr %19, align 4, !invariant.load !3
  %21 = trunc i64 %13 to i32
  %22 = fcmp ogt float %14, %20
  %23 = fcmp une float %14, %14
  %24 = or i1 %22, %23
  %25 = select i1 %24, float %14, float %20
  %26 = fcmp oeq float %14, %20
  %27 = icmp slt i32 %15, %21
  %28 = and i1 %26, %27
  %29 = or i1 %24, %28
  %30 = select i1 %29, i32 %15, i32 %21
  %31 = add i64 %13, 1
  br label %12

32:                                               ; preds = %12
  %33 = getelementptr inbounds [8 x float], ptr %1, i32 0, i64 %8
  store float %14, ptr %33, align 4
  %34 = getelementptr inbounds [8 x i32], ptr %2, i32 0, i64 %8
  store i32 %15, ptr %34, align 4
  %35 = add i64 %8, 1
  br label %7, !llvm.loop !6

36:                                               ; preds = %7
  ret void
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 2}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 64}
!5 = !{i64 32}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.unroll.disable"}
