; ModuleID = '__compute_module_multiply_multiply_fusion_kernel_module'
source_filename = "__compute_module_multiply_multiply_fusion_kernel_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin23.5.0"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

; Function Attrs: uwtable
define ptr @multiply_multiply_fusion(ptr %0) #0 {
  %2 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !4
  %6 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 1, i32 0
  %7 = load ptr, ptr %6, align 8, !invariant.load !3, !dereferenceable !4
  %8 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 2, i32 0
  %9 = load ptr, ptr %8, align 8, !invariant.load !3, !dereferenceable !4
  %10 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %11 = load ptr, ptr %10, align 8
  %12 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 0
  %13 = load i64, ptr %12, align 4, !invariant.load !3
  %14 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 1
  %15 = load i64, ptr %14, align 4, !invariant.load !3
  %16 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 2
  %17 = load i64, ptr %16, align 4, !invariant.load !3
  call void @multiply_multiply_fusion_wrapped(ptr %5, ptr %7, ptr %9, i64 %13, i64 %15, i64 %17)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @multiply_multiply_fusion_wrapped(ptr noalias align 64 dereferenceable(4096) %0, ptr noalias align 64 dereferenceable(4096) %1, ptr noalias align 64 dereferenceable(4096) %2, i64 %3, i64 %4, i64 %5) #1 {
  br label %7

7:                                                ; preds = %36, %6
  %8 = phi i64 [ %37, %36 ], [ 0, %6 ]
  %9 = icmp slt i64 %8, 2
  br i1 %9, label %10, label %38

10:                                               ; preds = %7
  %11 = mul nsw i64 %8, 512
  br label %12

12:                                               ; preds = %34, %10
  %13 = phi i64 [ %35, %34 ], [ 0, %10 ]
  %14 = icmp slt i64 %13, 8
  br i1 %14, label %15, label %36

15:                                               ; preds = %12
  %16 = mul nsw i64 %13, 64
  %17 = add nsw i64 %11, %16
  br label %18

18:                                               ; preds = %21, %15
  %19 = phi i64 [ %33, %21 ], [ 0, %15 ]
  %20 = icmp slt i64 %19, 64
  br i1 %20, label %21, label %34

21:                                               ; preds = %18
  %22 = add nsw i64 %17, %19
  %23 = getelementptr inbounds [1024 x float], ptr %1, i32 0, i64 %22
  %24 = load float, ptr %23, align 4, !invariant.load !3
  %25 = fneg float %24
  %26 = call float @llvm.exp.f32(float %25)
  %27 = fadd float %26, 1.000000e+00
  %28 = fdiv float 1.000000e+00, %27
  %29 = fmul float %24, %28
  %30 = getelementptr inbounds [1024 x float], ptr %0, i32 0, i64 %22
  %31 = load float, ptr %30, align 4
  %32 = fmul float %29, %31
  store float %32, ptr %30, align 4
  %33 = add i64 %19, 1
  br label %18

34:                                               ; preds = %18
  %35 = add i64 %13, 1
  br label %12, !llvm.loop !5

36:                                               ; preds = %12
  %37 = add i64 %8, 1
  br label %7, !llvm.loop !5

38:                                               ; preds = %7
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #2

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 3}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 4096}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.unroll.disable"}
