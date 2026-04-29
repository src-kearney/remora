; ModuleID = '__compute_module_wrapped_broadcast_kernel_module'
source_filename = "__compute_module_wrapped_broadcast_kernel_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin23.5.0"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable
define noalias noundef ptr @wrapped_broadcast(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !5
  tail call void @llvm.experimental.noalias.scope.decl(metadata !6)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !9)
  %7 = load float, ptr %4, align 4, !invariant.load !3, !alias.scope !6, !noalias !9
  br label %.preheader

.preheader:                                       ; preds = %1, %.preheader
  %8 = phi i64 [ 0, %1 ], [ %41, %.preheader ]
  %.idx = shl i64 %8, 7
  %9 = getelementptr i8, ptr %6, i64 %.idx
  store float %7, ptr %9, align 4, !alias.scope !9, !noalias !6
  %10 = getelementptr i8, ptr %9, i64 4
  store float %7, ptr %10, align 4, !alias.scope !9, !noalias !6
  %11 = getelementptr i8, ptr %9, i64 8
  store float %7, ptr %11, align 4, !alias.scope !9, !noalias !6
  %12 = getelementptr i8, ptr %9, i64 12
  store float %7, ptr %12, align 4, !alias.scope !9, !noalias !6
  %13 = getelementptr i8, ptr %9, i64 16
  store float %7, ptr %13, align 4, !alias.scope !9, !noalias !6
  %14 = getelementptr i8, ptr %9, i64 20
  store float %7, ptr %14, align 4, !alias.scope !9, !noalias !6
  %15 = getelementptr i8, ptr %9, i64 24
  store float %7, ptr %15, align 4, !alias.scope !9, !noalias !6
  %16 = getelementptr i8, ptr %9, i64 28
  store float %7, ptr %16, align 4, !alias.scope !9, !noalias !6
  %17 = getelementptr i8, ptr %9, i64 32
  store float %7, ptr %17, align 4, !alias.scope !9, !noalias !6
  %18 = getelementptr i8, ptr %9, i64 36
  store float %7, ptr %18, align 4, !alias.scope !9, !noalias !6
  %19 = getelementptr i8, ptr %9, i64 40
  store float %7, ptr %19, align 4, !alias.scope !9, !noalias !6
  %20 = getelementptr i8, ptr %9, i64 44
  store float %7, ptr %20, align 4, !alias.scope !9, !noalias !6
  %21 = getelementptr i8, ptr %9, i64 48
  store float %7, ptr %21, align 4, !alias.scope !9, !noalias !6
  %22 = getelementptr i8, ptr %9, i64 52
  store float %7, ptr %22, align 4, !alias.scope !9, !noalias !6
  %23 = getelementptr i8, ptr %9, i64 56
  store float %7, ptr %23, align 4, !alias.scope !9, !noalias !6
  %24 = getelementptr i8, ptr %9, i64 60
  store float %7, ptr %24, align 4, !alias.scope !9, !noalias !6
  %25 = getelementptr i8, ptr %9, i64 64
  store float %7, ptr %25, align 4, !alias.scope !9, !noalias !6
  %26 = getelementptr i8, ptr %9, i64 68
  store float %7, ptr %26, align 4, !alias.scope !9, !noalias !6
  %27 = getelementptr i8, ptr %9, i64 72
  store float %7, ptr %27, align 4, !alias.scope !9, !noalias !6
  %28 = getelementptr i8, ptr %9, i64 76
  store float %7, ptr %28, align 4, !alias.scope !9, !noalias !6
  %29 = getelementptr i8, ptr %9, i64 80
  store float %7, ptr %29, align 4, !alias.scope !9, !noalias !6
  %30 = getelementptr i8, ptr %9, i64 84
  store float %7, ptr %30, align 4, !alias.scope !9, !noalias !6
  %31 = getelementptr i8, ptr %9, i64 88
  store float %7, ptr %31, align 4, !alias.scope !9, !noalias !6
  %32 = getelementptr i8, ptr %9, i64 92
  store float %7, ptr %32, align 4, !alias.scope !9, !noalias !6
  %33 = getelementptr i8, ptr %9, i64 96
  store float %7, ptr %33, align 4, !alias.scope !9, !noalias !6
  %34 = getelementptr i8, ptr %9, i64 100
  store float %7, ptr %34, align 4, !alias.scope !9, !noalias !6
  %35 = getelementptr i8, ptr %9, i64 104
  store float %7, ptr %35, align 4, !alias.scope !9, !noalias !6
  %36 = getelementptr i8, ptr %9, i64 108
  store float %7, ptr %36, align 4, !alias.scope !9, !noalias !6
  %37 = getelementptr i8, ptr %9, i64 112
  store float %7, ptr %37, align 4, !alias.scope !9, !noalias !6
  %38 = getelementptr i8, ptr %9, i64 116
  store float %7, ptr %38, align 4, !alias.scope !9, !noalias !6
  %39 = getelementptr i8, ptr %9, i64 120
  store float %7, ptr %39, align 4, !alias.scope !9, !noalias !6
  %40 = getelementptr i8, ptr %9, i64 124
  store float %7, ptr %40, align 4, !alias.scope !9, !noalias !6
  %41 = add nuw nsw i64 %8, 1
  %exitcond.not = icmp eq i64 %41, 8
  br i1 %exitcond.not, label %wrapped_broadcast_wrapped.exit, label %.preheader, !llvm.loop !11

wrapped_broadcast_wrapped.exit:                   ; preds = %.preheader
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 0}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 4}
!5 = !{i64 1024}
!6 = !{!7}
!7 = distinct !{!7, !8, !"wrapped_broadcast_wrapped: argument 0"}
!8 = distinct !{!8, !"wrapped_broadcast_wrapped"}
!9 = !{!10}
!10 = distinct !{!10, !8, !"wrapped_broadcast_wrapped: argument 1"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.unroll.disable"}
