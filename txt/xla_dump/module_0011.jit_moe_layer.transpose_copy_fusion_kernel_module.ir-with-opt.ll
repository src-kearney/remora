; ModuleID = '__compute_module_transpose_copy_fusion_kernel_module'
source_filename = "__compute_module_transpose_copy_fusion_kernel_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin23.5.0"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable
define noalias noundef ptr @transpose_copy_fusion(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !4
  tail call void @llvm.experimental.noalias.scope.decl(metadata !5)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !8)
  br label %.preheader4

.preheader4:                                      ; preds = %1, %108
  %7 = phi i64 [ 0, %1 ], [ %109, %108 ]
  %.idx = shl i64 %7, 7
  %8 = getelementptr i8, ptr %4, i64 %.idx
  %.idx2 = shl i64 %7, 8
  %9 = getelementptr i8, ptr %6, i64 %.idx2
  br label %.preheader

.preheader:                                       ; preds = %.preheader4, %.preheader
  %10 = phi i64 [ 0, %.preheader4 ], [ %107, %.preheader ]
  %.idx1 = shl i64 %10, 10
  %11 = getelementptr i8, ptr %8, i64 %.idx1
  %.idx3 = shl i64 %10, 7
  %12 = getelementptr i8, ptr %9, i64 %.idx3
  %13 = load float, ptr %11, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  store float %13, ptr %12, align 4, !alias.scope !8, !noalias !5
  %14 = getelementptr i8, ptr %11, i64 4
  %15 = load float, ptr %14, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %16 = getelementptr i8, ptr %12, i64 4
  store float %15, ptr %16, align 4, !alias.scope !8, !noalias !5
  %17 = getelementptr i8, ptr %11, i64 8
  %18 = load float, ptr %17, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %19 = getelementptr i8, ptr %12, i64 8
  store float %18, ptr %19, align 4, !alias.scope !8, !noalias !5
  %20 = getelementptr i8, ptr %11, i64 12
  %21 = load float, ptr %20, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %22 = getelementptr i8, ptr %12, i64 12
  store float %21, ptr %22, align 4, !alias.scope !8, !noalias !5
  %23 = getelementptr i8, ptr %11, i64 16
  %24 = load float, ptr %23, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %25 = getelementptr i8, ptr %12, i64 16
  store float %24, ptr %25, align 4, !alias.scope !8, !noalias !5
  %26 = getelementptr i8, ptr %11, i64 20
  %27 = load float, ptr %26, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %28 = getelementptr i8, ptr %12, i64 20
  store float %27, ptr %28, align 4, !alias.scope !8, !noalias !5
  %29 = getelementptr i8, ptr %11, i64 24
  %30 = load float, ptr %29, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %31 = getelementptr i8, ptr %12, i64 24
  store float %30, ptr %31, align 4, !alias.scope !8, !noalias !5
  %32 = getelementptr i8, ptr %11, i64 28
  %33 = load float, ptr %32, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %34 = getelementptr i8, ptr %12, i64 28
  store float %33, ptr %34, align 4, !alias.scope !8, !noalias !5
  %35 = getelementptr i8, ptr %11, i64 32
  %36 = load float, ptr %35, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %37 = getelementptr i8, ptr %12, i64 32
  store float %36, ptr %37, align 4, !alias.scope !8, !noalias !5
  %38 = getelementptr i8, ptr %11, i64 36
  %39 = load float, ptr %38, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %40 = getelementptr i8, ptr %12, i64 36
  store float %39, ptr %40, align 4, !alias.scope !8, !noalias !5
  %41 = getelementptr i8, ptr %11, i64 40
  %42 = load float, ptr %41, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %43 = getelementptr i8, ptr %12, i64 40
  store float %42, ptr %43, align 4, !alias.scope !8, !noalias !5
  %44 = getelementptr i8, ptr %11, i64 44
  %45 = load float, ptr %44, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %46 = getelementptr i8, ptr %12, i64 44
  store float %45, ptr %46, align 4, !alias.scope !8, !noalias !5
  %47 = getelementptr i8, ptr %11, i64 48
  %48 = load float, ptr %47, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %49 = getelementptr i8, ptr %12, i64 48
  store float %48, ptr %49, align 4, !alias.scope !8, !noalias !5
  %50 = getelementptr i8, ptr %11, i64 52
  %51 = load float, ptr %50, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %52 = getelementptr i8, ptr %12, i64 52
  store float %51, ptr %52, align 4, !alias.scope !8, !noalias !5
  %53 = getelementptr i8, ptr %11, i64 56
  %54 = load float, ptr %53, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %55 = getelementptr i8, ptr %12, i64 56
  store float %54, ptr %55, align 4, !alias.scope !8, !noalias !5
  %56 = getelementptr i8, ptr %11, i64 60
  %57 = load float, ptr %56, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %58 = getelementptr i8, ptr %12, i64 60
  store float %57, ptr %58, align 4, !alias.scope !8, !noalias !5
  %59 = getelementptr i8, ptr %11, i64 64
  %60 = load float, ptr %59, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %61 = getelementptr i8, ptr %12, i64 64
  store float %60, ptr %61, align 4, !alias.scope !8, !noalias !5
  %62 = getelementptr i8, ptr %11, i64 68
  %63 = load float, ptr %62, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %64 = getelementptr i8, ptr %12, i64 68
  store float %63, ptr %64, align 4, !alias.scope !8, !noalias !5
  %65 = getelementptr i8, ptr %11, i64 72
  %66 = load float, ptr %65, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %67 = getelementptr i8, ptr %12, i64 72
  store float %66, ptr %67, align 4, !alias.scope !8, !noalias !5
  %68 = getelementptr i8, ptr %11, i64 76
  %69 = load float, ptr %68, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %70 = getelementptr i8, ptr %12, i64 76
  store float %69, ptr %70, align 4, !alias.scope !8, !noalias !5
  %71 = getelementptr i8, ptr %11, i64 80
  %72 = load float, ptr %71, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %73 = getelementptr i8, ptr %12, i64 80
  store float %72, ptr %73, align 4, !alias.scope !8, !noalias !5
  %74 = getelementptr i8, ptr %11, i64 84
  %75 = load float, ptr %74, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %76 = getelementptr i8, ptr %12, i64 84
  store float %75, ptr %76, align 4, !alias.scope !8, !noalias !5
  %77 = getelementptr i8, ptr %11, i64 88
  %78 = load float, ptr %77, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %79 = getelementptr i8, ptr %12, i64 88
  store float %78, ptr %79, align 4, !alias.scope !8, !noalias !5
  %80 = getelementptr i8, ptr %11, i64 92
  %81 = load float, ptr %80, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %82 = getelementptr i8, ptr %12, i64 92
  store float %81, ptr %82, align 4, !alias.scope !8, !noalias !5
  %83 = getelementptr i8, ptr %11, i64 96
  %84 = load float, ptr %83, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %85 = getelementptr i8, ptr %12, i64 96
  store float %84, ptr %85, align 4, !alias.scope !8, !noalias !5
  %86 = getelementptr i8, ptr %11, i64 100
  %87 = load float, ptr %86, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %88 = getelementptr i8, ptr %12, i64 100
  store float %87, ptr %88, align 4, !alias.scope !8, !noalias !5
  %89 = getelementptr i8, ptr %11, i64 104
  %90 = load float, ptr %89, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %91 = getelementptr i8, ptr %12, i64 104
  store float %90, ptr %91, align 4, !alias.scope !8, !noalias !5
  %92 = getelementptr i8, ptr %11, i64 108
  %93 = load float, ptr %92, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %94 = getelementptr i8, ptr %12, i64 108
  store float %93, ptr %94, align 4, !alias.scope !8, !noalias !5
  %95 = getelementptr i8, ptr %11, i64 112
  %96 = load float, ptr %95, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %97 = getelementptr i8, ptr %12, i64 112
  store float %96, ptr %97, align 4, !alias.scope !8, !noalias !5
  %98 = getelementptr i8, ptr %11, i64 116
  %99 = load float, ptr %98, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %100 = getelementptr i8, ptr %12, i64 116
  store float %99, ptr %100, align 4, !alias.scope !8, !noalias !5
  %101 = getelementptr i8, ptr %11, i64 120
  %102 = load float, ptr %101, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %103 = getelementptr i8, ptr %12, i64 120
  store float %102, ptr %103, align 4, !alias.scope !8, !noalias !5
  %104 = getelementptr i8, ptr %11, i64 124
  %105 = load float, ptr %104, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %106 = getelementptr i8, ptr %12, i64 124
  store float %105, ptr %106, align 4, !alias.scope !8, !noalias !5
  %107 = add nuw nsw i64 %10, 1
  %exitcond.not = icmp eq i64 %107, 2
  br i1 %exitcond.not, label %108, label %.preheader, !llvm.loop !10

108:                                              ; preds = %.preheader
  %109 = add nuw nsw i64 %7, 1
  %exitcond6.not = icmp eq i64 %109, 8
  br i1 %exitcond6.not, label %transpose_copy_fusion_wrapped.exit, label %.preheader4, !llvm.loop !10

transpose_copy_fusion_wrapped.exit:               ; preds = %108
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 4}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 2048}
!5 = !{!6}
!6 = distinct !{!6, !7, !"transpose_copy_fusion_wrapped: argument 0"}
!7 = distinct !{!7, !"transpose_copy_fusion_wrapped"}
!8 = !{!9}
!9 = distinct !{!9, !7, !"transpose_copy_fusion_wrapped: argument 1"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.unroll.disable"}
