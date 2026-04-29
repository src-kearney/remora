; ModuleID = '__compute_module_transpose_copy_fusion.1_kernel_module'
source_filename = "__compute_module_transpose_copy_fusion.1_kernel_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin23.5.0"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable
define noalias noundef ptr @transpose_copy_fusion.1(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !5
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %8 = load ptr, ptr %7, align 8, !invariant.load !3, !dereferenceable !6
  tail call void @llvm.experimental.noalias.scope.decl(metadata !7)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !10)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !12)
  br label %9

9:                                                ; preds = %1, %150
  %10 = phi i1 [ true, %1 ], [ false, %150 ]
  %11 = phi i64 [ 0, %1 ], [ 1, %150 ]
  %12 = trunc nuw nsw i64 %11 to i32
  %.idx = shl nuw nsw i64 %11, 10
  %13 = getelementptr i8, ptr %8, i64 %.idx
  br label %14

14:                                               ; preds = %9, %14
  %15 = phi i64 [ 0, %9 ], [ %149, %14 ]
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %15
  %17 = load i32, ptr %16, align 4, !invariant.load !3, !alias.scope !10, !noalias !14
  %18 = icmp eq i32 %17, %12
  %19 = uitofp i1 %18 to float
  %20 = shl nuw nsw i64 %15, 5
  %21 = getelementptr float, ptr %4, i64 %20
  %22 = getelementptr float, ptr %13, i64 %20
  %23 = load float, ptr %21, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %24 = fmul float %23, %19
  store float %24, ptr %22, align 4, !alias.scope !12, !noalias !16
  %25 = getelementptr i8, ptr %21, i64 4
  %26 = load float, ptr %25, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %27 = fmul float %26, %19
  %28 = getelementptr i8, ptr %22, i64 4
  store float %27, ptr %28, align 4, !alias.scope !12, !noalias !16
  %29 = getelementptr i8, ptr %21, i64 8
  %30 = load float, ptr %29, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %31 = fmul float %30, %19
  %32 = getelementptr i8, ptr %22, i64 8
  store float %31, ptr %32, align 4, !alias.scope !12, !noalias !16
  %33 = getelementptr i8, ptr %21, i64 12
  %34 = load float, ptr %33, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %35 = fmul float %34, %19
  %36 = getelementptr i8, ptr %22, i64 12
  store float %35, ptr %36, align 4, !alias.scope !12, !noalias !16
  %37 = getelementptr i8, ptr %21, i64 16
  %38 = load float, ptr %37, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %39 = fmul float %38, %19
  %40 = getelementptr i8, ptr %22, i64 16
  store float %39, ptr %40, align 4, !alias.scope !12, !noalias !16
  %41 = getelementptr i8, ptr %21, i64 20
  %42 = load float, ptr %41, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %43 = fmul float %42, %19
  %44 = getelementptr i8, ptr %22, i64 20
  store float %43, ptr %44, align 4, !alias.scope !12, !noalias !16
  %45 = getelementptr i8, ptr %21, i64 24
  %46 = load float, ptr %45, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %47 = fmul float %46, %19
  %48 = getelementptr i8, ptr %22, i64 24
  store float %47, ptr %48, align 4, !alias.scope !12, !noalias !16
  %49 = getelementptr i8, ptr %21, i64 28
  %50 = load float, ptr %49, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %51 = fmul float %50, %19
  %52 = getelementptr i8, ptr %22, i64 28
  store float %51, ptr %52, align 4, !alias.scope !12, !noalias !16
  %53 = getelementptr i8, ptr %21, i64 32
  %54 = load float, ptr %53, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %55 = fmul float %54, %19
  %56 = getelementptr i8, ptr %22, i64 32
  store float %55, ptr %56, align 4, !alias.scope !12, !noalias !16
  %57 = getelementptr i8, ptr %21, i64 36
  %58 = load float, ptr %57, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %59 = fmul float %58, %19
  %60 = getelementptr i8, ptr %22, i64 36
  store float %59, ptr %60, align 4, !alias.scope !12, !noalias !16
  %61 = getelementptr i8, ptr %21, i64 40
  %62 = load float, ptr %61, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %63 = fmul float %62, %19
  %64 = getelementptr i8, ptr %22, i64 40
  store float %63, ptr %64, align 4, !alias.scope !12, !noalias !16
  %65 = getelementptr i8, ptr %21, i64 44
  %66 = load float, ptr %65, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %67 = fmul float %66, %19
  %68 = getelementptr i8, ptr %22, i64 44
  store float %67, ptr %68, align 4, !alias.scope !12, !noalias !16
  %69 = getelementptr i8, ptr %21, i64 48
  %70 = load float, ptr %69, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %71 = fmul float %70, %19
  %72 = getelementptr i8, ptr %22, i64 48
  store float %71, ptr %72, align 4, !alias.scope !12, !noalias !16
  %73 = getelementptr i8, ptr %21, i64 52
  %74 = load float, ptr %73, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %75 = fmul float %74, %19
  %76 = getelementptr i8, ptr %22, i64 52
  store float %75, ptr %76, align 4, !alias.scope !12, !noalias !16
  %77 = getelementptr i8, ptr %21, i64 56
  %78 = load float, ptr %77, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %79 = fmul float %78, %19
  %80 = getelementptr i8, ptr %22, i64 56
  store float %79, ptr %80, align 4, !alias.scope !12, !noalias !16
  %81 = getelementptr i8, ptr %21, i64 60
  %82 = load float, ptr %81, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %83 = fmul float %82, %19
  %84 = getelementptr i8, ptr %22, i64 60
  store float %83, ptr %84, align 4, !alias.scope !12, !noalias !16
  %85 = getelementptr i8, ptr %21, i64 64
  %86 = load float, ptr %85, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %87 = fmul float %86, %19
  %88 = getelementptr i8, ptr %22, i64 64
  store float %87, ptr %88, align 4, !alias.scope !12, !noalias !16
  %89 = getelementptr i8, ptr %21, i64 68
  %90 = load float, ptr %89, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %91 = fmul float %90, %19
  %92 = getelementptr i8, ptr %22, i64 68
  store float %91, ptr %92, align 4, !alias.scope !12, !noalias !16
  %93 = getelementptr i8, ptr %21, i64 72
  %94 = load float, ptr %93, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %95 = fmul float %94, %19
  %96 = getelementptr i8, ptr %22, i64 72
  store float %95, ptr %96, align 4, !alias.scope !12, !noalias !16
  %97 = getelementptr i8, ptr %21, i64 76
  %98 = load float, ptr %97, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %99 = fmul float %98, %19
  %100 = getelementptr i8, ptr %22, i64 76
  store float %99, ptr %100, align 4, !alias.scope !12, !noalias !16
  %101 = getelementptr i8, ptr %21, i64 80
  %102 = load float, ptr %101, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %103 = fmul float %102, %19
  %104 = getelementptr i8, ptr %22, i64 80
  store float %103, ptr %104, align 4, !alias.scope !12, !noalias !16
  %105 = getelementptr i8, ptr %21, i64 84
  %106 = load float, ptr %105, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %107 = fmul float %106, %19
  %108 = getelementptr i8, ptr %22, i64 84
  store float %107, ptr %108, align 4, !alias.scope !12, !noalias !16
  %109 = getelementptr i8, ptr %21, i64 88
  %110 = load float, ptr %109, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %111 = fmul float %110, %19
  %112 = getelementptr i8, ptr %22, i64 88
  store float %111, ptr %112, align 4, !alias.scope !12, !noalias !16
  %113 = getelementptr i8, ptr %21, i64 92
  %114 = load float, ptr %113, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %115 = fmul float %114, %19
  %116 = getelementptr i8, ptr %22, i64 92
  store float %115, ptr %116, align 4, !alias.scope !12, !noalias !16
  %117 = getelementptr i8, ptr %21, i64 96
  %118 = load float, ptr %117, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %119 = fmul float %118, %19
  %120 = getelementptr i8, ptr %22, i64 96
  store float %119, ptr %120, align 4, !alias.scope !12, !noalias !16
  %121 = getelementptr i8, ptr %21, i64 100
  %122 = load float, ptr %121, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %123 = fmul float %122, %19
  %124 = getelementptr i8, ptr %22, i64 100
  store float %123, ptr %124, align 4, !alias.scope !12, !noalias !16
  %125 = getelementptr i8, ptr %21, i64 104
  %126 = load float, ptr %125, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %127 = fmul float %126, %19
  %128 = getelementptr i8, ptr %22, i64 104
  store float %127, ptr %128, align 4, !alias.scope !12, !noalias !16
  %129 = getelementptr i8, ptr %21, i64 108
  %130 = load float, ptr %129, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %131 = fmul float %130, %19
  %132 = getelementptr i8, ptr %22, i64 108
  store float %131, ptr %132, align 4, !alias.scope !12, !noalias !16
  %133 = getelementptr i8, ptr %21, i64 112
  %134 = load float, ptr %133, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %135 = fmul float %134, %19
  %136 = getelementptr i8, ptr %22, i64 112
  store float %135, ptr %136, align 4, !alias.scope !12, !noalias !16
  %137 = getelementptr i8, ptr %21, i64 116
  %138 = load float, ptr %137, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %139 = fmul float %138, %19
  %140 = getelementptr i8, ptr %22, i64 116
  store float %139, ptr %140, align 4, !alias.scope !12, !noalias !16
  %141 = getelementptr i8, ptr %21, i64 120
  %142 = load float, ptr %141, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %143 = fmul float %142, %19
  %144 = getelementptr i8, ptr %22, i64 120
  store float %143, ptr %144, align 4, !alias.scope !12, !noalias !16
  %145 = getelementptr i8, ptr %21, i64 124
  %146 = load float, ptr %145, align 4, !invariant.load !3, !alias.scope !7, !noalias !15
  %147 = fmul float %146, %19
  %148 = getelementptr i8, ptr %22, i64 124
  store float %147, ptr %148, align 4, !alias.scope !12, !noalias !16
  %149 = add nuw nsw i64 %15, 1
  %exitcond.not = icmp eq i64 %149, 8
  br i1 %exitcond.not, label %150, label %14, !llvm.loop !17

150:                                              ; preds = %14
  br i1 %10, label %9, label %transpose_copy_fusion.1_wrapped.exit, !llvm.loop !17

transpose_copy_fusion.1_wrapped.exit:             ; preds = %150
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 5}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 1024}
!5 = !{i64 32}
!6 = !{i64 2048}
!7 = !{!8}
!8 = distinct !{!8, !9, !"transpose_copy_fusion.1_wrapped: argument 0"}
!9 = distinct !{!9, !"transpose_copy_fusion.1_wrapped"}
!10 = !{!11}
!11 = distinct !{!11, !9, !"transpose_copy_fusion.1_wrapped: argument 1"}
!12 = !{!13}
!13 = distinct !{!13, !9, !"transpose_copy_fusion.1_wrapped: argument 2"}
!14 = !{!8, !13}
!15 = !{!11, !13}
!16 = !{!8, !11}
!17 = distinct !{!17, !18}
!18 = !{!"llvm.loop.unroll.disable"}
