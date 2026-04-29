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
  br label %.preheader2

.preheader2:                                      ; preds = %1, %76
  %exitcond3.not = phi i1 [ false, %1 ], [ true, %76 ]
  %8 = phi i64 [ 0, %1 ], [ 8192, %76 ]
  %9 = getelementptr i8, ptr %6, i64 %8
  br label %.preheader

.preheader:                                       ; preds = %.preheader2, %.preheader
  %10 = phi i64 [ 0, %.preheader2 ], [ %75, %.preheader ]
  %.idx1 = shl i64 %10, 8
  %11 = getelementptr i8, ptr %9, i64 %.idx1
  store float %7, ptr %11, align 4, !alias.scope !9, !noalias !6
  %12 = getelementptr i8, ptr %11, i64 4
  store float %7, ptr %12, align 4, !alias.scope !9, !noalias !6
  %13 = getelementptr i8, ptr %11, i64 8
  store float %7, ptr %13, align 4, !alias.scope !9, !noalias !6
  %14 = getelementptr i8, ptr %11, i64 12
  store float %7, ptr %14, align 4, !alias.scope !9, !noalias !6
  %15 = getelementptr i8, ptr %11, i64 16
  store float %7, ptr %15, align 4, !alias.scope !9, !noalias !6
  %16 = getelementptr i8, ptr %11, i64 20
  store float %7, ptr %16, align 4, !alias.scope !9, !noalias !6
  %17 = getelementptr i8, ptr %11, i64 24
  store float %7, ptr %17, align 4, !alias.scope !9, !noalias !6
  %18 = getelementptr i8, ptr %11, i64 28
  store float %7, ptr %18, align 4, !alias.scope !9, !noalias !6
  %19 = getelementptr i8, ptr %11, i64 32
  store float %7, ptr %19, align 4, !alias.scope !9, !noalias !6
  %20 = getelementptr i8, ptr %11, i64 36
  store float %7, ptr %20, align 4, !alias.scope !9, !noalias !6
  %21 = getelementptr i8, ptr %11, i64 40
  store float %7, ptr %21, align 4, !alias.scope !9, !noalias !6
  %22 = getelementptr i8, ptr %11, i64 44
  store float %7, ptr %22, align 4, !alias.scope !9, !noalias !6
  %23 = getelementptr i8, ptr %11, i64 48
  store float %7, ptr %23, align 4, !alias.scope !9, !noalias !6
  %24 = getelementptr i8, ptr %11, i64 52
  store float %7, ptr %24, align 4, !alias.scope !9, !noalias !6
  %25 = getelementptr i8, ptr %11, i64 56
  store float %7, ptr %25, align 4, !alias.scope !9, !noalias !6
  %26 = getelementptr i8, ptr %11, i64 60
  store float %7, ptr %26, align 4, !alias.scope !9, !noalias !6
  %27 = getelementptr i8, ptr %11, i64 64
  store float %7, ptr %27, align 4, !alias.scope !9, !noalias !6
  %28 = getelementptr i8, ptr %11, i64 68
  store float %7, ptr %28, align 4, !alias.scope !9, !noalias !6
  %29 = getelementptr i8, ptr %11, i64 72
  store float %7, ptr %29, align 4, !alias.scope !9, !noalias !6
  %30 = getelementptr i8, ptr %11, i64 76
  store float %7, ptr %30, align 4, !alias.scope !9, !noalias !6
  %31 = getelementptr i8, ptr %11, i64 80
  store float %7, ptr %31, align 4, !alias.scope !9, !noalias !6
  %32 = getelementptr i8, ptr %11, i64 84
  store float %7, ptr %32, align 4, !alias.scope !9, !noalias !6
  %33 = getelementptr i8, ptr %11, i64 88
  store float %7, ptr %33, align 4, !alias.scope !9, !noalias !6
  %34 = getelementptr i8, ptr %11, i64 92
  store float %7, ptr %34, align 4, !alias.scope !9, !noalias !6
  %35 = getelementptr i8, ptr %11, i64 96
  store float %7, ptr %35, align 4, !alias.scope !9, !noalias !6
  %36 = getelementptr i8, ptr %11, i64 100
  store float %7, ptr %36, align 4, !alias.scope !9, !noalias !6
  %37 = getelementptr i8, ptr %11, i64 104
  store float %7, ptr %37, align 4, !alias.scope !9, !noalias !6
  %38 = getelementptr i8, ptr %11, i64 108
  store float %7, ptr %38, align 4, !alias.scope !9, !noalias !6
  %39 = getelementptr i8, ptr %11, i64 112
  store float %7, ptr %39, align 4, !alias.scope !9, !noalias !6
  %40 = getelementptr i8, ptr %11, i64 116
  store float %7, ptr %40, align 4, !alias.scope !9, !noalias !6
  %41 = getelementptr i8, ptr %11, i64 120
  store float %7, ptr %41, align 4, !alias.scope !9, !noalias !6
  %42 = getelementptr i8, ptr %11, i64 124
  store float %7, ptr %42, align 4, !alias.scope !9, !noalias !6
  %43 = getelementptr i8, ptr %11, i64 128
  store float %7, ptr %43, align 4, !alias.scope !9, !noalias !6
  %44 = getelementptr i8, ptr %11, i64 132
  store float %7, ptr %44, align 4, !alias.scope !9, !noalias !6
  %45 = getelementptr i8, ptr %11, i64 136
  store float %7, ptr %45, align 4, !alias.scope !9, !noalias !6
  %46 = getelementptr i8, ptr %11, i64 140
  store float %7, ptr %46, align 4, !alias.scope !9, !noalias !6
  %47 = getelementptr i8, ptr %11, i64 144
  store float %7, ptr %47, align 4, !alias.scope !9, !noalias !6
  %48 = getelementptr i8, ptr %11, i64 148
  store float %7, ptr %48, align 4, !alias.scope !9, !noalias !6
  %49 = getelementptr i8, ptr %11, i64 152
  store float %7, ptr %49, align 4, !alias.scope !9, !noalias !6
  %50 = getelementptr i8, ptr %11, i64 156
  store float %7, ptr %50, align 4, !alias.scope !9, !noalias !6
  %51 = getelementptr i8, ptr %11, i64 160
  store float %7, ptr %51, align 4, !alias.scope !9, !noalias !6
  %52 = getelementptr i8, ptr %11, i64 164
  store float %7, ptr %52, align 4, !alias.scope !9, !noalias !6
  %53 = getelementptr i8, ptr %11, i64 168
  store float %7, ptr %53, align 4, !alias.scope !9, !noalias !6
  %54 = getelementptr i8, ptr %11, i64 172
  store float %7, ptr %54, align 4, !alias.scope !9, !noalias !6
  %55 = getelementptr i8, ptr %11, i64 176
  store float %7, ptr %55, align 4, !alias.scope !9, !noalias !6
  %56 = getelementptr i8, ptr %11, i64 180
  store float %7, ptr %56, align 4, !alias.scope !9, !noalias !6
  %57 = getelementptr i8, ptr %11, i64 184
  store float %7, ptr %57, align 4, !alias.scope !9, !noalias !6
  %58 = getelementptr i8, ptr %11, i64 188
  store float %7, ptr %58, align 4, !alias.scope !9, !noalias !6
  %59 = getelementptr i8, ptr %11, i64 192
  store float %7, ptr %59, align 4, !alias.scope !9, !noalias !6
  %60 = getelementptr i8, ptr %11, i64 196
  store float %7, ptr %60, align 4, !alias.scope !9, !noalias !6
  %61 = getelementptr i8, ptr %11, i64 200
  store float %7, ptr %61, align 4, !alias.scope !9, !noalias !6
  %62 = getelementptr i8, ptr %11, i64 204
  store float %7, ptr %62, align 4, !alias.scope !9, !noalias !6
  %63 = getelementptr i8, ptr %11, i64 208
  store float %7, ptr %63, align 4, !alias.scope !9, !noalias !6
  %64 = getelementptr i8, ptr %11, i64 212
  store float %7, ptr %64, align 4, !alias.scope !9, !noalias !6
  %65 = getelementptr i8, ptr %11, i64 216
  store float %7, ptr %65, align 4, !alias.scope !9, !noalias !6
  %66 = getelementptr i8, ptr %11, i64 220
  store float %7, ptr %66, align 4, !alias.scope !9, !noalias !6
  %67 = getelementptr i8, ptr %11, i64 224
  store float %7, ptr %67, align 4, !alias.scope !9, !noalias !6
  %68 = getelementptr i8, ptr %11, i64 228
  store float %7, ptr %68, align 4, !alias.scope !9, !noalias !6
  %69 = getelementptr i8, ptr %11, i64 232
  store float %7, ptr %69, align 4, !alias.scope !9, !noalias !6
  %70 = getelementptr i8, ptr %11, i64 236
  store float %7, ptr %70, align 4, !alias.scope !9, !noalias !6
  %71 = getelementptr i8, ptr %11, i64 240
  store float %7, ptr %71, align 4, !alias.scope !9, !noalias !6
  %72 = getelementptr i8, ptr %11, i64 244
  store float %7, ptr %72, align 4, !alias.scope !9, !noalias !6
  %73 = getelementptr i8, ptr %11, i64 248
  store float %7, ptr %73, align 4, !alias.scope !9, !noalias !6
  %74 = getelementptr i8, ptr %11, i64 252
  store float %7, ptr %74, align 4, !alias.scope !9, !noalias !6
  %75 = add nuw nsw i64 %10, 1
  %exitcond.not = icmp eq i64 %75, 32
  br i1 %exitcond.not, label %76, label %.preheader, !llvm.loop !11

76:                                               ; preds = %.preheader
  br i1 %exitcond3.not, label %wrapped_broadcast_wrapped.exit, label %.preheader2, !llvm.loop !11

wrapped_broadcast_wrapped.exit:                   ; preds = %76
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
!5 = !{i64 16384}
!6 = !{!7}
!7 = distinct !{!7, !8, !"wrapped_broadcast_wrapped: argument 0"}
!8 = distinct !{!8, !"wrapped_broadcast_wrapped"}
!9 = !{!10}
!10 = distinct !{!10, !8, !"wrapped_broadcast_wrapped: argument 1"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.unroll.disable"}
