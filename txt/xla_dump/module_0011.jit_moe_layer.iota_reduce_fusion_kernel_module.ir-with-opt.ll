; ModuleID = '__compute_module_iota_reduce_fusion_kernel_module'
source_filename = "__compute_module_iota_reduce_fusion_kernel_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin23.5.0"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable
define noalias noundef ptr @iota_reduce_fusion(ptr readonly captures(none) %0) local_unnamed_addr #0 {
vector.ph:
  %1 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %2 = load ptr, ptr %1, align 8, !invariant.load !3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3, !dereferenceable !4
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !5
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %7 = load ptr, ptr %6, align 8, !invariant.load !3, !dereferenceable !5
  tail call void @llvm.experimental.noalias.scope.decl(metadata !6)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !9)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !11)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %8 = shl i64 %index, 3
  %9 = getelementptr i8, ptr %3, i64 %8
  %wide.vec = load <8 x float>, ptr %9, align 4, !invariant.load !3, !alias.scope !6, !noalias !13
  %strided.vec = shufflevector <8 x float> %wide.vec, <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %strided.vec8 = shufflevector <8 x float> %wide.vec, <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %10 = fcmp ule <4 x float> %strided.vec, %strided.vec8
  %11 = fcmp ord <4 x float> %strided.vec, zeroinitializer
  %12 = and <4 x i1> %11, %10
  %13 = select <4 x i1> %12, <4 x float> %strided.vec8, <4 x float> %strided.vec
  %14 = fcmp une <4 x float> %strided.vec, %strided.vec8
  %15 = and <4 x i1> %12, %14
  %16 = zext <4 x i1> %15 to <4 x i32>
  %17 = getelementptr inbounds nuw float, ptr %5, i64 %index
  store <4 x float> %13, ptr %17, align 4, !alias.scope !9, !noalias !14
  %18 = getelementptr inbounds nuw i32, ptr %7, i64 %index
  store <4 x i32> %16, ptr %18, align 4, !alias.scope !11, !noalias !15
  %index.next = add nuw i64 %index, 4
  %19 = icmp eq i64 %index.next, 8
  br i1 %19, label %iota_reduce_fusion_wrapped.exit, label %vector.body, !llvm.loop !16

iota_reduce_fusion_wrapped.exit:                  ; preds = %vector.body
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 2}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 64}
!5 = !{i64 32}
!6 = !{!7}
!7 = distinct !{!7, !8, !"iota_reduce_fusion_wrapped: argument 0"}
!8 = distinct !{!8, !"iota_reduce_fusion_wrapped"}
!9 = !{!10}
!10 = distinct !{!10, !8, !"iota_reduce_fusion_wrapped: argument 1"}
!11 = !{!12}
!12 = distinct !{!12, !8, !"iota_reduce_fusion_wrapped: argument 2"}
!13 = !{!10, !12}
!14 = !{!7, !12}
!15 = !{!7, !10}
!16 = distinct !{!16, !17, !18, !19}
!17 = !{!"llvm.loop.unroll.disable"}
!18 = !{!"llvm.loop.isvectorized", i32 1}
!19 = !{!"llvm.loop.unroll.runtime.disable"}
