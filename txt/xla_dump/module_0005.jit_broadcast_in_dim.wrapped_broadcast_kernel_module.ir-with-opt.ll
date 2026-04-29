; ModuleID = '__compute_module_wrapped_broadcast_kernel_module'
source_filename = "__compute_module_wrapped_broadcast_kernel_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin23.5.0"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable
define noalias noundef ptr @wrapped_broadcast(ptr readonly captures(none) %0) local_unnamed_addr #0 {
vector.ph:
  %1 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %2 = load ptr, ptr %1, align 8, !invariant.load !3
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  tail call void @llvm.experimental.noalias.scope.decl(metadata !5)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !8)
  %5 = load ptr, ptr %2, align 8, !invariant.load !3, !dereferenceable !10
  %6 = load float, ptr %5, align 4, !invariant.load !3, !alias.scope !5, !noalias !8
  %broadcast.splatinsert = insertelement <4 x float> poison, float %6, i64 0
  %7 = shufflevector <4 x float> %broadcast.splatinsert, <4 x float> poison, <8 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %8 = shl i64 %index, 3
  %9 = getelementptr i8, ptr %4, i64 %8
  store <8 x float> %7, ptr %9, align 4, !alias.scope !8, !noalias !5
  %index.next = add nuw i64 %index, 4
  %10 = icmp eq i64 %index.next, 32
  br i1 %10, label %wrapped_broadcast_wrapped.exit, label %vector.body, !llvm.loop !11

wrapped_broadcast_wrapped.exit:                   ; preds = %vector.body
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
!4 = !{i64 256}
!5 = !{!6}
!6 = distinct !{!6, !7, !"wrapped_broadcast_wrapped: argument 0"}
!7 = distinct !{!7, !"wrapped_broadcast_wrapped"}
!8 = !{!9}
!9 = distinct !{!9, !7, !"wrapped_broadcast_wrapped: argument 1"}
!10 = !{i64 4}
!11 = distinct !{!11, !12, !13, !14}
!12 = !{!"llvm.loop.unroll.disable"}
!13 = !{!"llvm.loop.isvectorized", i32 1}
!14 = !{!"llvm.loop.unroll.runtime.disable"}
