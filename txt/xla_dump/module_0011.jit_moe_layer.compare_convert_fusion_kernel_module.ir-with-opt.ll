; ModuleID = '__compute_module_compare_convert_fusion_kernel_module'
source_filename = "__compute_module_compare_convert_fusion_kernel_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin23.5.0"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable
define noalias noundef ptr @compare_convert_fusion(ptr readonly captures(none) %0) local_unnamed_addr #0 {
vector.ph:
  %1 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %2 = load ptr, ptr %1, align 8, !invariant.load !3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3, !dereferenceable !4
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !5
  tail call void @llvm.experimental.noalias.scope.decl(metadata !6)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !9)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %6 = getelementptr inbounds nuw i32, ptr %3, i64 %index
  %wide.load = load <4 x i32>, ptr %6, align 4, !invariant.load !3, !alias.scope !6, !noalias !9
  %7 = shl nuw nsw i64 %index, 3
  %8 = getelementptr i8, ptr %5, i64 %7
  %9 = icmp eq <4 x i32> %wide.load, zeroinitializer
  %10 = uitofp <4 x i1> %9 to <4 x float>
  %11 = icmp eq <4 x i32> %wide.load, splat (i32 1)
  %12 = uitofp <4 x i1> %11 to <4 x float>
  %interleaved.vec = shufflevector <4 x float> %10, <4 x float> %12, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x float> %interleaved.vec, ptr %8, align 4, !alias.scope !9, !noalias !6
  %index.next = add nuw i64 %index, 4
  %13 = icmp eq i64 %index.next, 8
  br i1 %13, label %compare_convert_fusion_wrapped.exit, label %vector.body, !llvm.loop !11

compare_convert_fusion_wrapped.exit:              ; preds = %vector.body
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 1}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 32}
!5 = !{i64 64}
!6 = !{!7}
!7 = distinct !{!7, !8, !"compare_convert_fusion_wrapped: argument 0"}
!8 = distinct !{!8, !"compare_convert_fusion_wrapped"}
!9 = !{!10}
!10 = distinct !{!10, !8, !"compare_convert_fusion_wrapped: argument 1"}
!11 = distinct !{!11, !12, !13, !14}
!12 = !{!"llvm.loop.unroll.disable"}
!13 = !{!"llvm.loop.isvectorized", i32 1}
!14 = !{!"llvm.loop.unroll.runtime.disable"}
