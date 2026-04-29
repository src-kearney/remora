; ModuleID = '__compute_module_dot.1_elemental_kernel_module'
source_filename = "__compute_module_dot.1_elemental_kernel_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin23.5.0"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none) uwtable
define noalias noundef ptr @dot.1_kernel(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %workgroup_id_gep = getelementptr inbounds nuw i8, ptr %0, i64 8
  %workgroup_id = load ptr, ptr %workgroup_id_gep, align 8
  %workgroup_id_y_gep = getelementptr inbounds nuw i8, ptr %workgroup_id, i64 8
  %workgroup_id_y = load i64, ptr %workgroup_id_y_gep, align 4
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !2, !dereferenceable !3, !align !3
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !2, !dereferenceable !4, !align !3
  %arg2_gep = getelementptr i8, ptr %args, i64 32
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !2, !dereferenceable !5, !align !3
  br label %dot.1.bdot.loop_body

dot.1.bdot.loop_body:                             ; preds = %1, %dot.1.bdot.loop_body
  %dot.1.bdot.invar_address.06 = phi i64 [ 0, %1 ], [ %invar.inc, %dot.1.bdot.loop_body ]
  %invar.inc = add nuw nsw i64 %dot.1.bdot.invar_address.06, 1
  %2 = getelementptr inbounds nuw [2 x float], ptr %arg0, i64 %dot.1.bdot.invar_address.06
  %3 = getelementptr inbounds nuw [2 x [32 x float]], ptr %arg1, i64 %dot.1.bdot.invar_address.06
  %4 = getelementptr inbounds nuw [32 x float], ptr %arg2, i64 %dot.1.bdot.invar_address.06
  %.val = load float, ptr %2, align 8
  %5 = getelementptr i8, ptr %2, i64 4
  %.val5 = load float, ptr %5, align 4
  tail call fastcc void @col_major_gemv_F32_4_8_32_2_1(ptr nonnull %3, float %.val, float %.val5, ptr nonnull %4, i64 %workgroup_id_y)
  %exitcond = icmp eq i64 %invar.inc, 8
  br i1 %exitcond, label %return, label %dot.1.bdot.loop_body, !llvm.loop !6

return:                                           ; preds = %dot.1.bdot.loop_body
  ret ptr null
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define internal fastcc void @col_major_gemv_F32_4_8_32_2_1(ptr readonly captures(none) %0, float %.0.val, float %.4.val, ptr writeonly captures(none) %1, i64 %2) unnamed_addr #1 {
entry:
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 128
  %.splatinsert42 = insertelement <4 x float> poison, float %.0.val, i64 0
  %.splat43 = shufflevector <4 x float> %.splatinsert42, <4 x float> poison, <4 x i32> zeroinitializer
  %.splatinsert44 = insertelement <4 x float> poison, float %.4.val, i64 0
  %.splat45 = shufflevector <4 x float> %.splatinsert44, <4 x float> poison, <4 x i32> zeroinitializer
  %4 = shl i64 %2, 5
  %5 = add i64 %4, 32
  %6 = tail call i64 @llvm.smin.i64(i64 %5, i64 32)
  %.not1 = icmp ult i64 %4, %6
  br i1 %.not1, label %dot.inner.tiled.loop_body48, label %-after53

dot.inner.tiled.loop_body48:                      ; preds = %entry, %dot.inner.tiled.loop_body48
  %dot.inner.tiled.invar_address49.02 = phi i64 [ %invar.inc51, %dot.inner.tiled.loop_body48 ], [ %4, %entry ]
  %invar.inc51 = add nuw nsw i64 %dot.inner.tiled.invar_address49.02, 4
  %7 = getelementptr inbounds float, ptr %0, i64 %dot.inner.tiled.invar_address49.02
  %8 = load <4 x float>, ptr %7, align 4
  %9 = getelementptr inbounds float, ptr %3, i64 %dot.inner.tiled.invar_address49.02
  %10 = load <4 x float>, ptr %9, align 4
  %11 = fmul <4 x float> %.splat43, %8
  %12 = fadd <4 x float> %11, zeroinitializer
  %13 = fmul <4 x float> %.splat45, %10
  %14 = fadd <4 x float> %12, %13
  %15 = getelementptr inbounds float, ptr %1, i64 %dot.inner.tiled.invar_address49.02
  store <4 x float> %14, ptr %15, align 4
  %.not = icmp ult i64 %invar.inc51, %6
  br i1 %.not, label %dot.inner.tiled.loop_body48, label %-after53, !llvm.loop !9

-after53:                                         ; preds = %dot.inner.tiled.loop_body48, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #2

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "denormal-fp-math"="preserve-sign" "no-frame-pointer-elim"="false" }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!xla_cpu_memory_region_name = !{!0}
!llvm.module.flags = !{!1}

!0 = !{!"xla_cpu_emitter__dot_kernel_emitter__hlo_opcode__dot"}
!1 = !{i32 1, !"xla_dylib_index", i64 0}
!2 = !{}
!3 = !{i64 64}
!4 = !{i64 2048}
!5 = !{i64 1024}
!6 = distinct !{!6, !7, !8}
!7 = !{!"llvm.loop.unroll.disable"}
!8 = !{!"llvm.loop.vectorize.enable", i1 false}
!9 = distinct !{!9, !7, !8}
