; ModuleID = '__compute_module_dot.1_elemental_kernel_module'
source_filename = "__compute_module_dot.1_elemental_kernel_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin23.5.0"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_NumWorkGroups = type { i64, i64, i64 }
%XLA_CPU_WorkGroupId = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

; Function Attrs: uwtable
define ptr @dot.1_kernel(ptr %0) #0 {
  %dot.1.bdot.invar_address = alloca i64, align 8
  %num_workgroups_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 0
  %num_workgroups = load ptr, ptr %num_workgroups_gep, align 8
  %num_workgroups_x_gep = getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 0
  %num_workgroups_y_gep = getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 1
  %num_workgroups_z_gep = getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 2
  %num_workgroups_x = load i64, ptr %num_workgroups_x_gep, align 4
  %num_workgroups_y = load i64, ptr %num_workgroups_y_gep, align 4
  %num_workgroups_z = load i64, ptr %num_workgroups_z_gep, align 4
  %workgroup_id_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %workgroup_id = load ptr, ptr %workgroup_id_gep, align 8
  %workgroup_id_x_gep = getelementptr inbounds nuw %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 0
  %workgroup_id_y_gep = getelementptr inbounds nuw %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 1
  %workgroup_id_z_gep = getelementptr inbounds nuw %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 2
  %workgroup_id_x = load i64, ptr %workgroup_id_x_gep, align 4
  %workgroup_id_y = load i64, ptr %workgroup_id_y_gep, align 4
  %workgroup_id_z = load i64, ptr %workgroup_id_z_gep, align 4
  %args_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args = load ptr, ptr %args_gep, align 8
  %arg0_gep = getelementptr %XLA_CPU_KernelArg, ptr %args, i32 0, i32 0
  %arg0 = load ptr, ptr %arg0_gep, align 8, !invariant.load !2, !dereferenceable !3, !align !3
  %args_gep1 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args2 = load ptr, ptr %args_gep1, align 8
  %arg1_gep = getelementptr %XLA_CPU_KernelArg, ptr %args2, i32 1, i32 0
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !2, !dereferenceable !4, !align !3
  %args_gep3 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args4 = load ptr, ptr %args_gep3, align 8
  %arg2_gep = getelementptr %XLA_CPU_KernelArg, ptr %args4, i32 2, i32 0
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !2, !dereferenceable !5, !align !3
  store i64 0, ptr %dot.1.bdot.invar_address, align 4
  br label %dot.1.bdot.loop_header

dot.1.bdot.loop_header:                           ; preds = %dot.1.bdot.loop_body, %1
  %dot.1.bdot.indvar = load i64, ptr %dot.1.bdot.invar_address, align 4
  %2 = icmp uge i64 %dot.1.bdot.indvar, 8
  br i1 %2, label %dot.1.bdot.loop_exit, label %dot.1.bdot.loop_body

dot.1.bdot.loop_body:                             ; preds = %dot.1.bdot.loop_header
  %invar.inc = add nuw nsw i64 %dot.1.bdot.indvar, 1
  store i64 %invar.inc, ptr %dot.1.bdot.invar_address, align 4
  %3 = getelementptr inbounds [8 x [2 x float]], ptr %arg0, i64 0, i64 %dot.1.bdot.indvar, i64 0
  %4 = getelementptr inbounds [8 x [2 x [32 x float]]], ptr %arg1, i64 0, i64 %dot.1.bdot.indvar, i64 0, i64 0
  %5 = getelementptr inbounds [8 x [32 x float]], ptr %arg2, i64 0, i64 %dot.1.bdot.indvar, i64 0
  call void @col_major_gemv_F32_4_8_32_2_1(ptr %4, ptr %3, ptr %5, i64 %workgroup_id_y)
  br label %dot.1.bdot.loop_header, !llvm.loop !6

dot.1.bdot.loop_exit:                             ; preds = %dot.1.bdot.loop_header
  br label %return

return:                                           ; preds = %dot.1.bdot.loop_exit
  ret ptr null
}

; Function Attrs: uwtable
define internal void @col_major_gemv_F32_4_8_32_2_1(ptr %0, ptr %1, ptr %2, i64 %3) #1 {
entry:
  %dot.inner.tiled.invar_address49 = alloca i64, align 8
  %dot.inner.tiled.invar_address37 = alloca i64, align 8
  %dot.outer.tiled.invar_address = alloca i64, align 8
  %dot.inner.tiled.invar_address = alloca i64, align 8
  br i1 false, label %-true, label %-false

-after:                                           ; preds = %-false, %dot.outer.tiled.loop_exit
  %4 = getelementptr inbounds float, ptr %0, i64 0
  %5 = getelementptr inbounds float, ptr %0, i64 32
  %6 = getelementptr inbounds float, ptr %1, i64 0
  %7 = getelementptr inbounds float, ptr %6, i64 0
  %8 = load float, ptr %7, align 4
  %.splatinsert42 = insertelement <4 x float> poison, float %8, i64 0
  %.splat43 = shufflevector <4 x float> %.splatinsert42, <4 x float> poison, <4 x i32> zeroinitializer
  %9 = getelementptr inbounds float, ptr %6, i64 1
  %10 = load float, ptr %9, align 4
  %.splatinsert44 = insertelement <4 x float> poison, float %10, i64 0
  %.splat45 = shufflevector <4 x float> %.splatinsert44, <4 x float> poison, <4 x i32> zeroinitializer
  %11 = mul i64 %3, 32
  %12 = add i64 %11, 32
  %13 = icmp slt i64 %12, 32
  %14 = select i1 %13, i64 %12, i64 32
  store i64 %11, ptr %dot.inner.tiled.invar_address49, align 4
  br label %dot.inner.tiled.loop_header47

dot.inner.tiled.loop_header47:                    ; preds = %dot.inner.tiled.loop_body48, %-after
  %dot.inner.tiled.indvar50 = load i64, ptr %dot.inner.tiled.invar_address49, align 4
  %15 = icmp uge i64 %dot.inner.tiled.indvar50, %14
  br i1 %15, label %dot.inner.tiled.loop_exit46, label %dot.inner.tiled.loop_body48

dot.inner.tiled.loop_body48:                      ; preds = %dot.inner.tiled.loop_header47
  %invar.inc51 = add nuw nsw i64 %dot.inner.tiled.indvar50, 4
  store i64 %invar.inc51, ptr %dot.inner.tiled.invar_address49, align 4
  %16 = getelementptr inbounds float, ptr %4, i64 %dot.inner.tiled.indvar50
  %17 = load <4 x float>, ptr %16, align 4
  %18 = getelementptr inbounds float, ptr %5, i64 %dot.inner.tiled.indvar50
  %19 = load <4 x float>, ptr %18, align 4
  %20 = fmul <4 x float> %17, %.splat43
  %21 = fadd <4 x float> zeroinitializer, %20
  %22 = fmul <4 x float> %19, %.splat45
  %23 = fadd <4 x float> %21, %22
  %24 = getelementptr inbounds float, ptr %2, i64 %dot.inner.tiled.indvar50
  store <4 x float> %23, ptr %24, align 4
  br label %dot.inner.tiled.loop_header47, !llvm.loop !9

dot.inner.tiled.loop_exit46:                      ; preds = %dot.inner.tiled.loop_header47
  %25 = icmp eq i64 %3, 0
  br i1 %25, label %-true52, label %-after53

-after53:                                         ; preds = %-true52, %dot.inner.tiled.loop_exit46
  ret void

-true:                                            ; preds = %entry
  %26 = getelementptr inbounds float, ptr %0, i64 0
  %27 = getelementptr inbounds float, ptr %0, i64 32
  %28 = getelementptr inbounds float, ptr %0, i64 64
  %29 = getelementptr inbounds float, ptr %0, i64 96
  %30 = getelementptr inbounds float, ptr %0, i64 128
  %31 = getelementptr inbounds float, ptr %0, i64 160
  %32 = getelementptr inbounds float, ptr %0, i64 192
  %33 = getelementptr inbounds float, ptr %0, i64 224
  %34 = getelementptr inbounds float, ptr %1, i64 0
  %35 = getelementptr inbounds float, ptr %34, i64 0
  %36 = load float, ptr %35, align 4
  %.splatinsert = insertelement <4 x float> poison, float %36, i64 0
  %.splat = shufflevector <4 x float> %.splatinsert, <4 x float> poison, <4 x i32> zeroinitializer
  %37 = getelementptr inbounds float, ptr %34, i64 1
  %38 = load float, ptr %37, align 4
  %.splatinsert1 = insertelement <4 x float> poison, float %38, i64 0
  %.splat2 = shufflevector <4 x float> %.splatinsert1, <4 x float> poison, <4 x i32> zeroinitializer
  %39 = getelementptr inbounds float, ptr %34, i64 2
  %40 = load float, ptr %39, align 4
  %.splatinsert3 = insertelement <4 x float> poison, float %40, i64 0
  %.splat4 = shufflevector <4 x float> %.splatinsert3, <4 x float> poison, <4 x i32> zeroinitializer
  %41 = getelementptr inbounds float, ptr %34, i64 3
  %42 = load float, ptr %41, align 4
  %.splatinsert5 = insertelement <4 x float> poison, float %42, i64 0
  %.splat6 = shufflevector <4 x float> %.splatinsert5, <4 x float> poison, <4 x i32> zeroinitializer
  %43 = getelementptr inbounds float, ptr %34, i64 4
  %44 = load float, ptr %43, align 4
  %.splatinsert7 = insertelement <4 x float> poison, float %44, i64 0
  %.splat8 = shufflevector <4 x float> %.splatinsert7, <4 x float> poison, <4 x i32> zeroinitializer
  %45 = getelementptr inbounds float, ptr %34, i64 5
  %46 = load float, ptr %45, align 4
  %.splatinsert9 = insertelement <4 x float> poison, float %46, i64 0
  %.splat10 = shufflevector <4 x float> %.splatinsert9, <4 x float> poison, <4 x i32> zeroinitializer
  %47 = getelementptr inbounds float, ptr %34, i64 6
  %48 = load float, ptr %47, align 4
  %.splatinsert11 = insertelement <4 x float> poison, float %48, i64 0
  %.splat12 = shufflevector <4 x float> %.splatinsert11, <4 x float> poison, <4 x i32> zeroinitializer
  %49 = getelementptr inbounds float, ptr %34, i64 7
  %50 = load float, ptr %49, align 4
  %.splatinsert13 = insertelement <4 x float> poison, float %50, i64 0
  %.splat14 = shufflevector <4 x float> %.splatinsert13, <4 x float> poison, <4 x i32> zeroinitializer
  %51 = mul i64 %3, 32
  %52 = add i64 %51, 32
  %53 = icmp slt i64 %52, 32
  %54 = select i1 %53, i64 %52, i64 32
  store i64 %51, ptr %dot.inner.tiled.invar_address, align 4
  br label %dot.inner.tiled.loop_header

dot.inner.tiled.loop_header:                      ; preds = %dot.inner.tiled.loop_body, %-true
  %dot.inner.tiled.indvar = load i64, ptr %dot.inner.tiled.invar_address, align 4
  %55 = icmp uge i64 %dot.inner.tiled.indvar, %54
  br i1 %55, label %dot.inner.tiled.loop_exit, label %dot.inner.tiled.loop_body

dot.inner.tiled.loop_body:                        ; preds = %dot.inner.tiled.loop_header
  %invar.inc = add nuw nsw i64 %dot.inner.tiled.indvar, 4
  store i64 %invar.inc, ptr %dot.inner.tiled.invar_address, align 4
  %56 = getelementptr inbounds float, ptr %26, i64 %dot.inner.tiled.indvar
  %57 = load <4 x float>, ptr %56, align 4
  %58 = getelementptr inbounds float, ptr %27, i64 %dot.inner.tiled.indvar
  %59 = load <4 x float>, ptr %58, align 4
  %60 = getelementptr inbounds float, ptr %28, i64 %dot.inner.tiled.indvar
  %61 = load <4 x float>, ptr %60, align 4
  %62 = getelementptr inbounds float, ptr %29, i64 %dot.inner.tiled.indvar
  %63 = load <4 x float>, ptr %62, align 4
  %64 = getelementptr inbounds float, ptr %30, i64 %dot.inner.tiled.indvar
  %65 = load <4 x float>, ptr %64, align 4
  %66 = getelementptr inbounds float, ptr %31, i64 %dot.inner.tiled.indvar
  %67 = load <4 x float>, ptr %66, align 4
  %68 = getelementptr inbounds float, ptr %32, i64 %dot.inner.tiled.indvar
  %69 = load <4 x float>, ptr %68, align 4
  %70 = getelementptr inbounds float, ptr %33, i64 %dot.inner.tiled.indvar
  %71 = load <4 x float>, ptr %70, align 4
  %72 = fmul <4 x float> %57, %.splat
  %73 = fadd <4 x float> zeroinitializer, %72
  %74 = fmul <4 x float> %59, %.splat2
  %75 = fadd <4 x float> %73, %74
  %76 = fmul <4 x float> %61, %.splat4
  %77 = fadd <4 x float> %75, %76
  %78 = fmul <4 x float> %63, %.splat6
  %79 = fadd <4 x float> %77, %78
  %80 = fmul <4 x float> %65, %.splat8
  %81 = fadd <4 x float> %79, %80
  %82 = fmul <4 x float> %67, %.splat10
  %83 = fadd <4 x float> %81, %82
  %84 = fmul <4 x float> %69, %.splat12
  %85 = fadd <4 x float> %83, %84
  %86 = fmul <4 x float> %71, %.splat14
  %87 = fadd <4 x float> %85, %86
  %88 = getelementptr inbounds float, ptr %2, i64 %dot.inner.tiled.indvar
  store <4 x float> %87, ptr %88, align 4
  br label %dot.inner.tiled.loop_header, !llvm.loop !10

dot.inner.tiled.loop_exit:                        ; preds = %dot.inner.tiled.loop_header
  %89 = icmp eq i64 %3, 0
  br i1 %89, label %-true15, label %-after16

-after16:                                         ; preds = %-true15, %dot.inner.tiled.loop_exit
  store i64 8, ptr %dot.outer.tiled.invar_address, align 4
  br label %dot.outer.tiled.loop_header

dot.outer.tiled.loop_header:                      ; preds = %-after41, %-after16
  %dot.outer.tiled.indvar = load i64, ptr %dot.outer.tiled.invar_address, align 4
  %90 = icmp uge i64 %dot.outer.tiled.indvar, 0
  br i1 %90, label %dot.outer.tiled.loop_exit, label %dot.outer.tiled.loop_body

dot.outer.tiled.loop_body:                        ; preds = %dot.outer.tiled.loop_header
  %invar.inc17 = add nuw nsw i64 %dot.outer.tiled.indvar, 8
  store i64 %invar.inc17, ptr %dot.outer.tiled.invar_address, align 4
  %91 = add i64 0, %dot.outer.tiled.indvar
  %92 = mul i64 32, %91
  %93 = getelementptr inbounds float, ptr %0, i64 %92
  %94 = add i64 1, %dot.outer.tiled.indvar
  %95 = mul i64 32, %94
  %96 = getelementptr inbounds float, ptr %0, i64 %95
  %97 = add i64 2, %dot.outer.tiled.indvar
  %98 = mul i64 32, %97
  %99 = getelementptr inbounds float, ptr %0, i64 %98
  %100 = add i64 3, %dot.outer.tiled.indvar
  %101 = mul i64 32, %100
  %102 = getelementptr inbounds float, ptr %0, i64 %101
  %103 = add i64 4, %dot.outer.tiled.indvar
  %104 = mul i64 32, %103
  %105 = getelementptr inbounds float, ptr %0, i64 %104
  %106 = add i64 5, %dot.outer.tiled.indvar
  %107 = mul i64 32, %106
  %108 = getelementptr inbounds float, ptr %0, i64 %107
  %109 = add i64 6, %dot.outer.tiled.indvar
  %110 = mul i64 32, %109
  %111 = getelementptr inbounds float, ptr %0, i64 %110
  %112 = add i64 7, %dot.outer.tiled.indvar
  %113 = mul i64 32, %112
  %114 = getelementptr inbounds float, ptr %0, i64 %113
  %115 = getelementptr inbounds float, ptr %1, i64 %dot.outer.tiled.indvar
  %116 = getelementptr inbounds float, ptr %115, i64 0
  %117 = load float, ptr %116, align 4
  %.splatinsert18 = insertelement <4 x float> poison, float %117, i64 0
  %.splat19 = shufflevector <4 x float> %.splatinsert18, <4 x float> poison, <4 x i32> zeroinitializer
  %118 = getelementptr inbounds float, ptr %115, i64 1
  %119 = load float, ptr %118, align 4
  %.splatinsert20 = insertelement <4 x float> poison, float %119, i64 0
  %.splat21 = shufflevector <4 x float> %.splatinsert20, <4 x float> poison, <4 x i32> zeroinitializer
  %120 = getelementptr inbounds float, ptr %115, i64 2
  %121 = load float, ptr %120, align 4
  %.splatinsert22 = insertelement <4 x float> poison, float %121, i64 0
  %.splat23 = shufflevector <4 x float> %.splatinsert22, <4 x float> poison, <4 x i32> zeroinitializer
  %122 = getelementptr inbounds float, ptr %115, i64 3
  %123 = load float, ptr %122, align 4
  %.splatinsert24 = insertelement <4 x float> poison, float %123, i64 0
  %.splat25 = shufflevector <4 x float> %.splatinsert24, <4 x float> poison, <4 x i32> zeroinitializer
  %124 = getelementptr inbounds float, ptr %115, i64 4
  %125 = load float, ptr %124, align 4
  %.splatinsert26 = insertelement <4 x float> poison, float %125, i64 0
  %.splat27 = shufflevector <4 x float> %.splatinsert26, <4 x float> poison, <4 x i32> zeroinitializer
  %126 = getelementptr inbounds float, ptr %115, i64 5
  %127 = load float, ptr %126, align 4
  %.splatinsert28 = insertelement <4 x float> poison, float %127, i64 0
  %.splat29 = shufflevector <4 x float> %.splatinsert28, <4 x float> poison, <4 x i32> zeroinitializer
  %128 = getelementptr inbounds float, ptr %115, i64 6
  %129 = load float, ptr %128, align 4
  %.splatinsert30 = insertelement <4 x float> poison, float %129, i64 0
  %.splat31 = shufflevector <4 x float> %.splatinsert30, <4 x float> poison, <4 x i32> zeroinitializer
  %130 = getelementptr inbounds float, ptr %115, i64 7
  %131 = load float, ptr %130, align 4
  %.splatinsert32 = insertelement <4 x float> poison, float %131, i64 0
  %.splat33 = shufflevector <4 x float> %.splatinsert32, <4 x float> poison, <4 x i32> zeroinitializer
  %132 = mul i64 %3, 32
  %133 = add i64 %132, 32
  %134 = icmp slt i64 %133, 32
  %135 = select i1 %134, i64 %133, i64 32
  store i64 %132, ptr %dot.inner.tiled.invar_address37, align 4
  br label %dot.inner.tiled.loop_header35

dot.inner.tiled.loop_header35:                    ; preds = %dot.inner.tiled.loop_body36, %dot.outer.tiled.loop_body
  %dot.inner.tiled.indvar38 = load i64, ptr %dot.inner.tiled.invar_address37, align 4
  %136 = icmp uge i64 %dot.inner.tiled.indvar38, %135
  br i1 %136, label %dot.inner.tiled.loop_exit34, label %dot.inner.tiled.loop_body36

dot.inner.tiled.loop_body36:                      ; preds = %dot.inner.tiled.loop_header35
  %invar.inc39 = add nuw nsw i64 %dot.inner.tiled.indvar38, 4
  store i64 %invar.inc39, ptr %dot.inner.tiled.invar_address37, align 4
  %137 = getelementptr inbounds float, ptr %93, i64 %dot.inner.tiled.indvar38
  %138 = load <4 x float>, ptr %137, align 4
  %139 = getelementptr inbounds float, ptr %96, i64 %dot.inner.tiled.indvar38
  %140 = load <4 x float>, ptr %139, align 4
  %141 = getelementptr inbounds float, ptr %99, i64 %dot.inner.tiled.indvar38
  %142 = load <4 x float>, ptr %141, align 4
  %143 = getelementptr inbounds float, ptr %102, i64 %dot.inner.tiled.indvar38
  %144 = load <4 x float>, ptr %143, align 4
  %145 = getelementptr inbounds float, ptr %105, i64 %dot.inner.tiled.indvar38
  %146 = load <4 x float>, ptr %145, align 4
  %147 = getelementptr inbounds float, ptr %108, i64 %dot.inner.tiled.indvar38
  %148 = load <4 x float>, ptr %147, align 4
  %149 = getelementptr inbounds float, ptr %111, i64 %dot.inner.tiled.indvar38
  %150 = load <4 x float>, ptr %149, align 4
  %151 = getelementptr inbounds float, ptr %114, i64 %dot.inner.tiled.indvar38
  %152 = load <4 x float>, ptr %151, align 4
  %153 = getelementptr inbounds float, ptr %2, i64 %dot.inner.tiled.indvar38
  %154 = load <4 x float>, ptr %153, align 4
  %155 = fmul <4 x float> %138, %.splat19
  %156 = fadd <4 x float> %154, %155
  %157 = fmul <4 x float> %140, %.splat21
  %158 = fadd <4 x float> %156, %157
  %159 = fmul <4 x float> %142, %.splat23
  %160 = fadd <4 x float> %158, %159
  %161 = fmul <4 x float> %144, %.splat25
  %162 = fadd <4 x float> %160, %161
  %163 = fmul <4 x float> %146, %.splat27
  %164 = fadd <4 x float> %162, %163
  %165 = fmul <4 x float> %148, %.splat29
  %166 = fadd <4 x float> %164, %165
  %167 = fmul <4 x float> %150, %.splat31
  %168 = fadd <4 x float> %166, %167
  %169 = fmul <4 x float> %152, %.splat33
  %170 = fadd <4 x float> %168, %169
  %171 = getelementptr inbounds float, ptr %2, i64 %dot.inner.tiled.indvar38
  store <4 x float> %170, ptr %171, align 4
  br label %dot.inner.tiled.loop_header35, !llvm.loop !11

dot.inner.tiled.loop_exit34:                      ; preds = %dot.inner.tiled.loop_header35
  %172 = icmp eq i64 %3, 0
  br i1 %172, label %-true40, label %-after41

-after41:                                         ; preds = %-true40, %dot.inner.tiled.loop_exit34
  br label %dot.outer.tiled.loop_header, !llvm.loop !12

dot.outer.tiled.loop_exit:                        ; preds = %dot.outer.tiled.loop_header
  br label %-after

-false:                                           ; preds = %entry
  br label %-after

-true15:                                          ; preds = %dot.inner.tiled.loop_exit
  br label %-after16

-true40:                                          ; preds = %dot.inner.tiled.loop_exit34
  br label %-after41

-true52:                                          ; preds = %dot.inner.tiled.loop_exit46
  br label %-after53
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { uwtable "denormal-fp-math"="preserve-sign" "no-frame-pointer-elim"="false" }

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
!10 = distinct !{!10, !7, !8}
!11 = distinct !{!11, !7, !8}
!12 = distinct !{!12, !7, !8}
