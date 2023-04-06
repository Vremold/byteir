; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define void @Unknown0(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21) {
  br label %23

23:                                               ; preds = %26, %22
  %24 = phi i64 [ %115, %26 ], [ 0, %22 ]
  %25 = icmp slt i64 %24, 6272
  br i1 %25, label %26, label %116

26:                                               ; preds = %23
  %27 = srem i64 %24, 28
  %28 = sdiv i64 %24, 28
  %29 = mul i64 %27, 8
  %30 = mul i64 %28, 224
  %31 = add i64 %30, %29
  %32 = mul i64 %28, 672
  %33 = mul i64 %29, 3
  %34 = add i64 %32, %33
  %35 = add i64 %31, 0
  %36 = add i64 %35, 0
  %37 = add i64 %36, 0
  %38 = add i64 %37, 0
  %39 = getelementptr float, ptr %1, i64 %38
  %40 = load <8 x float>, ptr %39, align 4
  %41 = add i64 %31, 0
  %42 = add i64 %41, 50176
  %43 = add i64 %42, 0
  %44 = add i64 %43, 0
  %45 = getelementptr float, ptr %1, i64 %44
  %46 = load <8 x float>, ptr %45, align 4
  %47 = add i64 %31, 0
  %48 = add i64 %47, 100352
  %49 = add i64 %48, 0
  %50 = add i64 %49, 0
  %51 = getelementptr float, ptr %1, i64 %50
  %52 = load <8 x float>, ptr %51, align 4
  %53 = shufflevector <8 x float> %40, <8 x float> %46, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  %54 = shufflevector <8 x float> %40, <8 x float> %46, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  %55 = shufflevector <8 x float> %52, <8 x float> zeroinitializer, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  %56 = shufflevector <8 x float> %52, <8 x float> zeroinitializer, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  %57 = shufflevector <8 x float> %53, <8 x float> %55, <8 x i32> <i32 2, i32 3, i32 8, i32 9, i32 6, i32 7, i32 12, i32 13>
  %58 = shufflevector <8 x float> %54, <8 x float> %56, <8 x i32> <i32 2, i32 3, i32 8, i32 9, i32 6, i32 7, i32 12, i32 13>
  %59 = call <8 x float> asm inteldialect "vblendps $0, $1, $2, 0xcc", "=x,x,x"(<8 x float> %53, <8 x float> %57)
  %60 = call <8 x float> asm inteldialect "vblendps $0, $1, $2, 0x33", "=x,x,x"(<8 x float> %55, <8 x float> %57)
  %61 = call <8 x float> asm inteldialect "vblendps $0, $1, $2, 0xcc", "=x,x,x"(<8 x float> %54, <8 x float> %58)
  %62 = call <8 x float> asm inteldialect "vblendps $0, $1, $2, 0x33", "=x,x,x"(<8 x float> %56, <8 x float> %58)
  %63 = call <8 x float> asm inteldialect "vblendps $0, $1, $2, 0xcc", "=x,x,x"(<8 x float> zeroinitializer, <8 x float> zeroinitializer)
  %64 = call <8 x float> asm inteldialect "vblendps $0, $1, $2, 0x33", "=x,x,x"(<8 x float> zeroinitializer, <8 x float> zeroinitializer)
  %65 = call <8 x float> asm inteldialect "vblendps $0, $1, $2, 0xcc", "=x,x,x"(<8 x float> zeroinitializer, <8 x float> zeroinitializer)
  %66 = call <8 x float> asm inteldialect "vblendps $0, $1, $2, 0x33", "=x,x,x"(<8 x float> zeroinitializer, <8 x float> zeroinitializer)
  %67 = shufflevector <8 x float> %59, <8 x float> %63, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  %68 = shufflevector <8 x float> %60, <8 x float> %64, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  %69 = shufflevector <8 x float> %61, <8 x float> %65, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  %70 = shufflevector <8 x float> %62, <8 x float> %66, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  %71 = shufflevector <8 x float> %59, <8 x float> %63, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  %72 = shufflevector <8 x float> %60, <8 x float> %64, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  %73 = shufflevector <8 x float> %61, <8 x float> %65, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  %74 = shufflevector <8 x float> %62, <8 x float> %66, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  %75 = add i64 %34, 0
  %76 = add i64 %75, 0
  %77 = add i64 %76, 0
  %78 = add i64 %77, 0
  %79 = getelementptr float, ptr %12, i64 %78
  call void @llvm.masked.store.v8f32.p0(<8 x float> %67, ptr %79, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false>)
  %80 = add i64 %34, 0
  %81 = add i64 %80, 0
  %82 = add i64 %81, 3
  %83 = add i64 %82, 0
  %84 = getelementptr float, ptr %12, i64 %83
  call void @llvm.masked.store.v8f32.p0(<8 x float> %68, ptr %84, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false>)
  %85 = add i64 %34, 0
  %86 = add i64 %85, 0
  %87 = add i64 %86, 6
  %88 = add i64 %87, 0
  %89 = getelementptr float, ptr %12, i64 %88
  call void @llvm.masked.store.v8f32.p0(<8 x float> %69, ptr %89, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false>)
  %90 = add i64 %34, 0
  %91 = add i64 %90, 0
  %92 = add i64 %91, 9
  %93 = add i64 %92, 0
  %94 = getelementptr float, ptr %12, i64 %93
  call void @llvm.masked.store.v8f32.p0(<8 x float> %70, ptr %94, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false>)
  %95 = add i64 %34, 0
  %96 = add i64 %95, 0
  %97 = add i64 %96, 12
  %98 = add i64 %97, 0
  %99 = getelementptr float, ptr %12, i64 %98
  call void @llvm.masked.store.v8f32.p0(<8 x float> %71, ptr %99, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false>)
  %100 = add i64 %34, 0
  %101 = add i64 %100, 0
  %102 = add i64 %101, 15
  %103 = add i64 %102, 0
  %104 = getelementptr float, ptr %12, i64 %103
  call void @llvm.masked.store.v8f32.p0(<8 x float> %72, ptr %104, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false>)
  %105 = add i64 %34, 0
  %106 = add i64 %105, 0
  %107 = add i64 %106, 18
  %108 = add i64 %107, 0
  %109 = getelementptr float, ptr %12, i64 %108
  call void @llvm.masked.store.v8f32.p0(<8 x float> %73, ptr %109, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false>)
  %110 = add i64 %34, 0
  %111 = add i64 %110, 0
  %112 = add i64 %111, 21
  %113 = add i64 %112, 0
  %114 = getelementptr float, ptr %12, i64 %113
  call void @llvm.masked.store.v8f32.p0(<8 x float> %74, ptr %114, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false>)
  %115 = add i64 %24, 1
  br label %23

116:                                              ; preds = %23
  ret void
}

define void @_mlir_ciface_Unknown0(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, i64, [4 x i64], [4 x i64] }, ptr %0, align 8
  %4 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 0
  %5 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 1
  %6 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 2
  %7 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 3, 0
  %8 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 3, 1
  %9 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 3, 2
  %10 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 3, 3
  %11 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 4, 0
  %12 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 4, 1
  %13 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 4, 2
  %14 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 4, 3
  %15 = load { ptr, ptr, i64, [4 x i64], [4 x i64] }, ptr %1, align 8
  %16 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 0
  %17 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 1
  %18 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 2
  %19 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 3, 0
  %20 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 3, 1
  %21 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 3, 2
  %22 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 3, 3
  %23 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 4, 0
  %24 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 4, 1
  %25 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 4, 2
  %26 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, 4, 3
  call void @Unknown0(ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.masked.store.v8f32.p0(<8 x float>, ptr nocapture, i32 immarg, <8 x i1>) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
