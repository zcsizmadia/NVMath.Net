// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

namespace NVMathNet.Interop;

// 64-bit-count vector scaling (x = alpha * x). Signatures verified against cublas_api.h (CUDA 13.x).
public static unsafe partial class CuBlasNative
{
    /// <summary>Scales a single-precision vector in place: x = alpha * x (64-bit element count).</summary>
    public static void Sscal(nint handle, long n, float alpha, float* x) =>
        Check(((delegate* unmanaged[Cdecl]<nint, long, float*, float*, long, CuBlasStatus>)
            NativeLibraryLoader.GetExport(LibName, "cublasSscal_v2_64"))(handle, n, &alpha, x, 1), "cublasSscal_v2_64");

    /// <summary>Scales a double-precision vector in place: x = alpha * x (64-bit element count).</summary>
    public static void Dscal(nint handle, long n, double alpha, double* x) =>
        Check(((delegate* unmanaged[Cdecl]<nint, long, double*, double*, long, CuBlasStatus>)
            NativeLibraryLoader.GetExport(LibName, "cublasDscal_v2_64"))(handle, n, &alpha, x, 1), "cublasDscal_v2_64");
}
