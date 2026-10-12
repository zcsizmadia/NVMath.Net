// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

namespace NVMathNet.Interop;

// Double-precision complex triangular solve. Signature verified against cublas_api.h (CUDA 13.x).
public static unsafe partial class CuBlasNative
{
    /// <summary>
    /// Complex double-precision triangular solve:
    /// B = alpha * inv(op(A)) * B (side = Left) or B = alpha * B * inv(op(A)) (side = Right).
    /// </summary>
    public static void Ztrsm(
        nint handle, CuBlasSideMode side, CuBlasFillMode uplo, CuBlasOperation trans, CuBlasDiagType diag,
        int m, int n, Complex* alpha, Complex* a, int lda, Complex* b, int ldb) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuBlasSideMode, CuBlasFillMode, CuBlasOperation, CuBlasDiagType, int, int, Complex*, Complex*, int, Complex*, int, CuBlasStatus>)
            NativeLibraryLoader.GetExport(LibName, "cublasZtrsm_v2"))(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb), "cublasZtrsm_v2");
}
