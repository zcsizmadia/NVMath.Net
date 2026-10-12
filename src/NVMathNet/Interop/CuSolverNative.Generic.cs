// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

namespace NVMathNet.Interop;

// Additional cuSOLVER dense bindings: orgqr, ormqr, batched Cholesky.
// Signatures verified against cusolverDn.h (CUDA 13.x).
public static unsafe partial class CuSolverNative
{
    private static nint Fn(string name) => NativeLibraryLoader.GetExport(LibName, name);

    // ── orgqr: generate Q from a QR factorisation ────────────────────────────

    /// <summary>Queries workspace size for single-precision <c>orgqr</c>.</summary>
    public static int SorgqrBufferSize(nint handle, int m, int n, int k, float* A, int lda, float* tau)
    {
        int lwork;
        Check(((delegate* unmanaged[Cdecl]<nint, int, int, int, float*, int, float*, int*, CuSolverStatus>)Fn("cusolverDnSorgqr_bufferSize"))(
            handle, m, n, k, A, lda, tau, &lwork), "cusolverDnSorgqr_bufferSize");
        return lwork;
    }

    /// <summary>Queries workspace size for double-precision <c>orgqr</c>.</summary>
    public static int DorgqrBufferSize(nint handle, int m, int n, int k, double* A, int lda, double* tau)
    {
        int lwork;
        Check(((delegate* unmanaged[Cdecl]<nint, int, int, int, double*, int, double*, int*, CuSolverStatus>)Fn("cusolverDnDorgqr_bufferSize"))(
            handle, m, n, k, A, lda, tau, &lwork), "cusolverDnDorgqr_bufferSize");
        return lwork;
    }

    /// <summary>Overwrites A (holding Householder vectors from <c>geqrf</c>) with the first n columns of Q.</summary>
    public static void Sorgqr(nint handle, int m, int n, int k, float* A, int lda, float* tau, float* work, int lwork, int* info) =>
        Check(((delegate* unmanaged[Cdecl]<nint, int, int, int, float*, int, float*, float*, int, int*, CuSolverStatus>)Fn("cusolverDnSorgqr"))(
            handle, m, n, k, A, lda, tau, work, lwork, info), "cusolverDnSorgqr");

    /// <summary>Overwrites A (holding Householder vectors from <c>geqrf</c>) with the first n columns of Q.</summary>
    public static void Dorgqr(nint handle, int m, int n, int k, double* A, int lda, double* tau, double* work, int lwork, int* info) =>
        Check(((delegate* unmanaged[Cdecl]<nint, int, int, int, double*, int, double*, double*, int, int*, CuSolverStatus>)Fn("cusolverDnDorgqr"))(
            handle, m, n, k, A, lda, tau, work, lwork, info), "cusolverDnDorgqr");

    // ── ormqr: apply Q from a QR factorisation ───────────────────────────────

    /// <summary>Queries workspace size for single-precision <c>ormqr</c>.</summary>
    public static int SormqrBufferSize(
        nint handle, CuBlasSideMode side, CuBlasOperation trans, int m, int n, int k,
        float* A, int lda, float* tau, float* C, int ldc)
    {
        int lwork;
        Check(((delegate* unmanaged[Cdecl]<nint, CuBlasSideMode, CuBlasOperation, int, int, int, float*, int, float*, float*, int, int*, CuSolverStatus>)Fn("cusolverDnSormqr_bufferSize"))(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork), "cusolverDnSormqr_bufferSize");
        return lwork;
    }

    /// <summary>Queries workspace size for double-precision <c>ormqr</c>.</summary>
    public static int DormqrBufferSize(
        nint handle, CuBlasSideMode side, CuBlasOperation trans, int m, int n, int k,
        double* A, int lda, double* tau, double* C, int ldc)
    {
        int lwork;
        Check(((delegate* unmanaged[Cdecl]<nint, CuBlasSideMode, CuBlasOperation, int, int, int, double*, int, double*, double*, int, int*, CuSolverStatus>)Fn("cusolverDnDormqr_bufferSize"))(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork), "cusolverDnDormqr_bufferSize");
        return lwork;
    }

    /// <summary>Applies Q (or Q^T) from <c>geqrf</c> to C: C = op(Q) C or C op(Q).</summary>
    public static void Sormqr(
        nint handle, CuBlasSideMode side, CuBlasOperation trans, int m, int n, int k,
        float* A, int lda, float* tau, float* C, int ldc, float* work, int lwork, int* info) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuBlasSideMode, CuBlasOperation, int, int, int, float*, int, float*, float*, int, float*, int, int*, CuSolverStatus>)Fn("cusolverDnSormqr"))(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info), "cusolverDnSormqr");

    /// <summary>Applies Q (or Q^T) from <c>geqrf</c> to C: C = op(Q) C or C op(Q).</summary>
    public static void Dormqr(
        nint handle, CuBlasSideMode side, CuBlasOperation trans, int m, int n, int k,
        double* A, int lda, double* tau, double* C, int ldc, double* work, int lwork, int* info) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuBlasSideMode, CuBlasOperation, int, int, int, double*, int, double*, double*, int, double*, int, int*, CuSolverStatus>)Fn("cusolverDnDormqr"))(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info), "cusolverDnDormqr");

    // ── Batched Cholesky ─────────────────────────────────────────────────────

    /// <summary>
    /// Batched single-precision Cholesky. <paramref name="aArray"/> is a <b>device</b> array of
    /// <paramref name="batchSize"/> device pointers, each to an n×n matrix.
    /// </summary>
    public static void SpotrfBatched(nint handle, CuBlasFillMode uplo, int n, float** aArray, int lda, int* infoArray, int batchSize) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuBlasFillMode, int, float**, int, int*, int, CuSolverStatus>)Fn("cusolverDnSpotrfBatched"))(
            handle, uplo, n, aArray, lda, infoArray, batchSize), "cusolverDnSpotrfBatched");

    /// <summary>
    /// Batched double-precision Cholesky. <paramref name="aArray"/> is a <b>device</b> array of
    /// <paramref name="batchSize"/> device pointers, each to an n×n matrix.
    /// </summary>
    public static void DpotrfBatched(nint handle, CuBlasFillMode uplo, int n, double** aArray, int lda, int* infoArray, int batchSize) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuBlasFillMode, int, double**, int, int*, int, CuSolverStatus>)Fn("cusolverDnDpotrfBatched"))(
            handle, uplo, n, aArray, lda, infoArray, batchSize), "cusolverDnDpotrfBatched");
}
