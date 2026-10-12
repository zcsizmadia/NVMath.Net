// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

namespace NVMathNet.Interop;

// Double-precision complex (cuDoubleComplex == System.Numerics.Complex) cuSOLVER routines.
// Signatures verified against cusolverDn.h (CUDA 13.x).
public static unsafe partial class CuSolverNative
{
    // ── LU ────────────────────────────────────────────────────────────────────

    /// <summary>Queries workspace size for complex LU factorisation.</summary>
    public static int ZgetrfBufferSize(nint handle, int m, int n, Complex* A, int lda)
    {
        int lwork;
        Check(((delegate* unmanaged[Cdecl]<nint, int, int, Complex*, int, int*, CuSolverStatus>)Fn("cusolverDnZgetrf_bufferSize"))(
            handle, m, n, A, lda, &lwork), "cusolverDnZgetrf_bufferSize");
        return lwork;
    }

    /// <summary>Complex LU factorisation with partial pivoting.</summary>
    public static void Zgetrf(nint handle, int m, int n, Complex* A, int lda, Complex* workspace, int* devIpiv, int* devInfo) =>
        Check(((delegate* unmanaged[Cdecl]<nint, int, int, Complex*, int, Complex*, int*, int*, CuSolverStatus>)Fn("cusolverDnZgetrf"))(
            handle, m, n, A, lda, workspace, devIpiv, devInfo), "cusolverDnZgetrf");

    /// <summary>Complex LU solve from the factors of <see cref="Zgetrf"/>.</summary>
    public static void Zgetrs(nint handle, int trans, int n, int nrhs, Complex* A, int lda, int* devIpiv, Complex* B, int ldb, int* devInfo) =>
        Check(((delegate* unmanaged[Cdecl]<nint, int, int, int, Complex*, int, int*, Complex*, int, int*, CuSolverStatus>)Fn("cusolverDnZgetrs"))(
            handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo), "cusolverDnZgetrs");

    // ── QR ────────────────────────────────────────────────────────────────────

    /// <summary>Queries workspace size for complex QR factorisation.</summary>
    public static int ZgeqrfBufferSize(nint handle, int m, int n, Complex* A, int lda)
    {
        int lwork;
        Check(((delegate* unmanaged[Cdecl]<nint, int, int, Complex*, int, int*, CuSolverStatus>)Fn("cusolverDnZgeqrf_bufferSize"))(
            handle, m, n, A, lda, &lwork), "cusolverDnZgeqrf_bufferSize");
        return lwork;
    }

    /// <summary>Complex QR factorisation.</summary>
    public static void Zgeqrf(nint handle, int m, int n, Complex* A, int lda, Complex* tau, Complex* workspace, int lwork, int* devInfo) =>
        Check(((delegate* unmanaged[Cdecl]<nint, int, int, Complex*, int, Complex*, Complex*, int, int*, CuSolverStatus>)Fn("cusolverDnZgeqrf"))(
            handle, m, n, A, lda, tau, workspace, lwork, devInfo), "cusolverDnZgeqrf");

    /// <summary>Queries workspace size for <see cref="Zungqr"/>.</summary>
    public static int ZungqrBufferSize(nint handle, int m, int n, int k, Complex* A, int lda, Complex* tau)
    {
        int lwork;
        Check(((delegate* unmanaged[Cdecl]<nint, int, int, int, Complex*, int, Complex*, int*, CuSolverStatus>)Fn("cusolverDnZungqr_bufferSize"))(
            handle, m, n, k, A, lda, tau, &lwork), "cusolverDnZungqr_bufferSize");
        return lwork;
    }

    /// <summary>Forms the first n columns of the unitary factor Q from a complex QR factorisation.</summary>
    public static void Zungqr(nint handle, int m, int n, int k, Complex* A, int lda, Complex* tau, Complex* work, int lwork, int* info) =>
        Check(((delegate* unmanaged[Cdecl]<nint, int, int, int, Complex*, int, Complex*, Complex*, int, int*, CuSolverStatus>)Fn("cusolverDnZungqr"))(
            handle, m, n, k, A, lda, tau, work, lwork, info), "cusolverDnZungqr");

    /// <summary>Queries workspace size for <see cref="Zunmqr"/>.</summary>
    public static int ZunmqrBufferSize(
        nint handle, CuBlasSideMode side, CuBlasOperation trans, int m, int n, int k,
        Complex* A, int lda, Complex* tau, Complex* C, int ldc)
    {
        int lwork;
        Check(((delegate* unmanaged[Cdecl]<nint, CuBlasSideMode, CuBlasOperation, int, int, int, Complex*, int, Complex*, Complex*, int, int*, CuSolverStatus>)Fn("cusolverDnZunmqr_bufferSize"))(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork), "cusolverDnZunmqr_bufferSize");
        return lwork;
    }

    /// <summary>Applies Q (or Q^H) from a complex QR factorisation to C.</summary>
    public static void Zunmqr(
        nint handle, CuBlasSideMode side, CuBlasOperation trans, int m, int n, int k,
        Complex* A, int lda, Complex* tau, Complex* C, int ldc, Complex* work, int lwork, int* info) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuBlasSideMode, CuBlasOperation, int, int, int, Complex*, int, Complex*, Complex*, int, Complex*, int, int*, CuSolverStatus>)Fn("cusolverDnZunmqr"))(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info), "cusolverDnZunmqr");

    // ── Cholesky ──────────────────────────────────────────────────────────────

    /// <summary>Queries workspace size for complex Cholesky factorisation.</summary>
    public static int ZpotrfBufferSize(nint handle, CuBlasFillMode uplo, int n, Complex* A, int lda)
    {
        int lwork;
        Check(((delegate* unmanaged[Cdecl]<nint, CuBlasFillMode, int, Complex*, int, int*, CuSolverStatus>)Fn("cusolverDnZpotrf_bufferSize"))(
            handle, uplo, n, A, lda, &lwork), "cusolverDnZpotrf_bufferSize");
        return lwork;
    }

    /// <summary>Cholesky factorisation of a Hermitian positive-definite matrix.</summary>
    public static void Zpotrf(nint handle, CuBlasFillMode uplo, int n, Complex* A, int lda, Complex* workspace, int lwork, int* devInfo) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuBlasFillMode, int, Complex*, int, Complex*, int, int*, CuSolverStatus>)Fn("cusolverDnZpotrf"))(
            handle, uplo, n, A, lda, workspace, lwork, devInfo), "cusolverDnZpotrf");

    // ── Hermitian eigenproblem ────────────────────────────────────────────────

    /// <summary>Queries workspace size for <see cref="Zheevd"/>.</summary>
    public static int ZheevdBufferSize(nint handle, CuSolverEigMode jobz, CuBlasFillMode uplo, int n, Complex* A, int lda, double* W)
    {
        int lwork;
        Check(((delegate* unmanaged[Cdecl]<nint, CuSolverEigMode, CuBlasFillMode, int, Complex*, int, double*, int*, CuSolverStatus>)Fn("cusolverDnZheevd_bufferSize"))(
            handle, jobz, uplo, n, A, lda, W, &lwork), "cusolverDnZheevd_bufferSize");
        return lwork;
    }

    /// <summary>Eigenvalues (and optionally eigenvectors) of a Hermitian matrix; W holds real eigenvalues.</summary>
    public static void Zheevd(nint handle, CuSolverEigMode jobz, CuBlasFillMode uplo, int n, Complex* A, int lda, double* W, Complex* work, int lwork, int* info) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuSolverEigMode, CuBlasFillMode, int, Complex*, int, double*, Complex*, int, int*, CuSolverStatus>)Fn("cusolverDnZheevd"))(
            handle, jobz, uplo, n, A, lda, W, work, lwork, info), "cusolverDnZheevd");
}
