// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

namespace NVMathNet.Interop;

// Complex (double) SVD and batched cuSOLVER / cuBLAS routines.
// Signatures verified against cusolverDn.h and cublas_api.h (CUDA 13.x).
public static unsafe partial class CuSolverNative
{
    /// <summary>Queries workspace size for complex SVD (<see cref="Zgesvd"/>).</summary>
    public static int ZgesvdBufferSize(nint handle, int m, int n)
    {
        int lwork;
        Check(((delegate* unmanaged[Cdecl]<nint, int, int, int*, CuSolverStatus>)Fn("cusolverDnZgesvd_bufferSize"))(
            handle, m, n, &lwork), "cusolverDnZgesvd_bufferSize");
        return lwork;
    }

    /// <summary>Complex SVD: A = U·diag(S)·V^H (VT holds V^H). Requires m ≥ n.</summary>
    public static void Zgesvd(
        nint handle, byte jobu, byte jobvt, int m, int n, Complex* A, int lda, double* S,
        Complex* U, int ldu, Complex* VT, int ldvt, Complex* work, int lwork, double* rwork, int* info) =>
        Check(((delegate* unmanaged[Cdecl]<nint, byte, byte, int, int, Complex*, int, double*, Complex*, int, Complex*, int, Complex*, int, double*, int*, CuSolverStatus>)Fn("cusolverDnZgesvd"))(
            handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info), "cusolverDnZgesvd");

    /// <summary>
    /// Batched complex Cholesky. <paramref name="aArray"/> is a <b>device</b> array of device pointers.
    /// </summary>
    public static void ZpotrfBatched(nint handle, CuBlasFillMode uplo, int n, Complex** aArray, int lda, int* infoArray, int batchSize) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuBlasFillMode, int, Complex**, int, int*, int, CuSolverStatus>)Fn("cusolverDnZpotrfBatched"))(
            handle, uplo, n, aArray, lda, infoArray, batchSize), "cusolverDnZpotrfBatched");

    /// <summary>Queries workspace size for batched complex Jacobi SVD.</summary>
    public static int ZgesvdjBatchedBufferSize(
        nint handle, CuSolverEigMode jobz, int m, int n, Complex* A, int lda, double* S,
        Complex* U, int ldu, Complex* V, int ldv, nint gesvdjParams, int batchSize)
    {
        int lwork;
        Check(((delegate* unmanaged[Cdecl]<nint, CuSolverEigMode, int, int, Complex*, int, double*, Complex*, int, Complex*, int, int*, nint, int, CuSolverStatus>)Fn("cusolverDnZgesvdjBatched_bufferSize"))(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &lwork, gesvdjParams, batchSize), "cusolverDnZgesvdjBatched_bufferSize");
        return lwork;
    }

    /// <summary>Batched complex Jacobi SVD (m, n ≤ 32): A = U·diag(S)·V^H per matrix.</summary>
    public static void ZgesvdjBatched(
        nint handle, CuSolverEigMode jobz, int m, int n, Complex* A, int lda, double* S,
        Complex* U, int ldu, Complex* V, int ldv, Complex* work, int lwork, int* info, nint gesvdjParams, int batchSize) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuSolverEigMode, int, int, Complex*, int, double*, Complex*, int, Complex*, int, Complex*, int, int*, nint, int, CuSolverStatus>)Fn("cusolverDnZgesvdjBatched"))(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, gesvdjParams, batchSize), "cusolverDnZgesvdjBatched");
}

public static unsafe partial class CuBlasNative
{
    /// <summary>Batched complex LU factorisation with partial pivoting.</summary>
    public static void ZgetrfBatched(nint handle, int n, Complex** aArray, int lda, int* p, int* info, int batchSize) =>
        Check(((delegate* unmanaged[Cdecl]<nint, int, Complex**, int, int*, int*, int, CuBlasStatus>)BatchedFn("cublasZgetrfBatched"))(
            handle, n, aArray, lda, p, info, batchSize), "cublasZgetrfBatched");

    /// <summary>Batched complex solve from LU factors. <paramref name="info"/> is a <b>host</b> int.</summary>
    public static void ZgetrsBatched(
        nint handle, CuBlasOperation trans, int n, int nrhs, Complex** aArray, int lda,
        int* devIpiv, Complex** bArray, int ldb, int* info, int batchSize) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuBlasOperation, int, int, Complex**, int, int*, Complex**, int, int*, int, CuBlasStatus>)BatchedFn("cublasZgetrsBatched"))(
            handle, trans, n, nrhs, aArray, lda, devIpiv, bArray, ldb, info, batchSize), "cublasZgetrsBatched");
}
