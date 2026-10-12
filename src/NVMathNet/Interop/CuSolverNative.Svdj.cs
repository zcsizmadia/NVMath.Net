// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

namespace NVMathNet.Interop;

// Batched Jacobi SVD (gesvdjBatched). Signatures verified against cusolverDn.h (CUDA 13.x).
public static unsafe partial class CuSolverNative
{
    /// <summary>Creates a <c>gesvdjInfo_t</c> parameter object (Jacobi tolerance, sweeps, residual).</summary>
    public static nint CreateGesvdjInfo()
    {
        nint info;
        Check(((delegate* unmanaged[Cdecl]<nint*, CuSolverStatus>)Fn("cusolverDnCreateGesvdjInfo"))(&info), "cusolverDnCreateGesvdjInfo");
        return info;
    }

    /// <summary>Destroys a <c>gesvdjInfo_t</c> parameter object.</summary>
    public static void DestroyGesvdjInfo(nint info) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuSolverStatus>)Fn("cusolverDnDestroyGesvdjInfo"))(info), "cusolverDnDestroyGesvdjInfo");

    /// <summary>Queries workspace size for single-precision batched Jacobi SVD.</summary>
    public static int SgesvdjBatchedBufferSize(
        nint handle, CuSolverEigMode jobz, int m, int n, float* A, int lda, float* S,
        float* U, int ldu, float* V, int ldv, nint gesvdjParams, int batchSize)
    {
        int lwork;
        Check(((delegate* unmanaged[Cdecl]<nint, CuSolverEigMode, int, int, float*, int, float*, float*, int, float*, int, int*, nint, int, CuSolverStatus>)Fn("cusolverDnSgesvdjBatched_bufferSize"))(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &lwork, gesvdjParams, batchSize), "cusolverDnSgesvdjBatched_bufferSize");
        return lwork;
    }

    /// <summary>Queries workspace size for double-precision batched Jacobi SVD.</summary>
    public static int DgesvdjBatchedBufferSize(
        nint handle, CuSolverEigMode jobz, int m, int n, double* A, int lda, double* S,
        double* U, int ldu, double* V, int ldv, nint gesvdjParams, int batchSize)
    {
        int lwork;
        Check(((delegate* unmanaged[Cdecl]<nint, CuSolverEigMode, int, int, double*, int, double*, double*, int, double*, int, int*, nint, int, CuSolverStatus>)Fn("cusolverDnDgesvdjBatched_bufferSize"))(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &lwork, gesvdjParams, batchSize), "cusolverDnDgesvdjBatched_bufferSize");
        return lwork;
    }

    /// <summary>Single-precision batched Jacobi SVD (m, n ≤ 32): A = U·diag(S)·V^T per matrix.</summary>
    public static void SgesvdjBatched(
        nint handle, CuSolverEigMode jobz, int m, int n, float* A, int lda, float* S,
        float* U, int ldu, float* V, int ldv, float* work, int lwork, int* info, nint gesvdjParams, int batchSize) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuSolverEigMode, int, int, float*, int, float*, float*, int, float*, int, float*, int, int*, nint, int, CuSolverStatus>)Fn("cusolverDnSgesvdjBatched"))(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, gesvdjParams, batchSize), "cusolverDnSgesvdjBatched");

    /// <summary>Double-precision batched Jacobi SVD (m, n ≤ 32): A = U·diag(S)·V^T per matrix.</summary>
    public static void DgesvdjBatched(
        nint handle, CuSolverEigMode jobz, int m, int n, double* A, int lda, double* S,
        double* U, int ldu, double* V, int ldv, double* work, int lwork, int* info, nint gesvdjParams, int batchSize) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuSolverEigMode, int, int, double*, int, double*, double*, int, double*, int, double*, int, int*, nint, int, CuSolverStatus>)Fn("cusolverDnDgesvdjBatched"))(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, gesvdjParams, batchSize), "cusolverDnDgesvdjBatched");
}
