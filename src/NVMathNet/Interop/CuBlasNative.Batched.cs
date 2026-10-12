// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

namespace NVMathNet.Interop;

// Batched LU (getrfBatched / getrsBatched). Signatures verified against cublas_api.h (CUDA 13.x).
// The Aarray/Barray arguments are device arrays of device pointers.
public static unsafe partial class CuBlasNative
{
    private static nint BatchedFn(string name) => NativeLibraryLoader.GetExport(LibName, name);

    /// <summary>Batched single-precision LU factorisation with partial pivoting.</summary>
    /// <param name="handle">cuBLAS handle.</param>
    /// <param name="n">Order of each matrix.</param>
    /// <param name="aArray">Device array of device pointers to the matrices.</param>
    /// <param name="lda">Leading dimension.</param>
    /// <param name="p">Device array of <c>n * batchSize</c> pivot indices (output).</param>
    /// <param name="info">Device array of <c>batchSize</c> status values (output).</param>
    /// <param name="batchSize">Number of matrices.</param>
    public static void SgetrfBatched(nint handle, int n, float** aArray, int lda, int* p, int* info, int batchSize) =>
        Check(((delegate* unmanaged[Cdecl]<nint, int, float**, int, int*, int*, int, CuBlasStatus>)BatchedFn("cublasSgetrfBatched"))(
            handle, n, aArray, lda, p, info, batchSize), "cublasSgetrfBatched");

    /// <summary>Batched double-precision LU factorisation with partial pivoting.</summary>
    /// <inheritdoc cref="SgetrfBatched"/>
    public static void DgetrfBatched(nint handle, int n, double** aArray, int lda, int* p, int* info, int batchSize) =>
        Check(((delegate* unmanaged[Cdecl]<nint, int, double**, int, int*, int*, int, CuBlasStatus>)BatchedFn("cublasDgetrfBatched"))(
            handle, n, aArray, lda, p, info, batchSize), "cublasDgetrfBatched");

    /// <summary>Batched single-precision solve from LU factors. <paramref name="info"/> is a <b>host</b> int.</summary>
    /// <param name="handle">cuBLAS handle.</param>
    /// <param name="trans">Operation applied to A.</param>
    /// <param name="n">Order of each matrix.</param>
    /// <param name="nrhs">Right-hand sides per matrix.</param>
    /// <param name="aArray">Device array of device pointers to the LU factors.</param>
    /// <param name="lda">Leading dimension of A.</param>
    /// <param name="devIpiv">Device pivot array from the factorisation.</param>
    /// <param name="bArray">Device array of device pointers to the right-hand sides.</param>
    /// <param name="ldb">Leading dimension of B.</param>
    /// <param name="info">Host status value.</param>
    /// <param name="batchSize">Number of matrices.</param>
    public static void SgetrsBatched(
        nint handle, CuBlasOperation trans, int n, int nrhs, float** aArray, int lda,
        int* devIpiv, float** bArray, int ldb, int* info, int batchSize) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuBlasOperation, int, int, float**, int, int*, float**, int, int*, int, CuBlasStatus>)BatchedFn("cublasSgetrsBatched"))(
            handle, trans, n, nrhs, aArray, lda, devIpiv, bArray, ldb, info, batchSize), "cublasSgetrsBatched");

    /// <summary>Batched double-precision solve from LU factors. <paramref name="info"/> is a <b>host</b> int.</summary>
    /// <inheritdoc cref="SgetrsBatched"/>
    public static void DgetrsBatched(
        nint handle, CuBlasOperation trans, int n, int nrhs, double** aArray, int lda,
        int* devIpiv, double** bArray, int ldb, int* info, int batchSize) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuBlasOperation, int, int, double**, int, int*, double**, int, int*, int, CuBlasStatus>)BatchedFn("cublasDgetrsBatched"))(
            handle, trans, n, nrhs, aArray, lda, devIpiv, bArray, ldb, info, batchSize), "cublasDgetrsBatched");
}
