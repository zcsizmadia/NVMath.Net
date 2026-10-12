// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Interop;
using NVMathNet.Memory;

namespace NVMathNet.Sparse;

/// <summary>
/// Static helpers for sparse linear algebra operations using cuSPARSE.
/// </summary>
public static partial class SparseLinearAlgebra
{
    /// <summary>
    /// Sparse matrix-vector product: y = alpha * op(A) * x + beta * y.
    /// </summary>
    /// <param name="matrix">Sparse CSR matrix A.</param>
    /// <param name="x">Dense vector x on device.</param>
    /// <param name="y">Dense vector y on device; updated in place.</param>
    /// <param name="alpha">Scalar multiplier for the product. Default: 1.</param>
    /// <param name="beta">Scalar multiplier for existing y. Default: 0.</param>
    /// <param name="transpose">If true, uses the transpose of the sparse matrix.</param>
    /// <param name="stream">Optional CUDA stream. <c>null</c> creates a private stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task SpMVAsync<T>(
        CsrMatrix<T> matrix,
        DeviceBuffer<T> x,
        DeviceBuffer<T> y,
        float alpha = 1.0f,
        float beta  = 0.0f,
        bool transpose = false,
        CudaStream? stream = null,
        CancellationToken ct = default) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(matrix);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);

        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            SpMVCore(matrix.Descriptor, x, y, T.CreateChecked(alpha), T.CreateChecked(beta), transpose, s);
            await s.SynchronizeAsync(ct).ConfigureAwait(false);
        }
        finally
        {
            if (ownStream)
            {
                s.Dispose();
            }
        }
    }

    internal static unsafe void SpMVCore<T>(
        nint matrixDesc, DeviceBuffer<T> x, DeviceBuffer<T> y,
        T alpha, T beta, bool transpose, CudaStream s) where T : unmanaged, INumberBase<T>
    {
        var cudaType = CudaRuntime.GetCudaDataType<T>();
        Span<byte> alphaStore = stackalloc byte[AlignedScalar.StorageSize], betaStore = stackalloc byte[AlignedScalar.StorageSize];
        T* pAlpha = AlignedScalar.Store(alphaStore, alpha);
        T* pBeta = AlignedScalar.Store(betaStore, beta);
        var opA = transpose ? CuSparseOperation.Transpose : CuSparseOperation.None;
        CuSparseNative.SpMatGetSize(matrixDesc, out long rowsA, out long colsA, out _);
        RequireLength(x.Length, transpose ? rowsA : colsA, nameof(x));
        RequireLength(y.Length, transpose ? colsA : rowsA, nameof(y));
        var handle = CreateHandle(s);
        var dnVecX = CuSparseNative.CreateDnVec(x.Length, x.Pointer, cudaType);
        var dnVecY = CuSparseNative.CreateDnVec(y.Length, y.Pointer, cudaType);
        try
        {
            nuint bufferSize = CuSparseNative.SpMVBufferSize(
                handle,
                opA,
                pAlpha, matrixDesc, dnVecX,
                pBeta, dnVecY,
                cudaType, CuSparseSpMVAlg.Default);

            var ws = new Workspace(bufferSize);
            void* buffer = ws.Pointer;
            try
            {
                CuSparseNative.SpMV(
                    handle,
                    opA,
                    pAlpha, matrixDesc, dnVecX,
                    pBeta, dnVecY,
                    cudaType, CuSparseSpMVAlg.Default,
                    buffer);
            }
            finally
            {
                SyncQuietly(s); // the kernel may still be using the workspace
                ws.Dispose();
            }
        }
        finally
        {
            CuSparseNative.DestroyDnVec(dnVecX);
            CuSparseNative.DestroyDnVec(dnVecY);
            CuSparseNative.Destroy(handle);
        }
    }

    // -- SpMM ---------------------------------------------------------------

    /// <summary>
    /// Sparse matrix-dense matrix product: C = alpha * op(A) * B + beta * C.
    /// </summary>
    /// <param name="matrix">Sparse CSR matrix A (m × k).</param>
    /// <param name="b">Dense matrix B (k × n), column-major on device.</param>
    /// <param name="c">Dense output matrix C (m × n), column-major on device; updated in place.</param>
    /// <param name="n">Number of columns in B and C.</param>
    /// <param name="alpha">Scalar multiplier for the sparse-dense product. Default: 1.</param>
    /// <param name="beta">Scalar multiplier for the existing values in C. Default: 0.</param>
    /// <param name="transpose">If true, uses the transpose of the sparse matrix.</param>
    /// <param name="stream">Optional CUDA stream. <c>null</c> creates a private stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task SpMMAsync<T>(
        CsrMatrix<T> matrix,
        DeviceBuffer<T> b,
        DeviceBuffer<T> c,
        long n,
        float alpha = 1.0f,
        float beta  = 0.0f,
        bool transpose = false,
        CudaStream? stream = null,
        CancellationToken ct = default) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(matrix);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);

        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            SpMMCore(matrix.Descriptor, matrix.Rows, matrix.Cols, b, c, n, T.CreateChecked(alpha), T.CreateChecked(beta), transpose, s);
            await s.SynchronizeAsync(ct).ConfigureAwait(false);
        }
        finally
        {
            if (ownStream)
            {
                s.Dispose();
            }
        }
    }

    internal static unsafe void SpMMCore<T>(
        nint matrixDesc, long matrixRows, long matrixCols, DeviceBuffer<T> b, DeviceBuffer<T> c,
        long n, T alpha, T beta, bool transpose, CudaStream s) where T : unmanaged, INumberBase<T>
    {
        var cudaType = CudaRuntime.GetCudaDataType<T>();
        Span<byte> alphaStore = stackalloc byte[AlignedScalar.StorageSize], betaStore = stackalloc byte[AlignedScalar.StorageSize];
        T* pAlpha = AlignedScalar.Store(alphaStore, alpha);
        T* pBeta = AlignedScalar.Store(betaStore, beta);
        var opA = transpose ? CuSparseOperation.Transpose : CuSparseOperation.None;
        ArgumentOutOfRangeException.ThrowIfLessThan(n, 1);
        RequireLength(b.Length, (transpose ? matrixRows : matrixCols) * n, nameof(b));
        RequireLength(c.Length, (transpose ? matrixCols : matrixRows) * n, nameof(c));
        var handle = CreateHandle(s);
        // order 1 = CUSPARSE_ORDER_COL (column-major)
        // B dimensions: (k × n) where k depends on op(A)
        // For transpose: op(A) is n_cols × n_rows, so B is n_rows × n and C is n_cols × n
        long cRows = transpose ? matrixCols : matrixRows;
        var dnMatB = CuSparseNative.CreateDnMat(transpose ? matrixRows : matrixCols, n, transpose ? matrixRows : matrixCols, b.Pointer, cudaType, 1);
        var dnMatC = CuSparseNative.CreateDnMat(cRows, n, cRows, c.Pointer, cudaType, 1);
        try
        {
            nuint bufferSize = CuSparseNative.SpMMBufferSize(
                handle,
                opA,
                CuSparseOperation.None,
                pAlpha, matrixDesc, dnMatB,
                pBeta, dnMatC,
                cudaType, CuSparseSpMMAlg.Default);

            var ws = new Workspace(bufferSize);
            void* buffer = ws.Pointer;
            try
            {
                CuSparseNative.SpMMPreprocess(
                    handle,
                    opA,
                    CuSparseOperation.None,
                    pAlpha, matrixDesc, dnMatB,
                    pBeta, dnMatC,
                    cudaType, CuSparseSpMMAlg.Default,
                    buffer);

                CuSparseNative.SpMM(
                    handle,
                    opA,
                    CuSparseOperation.None,
                    pAlpha, matrixDesc, dnMatB,
                    pBeta, dnMatC,
                    cudaType, CuSparseSpMMAlg.Default,
                    buffer);
            }
            finally
            {
                SyncQuietly(s); // the kernel may still be using the workspace
                ws.Dispose();
            }
        }
        finally
        {
            CuSparseNative.DestroyDnMat(dnMatB);
            CuSparseNative.DestroyDnMat(dnMatC);
            CuSparseNative.Destroy(handle);
        }
    }

    // -- Helpers ------------------------------------------------------------

    // Before a workspace goes back to its memory manager (which may hand it out again immediately), any queued work
    // that uses it must have finished - also on the error path, where the call may have been submitted already.
    // cuSPARSE reads and writes the dense operands without any bounds checks, so an undersized buffer corrupts memory
    // or poisons the CUDA context instead of failing cleanly.
    internal static void RequireLength(long actual, long required, string name)
    {
        if (actual < required)
        {
            throw new ArgumentException($"The buffer holds {actual} elements but {required} are required.", name);
        }
    }

    internal static void SyncQuietly(CudaStream s)
    {
        try
        {
            s.Synchronize();
        }
        catch (Exception)
        {
            // The context may already be broken; the original exception is the one worth reporting.
        }
    }

    private static nint CreateHandle(CudaStream stream)
    {
        var h = CuSparseNative.Create();
        CuSparseNative.SetStream(h, stream.Handle);
        return h;
    }
}
