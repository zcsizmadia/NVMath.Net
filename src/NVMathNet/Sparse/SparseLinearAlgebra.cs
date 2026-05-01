// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Interop;

namespace NVMathNet.Sparse;

/// <summary>
/// Static helpers for sparse linear algebra operations using cuSPARSE.
/// </summary>
public static class SparseLinearAlgebra
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
            SpMVCore(matrix, x, y, alpha, beta, transpose, s);
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

    private static unsafe void SpMVCore<T>(
        CsrMatrix<T> matrix, DeviceBuffer<T> x, DeviceBuffer<T> y,
        float alpha, float beta, bool transpose, CudaStream s) where T : unmanaged, INumberBase<T>
    {
        var cudaType = CudaRuntime.GetCudaDataType<T>();
        var opA = transpose ? CuSparseOperation.Transpose : CuSparseOperation.None;
        var handle = CreateHandle(s);
        var dnVecX = CuSparseNative.CreateDnVec(x.Length, x.Pointer, cudaType);
        var dnVecY = CuSparseNative.CreateDnVec(y.Length, y.Pointer, cudaType);
        try
        {
            nuint bufferSize = CuSparseNative.SpMVBufferSize(
                handle,
                opA,
                &alpha, matrix.Descriptor, dnVecX,
                &beta, dnVecY,
                cudaType, CuSparseSpMVAlg.Default);

            void* buffer = bufferSize > 0 ? CudaRuntime.Malloc(bufferSize) : null;
            try
            {
                CuSparseNative.SpMV(
                    handle,
                    opA,
                    &alpha, matrix.Descriptor, dnVecX,
                    &beta, dnVecY,
                    cudaType, CuSparseSpMVAlg.Default,
                    buffer);
            }
            finally
            {
                if (buffer != null)
                {
                    CudaRuntime.Free(buffer);
                }
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
            SpMMCore(matrix, b, c, n, alpha, beta, transpose, s);
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

    private static unsafe void SpMMCore<T>(
        CsrMatrix<T> matrix, DeviceBuffer<T> b, DeviceBuffer<T> c,
        long n, float alpha, float beta, bool transpose, CudaStream s) where T : unmanaged, INumberBase<T>
    {
        var cudaType = CudaRuntime.GetCudaDataType<T>();
        var opA = transpose ? CuSparseOperation.Transpose : CuSparseOperation.None;
        var handle = CreateHandle(s);
        // order 1 = CUSPARSE_ORDER_COL (column-major)
        // B dimensions: (k × n) where k depends on op(A)
        // For transpose: op(A) is n_cols × n_rows, so B is n_rows × n and C is n_cols × n
        long cRows = transpose ? matrix.Cols : matrix.Rows;
        var dnMatB = CuSparseNative.CreateDnMat(transpose ? matrix.Rows : matrix.Cols, n, transpose ? matrix.Rows : matrix.Cols, b.Pointer, cudaType, 1);
        var dnMatC = CuSparseNative.CreateDnMat(cRows, n, cRows, c.Pointer, cudaType, 1);
        try
        {
            nuint bufferSize = CuSparseNative.SpMMBufferSize(
                handle,
                opA,
                CuSparseOperation.None,
                &alpha, matrix.Descriptor, dnMatB,
                &beta, dnMatC,
                cudaType, CuSparseSpMMAlg.Default);

            void* buffer = bufferSize > 0 ? CudaRuntime.Malloc(bufferSize) : null;
            try
            {
                CuSparseNative.SpMMPreprocess(
                    handle,
                    opA,
                    CuSparseOperation.None,
                    &alpha, matrix.Descriptor, dnMatB,
                    &beta, dnMatC,
                    cudaType, CuSparseSpMMAlg.Default,
                    buffer);

                CuSparseNative.SpMM(
                    handle,
                    opA,
                    CuSparseOperation.None,
                    &alpha, matrix.Descriptor, dnMatB,
                    &beta, dnMatC,
                    cudaType, CuSparseSpMMAlg.Default,
                    buffer);
            }
            finally
            {
                if (buffer != null)
                {
                    CudaRuntime.Free(buffer);
                }
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

    private static nint CreateHandle(CudaStream stream)
    {
        var h = CuSparseNative.Create();
        CuSparseNative.SetStream(h, stream.Handle);
        return h;
    }
}
