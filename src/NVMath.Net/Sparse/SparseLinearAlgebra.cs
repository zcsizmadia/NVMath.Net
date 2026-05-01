// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using System.Numerics;
using NVMathNet.Interop;

namespace NVMathNet.Sparse;

/// <summary>
/// Static helpers for sparse linear algebra operations using cuSPARSE.
/// </summary>
public static class SparseLinearAlgebra
{
    // â”€â”€ SpMV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// <summary>
    /// Sparse matrix-vector product: y = alpha * op(A) * x + beta * y.
    /// </summary>
    public static async Task SpMVAsync<T>(
        CsrMatrix<T> matrix,
        DeviceBuffer<T> x,
        DeviceBuffer<T> y,
        float alpha = 1.0f,
        float beta  = 0.0f,
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
            SpMVCore(matrix, x, y, alpha, beta, s);
            await s.SynchronizeAsync(ct).ConfigureAwait(false);
        }
        finally
        {
            if (ownStream) s.Dispose();
        }
    }

    private static unsafe void SpMVCore<T>(
        CsrMatrix<T> matrix, DeviceBuffer<T> x, DeviceBuffer<T> y,
        float alpha, float beta, CudaStream s) where T : unmanaged, INumberBase<T>
    {
        var cudaType = NativeCudaType<T>();
        var handle = CreateHandle(s);
        var dnVecX = CuSparseNative.CreateDnVec(x.Length, x.Pointer, cudaType);
        var dnVecY = CuSparseNative.CreateDnVec(y.Length, y.Pointer, cudaType);
        try
        {
            nuint bufferSize = CuSparseNative.SpMVBufferSize(
                handle,
                CuSparseOperation.None,
                &alpha, matrix.Descriptor, dnVecX,
                &beta, dnVecY,
                cudaType, CuSparseSpMVAlg.Default);

            void* buffer = bufferSize > 0 ? CudaRuntime.Malloc(bufferSize) : null;
            try
            {
                CuSparseNative.SpMV(
                    handle,
                    CuSparseOperation.None,
                    &alpha, matrix.Descriptor, dnVecX,
                    &beta, dnVecY,
                    cudaType, CuSparseSpMVAlg.Default,
                    buffer);
            }
            finally
            {
                if (buffer != null) CudaRuntime.Free(buffer);
            }
        }
        finally
        {
            CuSparseNative.DestroyDnVec(dnVecX);
            CuSparseNative.DestroyDnVec(dnVecY);
            CuSparseNative.Destroy(handle);
        }
    }

    // â”€â”€ SpMM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// <summary>
    /// Sparse matrix-dense matrix product: C = alpha * op(A) * B + beta * C.
    /// </summary>
    /// <param name="matrix">Sparse CSR matrix A (m × k).</param>
    /// <param name="b">Dense matrix B (k × n), column-major on device.</param>
    /// <param name="c">Dense output matrix C (m × n), column-major on device; updated in place.</param>
    /// <param name="n">Number of columns in B and C.</param>
    /// <param name="alpha">Scalar multiplier for the sparse-dense product. Default: 1.</param>
    /// <param name="beta">Scalar multiplier for the existing values in C. Default: 0.</param>
    /// <param name="stream">Optional CUDA stream. <c>null</c> creates a private stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task SpMMAsync<T>(
        CsrMatrix<T> matrix,
        DeviceBuffer<T> b,
        DeviceBuffer<T> c,
        long n,
        float alpha = 1.0f,
        float beta  = 0.0f,
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
            SpMMCore(matrix, b, c, n, alpha, beta, s);
            await s.SynchronizeAsync(ct).ConfigureAwait(false);
        }
        finally
        {
            if (ownStream) s.Dispose();
        }
    }

    private static unsafe void SpMMCore<T>(
        CsrMatrix<T> matrix, DeviceBuffer<T> b, DeviceBuffer<T> c,
        long n, float alpha, float beta, CudaStream s) where T : unmanaged, INumberBase<T>
    {
        var cudaType = NativeCudaType<T>();
        var handle = CreateHandle(s);
        // order 0 = CUSPARSE_ORDER_COL (column-major)
        var dnMatB = CuSparseNative.CreateDnMat(matrix.Cols, n, matrix.Cols, b.Pointer, cudaType, 0);
        var dnMatC = CuSparseNative.CreateDnMat(matrix.Rows, n, matrix.Rows, c.Pointer, cudaType, 0);
        try
        {
            nuint bufferSize = CuSparseNative.SpMMBufferSize(
                handle,
                CuSparseOperation.None,
                CuSparseOperation.None,
                &alpha, matrix.Descriptor, dnMatB,
                &beta, dnMatC,
                cudaType, CuSparseSpMMAlg.Default);

            void* buffer = bufferSize > 0 ? CudaRuntime.Malloc(bufferSize) : null;
            try
            {
                CuSparseNative.SpMMPreprocess(
                    handle,
                    CuSparseOperation.None,
                    CuSparseOperation.None,
                    &alpha, matrix.Descriptor, dnMatB,
                    &beta, dnMatC,
                    cudaType, CuSparseSpMMAlg.Default,
                    buffer);

                CuSparseNative.SpMM(
                    handle,
                    CuSparseOperation.None,
                    CuSparseOperation.None,
                    &alpha, matrix.Descriptor, dnMatB,
                    &beta, dnMatC,
                    cudaType, CuSparseSpMMAlg.Default,
                    buffer);
            }
            finally
            {
                if (buffer != null) CudaRuntime.Free(buffer);
            }
        }
        finally
        {
            CuSparseNative.DestroyDnMat(dnMatB);
            CuSparseNative.DestroyDnMat(dnMatC);
            CuSparseNative.Destroy(handle);
        }
    }

    // â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    private static nint CreateHandle(CudaStream stream)
    {
        var h = CuSparseNative.Create();
        CuSparseNative.SetStream(h, stream.Handle);
        return h;
    }

    private static CudaDataType NativeCudaType<T>() where T : unmanaged, INumberBase<T>
    {
        if (typeof(T) == typeof(float))  return CudaDataType.R_32F;
        if (typeof(T) == typeof(double)) return CudaDataType.R_64F;
        if (typeof(T) == typeof(Half))   return CudaDataType.R_16F;
        throw new NotSupportedException($"Unsupported element type {typeof(T)} for sparse operations.");
    }
}
