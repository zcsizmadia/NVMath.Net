// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Interop;

namespace NVMathNet.LinAlg;

/// <summary>
/// Stateless helpers for cuBLAS symmetric matrix-matrix multiply (SYMM):
/// C = alpha * A * B + beta * C (side=Left) or C = alpha * B * A + beta * C (side=Right)
/// where A is symmetric, in column-major layout.
/// </summary>
public static class SymmetricMultiply
{
    /// <summary>
    /// Symmetric matrix-matrix multiply (single-precision, async).
    /// </summary>
    /// <param name="a">Device buffer for symmetric matrix A.</param>
    /// <param name="b">Device buffer for matrix B (m×n).</param>
    /// <param name="c">Device buffer for matrix C (m×n), updated in place.</param>
    /// <param name="m">Rows of B and C.</param>
    /// <param name="n">Columns of B and C.</param>
    /// <param name="alpha">Scalar multiplier for A*B. Default: 1.</param>
    /// <param name="beta">Scalar multiplier for existing C. Default: 0.</param>
    /// <param name="leftSide">If true, A is on the left (A*B); if false, right (B*A).</param>
    /// <param name="upper">If true, upper triangle of A is stored; otherwise lower.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task SsymmAsync(
        DeviceBuffer<float> a, DeviceBuffer<float> b, DeviceBuffer<float> c,
        int m, int n,
        float alpha = 1.0f, float beta = 0.0f,
        bool leftSide = true, bool upper = false,
        CudaStream? stream = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);

        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            SsymmCore(a, b, c, m, n, alpha, beta, leftSide, upper, s);
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

    /// <summary>
    /// Symmetric matrix-matrix multiply (double-precision, async).
    /// </summary>
    public static async Task DsymmAsync(
        DeviceBuffer<double> a, DeviceBuffer<double> b, DeviceBuffer<double> c,
        int m, int n,
        double alpha = 1.0, double beta = 0.0,
        bool leftSide = true, bool upper = false,
        CudaStream? stream = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);

        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            DsymmCore(a, b, c, m, n, alpha, beta, leftSide, upper, s);
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

    private static unsafe void SsymmCore(
        DeviceBuffer<float> a, DeviceBuffer<float> b, DeviceBuffer<float> c,
        int m, int n, float alpha, float beta,
        bool leftSide, bool upper, CudaStream s)
    {
        var side = leftSide ? CuBlasSideMode.Left : CuBlasSideMode.Right;
        var uplo = upper ? CuBlasFillMode.Upper : CuBlasFillMode.Lower;
        int lda = leftSide ? m : n;

        nint handle = CuBlasNative.Create();
        CuBlasNative.SetStream(handle, s.Handle);
        try
        {
            CuBlasNative.Ssymm(handle, side, uplo, m, n,
                &alpha, (float*)a.Pointer, lda,
                (float*)b.Pointer, m,
                &beta, (float*)c.Pointer, m);
        }
        finally { CuBlasNative.Destroy(handle); }
    }

    private static unsafe void DsymmCore(
        DeviceBuffer<double> a, DeviceBuffer<double> b, DeviceBuffer<double> c,
        int m, int n, double alpha, double beta,
        bool leftSide, bool upper, CudaStream s)
    {
        var side = leftSide ? CuBlasSideMode.Left : CuBlasSideMode.Right;
        var uplo = upper ? CuBlasFillMode.Upper : CuBlasFillMode.Lower;
        int lda = leftSide ? m : n;

        nint handle = CuBlasNative.Create();
        CuBlasNative.SetStream(handle, s.Handle);
        try
        {
            CuBlasNative.Dsymm(handle, side, uplo, m, n,
                &alpha, (double*)a.Pointer, lda,
                (double*)b.Pointer, m,
                &beta, (double*)c.Pointer, m);
        }
        finally { CuBlasNative.Destroy(handle); }
    }
}
