// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Interop;

namespace NVMathNet.LinAlg;

/// <summary>
/// Stateless helpers for cuBLAS diagonal matrix multiply (DGMM):
/// C = diag(X) * A (side=Left) or C = A * diag(X) (side=Right)
/// in column-major layout.
/// </summary>
public static class DiagonalMultiply
{
    /// <summary>
    /// Diagonal matrix multiply (single-precision, async).
    /// C = diag(x) * A when <paramref name="leftSide"/> is true,
    /// or C = A * diag(x) when false.
    /// </summary>
    /// <param name="a">Device buffer for dense matrix A (m×n), column-major.</param>
    /// <param name="x">Device buffer for diagonal vector x (length m if leftSide, n if rightSide).</param>
    /// <param name="c">Device buffer for output matrix C (m×n), column-major.</param>
    /// <param name="m">Number of rows.</param>
    /// <param name="n">Number of columns.</param>
    /// <param name="leftSide">If true, left-multiply by diag(x); if false, right-multiply.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task SdgmmAsync(
        DeviceBuffer<float> a, DeviceBuffer<float> x, DeviceBuffer<float> c,
        int m, int n,
        bool leftSide = true,
        CudaStream? stream = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(c);

        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            SdgmmCore(a, x, c, m, n, leftSide, s);
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
    /// Diagonal matrix multiply (double-precision, async).
    /// C = diag(x) * A when <paramref name="leftSide"/> is true,
    /// or C = A * diag(x) when false.
    /// </summary>
    public static async Task DdgmmAsync(
        DeviceBuffer<double> a, DeviceBuffer<double> x, DeviceBuffer<double> c,
        int m, int n,
        bool leftSide = true,
        CudaStream? stream = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(c);

        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            DdgmmCore(a, x, c, m, n, leftSide, s);
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

    private static unsafe void SdgmmCore(
        DeviceBuffer<float> a, DeviceBuffer<float> x, DeviceBuffer<float> c,
        int m, int n, bool leftSide, CudaStream s)
    {
        var mode = leftSide ? CuBlasSideMode.Left : CuBlasSideMode.Right;

        nint handle = CuBlasNative.Create();
        CuBlasNative.SetStream(handle, s.Handle);
        try
        {
            CuBlasNative.Sdgmm(handle, mode, m, n,
                (float*)a.Pointer, m,
                (float*)x.Pointer, 1,
                (float*)c.Pointer, m);
        }
        finally { CuBlasNative.Destroy(handle); }
    }

    private static unsafe void DdgmmCore(
        DeviceBuffer<double> a, DeviceBuffer<double> x, DeviceBuffer<double> c,
        int m, int n, bool leftSide, CudaStream s)
    {
        var mode = leftSide ? CuBlasSideMode.Left : CuBlasSideMode.Right;

        nint handle = CuBlasNative.Create();
        CuBlasNative.SetStream(handle, s.Handle);
        try
        {
            CuBlasNative.Ddgmm(handle, mode, m, n,
                (double*)a.Pointer, m,
                (double*)x.Pointer, 1,
                (double*)c.Pointer, m);
        }
        finally { CuBlasNative.Destroy(handle); }
    }
}
