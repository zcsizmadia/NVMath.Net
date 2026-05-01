// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using System.Numerics;
using NVMathNet.Interop;

namespace NVMathNet.LinAlg;

/// <summary>
/// Stateless helpers for classic cuBLAS GEMM operations.
/// Supports single-precision (<see cref="float"/>), double-precision (<see cref="double"/>),
/// and half-precision (<see cref="Half"/>) element types.
/// <para>
/// Computes C = alpha * op(A) * op(B) + beta * C where all matrices are in
/// column-major layout on the GPU device.
/// </para>
/// <para>
/// For mixed-precision or epilog support, see <see cref="Matmul"/> (cuBLASLt).
/// </para>
/// </summary>
public static class Gemm
{
    // ── Async overloads ───────────────────────────────────────────────────────

    /// <summary>
    /// Asynchronously computes C = alpha * op(A) * op(B) + beta * C using single-precision.
    /// </summary>
    /// <param name="a">Device buffer for matrix A (m × k or k × m when transposed).</param>
    /// <param name="b">Device buffer for matrix B (k × n or n × k when transposed).</param>
    /// <param name="c">Device buffer for matrix C (m × n); updated in place.</param>
    /// <param name="m">Rows of op(A) and C.</param>
    /// <param name="n">Columns of op(B) and C.</param>
    /// <param name="k">Inner dimension (columns of op(A) / rows of op(B)).</param>
    /// <param name="alpha">Scalar multiplier for A*B. Default: 1.</param>
    /// <param name="beta">Scalar multiplier for existing values in C. Default: 0.</param>
    /// <param name="transposeA">If <c>true</c>, use A<sup>T</sup>.</param>
    /// <param name="transposeB">If <c>true</c>, use B<sup>T</sup>.</param>
    /// <param name="stream">Optional CUDA stream; <c>null</c> creates a private stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task SgemmAsync(
        DeviceBuffer<float> a, DeviceBuffer<float> b, DeviceBuffer<float> c,
        int m, int n, int k,
        float alpha = 1.0f, float beta = 0.0f,
        bool transposeA = false, bool transposeB = false,
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
            SgemmCore(a, b, c, m, n, k, alpha, beta, transposeA, transposeB, s);
            await s.SynchronizeAsync(ct).ConfigureAwait(false);
        }
        finally
        {
            if (ownStream) s.Dispose();
        }
    }

    /// <summary>
    /// Asynchronously computes C = alpha * op(A) * op(B) + beta * C using double-precision.
    /// </summary>
    /// <inheritdoc cref="SgemmAsync"/>
    public static async Task DgemmAsync(
        DeviceBuffer<double> a, DeviceBuffer<double> b, DeviceBuffer<double> c,
        int m, int n, int k,
        double alpha = 1.0, double beta = 0.0,
        bool transposeA = false, bool transposeB = false,
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
            DgemmCore(a, b, c, m, n, k, alpha, beta, transposeA, transposeB, s);
            await s.SynchronizeAsync(ct).ConfigureAwait(false);
        }
        finally
        {
            if (ownStream) s.Dispose();
        }
    }

    /// <summary>
    /// Asynchronously computes C = alpha * op(A) * op(B) + beta * C using half-precision.
    /// </summary>
    /// <inheritdoc cref="SgemmAsync(DeviceBuffer{float}, DeviceBuffer{float}, DeviceBuffer{float}, int, int, int, float, float, bool, bool, CudaStream?, CancellationToken)"/>
    public static async Task HgemmAsync(
        DeviceBuffer<Half> a, DeviceBuffer<Half> b, DeviceBuffer<Half> c,
        int m, int n, int k,
        Half alpha, Half beta,
        bool transposeA = false, bool transposeB = false,
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
            HgemmCore(a, b, c, m, n, k, alpha, beta, transposeA, transposeB, s);
            await s.SynchronizeAsync(ct).ConfigureAwait(false);
        }
        finally
        {
            if (ownStream) s.Dispose();
        }
    }

    // ── Synchronous overloads ─────────────────────────────────────────────────

    /// <summary>Synchronously computes C = alpha * op(A) * op(B) + beta * C using single-precision.</summary>
    /// <inheritdoc cref="SgemmAsync(DeviceBuffer{float}, DeviceBuffer{float}, DeviceBuffer{float}, int, int, int, float, float, bool, bool, CudaStream?, CancellationToken)"/>
    public static void Sgemm(
        DeviceBuffer<float> a, DeviceBuffer<float> b, DeviceBuffer<float> c,
        int m, int n, int k,
        float alpha = 1.0f, float beta = 0.0f,
        bool transposeA = false, bool transposeB = false,
        CudaStream? stream = null)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);

        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            SgemmCore(a, b, c, m, n, k, alpha, beta, transposeA, transposeB, s);
            s.Synchronize();
        }
        finally
        {
            if (ownStream) s.Dispose();
        }
    }

    /// <summary>Synchronously computes C = alpha * op(A) * op(B) + beta * C using double-precision.</summary>
    /// <inheritdoc cref="DgemmAsync"/>
    public static void Dgemm(
        DeviceBuffer<double> a, DeviceBuffer<double> b, DeviceBuffer<double> c,
        int m, int n, int k,
        double alpha = 1.0, double beta = 0.0,
        bool transposeA = false, bool transposeB = false,
        CudaStream? stream = null)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);

        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            DgemmCore(a, b, c, m, n, k, alpha, beta, transposeA, transposeB, s);
            s.Synchronize();
        }
        finally
        {
            if (ownStream) s.Dispose();
        }
    }

    // ── Core implementations ──────────────────────────────────────────────────

    private static unsafe void SgemmCore(
        DeviceBuffer<float> a, DeviceBuffer<float> b, DeviceBuffer<float> c,
        int m, int n, int k,
        float alpha, float beta,
        bool transposeA, bool transposeB,
        CudaStream s)
    {
        var opA = transposeA ? CuBlasOperation.Transpose : CuBlasOperation.None;
        var opB = transposeB ? CuBlasOperation.Transpose : CuBlasOperation.None;
        int lda = transposeA ? k : m;
        int ldb = transposeB ? n : k;

        nint handle = CuBlasNative.Create();
        CuBlasNative.SetStream(handle, s.Handle);
        try
        {
            CuBlasNative.Sgemm(handle, opA, opB, m, n, k,
                &alpha, (float*)a.Pointer, lda,
                (float*)b.Pointer, ldb,
                &beta,  (float*)c.Pointer, m);
        }
        finally { CuBlasNative.Destroy(handle); }
    }

    private static unsafe void DgemmCore(
        DeviceBuffer<double> a, DeviceBuffer<double> b, DeviceBuffer<double> c,
        int m, int n, int k,
        double alpha, double beta,
        bool transposeA, bool transposeB,
        CudaStream s)
    {
        var opA = transposeA ? CuBlasOperation.Transpose : CuBlasOperation.None;
        var opB = transposeB ? CuBlasOperation.Transpose : CuBlasOperation.None;
        int lda = transposeA ? k : m;
        int ldb = transposeB ? n : k;

        nint handle = CuBlasNative.Create();
        CuBlasNative.SetStream(handle, s.Handle);
        try
        {
            CuBlasNative.Dgemm(handle, opA, opB, m, n, k,
                &alpha, (double*)a.Pointer, lda,
                (double*)b.Pointer, ldb,
                &beta,  (double*)c.Pointer, m);
        }
        finally { CuBlasNative.Destroy(handle); }
    }

    private static unsafe void HgemmCore(
        DeviceBuffer<Half> a, DeviceBuffer<Half> b, DeviceBuffer<Half> c,
        int m, int n, int k,
        Half alpha, Half beta,
        bool transposeA, bool transposeB,
        CudaStream s)
    {
        var opA = transposeA ? CuBlasOperation.Transpose : CuBlasOperation.None;
        var opB = transposeB ? CuBlasOperation.Transpose : CuBlasOperation.None;
        int lda = transposeA ? k : m;
        int ldb = transposeB ? n : k;

        nint handle = CuBlasNative.Create();
        CuBlasNative.SetStream(handle, s.Handle);
        try
        {
            CuBlasNative.Hgemm(handle, opA, opB, m, n, k,
                &alpha, (Half*)a.Pointer, lda,
                (Half*)b.Pointer, ldb,
                &beta,  (Half*)c.Pointer, m);
        }
        finally { CuBlasNative.Destroy(handle); }
    }
}
