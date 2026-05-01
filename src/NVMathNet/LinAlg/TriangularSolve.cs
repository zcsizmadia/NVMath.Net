// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Interop;

namespace NVMathNet.LinAlg;

/// <summary>
/// Stateless helpers for cuBLAS triangular solve (TRSM):
/// B = alpha * inv(op(A)) * B (side=Left) or B = alpha * B * inv(op(A)) (side=Right)
/// where A is triangular, in column-major layout.
/// </summary>
public static class TriangularSolve
{
    /// <summary>
    /// Solves a triangular system asynchronously (single-precision).
    /// B is overwritten with the solution.
    /// </summary>
    /// <param name="a">Device buffer for triangular matrix A (m×m for Left, n×n for Right).</param>
    /// <param name="b">Device buffer for matrix B (m×n), overwritten with the solution.</param>
    /// <param name="m">Rows of B.</param>
    /// <param name="n">Columns of B.</param>
    /// <param name="alpha">Scalar multiplier. Default: 1.</param>
    /// <param name="leftSide">If true, solve from the left (A*X = alpha*B); otherwise right.</param>
    /// <param name="upper">If true, A is upper triangular; otherwise lower.</param>
    /// <param name="transpose">If true, use A^T instead of A.</param>
    /// <param name="unitDiag">If true, diagonal of A is assumed to be all ones.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task StrsmAsync(
        DeviceBuffer<float> a, DeviceBuffer<float> b,
        int m, int n,
        float alpha = 1.0f,
        bool leftSide = true, bool upper = false,
        bool transpose = false, bool unitDiag = false,
        CudaStream? stream = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            StrsmCore(a, b, m, n, alpha, leftSide, upper, transpose, unitDiag, s);
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
    /// Solves a triangular system asynchronously (double-precision).
    /// </summary>
    public static async Task DtrsmAsync(
        DeviceBuffer<double> a, DeviceBuffer<double> b,
        int m, int n,
        double alpha = 1.0,
        bool leftSide = true, bool upper = false,
        bool transpose = false, bool unitDiag = false,
        CudaStream? stream = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            DtrsmCore(a, b, m, n, alpha, leftSide, upper, transpose, unitDiag, s);
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

    private static unsafe void StrsmCore(
        DeviceBuffer<float> a, DeviceBuffer<float> b,
        int m, int n, float alpha,
        bool leftSide, bool upper, bool transpose, bool unitDiag,
        CudaStream s)
    {
        var side = leftSide ? CuBlasSideMode.Left : CuBlasSideMode.Right;
        var uplo = upper ? CuBlasFillMode.Upper : CuBlasFillMode.Lower;
        var trans = transpose ? CuBlasOperation.Transpose : CuBlasOperation.None;
        var diag = unitDiag ? CuBlasDiagType.Unit : CuBlasDiagType.NonUnit;
        int lda = leftSide ? m : n;

        nint handle = CuBlasNative.Create();
        CuBlasNative.SetStream(handle, s.Handle);
        try
        {
            CuBlasNative.Strsm(handle, side, uplo, trans, diag, m, n,
                &alpha, (float*)a.Pointer, lda, (float*)b.Pointer, m);
        }
        finally { CuBlasNative.Destroy(handle); }
    }

    private static unsafe void DtrsmCore(
        DeviceBuffer<double> a, DeviceBuffer<double> b,
        int m, int n, double alpha,
        bool leftSide, bool upper, bool transpose, bool unitDiag,
        CudaStream s)
    {
        var side = leftSide ? CuBlasSideMode.Left : CuBlasSideMode.Right;
        var uplo = upper ? CuBlasFillMode.Upper : CuBlasFillMode.Lower;
        var trans = transpose ? CuBlasOperation.Transpose : CuBlasOperation.None;
        var diag = unitDiag ? CuBlasDiagType.Unit : CuBlasDiagType.NonUnit;
        int lda = leftSide ? m : n;

        nint handle = CuBlasNative.Create();
        CuBlasNative.SetStream(handle, s.Handle);
        try
        {
            CuBlasNative.Dtrsm(handle, side, uplo, trans, diag, m, n,
                &alpha, (double*)a.Pointer, lda, (double*)b.Pointer, m);
        }
        finally { CuBlasNative.Destroy(handle); }
    }
}
