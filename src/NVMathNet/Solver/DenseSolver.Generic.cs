// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Interop;

namespace NVMathNet.Solver;

// LU factor/solve, QR, least squares and batched Cholesky. Element type must be float or double.
// All matrices are column-major and operated on in place.
public static partial class DenseSolver
{
    // ── LU ────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Computes the LU factorisation with partial pivoting <c>P·A = L·U</c> in place.
    /// </summary>
    /// <param name="a">Device buffer for the m×n matrix A; overwritten with the L\U factors.</param>
    /// <param name="ipiv">Device buffer of <c>min(m, n)</c> pivot indices (output).</param>
    /// <param name="m">Number of rows.</param>
    /// <param name="n">Number of columns.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>
    /// 0 on success; a positive value <c>i</c> means <c>U(i,i)</c> is exactly zero (A is singular).
    /// </returns>
    public static async Task<int> LuFactorAsync<T>(
        DeviceBuffer<T> a, DeviceBuffer<int> ipiv, int m, int n,
        CudaStream? stream = null, CancellationToken ct = default) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(ipiv);
        RequireFloatOrDouble<T>();

        int info = 0;
        await RunAsync(stream, ct, s => info = LuFactorCore(a, ipiv, m, n, s)).ConfigureAwait(false);
        return info;
    }

    private static unsafe int LuFactorCore<T>(DeviceBuffer<T> a, DeviceBuffer<int> ipiv, int m, int n, CudaStream s)
        where T : unmanaged, INumberBase<T>
    {
        Dim(m, nameof(m)); Dim(n, nameof(n));
        Need(a.Length, (long)m * n, nameof(a)); Need(ipiv.Length, Math.Min(m, n), nameof(ipiv));
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            using var devInfo = new DeviceBuffer<int>(1);
            if (typeof(T) == typeof(float))
            {
                int lwork = CuSolverNative.SgetrfBufferSize(handle, m, n, (float*)a.Pointer, m);
                using var ws = new DeviceBuffer<float>(Math.Max(lwork, 1));
                CuSolverNative.Sgetrf(handle, m, n, (float*)a.Pointer, m, (float*)ws.Pointer, (int*)ipiv.Pointer, (int*)devInfo.Pointer);
                s.Synchronize();
            }
            else
            {
                int lwork = CuSolverNative.DgetrfBufferSize(handle, m, n, (double*)a.Pointer, m);
                using var ws = new DeviceBuffer<double>(Math.Max(lwork, 1));
                CuSolverNative.Dgetrf(handle, m, n, (double*)a.Pointer, m, (double*)ws.Pointer, (int*)ipiv.Pointer, (int*)devInfo.Pointer);
                s.Synchronize();
            }

            return ReadInfo(devInfo, "getrf");
        }
        finally
        {
            CuSolverNative.Destroy(handle);
        }
    }

    /// <summary>
    /// Solves <c>op(A)·X = B</c> using a factorisation produced by <see cref="LuFactorAsync{T}"/>.
    /// </summary>
    /// <param name="a">Device buffer holding the n×n L\U factors.</param>
    /// <param name="ipiv">Pivot indices from <see cref="LuFactorAsync{T}"/>.</param>
    /// <param name="b">Device buffer for the n×nrhs right-hand sides; overwritten with the solution.</param>
    /// <param name="n">Order of A.</param>
    /// <param name="nrhs">Number of right-hand sides.</param>
    /// <param name="transpose">Solve with the transpose of A.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task LuSolveAsync<T>(
        DeviceBuffer<T> a, DeviceBuffer<int> ipiv, DeviceBuffer<T> b, int n, int nrhs = 1,
        bool transpose = false, CudaStream? stream = null, CancellationToken ct = default)
        where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(ipiv);
        ArgumentNullException.ThrowIfNull(b);
        RequireFloatOrDouble<T>();

        await RunAsync(stream, ct, s => LuSolveCore(a, ipiv, b, n, nrhs, transpose, s)).ConfigureAwait(false);
    }

    private static unsafe void LuSolveCore<T>(
        DeviceBuffer<T> a, DeviceBuffer<int> ipiv, DeviceBuffer<T> b, int n, int nrhs, bool transpose, CudaStream s)
        where T : unmanaged, INumberBase<T>
    {
        Dim(n, nameof(n)); Dim(nrhs, nameof(nrhs));
        Need(a.Length, (long)n * n, nameof(a)); Need(ipiv.Length, n, nameof(ipiv)); Need(b.Length, (long)n * nrhs, nameof(b));
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            using var devInfo = new DeviceBuffer<int>(1);
            int trans = transpose ? 1 : 0;
            if (typeof(T) == typeof(float))
            {
                CuSolverNative.Sgetrs(handle, trans, n, nrhs, (float*)a.Pointer, n, (int*)ipiv.Pointer, (float*)b.Pointer, n, (int*)devInfo.Pointer);
            }
            else
            {
                CuSolverNative.Dgetrs(handle, trans, n, nrhs, (double*)a.Pointer, n, (int*)ipiv.Pointer, (double*)b.Pointer, n, (int*)devInfo.Pointer);
            }

            s.Synchronize();
            ReadInfo(devInfo, "getrs");
        }
        finally
        {
            CuSolverNative.Destroy(handle);
        }
    }

    // ── QR ────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Computes the QR factorisation <c>A = Q·R</c> in place. On return the upper triangle of A
    /// holds R and the part below the diagonal, together with <paramref name="tau"/>,
    /// encodes the Householder reflectors of Q.
    /// </summary>
    /// <param name="a">Device buffer for the m×n matrix A.</param>
    /// <param name="tau">Device buffer of <c>min(m, n)</c> scalars (output).</param>
    /// <param name="m">Number of rows.</param>
    /// <param name="n">Number of columns.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task QrAsync<T>(
        DeviceBuffer<T> a, DeviceBuffer<T> tau, int m, int n,
        CudaStream? stream = null, CancellationToken ct = default) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(tau);
        RequireFloatOrDouble<T>();

        await RunAsync(stream, ct, s => QrCore(a, tau, m, n, s)).ConfigureAwait(false);
    }

    private static unsafe void QrCore<T>(DeviceBuffer<T> a, DeviceBuffer<T> tau, int m, int n, CudaStream s)
        where T : unmanaged, INumberBase<T>
    {
        Dim(m, nameof(m)); Dim(n, nameof(n));
        Need(a.Length, (long)m * n, nameof(a)); Need(tau.Length, Math.Min(m, n), nameof(tau));
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            using var devInfo = new DeviceBuffer<int>(1);
            if (typeof(T) == typeof(float))
            {
                int lwork = CuSolverNative.SgeqrfBufferSize(handle, m, n, (float*)a.Pointer, m);
                using var ws = new DeviceBuffer<float>(Math.Max(lwork, 1));
                CuSolverNative.Sgeqrf(handle, m, n, (float*)a.Pointer, m, (float*)tau.Pointer, (float*)ws.Pointer, lwork, (int*)devInfo.Pointer);
            }
            else
            {
                int lwork = CuSolverNative.DgeqrfBufferSize(handle, m, n, (double*)a.Pointer, m);
                using var ws = new DeviceBuffer<double>(Math.Max(lwork, 1));
                CuSolverNative.Dgeqrf(handle, m, n, (double*)a.Pointer, m, (double*)tau.Pointer, (double*)ws.Pointer, lwork, (int*)devInfo.Pointer);
            }

            s.Synchronize();
            ReadInfo(devInfo, "geqrf");
        }
        finally
        {
            CuSolverNative.Destroy(handle);
        }
    }

    /// <summary>
    /// Forms the explicit orthonormal factor Q (first <paramref name="n"/> columns, m×n) from a
    /// factorisation produced by <see cref="QrAsync{T}"/>. <paramref name="a"/> is overwritten with Q.
    /// Requires <c>m ≥ n</c>.
    /// </summary>
    /// <param name="a">Device buffer holding the output of <see cref="QrAsync{T}"/>; overwritten with Q.</param>
    /// <param name="tau">Scalars from <see cref="QrAsync{T}"/>.</param>
    /// <param name="m">Number of rows.</param>
    /// <param name="n">Number of columns of Q (≤ m).</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task QrFormQAsync<T>(
        DeviceBuffer<T> a, DeviceBuffer<T> tau, int m, int n,
        CudaStream? stream = null, CancellationToken ct = default) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(tau);
        RequireFloatOrDouble<T>();
        if (m < n)
        {
            throw new ArgumentException("QrFormQ requires m >= n.");
        }

        await RunAsync(stream, ct, s => QrFormQCore(a, tau, m, n, s)).ConfigureAwait(false);
    }

    private static unsafe void QrFormQCore<T>(DeviceBuffer<T> a, DeviceBuffer<T> tau, int m, int n, CudaStream s)
        where T : unmanaged, INumberBase<T>
    {
        Dim(m, nameof(m)); Dim(n, nameof(n));
        Need(a.Length, (long)m * n, nameof(a)); Need(tau.Length, Math.Min(m, n), nameof(tau));
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            using var devInfo = new DeviceBuffer<int>(1);
            if (typeof(T) == typeof(float))
            {
                int lwork = CuSolverNative.SorgqrBufferSize(handle, m, n, n, (float*)a.Pointer, m, (float*)tau.Pointer);
                using var ws = new DeviceBuffer<float>(Math.Max(lwork, 1));
                CuSolverNative.Sorgqr(handle, m, n, n, (float*)a.Pointer, m, (float*)tau.Pointer, (float*)ws.Pointer, lwork, (int*)devInfo.Pointer);
            }
            else
            {
                int lwork = CuSolverNative.DorgqrBufferSize(handle, m, n, n, (double*)a.Pointer, m, (double*)tau.Pointer);
                using var ws = new DeviceBuffer<double>(Math.Max(lwork, 1));
                CuSolverNative.Dorgqr(handle, m, n, n, (double*)a.Pointer, m, (double*)tau.Pointer, (double*)ws.Pointer, lwork, (int*)devInfo.Pointer);
            }

            s.Synchronize();
            ReadInfo(devInfo, "orgqr");
        }
        finally
        {
            CuSolverNative.Destroy(handle);
        }
    }

    // ── Least squares ─────────────────────────────────────────────────────────

    /// <summary>
    /// Solves the overdetermined least-squares problem <c>min ‖A·X − B‖₂</c> via QR
    /// (<c>m ≥ n</c>, A of full column rank).
    /// </summary>
    /// <param name="a">Device buffer for the m×n matrix A; overwritten with its QR factorisation.</param>
    /// <param name="b">
    /// Device buffer for the m×nrhs right-hand sides (leading dimension m). On return the first
    /// n rows of each column hold the solution; the remaining rows hold the transformed residual.
    /// </param>
    /// <param name="m">Number of rows of A.</param>
    /// <param name="n">Number of columns of A (≤ m).</param>
    /// <param name="nrhs">Number of right-hand sides.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task LeastSquaresAsync<T>(
        DeviceBuffer<T> a, DeviceBuffer<T> b, int m, int n, int nrhs = 1,
        CudaStream? stream = null, CancellationToken ct = default) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        RequireFloatOrDouble<T>();
        if (m < n)
        {
            throw new ArgumentException("LeastSquares requires m >= n (overdetermined or square).");
        }

        await RunAsync(stream, ct, s => LeastSquaresCore(a, b, m, n, nrhs, s)).ConfigureAwait(false);
    }

    private static unsafe void LeastSquaresCore<T>(
        DeviceBuffer<T> a, DeviceBuffer<T> b, int m, int n, int nrhs, CudaStream s)
        where T : unmanaged, INumberBase<T>
    {
        Dim(m, nameof(m)); Dim(n, nameof(n)); Dim(nrhs, nameof(nrhs));
        Need(a.Length, (long)m * n, nameof(a)); Need(b.Length, (long)Math.Max(m, n) * nrhs, nameof(b));
        nint solver = CuSolverNative.Create();
        nint blas = CuBlasNative.Create();
        CuSolverNative.SetStream(solver, s.Handle);
        CuBlasNative.SetStream(blas, s.Handle);
        try
        {
            using var devInfo = new DeviceBuffer<int>(1);
            using var tau = new DeviceBuffer<T>(n);
            if (typeof(T) == typeof(float))
            {
                float one = 1f;
                float* pA = (float*)a.Pointer, pB = (float*)b.Pointer, pTau = (float*)tau.Pointer;

                int lwork = CuSolverNative.SgeqrfBufferSize(solver, m, n, pA, m);
                using (var ws = new DeviceBuffer<float>(Math.Max(lwork, 1)))
                {
                    CuSolverNative.Sgeqrf(solver, m, n, pA, m, pTau, (float*)ws.Pointer, lwork, (int*)devInfo.Pointer);
                    s.Synchronize();
                    ReadInfo(devInfo, "geqrf");
                }

                // B <- Q^T B
                lwork = CuSolverNative.SormqrBufferSize(solver, CuBlasSideMode.Left, CuBlasOperation.Transpose, m, nrhs, n, pA, m, pTau, pB, m);
                using (var ws = new DeviceBuffer<float>(Math.Max(lwork, 1)))
                {
                    CuSolverNative.Sormqr(solver, CuBlasSideMode.Left, CuBlasOperation.Transpose, m, nrhs, n, pA, m, pTau, pB, m, (float*)ws.Pointer, lwork, (int*)devInfo.Pointer);
                    s.Synchronize();
                    ReadInfo(devInfo, "ormqr");
                }

                // Solve R x = (Q^T B)[0:n]
                CuBlasNative.Strsm(blas, CuBlasSideMode.Left, CuBlasFillMode.Upper, CuBlasOperation.None, CuBlasDiagType.NonUnit,
                    n, nrhs, &one, pA, m, pB, m);
            }
            else
            {
                double one = 1.0;
                double* pA = (double*)a.Pointer, pB = (double*)b.Pointer, pTau = (double*)tau.Pointer;

                int lwork = CuSolverNative.DgeqrfBufferSize(solver, m, n, pA, m);
                using (var ws = new DeviceBuffer<double>(Math.Max(lwork, 1)))
                {
                    CuSolverNative.Dgeqrf(solver, m, n, pA, m, pTau, (double*)ws.Pointer, lwork, (int*)devInfo.Pointer);
                    s.Synchronize();
                    ReadInfo(devInfo, "geqrf");
                }

                lwork = CuSolverNative.DormqrBufferSize(solver, CuBlasSideMode.Left, CuBlasOperation.Transpose, m, nrhs, n, pA, m, pTau, pB, m);
                using (var ws = new DeviceBuffer<double>(Math.Max(lwork, 1)))
                {
                    CuSolverNative.Dormqr(solver, CuBlasSideMode.Left, CuBlasOperation.Transpose, m, nrhs, n, pA, m, pTau, pB, m, (double*)ws.Pointer, lwork, (int*)devInfo.Pointer);
                    s.Synchronize();
                    ReadInfo(devInfo, "ormqr");
                }

                CuBlasNative.Dtrsm(blas, CuBlasSideMode.Left, CuBlasFillMode.Upper, CuBlasOperation.None, CuBlasDiagType.NonUnit,
                    n, nrhs, &one, pA, m, pB, m);
            }

            s.Synchronize();
        }
        finally
        {
            CuBlasNative.Destroy(blas);
            CuSolverNative.Destroy(solver);
        }
    }

    // ── Batched Cholesky ──────────────────────────────────────────────────────

    /// <summary>
    /// Cholesky factorisation of a batch of independent symmetric positive-definite matrices,
    /// stored back-to-back (matrix <c>i</c> starts at element <c>i·n·n</c>). Each matrix is
    /// overwritten with its factor.
    /// </summary>
    /// <param name="a">Device buffer of <c>batchCount·n·n</c> elements.</param>
    /// <param name="n">Order of each matrix.</param>
    /// <param name="batchCount">Number of matrices.</param>
    /// <param name="upper">Compute the upper factor instead of the lower.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Per-matrix status: 0 = success; a positive value means the matrix is not positive definite.</returns>
    public static async Task<int[]> CholeskyBatchedAsync<T>(
        DeviceBuffer<T> a, int n, int batchCount, bool upper = false,
        CudaStream? stream = null, CancellationToken ct = default) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(a);
        RequireFloatOrDouble<T>();
        if (a.Length < (long)n * n * batchCount)
        {
            throw new ArgumentException("Buffer is too small for the requested batch.", nameof(a));
        }

        int[] infos = [];
        await RunAsync(stream, ct, s => infos = CholeskyBatchedCore(a, n, batchCount, upper, s)).ConfigureAwait(false);
        return infos;
    }

    private static unsafe int[] CholeskyBatchedCore<T>(DeviceBuffer<T> a, int n, int batchCount, bool upper, CudaStream s)
        where T : unmanaged, INumberBase<T>
    {
        Dim(n, nameof(n)); Dim(batchCount, nameof(batchCount));
        Need(a.Length, (long)n * n * batchCount, nameof(a));
        var uplo = upper ? CuBlasFillMode.Upper : CuBlasFillMode.Lower;
        long stride = (long)n * n * sizeof(T);
        nint[] hostPtrs = new nint[batchCount];
        for (int i = 0; i < batchCount; i++)
        {
            hostPtrs[i] = a.PointerAsInt + (nint)(i * stride);
        }

        using var ptrs = new DeviceBuffer<nint>(batchCount);
        ptrs.CopyFrom(hostPtrs.AsSpan());
        using var infos = new DeviceBuffer<int>(batchCount);

        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            if (typeof(T) == typeof(float))
            {
                CuSolverNative.SpotrfBatched(handle, uplo, n, (float**)ptrs.Pointer, n, (int*)infos.Pointer, batchCount);
            }
            else
            {
                CuSolverNative.DpotrfBatched(handle, uplo, n, (double**)ptrs.Pointer, n, (int*)infos.Pointer, batchCount);
            }

            s.Synchronize();
        }
        finally
        {
            CuSolverNative.Destroy(handle);
        }

        int[] result = infos.ToArray();
        for (int i = 0; i < result.Length; i++)
        {
            if (result[i] < 0)
            {
                throw new InvalidOperationException($"potrfBatched: invalid argument {-result[i]}.");
            }
        }

        return result;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static void RequireFloatOrDouble<T>()
    {
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            throw new NotSupportedException($"Element type {typeof(T)} is not supported; use float or double.");
        }
    }

    /// <summary>Throws if cuSOLVER's <c>devInfo</c> is non-zero (failure or invalid argument).</summary>
    private static void ThrowIfInfo(DeviceBuffer<int> devInfo, string what)
    {
        int info = devInfo.ToArray()[0];
        if (info != 0)
        {
            throw new InvalidOperationException($"cuSOLVER {what}: info = {info}.");
        }
    }

    /// <summary>Reads the cuSOLVER <c>devInfo</c> value; throws for invalid-argument errors (info &lt; 0).</summary>
    private static int ReadInfo(DeviceBuffer<int> devInfo, string routine)
    {
        int info = devInfo.ToArray()[0];
        if (info < 0)
        {
            throw new InvalidOperationException($"cuSOLVER {routine}: argument {-info} had an illegal value.");
        }

        return info;
    }

    private static async Task RunAsync(CudaStream? stream, CancellationToken ct, Action<CudaStream> body)
    {
        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            body(s);
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
}
