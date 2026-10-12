// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Interop;

namespace NVMathNet.Solver;

// Complex (double) SVD and batched routines.
public static partial class DenseSolver
{
    // ── SVD ───────────────────────────────────────────────────────────────────

    /// <summary>
    /// Singular value decomposition of a complex matrix, <c>A = U · diag(S) · VH</c>, where
    /// <c>VH = Vᴴ</c> is the conjugate transpose of the right singular vectors. Requires <c>m ≥ n</c>.
    /// </summary>
    /// <param name="a">Device buffer for the m×n matrix A; destroyed on return.</param>
    /// <param name="s">Device buffer of <c>min(m, n)</c> real singular values in descending order (output).</param>
    /// <param name="u">Device buffer for the m×m left singular vectors (output).</param>
    /// <param name="vh">Device buffer for the n×n matrix <c>Vᴴ</c> (output).</param>
    /// <param name="m">Rows of A.</param>
    /// <param name="n">Columns of A (≤ m).</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task SvdAsync(
        DeviceBuffer<Complex> a, DeviceBuffer<double> s,
        DeviceBuffer<Complex> u, DeviceBuffer<Complex> vh,
        int m, int n,
        CudaStream? stream = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(s);
        ArgumentNullException.ThrowIfNull(u);
        ArgumentNullException.ThrowIfNull(vh);
        if (m < n)
        {
            throw new ArgumentException("The complex SVD requires m >= n; pass the conjugate transpose otherwise.");
        }

        RequireSize(a.Length, (long)m * n, nameof(a));
        RequireSize(s.Length, n, nameof(s));
        RequireSize(u.Length, (long)m * m, nameof(u));
        RequireSize(vh.Length, (long)n * n, nameof(vh));

        await RunAsync(stream, ct, st => SvdComplexCore(a, s, u, vh, m, n, st)).ConfigureAwait(false);
    }

    private static unsafe void SvdComplexCore(
        DeviceBuffer<Complex> a, DeviceBuffer<double> s, DeviceBuffer<Complex> u, DeviceBuffer<Complex> vh,
        int m, int n, CudaStream stream)
    {
        Dim(m, nameof(m)); Dim(n, nameof(n));
        Need(a.Length, (long)m * n, nameof(a)); Need(s.Length, Math.Min(m, n), nameof(s));
        RequireTall(m, n);
        Need(u.Length, (long)m * m, nameof(u)); Need(vh.Length, (long)n * n, nameof(vh));
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, stream.Handle);
        try
        {
            int lwork = CuSolverNative.ZgesvdBufferSize(handle, m, n);
            using var ws = new DeviceBuffer<Complex>(Math.Max(lwork, 1));
            using var devInfo = new DeviceBuffer<int>(1);
            CuSolverNative.Zgesvd(handle, (byte)'A', (byte)'A', m, n,
                (Complex*)a.Pointer, m, (double*)s.Pointer,
                (Complex*)u.Pointer, m, (Complex*)vh.Pointer, n,
                (Complex*)ws.Pointer, lwork, null, (int*)devInfo.Pointer);
            stream.Synchronize();
            ThrowIfInfo(devInfo, "zgesvd (did not converge)");
        }
        finally
        {
            CuSolverNative.Destroy(handle);
        }
    }

    // ── Batched Cholesky ──────────────────────────────────────────────────────

    /// <summary>
    /// Cholesky factorisation of a batch of independent Hermitian positive-definite matrices, packed back-to-back.
    /// </summary>
    /// <param name="a">Device buffer of <c>batchCount·n·n</c> elements; each matrix is overwritten with its factor.</param>
    /// <param name="n">Order of each matrix.</param>
    /// <param name="batchCount">Number of matrices.</param>
    /// <param name="upper">Compute the upper factor instead of the lower.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Per-matrix status: 0 = success; a positive value means the matrix is not positive definite.</returns>
    public static async Task<int[]> CholeskyBatchedAsync(
        DeviceBuffer<Complex> a, int n, int batchCount, bool upper = false,
        CudaStream? stream = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        RequireSize(a.Length, (long)n * n * batchCount, nameof(a));

        int[] infos = [];
        await RunAsync(stream, ct, s => infos = CholeskyBatchedComplexCore(a, n, batchCount, upper, s)).ConfigureAwait(false);
        return infos;
    }

    private static unsafe int[] CholeskyBatchedComplexCore(DeviceBuffer<Complex> a, int n, int batchCount, bool upper, CudaStream s)
    {
        Dim(n, nameof(n)); Dim(batchCount, nameof(batchCount));
        Need(a.Length, (long)n * n * batchCount, nameof(a));
        var uplo = upper ? CuBlasFillMode.Upper : CuBlasFillMode.Lower;
        using var ptrs = MakePointerArray(a, (long)n * n, batchCount);
        using var infos = new DeviceBuffer<int>(batchCount);
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            CuSolverNative.ZpotrfBatched(handle, uplo, n, (Complex**)ptrs.Pointer, n, (int*)infos.Pointer, batchCount);
            s.Synchronize();
        }
        finally
        {
            CuSolverNative.Destroy(handle);
        }

        int[] result = infos.ToArray();
        if (result.Any(i => i < 0))
        {
            throw new InvalidOperationException("potrfBatched: invalid argument.");
        }

        return result;
    }

    // ── Batched LU ────────────────────────────────────────────────────────────

    /// <summary>
    /// LU factorisation with partial pivoting of a batch of independent complex n×n matrices, in place.
    /// </summary>
    /// <param name="a">Device buffer of <c>batchCount·n·n</c> elements; overwritten with the L\U factors.</param>
    /// <param name="ipiv">Device buffer of <c>batchCount·n</c> pivot indices (output).</param>
    /// <param name="n">Order of each matrix.</param>
    /// <param name="batchCount">Number of matrices.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Per-matrix status: 0 = success; a positive value <c>i</c> means <c>U(i,i)</c> is exactly zero.</returns>
    public static async Task<int[]> LuFactorBatchedAsync(
        DeviceBuffer<Complex> a, DeviceBuffer<int> ipiv, int n, int batchCount,
        CudaStream? stream = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(ipiv);
        RequireSize(a.Length, (long)n * n * batchCount, nameof(a));
        RequireSize(ipiv.Length, (long)n * batchCount, nameof(ipiv));

        int[] infos = [];
        await RunAsync(stream, ct, s => infos = LuFactorBatchedComplexCore(a, ipiv, n, batchCount, s)).ConfigureAwait(false);
        return infos;
    }

    private static unsafe int[] LuFactorBatchedComplexCore(
        DeviceBuffer<Complex> a, DeviceBuffer<int> ipiv, int n, int batchCount, CudaStream s)
    {
        Dim(n, nameof(n)); Dim(batchCount, nameof(batchCount));
        Need(a.Length, (long)n * n * batchCount, nameof(a)); Need(ipiv.Length, (long)n * batchCount, nameof(ipiv));
        using var ptrs = MakePointerArray(a, (long)n * n, batchCount);
        using var infos = new DeviceBuffer<int>(batchCount);
        nint handle = CuBlasNative.Create();
        CuBlasNative.SetStream(handle, s.Handle);
        try
        {
            CuBlasNative.ZgetrfBatched(handle, n, (Complex**)ptrs.Pointer, n, (int*)ipiv.Pointer, (int*)infos.Pointer, batchCount);
            s.Synchronize();
        }
        finally
        {
            CuBlasNative.Destroy(handle);
        }

        int[] result = infos.ToArray();
        if (result.Any(i => i < 0))
        {
            throw new InvalidOperationException("cublas getrfBatched: illegal argument.");
        }

        return result;
    }

    /// <summary>
    /// Solves <c>op(A_i)·X_i = B_i</c> for every matrix of a complex batch, where <c>op</c> is the identity or, with
    /// <paramref name="transpose"/>, the plain transpose. Use the overload with a <c>conjugate</c> argument for the
    /// conjugate transpose.
    /// </summary>
    /// <param name="a">Device buffer holding the batch of L\U factors.</param>
    /// <param name="ipiv">Pivot indices from the factorisation.</param>
    /// <param name="b">Device buffer of <c>batchCount·n·nrhs</c> right-hand sides; overwritten with the solutions.</param>
    /// <param name="n">Order of each matrix.</param>
    /// <param name="nrhs">Right-hand sides per matrix.</param>
    /// <param name="batchCount">Number of matrices.</param>
    /// <param name="transpose">Solve with the transpose of each A.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static Task LuSolveBatchedAsync(
        DeviceBuffer<Complex> a, DeviceBuffer<int> ipiv, DeviceBuffer<Complex> b, int n, int nrhs, int batchCount,
        bool transpose = false, CudaStream? stream = null, CancellationToken ct = default)
        => LuSolveBatchedAsync(a, ipiv, b, n, nrhs, batchCount, transpose, false, stream, ct);

    /// <summary>
    /// Solves <c>op(A_i)·X_i = B_i</c> for every matrix of a complex batch from the factors produced by
    /// <see cref="LuFactorBatchedAsync(DeviceBuffer{Complex}, DeviceBuffer{int}, int, int, CudaStream, CancellationToken)"/>.
    /// </summary>
    /// <param name="a">Device buffer holding the batch of L\U factors.</param>
    /// <param name="ipiv">Pivot indices from the factorisation.</param>
    /// <param name="b">Device buffer of <c>batchCount·n·nrhs</c> right-hand sides; overwritten with the solutions.</param>
    /// <param name="n">Order of each matrix.</param>
    /// <param name="nrhs">Right-hand sides per matrix.</param>
    /// <param name="batchCount">Number of matrices.</param>
    /// <param name="transpose">Solve with the transpose of each A.</param>
    /// <param name="conjugate">With <paramref name="transpose"/>, solve with the conjugate transpose.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task LuSolveBatchedAsync(
        DeviceBuffer<Complex> a, DeviceBuffer<int> ipiv, DeviceBuffer<Complex> b, int n, int nrhs, int batchCount,
        bool transpose, bool conjugate,
        CudaStream? stream = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(ipiv);
        ArgumentNullException.ThrowIfNull(b);
        RequireSize(a.Length, (long)n * n * batchCount, nameof(a));
        RequireSize(ipiv.Length, (long)n * batchCount, nameof(ipiv));
        RequireSize(b.Length, (long)n * nrhs * batchCount, nameof(b));
        var trans = !transpose ? CuBlasOperation.None : conjugate ? CuBlasOperation.ConjTranspose : CuBlasOperation.Transpose;

        await RunAsync(stream, ct, s => LuSolveBatchedComplexCore(a, ipiv, b, n, nrhs, batchCount, trans, s)).ConfigureAwait(false);
    }

    private static unsafe void LuSolveBatchedComplexCore(
        DeviceBuffer<Complex> a, DeviceBuffer<int> ipiv, DeviceBuffer<Complex> b,
        int n, int nrhs, int batchCount, CuBlasOperation trans, CudaStream s)
    {
        Dim(n, nameof(n)); Dim(nrhs, nameof(nrhs)); Dim(batchCount, nameof(batchCount));
        Need(a.Length, (long)n * n * batchCount, nameof(a)); Need(ipiv.Length, (long)n * batchCount, nameof(ipiv)); Need(b.Length, (long)n * nrhs * batchCount, nameof(b));
        using var aPtrs = MakePointerArray(a, (long)n * n, batchCount);
        using var bPtrs = MakePointerArray(b, (long)n * nrhs, batchCount);
        int info = 0;
        nint handle = CuBlasNative.Create();
        CuBlasNative.SetStream(handle, s.Handle);
        try
        {
            CuBlasNative.ZgetrsBatched(handle, trans, n, nrhs, (Complex**)aPtrs.Pointer, n, (int*)ipiv.Pointer, (Complex**)bPtrs.Pointer, n, &info, batchCount);
            s.Synchronize();
        }
        finally
        {
            CuBlasNative.Destroy(handle);
        }

        if (info < 0)
        {
            throw new InvalidOperationException($"cublas getrsBatched: argument {-info} had an illegal value.");
        }
    }

    // ── Batched Jacobi SVD ────────────────────────────────────────────────────

    /// <summary>
    /// Singular value decomposition <c>A_i = U_i · diag(S_i) · V_iᴴ</c> of a batch of small complex matrices
    /// (<c>m</c> and <c>n</c> at most 32) using the Jacobi method; matrices are packed back-to-back and column-major.
    /// </summary>
    /// <param name="a">Device buffer of <c>batchCount·m·n</c> elements; destroyed on return.</param>
    /// <param name="s">Device buffer of <c>batchCount·min(m,n)</c> real singular values, descending (output).</param>
    /// <param name="u">Device buffer of <c>batchCount·m·m</c> left vectors; may be <c>null</c> when <paramref name="computeVectors"/> is <c>false</c> (a scratch buffer is used).</param>
    /// <param name="v">Device buffer of <c>batchCount·n·n</c> right vectors — <b>V, not Vᴴ</b>; may be <c>null</c> when <paramref name="computeVectors"/> is <c>false</c>.</param>
    /// <param name="m">Rows of each matrix.</param>
    /// <param name="n">Columns of each matrix.</param>
    /// <param name="batchCount">Number of matrices.</param>
    /// <param name="computeVectors">Whether to compute U and V.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Per-matrix status: 0 = converged; a positive value means the matrix did not converge.</returns>
    public static async Task<int[]> SvdBatchedAsync(
        DeviceBuffer<Complex> a, DeviceBuffer<double> s, DeviceBuffer<Complex>? u, DeviceBuffer<Complex>? v,
        int m, int n, int batchCount, bool computeVectors = true,
        CudaStream? stream = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(s);
        if (m < 1 || n < 1 || m > 32 || n > 32)
        {
            throw new ArgumentOutOfRangeException(nameof(m), "Batched Jacobi SVD supports 1 <= m, n <= 32.");
        }

        RequireSize(a.Length, (long)m * n * batchCount, nameof(a));
        RequireSize(s.Length, (long)Math.Min(m, n) * batchCount, nameof(s));
        if (computeVectors)
        {
            ArgumentNullException.ThrowIfNull(u);
            ArgumentNullException.ThrowIfNull(v);
            RequireSize(u.Length, (long)m * m * batchCount, nameof(u));
            RequireSize(v.Length, (long)n * n * batchCount, nameof(v));
        }

        int[] infos = [];
        await RunAsync(stream, ct, st => infos = SvdBatchedComplexCore(a, s, u, v, m, n, batchCount, computeVectors, st)).ConfigureAwait(false);
        return infos;
    }

    private static unsafe int[] SvdBatchedComplexCore(
        DeviceBuffer<Complex> a, DeviceBuffer<double> s, DeviceBuffer<Complex>? u, DeviceBuffer<Complex>? v,
        int m, int n, int batchCount, bool computeVectors, CudaStream stream)
    {
        Dim(m, nameof(m)); Dim(n, nameof(n)); Dim(batchCount, nameof(batchCount));
        Need(a.Length, (long)m * n * batchCount, nameof(a)); Need(s.Length, (long)Math.Min(m, n) * batchCount, nameof(s));
        if (u is not null) { Need(u.Length, (long)m * m * batchCount, nameof(u)); }
        if (v is not null) { Need(v.Length, (long)n * n * batchCount, nameof(v)); }
        var jobz = computeVectors ? CuSolverEigMode.Vector : CuSolverEigMode.NoVector;
        // cuSOLVER dereferences U and V even for NOVECTOR, so supply scratch buffers when the caller has none.
        using var scratchU = u is null ? new DeviceBuffer<Complex>((long)m * m * batchCount) : null;
        using var scratchV = v is null ? new DeviceBuffer<Complex>((long)n * n * batchCount) : null;
        u ??= scratchU;
        v ??= scratchV;
        using var infos = new DeviceBuffer<int>(batchCount);
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, stream.Handle);
        nint gesvdjParams = CuSolverNative.CreateGesvdjInfo();
        try
        {
            Complex* pU = (Complex*)u!.Pointer;
            Complex* pV = (Complex*)v!.Pointer;
            int lwork = CuSolverNative.ZgesvdjBatchedBufferSize(handle, jobz, m, n, (Complex*)a.Pointer, m, (double*)s.Pointer, pU, m, pV, n, gesvdjParams, batchCount);
            using var ws = new DeviceBuffer<Complex>(Math.Max(lwork, 1));
            CuSolverNative.ZgesvdjBatched(handle, jobz, m, n, (Complex*)a.Pointer, m, (double*)s.Pointer, pU, m, pV, n, (Complex*)ws.Pointer, lwork, (int*)infos.Pointer, gesvdjParams, batchCount);
            stream.Synchronize();
        }
        finally
        {
            CuSolverNative.DestroyGesvdjInfo(gesvdjParams);
            CuSolverNative.Destroy(handle);
        }

        return infos.ToArray();
    }
}
