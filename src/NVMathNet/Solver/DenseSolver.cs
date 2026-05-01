// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Interop;

namespace NVMathNet.Solver;

/// <summary>
/// Stateless helpers for cuSOLVER dense operations: LU, Cholesky, SVD, eigenvalue
/// decomposition, and QR factorisation. All matrices use column-major layout.
/// </summary>
public static class DenseSolver
{
    // ── LU: Solve A*X = B ────────────────────────────────────────────────────

    /// <summary>
    /// Solves A*X = B using LU factorisation (single-precision).
    /// A is overwritten with L\U factors; B is overwritten with the solution X.
    /// </summary>
    /// <param name="a">Device buffer for n×n matrix A (overwritten with LU factors).</param>
    /// <param name="b">Device buffer for n×nrhs matrix B (overwritten with solution X).</param>
    /// <param name="n">Order of the square matrix A.</param>
    /// <param name="nrhs">Number of right-hand sides (columns of B).</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task SolveAsync(
        DeviceBuffer<float> a, DeviceBuffer<float> b,
        int n, int nrhs = 1,
        CudaStream? stream = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            SolveCore(a, b, n, nrhs, s);
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
    /// Solves A*X = B using LU factorisation (double-precision).
    /// </summary>
    public static async Task SolveAsync(
        DeviceBuffer<double> a, DeviceBuffer<double> b,
        int n, int nrhs = 1,
        CudaStream? stream = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            SolveCore(a, b, n, nrhs, s);
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

    // ── Cholesky factorisation ───────────────────────────────────────────────

    /// <summary>
    /// Computes the Cholesky factorisation of a symmetric positive-definite matrix (single-precision).
    /// A is overwritten with the factor L (lower) or U (upper).
    /// </summary>
    /// <param name="a">Device buffer for n×n SPD matrix A (overwritten).</param>
    /// <param name="n">Order of the matrix.</param>
    /// <param name="upper">If true, compute upper factor; otherwise lower.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task CholeskyAsync(
        DeviceBuffer<float> a, int n,
        bool upper = false,
        CudaStream? stream = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);

        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            CholeskyCore(a, n, upper, s);
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
    /// Computes the Cholesky factorisation (double-precision).
    /// </summary>
    public static async Task CholeskyAsync(
        DeviceBuffer<double> a, int n,
        bool upper = false,
        CudaStream? stream = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);

        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            CholeskyCore(a, n, upper, s);
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

    // ── SVD ──────────────────────────────────────────────────────────────────

    /// <summary>
    /// Computes the singular value decomposition A = U*diag(S)*VT (single-precision).
    /// </summary>
    /// <param name="a">Device buffer for m×n matrix A (overwritten).</param>
    /// <param name="s">Device buffer for min(m,n) singular values.</param>
    /// <param name="u">Device buffer for m×m unitary matrix U.</param>
    /// <param name="vt">Device buffer for n×n unitary matrix VT.</param>
    /// <param name="m">Rows of A.</param>
    /// <param name="n">Columns of A.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task SvdAsync(
        DeviceBuffer<float> a, DeviceBuffer<float> s,
        DeviceBuffer<float> u, DeviceBuffer<float> vt,
        int m, int n,
        CudaStream? stream = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(s);
        ArgumentNullException.ThrowIfNull(u);
        ArgumentNullException.ThrowIfNull(vt);

        bool ownStream = stream is null;
        CudaStream s2 = stream ?? new CudaStream();
        try
        {
            SvdCore(a, s, u, vt, m, n, s2);
            await s2.SynchronizeAsync(ct).ConfigureAwait(false);
        }
        finally
        {
            if (ownStream)
            {
                s2.Dispose();
            }
        }
    }

    /// <summary>
    /// Computes the singular value decomposition A = U*diag(S)*VT (double-precision).
    /// </summary>
    public static async Task SvdAsync(
        DeviceBuffer<double> a, DeviceBuffer<double> s,
        DeviceBuffer<double> u, DeviceBuffer<double> vt,
        int m, int n,
        CudaStream? stream = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(s);
        ArgumentNullException.ThrowIfNull(u);
        ArgumentNullException.ThrowIfNull(vt);

        bool ownStream = stream is null;
        CudaStream s2 = stream ?? new CudaStream();
        try
        {
            SvdCore(a, s, u, vt, m, n, s2);
            await s2.SynchronizeAsync(ct).ConfigureAwait(false);
        }
        finally
        {
            if (ownStream)
            {
                s2.Dispose();
            }
        }
    }

    // ── Eigenvalue decomposition ─────────────────────────────────────────────

    /// <summary>
    /// Computes eigenvalues (and optionally eigenvectors) of a symmetric matrix (single-precision).
    /// A is overwritten with eigenvectors (if requested); W receives eigenvalues in ascending order.
    /// </summary>
    /// <param name="a">Device buffer for n×n symmetric matrix A (overwritten with eigenvectors if computeVectors=true).</param>
    /// <param name="w">Device buffer for n eigenvalues.</param>
    /// <param name="n">Order of the matrix.</param>
    /// <param name="computeVectors">If true, compute eigenvectors in A.</param>
    /// <param name="upper">If true, read upper triangle; otherwise lower.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task EigAsync(
        DeviceBuffer<float> a, DeviceBuffer<float> w,
        int n, bool computeVectors = true, bool upper = false,
        CudaStream? stream = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(w);

        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            EigCore(a, w, n, computeVectors, upper, s);
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
    /// Computes eigenvalues (and optionally eigenvectors) of a symmetric matrix (double-precision).
    /// </summary>
    public static async Task EigAsync(
        DeviceBuffer<double> a, DeviceBuffer<double> w,
        int n, bool computeVectors = true, bool upper = false,
        CudaStream? stream = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(w);

        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            EigCore(a, w, n, computeVectors, upper, s);
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

    // ── Core implementations ─────────────────────────────────────────────────

    private static unsafe void SolveCore(DeviceBuffer<float> a, DeviceBuffer<float> b, int n, int nrhs, CudaStream s)
    {
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            int lwork = CuSolverNative.SgetrfBufferSize(handle, n, n, (float*)a.Pointer, n);
            using var workspace = new DeviceBuffer<float>(lwork);
            using var devIpiv = new DeviceBuffer<int>(n);
            using var devInfo = new DeviceBuffer<int>(1);

            CuSolverNative.Sgetrf(handle, n, n, (float*)a.Pointer, n,
                (float*)workspace.Pointer, (int*)devIpiv.Pointer, (int*)devInfo.Pointer);
            CuSolverNative.Sgetrs(handle, 0 /* CUBLAS_OP_N */, n, nrhs,
                (float*)a.Pointer, n, (int*)devIpiv.Pointer, (float*)b.Pointer, n, (int*)devInfo.Pointer);
        }
        finally { CuSolverNative.Destroy(handle); }
    }

    private static unsafe void SolveCore(DeviceBuffer<double> a, DeviceBuffer<double> b, int n, int nrhs, CudaStream s)
    {
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            int lwork = CuSolverNative.DgetrfBufferSize(handle, n, n, (double*)a.Pointer, n);
            using var workspace = new DeviceBuffer<double>(lwork);
            using var devIpiv = new DeviceBuffer<int>(n);
            using var devInfo = new DeviceBuffer<int>(1);

            CuSolverNative.Dgetrf(handle, n, n, (double*)a.Pointer, n,
                (double*)workspace.Pointer, (int*)devIpiv.Pointer, (int*)devInfo.Pointer);
            CuSolverNative.Dgetrs(handle, 0, n, nrhs,
                (double*)a.Pointer, n, (int*)devIpiv.Pointer, (double*)b.Pointer, n, (int*)devInfo.Pointer);
        }
        finally { CuSolverNative.Destroy(handle); }
    }

    private static unsafe void CholeskyCore(DeviceBuffer<float> a, int n, bool upper, CudaStream s)
    {
        var uplo = upper ? CuBlasFillMode.Upper : CuBlasFillMode.Lower;
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            int lwork = CuSolverNative.SpotrfBufferSize(handle, uplo, n, (float*)a.Pointer, n);
            using var workspace = new DeviceBuffer<float>(lwork);
            using var devInfo = new DeviceBuffer<int>(1);

            CuSolverNative.Spotrf(handle, uplo, n, (float*)a.Pointer, n,
                (float*)workspace.Pointer, lwork, (int*)devInfo.Pointer);
        }
        finally { CuSolverNative.Destroy(handle); }
    }

    private static unsafe void CholeskyCore(DeviceBuffer<double> a, int n, bool upper, CudaStream s)
    {
        var uplo = upper ? CuBlasFillMode.Upper : CuBlasFillMode.Lower;
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            int lwork = CuSolverNative.DpotrfBufferSize(handle, uplo, n, (double*)a.Pointer, n);
            using var workspace = new DeviceBuffer<double>(lwork);
            using var devInfo = new DeviceBuffer<int>(1);

            CuSolverNative.Dpotrf(handle, uplo, n, (double*)a.Pointer, n,
                (double*)workspace.Pointer, lwork, (int*)devInfo.Pointer);
        }
        finally { CuSolverNative.Destroy(handle); }
    }

    private static unsafe void SvdCore(
        DeviceBuffer<float> a, DeviceBuffer<float> sv,
        DeviceBuffer<float> u, DeviceBuffer<float> vt,
        int m, int n, CudaStream s)
    {
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            int lwork = CuSolverNative.SgesvdBufferSize(handle, m, n);
            using var workspace = new DeviceBuffer<float>(lwork);
            using var devInfo = new DeviceBuffer<int>(1);

            CuSolverNative.Sgesvd(handle,
                (byte)'A', (byte)'A', m, n,
                (float*)a.Pointer, m,
                (float*)sv.Pointer,
                (float*)u.Pointer, m,
                (float*)vt.Pointer, n,
                (float*)workspace.Pointer, lwork,
                null, (int*)devInfo.Pointer);
        }
        finally { CuSolverNative.Destroy(handle); }
    }

    private static unsafe void SvdCore(
        DeviceBuffer<double> a, DeviceBuffer<double> sv,
        DeviceBuffer<double> u, DeviceBuffer<double> vt,
        int m, int n, CudaStream s)
    {
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            int lwork = CuSolverNative.DgesvdBufferSize(handle, m, n);
            using var workspace = new DeviceBuffer<double>(lwork);
            using var devInfo = new DeviceBuffer<int>(1);

            CuSolverNative.Dgesvd(handle,
                (byte)'A', (byte)'A', m, n,
                (double*)a.Pointer, m,
                (double*)sv.Pointer,
                (double*)u.Pointer, m,
                (double*)vt.Pointer, n,
                (double*)workspace.Pointer, lwork,
                null, (int*)devInfo.Pointer);
        }
        finally { CuSolverNative.Destroy(handle); }
    }

    private static unsafe void EigCore(DeviceBuffer<float> a, DeviceBuffer<float> w, int n, bool computeVectors, bool upper, CudaStream s)
    {
        var jobz = computeVectors ? CuSolverEigMode.Vector : CuSolverEigMode.NoVector;
        var uplo = upper ? CuBlasFillMode.Upper : CuBlasFillMode.Lower;

        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            int lwork = CuSolverNative.SsyevdBufferSize(handle, jobz, uplo, n, (float*)a.Pointer, n, (float*)w.Pointer);
            using var workspace = new DeviceBuffer<float>(lwork);
            using var devInfo = new DeviceBuffer<int>(1);

            CuSolverNative.Ssyevd(handle, jobz, uplo, n,
                (float*)a.Pointer, n, (float*)w.Pointer,
                (float*)workspace.Pointer, lwork, (int*)devInfo.Pointer);
        }
        finally { CuSolverNative.Destroy(handle); }
    }

    private static unsafe void EigCore(DeviceBuffer<double> a, DeviceBuffer<double> w, int n, bool computeVectors, bool upper, CudaStream s)
    {
        var jobz = computeVectors ? CuSolverEigMode.Vector : CuSolverEigMode.NoVector;
        var uplo = upper ? CuBlasFillMode.Upper : CuBlasFillMode.Lower;

        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            int lwork = CuSolverNative.DsyevdBufferSize(handle, jobz, uplo, n, (double*)a.Pointer, n, (double*)w.Pointer);
            using var workspace = new DeviceBuffer<double>(lwork);
            using var devInfo = new DeviceBuffer<int>(1);

            CuSolverNative.Dsyevd(handle, jobz, uplo, n,
                (double*)a.Pointer, n, (double*)w.Pointer,
                (double*)workspace.Pointer, lwork, (int*)devInfo.Pointer);
        }
        finally { CuSolverNative.Destroy(handle); }
    }
}
