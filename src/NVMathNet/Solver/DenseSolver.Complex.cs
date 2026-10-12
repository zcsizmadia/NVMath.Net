// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Interop;

namespace NVMathNet.Solver;

// Double-precision complex variants (System.Numerics.Complex == cuDoubleComplex).
// .NET has no single-precision complex type, so complex64 is not offered.
public static partial class DenseSolver
{
    // ── LU ────────────────────────────────────────────────────────────────────

    /// <inheritdoc cref="LuFactorAsync{T}(DeviceBuffer{T}, DeviceBuffer{int}, int, int, CudaStream, CancellationToken)"/>
    public static async Task<int> LuFactorAsync(
        DeviceBuffer<Complex> a, DeviceBuffer<int> ipiv, int m, int n,
        CudaStream? stream = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(ipiv);

        int info = 0;
        await RunAsync(stream, ct, s => info = LuFactorComplexCore(a, ipiv, m, n, s)).ConfigureAwait(false);
        return info;
    }

    private static unsafe int LuFactorComplexCore(DeviceBuffer<Complex> a, DeviceBuffer<int> ipiv, int m, int n, CudaStream s)
    {
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            using var devInfo = new DeviceBuffer<int>(1);
            int lwork = CuSolverNative.ZgetrfBufferSize(handle, m, n, (Complex*)a.Pointer, m);
            using var ws = new DeviceBuffer<Complex>(Math.Max(lwork, 1));
            CuSolverNative.Zgetrf(handle, m, n, (Complex*)a.Pointer, m, (Complex*)ws.Pointer, (int*)ipiv.Pointer, (int*)devInfo.Pointer);
            s.Synchronize();
            return ReadInfo(devInfo, "zgetrf");
        }
        finally
        {
            CuSolverNative.Destroy(handle);
        }
    }

    /// <summary>
    /// Solves <c>op(A)·X = B</c> from the factors produced by
    /// <see cref="LuFactorAsync(DeviceBuffer{Complex}, DeviceBuffer{int}, int, int, CudaStream, CancellationToken)"/>.
    /// </summary>
    /// <param name="a">Device buffer holding the n×n L\U factors.</param>
    /// <param name="ipiv">Pivot indices from the factorisation.</param>
    /// <param name="b">Device buffer for the n×nrhs right-hand sides; overwritten with the solution.</param>
    /// <param name="n">Order of A.</param>
    /// <param name="nrhs">Number of right-hand sides.</param>
    /// <param name="transpose">Solve with the transpose of A.</param>
    /// <param name="conjugate">With <paramref name="transpose"/>, solve with the conjugate transpose.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task LuSolveAsync(
        DeviceBuffer<Complex> a, DeviceBuffer<int> ipiv, DeviceBuffer<Complex> b, int n, int nrhs = 1,
        bool transpose = false, bool conjugate = false,
        CudaStream? stream = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(ipiv);
        ArgumentNullException.ThrowIfNull(b);
        int trans = !transpose ? 0 : conjugate ? 2 : 1;

        await RunAsync(stream, ct, s => LuSolveComplexCore(a, ipiv, b, n, nrhs, trans, s)).ConfigureAwait(false);
    }

    private static unsafe void LuSolveComplexCore(
        DeviceBuffer<Complex> a, DeviceBuffer<int> ipiv, DeviceBuffer<Complex> b, int n, int nrhs, int trans, CudaStream s)
    {
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            using var devInfo = new DeviceBuffer<int>(1);
            CuSolverNative.Zgetrs(handle, trans, n, nrhs, (Complex*)a.Pointer, n, (int*)ipiv.Pointer, (Complex*)b.Pointer, n, (int*)devInfo.Pointer);
            s.Synchronize();
            ReadInfo(devInfo, "zgetrs");
        }
        finally
        {
            CuSolverNative.Destroy(handle);
        }
    }

    // ── QR ────────────────────────────────────────────────────────────────────

    /// <inheritdoc cref="QrAsync{T}(DeviceBuffer{T}, DeviceBuffer{T}, int, int, CudaStream, CancellationToken)"/>
    public static async Task QrAsync(
        DeviceBuffer<Complex> a, DeviceBuffer<Complex> tau, int m, int n,
        CudaStream? stream = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(tau);
        await RunAsync(stream, ct, s => QrComplexCore(a, tau, m, n, s)).ConfigureAwait(false);
    }

    private static unsafe void QrComplexCore(DeviceBuffer<Complex> a, DeviceBuffer<Complex> tau, int m, int n, CudaStream s)
    {
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            using var devInfo = new DeviceBuffer<int>(1);
            int lwork = CuSolverNative.ZgeqrfBufferSize(handle, m, n, (Complex*)a.Pointer, m);
            using var ws = new DeviceBuffer<Complex>(Math.Max(lwork, 1));
            CuSolverNative.Zgeqrf(handle, m, n, (Complex*)a.Pointer, m, (Complex*)tau.Pointer, (Complex*)ws.Pointer, lwork, (int*)devInfo.Pointer);
            s.Synchronize();
            ReadInfo(devInfo, "zgeqrf");
        }
        finally
        {
            CuSolverNative.Destroy(handle);
        }
    }

    /// <inheritdoc cref="QrFormQAsync{T}(DeviceBuffer{T}, DeviceBuffer{T}, int, int, CudaStream, CancellationToken)"/>
    public static async Task QrFormQAsync(
        DeviceBuffer<Complex> a, DeviceBuffer<Complex> tau, int m, int n,
        CudaStream? stream = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(tau);
        if (m < n)
        {
            throw new ArgumentException("QrFormQ requires m >= n.");
        }

        await RunAsync(stream, ct, s => QrFormQComplexCore(a, tau, m, n, s)).ConfigureAwait(false);
    }

    private static unsafe void QrFormQComplexCore(DeviceBuffer<Complex> a, DeviceBuffer<Complex> tau, int m, int n, CudaStream s)
    {
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            using var devInfo = new DeviceBuffer<int>(1);
            int lwork = CuSolverNative.ZungqrBufferSize(handle, m, n, n, (Complex*)a.Pointer, m, (Complex*)tau.Pointer);
            using var ws = new DeviceBuffer<Complex>(Math.Max(lwork, 1));
            CuSolverNative.Zungqr(handle, m, n, n, (Complex*)a.Pointer, m, (Complex*)tau.Pointer, (Complex*)ws.Pointer, lwork, (int*)devInfo.Pointer);
            s.Synchronize();
            ReadInfo(devInfo, "zungqr");
        }
        finally
        {
            CuSolverNative.Destroy(handle);
        }
    }

    // ── Least squares ─────────────────────────────────────────────────────────

    /// <inheritdoc cref="LeastSquaresAsync{T}(DeviceBuffer{T}, DeviceBuffer{T}, int, int, int, CudaStream, CancellationToken)"/>
    public static async Task LeastSquaresAsync(
        DeviceBuffer<Complex> a, DeviceBuffer<Complex> b, int m, int n, int nrhs = 1,
        CudaStream? stream = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (m < n)
        {
            throw new ArgumentException("LeastSquares requires m >= n (overdetermined or square).");
        }

        await RunAsync(stream, ct, s => LeastSquaresComplexCore(a, b, m, n, nrhs, s)).ConfigureAwait(false);
    }

    private static unsafe void LeastSquaresComplexCore(
        DeviceBuffer<Complex> a, DeviceBuffer<Complex> b, int m, int n, int nrhs, CudaStream s)
    {
        nint solver = CuSolverNative.Create();
        nint blas = CuBlasNative.Create();
        CuSolverNative.SetStream(solver, s.Handle);
        CuBlasNative.SetStream(blas, s.Handle);
        try
        {
            using var devInfo = new DeviceBuffer<int>(1);
            using var tau = new DeviceBuffer<Complex>(n);
            Span<byte> oneStore = stackalloc byte[AlignedScalar.StorageSize];
            Complex* one = AlignedScalar.Store(oneStore, Complex.One);
            Complex* pA = (Complex*)a.Pointer, pB = (Complex*)b.Pointer, pTau = (Complex*)tau.Pointer;

            int lwork = CuSolverNative.ZgeqrfBufferSize(solver, m, n, pA, m);
            using (var ws = new DeviceBuffer<Complex>(Math.Max(lwork, 1)))
            {
                CuSolverNative.Zgeqrf(solver, m, n, pA, m, pTau, (Complex*)ws.Pointer, lwork, (int*)devInfo.Pointer);
                s.Synchronize();
                ReadInfo(devInfo, "zgeqrf");
            }

            // B <- Q^H B
            lwork = CuSolverNative.ZunmqrBufferSize(solver, CuBlasSideMode.Left, CuBlasOperation.ConjTranspose, m, nrhs, n, pA, m, pTau, pB, m);
            using (var ws = new DeviceBuffer<Complex>(Math.Max(lwork, 1)))
            {
                CuSolverNative.Zunmqr(solver, CuBlasSideMode.Left, CuBlasOperation.ConjTranspose, m, nrhs, n, pA, m, pTau, pB, m, (Complex*)ws.Pointer, lwork, (int*)devInfo.Pointer);
                s.Synchronize();
                ReadInfo(devInfo, "zunmqr");
            }

            CuBlasNative.Ztrsm(blas, CuBlasSideMode.Left, CuBlasFillMode.Upper, CuBlasOperation.None, CuBlasDiagType.NonUnit,
                n, nrhs, one, pA, m, pB, m);
            s.Synchronize();
        }
        finally
        {
            CuBlasNative.Destroy(blas);
            CuSolverNative.Destroy(solver);
        }
    }

    // ── Cholesky / Hermitian eigenproblem ─────────────────────────────────────

    /// <summary>
    /// Cholesky factorisation of a Hermitian positive-definite matrix, in place. Throws
    /// <see cref="InvalidOperationException"/> if the matrix is not positive definite.
    /// </summary>
    /// <param name="a">Device buffer for the n×n matrix (overwritten with the factor).</param>
    /// <param name="n">Order of the matrix.</param>
    /// <param name="upper">Compute the upper factor instead of the lower.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task CholeskyAsync(
        DeviceBuffer<Complex> a, int n, bool upper = false,
        CudaStream? stream = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        await RunAsync(stream, ct, s => CholeskyComplexCore(a, n, upper, s)).ConfigureAwait(false);
    }

    private static unsafe void CholeskyComplexCore(DeviceBuffer<Complex> a, int n, bool upper, CudaStream s)
    {
        var uplo = upper ? CuBlasFillMode.Upper : CuBlasFillMode.Lower;
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            using var devInfo = new DeviceBuffer<int>(1);
            int lwork = CuSolverNative.ZpotrfBufferSize(handle, uplo, n, (Complex*)a.Pointer, n);
            using var ws = new DeviceBuffer<Complex>(Math.Max(lwork, 1));
            CuSolverNative.Zpotrf(handle, uplo, n, (Complex*)a.Pointer, n, (Complex*)ws.Pointer, lwork, (int*)devInfo.Pointer);
            s.Synchronize();
            ThrowIfInfo(devInfo, "zpotrf (matrix is not positive definite)");
        }
        finally
        {
            CuSolverNative.Destroy(handle);
        }
    }

    /// <summary>
    /// Eigenvalue decomposition of a Hermitian matrix: eigenvalues (real, ascending) in
    /// <paramref name="w"/> and, if requested, orthonormal eigenvectors overwriting <paramref name="a"/>.
    /// </summary>
    /// <param name="a">Device buffer for the n×n Hermitian matrix.</param>
    /// <param name="w">Device buffer of n real eigenvalues (output).</param>
    /// <param name="n">Order of the matrix.</param>
    /// <param name="computeVectors">Whether to compute eigenvectors.</param>
    /// <param name="upper">Whether the upper triangle is referenced instead of the lower.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task HermitianEigAsync(
        DeviceBuffer<Complex> a, DeviceBuffer<double> w, int n,
        bool computeVectors = true, bool upper = false,
        CudaStream? stream = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(w);
        await RunAsync(stream, ct, s => HermitianEigCore(a, w, n, computeVectors, upper, s)).ConfigureAwait(false);
    }

    private static unsafe void HermitianEigCore(
        DeviceBuffer<Complex> a, DeviceBuffer<double> w, int n, bool computeVectors, bool upper, CudaStream s)
    {
        var jobz = computeVectors ? CuSolverEigMode.Vector : CuSolverEigMode.NoVector;
        var uplo = upper ? CuBlasFillMode.Upper : CuBlasFillMode.Lower;
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, s.Handle);
        try
        {
            using var devInfo = new DeviceBuffer<int>(1);
            int lwork = CuSolverNative.ZheevdBufferSize(handle, jobz, uplo, n, (Complex*)a.Pointer, n, (double*)w.Pointer);
            using var ws = new DeviceBuffer<Complex>(Math.Max(lwork, 1));
            CuSolverNative.Zheevd(handle, jobz, uplo, n, (Complex*)a.Pointer, n, (double*)w.Pointer, (Complex*)ws.Pointer, lwork, (int*)devInfo.Pointer);
            s.Synchronize();
            ThrowIfInfo(devInfo, "zheevd (did not converge)");
        }
        finally
        {
            CuSolverNative.Destroy(handle);
        }
    }
}
