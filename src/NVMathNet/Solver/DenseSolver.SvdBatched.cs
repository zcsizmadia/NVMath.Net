// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Interop;

namespace NVMathNet.Solver;

// Batched SVD for small matrices (cuSOLVER gesvdjBatched, Jacobi method).
public static partial class DenseSolver
{
    /// <summary>
    /// Singular value decomposition <c>A_i = U_i · diag(S_i) · V_i^T</c> of a batch of small
    /// matrices (<c>m</c> and <c>n</c> at most 32) using the Jacobi method. Matrices are packed
    /// back-to-back and column-major.
    /// </summary>
    /// <param name="a">Device buffer of <c>batchCount·m·n</c> elements; destroyed on return.</param>
    /// <param name="s">Device buffer of <c>batchCount·min(m,n)</c> singular values, sorted descending (output).</param>
    /// <param name="u">Device buffer of <c>batchCount·m·m</c> left vectors (output); may be <c>null</c> when <paramref name="computeVectors"/> is <c>false</c> (a scratch buffer is used).</param>
    /// <param name="v">Device buffer of <c>batchCount·n·n</c> right vectors — <b>V, not V^T</b> (output); may be <c>null</c> when <paramref name="computeVectors"/> is <c>false</c> (a scratch buffer is used).</param>
    /// <param name="m">Rows of each matrix.</param>
    /// <param name="n">Columns of each matrix.</param>
    /// <param name="batchCount">Number of matrices.</param>
    /// <param name="computeVectors">Whether to compute U and V.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Per-matrix status: 0 = converged; a positive value means the matrix did not converge.</returns>
    public static async Task<int[]> SvdBatchedAsync<T>(
        DeviceBuffer<T> a, DeviceBuffer<T> s, DeviceBuffer<T>? u, DeviceBuffer<T>? v,
        int m, int n, int batchCount, bool computeVectors = true,
        CudaStream? stream = null, CancellationToken ct = default) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(s);
        RequireFloatOrDouble<T>();
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
        await RunAsync(stream, ct, st => infos = SvdBatchedCore(a, s, u, v, m, n, batchCount, computeVectors, st)).ConfigureAwait(false);
        return infos;
    }

    private static unsafe int[] SvdBatchedCore<T>(
        DeviceBuffer<T> a, DeviceBuffer<T> s, DeviceBuffer<T>? u, DeviceBuffer<T>? v,
        int m, int n, int batchCount, bool computeVectors, CudaStream stream)
        where T : unmanaged, INumberBase<T>
    {
        Dim(m, nameof(m)); Dim(n, nameof(n)); Dim(batchCount, nameof(batchCount));
        Need(a.Length, (long)m * n * batchCount, nameof(a)); Need(s.Length, (long)Math.Min(m, n) * batchCount, nameof(s));
        if (u is not null) { Need(u.Length, (long)m * m * batchCount, nameof(u)); }
        if (v is not null) { Need(v.Length, (long)n * n * batchCount, nameof(v)); }
        var jobz = computeVectors ? CuSolverEigMode.Vector : CuSolverEigMode.NoVector;
        // cuSOLVER dereferences U and V even for NOVECTOR, so supply scratch buffers when the caller has none.
        using var scratchU = u is null ? new DeviceBuffer<T>((long)m * m * batchCount) : null;
        using var scratchV = v is null ? new DeviceBuffer<T>((long)n * n * batchCount) : null;
        u ??= scratchU;
        v ??= scratchV;
        using var infos = new DeviceBuffer<int>(batchCount);
        nint handle = CuSolverNative.Create();
        CuSolverNative.SetStream(handle, stream.Handle);
        nint gesvdjParams = CuSolverNative.CreateGesvdjInfo();
        try
        {
            if (typeof(T) == typeof(float))
            {
                float* pU = (float*)u!.Pointer;
                float* pV = (float*)v!.Pointer;
                int lwork = CuSolverNative.SgesvdjBatchedBufferSize(handle, jobz, m, n, (float*)a.Pointer, m, (float*)s.Pointer, pU, m, pV, n, gesvdjParams, batchCount);
                using var ws = new DeviceBuffer<float>(Math.Max(lwork, 1));
                CuSolverNative.SgesvdjBatched(handle, jobz, m, n, (float*)a.Pointer, m, (float*)s.Pointer, pU, m, pV, n, (float*)ws.Pointer, lwork, (int*)infos.Pointer, gesvdjParams, batchCount);
            }
            else
            {
                double* pU = (double*)u!.Pointer;
                double* pV = (double*)v!.Pointer;
                int lwork = CuSolverNative.DgesvdjBatchedBufferSize(handle, jobz, m, n, (double*)a.Pointer, m, (double*)s.Pointer, pU, m, pV, n, gesvdjParams, batchCount);
                using var ws = new DeviceBuffer<double>(Math.Max(lwork, 1));
                CuSolverNative.DgesvdjBatched(handle, jobz, m, n, (double*)a.Pointer, m, (double*)s.Pointer, pU, m, pV, n, (double*)ws.Pointer, lwork, (int*)infos.Pointer, gesvdjParams, batchCount);
            }

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
