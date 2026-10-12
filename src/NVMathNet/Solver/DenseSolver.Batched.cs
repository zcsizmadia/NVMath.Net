// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Interop;

namespace NVMathNet.Solver;

// Batched LU factorisation and solve (cuBLAS getrfBatched/getrsBatched).
// Matrices of a batch are packed back-to-back in one column-major buffer.
public static partial class DenseSolver
{
    /// <summary>
    /// LU factorisation with partial pivoting of a batch of independent n×n matrices, in place.
    /// </summary>
    /// <param name="a">Device buffer of <c>batchCount·n·n</c> elements; overwritten with the L\U factors.</param>
    /// <param name="ipiv">Device buffer of <c>batchCount·n</c> pivot indices (output).</param>
    /// <param name="n">Order of each matrix.</param>
    /// <param name="batchCount">Number of matrices.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Per-matrix status: 0 = success; a positive value <c>i</c> means <c>U(i,i)</c> is exactly zero.</returns>
    public static async Task<int[]> LuFactorBatchedAsync<T>(
        DeviceBuffer<T> a, DeviceBuffer<int> ipiv, int n, int batchCount,
        CudaStream? stream = null, CancellationToken ct = default) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(ipiv);
        RequireFloatOrDouble<T>();
        RequireSize(a.Length, (long)n * n * batchCount, nameof(a));
        RequireSize(ipiv.Length, (long)n * batchCount, nameof(ipiv));

        int[] infos = [];
        await RunAsync(stream, ct, s => infos = LuFactorBatchedCore(a, ipiv, n, batchCount, s)).ConfigureAwait(false);
        return infos;
    }

    private static unsafe int[] LuFactorBatchedCore<T>(
        DeviceBuffer<T> a, DeviceBuffer<int> ipiv, int n, int batchCount, CudaStream s)
        where T : unmanaged, INumberBase<T>
    {
        using var ptrs = MakePointerArray(a, (long)n * n, batchCount);
        using var infos = new DeviceBuffer<int>(batchCount);
        nint handle = CuBlasNative.Create();
        CuBlasNative.SetStream(handle, s.Handle);
        try
        {
            if (typeof(T) == typeof(float))
            {
                CuBlasNative.SgetrfBatched(handle, n, (float**)ptrs.Pointer, n, (int*)ipiv.Pointer, (int*)infos.Pointer, batchCount);
            }
            else
            {
                CuBlasNative.DgetrfBatched(handle, n, (double**)ptrs.Pointer, n, (int*)ipiv.Pointer, (int*)infos.Pointer, batchCount);
            }

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
    /// Solves <c>op(A_i)·X_i = B_i</c> for every matrix of a batch from the factors produced by
    /// <see cref="LuFactorBatchedAsync{T}"/>.
    /// </summary>
    /// <param name="a">Device buffer holding the batch of L\U factors.</param>
    /// <param name="ipiv">Pivot indices from <see cref="LuFactorBatchedAsync{T}"/>.</param>
    /// <param name="b">Device buffer of <c>batchCount·n·nrhs</c> right-hand sides; overwritten with the solutions.</param>
    /// <param name="n">Order of each matrix.</param>
    /// <param name="nrhs">Right-hand sides per matrix.</param>
    /// <param name="batchCount">Number of matrices.</param>
    /// <param name="transpose">Solve with the transpose of each A.</param>
    /// <param name="stream">Optional CUDA stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task LuSolveBatchedAsync<T>(
        DeviceBuffer<T> a, DeviceBuffer<int> ipiv, DeviceBuffer<T> b, int n, int nrhs, int batchCount,
        bool transpose = false, CudaStream? stream = null, CancellationToken ct = default)
        where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(ipiv);
        ArgumentNullException.ThrowIfNull(b);
        RequireFloatOrDouble<T>();
        RequireSize(a.Length, (long)n * n * batchCount, nameof(a));
        RequireSize(ipiv.Length, (long)n * batchCount, nameof(ipiv));
        RequireSize(b.Length, (long)n * nrhs * batchCount, nameof(b));

        await RunAsync(stream, ct, s => LuSolveBatchedCore(a, ipiv, b, n, nrhs, batchCount, transpose, s)).ConfigureAwait(false);
    }

    private static unsafe void LuSolveBatchedCore<T>(
        DeviceBuffer<T> a, DeviceBuffer<int> ipiv, DeviceBuffer<T> b,
        int n, int nrhs, int batchCount, bool transpose, CudaStream s)
        where T : unmanaged, INumberBase<T>
    {
        using var aPtrs = MakePointerArray(a, (long)n * n, batchCount);
        using var bPtrs = MakePointerArray(b, (long)n * nrhs, batchCount);
        var trans = transpose ? CuBlasOperation.Transpose : CuBlasOperation.None;
        int info = 0;
        nint handle = CuBlasNative.Create();
        CuBlasNative.SetStream(handle, s.Handle);
        try
        {
            if (typeof(T) == typeof(float))
            {
                CuBlasNative.SgetrsBatched(handle, trans, n, nrhs, (float**)aPtrs.Pointer, n, (int*)ipiv.Pointer, (float**)bPtrs.Pointer, n, &info, batchCount);
            }
            else
            {
                CuBlasNative.DgetrsBatched(handle, trans, n, nrhs, (double**)aPtrs.Pointer, n, (int*)ipiv.Pointer, (double**)bPtrs.Pointer, n, &info, batchCount);
            }

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

    // Device array holding the address of each matrix of a packed batch.
    private static unsafe DeviceBuffer<nint> MakePointerArray<T>(DeviceBuffer<T> buffer, long strideElements, int count)
        where T : unmanaged, INumberBase<T>
    {
        nint[] host = new nint[count];
        for (int i = 0; i < count; i++)
        {
            host[i] = buffer.PointerAsInt + (nint)(i * strideElements * sizeof(T));
        }

        var ptrs = new DeviceBuffer<nint>(count);
        ptrs.CopyFrom(host.AsSpan());
        return ptrs;
    }

    private static void RequireSize(long actual, long required, string name)
    {
        if (actual < required)
        {
            throw new ArgumentException($"Buffer holds {actual} elements but {required} are required.", name);
        }
    }
}
