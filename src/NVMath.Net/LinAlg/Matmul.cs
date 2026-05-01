// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using System.Numerics;
using NVMathNet.Interop;

namespace NVMathNet.LinAlg;

/// <summary>
/// Stateful cuBLASLt matrix multiplication wrapper.
/// Mirrors the design of <c>nvmath.linalg.Matmul</c>:
/// <list type="number">
///   <item><description>Construct with dimension info and options.</description></item>
///   <item><description>Call <see cref="PlanAsync"/> to select an algorithm.</description></item>
///   <item><description>Call <see cref="ExecuteAsync"/> (possibly multiple times).</description></item>
///   <item><description>Dispose to free CUDA resources.</description></item>
/// </list>
/// Computes: D = alpha * op(A) * op(B) + beta * C
/// where op() is optional transpose.
/// </summary>
public sealed class Matmul : IAsyncDisposable, IDisposable
{
    // ── cuBLASLt handles ───────────────────────────────────────────────────────

    private readonly nint _ltHandle;
    private nint _matmulDesc;
    private nint _layoutA;
    private nint _layoutB;
    private nint _layoutC;
    private nint _layoutD;
    private nint _preference;
    private byte[]? _algoBuffer;
    private nuint  _algoWorkspaceSize;
    private bool _planned;
    private bool _disposed;

    // ── Workspace ─────────────────────────────────────────────────────────────

    private nint   _workspacePtr;

    // ── Dimensions ────────────────────────────────────────────────────────────

    private readonly long _m, _n, _k;
    private readonly long _batchCount;
    private readonly MatmulOptions _options;
    private readonly CudaStream _ownedStream;

    /// <summary>
    /// Creates a Matmul descriptor.
    /// </summary>
    /// <param name="m">Rows of A and C/D.</param>
    /// <param name="n">Columns of B and C/D.</param>
    /// <param name="k">Inner dimension (cols of A / rows of B).</param>
    /// <param name="batchCount">Number of independent GEMMs in a batched operation.  Default: 1.</param>
    /// <param name="options">Operation options; <c>null</c> uses defaults.</param>
    /// <param name="stream">Optional external stream.</param>
    public Matmul(
        long m, long n, long k,
        long batchCount = 1,
        MatmulOptions? options = null,
        CudaStream? stream = null)
    {
        _m = m; _n = n; _k = k;
        _batchCount = batchCount;
        _options = options ?? new MatmulOptions();
        _ownedStream = stream ?? new CudaStream();

        _ltHandle = CuBlasNative.LtCreate();
        CreateDescriptors();
    }

    private unsafe void CreateDescriptors()
    {
        int computeType = (int)_options.ComputeType;
        int scaleType   = (int)_options.ScaleType;
        int typeA = (int)_options.TypeA;
        int typeB = (int)_options.TypeB;
        int typeC = (int)_options.TypeC;
        int typeD = (int)_options.TypeD;

        _matmulDesc = CuBlasNative.MatmulDescCreate((CuBlasComputeType)computeType, (CudaDataType)scaleType);

        // Set transpose operations
        if (_options.TransposeA)
        {
            int opT = (int)CuBlasOperation.Transpose;
            CuBlasNative.MatmulDescSetAttribute(
                _matmulDesc,
                0, // CUBLASLT_MATMUL_DESC_TRANSA
                &opT, sizeof(int));
        }
        if (_options.TransposeB)
        {
            int opT = (int)CuBlasOperation.Transpose;
            CuBlasNative.MatmulDescSetAttribute(
                _matmulDesc,
                1, // CUBLASLT_MATMUL_DESC_TRANSB,
                &opT, sizeof(int));
        }

        // Set epilog
        if (_options.Epilog != MatmulEpilog.Default)
        {
            int epilog = (int)_options.Epilog;
            CuBlasNative.MatmulDescSetAttribute(
                _matmulDesc,
                5, // CUBLASLT_MATMUL_DESC_EPILOGUE
                &epilog, sizeof(int));
        }

        long ldA = _options.TransposeA ? _k : _m;
        long ldB = _options.TransposeB ? _n : _k;
        long ldC = _m;

        _layoutA = CuBlasNative.MatrixLayoutCreate((CudaDataType)typeA, (ulong)_m, (ulong)_k, ldA);
        _layoutB = CuBlasNative.MatrixLayoutCreate((CudaDataType)typeB, (ulong)_k, (ulong)_n, ldB);
        _layoutC = CuBlasNative.MatrixLayoutCreate((CudaDataType)typeC, (ulong)_m, (ulong)_n, ldC);
        _layoutD = CuBlasNative.MatrixLayoutCreate((CudaDataType)typeD, (ulong)_m, (ulong)_n, ldC);

        _preference = CuBlasNative.MatmulPreferenceCreate();
        nuint ws = _options.MaxWorkspaceBytes;
        CuBlasNative.MatmulPreferenceSetAttribute(
            _preference,
            0, // CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
            &ws, (nuint)sizeof(nuint));
    }

    // ── Plan ──────────────────────────────────────────────────────────────────

    /// <summary>Selects the best algorithm on a background thread.</summary>
    /// <param name="preferences">
    /// Optional fine-grained algorithm preferences.  <c>null</c> uses the defaults
    /// from the <see cref="MatmulOptions"/> supplied at construction.
    /// </param>
    /// <param name="ct">Cancellation token.</param>
    public async Task PlanAsync(MatmulPlanPreferences? preferences = null, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_planned) return;
        await Task.Run(() => DoPlan(preferences), ct).ConfigureAwait(false);
    }

    /// <summary>Synchronously selects the best algorithm.</summary>
    /// <param name="preferences">
    /// Optional fine-grained algorithm preferences.  <c>null</c> uses the defaults
    /// from the <see cref="MatmulOptions"/> supplied at construction.
    /// </param>
    public void Plan(MatmulPlanPreferences? preferences = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_planned) return;
        DoPlan(preferences);
    }

    private unsafe void DoPlan(MatmulPlanPreferences? prefs)
    {
        // Apply optional workspace override from preferences
        if (prefs?.MaxWorkspaceBytes is nuint wsOverride)
        {
            CuBlasNative.MatmulPreferenceSetAttribute(
                _preference,
                0, // CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
                &wsOverride, (nuint)sizeof(nuint));
        }

        // Apply numerical implementation mask
        if (prefs is { NumericalImplMask: not 0 })
        {
            ulong mask = prefs.NumericalImplMask;
            CuBlasNative.MatmulPreferenceSetAttribute(
                _preference,
                4, // CUBLASLT_MATMUL_PREF_IMPL_MASK
                &mask, sizeof(ulong));
        }

        int algoCount = prefs?.RequestedAlgoCount ?? 1;
        _algoBuffer = new byte[CuBlasNative.HeuristicResultSize * algoCount];
        int count = CuBlasNative.MatmulAlgoGetHeuristic(
            _ltHandle, _matmulDesc,
            _layoutA, _layoutB, _layoutC, _layoutD,
            _preference, algoCount, _algoBuffer);

        if (count == 0)
            throw new InvalidOperationException("No cuBLASLt algorithm found for the given matmul configuration.");

        // Trim to first (highest-scored) result
        if (algoCount > 1 && _algoBuffer.Length > CuBlasNative.HeuristicResultSize)
        {
            byte[] first = new byte[CuBlasNative.HeuristicResultSize];
            Array.Copy(_algoBuffer, 0, first, 0, CuBlasNative.HeuristicResultSize);
            _algoBuffer = first;
        }

        // Extract workspace size from result (offset 64 in the struct layout)
        _algoWorkspaceSize = (nuint)System.Runtime.InteropServices.MemoryMarshal
            .Read<ulong>(_algoBuffer.AsSpan(64));

        if (_algoWorkspaceSize > 0)
            _workspacePtr = (nint)CudaRuntime.Malloc(_algoWorkspaceSize);

        _planned = true;
    }

    // ── Execute ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Executes D = alpha*op(A)*op(B) + beta*C on device buffers.
    /// Prefer <see cref="ExecuteAsync"/> in async contexts.
    /// </summary>
    public void Execute(
        nint pA, nint pB, nint pC, nint pD,
        float alpha = 1.0f, float beta = 0.0f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_planned)
            throw new InvalidOperationException("Call Plan() or PlanAsync() before Execute.");

        RunGemm(pA, pB, pC, pD, alpha, beta);
        if (_options.Blocking) _ownedStream.Synchronize();
    }

    /// <summary>
    /// Asynchronously executes the matrix multiplication and returns a
    /// <see cref="Task"/> that completes when the GPU kernel finishes.
    /// </summary>
    public async Task ExecuteAsync(
        nint pA, nint pB, nint pC, nint pD,
        float alpha = 1.0f, float beta = 0.0f,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_planned)
            throw new InvalidOperationException("Call Plan() or PlanAsync() before Execute.");

        RunGemm(pA, pB, pC, pD, alpha, beta);
        if (_options.Blocking)
            await _ownedStream.SynchronizeAsync(ct).ConfigureAwait(false);
    }

    private unsafe void RunGemm(nint pA, nint pB, nint pC, nint pD, float alpha, float beta)
    {
        fixed (byte* pAlgo = _algoBuffer)
        CuBlasNative.Matmul(
            _ltHandle,
            _matmulDesc,
            &alpha,
            (void*)pA, _layoutA,
            (void*)pB, _layoutB,
            &beta,
            (void*)pC, _layoutC,
            (void*)pD, _layoutD,
            pAlgo,
            (void*)_workspacePtr, _algoWorkspaceSize,
            _ownedStream.Handle);
    }

    /// <summary>Waits for all work on the internal stream to complete.</summary>
    public Task SynchronizeAsync(CancellationToken ct = default) =>
        _ownedStream.SynchronizeAsync(ct);

    // ── Convenience overloads ─────────────────────────────────────────────────

    /// <inheritdoc cref="ExecuteAsync(nint, nint, nint, nint, float, float, CancellationToken)"/>
    public Task ExecuteAsync<T>(
        DeviceBuffer<T> a, DeviceBuffer<T> b, DeviceBuffer<T> c, DeviceBuffer<T> d,
        float alpha = 1.0f, float beta = 0.0f,
        CancellationToken ct = default) where T : unmanaged, INumberBase<T>
        => ExecuteAsync(a.PointerAsInt, b.PointerAsInt, c.PointerAsInt, d.PointerAsInt, alpha, beta, ct);

    // ── Dispose ───────────────────────────────────────────────────────────────
    /// <inheritdoc/>
    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        await _ownedStream.SynchronizeAsync().ConfigureAwait(false);
        DisposeCore();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        DisposeCore();
    }

    private unsafe void DisposeCore()
    {
        if (_disposed) return;
        _disposed = true;

        if (_preference != default) CuBlasNative.MatmulPreferenceDestroy(_preference);
        if (_layoutD    != default) CuBlasNative.MatrixLayoutDestroy(_layoutD);
        if (_layoutC    != default) CuBlasNative.MatrixLayoutDestroy(_layoutC);
        if (_layoutB    != default) CuBlasNative.MatrixLayoutDestroy(_layoutB);
        if (_layoutA    != default) CuBlasNative.MatrixLayoutDestroy(_layoutA);
        if (_matmulDesc != default) CuBlasNative.MatmulDescDestroy(_matmulDesc);
        if (_ltHandle   != default) CuBlasNative.LtDestroy(_ltHandle);

        if (_workspacePtr != 0)
        {
            unsafe { CudaRuntime.Free((void*)_workspacePtr); }
            _workspacePtr = 0;
        }

        _ownedStream.Dispose();
    }
}
