// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Interop;
using NVMathNet.Memory;

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
    private byte[] _allAlgos = [];
    private int _algoCount;
    private nuint  _algoWorkspaceSize;
    private bool _planned;
    private bool _disposed;

    // ── Workspace ─────────────────────────────────────────────────────────────

    private nint   _workspacePtr;
    private Workspace _workspace;

    // ── cuBLASLt attribute IDs (see cublasLt.h) ───────────────────────────────

    private const int DescTransA = 3;
    private const int DescTransB = 4;
    private const int DescEpilogue = 7;
    private const int DescBiasPointer = 8;
    private const int DescBiasBatchStride = 10;
    private const int DescAScalePointer = 17;
    private const int DescBScalePointer = 18;
    private const int DescCScalePointer = 19;
    private const int DescDScalePointer = 20;
    private const int DescAmaxDPointer = 21;
    private const int DescFastAccum = 25;
    private const int DescBiasDataType = 26;
    private const int DescEpilogueAuxPointer = 11;
    private const int DescEpilogueAuxLd = 12;
    private const int DescEpilogueAuxBatchStride = 13;
    private const int DescEpilogueAuxDataType = 22;
    private const int LayoutBatchCount = 5;
    private const int LayoutStridedBatchOffset = 6;
    private const int PrefMaxWorkspaceBytes = 1;
    private const int PrefImplMask = 12;

    // ── Dimensions ────────────────────────────────────────────────────────────

    private readonly long _m, _n, _k;
    private readonly long _batchCount;
    private readonly MatmulOptions _options;
    private readonly CudaStream _ownedStream;
    private readonly bool _streamOwned;

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
        _streamOwned = stream is null;

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
                DescTransA,
                &opT, sizeof(int));
        }
        if (_options.TransposeB)
        {
            int opT = (int)CuBlasOperation.Transpose;
            CuBlasNative.MatmulDescSetAttribute(
                _matmulDesc,
                DescTransB,
                &opT, sizeof(int));
        }

        // Set epilog
        if (_options.Epilog != MatmulEpilog.Default)
        {
            int epilog = (int)_options.Epilog;
            CuBlasNative.MatmulDescSetAttribute(
                _matmulDesc,
                DescEpilogue,
                &epilog, sizeof(int));
        }

        long ldA = _options.TransposeA ? _k : _m;
        long ldB = _options.TransposeB ? _n : _k;
        long ldC = _m;

        // Layout describes the physical (pre-transpose) matrix in memory.
        // When transposed, rows and cols of the stored matrix are swapped.
        _layoutA = _options.TransposeA
            ? CuBlasNative.MatrixLayoutCreate((CudaDataType)typeA, (ulong)_k, (ulong)_m, ldA)
            : CuBlasNative.MatrixLayoutCreate((CudaDataType)typeA, (ulong)_m, (ulong)_k, ldA);
        _layoutB = _options.TransposeB
            ? CuBlasNative.MatrixLayoutCreate((CudaDataType)typeB, (ulong)_n, (ulong)_k, ldB)
            : CuBlasNative.MatrixLayoutCreate((CudaDataType)typeB, (ulong)_k, (ulong)_n, ldB);
        _layoutC = CuBlasNative.MatrixLayoutCreate((CudaDataType)typeC, (ulong)_m, (ulong)_n, ldC);
        _layoutD = CuBlasNative.MatrixLayoutCreate((CudaDataType)typeD, (ulong)_m, (ulong)_n, ldC);

        if (_options.FastAccum)
        {
            int fast = 1;
            CuBlasNative.MatmulDescSetAttribute(_matmulDesc, DescFastAccum, &fast, sizeof(int));
        }

        if (_batchCount > 1)
        {
            // Strides are in elements; matrices are packed back-to-back.
            SetBatch(_layoutA, _m * _k);
            SetBatch(_layoutB, _k * _n);
            SetBatch(_layoutC, _m * _n);
            SetBatch(_layoutD, _m * _n);
        }

        _preference = CuBlasNative.MatmulPreferenceCreate();
        ulong ws = _options.MaxWorkspaceBytes;
        CuBlasNative.MatmulPreferenceSetAttribute(
            _preference,
            PrefMaxWorkspaceBytes,
            &ws, (nuint)sizeof(ulong));
    }

    private unsafe void SetBatch(nint layout, long strideElements)
    {
        int count = checked((int)_batchCount);
        CuBlasNative.MatrixLayoutSetAttribute(layout, LayoutBatchCount, &count, sizeof(int));
        CuBlasNative.MatrixLayoutSetAttribute(layout, LayoutStridedBatchOffset, &strideElements, sizeof(long));
    }

    // ── Scale / bias pointers ─────────────────────────────────────────────────

    /// <summary>
    /// Sets the device pointers to the scalar scale factors used by FP8 matmul
    /// and the optional <c>amax(D)</c> output. Pass <c>0</c> to leave a pointer
    /// unchanged. Call before <see cref="Execute"/>.
    /// </summary>
    /// <param name="aScale">Device pointer to the float scale of A.</param>
    /// <param name="bScale">Device pointer to the float scale of B.</param>
    /// <param name="cScale">Device pointer to the float scale of C.</param>
    /// <param name="dScale">Device pointer to the float scale of D.</param>
    /// <param name="amaxD">Device pointer receiving the absolute maximum of D.</param>
    public void SetScalePointers(nint aScale = 0, nint bScale = 0, nint cScale = 0, nint dScale = 0, nint amaxD = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        SetPointerAttr(DescAScalePointer, aScale);
        SetPointerAttr(DescBScalePointer, bScale);
        SetPointerAttr(DescCScalePointer, cScale);
        SetPointerAttr(DescDScalePointer, dScale);
        SetPointerAttr(DescAmaxDPointer, amaxD);
    }

    /// <summary>
    /// Sets the bias vector used by the <c>Bias</c> family of epilogs.
    /// </summary>
    /// <param name="bias">Device pointer to the bias vector (length M).</param>
    /// <param name="biasType">Element type of the bias; <c>null</c> keeps the cuBLASLt default.</param>
    /// <param name="batchStride">Stride in elements between batches; 0 shares one bias.</param>
    public unsafe void SetBias(nint bias, MatrixDataType? biasType = null, long batchStride = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        SetPointerAttr(DescBiasPointer, bias);
        if (biasType is { } t)
        {
            int dt = (int)t;
            CuBlasNative.MatmulDescSetAttribute(_matmulDesc, DescBiasDataType, &dt, sizeof(int));
        }

        if (batchStride != 0)
        {
            CuBlasNative.MatmulDescSetAttribute(_matmulDesc, DescBiasBatchStride, &batchStride, sizeof(long));
        }
    }

    /// <summary>
    /// Sets the auxiliary output written by the <c>*Aux</c> epilogs (<see cref="MatmulEpilog.GeluAux"/>,
    /// <see cref="MatmulEpilog.GeluAuxBias"/>, <see cref="MatmulEpilog.ReluAux"/>,
    /// <see cref="MatmulEpilog.ReluAuxBias"/>), used by the backward pass.
    /// </summary>
    /// <param name="aux">Device pointer to the aux buffer (16-byte aligned).</param>
    /// <param name="ld">
    /// Leading dimension in elements; for GELU a multiple of 8 and at least M. For ReLU the aux
    /// buffer is a bit mask and <paramref name="ld"/> counts bits (a multiple of 128).
    /// </param>
    /// <param name="auxType">Element type of the GELU aux output; <c>null</c> keeps the cuBLASLt default.</param>
    /// <param name="batchStride">Stride between batches in elements (bits for ReLU); 0 for non-batched.</param>
    public unsafe void SetEpilogAux(nint aux, long ld, MatrixDataType? auxType = null, long batchStride = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        SetPointerAttr(DescEpilogueAuxPointer, aux);
        CuBlasNative.MatmulDescSetAttribute(_matmulDesc, DescEpilogueAuxLd, &ld, sizeof(long));
        if (auxType is { } t)
        {
            int dt = (int)t;
            CuBlasNative.MatmulDescSetAttribute(_matmulDesc, DescEpilogueAuxDataType, &dt, sizeof(int));
        }

        if (batchStride != 0)
        {
            CuBlasNative.MatmulDescSetAttribute(_matmulDesc, DescEpilogueAuxBatchStride, &batchStride, sizeof(long));
        }
    }

    private unsafe void SetPointerAttr(int attr, nint value)
    {
        if (value == 0)
        {
            return;
        }

        void* p = (void*)value;
        CuBlasNative.MatmulDescSetAttribute(_matmulDesc, attr, &p, (nuint)sizeof(void*));
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
        if (_planned)
        {
            return;
        }

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
        if (_planned)
        {
            return;
        }

        DoPlan(preferences);
    }

    private unsafe void DoPlan(MatmulPlanPreferences? prefs)
    {
        // Apply optional workspace override from preferences
        if (prefs?.MaxWorkspaceBytes is nuint wsOverride)
        {
            ulong wsVal = wsOverride;
            CuBlasNative.MatmulPreferenceSetAttribute(
                _preference,
                PrefMaxWorkspaceBytes,
                &wsVal, (nuint)sizeof(ulong));
        }

        // Apply numerical implementation mask
        if (prefs is { NumericalImplMask: not 0 })
        {
            ulong mask = prefs.NumericalImplMask;
            CuBlasNative.MatmulPreferenceSetAttribute(
                _preference,
                PrefImplMask,
                &mask, sizeof(ulong));
        }

        int algoCount = prefs?.RequestedAlgoCount ?? 1;
        _algoBuffer = new byte[CuBlasNative.HeuristicResultSize * algoCount];
        int count = CuBlasNative.MatmulAlgoGetHeuristic(
            _ltHandle, _matmulDesc,
            _layoutA, _layoutB, _layoutC, _layoutD,
            _preference, algoCount, _algoBuffer);

        if (count == 0)
        {
            throw new InvalidOperationException("No cuBLASLt algorithm found for the given matmul configuration.");
        }

        // Keep every candidate (ordered best-first by the heuristic) so Autotune can time them.
        _algoCount = count;
        _allAlgos = _algoBuffer[..(CuBlasNative.HeuristicResultSize * count)];

        // One workspace large enough for any candidate.
        nuint maxWorkspace = 0;
        for (int i = 0; i < count; i++)
        {
            maxWorkspace = Math.Max(maxWorkspace, CandidateWorkspace(i));
        }

        if (maxWorkspace > 0)
        {
            _workspace = new Workspace(maxWorkspace);
            _workspacePtr = _workspace.PointerAsInt;
        }

        SelectAlgorithm(0);
        _planned = true;
    }

    // Workspace size sits at offset 64 of cublasLtMatmulHeuristicResult_t (after the 64-byte algo).
    private nuint CandidateWorkspace(int index) => (nuint)System.Runtime.InteropServices.MemoryMarshal
        .Read<ulong>(_allAlgos.AsSpan(index * CuBlasNative.HeuristicResultSize + 64));

    /// <summary>Number of candidate algorithms found by <see cref="Plan"/> (best-first).</summary>
    public int AlgorithmCount => _algoCount;

    /// <summary>Index of the algorithm that <see cref="Execute"/> uses (0 = heuristic's top choice).</summary>
    public int SelectedAlgorithm { get; private set; }

    /// <summary>
    /// Selects which planned candidate algorithm <see cref="Execute"/> uses.
    /// Request several candidates through <see cref="MatmulPlanPreferences.RequestedAlgoCount"/>.
    /// </summary>
    /// <param name="index">Candidate index in <c>[0, AlgorithmCount)</c>.</param>
    public void SelectAlgorithm(int index)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, _algoCount);

        _algoBuffer = _allAlgos.AsSpan(index * CuBlasNative.HeuristicResultSize, CuBlasNative.HeuristicResultSize).ToArray();
        _algoWorkspaceSize = CandidateWorkspace(index);
        SelectedAlgorithm = index;
    }

    // ── Autotune ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Times every candidate algorithm on the given operands and selects the fastest.
    /// <b>D is overwritten</b> by the trial runs (and so is C when it aliases D), so run this on
    /// scratch data or with <c>beta = 0</c> before the real computation.
    /// </summary>
    /// <param name="pA">Device pointer to A.</param>
    /// <param name="pB">Device pointer to B.</param>
    /// <param name="pC">Device pointer to C.</param>
    /// <param name="pD">Device pointer to D.</param>
    /// <param name="alpha">Scalar for the product.</param>
    /// <param name="beta">Scalar for C.</param>
    /// <param name="iterations">Timed runs per candidate (after one warm-up run).</param>
    /// <returns>Average milliseconds per candidate; <see cref="float.PositiveInfinity"/> for candidates that failed.</returns>
    public float[] Autotune(
        nint pA, nint pB, nint pC, nint pD,
        double alpha = 1.0, double beta = 0.0, int iterations = 3)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_planned)
        {
            throw new InvalidOperationException("Call Plan() or PlanAsync() before Autotune.");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(iterations, 1);

        float[] times = new float[_algoCount];
        using var start = new CudaEvent(disableTiming: false);
        using var stop = new CudaEvent(disableTiming: false);
        for (int i = 0; i < _algoCount; i++)
        {
            byte[] algo = _allAlgos.AsSpan(i * CuBlasNative.HeuristicResultSize, CuBlasNative.HeuristicResultSize).ToArray();
            nuint workspace = CandidateWorkspace(i);
            try
            {
                RunGemm(pA, pB, pC, pD, alpha, beta, algo, workspace); // warm-up
                _ownedStream.Synchronize();
                start.Record(_ownedStream);
                for (int it = 0; it < iterations; it++)
                {
                    RunGemm(pA, pB, pC, pD, alpha, beta, algo, workspace);
                }

                stop.Record(_ownedStream);
                _ownedStream.Synchronize();
                times[i] = stop.ElapsedMilliseconds(start) / iterations;
            }
            catch (CuBlasException)
            {
                times[i] = float.PositiveInfinity;
            }
        }

        SelectAlgorithm(Array.IndexOf(times, times.Min()));
        return times;
    }

    /// <summary>Asynchronous version of <see cref="Autotune"/>.</summary>
    public Task<float[]> AutotuneAsync(
        nint pA, nint pB, nint pC, nint pD,
        double alpha = 1.0, double beta = 0.0, int iterations = 3,
        CancellationToken ct = default) =>
        Task.Run(() => Autotune(pA, pB, pC, pD, alpha, beta, iterations), ct);

    // ── Execute ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Executes D = alpha*op(A)*op(B) + beta*C on device buffers.
    /// Prefer <see cref="ExecuteAsync"/> in async contexts.
    /// </summary>
    public void Execute(
        nint pA, nint pB, nint pC, nint pD,
        double alpha = 1.0, double beta = 0.0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_planned)
        {
            throw new InvalidOperationException("Call Plan() or PlanAsync() before Execute.");
        }

        RunGemm(pA, pB, pC, pD, alpha, beta, _algoBuffer!, _algoWorkspaceSize);
        if (_options.Blocking)
        {
            _ownedStream.Synchronize();
        }
    }

    /// <summary>
    /// Asynchronously executes the matrix multiplication and returns a
    /// <see cref="Task"/> that completes when the GPU kernel finishes.
    /// </summary>
    public async Task ExecuteAsync(
        nint pA, nint pB, nint pC, nint pD,
        double alpha = 1.0, double beta = 0.0,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_planned)
        {
            throw new InvalidOperationException("Call Plan() or PlanAsync() before Execute.");
        }

        RunGemm(pA, pB, pC, pD, alpha, beta, _algoBuffer!, _algoWorkspaceSize);
        if (_options.Blocking)
        {
            await _ownedStream.SynchronizeAsync(ct).ConfigureAwait(false);
        }
    }

    private unsafe void RunGemm(
        nint pA, nint pB, nint pC, nint pD, double alpha, double beta,
        byte[] algo, nuint workspaceSize)
    {
        // cuBLASLt reads alpha/beta with the scale type, so encode them accordingly.
        byte* a = stackalloc byte[8];
        byte* b = stackalloc byte[8];
        EncodeScalar(alpha, a);
        EncodeScalar(beta, b);

        fixed (byte* pAlgo = algo)
        {
            CuBlasNative.Matmul(
                _ltHandle,
                _matmulDesc,
                a,
                (void*)pA, _layoutA,
                (void*)pB, _layoutB,
                b,
                (void*)pC, _layoutC,
                (void*)pD, _layoutD,
                pAlgo,
                (void*)_workspacePtr, workspaceSize,
                _ownedStream.Handle);
        }
    }

    private unsafe void EncodeScalar(double value, byte* dst)
    {
        *(ulong*)dst = 0;
        switch (_options.ScaleType)
        {
            case MatrixDataType.Float64:
                *(double*)dst = value;
                break;
            case MatrixDataType.Float16:
                *(Half*)dst = (Half)value;
                break;
            case MatrixDataType.BFloat16:
                // bfloat16 is the upper 16 bits of an IEEE float32.
                *(ushort*)dst = (ushort)(BitConverter.SingleToUInt32Bits((float)value) >> 16);
                break;
            case MatrixDataType.Int32:
                *(int*)dst = (int)value;
                break;
            default:
                *(float*)dst = (float)value;
                break;
        }
    }

    /// <summary>Waits for all work on the internal stream to complete.</summary>
    public Task SynchronizeAsync(CancellationToken ct = default) =>
        _ownedStream.SynchronizeAsync(ct);

    // ── Convenience overloads ─────────────────────────────────────────────────

    /// <inheritdoc cref="ExecuteAsync(nint, nint, nint, nint, double, double, CancellationToken)"/>
    public Task ExecuteAsync<T>(
        DeviceBuffer<T> a, DeviceBuffer<T> b, DeviceBuffer<T> c, DeviceBuffer<T> d,
        double alpha = 1.0, double beta = 0.0,
        CancellationToken ct = default) where T : unmanaged, INumberBase<T>
        => ExecuteAsync(a.PointerAsInt, b.PointerAsInt, c.PointerAsInt, d.PointerAsInt, alpha, beta, ct);

    // ── Dispose ───────────────────────────────────────────────────────────────
    /// <inheritdoc/>
    public async ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return;
        }

        await _ownedStream.SynchronizeAsync().ConfigureAwait(false);
        DisposeCore();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        DisposeCore();
    }

    private unsafe void DisposeCore()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_preference != default)
        {
            CuBlasNative.MatmulPreferenceDestroy(_preference);
        }

        if (_layoutD    != default)
        {
            CuBlasNative.MatrixLayoutDestroy(_layoutD);
        }

        if (_layoutC    != default)
        {
            CuBlasNative.MatrixLayoutDestroy(_layoutC);
        }

        if (_layoutB    != default)
        {
            CuBlasNative.MatrixLayoutDestroy(_layoutB);
        }

        if (_layoutA    != default)
        {
            CuBlasNative.MatrixLayoutDestroy(_layoutA);
        }

        if (_matmulDesc != default)
        {
            CuBlasNative.MatmulDescDestroy(_matmulDesc);
        }

        if (_ltHandle   != default)
        {
            CuBlasNative.LtDestroy(_ltHandle);
        }

        if (_workspacePtr != 0)
        {
            _workspace.Dispose();
            _workspacePtr = 0;
        }

        if (_streamOwned)
        {
            _ownedStream.Dispose();
        }
    }
}
