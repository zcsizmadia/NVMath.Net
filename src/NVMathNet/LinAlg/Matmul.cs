// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
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
///   <item><description>Call <see cref="ExecuteAsync(nint, nint, nint, nint, double, double, CancellationToken)"/> (possibly multiple times).</description></item>
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
    private readonly int _deviceId;
    private MatrixDataType? _biasType;
    private long _biasStride;
    private MatrixDataType? _auxType;
    private long _auxStride;
    private nint _biasPointer;
    private nint _auxPointer;
    private long _auxLd;
    private bool _hasScalePointers;
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
        ArgumentOutOfRangeException.ThrowIfLessThan(m, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(n, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(k, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(batchCount, 1);

        _m = m; _n = n; _k = k;
        _batchCount = batchCount;
        _options = options?.Clone() ?? new MatmulOptions();
        _deviceId = CudaRuntime.GetDevice();
        _ownedStream = stream ?? new CudaStream();
        _streamOwned = stream is null;

        try
        {
            _ltHandle = CuBlasNative.LtCreate();
            CreateDescriptors();
        }
        catch
        {
            DisposeCore(); // release whatever was created before the failure
            throw;
        }
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

        // Operations on A and B: none, transpose or conjugate transpose.
        int opA = _options.ConjugateTransposeA ? (int)CuBlasOperation.ConjTranspose
            : _options.TransposeA ? (int)CuBlasOperation.Transpose : (int)CuBlasOperation.None;
        int opB = _options.ConjugateTransposeB ? (int)CuBlasOperation.ConjTranspose
            : _options.TransposeB ? (int)CuBlasOperation.Transpose : (int)CuBlasOperation.None;
        bool transA = opA != (int)CuBlasOperation.None;
        bool transB = opB != (int)CuBlasOperation.None;
        if (transA)
        {
            CuBlasNative.MatmulDescSetAttribute(_matmulDesc, DescTransA, &opA, sizeof(int));
        }

        if (transB)
        {
            CuBlasNative.MatmulDescSetAttribute(_matmulDesc, DescTransB, &opB, sizeof(int));
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

        long ldA = transA ? _k : _m;
        long ldB = transB ? _n : _k;
        long ldC = _m;

        // Layout describes the physical (pre-transpose) matrix in memory.
        // When transposed, rows and cols of the stored matrix are swapped.
        _layoutA = transA
            ? CuBlasNative.MatrixLayoutCreate((CudaDataType)typeA, (ulong)_k, (ulong)_m, ldA)
            : CuBlasNative.MatrixLayoutCreate((CudaDataType)typeA, (ulong)_m, (ulong)_k, ldA);
        _layoutB = transB
            ? CuBlasNative.MatrixLayoutCreate((CudaDataType)typeB, (ulong)_n, (ulong)_k, ldB)
            : CuBlasNative.MatrixLayoutCreate((CudaDataType)typeB, (ulong)_k, (ulong)_n, ldB);
        _layoutC = CuBlasNative.MatrixLayoutCreate((CudaDataType)typeC, (ulong)_m, (ulong)_n, ldC);
        _layoutD = CuBlasNative.MatrixLayoutCreate((CudaDataType)typeD, (ulong)_m, (ulong)_n, ldC);

        if (_options.FastAccum)
        {
            sbyte fast = 1; // CUBLASLT_MATMUL_DESC_FAST_ACCUM is an int8_t
            CuBlasNative.MatmulDescSetAttribute(_matmulDesc, DescFastAccum, &fast, sizeof(sbyte));
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
    /// unchanged. Call before <see cref="Execute(nint, nint, nint, nint, double, double)"/>.
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
        _hasScalePointers = (aScale | bScale | cScale | dScale | amaxD) != 0;
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
        _biasPointer = bias;
        _biasType = biasType;
        _biasStride = batchStride;
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
        _auxPointer = aux;
        _auxLd = ld;
        _auxType = auxType;
        _auxStride = batchStride;
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

    // cuBLASLt handles are bound to the device that was current at creation, but planning may run on a thread-pool
    // thread whose current device differs, so every entry point re-selects the device first.
    private void EnsureDevice()
    {
        if (CudaRuntime.GetDevice() != _deviceId)
        {
            CudaRuntime.SetDevice(_deviceId);
        }
    }

    private unsafe void DoPlan(MatmulPlanPreferences? prefs)
    {
        EnsureDevice();
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
        if (algoCount < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(prefs), "RequestedAlgoCount must be at least 1.");
        }

        int count;
        int cachedSelection = 0;
        _cacheKey = MatmulPlanCache.Enabled ? BuildCacheKey(prefs, algoCount) : null;
        if (_cacheKey is not null && MatmulPlanCache.TryGet(_cacheKey, out var cached))
        {
            // Reuse the heuristic result (and any autotuned choice) of an identical earlier plan.
            _algoBuffer = new byte[CuBlasNative.HeuristicResultSize];
            _allAlgos = cached.Algorithms;
            count = cached.Count;
            cachedSelection = cached.Selected;
            _algoCount = count;
            NVMathLog.Debug("Matmul", () => $"Plan cache hit for {_m}x{_n}x{_k} (batch {_batchCount}); selected algorithm #{cachedSelection}.");
        }
        else
        {
            _algoBuffer = new byte[CuBlasNative.HeuristicResultSize * algoCount];
            count = CuBlasNative.MatmulAlgoGetHeuristic(
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
            if (_cacheKey is not null)
            {
                MatmulPlanCache.Store(_cacheKey, _allAlgos, count);
            }
        }

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

        NVMathLog.Debug("Matmul", () => $"Planned {_m}x{_n}x{_k} (batch {_batchCount}): {count} candidate algorithm(s), workspace {maxWorkspace} bytes.");
        SelectAlgorithm(cachedSelection);
        _planned = true;
    }

    private string? _cacheKey;

    private unsafe string BuildCacheKey(MatmulPlanPreferences? prefs, int requested)
    {
        // Anything that can change the heuristic's answer: shape, types, operations, epilog, alignment
        // of the epilog pointers and the planning preferences.
        var o = _options;
        string pointers = $"{(_biasPointer != 0 ? 1 : 0)}{(_biasPointer % 16 == 0 ? 1 : 0)}{(_auxPointer != 0 ? 1 : 0)}{(_auxPointer % 16 == 0 ? 1 : 0)}{(_hasScalePointers ? 1 : 0)}";
        return string.Join('|',
            _deviceId, _m, _n, _k, _batchCount,
            (int)o.ComputeType, (int)o.ScaleType, (int)o.TypeA, (int)o.TypeB, (int)o.TypeC, (int)o.TypeD,
            (int)o.Epilog, o.TransposeA, o.TransposeB, o.ConjugateTransposeA, o.ConjugateTransposeB, o.FastAccum,
            o.MaxWorkspaceBytes, prefs?.MaxWorkspaceBytes?.ToString() ?? "-", prefs?.NumericalImplMask ?? 0, requested,
            pointers, _auxLd, _biasType?.ToString() ?? "-", _biasStride, _auxType?.ToString() ?? "-", _auxStride);
    }

    // Workspace size sits at offset 64 of cublasLtMatmulHeuristicResult_t (after the 64-byte algo).
    private nuint CandidateWorkspace(int index) => (nuint)System.Runtime.InteropServices.MemoryMarshal
        .Read<ulong>(_allAlgos.AsSpan(index * CuBlasNative.HeuristicResultSize + 64));

    /// <summary>Number of candidate algorithms found by <see cref="Plan"/> (best-first).</summary>
    public int AlgorithmCount => _algoCount;

    /// <summary>Index of the algorithm that <see cref="Execute(nint, nint, nint, nint, double, double)"/> uses (0 = heuristic's top choice).</summary>
    public int SelectedAlgorithm { get; private set; }

    /// <summary>
    /// Selects which planned candidate algorithm <see cref="Execute(nint, nint, nint, nint, double, double)"/> uses.
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
    /// Times every candidate algorithm on the given operands and selects the fastest. Only the candidates found while
    /// planning are timed: plan with <see cref="MatmulPlanPreferences.RequestedAlgoCount"/> greater than 1, otherwise
    /// there is a single candidate and nothing to choose between.
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
        EnsureDevice();

        float[] times = new float[_algoCount];
        using var start = new CudaEvent(disableTiming: false);
        using var stop = new CudaEvent(disableTiming: false);
        for (int i = 0; i < _algoCount; i++)
        {
            byte[] algo = _allAlgos.AsSpan(i * CuBlasNative.HeuristicResultSize, CuBlasNative.HeuristicResultSize).ToArray();
            nuint workspace = CandidateWorkspace(i);
            try
            {
                RunGemm(pA, pB, pC, pD, new Complex(alpha, 0), new Complex(beta, 0), algo, workspace); // warm-up
                _ownedStream.Synchronize();
                start.Record(_ownedStream);
                for (int it = 0; it < iterations; it++)
                {
                    RunGemm(pA, pB, pC, pD, new Complex(alpha, 0), new Complex(beta, 0), algo, workspace);
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
        if (_cacheKey is not null)
        {
            MatmulPlanCache.UpdateSelected(_cacheKey, SelectedAlgorithm);
        }

        NVMathLog.Info("Matmul", () => $"Autotuned {_algoCount} candidate(s) over {iterations} iteration(s); selected #{SelectedAlgorithm} ({times[SelectedAlgorithm]:F3} ms): [{string.Join(", ", times.Select(t => t.ToString("F3")))}] ms.");
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
    /// Prefer <see cref="ExecuteAsync(nint, nint, nint, nint, double, double, CancellationToken)"/> in async contexts.
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

        RunGemm(pA, pB, pC, pD, new Complex(alpha, 0), new Complex(beta, 0), _algoBuffer!, _algoWorkspaceSize);
        if (_options.Blocking)
        {
            _ownedStream.Synchronize();
        }
    }

    /// <summary>
    /// Asynchronously executes the matrix multiplication and returns a
    /// <see cref="Task"/> that completes once the work has been submitted, or once it has finished when
    /// <see cref="MatmulOptions.Blocking"/> is set.
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

        RunGemm(pA, pB, pC, pD, new Complex(alpha, 0), new Complex(beta, 0), _algoBuffer!, _algoWorkspaceSize);
        if (_options.Blocking)
        {
            await _ownedStream.SynchronizeAsync(ct).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Executes D = alpha*op(A)*op(B) + beta*C with complex scalars (for complex data types and
    /// <see cref="MatrixDataType.Complex64"/> / <see cref="MatrixDataType.Complex128"/> scale types).
    /// For real scale types only the real part is used.
    /// </summary>
    public void Execute(nint pA, nint pB, nint pC, nint pD, Complex alpha, Complex beta)
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

    /// <inheritdoc cref="Execute(nint, nint, nint, nint, Complex, Complex)"/>
    public async Task ExecuteAsync(
        nint pA, nint pB, nint pC, nint pD, Complex alpha, Complex beta, CancellationToken ct = default)
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

    /// <inheritdoc cref="Execute(nint, nint, nint, nint, Complex, Complex)"/>
    public Task ExecuteAsync<T>(
        DeviceBuffer<T> a, DeviceBuffer<T> b, DeviceBuffer<T> c, DeviceBuffer<T> d,
        Complex alpha, Complex beta, CancellationToken ct = default) where T : unmanaged, INumberBase<T>
        => ExecuteAsync(a.PointerAsInt, b.PointerAsInt, c.PointerAsInt, d.PointerAsInt, alpha, beta, ct);

    private unsafe void RunGemm(
        nint pA, nint pB, nint pC, nint pD, Complex alpha, Complex beta,
        byte[] algo, nuint workspaceSize)
    {
        EnsureDevice();

        // cuBLASLt reads alpha/beta with the scale type, so encode them accordingly. Complex scale types
        // (cuDoubleComplex) must be 16-byte aligned.
        Span<byte> aStore = stackalloc byte[AlignedScalar.StorageSize], bStore = stackalloc byte[AlignedScalar.StorageSize];
        byte* a = AlignedScalar.Store(aStore, (byte)0);
        byte* b = AlignedScalar.Store(bStore, (byte)0);
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

    private unsafe void EncodeScalar(Complex scalar, byte* dst)
    {
        *(ulong*)dst = 0;
        *((ulong*)dst + 1) = 0;
        double value = scalar.Real;
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
            case MatrixDataType.Complex64:
                *(float*)dst = (float)scalar.Real;
                *((float*)dst + 1) = (float)scalar.Imaginary;
                break;
            case MatrixDataType.Complex128:
                *(double*)dst = scalar.Real;
                *((double*)dst + 1) = scalar.Imaginary;
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

        // Work may still be queued on the stream (Blocking = false): wait for it before the workspace is returned
        // to its memory manager, which may hand the block to someone else immediately.
        try
        {
            _ownedStream.Synchronize();
        }
        catch (Exception)
        {
            // The context may already be broken; continue releasing resources.
        }

        // One failing destroy call must not leak everything that comes after it.
        static void Try(Action destroy)
        {
            try
            {
                destroy();
            }
            catch (Exception)
            {
                // Already reported through NVMathLog by the native Check helper.
            }
        }

        if (_preference != default)
        {
            Try(() => CuBlasNative.MatmulPreferenceDestroy(_preference));
        }

        if (_layoutD != default)
        {
            Try(() => CuBlasNative.MatrixLayoutDestroy(_layoutD));
        }

        if (_layoutC != default)
        {
            Try(() => CuBlasNative.MatrixLayoutDestroy(_layoutC));
        }

        if (_layoutB != default)
        {
            Try(() => CuBlasNative.MatrixLayoutDestroy(_layoutB));
        }

        if (_layoutA != default)
        {
            Try(() => CuBlasNative.MatrixLayoutDestroy(_layoutA));
        }

        if (_matmulDesc != default)
        {
            Try(() => CuBlasNative.MatmulDescDestroy(_matmulDesc));
        }

        if (_ltHandle != default)
        {
            Try(() => CuBlasNative.LtDestroy(_ltHandle));
        }

        if (_workspacePtr != 0)
        {
            Try(() => _workspace.Dispose());
            _workspacePtr = 0;
        }

        if (_streamOwned)
        {
            Try(() => _ownedStream.Dispose());
        }
    }
}
