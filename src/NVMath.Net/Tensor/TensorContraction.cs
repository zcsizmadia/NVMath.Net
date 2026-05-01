// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using System.Numerics;
using NVMathNet.Interop;

namespace NVMathNet.Tensor;

/// <summary>
/// Stateful cuTENSOR contraction wrapper.
/// Mirrors the design of <c>nvmath.tensor.TensorContraction</c>:
/// <list type="number">
///   <item><description>Construct with tensor descriptors.</description></item>
///   <item><description>Call <see cref="PlanAsync"/> to build the plan and select an algorithm.</description></item>
///   <item><description>Call <see cref="ExecuteAsync"/> to perform the contraction.</description></item>
///   <item><description>Dispose to release CUDA resources.</description></item>
/// </list>
/// Computes: D[modeD] = alpha * A[modeA] * B[modeB] + beta * C[modeC]
/// </summary>
public sealed class TensorContraction : IAsyncDisposable, IDisposable
{
    // ── cuTENSOR handles ───────────────────────────────────────────────────────

    private nint _handle;
    private nint _descA;
    private nint _descB;
    private nint _descC;
    private nint _descD;
    private nint _opDesc;
    private nint _find;
    private nint _plan;
    private bool _planned;
    private bool _disposed;

    // ── Workspace ─────────────────────────────────────────────────────────────

    private nint   _workspacePtr;
    private nuint  _workspaceSize;

    private readonly TensorContractionOptions _options;
    private readonly CudaStream _ownedStream;

    // ── Tensor descriptors (passed in by caller) ──────────────────────────────

    private readonly long[] _extentA, _extentB, _extentC, _extentD;
    private readonly int[]  _modeA,   _modeB,   _modeC,   _modeD;
    private CudaDataType    _typeA,   _typeB,   _typeC,   _typeD;
    private CuBlasComputeType _computeType;

    /// <summary>
    /// Creates a TensorContraction descriptor.
    /// </summary>
    /// <param name="extentA">Sizes of each dimension of A.</param>
    /// <param name="modeA">Mode labels for A (one per dimension).</param>
    /// <param name="extentB">Sizes of each dimension of B.</param>
    /// <param name="modeB">Mode labels for B.</param>
    /// <param name="extentC">Sizes of each dimension of C.</param>
    /// <param name="modeC">Mode labels for C.</param>
    /// <param name="extentD">Sizes of each dimension of D (output); often same as extentC.</param>
    /// <param name="modeD">Mode labels for D.</param>
    /// <param name="dataType">Element type for all tensors (default: float32).</param>
    /// <param name="options">Operation options; <c>null</c> uses defaults.</param>
    /// <param name="stream">Optional external stream.</param>
    public TensorContraction(
        long[] extentA, int[] modeA,
        long[] extentB, int[] modeB,
        long[] extentC, int[] modeC,
        long[] extentD, int[] modeD,
        TensorDataType dataType = TensorDataType.Float32,
        TensorContractionOptions? options = null,
        CudaStream? stream = null)
    {
        ArgumentNullException.ThrowIfNull(extentA); ArgumentNullException.ThrowIfNull(modeA);
        ArgumentNullException.ThrowIfNull(extentB); ArgumentNullException.ThrowIfNull(modeB);
        ArgumentNullException.ThrowIfNull(extentC); ArgumentNullException.ThrowIfNull(modeC);
        ArgumentNullException.ThrowIfNull(extentD); ArgumentNullException.ThrowIfNull(modeD);

        _extentA = (long[])extentA.Clone(); _modeA = (int[])modeA.Clone();
        _extentB = (long[])extentB.Clone(); _modeB = (int[])modeB.Clone();
        _extentC = (long[])extentC.Clone(); _modeC = (int[])modeC.Clone();
        _extentD = (long[])extentD.Clone(); _modeD = (int[])modeD.Clone();

        _typeA = _typeB = _typeC = _typeD = (CudaDataType)(int)dataType;
        _computeType = ResolveComputeType(dataType);
        _options = options ?? new TensorContractionOptions();
        _ownedStream = stream ?? new CudaStream();

        _handle = CuTensorNative.Create();

        _descA = CuTensorNative.CreateTensorDescriptor(_handle, (uint)_extentA.Length, _extentA, null, _typeA);
        _descB = CuTensorNative.CreateTensorDescriptor(_handle, (uint)_extentB.Length, _extentB, null, _typeB);
        _descC = CuTensorNative.CreateTensorDescriptor(_handle, (uint)_extentC.Length, _extentC, null, _typeC);
        _descD = CuTensorNative.CreateTensorDescriptor(_handle, (uint)_extentD.Length, _extentD, null, _typeD);

        _opDesc = CuTensorNative.CreateContractionDescriptor(
            _handle,
            _descA, _modeA, 128,
            _descB, _modeB, 128,
            _descC, _modeC, 128,
            _descD, _modeD, 128,
            _computeType);

        _find = CuTensorNative.CreateContractionFind(_handle, ResolveAlgo());
    }

    private CuTensorAlgo ResolveAlgo() =>
        _options.AutotuneMode switch
        {
            ContractionAutotuneMode.Auto      => CuTensorAlgo.DefaultPatient,
            ContractionAutotuneMode.Benchmark => CuTensorAlgo.Optimal,
            _                                 => (CuTensorAlgo)_options.Algorithm,
        };

    private static CuBlasComputeType ResolveComputeType(TensorDataType dt) =>
        dt switch
        {
            TensorDataType.Float16   => CuBlasComputeType.Compute16F,
            TensorDataType.Float32   => CuBlasComputeType.Compute32FFastTF32,
            TensorDataType.Float64   => CuBlasComputeType.Compute64F,
            TensorDataType.Complex64  => CuBlasComputeType.Compute32FFastTF32,
            TensorDataType.Complex128 => CuBlasComputeType.Compute64F,
            _ => CuBlasComputeType.Compute32FFastTF32
        };

    // ── Plan ──────────────────────────────────────────────────────────────────

    /// <summary>Selects the best contraction algorithm on a background thread.</summary>
    public async Task PlanAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_planned) return;
        await Task.Run(DoPlan, ct).ConfigureAwait(false);
    }

    /// <summary>Synchronously selects the best contraction algorithm.</summary>
    public void Plan()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_planned) return;
        DoPlan();
    }

    private unsafe void DoPlan()
    {
        _workspaceSize = CuTensorNative.GetWorkspaceSize(
            _handle, _opDesc, _find,
            (CuTensorWorksizePreference)_options.WorksizePreference);

        if (_workspaceSize > 0)
            _workspacePtr = (nint)CudaRuntime.Malloc(_workspaceSize);

        _plan = CuTensorNative.CreateContractionPlan(_handle, _opDesc, _find, _workspaceSize);
        _planned = true;
    }

    // ── Execute ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Asynchronously performs D = alpha*A*B + beta*C.
    /// Completes when the GPU kernel finishes if <see cref="TensorContractionOptions.Blocking"/> is <c>true</c>.
    /// </summary>
    public async Task ExecuteAsync(
        nint pA, nint pB, nint pC, nint pD,
        double alpha = 1.0, double beta = 0.0,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_planned)
            throw new InvalidOperationException("Call Plan() or PlanAsync() before Execute.");

        RunContraction(pA, pB, pC, pD, alpha, beta);
        if (_options.Blocking)
            await _ownedStream.SynchronizeAsync(ct).ConfigureAwait(false);
    }

    /// <summary>Synchronously performs the tensor contraction.</summary>
    public void Execute(
        nint pA, nint pB, nint pC, nint pD,
        double alpha = 1.0, double beta = 0.0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_planned)
            throw new InvalidOperationException("Call Plan() or PlanAsync() before Execute.");

        RunContraction(pA, pB, pC, pD, alpha, beta);
        if (_options.Blocking) _ownedStream.Synchronize();
    }

    /// <inheritdoc cref="ExecuteAsync(nint, nint, nint, nint, double, double, CancellationToken)"/>
    public Task ExecuteAsync<T>(
        DeviceBuffer<T> a, DeviceBuffer<T> b, DeviceBuffer<T> c, DeviceBuffer<T> d,
        double alpha = 1.0, double beta = 0.0,
        CancellationToken ct = default) where T : unmanaged, INumberBase<T>
        => ExecuteAsync(a.PointerAsInt, b.PointerAsInt, c.PointerAsInt, d.PointerAsInt, alpha, beta, ct);

    private unsafe void RunContraction(nint pA, nint pB, nint pC, nint pD, double alpha, double beta)
    {
        CuTensorNative.Contract(
            _handle, _plan,
            &alpha, (void*)pA, (void*)pB,
            &beta,  (void*)pC, (void*)pD,
            (void*)_workspacePtr, _workspaceSize,
            _ownedStream.Handle);
    }

    /// <summary>Waits for all work on the internal stream to complete.</summary>
    public Task SynchronizeAsync(CancellationToken ct = default) =>
        _ownedStream.SynchronizeAsync(ct);

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

        if (_plan    != default) CuTensorNative.DestroyContractionPlan(_plan);
        if (_find    != default) CuTensorNative.DestroyContractionFind(_find);
        if (_opDesc  != default) CuTensorNative.DestroyOperationDescriptor(_opDesc);
        if (_descD   != default) CuTensorNative.DestroyTensorDescriptor(_descD);
        if (_descC   != default) CuTensorNative.DestroyTensorDescriptor(_descC);
        if (_descB   != default) CuTensorNative.DestroyTensorDescriptor(_descB);
        if (_descA   != default) CuTensorNative.DestroyTensorDescriptor(_descA);
        if (_handle  != default) CuTensorNative.Destroy(_handle);

        if (_workspacePtr != 0)
        {
            CudaRuntime.Free((void*)_workspacePtr);
            _workspacePtr = 0;
        }

        _ownedStream.Dispose();
    }
}
