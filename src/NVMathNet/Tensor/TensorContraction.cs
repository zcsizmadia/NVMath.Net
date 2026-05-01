// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

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

    private readonly nint _handle;
    private readonly nint _descA;
    private readonly nint _descB;
    private readonly nint _descC;
    private readonly nint _descD;
    private readonly nint _opDesc;
    private readonly nint _planPref;
    private nint _plan;
    private bool _planned;
    private bool _disposed;

    // ── Workspace ─────────────────────────────────────────────────────────────

    private nint   _workspacePtr;
    private ulong  _workspaceSize;

    private readonly TensorContractionOptions _options;
    private readonly CudaStream _ownedStream;
    private readonly bool _streamOwned;

    // ── Tensor descriptors (passed in by caller) ──────────────────────────────

    private readonly long[] _extentA, _extentB, _extentC, _extentD;
    private readonly int[]  _modeA,   _modeB,   _modeC,   _modeD;
    private readonly CudaDataType    _typeA,   _typeB,   _typeC,   _typeD;
    private readonly nint            _computeDesc;

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
        _computeDesc = ResolveComputeDesc(dataType);
        _options = options ?? new TensorContractionOptions();
        _ownedStream = stream ?? new CudaStream();
        _streamOwned = stream is null;

        _handle = CuTensorNative.Create();

        _descA = CuTensorNative.CreateTensorDescriptor(_handle, (uint)_extentA.Length, _extentA, null, _typeA);
        _descB = CuTensorNative.CreateTensorDescriptor(_handle, (uint)_extentB.Length, _extentB, null, _typeB);
        _descC = CuTensorNative.CreateTensorDescriptor(_handle, (uint)_extentC.Length, _extentC, null, _typeC);
        _descD = CuTensorNative.CreateTensorDescriptor(_handle, (uint)_extentD.Length, _extentD, null, _typeD);

        _opDesc = CuTensorNative.CreateContraction(
            _handle,
            _descA, _modeA, CuTensorOperator.Identity,
            _descB, _modeB, CuTensorOperator.Identity,
            _descC, _modeC, CuTensorOperator.Identity,
            _descD, _modeD,
            _computeDesc);

        _planPref = CuTensorNative.CreatePlanPreference(_handle, ResolveAlgo());
    }

    private CuTensorAlgo ResolveAlgo() =>
        _options.AutotuneMode switch
        {
            ContractionAutotuneMode.Auto      => CuTensorAlgo.DefaultPatient,
            ContractionAutotuneMode.Benchmark => CuTensorAlgo.DefaultPatient,
            _                                 => (CuTensorAlgo)_options.Algorithm,
        };

    private static nint ResolveComputeDesc(TensorDataType dt) =>
        dt switch
        {
            TensorDataType.Float16    => CuTensorNative.ComputeDesc16F,
            TensorDataType.Float32    => CuTensorNative.ComputeDesc32F,
            TensorDataType.Float64    => CuTensorNative.ComputeDesc64F,
            TensorDataType.Complex64  => CuTensorNative.ComputeDesc32F,
            TensorDataType.Complex128 => CuTensorNative.ComputeDesc64F,
            _ => CuTensorNative.ComputeDesc32F
        };

    private bool IsDoubleCompute =>
        _computeDesc == CuTensorNative.ComputeDesc64F;

    // ── Plan ──────────────────────────────────────────────────────────────────

    /// <summary>Selects the best contraction algorithm on a background thread.</summary>
    public async Task PlanAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_planned)
        {
            return;
        }

        await Task.Run(DoPlan, ct).ConfigureAwait(false);
    }

    /// <summary>Synchronously selects the best contraction algorithm.</summary>
    public void Plan()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_planned)
        {
            return;
        }

        DoPlan();
    }

    private unsafe void DoPlan()
    {
        _workspaceSize = CuTensorNative.EstimateWorkspaceSize(
            _handle, _opDesc, _planPref,
            (CuTensorWorksizePreference)_options.WorksizePreference);

        if (_workspaceSize > 0)
        {
            _workspacePtr = (nint)CudaRuntime.Malloc((nuint)_workspaceSize);
        }

        _plan = CuTensorNative.CreatePlan(_handle, _opDesc, _planPref, _workspaceSize);
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
        {
            throw new InvalidOperationException("Call Plan() or PlanAsync() before Execute.");
        }

        RunContraction(pA, pB, pC, pD, alpha, beta);
        if (_options.Blocking)
        {
            await _ownedStream.SynchronizeAsync(ct).ConfigureAwait(false);
        }
    }

    /// <summary>Synchronously performs the tensor contraction.</summary>
    public void Execute(
        nint pA, nint pB, nint pC, nint pD,
        double alpha = 1.0, double beta = 0.0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_planned)
        {
            throw new InvalidOperationException("Call Plan() or PlanAsync() before Execute.");
        }

        RunContraction(pA, pB, pC, pD, alpha, beta);
        if (_options.Blocking)
        {
            _ownedStream.Synchronize();
        }
    }

    /// <inheritdoc cref="ExecuteAsync(nint, nint, nint, nint, double, double, CancellationToken)"/>
    public Task ExecuteAsync<T>(
        DeviceBuffer<T> a, DeviceBuffer<T> b, DeviceBuffer<T> c, DeviceBuffer<T> d,
        double alpha = 1.0, double beta = 0.0,
        CancellationToken ct = default) where T : unmanaged, INumberBase<T>
        => ExecuteAsync(a.PointerAsInt, b.PointerAsInt, c.PointerAsInt, d.PointerAsInt, alpha, beta, ct);

    private unsafe void RunContraction(nint pA, nint pB, nint pC, nint pD, double alpha, double beta)
    {
        // cuTENSOR expects the scalar type to match the compute type.
        // For 64-bit compute, pass double*; for 32-bit, pass float*.
        if (IsDoubleCompute)
        {
            CuTensorNative.Contract(
                _handle, _plan,
                &alpha, (void*)pA, (void*)pB,
                &beta,  (void*)pC, (void*)pD,
                (void*)_workspacePtr, _workspaceSize,
                _ownedStream.Handle);
        }
        else
        {
            float fa = (float)alpha, fb = (float)beta;
            CuTensorNative.Contract(
                _handle, _plan,
                &fa, (void*)pA, (void*)pB,
                &fb, (void*)pC, (void*)pD,
                (void*)_workspacePtr, _workspaceSize,
                _ownedStream.Handle);
        }
    }

    /// <summary>Waits for all work on the internal stream to complete.</summary>
    public Task SynchronizeAsync(CancellationToken ct = default) =>
        _ownedStream.SynchronizeAsync(ct);

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

        if (_plan    != default)
        {
            CuTensorNative.DestroyPlan(_plan);
        }

        if (_planPref != default)
        {
            CuTensorNative.DestroyPlanPreference(_planPref);
        }

        if (_opDesc  != default)
        {
            CuTensorNative.DestroyOperationDescriptor(_opDesc);
        }

        if (_descD   != default)
        {
            CuTensorNative.DestroyTensorDescriptor(_descD);
        }

        if (_descC   != default)
        {
            CuTensorNative.DestroyTensorDescriptor(_descC);
        }

        if (_descB   != default)
        {
            CuTensorNative.DestroyTensorDescriptor(_descB);
        }

        if (_descA   != default)
        {
            CuTensorNative.DestroyTensorDescriptor(_descA);
        }

        if (_handle  != default)
        {
            CuTensorNative.Destroy(_handle);
        }

        if (_workspacePtr != 0)
        {
            CudaRuntime.Free((void*)_workspacePtr);
            _workspacePtr = 0;
        }

        if (_streamOwned)
        {
            _ownedStream.Dispose();
        }
    }
}
