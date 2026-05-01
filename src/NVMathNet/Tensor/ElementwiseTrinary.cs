// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Interop;

namespace NVMathNet.Tensor;

/// <summary>
/// Stateful cuTENSOR element-wise trinary operation wrapper.
/// Computes: D[modeD] = opABC(alpha * opA(A)[modeA], opAB(beta * opB(B)[modeB], gamma * opC(C)[modeC]))
/// <para>
/// Common use-cases include fused add/multiply of three same-shape tensors,
/// e.g. D = alpha * A + beta * B + gamma * C (using <see cref="CuTensorOperator.Add"/> for both binary ops).
/// </para>
/// Workflow:
/// <list type="number">
///   <item><description>Construct with tensor descriptors and operator choices.</description></item>
///   <item><description>Call <see cref="ExecuteAsync"/> to perform the operation.</description></item>
///   <item><description>Dispose (or use <c>await using</c>) to free CUDA resources.</description></item>
/// </list>
/// </summary>
public sealed class ElementwiseTrinary : IAsyncDisposable, IDisposable
{
    // ── cuTENSOR handles ───────────────────────────────────────────────────────

    private readonly nint _handle;
    private readonly nint _descA, _descB, _descC, _descD;
    private readonly nint _opDesc;
    private readonly nint _plan;
    private bool _disposed;

    private readonly CudaStream _ownedStream;
    private readonly bool _streamOwned;
    private readonly TensorContractionOptions _options;

    /// <summary>
    /// Creates an element-wise trinary operation descriptor.
    /// </summary>
    /// <param name="extentA">Sizes of each dimension of A.</param>
    /// <param name="modeA">Mode labels for A.</param>
    /// <param name="extentB">Sizes of each dimension of B.</param>
    /// <param name="modeB">Mode labels for B.</param>
    /// <param name="extentC">Sizes of each dimension of C.</param>
    /// <param name="modeC">Mode labels for C.</param>
    /// <param name="extentD">Sizes of each dimension of D (output).</param>
    /// <param name="modeD">Mode labels for D.</param>
    /// <param name="dataType">Element type for all tensors. Default: <see cref="TensorDataType.Float32"/>.</param>
    /// <param name="opA">Unary operator applied to A before scaling. Default: identity.</param>
    /// <param name="opB">Unary operator applied to B before scaling. Default: identity.</param>
    /// <param name="opC">Unary operator applied to C before scaling. Default: identity.</param>
    /// <param name="opAB">
    /// Binary operator combining alpha*opA(A) and beta*opB(B).
    /// Default: <see cref="CuTensorOperator.Add"/>.
    /// </param>
    /// <param name="opABC">
    /// Binary operator combining the prior result with gamma*opC(C).
    /// Default: <see cref="CuTensorOperator.Add"/>.
    /// </param>
    /// <param name="options">Operation options; <c>null</c> uses defaults.</param>
    /// <param name="stream">Optional external stream.</param>
    public ElementwiseTrinary(
        long[] extentA, int[] modeA,
        long[] extentB, int[] modeB,
        long[] extentC, int[] modeC,
        long[] extentD, int[] modeD,
        TensorDataType dataType = TensorDataType.Float32,
        CuTensorOperator opA   = CuTensorOperator.Identity,
        CuTensorOperator opB   = CuTensorOperator.Identity,
        CuTensorOperator opC   = CuTensorOperator.Identity,
        CuTensorOperator opAB  = CuTensorOperator.Add,
        CuTensorOperator opABC = CuTensorOperator.Add,
        TensorContractionOptions? options = null,
        CudaStream? stream = null)
    {
        ArgumentNullException.ThrowIfNull(extentA); ArgumentNullException.ThrowIfNull(modeA);
        ArgumentNullException.ThrowIfNull(extentB); ArgumentNullException.ThrowIfNull(modeB);
        ArgumentNullException.ThrowIfNull(extentC); ArgumentNullException.ThrowIfNull(modeC);
        ArgumentNullException.ThrowIfNull(extentD); ArgumentNullException.ThrowIfNull(modeD);

        _options = options ?? new TensorContractionOptions();
        _ownedStream = stream ?? new CudaStream();
        _streamOwned = stream is null;

        var nativeType = (CudaDataType)(int)dataType;

        _handle = CuTensorNative.Create();

        _descA = CuTensorNative.CreateTensorDescriptor(_handle, (uint)extentA.Length, extentA, null, nativeType);
        _descB = CuTensorNative.CreateTensorDescriptor(_handle, (uint)extentB.Length, extentB, null, nativeType);
        _descC = CuTensorNative.CreateTensorDescriptor(_handle, (uint)extentC.Length, extentC, null, nativeType);
        _descD = CuTensorNative.CreateTensorDescriptor(_handle, (uint)extentD.Length, extentD, null, nativeType);

        _opDesc = CuTensorNative.CreateElementwiseTrinary(
            _handle,
            _descA, (int[])modeA.Clone(), opA,
            _descB, (int[])modeB.Clone(), opB,
            _descC, (int[])modeC.Clone(), opC,
            _descD, (int[])modeD.Clone(),
            opAB, opABC);

        // v2 requires a plan for execution
        var planPref = CuTensorNative.CreatePlanPreference(_handle);
        try
        {
            ulong wsSize = CuTensorNative.EstimateWorkspaceSize(_handle, _opDesc, planPref);
            _plan = CuTensorNative.CreatePlan(_handle, _opDesc, planPref, wsSize);
        }
        finally
        {
            CuTensorNative.DestroyPlanPreference(planPref);
        }
    }

    // ── Execute ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Asynchronously computes D = opABC(alpha*opA(A), opAB(beta*opB(B), gamma*opC(C))).
    /// </summary>
    /// <param name="pA">Device pointer to tensor A.</param>
    /// <param name="pB">Device pointer to tensor B.</param>
    /// <param name="pC">Device pointer to tensor C.</param>
    /// <param name="pD">Device pointer to output tensor D.</param>
    /// <param name="alpha">Scalar multiplier for A. Default: 1.</param>
    /// <param name="beta">Scalar multiplier for B. Default: 1.</param>
    /// <param name="gamma">Scalar multiplier for C. Default: 1.</param>
    /// <param name="ct">Cancellation token.</param>
    public async Task ExecuteAsync(
        nint pA, nint pB, nint pC, nint pD,
        double alpha = 1.0, double beta = 1.0, double gamma = 1.0,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        RunEwTrinary(pA, pB, pC, pD, alpha, beta, gamma);
        if (_options.Blocking)
        {
            await _ownedStream.SynchronizeAsync(ct).ConfigureAwait(false);
        }
    }

    /// <summary>Synchronously computes the element-wise trinary operation.</summary>
    /// <inheritdoc cref="ExecuteAsync(nint, nint, nint, nint, double, double, double, CancellationToken)"/>
    public void Execute(
        nint pA, nint pB, nint pC, nint pD,
        double alpha = 1.0, double beta = 1.0, double gamma = 1.0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        RunEwTrinary(pA, pB, pC, pD, alpha, beta, gamma);
        if (_options.Blocking)
        {
            _ownedStream.Synchronize();
        }
    }

    /// <inheritdoc cref="ExecuteAsync(nint, nint, nint, nint, double, double, double, CancellationToken)"/>
    public Task ExecuteAsync<T>(
        DeviceBuffer<T> a, DeviceBuffer<T> b, DeviceBuffer<T> c, DeviceBuffer<T> d,
        double alpha = 1.0, double beta = 1.0, double gamma = 1.0,
        CancellationToken ct = default) where T : unmanaged, INumberBase<T>
        => ExecuteAsync(a.PointerAsInt, b.PointerAsInt, c.PointerAsInt, d.PointerAsInt, alpha, beta, gamma, ct);

    private unsafe void RunEwTrinary(nint pA, nint pB, nint pC, nint pD, double alpha, double beta, double gamma)
    {
        float fa = (float)alpha, fb = (float)beta, fg = (float)gamma;
        CuTensorNative.ElementwiseTrinaryExecute(
            _handle, _plan,
            &fa, (void*)pA,
            &fb, (void*)pB,
            &fg, (void*)pC,
            (void*)pD, _ownedStream.Handle);
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

    private void DisposeCore()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_plan   != default)
        {
            CuTensorNative.DestroyPlan(_plan);
        }

        if (_opDesc != default)
        {
            CuTensorNative.DestroyOperationDescriptor(_opDesc);
        }

        if (_descD  != default)
        {
            CuTensorNative.DestroyTensorDescriptor(_descD);
        }

        if (_descC  != default)
        {
            CuTensorNative.DestroyTensorDescriptor(_descC);
        }

        if (_descB  != default)
        {
            CuTensorNative.DestroyTensorDescriptor(_descB);
        }

        if (_descA  != default)
        {
            CuTensorNative.DestroyTensorDescriptor(_descA);
        }

        if (_handle != default)
        {
            CuTensorNative.Destroy(_handle);
        }

        if (_streamOwned)
        {
            _ownedStream.Dispose();
        }
    }
}
