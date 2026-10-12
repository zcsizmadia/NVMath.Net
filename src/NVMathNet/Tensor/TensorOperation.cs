// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Interop;
using NVMathNet.Memory;

namespace NVMathNet.Tensor;

/// <summary>
/// Common plumbing for the single-plan cuTENSOR operations (permutation, element-wise binary,
/// reduction): handle, stream, tensor descriptors, plan, workspace and disposal.
/// Tensors use a packed <b>column-major</b> layout: the first mode varies fastest in memory.
/// </summary>
public abstract class TensorOperation : IAsyncDisposable, IDisposable
{
    private readonly List<nint> _descriptors = [];
    private readonly bool _streamOwned;
    private nint _opDesc;
    private nint _plan;
    private Workspace _workspace;
    private bool _disposed;

    /// <summary>The cuTENSOR handle.</summary>
    internal nint Handle { get; }

    /// <summary>The stream the operation runs on.</summary>
    internal CudaStream Stream { get; }

    /// <summary>The compute descriptor chosen from the data type.</summary>
    internal nint ComputeDesc { get; }

    /// <summary>The CUDA data type of all tensors.</summary>
    internal CudaDataType NativeType { get; }

    /// <summary>The execution plan; valid once <see cref="Build"/> has run.</summary>
    internal nint Plan => _plan;

    /// <summary>Workspace size in bytes required by the plan.</summary>
    internal ulong WorkspaceSize { get; private set; }

    /// <summary>Pointer to the workspace (0 when none is needed).</summary>
    internal nint WorkspacePointer => _workspace.PointerAsInt;

    private readonly bool _blocking;

    internal TensorOperation(TensorDataType dataType, TensorContractionOptions? options, CudaStream? stream)
    {
        _blocking = (options ?? new TensorContractionOptions()).Blocking;
        _streamOwned = stream is null;
        Stream = stream ?? new CudaStream();
        NativeType = (CudaDataType)(int)dataType;
        ComputeDesc = TensorContraction.ResolveComputeDesc(dataType);
        Handle = CuTensorNative.Create();
    }

    /// <summary>Creates a packed tensor descriptor owned (and later destroyed) by this operation.</summary>
    internal nint NewDescriptor(long[] extent)
    {
        ArgumentNullException.ThrowIfNull(extent);
        nint desc = CuTensorNative.CreateTensorDescriptor(Handle, (uint)extent.Length, extent, null, NativeType);
        _descriptors.Add(desc);
        return desc;
    }

    /// <summary>Creates the execution plan (and workspace) for <paramref name="opDesc"/>.</summary>
    internal void Build(nint opDesc)
    {
        _opDesc = opDesc;
        nint pref = CuTensorNative.CreatePlanPreference(Handle);
        try
        {
            WorkspaceSize = CuTensorNative.EstimateWorkspaceSize(Handle, opDesc, pref);
            _plan = CuTensorNative.CreatePlan(Handle, opDesc, pref, WorkspaceSize);
        }
        finally
        {
            CuTensorNative.DestroyPlanPreference(pref);
        }

        _workspace = new Workspace((nuint)WorkspaceSize);
        NVMathLog.Debug("Tensor", () => $"{GetType().Name} planned, workspace {WorkspaceSize} bytes.");
    }

    /// <summary>Validates that mode labels are unique and match the tensor rank.</summary>
    internal static void ValidateModes(long[] extent, int[] modes, string name)
    {
        ArgumentNullException.ThrowIfNull(extent);
        ArgumentNullException.ThrowIfNull(modes);
        if (extent.Length != modes.Length)
        {
            throw new ArgumentException($"{name}: extent and mode arrays must have the same length.");
        }

        if (modes.Distinct().Count() != modes.Length)
        {
            throw new ArgumentException($"{name}: mode labels must be unique.");
        }
    }

    internal void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    /// <summary>Synchronises when the operation was created with <c>Blocking = true</c>.</summary>
    internal void AfterLaunch()
    {
        if (_blocking)
        {
            Stream.Synchronize();
        }
    }

    internal async Task AfterLaunchAsync(CancellationToken ct)
    {
        if (_blocking)
        {
            await Stream.SynchronizeAsync(ct).ConfigureAwait(false);
        }
    }

    /// <summary>Waits for all work on the internal stream to complete.</summary>
    public Task SynchronizeAsync(CancellationToken ct = default) => Stream.SynchronizeAsync(ct);

    /// <inheritdoc/>
    public async ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return;
        }

        await Stream.SynchronizeAsync().ConfigureAwait(false);
        Dispose();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (_plan != 0)
        {
            CuTensorNative.DestroyPlan(_plan);
        }

        if (_opDesc != 0)
        {
            CuTensorNative.DestroyOperationDescriptor(_opDesc);
        }

        foreach (nint d in _descriptors)
        {
            CuTensorNative.DestroyTensorDescriptor(d);
        }

        _workspace.Dispose();

        if (Handle != 0)
        {
            CuTensorNative.Destroy(Handle);
        }

        if (_streamOwned)
        {
            Stream.Dispose();
        }
    }
}

/// <summary>
/// Tensor permutation (generalised transpose): <c>B[modeB] = alpha · opA(A[modeA])</c>.
/// The B mode labels must be a reordering of the A mode labels.
/// </summary>
public sealed class TensorPermutation : TensorOperation
{
    /// <summary>Creates a permutation operation.</summary>
    /// <param name="extentA">Sizes of the dimensions of A.</param>
    /// <param name="modeA">Mode labels of A.</param>
    /// <param name="modeB">Mode labels of B: the same labels as <paramref name="modeA"/> in the desired order.</param>
    /// <param name="dataType">Element type of both tensors. Default: float.</param>
    /// <param name="opA">Unary operator applied to each element of A. Default: identity.</param>
    /// <param name="options">Options; <c>null</c> uses defaults.</param>
    /// <param name="stream">Optional external stream.</param>
    public TensorPermutation(
        long[] extentA, int[] modeA, int[] modeB,
        TensorDataType dataType = TensorDataType.Float32,
        CuTensorOperator opA = CuTensorOperator.Identity,
        TensorContractionOptions? options = null,
        CudaStream? stream = null)
        : base(dataType, options, stream)
    {
        try
        {
            ValidateModes(extentA, modeA, nameof(modeA));
            ArgumentNullException.ThrowIfNull(modeB);
            if (modeB.Length != modeA.Length || modeB.Except(modeA).Any())
            {
                throw new ArgumentException("modeB must be a reordering of modeA.", nameof(modeB));
            }

            long[] extentB = [.. modeB.Select(m => extentA[Array.IndexOf(modeA, m)])];
            nint descA = NewDescriptor(extentA);
            nint descB = NewDescriptor(extentB);
            Build(CuTensorNative.CreatePermutation(Handle, descA, (int[])modeA.Clone(), opA, descB, (int[])modeB.Clone(), ComputeDesc));
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>Executes <c>B = alpha · opA(permute(A))</c> on device pointers.</summary>
    public unsafe void Execute(nint pA, nint pB, double alpha = 1.0)
    {
        ThrowIfDisposed();
        byte* a = stackalloc byte[8];
        TensorScalars.Encode(alpha, ComputeDesc, a);
        CuTensorNative.Permute(Handle, Plan, a, (void*)pA, (void*)pB, Stream.Handle);
        AfterLaunch();
    }

    /// <summary>Asynchronous version of <see cref="Execute"/>; completes when the GPU work is done if <c>Blocking</c> is set.</summary>
    public Task ExecuteAsync(nint pA, nint pB, double alpha = 1.0, CancellationToken ct = default)
    {
        Execute(pA, pB, alpha);
        return Task.CompletedTask;
    }

    /// <inheritdoc cref="Execute"/>
    public Task ExecuteAsync<T>(DeviceBuffer<T> a, DeviceBuffer<T> b, double alpha = 1.0, CancellationToken ct = default)
        where T : unmanaged, INumberBase<T>
        => ExecuteAsync(a.PointerAsInt, b.PointerAsInt, alpha, ct);
}

/// <summary>
/// Element-wise binary operation: <c>D[modeD] = opAC(alpha · opA(A[modeA]), gamma · opC(C[modeC]))</c>.
/// </summary>
public sealed class ElementwiseBinary : TensorOperation
{
    /// <summary>Creates an element-wise binary operation.</summary>
    /// <param name="extentA">Sizes of the dimensions of A.</param>
    /// <param name="modeA">Mode labels of A.</param>
    /// <param name="extentC">Sizes of the dimensions of C.</param>
    /// <param name="modeC">Mode labels of C; must equal <paramref name="modeD"/>.</param>
    /// <param name="extentD">Sizes of the dimensions of D (output).</param>
    /// <param name="modeD">Mode labels of D.</param>
    /// <param name="dataType">Element type of all tensors. Default: float.</param>
    /// <param name="opA">Unary operator applied to A. Default: identity.</param>
    /// <param name="opC">Unary operator applied to C. Default: identity.</param>
    /// <param name="opAC">Binary operator combining the two terms. Default: add.</param>
    /// <param name="options">Options; <c>null</c> uses defaults.</param>
    /// <param name="stream">Optional external stream.</param>
    public ElementwiseBinary(
        long[] extentA, int[] modeA,
        long[] extentC, int[] modeC,
        long[] extentD, int[] modeD,
        TensorDataType dataType = TensorDataType.Float32,
        CuTensorOperator opA = CuTensorOperator.Identity,
        CuTensorOperator opC = CuTensorOperator.Identity,
        CuTensorOperator opAC = CuTensorOperator.Add,
        TensorContractionOptions? options = null,
        CudaStream? stream = null)
        : base(dataType, options, stream)
    {
        try
        {
            ValidateModes(extentA, modeA, nameof(modeA));
            ValidateModes(extentC, modeC, nameof(modeC));
            ValidateModes(extentD, modeD, nameof(modeD));
            if (!modeC.SequenceEqual(modeD) || !extentC.SequenceEqual(extentD))
            {
                throw new ArgumentException("cuTENSOR requires C to have the same modes and extents as D (only A may be permuted).", nameof(modeC));
            }

            nint descA = NewDescriptor(extentA);
            nint descC = NewDescriptor(extentC);
            nint descD = NewDescriptor(extentD);
            Build(CuTensorNative.CreateElementwiseBinary(
                Handle,
                descA, (int[])modeA.Clone(), opA,
                descC, (int[])modeC.Clone(), opC,
                descD, (int[])modeD.Clone(),
                opAC, ComputeDesc));
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>Executes the operation on device pointers.</summary>
    public unsafe void Execute(nint pA, nint pC, nint pD, double alpha = 1.0, double gamma = 1.0)
    {
        ThrowIfDisposed();
        byte* a = stackalloc byte[8];
        byte* g = stackalloc byte[8];
        TensorScalars.Encode(alpha, ComputeDesc, a);
        TensorScalars.Encode(gamma, ComputeDesc, g);
        CuTensorNative.ElementwiseBinaryExecute(Handle, Plan, a, (void*)pA, g, (void*)pC, (void*)pD, Stream.Handle);
        AfterLaunch();
    }

    /// <summary>Asynchronous version of <see cref="Execute"/>.</summary>
    public Task ExecuteAsync(nint pA, nint pC, nint pD, double alpha = 1.0, double gamma = 1.0, CancellationToken ct = default)
    {
        Execute(pA, pC, pD, alpha, gamma);
        return Task.CompletedTask;
    }

    /// <inheritdoc cref="Execute"/>
    public Task ExecuteAsync<T>(
        DeviceBuffer<T> a, DeviceBuffer<T> c, DeviceBuffer<T> d,
        double alpha = 1.0, double gamma = 1.0, CancellationToken ct = default)
        where T : unmanaged, INumberBase<T>
        => ExecuteAsync(a.PointerAsInt, c.PointerAsInt, d.PointerAsInt, alpha, gamma, ct);
}

/// <summary>
/// Reduction: <c>D[modeD] = alpha · opReduce(opA(A[modeA])) + beta · opC(C[modeD])</c>, reducing the
/// modes of A that do not appear in D.
/// </summary>
public sealed class TensorReduction : TensorOperation
{
    /// <summary>Creates a reduction.</summary>
    /// <param name="extentA">Sizes of the dimensions of A.</param>
    /// <param name="modeA">Mode labels of A.</param>
    /// <param name="extentD">Sizes of the dimensions of D (and C).</param>
    /// <param name="modeD">Mode labels of D: a subset of <paramref name="modeA"/>.</param>
    /// <param name="dataType">Element type of all tensors. Default: float.</param>
    /// <param name="opReduce">Reduction operator: <see cref="CuTensorOperator.Add"/>, <see cref="CuTensorOperator.Mul"/>, <see cref="CuTensorOperator.Max"/> or <see cref="CuTensorOperator.Min"/>.</param>
    /// <param name="opA">Unary operator applied to each element of A before reducing. Default: identity.</param>
    /// <param name="opC">Unary operator applied to C. Default: identity.</param>
    /// <param name="options">Options; <c>null</c> uses defaults.</param>
    /// <param name="stream">Optional external stream.</param>
    public TensorReduction(
        long[] extentA, int[] modeA,
        long[] extentD, int[] modeD,
        TensorDataType dataType = TensorDataType.Float32,
        CuTensorOperator opReduce = CuTensorOperator.Add,
        CuTensorOperator opA = CuTensorOperator.Identity,
        CuTensorOperator opC = CuTensorOperator.Identity,
        TensorContractionOptions? options = null,
        CudaStream? stream = null)
        : base(dataType, options, stream)
    {
        try
        {
            ValidateModes(extentA, modeA, nameof(modeA));
            ValidateModes(extentD, modeD, nameof(modeD));
            if (modeD.Except(modeA).Any())
            {
                throw new ArgumentException("modeD must be a subset of modeA.", nameof(modeD));
            }

            nint descA = NewDescriptor(extentA);
            nint descC = NewDescriptor(extentD);
            nint descD = NewDescriptor(extentD);
            Build(CuTensorNative.CreateReduction(
                Handle,
                descA, (int[])modeA.Clone(), opA,
                descC, (int[])modeD.Clone(), opC,
                descD, (int[])modeD.Clone(),
                opReduce, ComputeDesc));
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>
    /// Executes <c>D = alpha · reduce(A) + beta · C</c>. <paramref name="pC"/> and <paramref name="pD"/>
    /// may be the same pointer.
    /// </summary>
    public unsafe void Execute(nint pA, nint pC, nint pD, double alpha = 1.0, double beta = 0.0)
    {
        ThrowIfDisposed();
        byte* a = stackalloc byte[8];
        byte* b = stackalloc byte[8];
        TensorScalars.Encode(alpha, ComputeDesc, a);
        TensorScalars.Encode(beta, ComputeDesc, b);
        CuTensorNative.Reduce(Handle, Plan, a, (void*)pA, b, (void*)pC, (void*)pD,
            (void*)WorkspacePointer, WorkspaceSize, Stream.Handle);
        AfterLaunch();
    }

    /// <summary>Executes <c>D = alpha · reduce(A) + beta · D</c> in place on D.</summary>
    public void Execute(nint pA, nint pD, double alpha = 1.0, double beta = 0.0) =>
        Execute(pA, pD, pD, alpha, beta);

    /// <summary>Asynchronous version of <see cref="Execute(nint, nint, nint, double, double)"/>.</summary>
    public Task ExecuteAsync(nint pA, nint pC, nint pD, double alpha = 1.0, double beta = 0.0, CancellationToken ct = default)
    {
        Execute(pA, pC, pD, alpha, beta);
        return Task.CompletedTask;
    }

    /// <summary>Reduces A into D (<c>beta = 0</c> by default).</summary>
    public Task ExecuteAsync<T>(
        DeviceBuffer<T> a, DeviceBuffer<T> d, double alpha = 1.0, double beta = 0.0, CancellationToken ct = default)
        where T : unmanaged, INumberBase<T>
        => ExecuteAsync(a.PointerAsInt, d.PointerAsInt, d.PointerAsInt, alpha, beta, ct);
}
