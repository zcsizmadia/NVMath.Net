// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using System.Numerics;
using System.Runtime.CompilerServices;
using NVMathNet.Interop;

namespace NVMathNet.Fft;

/// <summary>
/// Stateful cuFFT wrapper.  Mirrors the design of <c>nvmath.fft.FFT</c>:
/// <list type="number">
///   <item><description>Construct with an input descriptor.</description></item>
///   <item><description>Call <see cref="PlanAsync"/> to build the cuFFT plan.</description></item>
///   <item><description>Call <see cref="ExecuteAsync"/> (possibly multiple times with <see cref="ResetOperand"/>).</description></item>
///   <item><description>Dispose (or use <c>await using</c>) to free all CUDA resources.</description></item>
/// </list>
/// </summary>
public sealed class Fft : IAsyncDisposable, IDisposable
{
    // ── Plan state ─────────────────────────────────────────────────────────────

    private nint _plan;
    private bool _planned;
    private bool _disposed;

    // ── Descriptor ─────────────────────────────────────────────────────────────

    private readonly long[] _shape;
    private readonly int[]  _axes;
    private readonly FftOptions _options;
    private readonly FftType _fftType;
    private readonly CuFftType _nativeType;
    private readonly bool _doublePrecision;

    // ── Device memory ──────────────────────────────────────────────────────────

    private readonly CudaStream _ownedStream;
    private nint   _inputPtr;
    private nint   _outputPtr;
    private nuint  _workspaceSize;
    private nint   _workspacePtr;

    // ── Construction ──────────────────────────────────────────────────────────

    /// <summary>
    /// Creates an FFT descriptor.
    /// </summary>
    /// <param name="shape">
    /// Full shape of the input tensor (all dimensions, including batch).
    /// </param>
    /// <param name="axes">
    /// The axes over which to perform the FFT.  <c>null</c> means all axes.
    /// Must be contiguous and include either the first or last dimension.
    /// </param>
    /// <param name="options">FFT options; <c>null</c> uses defaults.</param>
    /// <param name="stream">
    /// Optional external stream to use.  If <c>null</c>, a private stream is created.
    /// </param>
    /// <param name="doublePrecision">
    /// <c>true</c> for double / complex128 precision; <c>false</c> (default) for float / complex64.
    /// </param>
    public Fft(
        long[] shape,
        int[]? axes = null,
        FftOptions? options = null,
        CudaStream? stream = null,
        bool doublePrecision = false)
    {
        ArgumentNullException.ThrowIfNull(shape);
        if (shape.Length == 0)
            throw new ArgumentException("Shape must have at least one dimension.", nameof(shape));

        _shape = (long[])shape.Clone();
        _options = options ?? new FftOptions();
        _doublePrecision = doublePrecision;

        // Resolve axes
        _axes = axes is null
            ? Enumerable.Range(0, shape.Length).ToArray()
            : (int[])axes.Clone();

        // Resolve FFT type
        _fftType = _options.FftType ?? FftType.C2C;
        _nativeType = ResolveCuFftType(_fftType, doublePrecision);

        // Create or adopt stream
        _ownedStream = stream ?? new CudaStream();

        // Create plan handle
        _plan = CuFftNative.Create();
        CuFftNative.SetStream(_plan, _ownedStream.Handle);
        CuFftNative.SetAutoAllocation(_plan, autoAllocate: false);
    }

    // ── Mapping helpers ───────────────────────────────────────────────────────

    private static CuFftType ResolveCuFftType(FftType type, bool doublePrecision) =>
        (type, doublePrecision) switch
        {
            (FftType.C2C, false) => CuFftType.C2C,
            (FftType.C2C, true)  => CuFftType.Z2Z,
            (FftType.R2C, false) => CuFftType.R2C,
            (FftType.R2C, true)  => CuFftType.D2Z,
            (FftType.C2R, false) => CuFftType.C2R,
            (FftType.C2R, true)  => CuFftType.Z2D,
            _                    => throw new InvalidOperationException($"Unsupported FFT type: {type}"),
        };

    // ── Plan ──────────────────────────────────────────────────────────────────

    /// <summary>Builds the cuFFT plan on the current stream.</summary>
    public async Task PlanAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_planned) return;

        await Task.Run(() => BuildPlan(), ct).ConfigureAwait(false);
    }

    /// <summary>Synchronously builds the cuFFT plan.</summary>
    public void Plan()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_planned) return;
        BuildPlan();
    }

    private void BuildPlan()
    {
        // Derive the FFT dimensions from the selected axes
        long[] n    = _axes.Select(a => _shape[a]).ToArray();
        long   rank = n.Length;

        // For a simple contiguous layout the strides and distances follow from n
        long batchSize = _shape.Aggregate(1L, (acc, s) => acc * s) / n.Aggregate(1L, (acc, s) => acc * s);

        _workspaceSize = CuFftNative.MakePlanMany(
            _plan, (int)rank, n,
            inembed: null, istride: 1, idist: n.Aggregate(1L, (a, s) => a * s),
            onembed: null, ostride: 1, odist: OutputDist(n),
            _nativeType, batchSize);

        // Allocate workspace
        if (_workspaceSize > 0)
        {
            unsafe { _workspacePtr = (nint)CudaRuntime.Malloc(_workspaceSize); }
            unsafe { CuFftNative.SetWorkArea(_plan, (void*)_workspacePtr); }
        }

        _planned = true;
    }

    private long OutputDist(long[] n)
    {
        return _fftType switch
        {
            FftType.R2C => n[..^1].Aggregate(1L, (a, s) => a * s) * (n[^1] / 2 + 1),
            FftType.C2R => n.Aggregate(1L, (a, s) => a * s),      // output is real
            _           => n.Aggregate(1L, (a, s) => a * s),
        };
    }

    // ── Operand management ────────────────────────────────────────────────────

    /// <summary>
    /// Sets the device pointers for the current execution.
    /// Both <paramref name="inputPtr"/> and <paramref name="outputPtr"/> must remain
    /// valid until <see cref="SynchronizeAsync"/> completes.
    /// </summary>
    public void ResetOperand(nint inputPtr, nint outputPtr)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _inputPtr  = inputPtr;
        _outputPtr = outputPtr;
    }

    // ── Execute ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Executes the FFT on device pointers previously set with
    /// <see cref="ResetOperand"/>.  Returns a <see cref="Task"/>
    /// that completes when the GPU kernel finishes.
    /// </summary>
    public async Task ExecuteAsync(
        FftDirection direction = FftDirection.Forward,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_planned)
            throw new InvalidOperationException("Call Plan() or PlanAsync() before Execute.");

        LaunchKernel(direction);

        if (_options.Blocking)
            await _ownedStream.SynchronizeAsync(ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Synchronously executes the FFT on device memory and returns when done.
    /// Prefer <see cref="ExecuteAsync"/> in async contexts.
    /// </summary>
    public void Execute(FftDirection direction = FftDirection.Forward)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_planned)
            throw new InvalidOperationException("Call Plan() or PlanAsync() before Execute.");

        LaunchKernel(direction);

        if (_options.Blocking)
            _ownedStream.Synchronize();
    }

    private unsafe void LaunchKernel(FftDirection direction)
    {
        void* inp = (void*)_inputPtr;
        void* out_ = _options.InPlace ? inp : (void*)_outputPtr;

        switch (_nativeType)
        {
            case CuFftType.C2C:
                CuFftNative.ExecC2C(_plan, inp, out_, (CuFftDirection)direction);
                break;
            case CuFftType.Z2Z:
                CuFftNative.ExecZ2Z(_plan, inp, out_, (CuFftDirection)direction);
                break;
            case CuFftType.R2C:
                CuFftNative.ExecR2C(_plan, inp, out_);
                break;
            case CuFftType.C2R:
                CuFftNative.ExecC2R(_plan, inp, out_);
                break;
            case CuFftType.D2Z:
                CuFftNative.ExecD2Z(_plan, inp, out_);
                break;
            case CuFftType.Z2D:
                CuFftNative.ExecZ2D(_plan, inp, out_);
                break;
        }
    }

    // ── Synchronization ───────────────────────────────────────────────────────

    /// <summary>
    /// Returns a <see cref="Task"/> that completes once all work on the internal
    /// stream has finished.
    /// </summary>
    public Task SynchronizeAsync(CancellationToken ct = default) =>
        _ownedStream.SynchronizeAsync(ct);

    // ── Convenience static wrappers ───────────────────────────────────────────

    /// <summary>
    /// Performs a forward complex-to-complex FFT.
    /// Allocates and frees plan resources automatically.
    /// </summary>
    /// <param name="input">Device buffer of complex64 (interleaved float pairs) elements.</param>
    /// <param name="shape">Full tensor shape.</param>
    /// <param name="axes">FFT axes; <c>null</c> = all.</param>
    /// <param name="stream">Optional external stream.</param>
    /// <param name="options">Optional options.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task<DeviceBuffer<Complex>> FftAsync(
        DeviceBuffer<Complex> input,
        long[] shape,
        int[]? axes = null,
        CudaStream? stream = null,
        FftOptions? options = null,
        CancellationToken ct = default)
    {
        var opts = options ?? new FftOptions { FftType = FftType.C2C };
        opts.FftType ??= FftType.C2C;

        long total = shape.Aggregate(1L, (a, s) => a * s);
        await using var fft = new Fft(shape, axes, opts, stream, doublePrecision: false);

        var output = new DeviceBuffer<Complex>(total);
        fft.ResetOperand(input.PointerAsInt, output.PointerAsInt);
        await fft.PlanAsync(ct).ConfigureAwait(false);
        await fft.ExecuteAsync(FftDirection.Forward, ct).ConfigureAwait(false);
        await fft.SynchronizeAsync(ct).ConfigureAwait(false);
        return output;
    }

    /// <summary>
    /// Performs an inverse complex-to-complex FFT.
    /// </summary>
    public static Task<DeviceBuffer<Complex>> IFftAsync(
        DeviceBuffer<Complex> input,
        long[] shape,
        int[]? axes = null,
        CudaStream? stream = null,
        FftOptions? options = null,
        CancellationToken ct = default)
    {
        var opts = options ?? new FftOptions { FftType = FftType.C2C };
        return FftInternalAsync(input, shape, axes, stream, opts, FftDirection.Inverse, false, ct);
    }

    private static async Task<DeviceBuffer<Complex>> FftInternalAsync(
        DeviceBuffer<Complex> input, long[] shape, int[]? axes,
        CudaStream? stream, FftOptions opts, FftDirection direction, bool doublePrecision,
        CancellationToken ct)
    {
        long total = shape.Aggregate(1L, (a, s) => a * s);
        await using var fft = new Fft(shape, axes, opts, stream, doublePrecision);
        var output = new DeviceBuffer<Complex>(total);
        fft.ResetOperand(input.PointerAsInt, output.PointerAsInt);
        await fft.PlanAsync(ct).ConfigureAwait(false);
        await fft.ExecuteAsync(direction, ct).ConfigureAwait(false);
        await fft.SynchronizeAsync(ct).ConfigureAwait(false);
        return output;
    }

    /// <summary>
    /// Performs a real-to-complex (forward half-spectrum) FFT.
    /// The output has shape <c>[..., n/2+1]</c> complex elements along the last transform axis.
    /// </summary>
    /// <param name="input">Device buffer of <see cref="float"/> real elements.</param>
    /// <param name="shape">Full tensor shape of the real input (all dimensions, including batch).</param>
    /// <param name="axes">FFT axes; <c>null</c> = all axes.</param>
    /// <param name="stream">Optional external stream.</param>
    /// <param name="options">Optional options; <see cref="FftOptions.FftType"/> is overridden to <see cref="FftType.R2C"/>.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Device buffer of <see cref="Complex"/> elements (half-spectrum).</returns>
    public static async Task<DeviceBuffer<Complex>> RFftAsync(
        DeviceBuffer<float> input,
        long[] shape,
        int[]? axes = null,
        CudaStream? stream = null,
        FftOptions? options = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(shape);

        var opts = options is null ? new FftOptions() : new FftOptions
        {
            InPlace = options.InPlace,
            LastAxisParity = options.LastAxisParity,
            DeviceId = options.DeviceId,
            Blocking = options.Blocking,
        };
        opts.FftType = FftType.R2C;

        // Compute half-spectrum output size: last axis shrinks to n/2+1
        int[] resolvedAxes = axes ?? Enumerable.Range(0, shape.Length).ToArray();
        long outputElements = ComputeR2COutputSize(shape, resolvedAxes);

        await using var fft = new Fft(shape, resolvedAxes, opts, stream, doublePrecision: false);
        var output = new DeviceBuffer<Complex>(outputElements);
        fft.ResetOperand(input.PointerAsInt, output.PointerAsInt);
        await fft.PlanAsync(ct).ConfigureAwait(false);
        await fft.ExecuteAsync(FftDirection.Forward, ct).ConfigureAwait(false);
        await fft.SynchronizeAsync(ct).ConfigureAwait(false);
        return output;
    }

    /// <summary>
    /// Performs a complex-to-real (inverse half-spectrum) FFT.
    /// The output contains the reconstructed real signal.
    /// </summary>
    /// <param name="input">Device buffer of <see cref="Complex"/> half-spectrum elements.</param>
    /// <param name="outputShape">
    /// Full shape of the desired real output tensor (all dimensions, including batch).
    /// The last transform-axis size must equal <c>2*(m-1)</c> (even) or <c>2*m-1</c> (odd)
    /// where <c>m</c> is the last axis size of <paramref name="input"/>.
    /// </param>
    /// <param name="axes">FFT axes; <c>null</c> = all axes.</param>
    /// <param name="stream">Optional external stream.</param>
    /// <param name="options">
    /// Optional options; <see cref="FftOptions.FftType"/> is overridden to <see cref="FftType.C2R"/>.
    /// Set <see cref="FftOptions.LastAxisParity"/> if the output last-axis size is odd.
    /// </param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Device buffer of <see cref="float"/> reconstructed real elements.</returns>
    public static async Task<DeviceBuffer<float>> IRFftAsync(
        DeviceBuffer<Complex> input,
        long[] outputShape,
        int[]? axes = null,
        CudaStream? stream = null,
        FftOptions? options = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(outputShape);

        var opts = options is null ? new FftOptions() : new FftOptions
        {
            InPlace = options.InPlace,
            LastAxisParity = options.LastAxisParity,
            DeviceId = options.DeviceId,
            Blocking = options.Blocking,
        };
        opts.FftType = FftType.C2R;

        long outputElements = outputShape.Aggregate(1L, (a, s) => a * s);

        // The plan uses the *output* (real) shape for C2R
        await using var fft = new Fft(outputShape, axes, opts, stream, doublePrecision: false);
        var output = new DeviceBuffer<float>(outputElements);
        fft.ResetOperand(input.PointerAsInt, output.PointerAsInt);
        await fft.PlanAsync(ct).ConfigureAwait(false);
        await fft.ExecuteAsync(FftDirection.Inverse, ct).ConfigureAwait(false);
        await fft.SynchronizeAsync(ct).ConfigureAwait(false);
        return output;
    }

    private static long ComputeR2COutputSize(long[] shape, int[] axes)
    {
        // Half-spectrum: last transform axis becomes n/2+1, others stay the same.
        long total = 1;
        int lastAxis = axes[^1];
        for (int i = 0; i < shape.Length; i++)
            total *= (i == lastAxis) ? (shape[i] / 2 + 1) : shape[i];
        return total;
    }

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

        if (_plan != default)
        {
            CuFftNative.Destroy(_plan);
            _plan = default;
        }

        if (_workspacePtr != 0)
        {
            unsafe { CudaRuntime.Free((void*)_workspacePtr); }
            _workspacePtr = 0;
        }

        _ownedStream.Dispose();
    }
}
