// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;
using NVMathNet.Interop;
using NVMathNet.Memory;

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
    private readonly bool _streamOwned;
    private nint   _inputPtr;
    private nint   _outputPtr;
    private nuint  _workspaceSize;
    private nint   _workspacePtr;
    private Workspace _workspace;
    private nint   _blas;              // lazily created, only for output normalization
    private long   _signalSize;        // N: product of the transform-axis lengths
    private long   _outputElements;    // output elements across the whole batch

    // ── Construction ──────────────────────────────────────────────────────────

    /// <summary>
    /// Creates an FFT descriptor.
    /// </summary>
    /// <param name="shape">
    /// Full shape of the input tensor (all dimensions, including batch).
    /// </param>
    /// <param name="axes">
    /// The axes over which to perform the FFT.  <c>null</c> means all axes.
    /// Negative values count from the end. Must be ascending, contiguous and include
    /// either the first or last dimension. Leading axes are supported through strided layouts.
    /// For R2C the shape is the real input; for C2R it is the real output.
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
        {
            throw new ArgumentException("Shape must have at least one dimension.", nameof(shape));
        }

        _shape = (long[])shape.Clone();
        _options = options?.Clone() ?? new FftOptions();
        _doublePrecision = doublePrecision;

        // Resolve axes
        _axes = axes is null
            ? [.. Enumerable.Range(0, shape.Length)]
            : NormalizeAxes(axes, shape.Length);

        // Resolve FFT type
        _fftType = _options.FftType ?? FftType.C2C;
        if (_options.InPlace && _fftType != FftType.C2C)
        {
            throw new NotSupportedException(
                "In-place transforms are supported for C2C only: in-place R2C/C2R needs a padded real layout. Use an out-of-place plan.");
        }

        _nativeType = ResolveCuFftType(_fftType, doublePrecision);

        // Create or adopt stream
        _ownedStream = stream ?? new CudaStream();
        _streamOwned = stream is null;

        // Create plan handle; do not leak the stream or the plan if configuration fails.
        try
        {
            _plan = CuFftNative.Create();
            CuFftNative.SetStream(_plan, _ownedStream.Handle);
            CuFftNative.SetAutoAllocation(_plan, autoAllocate: false);
        }
        catch
        {
            DisposeCore();
            throw;
        }
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
        if (_planned)
        {
            return;
        }

        await Task.Run(() => BuildPlan(), ct).ConfigureAwait(false);
    }

    /// <summary>Synchronously builds the cuFFT plan.</summary>
    public void Plan()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_planned)
        {
            return;
        }

        BuildPlan();
    }

    private void BuildPlan()
    {
        // Derive the FFT dimensions from the selected axes (validated in the constructor)
        long[] n    = [.. _axes.Select(a => _shape[a])];
        int    rank = n.Length;

        // Transform-axis extents of the input and output operands. R2C halves the
        // last transform axis of the output; C2R halves it on the input.
        long[] half = (long[])n.Clone();
        half[^1] = n[^1] / 2 + 1;
        long[] inDims  = _fftType == FftType.C2R ? half : n;
        long[] outDims = _fftType == FftType.R2C ? half : n;

        // Axes are one contiguous block, so the remaining dimensions split into
        // `outer` (before the block) and `inner` (after it).
        long outer = 1, inner = 1;
        for (int i = 0; i < _axes[0]; i++)
        {
            outer *= _shape[i];
        }

        for (int i = _axes[^1] + 1; i < _shape.Length; i++)
        {
            inner *= _shape[i];
        }

        long batchSize = outer * inner;
        if (inner == 1)
        {
            // Trailing axes: each transform is contiguous, batches are packed back-to-back.
            _workspaceSize = CuFftNative.MakePlanMany(
                _plan, rank, n,
                inembed: null, istride: 1, idist: Product(inDims),
                onembed: null, ostride: 1, odist: Product(outDims),
                _nativeType, batchSize);
        }
        else
        {
            // Leading axes (outer == 1): transform elements are `inner` apart and the
            // `inner` batches are interleaved with a distance of one element.
            _workspaceSize = CuFftNative.MakePlanMany(
                _plan, rank, n,
                inembed: inDims, istride: inner, idist: 1,
                onembed: outDims, ostride: inner, odist: 1,
                _nativeType, batchSize);
        }

        // Allocate workspace
        if (_workspaceSize > 0)
        {
            _workspace = new Workspace(_workspaceSize);
            _workspacePtr = _workspace.PointerAsInt;
            unsafe { CuFftNative.SetWorkArea(_plan, (void*)_workspacePtr); }
        }

        NVMathLog.Debug("Fft", () => $"Planned {_fftType} shape [{string.Join(", ", _shape)}] axes [{string.Join(", ", _axes)}]: batch {batchSize}, workspace {_workspaceSize} bytes.");
        _signalSize = Product(n);
        _outputElements = batchSize * Product(outDims);
        _planned = true;
    }

    private static long Product(long[] dims) => dims.Aggregate(1L, (a, s) => a * s);

    /// <summary>
    /// Validates the transform axes: in range, strictly ascending, contiguous, and
    /// covering either the first or the last tensor dimension (cuFFT can express
    /// only a single batch stride).
    /// </summary>
    private static int[] NormalizeAxes(int[] axes, int rank)
    {
        if (axes.Length == 0)
        {
            throw new ArgumentException("At least one FFT axis is required.", nameof(axes));
        }

        int[] result = [.. axes.Select(a => a < 0 ? a + rank : a)];
        for (int i = 0; i < result.Length; i++)
        {
            if (result[i] < 0 || result[i] >= rank)
            {
                throw new ArgumentOutOfRangeException(nameof(axes), $"Axis {axes[i]} is out of range for a {rank}-D tensor.");
            }

            if (i > 0 && result[i] != result[i - 1] + 1)
            {
                throw new ArgumentException("FFT axes must be ascending and contiguous.", nameof(axes));
            }
        }

        if (result[0] != 0 && result[^1] != rank - 1)
        {
            throw new NotSupportedException("FFT axes must include the first or the last dimension.");
        }

        return result;
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
        {
            throw new InvalidOperationException("Call Plan() or PlanAsync() before Execute.");
        }

        LaunchKernel(direction);

        if (_options.Blocking)
        {
            await _ownedStream.SynchronizeAsync(ct).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Synchronously executes the FFT on device memory and returns when done.
    /// Prefer <see cref="ExecuteAsync"/> in async contexts.
    /// </summary>
    public void Execute(FftDirection direction = FftDirection.Forward)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_planned)
        {
            throw new InvalidOperationException("Call Plan() or PlanAsync() before Execute.");
        }

        LaunchKernel(direction);

        if (_options.Blocking)
        {
            _ownedStream.Synchronize();
        }
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

        ApplyNormalization(direction, out_);
    }

    private unsafe void ApplyNormalization(FftDirection direction, void* output)
    {
        if (_options.Normalization == FftNormalization.None)
        {
            return;
        }

        // R2C is always a forward transform and C2R an inverse one; C2C follows the requested direction.
        bool forward = _fftType switch
        {
            FftType.R2C => true,
            FftType.C2R => false,
            _ => direction == FftDirection.Forward,
        };

        double scale = _options.Normalization switch
        {
            FftNormalization.Backward => forward ? 1.0 : 1.0 / _signalSize,
            FftNormalization.Forward => forward ? 1.0 / _signalSize : 1.0,
            FftNormalization.Orthonormal => 1.0 / Math.Sqrt(_signalSize),
            _ => 1.0,
        };

        if (scale == 1.0)
        {
            return;
        }

        if (_blas == 0)
        {
            _blas = CuBlasNative.Create();
            CuBlasNative.SetStream(_blas, _ownedStream.Handle);
        }

        // Complex outputs are scaled as interleaved real pairs.
        bool complexOutput = _fftType != FftType.C2R;
        long scalars = _outputElements * (complexOutput ? 2 : 1);
        if (_doublePrecision)
        {
            CuBlasNative.Dscal(_blas, scalars, scale, (double*)output);
        }
        else
        {
            CuBlasNative.Sscal(_blas, scalars, (float)scale, (float*)output);
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
        var opts = options?.Clone() ?? new FftOptions();
        opts.FftType = FftType.C2C;

        long total = shape.Aggregate(1L, (a, s) => a * s);
        await using var fft = new Fft(shape, axes, opts, stream, doublePrecision: true);

        var output = new DeviceBuffer<Complex>(total);
        try
        {
            fft.ResetOperand(input.PointerAsInt, output.PointerAsInt);
            await fft.PlanAsync(ct).ConfigureAwait(false);
            await fft.ExecuteAsync(FftDirection.Forward, ct).ConfigureAwait(false);
            await fft.SynchronizeAsync(ct).ConfigureAwait(false);
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
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
        var opts = options?.Clone() ?? new FftOptions();
        opts.FftType = FftType.C2C;
        return FftInternalAsync(input, shape, axes, stream, opts, FftDirection.Inverse, true, ct);
    }

    private static async Task<DeviceBuffer<Complex>> FftInternalAsync(
        DeviceBuffer<Complex> input, long[] shape, int[]? axes,
        CudaStream? stream, FftOptions opts, FftDirection direction, bool doublePrecision,
        CancellationToken ct)
    {
        long total = shape.Aggregate(1L, (a, s) => a * s);
        await using var fft = new Fft(shape, axes, opts, stream, doublePrecision);
        var output = new DeviceBuffer<Complex>(total);
        try
        {
            fft.ResetOperand(input.PointerAsInt, output.PointerAsInt);
            await fft.PlanAsync(ct).ConfigureAwait(false);
            await fft.ExecuteAsync(direction, ct).ConfigureAwait(false);
            await fft.SynchronizeAsync(ct).ConfigureAwait(false);
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Performs a real-to-complex (forward half-spectrum) FFT in <b>double precision</b>.
    /// The output has <c>n/2+1</c> complex elements along the last transform axis.
    /// </summary>
    /// <param name="input">Device buffer of <see cref="double"/> real elements.</param>
    /// <param name="shape">Full tensor shape of the real input (all dimensions, including batch), row-major.</param>
    /// <param name="axes">FFT axes; <c>null</c> = all axes.</param>
    /// <param name="stream">Optional external stream.</param>
    /// <param name="options">Optional options; <see cref="FftOptions.FftType"/> is overridden to <see cref="FftType.R2C"/>.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Device buffer of <see cref="Complex"/> elements (half-spectrum).</returns>
    public static async Task<DeviceBuffer<Complex>> RFftAsync(
        DeviceBuffer<double> input,
        long[] shape,
        int[]? axes = null,
        CudaStream? stream = null,
        FftOptions? options = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(shape);
        var (opts, resolvedAxes, outputElements) = PrepareRealToComplex(input.Length, shape, axes, options);
        return await RunTransformAsync<Complex>(
            shape, resolvedAxes, opts, stream, doublePrecision: true,
            input.PointerAsInt, outputElements, FftDirection.Forward, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Performs a real-to-complex (forward half-spectrum) FFT in <b>single precision</b>. .NET has no single-precision
    /// complex type, so the half-spectrum is returned as interleaved <c>(re, im)</c> floats: element <c>k</c> of the
    /// spectrum is <c>output[2k]</c> + i·<c>output[2k+1]</c>, and the buffer holds <c>2 · (number of complex elements)</c> floats.
    /// </summary>
    /// <param name="input">Device buffer of <see cref="float"/> real elements.</param>
    /// <param name="shape">Full tensor shape of the real input (all dimensions, including batch), row-major.</param>
    /// <param name="axes">FFT axes; <c>null</c> = all axes.</param>
    /// <param name="stream">Optional external stream.</param>
    /// <param name="options">Optional options; <see cref="FftOptions.FftType"/> is overridden to <see cref="FftType.R2C"/>.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Device buffer of interleaved single-precision complex values.</returns>
    public static async Task<DeviceBuffer<float>> RFftAsync(
        DeviceBuffer<float> input,
        long[] shape,
        int[]? axes = null,
        CudaStream? stream = null,
        FftOptions? options = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(shape);
        var (opts, resolvedAxes, outputElements) = PrepareRealToComplex(input.Length, shape, axes, options);
        return await RunTransformAsync<float>(
            shape, resolvedAxes, opts, stream, doublePrecision: false,
            input.PointerAsInt, 2 * outputElements, FftDirection.Forward, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Performs a complex-to-real (inverse half-spectrum) FFT in <b>double precision</b>.
    /// The output contains the reconstructed real signal. The result is not scaled unless
    /// <see cref="FftOptions.Normalization"/> says so.
    /// </summary>
    /// <param name="input">Device buffer of <see cref="Complex"/> half-spectrum elements.</param>
    /// <param name="outputShape">
    /// Full shape of the desired real output tensor (all dimensions, including batch), row-major. The last
    /// transform-axis size <c>n</c> determines the half-spectrum length <c>n/2+1</c> that <paramref name="input"/> must hold.
    /// </param>
    /// <param name="axes">FFT axes; <c>null</c> = all axes.</param>
    /// <param name="stream">Optional external stream.</param>
    /// <param name="options">Optional options; <see cref="FftOptions.FftType"/> is overridden to <see cref="FftType.C2R"/>.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Device buffer of <see cref="double"/> reconstructed real elements.</returns>
    public static async Task<DeviceBuffer<double>> IRFftAsync(
        DeviceBuffer<Complex> input,
        long[] outputShape,
        int[]? axes = null,
        CudaStream? stream = null,
        FftOptions? options = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(outputShape);
        var (opts, resolvedAxes, outputElements) = PrepareComplexToReal(input.Length, outputShape, axes, options, 1);
        return await RunTransformAsync<double>(
            outputShape, resolvedAxes, opts, stream, doublePrecision: true,
            input.PointerAsInt, outputElements, FftDirection.Inverse, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Performs a complex-to-real (inverse half-spectrum) FFT in <b>single precision</b>. The half-spectrum is
    /// supplied as interleaved <c>(re, im)</c> floats (see <see cref="RFftAsync(DeviceBuffer{float}, long[], int[], CudaStream, FftOptions, CancellationToken)"/>).
    /// </summary>
    /// <param name="input">Device buffer of interleaved single-precision complex half-spectrum values.</param>
    /// <param name="outputShape">Full shape of the desired real output tensor, row-major.</param>
    /// <param name="axes">FFT axes; <c>null</c> = all axes.</param>
    /// <param name="stream">Optional external stream.</param>
    /// <param name="options">Optional options; <see cref="FftOptions.FftType"/> is overridden to <see cref="FftType.C2R"/>.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Device buffer of <see cref="float"/> reconstructed real elements.</returns>
    public static async Task<DeviceBuffer<float>> IRFftAsync(
        DeviceBuffer<float> input,
        long[] outputShape,
        int[]? axes = null,
        CudaStream? stream = null,
        FftOptions? options = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(outputShape);
        var (opts, resolvedAxes, outputElements) = PrepareComplexToReal(input.Length, outputShape, axes, options, 2);
        return await RunTransformAsync<float>(
            outputShape, resolvedAxes, opts, stream, doublePrecision: false,
            input.PointerAsInt, outputElements, FftDirection.Inverse, ct).ConfigureAwait(false);
    }

    private static (FftOptions Options, int[] Axes, long OutputElements) PrepareRealToComplex(
        long inputLength, long[] shape, int[]? axes, FftOptions? options)
    {
        int[] resolvedAxes = axes is null
            ? [.. Enumerable.Range(0, shape.Length)]
            : NormalizeAxes(axes, shape.Length);
        long inputElements = Product(shape);
        if (inputLength < inputElements)
        {
            throw new ArgumentException($"The input holds {inputLength} elements but shape [{string.Join(", ", shape)}] needs {inputElements}.", nameof(shape));
        }

        var opts = options?.Clone() ?? new FftOptions();
        opts.FftType = FftType.R2C;
        return (opts, resolvedAxes, ComputeR2COutputSize(shape, resolvedAxes));
    }

    private static (FftOptions Options, int[] Axes, long OutputElements) PrepareComplexToReal(
        long inputLength, long[] outputShape, int[]? axes, FftOptions? options, int scalarsPerElement)
    {
        int[] resolvedAxes = axes is null
            ? [.. Enumerable.Range(0, outputShape.Length)]
            : NormalizeAxes(axes, outputShape.Length);
        long required = ComputeR2COutputSize(outputShape, resolvedAxes) * scalarsPerElement;
        if (inputLength < required)
        {
            throw new ArgumentException($"The half-spectrum input holds {inputLength} values but {required} are required for output shape [{string.Join(", ", outputShape)}].", nameof(outputShape));
        }

        var opts = options?.Clone() ?? new FftOptions();
        opts.FftType = FftType.C2R;
        return (opts, resolvedAxes, Product(outputShape));
    }

    private static async Task<DeviceBuffer<TOut>> RunTransformAsync<TOut>(
        long[] planShape, int[] axes, FftOptions opts, CudaStream? stream, bool doublePrecision,
        nint inputPointer, long outputLength, FftDirection direction, CancellationToken ct)
        where TOut : unmanaged, INumberBase<TOut>
    {
        await using var fft = new Fft(planShape, axes, opts, stream, doublePrecision);
        var output = new DeviceBuffer<TOut>(outputLength);
        try
        {
            fft.ResetOperand(inputPointer, output.PointerAsInt);
            await fft.PlanAsync(ct).ConfigureAwait(false);
            await fft.ExecuteAsync(direction, ct).ConfigureAwait(false);
            await fft.SynchronizeAsync(ct).ConfigureAwait(false);
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }

    private static long ComputeR2COutputSize(long[] shape, int[] axes)
    {
        // Half-spectrum: last transform axis becomes n/2+1, others stay the same.
        long total = 1;
        int lastAxis = axes[^1];
        for (int i = 0; i < shape.Length; i++)
        {
            total *= (i == lastAxis) ? (shape[i] / 2 + 1) : shape[i];
        }

        return total;
    }

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

        try
        {
            _ownedStream.Synchronize(); // queued work may still use the plan and workspace
        }
        catch (Exception)
        {
            // The context may already be broken; keep releasing resources.
        }

        if (_blas != 0)
        {
            CuBlasNative.Destroy(_blas);
            _blas = 0;
        }

        if (_plan != default)
        {
            CuFftNative.Destroy(_plan);
            _plan = default;
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
