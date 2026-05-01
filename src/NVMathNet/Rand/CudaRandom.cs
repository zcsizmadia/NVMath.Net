// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Interop;

namespace NVMathNet.Rand;

/// <summary>
/// GPU random number generator backed by cuRAND. Wraps a cuRAND generator handle
/// and provides methods to fill device buffers with random numbers.
/// </summary>
public sealed class CudaRandom : IDisposable
{
    private nint _generator;
    private readonly CudaStream? _stream;
    private readonly bool _streamOwned;
    private bool _disposed;

    /// <summary>
    /// Creates a new GPU random number generator.
    /// </summary>
    /// <param name="rngType">The random number generator algorithm.</param>
    /// <param name="seed">Seed value for the generator.</param>
    /// <param name="stream">Optional CUDA stream; a private stream is created if null.</param>
    public CudaRandom(
        CuRandRngType rngType = CuRandRngType.PseudoDefault,
        ulong seed = 0,
        CudaStream? stream = null)
    {
        _streamOwned = stream is null;
        _stream = stream ?? new CudaStream();
        _generator = CuRandNative.CreateGenerator(rngType);
        CuRandNative.SetStream(_generator, _stream.Handle);
        if (seed != 0)
        {
            CuRandNative.SetSeed(_generator, seed);
        }
    }

    /// <summary>Sets the seed for the pseudorandom generator.</summary>
    public void SetSeed(ulong seed)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CuRandNative.SetSeed(_generator, seed);
    }

    // ── Uniform distribution ─────────────────────────────────────────────────

    /// <summary>
    /// Fills the device buffer with uniformly distributed single-precision
    /// floats in the range (0, 1].
    /// </summary>
    public unsafe void FillUniform(DeviceBuffer<float> buffer)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(buffer);
        CuRandNative.GenerateUniform(_generator, (float*)buffer.Pointer, (nuint)buffer.Length);
    }

    /// <summary>
    /// Fills the device buffer with uniformly distributed double-precision
    /// floats in the range (0, 1].
    /// </summary>
    public unsafe void FillUniform(DeviceBuffer<double> buffer)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(buffer);
        CuRandNative.GenerateUniformDouble(_generator, (double*)buffer.Pointer, (nuint)buffer.Length);
    }

    /// <summary>
    /// Fills the device buffer with uniformly distributed values and synchronises.
    /// </summary>
    public async Task FillUniformAsync(DeviceBuffer<float> buffer, CancellationToken ct = default)
    {
        FillUniform(buffer);
        await _stream!.SynchronizeAsync(ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Fills the device buffer with uniformly distributed values and synchronises.
    /// </summary>
    public async Task FillUniformAsync(DeviceBuffer<double> buffer, CancellationToken ct = default)
    {
        FillUniform(buffer);
        await _stream!.SynchronizeAsync(ct).ConfigureAwait(false);
    }

    // ── Normal distribution ──────────────────────────────────────────────────

    /// <summary>
    /// Fills the device buffer with normally distributed single-precision floats.
    /// Count must be even for cuRAND.
    /// </summary>
    public unsafe void FillNormal(DeviceBuffer<float> buffer, float mean = 0f, float stddev = 1f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(buffer);
        CuRandNative.GenerateNormal(_generator, (float*)buffer.Pointer, (nuint)buffer.Length, mean, stddev);
    }

    /// <summary>
    /// Fills the device buffer with normally distributed double-precision floats.
    /// Count must be even for cuRAND.
    /// </summary>
    public unsafe void FillNormal(DeviceBuffer<double> buffer, double mean = 0.0, double stddev = 1.0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(buffer);
        CuRandNative.GenerateNormalDouble(_generator, (double*)buffer.Pointer, (nuint)buffer.Length, mean, stddev);
    }

    /// <summary>
    /// Fills the device buffer with normally distributed values and synchronises.
    /// </summary>
    public async Task FillNormalAsync(DeviceBuffer<float> buffer, float mean = 0f, float stddev = 1f, CancellationToken ct = default)
    {
        FillNormal(buffer, mean, stddev);
        await _stream!.SynchronizeAsync(ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Fills the device buffer with normally distributed values and synchronises.
    /// </summary>
    public async Task FillNormalAsync(DeviceBuffer<double> buffer, double mean = 0.0, double stddev = 1.0, CancellationToken ct = default)
    {
        FillNormal(buffer, mean, stddev);
        await _stream!.SynchronizeAsync(ct).ConfigureAwait(false);
    }

    // ── Log-normal distribution ──────────────────────────────────────────────

    /// <summary>
    /// Fills the device buffer with log-normally distributed single-precision floats.
    /// Count must be even for cuRAND.
    /// </summary>
    public unsafe void FillLogNormal(DeviceBuffer<float> buffer, float mean = 0f, float stddev = 1f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(buffer);
        CuRandNative.GenerateLogNormal(_generator, (float*)buffer.Pointer, (nuint)buffer.Length, mean, stddev);
    }

    /// <summary>
    /// Fills the device buffer with log-normally distributed double-precision floats.
    /// </summary>
    public unsafe void FillLogNormal(DeviceBuffer<double> buffer, double mean = 0.0, double stddev = 1.0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(buffer);
        CuRandNative.GenerateLogNormalDouble(_generator, (double*)buffer.Pointer, (nuint)buffer.Length, mean, stddev);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_generator != 0)
        {
            CuRandNative.DestroyGenerator(_generator);
            _generator = 0;
        }

        if (_streamOwned)
        {
            _stream?.Dispose();
        }
    }
}
