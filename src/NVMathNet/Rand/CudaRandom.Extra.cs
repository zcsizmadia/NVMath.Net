// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Interop;

namespace NVMathNet.Rand;

// Integer and Poisson generation, skip-ahead and quasirandom dimensions.
public sealed partial class CudaRandom
{
    /// <summary>The generator algorithm this instance was created with.</summary>
    public CuRandRngType RngType { get; }

    /// <summary><c>true</c> for quasirandom (Sobol) generators.</summary>
    public bool IsQuasiRandom => RngType >= CuRandRngType.QuasiDefault;

    // ── Integers ──────────────────────────────────────────────────────────────

    /// <summary>Fills the buffer with uniformly distributed 32-bit unsigned integers.</summary>
    public unsafe void FillUInt32(DeviceBuffer<uint> buffer)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(buffer);
        CuRandNative.Generate(_generator, (uint*)buffer.Pointer, (nuint)buffer.Length);
    }

    /// <summary>Fills the buffer with 32-bit unsigned integers and synchronises.</summary>
    public async Task FillUInt32Async(DeviceBuffer<uint> buffer, CancellationToken ct = default)
    {
        FillUInt32(buffer);
        await _stream!.SynchronizeAsync(ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Fills the buffer with 64-bit unsigned integers. Only valid for the 64-bit
    /// quasirandom generators (<see cref="CuRandRngType.QuasiSobol64"/> and
    /// <see cref="CuRandRngType.QuasiScrambledSobol64"/>).
    /// </summary>
    public unsafe void FillUInt64(DeviceBuffer<ulong> buffer)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(buffer);
        CuRandNative.GenerateLongLong(_generator, (ulong*)buffer.Pointer, (nuint)buffer.Length);
    }

    /// <summary>Fills the buffer with 64-bit unsigned integers and synchronises.</summary>
    public async Task FillUInt64Async(DeviceBuffer<ulong> buffer, CancellationToken ct = default)
    {
        FillUInt64(buffer);
        await _stream!.SynchronizeAsync(ct).ConfigureAwait(false);
    }

    // ── Poisson ───────────────────────────────────────────────────────────────

    /// <summary>Fills the buffer with Poisson-distributed integers with the given mean.</summary>
    /// <param name="buffer">Output buffer.</param>
    /// <param name="lambda">Mean of the distribution; must be positive.</param>
    public unsafe void FillPoisson(DeviceBuffer<uint> buffer, double lambda)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(buffer);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(lambda, 0.0);
        CuRandNative.GeneratePoisson(_generator, (uint*)buffer.Pointer, (nuint)buffer.Length, lambda);
    }

    /// <summary>Fills the buffer with Poisson-distributed integers and synchronises.</summary>
    public async Task FillPoissonAsync(DeviceBuffer<uint> buffer, double lambda, CancellationToken ct = default)
    {
        FillPoisson(buffer, lambda);
        await _stream!.SynchronizeAsync(ct).ConfigureAwait(false);
    }

    // ── Skip-ahead / dimensions ───────────────────────────────────────────────

    /// <summary>
    /// Skips ahead in the sequence: the next values generated are those that would follow the
    /// first <paramref name="offset"/> values. Supported by the Philox, XORWOW, MRG32k3a and
    /// Sobol generators; some generators round the offset to a multiple of their block size.
    /// </summary>
    public void SetOffset(ulong offset)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CuRandNative.SetOffset(_generator, offset);
    }

    /// <summary>
    /// Sets the number of dimensions of a quasirandom generator (1–20000). Successive values are
    /// interleaved across dimensions. Only valid for quasirandom generators.
    /// </summary>
    public void SetQuasiRandomDimensions(uint dimensions)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!IsQuasiRandom)
        {
            throw new InvalidOperationException("Dimensions can only be set on a quasirandom generator.");
        }

        CuRandNative.SetQuasiRandomGeneratorDimensions(_generator, dimensions);
    }
}
