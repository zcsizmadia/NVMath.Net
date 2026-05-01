// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

namespace NVMathNet.Fft;

/// <summary>Direction of the FFT transform.</summary>
public enum FftDirection
{
    /// <summary>Forward FFT (DFT).</summary>
    Forward = -1,
    /// <summary>Inverse FFT (IDFT).</summary>
    Inverse = 1,
}

/// <summary>The type of FFT transform.</summary>
public enum FftType
{
    /// <summary>Complex-to-complex FFT.</summary>
    C2C,
    /// <summary>Real-to-complex (forward half-spectrum) FFT.</summary>
    R2C,
    /// <summary>Complex-to-real (inverse half-spectrum) FFT.</summary>
    C2R,
}

/// <summary>
/// Options controlling the behaviour of an <see cref="Fft"/> operation.
/// </summary>
public sealed class FftOptions
{
    /// <summary>
    /// The type of FFT to perform.
    /// <c>null</c> means auto-detect from the input element type:
    /// <see cref="FftType.C2C"/> for complex input, <see cref="FftType.R2C"/> for real input.
    /// </summary>
    public FftType? FftType { get; set; }

    /// <summary>
    /// If <c>true</c>, the result overwrites the input buffer (C2C only).
    /// Default: <c>false</c>.
    /// </summary>
    public bool InPlace { get; set; }

    /// <summary>
    /// For C2R FFTs: whether the logical last-axis size is even or odd.
    /// Default: <see cref="LastAxisParity.Even"/>.
    /// </summary>
    public LastAxisParity LastAxisParity { get; set; } = LastAxisParity.Even;

    /// <summary>
    /// CUDA device to use when input resides on the host.
    /// Default: device 0.
    /// </summary>
    public int DeviceId { get; set; }

    /// <summary>
    /// When <c>true</c>, execute calls block until the GPU operation is complete.
    /// When <c>false</c> (default), execute launches the kernel and returns immediately.
    /// Use <see cref="Fft.SynchronizeAsync"/> to wait for completion.
    /// </summary>
    public bool Blocking { get; set; }
}

/// <summary>For C2R FFTs, specifies whether the reconstructed last-axis size is even or odd.</summary>
public enum LastAxisParity
{
    /// <summary>Last axis size = 2*(m-1) where m is the last input axis size.</summary>
    Even,
    /// <summary>Last axis size = 2*m-1 where m is the last input axis size.</summary>
    Odd,
}

/// <summary>
/// Strongly-typed element types supported by the FFT.
/// </summary>
internal enum FftElementType
{
    Float32,   // float  — R2C/C2R/C2C single
    Float64,   // double — D2Z/Z2D/Z2Z double
    Complex64,  // (float,  float) — C2C / C2R input / R2C output
    Complex128, // (double, double) — Z2Z / Z2D input / D2Z output
}
