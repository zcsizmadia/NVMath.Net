// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

namespace NVMathNet.Interop;

/// <summary>
/// Represents an error returned by the CUDA Runtime API.
/// </summary>
public sealed class CudaException : Exception
{
    /// <summary>The raw CUDA error code.</summary>
    public CudaError ErrorCode { get; }

    /// <summary>Initialises a new instance with <paramref name="code"/> and <paramref name="message"/>.</summary>
    public CudaException(CudaError code, string message)
        : base(message)
    {
        ErrorCode = code;
    }

    /// <summary>Initialises a new instance with <paramref name="code"/>, <paramref name="message"/>, and an inner exception.</summary>
    public CudaException(CudaError code, string message, Exception inner)
        : base(message, inner)
    {
        ErrorCode = code;
    }
}

/// <summary>
/// Represents an error returned by cuFFT.
/// </summary>
/// <remarks>Initialises a new instance with <paramref name="code"/> and <paramref name="message"/>.</remarks>
public sealed class CuFftException(int code, string message) : Exception(message)
{
    /// <summary>The raw cuFFT error code.</summary>
    public int ErrorCode { get; } = code;
}

/// <summary>
/// Represents an error returned by cuBLAS / cuBLASLt.
/// </summary>
/// <remarks>Initialises a new instance with <paramref name="code"/> and <paramref name="message"/>.</remarks>
public sealed class CuBlasException(int code, string message) : Exception(message)
{
    /// <summary>The raw cuBLAS error code.</summary>
    public int ErrorCode { get; } = code;
}

/// <summary>
/// Represents an error returned by cuSPARSE.
/// </summary>
/// <remarks>Initialises a new instance with <paramref name="code"/> and <paramref name="message"/>.</remarks>
public sealed class CuSparseException(int code, string message) : Exception(message)
{
    /// <summary>The raw cuSPARSE error code.</summary>
    public int ErrorCode { get; } = code;
}

/// <summary>
/// Represents an error returned by cuTENSOR.
/// </summary>
/// <remarks>Initialises a new instance with <paramref name="code"/> and <paramref name="message"/>.</remarks>
public sealed class CuTensorException(int code, string message) : Exception(message)
{
    /// <summary>The raw cuTENSOR error code.</summary>
    public int ErrorCode { get; } = code;
}

/// <summary>
/// Represents an error returned by cuSOLVER.
/// </summary>
/// <remarks>Initialises a new instance with <paramref name="code"/> and <paramref name="message"/>.</remarks>
public sealed class CuSolverException(int code, string message) : Exception(message)
{
    /// <summary>The raw cuSOLVER error code.</summary>
    public int ErrorCode { get; } = code;
}

/// <summary>
/// Represents an error returned by cuRAND.
/// </summary>
/// <remarks>Initialises a new instance with <paramref name="code"/> and <paramref name="message"/>.</remarks>
public sealed class CuRandException(int code, string message) : Exception(message)
{
    /// <summary>The raw cuRAND error code.</summary>
    public int ErrorCode { get; } = code;
}
