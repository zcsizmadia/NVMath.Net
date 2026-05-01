// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

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
public sealed class CuFftException : Exception
{
    /// <summary>The raw cuFFT error code.</summary>
    public int ErrorCode { get; }

    /// <summary>Initialises a new instance with <paramref name="code"/> and <paramref name="message"/>.</summary>
    public CuFftException(int code, string message)
        : base(message)
    {
        ErrorCode = code;
    }
}

/// <summary>
/// Represents an error returned by cuBLAS / cuBLASLt.
/// </summary>
public sealed class CuBlasException : Exception
{
    /// <summary>The raw cuBLAS error code.</summary>
    public int ErrorCode { get; }

    /// <summary>Initialises a new instance with <paramref name="code"/> and <paramref name="message"/>.</summary>
    public CuBlasException(int code, string message)
        : base(message)
    {
        ErrorCode = code;
    }
}

/// <summary>
/// Represents an error returned by cuSPARSE.
/// </summary>
public sealed class CuSparseException : Exception
{
    /// <summary>The raw cuSPARSE error code.</summary>
    public int ErrorCode { get; }

    /// <summary>Initialises a new instance with <paramref name="code"/> and <paramref name="message"/>.</summary>
    public CuSparseException(int code, string message)
        : base(message)
    {
        ErrorCode = code;
    }
}

/// <summary>
/// Represents an error returned by cuTENSOR.
/// </summary>
public sealed class CuTensorException : Exception
{
    /// <summary>The raw cuTENSOR error code.</summary>
    public int ErrorCode { get; }

    /// <summary>Initialises a new instance with <paramref name="code"/> and <paramref name="message"/>.</summary>
    public CuTensorException(int code, string message)
        : base(message)
    {
        ErrorCode = code;
    }
}
