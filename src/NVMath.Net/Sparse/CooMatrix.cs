// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using System.Numerics;
using NVMathNet.Interop;

namespace NVMathNet.Sparse;

/// <summary>
/// A sparse matrix stored in COOrdinate (COO) format on the GPU.
/// </summary>
/// <typeparam name="T">Unmanaged numeric value element type (float, double, etc.).</typeparam>
public sealed unsafe class CooMatrix<T> : IDisposable where T : unmanaged, INumberBase<T>
{
    private nint _descriptor;
    private bool _disposed;

    /// <summary>Number of rows.</summary>
    public long Rows { get; }
    /// <summary>Number of columns.</summary>
    public long Cols { get; }
    /// <summary>Number of non-zero elements.</summary>
    public long NNZ => Values.Length;

    /// <summary>Row indices — length NNZ.</summary>
    public DeviceBuffer<int> RowIndices { get; }
    /// <summary>Column indices — length NNZ.</summary>
    public DeviceBuffer<int> ColIndices { get; }
    /// <summary>Non-zero values — length NNZ.</summary>
    public DeviceBuffer<T> Values { get; }

    /// <summary>Internal cuSPARSE sparse matrix descriptor.</summary>
    internal nint Descriptor
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _descriptor;
        }
    }

    /// <summary>
    /// Creates a COO sparse matrix from pre-allocated device buffers.
    /// </summary>
    public CooMatrix(
        long rows, long cols,
        DeviceBuffer<int> rowIndices,
        DeviceBuffer<int> colIndices,
        DeviceBuffer<T> values)
    {
        ArgumentNullException.ThrowIfNull(rowIndices);
        ArgumentNullException.ThrowIfNull(colIndices);
        ArgumentNullException.ThrowIfNull(values);

        Rows = rows;
        Cols = cols;
        RowIndices = rowIndices;
        ColIndices = colIndices;
        Values = values;

        var cudaType = NativeCudaType();
        _descriptor = CuSparseNative.CreateCoo(
            rows, cols, values.Length,
            rowIndices.Pointer,
            colIndices.Pointer,
            values.Pointer,
            CuSparseIndexType.Int32,
            CuSparseIndexBase.Zero,
            cudaType);
    }

    private static CudaDataType NativeCudaType()
    {
        if (typeof(T) == typeof(float))   return CudaDataType.R_32F;
        if (typeof(T) == typeof(double))  return CudaDataType.R_64F;
        if (typeof(T) == typeof(Half))    return CudaDataType.R_16F;
        throw new NotSupportedException($"Unsupported element type {typeof(T)} for sparse matrix.");
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_descriptor != default)
        {
            CuSparseNative.DestroySpMat(_descriptor);
            _descriptor = default;
        }
    }
}
