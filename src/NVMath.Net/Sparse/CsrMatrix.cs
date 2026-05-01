// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using System.Numerics;
using NVMathNet.Interop;

namespace NVMathNet.Sparse;

/// <summary>
/// A sparse matrix stored in Compressed Sparse Row (CSR) format on the GPU.
/// Row offsets, column indices, and values are each held in a separate
/// <see cref="DeviceBuffer{T}"/>.
/// </summary>
/// <typeparam name="T">Unmanaged numeric value element type (float, double, etc.).</typeparam>
public sealed unsafe class CsrMatrix<T> : IDisposable where T : unmanaged, INumberBase<T>
{
    private nint _descriptor;
    private bool _disposed;

    /// <summary>Number of rows.</summary>
    public long Rows { get; }
    /// <summary>Number of columns.</summary>
    public long Cols { get; }
    /// <summary>Number of non-zero elements.</summary>
    public long NNZ => Values.Length;

    /// <summary>Row offsets — length Rows+1.</summary>
    public DeviceBuffer<int> RowOffsets { get; }
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
    /// Creates a CSR sparse matrix from pre-allocated device buffers.
    /// </summary>
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns.</param>
    /// <param name="rowOffsets">Row offsets (length rows+1).</param>
    /// <param name="colIndices">Column indices (length NNZ).</param>
    /// <param name="values">Non-zero values (length NNZ).</param>
    public CsrMatrix(
        long rows, long cols,
        DeviceBuffer<int> rowOffsets,
        DeviceBuffer<int> colIndices,
        DeviceBuffer<T> values)
    {
        ArgumentNullException.ThrowIfNull(rowOffsets);
        ArgumentNullException.ThrowIfNull(colIndices);
        ArgumentNullException.ThrowIfNull(values);

        Rows = rows;
        Cols = cols;
        RowOffsets = rowOffsets;
        ColIndices = colIndices;
        Values = values;

        var cudaType = NativeCudaType();
        _descriptor = CuSparseNative.CreateCsr(
            rows, cols, values.Length,
            rowOffsets.Pointer,
            colIndices.Pointer,
            values.Pointer,
            CuSparseIndexType.Int32,
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
