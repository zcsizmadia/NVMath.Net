// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

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
    /// Creates an additional descriptor over the same arrays. Operations that attach attributes to the descriptor
    /// (triangular solves) use it so the matrix's own descriptor stays unmodified; the caller destroys it.
    /// </summary>
    internal nint CreateTemporaryDescriptor()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return CuSparseNative.CreateCoo(
            Rows, Cols, Values.Length,
            RowIndices.Pointer, ColIndices.Pointer, Values.Pointer,
            CuSparseIndexType.Int32, CuSparseIndexBase.Zero,
            CudaRuntime.GetCudaDataType<T>());
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
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(cols);
        if (rowIndices.Length < values.Length || colIndices.Length < values.Length)
        {
            throw new ArgumentException($"rowIndices and colIndices must each hold at least {values.Length} elements (one per value).", nameof(rowIndices));
        }

        Rows = rows;
        Cols = cols;
        RowIndices = rowIndices;
        ColIndices = colIndices;
        Values = values;

        var cudaType = CudaRuntime.GetCudaDataType<T>();
        _descriptor = CuSparseNative.CreateCoo(
            rows, cols, values.Length,
            rowIndices.Pointer,
            colIndices.Pointer,
            values.Pointer,
            CuSparseIndexType.Int32,
            CuSparseIndexBase.Zero,
            cudaType);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (_descriptor != default)
        {
            CuSparseNative.DestroySpMat(_descriptor);
            _descriptor = default;
        }
        Values.Dispose();
        ColIndices.Dispose();
        RowIndices.Dispose();
    }
}
