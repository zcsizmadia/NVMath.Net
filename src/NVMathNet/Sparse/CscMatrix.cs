// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Interop;

namespace NVMathNet.Sparse;

/// <summary>
/// A sparse matrix stored in Compressed Sparse Column (CSC) format on the GPU.
/// Column offsets, row indices and values are each held in a separate <see cref="DeviceBuffer{T}"/>.
/// Obtain one from a CSR matrix with <see cref="CsrMatrix{T}.ToCsc"/>.
/// </summary>
/// <typeparam name="T">Unmanaged numeric value element type (float, double, etc.).</typeparam>
public sealed class CscMatrix<T> : IDisposable where T : unmanaged, INumberBase<T>
{
    private nint _descriptor;
    private bool _disposed;

    /// <summary>Number of rows.</summary>
    public long Rows { get; }

    /// <summary>Number of columns.</summary>
    public long Cols { get; }

    /// <summary>Number of non-zero elements.</summary>
    public long NNZ => Values.Length;

    /// <summary>Column offsets — length Cols+1.</summary>
    public DeviceBuffer<int> ColOffsets { get; }

    /// <summary>Row indices — length NNZ.</summary>
    public DeviceBuffer<int> RowIndices { get; }

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
    /// Creates a CSC sparse matrix from pre-allocated device buffers; the matrix takes ownership of them.
    /// </summary>
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns.</param>
    /// <param name="colOffsets">Column offsets (length cols+1).</param>
    /// <param name="rowIndices">Row indices (length NNZ).</param>
    /// <param name="values">Non-zero values (length NNZ).</param>
    public unsafe CscMatrix(
        long rows, long cols,
        DeviceBuffer<int> colOffsets,
        DeviceBuffer<int> rowIndices,
        DeviceBuffer<T> values)
    {
        ArgumentNullException.ThrowIfNull(colOffsets);
        ArgumentNullException.ThrowIfNull(rowIndices);
        ArgumentNullException.ThrowIfNull(values);

        Rows = rows;
        Cols = cols;
        ColOffsets = colOffsets;
        RowIndices = rowIndices;
        Values = values;

        _descriptor = CuSparseNative.CreateCsc(
            rows, cols, values.Length,
            colOffsets.Pointer, rowIndices.Pointer, values.Pointer,
            CuSparseIndexType.Int32, CuSparseIndexType.Int32, CuSparseIndexBase.Zero,
            CudaRuntime.GetCudaDataType<T>());
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
        RowIndices.Dispose();
        ColOffsets.Dispose();
    }
}
