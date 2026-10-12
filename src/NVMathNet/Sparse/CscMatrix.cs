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
    public long NNZ { get; }

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
    /// <param name="nnz">
    /// Number of stored entries when it is smaller than the buffer length (for example <c>0</c> for an
    /// empty matrix backed by one-element placeholder buffers); <c>null</c> uses the length of <paramref name="values"/>.
    /// </param>
    public unsafe CscMatrix(
        long rows, long cols,
        DeviceBuffer<int> colOffsets,
        DeviceBuffer<int> rowIndices,
        DeviceBuffer<T> values,
        long? nnz = null)
    {
        ArgumentNullException.ThrowIfNull(colOffsets);
        ArgumentNullException.ThrowIfNull(rowIndices);
        ArgumentNullException.ThrowIfNull(values);
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(cols);

        long count = nnz ?? values.Length;
        ArgumentOutOfRangeException.ThrowIfNegative(count, nameof(nnz));
        if (colOffsets.Length < cols + 1)
        {
            throw new ArgumentException($"colOffsets holds {colOffsets.Length} elements but cols + 1 = {cols + 1} are required.", nameof(colOffsets));
        }

        if (rowIndices.Length < count || values.Length < count)
        {
            throw new ArgumentException($"rowIndices and values must each hold at least {count} elements.", nameof(values));
        }

        NNZ = count;
        Rows = rows;
        Cols = cols;
        ColOffsets = colOffsets;
        RowIndices = rowIndices;
        Values = values;

        _descriptor = CuSparseNative.CreateCsc(
            rows, cols, count,
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
