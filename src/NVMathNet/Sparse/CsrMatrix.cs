// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Interop;
using NVMathNet.Memory;

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
    private readonly long _nnz;
    private bool _disposed;

    /// <summary>Number of rows.</summary>
    public long Rows { get; }
    /// <summary>Number of columns.</summary>
    public long Cols { get; }
    /// <summary>Number of non-zero elements.</summary>
    public long NNZ => _nnz;

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
    /// Creates an additional descriptor over the same arrays. Operations that attach attributes to the descriptor
    /// (triangular solves) use it so the matrix's own descriptor stays unmodified; the caller destroys it.
    /// </summary>
    internal nint CreateTemporaryDescriptor()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return CuSparseNative.CreateCsr(
            Rows, Cols, _nnz,
            RowOffsets.Pointer, ColIndices.Pointer, Values.Pointer,
            CuSparseIndexType.Int32, CuSparseIndexType.Int32, CuSparseIndexBase.Zero,
            CudaRuntime.GetCudaDataType<T>());
    }

    /// <summary>
    /// Creates a CSR sparse matrix from pre-allocated device buffers.
    /// </summary>
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns.</param>
    /// <param name="rowOffsets">Row offsets (length rows+1).</param>
    /// <param name="colIndices">Column indices (length NNZ).</param>
    /// <param name="values">Non-zero values (length NNZ).</param>
    /// <param name="nnz">
    /// Number of stored entries when it is smaller than the buffer length (for example <c>0</c> for an
    /// empty matrix backed by one-element placeholder buffers); <c>null</c> uses the length of <paramref name="values"/>.
    /// </param>
    public CsrMatrix(
        long rows, long cols,
        DeviceBuffer<int> rowOffsets,
        DeviceBuffer<int> colIndices,
        DeviceBuffer<T> values,
        long? nnz = null)
    {
        ArgumentNullException.ThrowIfNull(rowOffsets);
        ArgumentNullException.ThrowIfNull(colIndices);
        ArgumentNullException.ThrowIfNull(values);
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(cols);

        long count = nnz ?? values.Length;
        ArgumentOutOfRangeException.ThrowIfNegative(count, nameof(nnz));
        if (rowOffsets.Length < rows + 1)
        {
            throw new ArgumentException($"rowOffsets holds {rowOffsets.Length} elements but rows + 1 = {rows + 1} are required.", nameof(rowOffsets));
        }

        if (colIndices.Length < count || values.Length < count)
        {
            throw new ArgumentException($"colIndices and values must each hold at least {count} elements.", nameof(values));
        }

        Rows = rows;
        Cols = cols;
        RowOffsets = rowOffsets;
        ColIndices = colIndices;
        Values = values;
        _nnz = count;

        var cudaType = CudaRuntime.GetCudaDataType<T>();
        _descriptor = CuSparseNative.CreateCsr(
            rows, cols, _nnz,
            rowOffsets.Pointer,
            colIndices.Pointer,
            values.Pointer,
            CuSparseIndexType.Int32,
            CuSparseIndexType.Int32,
            CuSparseIndexBase.Zero,
            cudaType);
    }

    /// <summary>
    /// Returns a new CSR matrix holding the transpose of this matrix
    /// (equivalently, this matrix converted to CSC). The caller owns the result.
    /// </summary>
    public CsrMatrix<T> Transpose()
    {
        var (colPtr, rowInd, vals) = TransposeBuffers();
        return new CsrMatrix<T>(Cols, Rows, colPtr, rowInd, vals, NNZ);
    }

    /// <summary>
    /// Returns this matrix converted to CSC. The caller owns the result.
    /// </summary>
    public CscMatrix<T> ToCsc()
    {
        var (colPtr, rowInd, vals) = TransposeBuffers();
        return new CscMatrix<T>(Rows, Cols, colPtr, rowInd, vals, NNZ);
    }

    private (DeviceBuffer<int> ColOffsets, DeviceBuffer<int> RowIndices, DeviceBuffer<T> Values) TransposeBuffers()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (Rows > int.MaxValue || Cols > int.MaxValue || NNZ > int.MaxValue)
        {
            throw new NotSupportedException("Transpose supports 32-bit indexed matrices only.");
        }

        var cudaType = CudaRuntime.GetCudaDataType<T>();
        var colPtr = new DeviceBuffer<int>(Cols + 1);
        // At least one element, so an empty matrix does not need a zero-sized allocation.
        var rowInd = new DeviceBuffer<int>(Math.Max(NNZ, 1));
        var vals = new DeviceBuffer<T>(Math.Max(NNZ, 1));
        nint handle = CuSparseNative.Create();
        void* buffer = null;
        Workspace ws = default;
        try
        {
            nuint size = CuSparseNative.Csr2CscEx2BufferSize(
                handle, (int)Rows, (int)Cols, (int)NNZ,
                Values.Pointer, (int*)RowOffsets.Pointer, (int*)ColIndices.Pointer,
                vals.Pointer, (int*)colPtr.Pointer, (int*)rowInd.Pointer, cudaType);
            ws = new Workspace(size);
            buffer = ws.Pointer;
            CuSparseNative.Csr2CscEx2(
                handle, (int)Rows, (int)Cols, (int)NNZ,
                Values.Pointer, (int*)RowOffsets.Pointer, (int*)ColIndices.Pointer,
                vals.Pointer, (int*)colPtr.Pointer, (int*)rowInd.Pointer, cudaType, buffer);
            CudaRuntime.DeviceSynchronize();

            // The CSC arrays of A are the CSR arrays of A^T.
            return (colPtr, rowInd, vals);
        }
        catch
        {
            colPtr.Dispose();
            rowInd.Dispose();
            vals.Dispose();
            throw;
        }
        finally
        {
            ws.Dispose();

            CuSparseNative.Destroy(handle);
        }
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
        RowOffsets.Dispose();
    }
}
