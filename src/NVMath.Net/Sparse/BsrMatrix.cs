// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using System.Numerics;
using NVMathNet.Interop;

namespace NVMathNet.Sparse;

/// <summary>
/// A sparse matrix stored in Block Sparse Row (BSR) format on the GPU.
/// All non-zero values are grouped into dense sub-blocks of a fixed size
/// (<see cref="RowBlockDim"/> × <see cref="ColBlockDim"/>), which enables
/// better memory coalescing and higher throughput than element-level CSR on modern GPUs.
/// </summary>
/// <typeparam name="T">Unmanaged numeric value type (float, double, etc.).</typeparam>
public sealed unsafe class BsrMatrix<T> : IDisposable where T : unmanaged, INumberBase<T>
{
    private nint _descriptor;
    private bool _disposed;

    /// <summary>Number of block rows.</summary>
    public long BlockRows { get; }

    /// <summary>Number of block columns.</summary>
    public long BlockCols { get; }

    /// <summary>Number of non-zero blocks.</summary>
    public long BlockNNZ => Values.Length / ((long)RowBlockDim * ColBlockDim);

    /// <summary>Number of rows in each block.</summary>
    public int RowBlockDim { get; }

    /// <summary>Number of columns in each block.</summary>
    public int ColBlockDim { get; }

    /// <summary>Total matrix row count (<c>BlockRows * RowBlockDim</c>).</summary>
    public long Rows => BlockRows * RowBlockDim;

    /// <summary>Total matrix column count (<c>BlockCols * ColBlockDim</c>).</summary>
    public long Cols => BlockCols * ColBlockDim;

    /// <summary>Block row offsets — length BlockRows+1.</summary>
    public DeviceBuffer<int> RowOffsets { get; }

    /// <summary>Block column indices — length BlockNNZ.</summary>
    public DeviceBuffer<int> ColIndices { get; }

    /// <summary>
    /// Non-zero block values — length BlockNNZ × RowBlockDim × ColBlockDim.
    /// Blocks are stored in row-major order by default.
    /// </summary>
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
    /// Creates a BSR sparse matrix from pre-allocated device buffers.
    /// </summary>
    /// <param name="blockRows">Number of block rows.</param>
    /// <param name="blockCols">Number of block columns.</param>
    /// <param name="rowBlockDim">Number of rows in each dense block.</param>
    /// <param name="colBlockDim">Number of columns in each dense block.</param>
    /// <param name="rowOffsets">Block row offsets (length <paramref name="blockRows"/>+1).</param>
    /// <param name="colIndices">Block column indices (length BlockNNZ).</param>
    /// <param name="values">Non-zero block values (length BlockNNZ × rowBlockDim × colBlockDim).</param>
    /// <param name="columnMajorBlocks">
    /// <c>true</c> if block data is column-major; <c>false</c> (default) for row-major.
    /// </param>
    public BsrMatrix(
        long blockRows, long blockCols,
        int rowBlockDim, int colBlockDim,
        DeviceBuffer<int> rowOffsets,
        DeviceBuffer<int> colIndices,
        DeviceBuffer<T>   values,
        bool columnMajorBlocks = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockRows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockCols);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rowBlockDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(colBlockDim);
        ArgumentNullException.ThrowIfNull(rowOffsets);
        ArgumentNullException.ThrowIfNull(colIndices);
        ArgumentNullException.ThrowIfNull(values);

        BlockRows   = blockRows;
        BlockCols   = blockCols;
        RowBlockDim = rowBlockDim;
        ColBlockDim = colBlockDim;
        RowOffsets  = rowOffsets;
        ColIndices  = colIndices;
        Values      = values;

        long bnnz = colIndices.Length;
        var cudaType = NativeCudaType();

        _descriptor = CuSparseNative.CreateBsr(
            blockRows, blockCols, bnnz,
            rowBlockDim, colBlockDim,
            rowOffsets.Pointer, colIndices.Pointer, values.Pointer,
            CuSparseIndexType.Int32, CuSparseIndexType.Int32,
            CuSparseIndexBase.Zero, cudaType,
            columnMajorBlocks);
    }

    private static CudaDataType NativeCudaType()
    {
        if (typeof(T) == typeof(float))  return CudaDataType.R_32F;
        if (typeof(T) == typeof(double)) return CudaDataType.R_64F;
        if (typeof(T) == typeof(Half))   return CudaDataType.R_16F;
        throw new NotSupportedException($"Unsupported element type {typeof(T)} for BSR matrix.");
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_descriptor != default)
            CuSparseNative.DestroySpMat(_descriptor);
    }
}
