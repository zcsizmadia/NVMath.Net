// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Interop;
using NVMathNet.Memory;

namespace NVMathNet.Sparse;

/// <summary>Which triangle of a sparse matrix a triangular solve reads.</summary>
public enum SparseTriangle
{
    /// <summary>Lower triangle (including the diagonal unless <see cref="SparseDiagonal.Unit"/>).</summary>
    Lower = 0,
    /// <summary>Upper triangle (including the diagonal unless <see cref="SparseDiagonal.Unit"/>).</summary>
    Upper = 1,
}

/// <summary>Whether the diagonal of a triangular matrix is stored or implicitly all ones.</summary>
public enum SparseDiagonal
{
    /// <summary>Diagonal values are read from the matrix.</summary>
    NonUnit = 0,
    /// <summary>The diagonal is assumed to be all ones.</summary>
    Unit = 1,
}

// cuSPARSE generic API: SpSV, SpSM, SDDMM, SpGEMM.
public static partial class SparseLinearAlgebra
{
    // -- SpSV ---------------------------------------------------------------

    /// <summary>
    /// Sparse triangular solve with a vector right-hand side: solves
    /// <c>op(A) · y = alpha · x</c> for <paramref name="y"/>.
    /// </summary>
    /// <param name="matrix">Square sparse CSR matrix A whose <paramref name="triangle"/> is referenced.</param>
    /// <param name="x">Right-hand side vector (device).</param>
    /// <param name="y">Solution vector (device); must not alias <paramref name="x"/>.</param>
    /// <param name="triangle">Triangle of A to read.</param>
    /// <param name="diagonal">Whether the diagonal is implicit ones.</param>
    /// <param name="alpha">Scale applied to x. Default: 1.</param>
    /// <param name="transpose">Solve with the transpose of A.</param>
    /// <param name="stream">Optional CUDA stream. <c>null</c> creates a private stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task SpSVAsync<T>(
        CsrMatrix<T> matrix,
        DeviceBuffer<T> x,
        DeviceBuffer<T> y,
        SparseTriangle triangle = SparseTriangle.Lower,
        SparseDiagonal diagonal = SparseDiagonal.NonUnit,
        T? alpha = null,
        bool transpose = false,
        CudaStream? stream = null,
        CancellationToken ct = default) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(matrix);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (matrix.Rows != matrix.Cols)
        {
            throw new ArgumentException("SpSV requires a square matrix.", nameof(matrix));
        }

        await RunAsync(stream, ct, s => SpSVCore(matrix, x, y, triangle, diagonal, alpha ?? T.One, transpose, s)).ConfigureAwait(false);
    }

    private static unsafe void SpSVCore<T>(
        CsrMatrix<T> matrix, DeviceBuffer<T> x, DeviceBuffer<T> y,
        SparseTriangle triangle, SparseDiagonal diagonal, T alpha, bool transpose, CudaStream s)
        where T : unmanaged, INumberBase<T>
    {
        var cudaType = CudaRuntime.GetCudaDataType<T>();
        var opA = transpose ? CuSparseOperation.Transpose : CuSparseOperation.None;
        var handle = CreateHandle(s);
        nint vecX = 0, vecY = 0, descr = 0;
        void* buffer = null;
        Workspace ws = default;
        try
        {
            CuSparseNative.SpMatSetAttribute(matrix.Descriptor, CuSparseSpMatAttribute.FillMode, (int)triangle);
            CuSparseNative.SpMatSetAttribute(matrix.Descriptor, CuSparseSpMatAttribute.DiagType, (int)diagonal);
            vecX = CuSparseNative.CreateDnVec(x.Length, x.Pointer, cudaType);
            vecY = CuSparseNative.CreateDnVec(y.Length, y.Pointer, cudaType);
            descr = CuSparseNative.SpSVCreateDescr();

            nuint size = CuSparseNative.SpSVBufferSize(handle, opA, &alpha, matrix.Descriptor, vecX, vecY, cudaType, CuSparseSpSVAlg.Default, descr);
            ws = new Workspace(size);
            buffer = ws.Pointer;
            CuSparseNative.SpSVAnalysis(handle, opA, &alpha, matrix.Descriptor, vecX, vecY, cudaType, CuSparseSpSVAlg.Default, descr, buffer);
            CuSparseNative.SpSVSolve(handle, opA, &alpha, matrix.Descriptor, vecX, vecY, cudaType, CuSparseSpSVAlg.Default, descr);
            s.Synchronize();
        }
        finally
        {
            if (descr != 0)
            {
                CuSparseNative.SpSVDestroyDescr(descr);
            }

            if (vecX != 0)
            {
                CuSparseNative.DestroyDnVec(vecX);
            }

            if (vecY != 0)
            {
                CuSparseNative.DestroyDnVec(vecY);
            }

            ws.Dispose();

            CuSparseNative.Destroy(handle);
        }
    }

    // -- SpSM ---------------------------------------------------------------

    /// <summary>
    /// Sparse triangular solve with a dense matrix right-hand side: solves
    /// <c>op(A) · C = alpha · B</c> for <paramref name="c"/>.
    /// </summary>
    /// <param name="matrix">Square sparse CSR matrix A (m × m).</param>
    /// <param name="b">Dense right-hand side (m × n), column-major on device.</param>
    /// <param name="c">Dense solution (m × n), column-major on device.</param>
    /// <param name="n">Number of right-hand-side columns.</param>
    /// <param name="triangle">Triangle of A to read.</param>
    /// <param name="diagonal">Whether the diagonal is implicit ones.</param>
    /// <param name="alpha">Scale applied to B. Default: 1.</param>
    /// <param name="transpose">Solve with the transpose of A.</param>
    /// <param name="stream">Optional CUDA stream. <c>null</c> creates a private stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task SpSMAsync<T>(
        CsrMatrix<T> matrix,
        DeviceBuffer<T> b,
        DeviceBuffer<T> c,
        long n,
        SparseTriangle triangle = SparseTriangle.Lower,
        SparseDiagonal diagonal = SparseDiagonal.NonUnit,
        T? alpha = null,
        bool transpose = false,
        CudaStream? stream = null,
        CancellationToken ct = default) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(matrix);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);
        if (matrix.Rows != matrix.Cols)
        {
            throw new ArgumentException("SpSM requires a square matrix.", nameof(matrix));
        }

        await RunAsync(stream, ct, s => SpSMCore(matrix, b, c, n, triangle, diagonal, alpha ?? T.One, transpose, s)).ConfigureAwait(false);
    }

    private static unsafe void SpSMCore<T>(
        CsrMatrix<T> matrix, DeviceBuffer<T> b, DeviceBuffer<T> c, long n,
        SparseTriangle triangle, SparseDiagonal diagonal, T alpha, bool transpose, CudaStream s)
        where T : unmanaged, INumberBase<T>
    {
        var cudaType = CudaRuntime.GetCudaDataType<T>();
        var opA = transpose ? CuSparseOperation.Transpose : CuSparseOperation.None;
        long m = matrix.Rows;
        var handle = CreateHandle(s);
        nint matB = 0, matC = 0, descr = 0;
        void* buffer = null;
        Workspace ws = default;
        try
        {
            CuSparseNative.SpMatSetAttribute(matrix.Descriptor, CuSparseSpMatAttribute.FillMode, (int)triangle);
            CuSparseNative.SpMatSetAttribute(matrix.Descriptor, CuSparseSpMatAttribute.DiagType, (int)diagonal);
            matB = CuSparseNative.CreateDnMat(m, n, m, b.Pointer, cudaType, ColMajor);
            matC = CuSparseNative.CreateDnMat(m, n, m, c.Pointer, cudaType, ColMajor);
            descr = CuSparseNative.SpSMCreateDescr();

            nuint size = CuSparseNative.SpSMBufferSize(handle, opA, CuSparseOperation.None, &alpha, matrix.Descriptor, matB, matC, cudaType, CuSparseSpSMAlg.Default, descr);
            ws = new Workspace(size);
            buffer = ws.Pointer;
            CuSparseNative.SpSMAnalysis(handle, opA, CuSparseOperation.None, &alpha, matrix.Descriptor, matB, matC, cudaType, CuSparseSpSMAlg.Default, descr, buffer);
            CuSparseNative.SpSMSolve(handle, opA, CuSparseOperation.None, &alpha, matrix.Descriptor, matB, matC, cudaType, CuSparseSpSMAlg.Default, descr);
            s.Synchronize();
        }
        finally
        {
            if (descr != 0)
            {
                CuSparseNative.SpSMDestroyDescr(descr);
            }

            if (matB != 0)
            {
                CuSparseNative.DestroyDnMat(matB);
            }

            if (matC != 0)
            {
                CuSparseNative.DestroyDnMat(matC);
            }

            ws.Dispose();

            CuSparseNative.Destroy(handle);
        }
    }

    // -- SDDMM --------------------------------------------------------------

    /// <summary>
    /// Sampled dense-dense matrix product:
    /// <c>C = alpha · (A · B) ∘ spy(C) + beta · C</c>, where only the entries in the
    /// sparsity pattern of C are computed and C's values are updated in place.
    /// </summary>
    /// <param name="a">Dense A (m × k), column-major on device.</param>
    /// <param name="b">Dense B (k × n), column-major on device.</param>
    /// <param name="c">Sparse CSR matrix C (m × n); its values are updated in place.</param>
    /// <param name="k">Inner dimension.</param>
    /// <param name="alpha">Scale for the product. Default: 1.</param>
    /// <param name="beta">Scale for the existing values of C. Default: 0.</param>
    /// <param name="stream">Optional CUDA stream. <c>null</c> creates a private stream.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task SddmmAsync<T>(
        DeviceBuffer<T> a,
        DeviceBuffer<T> b,
        CsrMatrix<T> c,
        long k,
        T? alpha = null,
        T? beta = null,
        CudaStream? stream = null,
        CancellationToken ct = default) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);

        await RunAsync(stream, ct, s => SddmmCore(a, b, c, k, alpha ?? T.One, beta ?? T.Zero, s)).ConfigureAwait(false);
    }

    private static unsafe void SddmmCore<T>(
        DeviceBuffer<T> a, DeviceBuffer<T> b, CsrMatrix<T> c, long k, T alpha, T beta, CudaStream s)
        where T : unmanaged, INumberBase<T>
    {
        var cudaType = CudaRuntime.GetCudaDataType<T>();
        long m = c.Rows, n = c.Cols;
        var handle = CreateHandle(s);
        nint matA = 0, matB = 0;
        void* buffer = null;
        Workspace ws = default;
        try
        {
            matA = CuSparseNative.CreateDnMat(m, k, m, a.Pointer, cudaType, ColMajor);
            matB = CuSparseNative.CreateDnMat(k, n, k, b.Pointer, cudaType, ColMajor);

            nuint size = CuSparseNative.SddmmBufferSize(handle, CuSparseOperation.None, CuSparseOperation.None, &alpha, matA, matB, &beta, c.Descriptor, cudaType, CuSparseSddmmAlg.Default);
            ws = new Workspace(size);
            buffer = ws.Pointer;
            CuSparseNative.SddmmPreprocess(handle, CuSparseOperation.None, CuSparseOperation.None, &alpha, matA, matB, &beta, c.Descriptor, cudaType, CuSparseSddmmAlg.Default, buffer);
            CuSparseNative.Sddmm(handle, CuSparseOperation.None, CuSparseOperation.None, &alpha, matA, matB, &beta, c.Descriptor, cudaType, CuSparseSddmmAlg.Default, buffer);
            s.Synchronize();
        }
        finally
        {
            if (matA != 0)
            {
                CuSparseNative.DestroyDnMat(matA);
            }

            if (matB != 0)
            {
                CuSparseNative.DestroyDnMat(matB);
            }

            ws.Dispose();

            CuSparseNative.Destroy(handle);
        }
    }

    // -- SpGEMM -------------------------------------------------------------

    /// <summary>
    /// Sparse-sparse matrix product: returns a new CSR matrix <c>C = alpha · A · B</c>.
    /// </summary>
    /// <param name="a">Sparse CSR matrix A (m × k).</param>
    /// <param name="b">Sparse CSR matrix B (k × n).</param>
    /// <param name="alpha">Scale for the product. Default: 1.</param>
    /// <param name="stream">Optional CUDA stream. <c>null</c> creates a private stream.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>The product C; the caller owns and must dispose it.</returns>
    public static async Task<CsrMatrix<T>> SpGemmAsync<T>(
        CsrMatrix<T> a,
        CsrMatrix<T> b,
        T? alpha = null,
        CudaStream? stream = null,
        CancellationToken ct = default) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.Cols != b.Rows)
        {
            throw new ArgumentException($"Dimension mismatch: A is {a.Rows}x{a.Cols}, B is {b.Rows}x{b.Cols}.");
        }

        CsrMatrix<T>? result = null;
        await RunAsync(stream, ct, s => result = SpGemmCore(a, b, alpha ?? T.One, s)).ConfigureAwait(false);
        return result!;
    }

    private static unsafe CsrMatrix<T> SpGemmCore<T>(CsrMatrix<T> a, CsrMatrix<T> b, T alpha, CudaStream s)
        where T : unmanaged, INumberBase<T>
    {
        var cudaType = CudaRuntime.GetCudaDataType<T>();
        var op = CuSparseOperation.None;
        var alg = CuSparseSpGemmAlg.Default;
        T beta = T.Zero;
        var handle = CreateHandle(s);
        nint matC = 0, descr = 0;
        void* buf1 = null, buf2 = null;
        Workspace ws1 = default, ws2 = default;
        DeviceBuffer<int>? offsets = null, cols = null;
        DeviceBuffer<T>? vals = null;
        try
        {
            // C starts empty; its arrays are bound once the result size is known.
            matC = CuSparseNative.CreateCsr(
                a.Rows, b.Cols, 0, null, null, null,
                CuSparseIndexType.Int32, CuSparseIndexType.Int32, CuSparseIndexBase.Zero, cudaType);
            descr = CuSparseNative.SpGemmCreateDescr();

            nuint size1 = 0;
            CuSparseNative.SpGemmWorkEstimation(handle, op, op, &alpha, a.Descriptor, b.Descriptor, &beta, matC, cudaType, alg, descr, ref size1, null);
            ws1 = new Workspace(size1);
            buf1 = ws1.Pointer;
            CuSparseNative.SpGemmWorkEstimation(handle, op, op, &alpha, a.Descriptor, b.Descriptor, &beta, matC, cudaType, alg, descr, ref size1, buf1);

            nuint size2 = 0;
            CuSparseNative.SpGemmCompute(handle, op, op, &alpha, a.Descriptor, b.Descriptor, &beta, matC, cudaType, alg, descr, ref size2, null);
            ws2 = new Workspace(size2);
            buf2 = ws2.Pointer;
            CuSparseNative.SpGemmCompute(handle, op, op, &alpha, a.Descriptor, b.Descriptor, &beta, matC, cudaType, alg, descr, ref size2, buf2);

            CuSparseNative.SpMatGetSize(matC, out long rows, out long colCount, out long nnz);
            if (nnz == 0)
            {
                throw new InvalidOperationException("The sparse product has no non-zero entries.");
            }

            offsets = new DeviceBuffer<int>(rows + 1);
            cols = new DeviceBuffer<int>(nnz);
            vals = new DeviceBuffer<T>(nnz);
            CuSparseNative.CsrSetPointers(matC, offsets.Pointer, cols.Pointer, vals.Pointer);
            CuSparseNative.SpGemmCopy(handle, op, op, &alpha, a.Descriptor, b.Descriptor, &beta, matC, cudaType, alg, descr);
            s.Synchronize();

            // The CsrMatrix takes ownership of the buffers.
            var result = new CsrMatrix<T>(rows, colCount, offsets, cols, vals);
            offsets = null;
            cols = null;
            vals = null;
            return result;
        }
        finally
        {
            offsets?.Dispose();
            cols?.Dispose();
            vals?.Dispose();
            if (descr != 0)
            {
                CuSparseNative.SpGemmDestroyDescr(descr);
            }

            if (matC != 0)
            {
                CuSparseNative.DestroySpMat(matC);
            }

            ws1.Dispose();

            ws2.Dispose();

            CuSparseNative.Destroy(handle);
        }
    }

    // -- SpMV / SpMM for COO and BSR ----------------------------------------

    /// <summary>Sparse matrix-vector product <c>y = alpha · op(A) · x + beta · y</c> for a COO matrix.</summary>
    /// <inheritdoc cref="SpMVAsync{T}(CsrMatrix{T}, DeviceBuffer{T}, DeviceBuffer{T}, float, float, bool, CudaStream, CancellationToken)"/>
    public static Task SpMVAsync<T>(
        CooMatrix<T> matrix, DeviceBuffer<T> x, DeviceBuffer<T> y,
        float alpha = 1.0f, float beta = 0.0f, bool transpose = false,
        CudaStream? stream = null, CancellationToken ct = default) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(matrix);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        return RunAsync(stream, ct, s => SpMVCore(matrix.Descriptor, x, y, alpha, beta, transpose, s));
    }

    /// <summary>Sparse matrix-vector product <c>y = alpha · op(A) · x + beta · y</c> for a BSR matrix.</summary>
    /// <inheritdoc cref="SpMVAsync{T}(CsrMatrix{T}, DeviceBuffer{T}, DeviceBuffer{T}, float, float, bool, CudaStream, CancellationToken)"/>
    public static Task SpMVAsync<T>(
        BsrMatrix<T> matrix, DeviceBuffer<T> x, DeviceBuffer<T> y,
        float alpha = 1.0f, float beta = 0.0f, bool transpose = false,
        CudaStream? stream = null, CancellationToken ct = default) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(matrix);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        return RunAsync(stream, ct, s => SpMVCore(matrix.Descriptor, x, y, alpha, beta, transpose, s));
    }

    /// <summary>Sparse-dense matrix product <c>C = alpha · op(A) · B + beta · C</c> for a COO matrix.</summary>
    /// <inheritdoc cref="SpMMAsync{T}(CsrMatrix{T}, DeviceBuffer{T}, DeviceBuffer{T}, long, float, float, bool, CudaStream, CancellationToken)"/>
    public static Task SpMMAsync<T>(
        CooMatrix<T> matrix, DeviceBuffer<T> b, DeviceBuffer<T> c, long n,
        float alpha = 1.0f, float beta = 0.0f, bool transpose = false,
        CudaStream? stream = null, CancellationToken ct = default) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(matrix);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);
        return RunAsync(stream, ct, s => SpMMCore(matrix.Descriptor, matrix.Rows, matrix.Cols, b, c, n, alpha, beta, transpose, s));
    }

    /// <summary>Sparse-dense matrix product <c>C = alpha · op(A) · B + beta · C</c> for a BSR matrix.</summary>
    /// <inheritdoc cref="SpMMAsync{T}(CsrMatrix{T}, DeviceBuffer{T}, DeviceBuffer{T}, long, float, float, bool, CudaStream, CancellationToken)"/>
    public static Task SpMMAsync<T>(
        BsrMatrix<T> matrix, DeviceBuffer<T> b, DeviceBuffer<T> c, long n,
        float alpha = 1.0f, float beta = 0.0f, bool transpose = false,
        CudaStream? stream = null, CancellationToken ct = default) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(matrix);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);
        return RunAsync(stream, ct, s => SpMMCore(matrix.Descriptor, matrix.Rows, matrix.Cols, b, c, n, alpha, beta, transpose, s));
    }

    // -- Helpers ------------------------------------------------------------

    private const int ColMajor = 1; // CUSPARSE_ORDER_COL

    private static async Task RunAsync(CudaStream? stream, CancellationToken ct, Action<CudaStream> body)
    {
        bool ownStream = stream is null;
        CudaStream s = stream ?? new CudaStream();
        try
        {
            body(s);
            await s.SynchronizeAsync(ct).ConfigureAwait(false);
        }
        finally
        {
            if (ownStream)
            {
                s.Dispose();
            }
        }
    }
}
