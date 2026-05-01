// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.InteropServices;

namespace NVMathNet.Interop;

// TODO: VALIDATE ENUMS

/// <summary>cuSPARSE status codes.</summary>
public enum CuSparseStatus : int
{
    /// <summary>Operation completed successfully.</summary>
    Success                  = 0,
    /// <summary>cuSPARSE library not initialised.</summary>
    NotInitialized           = 1,
    /// <summary>GPU memory allocation failed.</summary>
    AllocFailed              = 2,
    /// <summary>A parameter has an invalid value.</summary>
    InvalidValue             = 3,
    /// <summary>GPU architecture does not support the operation.</summary>
    ArchMismatch             = 4,
    /// <summary>GPU memory mapping failed.</summary>
    MappingError             = 5,
    /// <summary>Execution on the GPU failed.</summary>
    ExecutionFailed          = 6,
    /// <summary>Internal driver error.</summary>
    InternalError            = 7,
    /// <summary>Matrix type is not supported.</summary>
    MatrixTypeNotSupported   = 8,
    /// <summary>A zero pivot was encountered during factorisation.</summary>
    ZeroPivot                = 9,
    /// <summary>Operation is not supported with the given parameters.</summary>
    NotSupported             = 10,
    /// <summary>Insufficient GPU resources to complete the operation.</summary>
    InsufficientResources    = 11,
}

/// <summary>cuSPARSE sparse matrix format.</summary>
public enum CuSparseFormat : int
{
    /// <summary>Compressed Sparse Row.</summary>
    Csr          = 1,
    /// <summary>Compressed Sparse Column.</summary>
    Csc          = 2,
    /// <summary>Coordinate (COO) sorted by row.</summary>
    Coo          = 3,
    /// <summary>Blocked Ellpack.</summary>
    BlockedEll   = 5,
    /// <summary>Fixed-size Block Sparse Row.</summary>
    BsrFixed     = 6,
    /// <summary>Sliced Ellpack.</summary>
    Sliced_Ell   = 7,
}

/// <summary>cuSPARSE index types.</summary>
public enum CuSparseIndexType : int
{
    /// <summary>16-bit integer indices.</summary>
    Int16 = 1,
    /// <summary>32-bit integer indices.</summary>
    Int32 = 2,
    /// <summary>64-bit integer indices.</summary>
    Int64 = 3,
}

/// <summary>cuSPARSE index base.</summary>
public enum CuSparseIndexBase : int
{
    /// <summary>Zero-based (C-style) indexing.</summary>
    Zero = 0,
    /// <summary>One-based (Fortran-style) indexing.</summary>
    One  = 1,
}

/// <summary>cuSPARSE SpMM algorithms.</summary>
public enum CuSparseSpMMAlg : int
{
    /// <summary>Auto-select the best algorithm.</summary>
    Default        = 0,
    /// <summary>COO SpMM algorithm 1.</summary>
    CooAlg1        = 1,
    /// <summary>COO SpMM algorithm 2.</summary>
    CooAlg2        = 2,
    /// <summary>COO SpMM algorithm 3.</summary>
    CooAlg3        = 3,
    /// <summary>CSR SpMM algorithm 1.</summary>
    CsrAlg1        = 4,
    /// <summary>COO SpMM algorithm 4.</summary>
    CooAlg4        = 5,
    /// <summary>CSR SpMM algorithm 2.</summary>
    CsrAlg2        = 6,
    /// <summary>CSR SpMM algorithm 3.</summary>
    CsrAlg3        = 12,
    /// <summary>Blocked ELL SpMM algorithm 1.</summary>
    BlockedEllAlg1 = 13,
    /// <summary>BSR SpMM algorithm 1.</summary>
    BsrAlg1        = 14,
}

/// <summary>cuSPARSE SpMV algorithms.</summary>
public enum CuSparseSpMVAlg : int
{
    /// <summary>Auto-select the best algorithm.</summary>
    Default  = 0,
    /// <summary>COO SpMV algorithm 1.</summary>
    CooAlg1  = 1,
    /// <summary>CSR SpMV algorithm 1.</summary>
    CsrAlg1  = 2,
    /// <summary>CSR SpMV algorithm 2.</summary>
    CsrAlg2  = 3,
    /// <summary>COO SpMV algorithm 2.</summary>
    CooAlg2  = 4,
    /// <summary>Sliced ELL SpMV algorithm 1.</summary>
    SellAlg1 = 5,
    /// <summary>BSR SpMV algorithm 1.</summary>
    BsrAlg1  = 6,
}

/// <summary>cuSPARSE operation type applied to a sparse or dense matrix operand.</summary>
public enum CuSparseOperation : int
{
    /// <summary>Use the matrix as-is (no transpose).</summary>
    None         = 0,
    /// <summary>Use the transpose of the matrix.</summary>
    Transpose    = 1,
    /// <summary>Use the conjugate transpose (Hermitian) of the matrix.</summary>
    ConjTranspose = 2,
}

/// <summary>
/// Raw P/Invoke bindings for cuSPARSE.
/// </summary>
public static unsafe class CuSparseNative
{
    private const string LibWindows = "cusparse64_12.dll";
    private const string LibLinux   = "libcusparse.so.12";

    private static readonly string LibName =
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? LibWindows : LibLinux;

    // ── Delegate types ─────────────────────────────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSparseStatus cusparseCreateDelegate(nint* handle);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSparseStatus cusparseDestroyDelegate(nint handle);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSparseStatus cusparseSetStreamDelegate(nint handle, nint stream);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSparseStatus cusparseCreateCsrDelegate(
        nint* spMatDescr, long rows, long cols, long nnz,
        void* csrRowOffsets, void* csrColInd, void* csrValues,
        CuSparseIndexType csrRowOffsetsType, CuSparseIndexType csrColIndType,
        CuSparseIndexBase idxBase, CudaDataType valueType);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSparseStatus cusparseCreateCooDelegate(
        nint* spMatDescr, long rows, long cols, long nnz,
        void* cooRowInd, void* cooColInd, void* cooValues,
        CuSparseIndexType cooIdxType, CuSparseIndexBase idxBase, CudaDataType valueType);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSparseStatus cusparseCreateBsrDelegate(
        nint* spMatDescr,
        long brows, long bcols, long bnnz,
        long rowBlockDim, long colBlockDim,
        void* bsrRowOffsets, void* bsrColInd, void* bsrValues,
        CuSparseIndexType bsrRowOffsetsType, CuSparseIndexType bsrColIndType,
        CuSparseIndexBase idxBase, CudaDataType valueType, int order);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSparseStatus cusparseDestroySpMatDelegate(nint spMatDescr);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSparseStatus cusparseCreateDnVecDelegate(nint* dnVecDescr, long size, void* values, CudaDataType valueType);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSparseStatus cusparseDestroyDnVecDelegate(nint dnVecDescr);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSparseStatus cusparseCreateDnMatDelegate(nint* dnMatDescr, long rows, long cols, long ld, void* values, CudaDataType valueType, int order);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSparseStatus cusparseDestroyDnMatDelegate(nint dnMatDescr);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSparseStatus cusparseSpMV_bufferSizeDelegate(
        nint handle, CuSparseOperation opA,
        void* alpha, nint matA, nint vecX, void* beta, nint vecY,
        CudaDataType computeType, CuSparseSpMVAlg alg, nuint* bufferSize);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSparseStatus cusparseSpMVDelegate(
        nint handle, CuSparseOperation opA,
        void* alpha, nint matA, nint vecX, void* beta, nint vecY,
        CudaDataType computeType, CuSparseSpMVAlg alg, void* externalBuffer);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSparseStatus cusparseSpMM_bufferSizeDelegate(
        nint handle, CuSparseOperation opA, CuSparseOperation opB,
        void* alpha, nint matA, nint matB, void* beta, nint matC,
        CudaDataType computeType, CuSparseSpMMAlg alg, nuint* bufferSize);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSparseStatus cusparseSpMMPreprocessDelegate(
        nint handle, CuSparseOperation opA, CuSparseOperation opB,
        void* alpha, nint matA, nint matB, void* beta, nint matC,
        CudaDataType computeType, CuSparseSpMMAlg alg, void* externalBuffer);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSparseStatus cusparseSpMMDelegate(
        nint handle, CuSparseOperation opA, CuSparseOperation opB,
        void* alpha, nint matA, nint matB, void* beta, nint matC,
        CudaDataType computeType, CuSparseSpMMAlg alg, void* externalBuffer);

    // ── Lazy-loaded delegates ─────────────────────────────────────────────────

    private static T Load<T>(string name) where T : Delegate =>
        Marshal.GetDelegateForFunctionPointer<T>(NativeLibraryLoader.GetExport(LibName, name));

    private static readonly Lazy<cusparseCreateDelegate>              _create          = new(() => Load<cusparseCreateDelegate>("cusparseCreate"));
    private static readonly Lazy<cusparseDestroyDelegate>             _destroy         = new(() => Load<cusparseDestroyDelegate>("cusparseDestroy"));
    private static readonly Lazy<cusparseSetStreamDelegate>           _setStream       = new(() => Load<cusparseSetStreamDelegate>("cusparseSetStream"));
    private static readonly Lazy<cusparseCreateCsrDelegate>           _createCsr       = new(() => Load<cusparseCreateCsrDelegate>("cusparseCreateCsr"));
    private static readonly Lazy<cusparseCreateCooDelegate>           _createCoo       = new(() => Load<cusparseCreateCooDelegate>("cusparseCreateCoo"));    private static readonly Lazy<cusparseCreateBsrDelegate>              _createBsr       = new(() => Load<cusparseCreateBsrDelegate>("cusparseCreateBsr"));    private static readonly Lazy<cusparseDestroySpMatDelegate>        _destroySpMat    = new(() => Load<cusparseDestroySpMatDelegate>("cusparseDestroySpMat"));
    private static readonly Lazy<cusparseCreateDnVecDelegate>         _createDnVec     = new(() => Load<cusparseCreateDnVecDelegate>("cusparseCreateDnVec"));
    private static readonly Lazy<cusparseDestroyDnVecDelegate>        _destroyDnVec    = new(() => Load<cusparseDestroyDnVecDelegate>("cusparseDestroyDnVec"));
    private static readonly Lazy<cusparseCreateDnMatDelegate>         _createDnMat     = new(() => Load<cusparseCreateDnMatDelegate>("cusparseCreateDnMat"));
    private static readonly Lazy<cusparseDestroyDnMatDelegate>        _destroyDnMat    = new(() => Load<cusparseDestroyDnMatDelegate>("cusparseDestroyDnMat"));
    private static readonly Lazy<cusparseSpMV_bufferSizeDelegate>     _spMVBufSize     = new(() => Load<cusparseSpMV_bufferSizeDelegate>("cusparseSpMV_bufferSize"));
    private static readonly Lazy<cusparseSpMVDelegate>                _spMV            = new(() => Load<cusparseSpMVDelegate>("cusparseSpMV"));
    private static readonly Lazy<cusparseSpMM_bufferSizeDelegate>     _spMMBufSize     = new(() => Load<cusparseSpMM_bufferSizeDelegate>("cusparseSpMM_bufferSize"));
    private static readonly Lazy<cusparseSpMMPreprocessDelegate>      _spMMPreprocess  = new(() => Load<cusparseSpMMPreprocessDelegate>("cusparseSpMM_preprocess"));
    private static readonly Lazy<cusparseSpMMDelegate>                _spMM            = new(() => Load<cusparseSpMMDelegate>("cusparseSpMM"));

    // ── Public helpers ─────────────────────────────────────────────────────────

    /// <summary>
    /// Throws <see cref="CuSparseException"/> if <paramref name="status"/> is not
    /// <see cref="CuSparseStatus.Success"/>. Optionally prefixes the message with
    /// <paramref name="context"/>.
    /// </summary>
    public static void Check(CuSparseStatus status, string? context = null)
    {
        if (status == CuSparseStatus.Success)
        {
            return;
        }

        string msg = $"cuSPARSE error {status} ({(int)status})";
        throw new CuSparseException((int)status, context is null ? msg : $"{context}: {msg}");
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    /// <summary>Creates a cuSPARSE library handle. Caller must call <c>cusparseDestroy</c> when done.</summary>
    public static nint Create()
    {
        nint h;
        Check(_create.Value(&h), "cusparseCreate");
        return h;
    }

    /// <summary>Destroys a cuSPARSE library handle.</summary>
    public static void Destroy(nint handle) =>
        Check(_destroy.Value(handle), "cusparseDestroy");

    /// <summary>Associates a CUDA stream with the cuSPARSE handle.</summary>
    public static void SetStream(nint handle, nint stream) =>
        Check(_setStream.Value(handle, stream), "cusparseSetStream");

    /// <summary>
    /// Creates a sparse matrix descriptor in CSR format.
    /// All data pointers must point to device memory.
    /// </summary>
    public static nint CreateCsr(long rows, long cols, long nnz,
        void* csrRowOffsets, void* csrColInd, void* csrValues,
        CuSparseIndexType rowOffType, CuSparseIndexType colIndType,
        CuSparseIndexBase idxBase, CudaDataType valType)
    {
        nint d;
        Check(_createCsr.Value(&d, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues,
            rowOffType, colIndType, idxBase, valType), "cusparseCreateCsr");
        return d;
    }

    /// <summary>
    /// Creates a sparse matrix descriptor in COO format.
    /// All data pointers must point to device memory.
    /// </summary>
    public static nint CreateCoo(long rows, long cols, long nnz,
        void* cooRowInd, void* cooColInd, void* cooValues,
        CuSparseIndexType idxType, CuSparseIndexBase idxBase, CudaDataType valType)
    {
        nint d;
        Check(_createCoo.Value(&d, rows, cols, nnz, cooRowInd, cooColInd, cooValues,
            idxType, idxBase, valType), "cusparseCreateCoo");
        return d;
    }

    /// <summary>
    /// Creates a sparse matrix descriptor in Block Sparse Row (BSR) format.
    /// All data pointers must point to device memory.
    /// </summary>
    /// <param name="brows">Number of block rows.</param>
    /// <param name="bcols">Number of block columns.</param>
    /// <param name="bnnz">Number of non-zero blocks.</param>
    /// <param name="rowBlockDim">Number of rows per block.</param>
    /// <param name="colBlockDim">Number of columns per block.</param>
    /// <param name="bsrRowOffsets">Block row offsets (length brows+1).</param>
    /// <param name="bsrColInd">Block column indices (length bnnz).</param>
    /// <param name="bsrValues">Non-zero block values (length bnnz * rowBlockDim * colBlockDim).</param>
    /// <param name="rowOffType">Index type for row offsets.</param>
    /// <param name="colIndType">Index type for column indices.</param>
    /// <param name="idxBase">Zero-based or one-based indexing.</param>
    /// <param name="valType">Element data type.</param>
    /// <param name="columnMajorBlocks">
    /// <c>true</c> if blocks are stored in column-major order; <c>false</c> for row-major. Default: <c>false</c>.
    /// </param>
    public static nint CreateBsr(
        long brows, long bcols, long bnnz,
        long rowBlockDim, long colBlockDim,
        void* bsrRowOffsets, void* bsrColInd, void* bsrValues,
        CuSparseIndexType rowOffType, CuSparseIndexType colIndType,
        CuSparseIndexBase idxBase, CudaDataType valType,
        bool columnMajorBlocks = false)
    {
        nint d;
        Check(_createBsr.Value(&d, brows, bcols, bnnz, rowBlockDim, colBlockDim,
            bsrRowOffsets, bsrColInd, bsrValues,
            rowOffType, colIndType, idxBase, valType,
            columnMajorBlocks ? 1 : 0), "cusparseCreateBsr");
        return d;
    }

    /// <summary>Destroys a sparse matrix descriptor (CSR or COO).</summary>
    public static void DestroySpMat(nint desc) =>
        Check(_destroySpMat.Value(desc), "cusparseDestroySpMat");

    /// <summary>Creates a dense vector descriptor pointing to device memory.</summary>
    public static nint CreateDnVec(long size, void* values, CudaDataType valType)
    {
        nint d;
        Check(_createDnVec.Value(&d, size, values, valType), "cusparseCreateDnVec");
        return d;
    }

    /// <summary>Destroys a dense vector descriptor.</summary>
    public static void DestroyDnVec(nint desc) =>
        Check(_destroyDnVec.Value(desc), "cusparseDestroyDnVec");

    /// <summary>
    /// Creates a dense matrix descriptor pointing to device memory.
    /// </summary>
    /// <param name="rows">Number of rows in the matrix.</param>
    /// <param name="cols">Number of columns in the matrix.</param>
    /// <param name="ld">Leading dimension of the matrix.</param>
    /// <param name="values">Pointer to the matrix values in device memory.</param>
    /// <param name="valType">Data type of the matrix elements.</param>
    /// <param name="order">Storage order: 0 = column-major, 1 = row-major.</param>
    public static nint CreateDnMat(long rows, long cols, long ld, void* values, CudaDataType valType, int order)
    {
        nint d;
        Check(_createDnMat.Value(&d, rows, cols, ld, values, valType, order), "cusparseCreateDnMat");
        return d;
    }

    /// <summary>Destroys a dense matrix descriptor.</summary>
    public static void DestroyDnMat(nint desc) =>
        Check(_destroyDnMat.Value(desc), "cusparseDestroyDnMat");

    /// <summary>
    /// Returns the size of the external workspace buffer required for
    /// <see cref="SpMV"/> with the given configuration.
    /// </summary>
    public static nuint SpMVBufferSize(
        nint handle, CuSparseOperation opA,
        void* alpha, nint matA, nint vecX, void* beta, nint vecY,
        CudaDataType computeType, CuSparseSpMVAlg alg)
    {
        nuint sz;
        Check(_spMVBufSize.Value(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, &sz), "cusparseSpMV_bufferSize");
        return sz;
    }

    /// <summary>
    /// Sparse matrix-vector product: y = alpha * op(A) * x + beta * y.
    /// All pointers must reside in device memory.
    /// </summary>
    public static void SpMV(
        nint handle, CuSparseOperation opA,
        void* alpha, nint matA, nint vecX, void* beta, nint vecY,
        CudaDataType computeType, CuSparseSpMVAlg alg, void* externalBuffer) =>
        Check(_spMV.Value(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer), "cusparseSpMV");

    /// <summary>
    /// Returns the size of the external workspace buffer required for <see cref="SpMM"/>.
    /// </summary>
    public static nuint SpMMBufferSize(
        nint handle, CuSparseOperation opA, CuSparseOperation opB,
        void* alpha, nint matA, nint matB, void* beta, nint matC,
        CudaDataType computeType, CuSparseSpMMAlg alg)
    {
        nuint sz;
        Check(_spMMBufSize.Value(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, &sz), "cusparseSpMM_bufferSize");
        return sz;
    }

    /// <summary>
    /// Optional preprocessing step for <see cref="SpMM"/> that may improve performance
    /// when the same sparse matrix is multiplied against multiple dense matrices.
    /// </summary>
    public static void SpMMPreprocess(
        nint handle, CuSparseOperation opA, CuSparseOperation opB,
        void* alpha, nint matA, nint matB, void* beta, nint matC,
        CudaDataType computeType, CuSparseSpMMAlg alg, void* externalBuffer) =>
        Check(_spMMPreprocess.Value(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer), "cusparseSpMM_preprocess");

    /// <summary>
    /// Sparse matrix-matrix product: C = alpha * op(A) * op(B) + beta * C.
    /// All pointers must reside in device memory.
    /// </summary>
    public static void SpMM(
        nint handle, CuSparseOperation opA, CuSparseOperation opB,
        void* alpha, nint matA, nint matB, void* beta, nint matC,
        CudaDataType computeType, CuSparseSpMMAlg alg, void* externalBuffer) =>
        Check(_spMM.Value(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer), "cusparseSpMM");
}
