// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using System.Runtime.InteropServices;

namespace NVMathNet.Interop;

/// <summary>cuBLAS status codes.</summary>
public enum CuBlasStatus : int
{
    /// <summary>Operation completed successfully.</summary>
    Success         = 0,
    /// <summary>cuBLAS library not initialised.</summary>
    NotInitialized  = 1,
    /// <summary>GPU memory allocation failed.</summary>
    AllocFailed     = 3,
    /// <summary>A parameter has an invalid value.</summary>
    InvalidValue    = 7,
    /// <summary>GPU architecture does not support the operation.</summary>
    ArchMismatch    = 8,
    /// <summary>GPU memory mapping failed.</summary>
    MappingError    = 11,
    /// <summary>Execution on the GPU failed.</summary>
    ExecutionFailed = 13,
    /// <summary>Internal driver error.</summary>
    InternalError   = 14,
    /// <summary>Operation is not supported with the given parameters.</summary>
    NotSupported    = 15,
    /// <summary>cuBLAS license check failed.</summary>
    LicenseError    = 16,
}

/// <summary>cuBLAS / cuBLASLt compute types.</summary>
public enum CuBlasComputeType : int
{
    /// <summary>16-bit floating-point accumulation.</summary>
    Compute16F           = 64,
    /// <summary>16-bit floating-point, strict IEEE compliance.</summary>
    Compute16FPedantic   = 65,
    /// <summary>32-bit floating-point accumulation (default for SGEMM).</summary>
    Compute32F           = 68,
    /// <summary>32-bit floating-point, strict IEEE compliance.</summary>
    Compute32FPedantic   = 69,
    /// <summary>32-bit float compute accelerated by 16-bit tensor cores.</summary>
    Compute32FFast16F    = 74,
    /// <summary>32-bit float compute accelerated by bfloat16 tensor cores.</summary>
    Compute32FFast16BF   = 75,
    /// <summary>32-bit float compute using TF32 tensor cores.</summary>
    Compute32FFastTF32   = 77,
    /// <summary>64-bit floating-point accumulation (DGEMM).</summary>
    Compute64F           = 70,
    /// <summary>64-bit floating-point, strict IEEE compliance.</summary>
    Compute64FPedantic   = 71,
    /// <summary>32-bit integer accumulation (IGEMM).</summary>
    Compute32I           = 72,
    /// <summary>32-bit integer, strict mode.</summary>
    Compute32IPedantic   = 73,
}

/// <summary>cuBLAS operation types (transpose).</summary>
/// <summary>Matrix transpose operations used by cuBLAS/cuBLASLt.</summary>
public enum CuBlasOperation : int
{
    /// <summary>No operation — matrix is used as-is.</summary>
    None          = 0,
    /// <summary>Transpose the matrix.</summary>
    Transpose     = 1,
    /// <summary>Conjugate-transpose (Hermitian) the matrix.</summary>
    ConjTranspose = 2,
}

/// <summary>CUDA data types used across cuBLASLt, cuSPARSE and cuTENSOR.</summary>
public enum CudaDataType : int
{
    /// <summary>16-bit real floating-point (half).</summary>
    R_16F   = 2,
    /// <summary>16-bit complex floating-point.</summary>
    C_16F   = 6,
    /// <summary>16-bit real bfloat16.</summary>
    R_16BF  = 14,
    /// <summary>16-bit complex bfloat16.</summary>
    C_16BF  = 15,
    /// <summary>32-bit real floating-point (single).</summary>
    R_32F   = 0,
    /// <summary>32-bit complex floating-point.</summary>
    C_32F   = 4,
    /// <summary>64-bit real floating-point (double).</summary>
    R_64F   = 1,
    /// <summary>64-bit complex floating-point.</summary>
    C_64F   = 5,
    /// <summary>8-bit signed integer.</summary>
    R_8I    = 3,
    /// <summary>8-bit complex signed integer.</summary>
    C_8I    = 7,
    /// <summary>8-bit unsigned integer.</summary>
    R_8U    = 8,
    /// <summary>8-bit complex unsigned integer.</summary>
    C_8U    = 9,
    /// <summary>32-bit signed integer.</summary>
    R_32I   = 10,
    /// <summary>32-bit complex signed integer.</summary>
    C_32I   = 11,
    /// <summary>8-bit floating-point E4M3 (FP8).</summary>
    R_8F_E4M3 = 28,
    /// <summary>8-bit floating-point E5M2 (FP8).</summary>
    R_8F_E5M2 = 29,
}

/// <summary>cuBLASLt epilog operations applied after the matrix multiplication.</summary>
public enum CuBlasLtEpilogue : int
{
    /// <summary>No epilog \u2014 output D = alpha*A*B + beta*C unchanged.</summary>
    Default         = 1,
    /// <summary>Apply ReLU activation: D = max(0, alpha*A*B + beta*C).</summary>
    Relu            = 2,
    /// <summary>Add a per-row bias vector to the output.</summary>
    Bias            = 4,
    /// <summary>Add per-row bias, then apply ReLU.</summary>
    ReluBias        = 6,
    /// <summary>Apply GELU activation.</summary>
    Gelu            = 32,
    /// <summary>Apply GELU and also save the pre-GELU values for backward pass.</summary>
    GeluAux         = 96,
    /// <summary>Add per-row bias, then apply GELU.</summary>
    GeluBias        = 100,
    /// <summary>Backward ReLU \u2014 compute gradient through ReLU.</summary>
    DreLu           = 128,
    /// <summary>Backward ReLU with bias gradient.</summary>
    DreLuBias       = 132,
    /// <summary>Backward GELU \u2014 compute gradient through GELU.</summary>
    DGelu           = 160,
    /// <summary>Backward GELU with bias gradient.</summary>
    DGeluBias       = 164,
    /// <summary>Backward bias-gradient accumulation.</summary>
    BgraD           = 192,
}

/// <summary>
/// Raw P/Invoke bindings for cuBLAS and cuBLASLt.
/// </summary>
public static unsafe class CuBlasNative
{
    private const string LibWindows    = "cublas64_12.dll";
    private const string LibLinux      = "libcublas.so.12";
    private const string LibLtWindows  = "cublasLt64_12.dll";
    private const string LibLtLinux    = "libcublasLt.so.12";

    private static readonly string LibName   = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? LibWindows   : LibLinux;
    private static readonly string LibLtName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? LibLtWindows : LibLtLinux;

    // ── Delegate types (classic cuBLAS) ────────────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasCreateDelegate(nint* handle);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasDestroyDelegate(nint handle);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasSetStreamDelegate(nint handle, nint stream);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasSgemmDelegate(
        nint handle, CuBlasOperation transa, CuBlasOperation transb,
        int m, int n, int k,
        float* alpha, float* a, int lda,
        float* b, int ldb,
        float* beta, float* c, int ldc);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasDgemmDelegate(
        nint handle, CuBlasOperation transa, CuBlasOperation transb,
        int m, int n, int k,
        double* alpha, double* a, int lda,
        double* b, int ldb,
        double* beta, double* c, int ldc);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasHgemmDelegate(
        nint handle, CuBlasOperation transa, CuBlasOperation transb,
        int m, int n, int k,
        Half* alpha, Half* a, int lda,
        Half* b, int ldb,
        Half* beta, Half* c, int ldc);

    // ── Delegate types (cuBLASLt) ─────────────────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasLtCreateDelegate(nint* lightHandle);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasLtDestroyDelegate(nint lightHandle);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasLtMatmulDescCreateDelegate(nint* matmulDesc, CuBlasComputeType computeType, CudaDataType scaleType);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasLtMatmulDescDestroyDelegate(nint matmulDesc);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasLtMatmulDescSetAttributeDelegate(nint matmulDesc, int attr, void* buf, nuint sizeInBytes);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasLtMatrixLayoutCreateDelegate(nint* matLayout, CudaDataType type, ulong rows, ulong cols, long ld);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasLtMatrixLayoutDestroyDelegate(nint matLayout);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasLtMatrixLayoutSetAttributeDelegate(nint matLayout, int attr, void* buf, nuint sizeInBytes);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasLtMatmulPreferenceCreateDelegate(nint* pref);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasLtMatmulPreferenceDestroyDelegate(nint pref);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasLtMatmulPreferenceSetAttributeDelegate(nint pref, int attr, void* buf, nuint sizeInBytes);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasLtMatmulAlgoGetHeuristicDelegate(
        nint lightHandle, nint operationDesc,
        nint adesc, nint bdesc, nint cdesc, nint ddesc,
        nint preference, int requestedAlgoCount,
        void* heuristicResultsArray, int* returnAlgoCount);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuBlasStatus cublasLtMatmulDelegate(
        nint lightHandle, nint computeDesc,
        void* alpha,
        void* a, nint adesc,
        void* b, nint bdesc,
        void* beta,
        void* c, nint cdesc,
        void* d, nint ddesc,
        void* algo, void* workspace, nuint workspaceSizeInBytes,
        nint stream);

    // ── Lazy-loaded delegates ─────────────────────────────────────────────────

    private static T Load<T>(string lib, string name) where T : Delegate =>
        Marshal.GetDelegateForFunctionPointer<T>(NativeLibraryLoader.GetExport(lib, name));

    private static readonly Lazy<cublasCreateDelegate>                      _create          = new(() => Load<cublasCreateDelegate>(LibName, "cublasCreate_v2"));
    private static readonly Lazy<cublasDestroyDelegate>                     _destroy         = new(() => Load<cublasDestroyDelegate>(LibName, "cublasDestroy_v2"));
    private static readonly Lazy<cublasSetStreamDelegate>                   _setStream       = new(() => Load<cublasSetStreamDelegate>(LibName, "cublasSetStream_v2"));
    private static readonly Lazy<cublasSgemmDelegate>                       _sgemm           = new(() => Load<cublasSgemmDelegate>(LibName, "cublasSgemm_v2"));
    private static readonly Lazy<cublasDgemmDelegate>                       _dgemm           = new(() => Load<cublasDgemmDelegate>(LibName, "cublasDgemm_v2"));
    private static readonly Lazy<cublasHgemmDelegate>                       _hgemm           = new(() => Load<cublasHgemmDelegate>(LibName, "cublasHgemm"));
    private static readonly Lazy<cublasLtCreateDelegate>                    _ltCreate        = new(() => Load<cublasLtCreateDelegate>(LibLtName, "cublasLtCreate"));
    private static readonly Lazy<cublasLtDestroyDelegate>                   _ltDestroy       = new(() => Load<cublasLtDestroyDelegate>(LibLtName, "cublasLtDestroy"));
    private static readonly Lazy<cublasLtMatmulDescCreateDelegate>          _descCreate      = new(() => Load<cublasLtMatmulDescCreateDelegate>(LibLtName, "cublasLtMatmulDescCreate"));
    private static readonly Lazy<cublasLtMatmulDescDestroyDelegate>         _descDestroy     = new(() => Load<cublasLtMatmulDescDestroyDelegate>(LibLtName, "cublasLtMatmulDescDestroy"));
    private static readonly Lazy<cublasLtMatmulDescSetAttributeDelegate>    _descSetAttr     = new(() => Load<cublasLtMatmulDescSetAttributeDelegate>(LibLtName, "cublasLtMatmulDescSetAttribute"));
    private static readonly Lazy<cublasLtMatrixLayoutCreateDelegate>        _layoutCreate    = new(() => Load<cublasLtMatrixLayoutCreateDelegate>(LibLtName, "cublasLtMatrixLayoutCreate"));
    private static readonly Lazy<cublasLtMatrixLayoutDestroyDelegate>       _layoutDestroy   = new(() => Load<cublasLtMatrixLayoutDestroyDelegate>(LibLtName, "cublasLtMatrixLayoutDestroy"));
    private static readonly Lazy<cublasLtMatrixLayoutSetAttributeDelegate>  _layoutSetAttr   = new(() => Load<cublasLtMatrixLayoutSetAttributeDelegate>(LibLtName, "cublasLtMatrixLayoutSetAttribute"));
    private static readonly Lazy<cublasLtMatmulPreferenceCreateDelegate>    _prefCreate      = new(() => Load<cublasLtMatmulPreferenceCreateDelegate>(LibLtName, "cublasLtMatmulPreferenceCreate"));
    private static readonly Lazy<cublasLtMatmulPreferenceDestroyDelegate>   _prefDestroy     = new(() => Load<cublasLtMatmulPreferenceDestroyDelegate>(LibLtName, "cublasLtMatmulPreferenceDestroy"));
    private static readonly Lazy<cublasLtMatmulPreferenceSetAttributeDelegate> _prefSetAttr  = new(() => Load<cublasLtMatmulPreferenceSetAttributeDelegate>(LibLtName, "cublasLtMatmulPreferenceSetAttribute"));
    private static readonly Lazy<cublasLtMatmulAlgoGetHeuristicDelegate>    _algoGetHeur     = new(() => Load<cublasLtMatmulAlgoGetHeuristicDelegate>(LibLtName, "cublasLtMatmulAlgoGetHeuristic"));
    private static readonly Lazy<cublasLtMatmulDelegate>                    _matmul          = new(() => Load<cublasLtMatmulDelegate>(LibLtName, "cublasLtMatmul"));

    // ── Public helpers ─────────────────────────────────────────────────────────

    /// <summary>
    /// Throws <see cref="CuBlasException"/> if <paramref name="status"/> is not
    /// <see cref="CuBlasStatus.Success"/>. Optionally prefixes the message with
    /// <paramref name="context"/>.
    /// </summary>
    public static void Check(CuBlasStatus status, string? context = null)
    {
        if (status == CuBlasStatus.Success) return;
        string msg = $"cuBLAS error {status} ({(int)status})";
        throw new CuBlasException((int)status, context is null ? msg : $"{context}: {msg}");
    }

    // ── Public API (cuBLASLt) ─────────────────────────────────────────────────

    /// <summary>Creates a cuBLASLt handle. Caller must call <see cref="LtDestroy"/> when done.</summary>
    public static nint LtCreate()
    {
        nint h;
        Check(_ltCreate.Value(&h), "cublasLtCreate");
        return h;
    }

    /// <summary>Destroys a cuBLASLt handle created with <see cref="LtCreate"/>.</summary>
    public static void LtDestroy(nint handle) =>
        Check(_ltDestroy.Value(handle), "cublasLtDestroy");

    /// <summary>
    /// Creates a matmul operation descriptor.
    /// </summary>
    /// <param name="computeType">Compute precision.</param>
    /// <param name="scaleType">Data type for alpha/beta scalars.</param>
    public static nint MatmulDescCreate(CuBlasComputeType computeType, CudaDataType scaleType)
    {
        nint desc;
        Check(_descCreate.Value(&desc, computeType, scaleType), "cublasLtMatmulDescCreate");
        return desc;
    }

    /// <summary>Destroys a matmul descriptor.</summary>
    public static void MatmulDescDestroy(nint desc) =>
        Check(_descDestroy.Value(desc), "cublasLtMatmulDescDestroy");

    /// <summary>Sets an attribute on a matmul descriptor (e.g. transpose, epilog).</summary>
    public static void MatmulDescSetAttribute(nint desc, int attr, void* buf, nuint size) =>
        Check(_descSetAttr.Value(desc, attr, buf, size), "cublasLtMatmulDescSetAttribute");

    /// <summary>Creates a matrix layout descriptor.</summary>
    public static nint MatrixLayoutCreate(CudaDataType type, ulong rows, ulong cols, long ld)
    {
        nint layout;
        Check(_layoutCreate.Value(&layout, type, rows, cols, ld), "cublasLtMatrixLayoutCreate");
        return layout;
    }

    /// <summary>Destroys a matrix layout descriptor.</summary>
    public static void MatrixLayoutDestroy(nint layout) =>
        Check(_layoutDestroy.Value(layout), "cublasLtMatrixLayoutDestroy");

    /// <summary>Sets an attribute on a matrix layout descriptor.</summary>
    public static void MatrixLayoutSetAttribute(nint layout, int attr, void* buf, nuint size) =>
        Check(_layoutSetAttr.Value(layout, attr, buf, size), "cublasLtMatrixLayoutSetAttribute");

    /// <summary>Creates a matmul algorithm preference descriptor.</summary>
    public static nint MatmulPreferenceCreate()
    {
        nint pref;
        Check(_prefCreate.Value(&pref), "cublasLtMatmulPreferenceCreate");
        return pref;
    }

    /// <summary>Destroys a matmul preference descriptor.</summary>
    public static void MatmulPreferenceDestroy(nint pref) =>
        Check(_prefDestroy.Value(pref), "cublasLtMatmulPreferenceDestroy");

    /// <summary>Sets an attribute on a matmul preference descriptor (e.g. max workspace bytes).</summary>
    public static void MatmulPreferenceSetAttribute(nint pref, int attr, void* buf, nuint size) =>
        Check(_prefSetAttr.Value(pref, attr, buf, size), "cublasLtMatmulPreferenceSetAttribute");

    /// <summary>
    /// Size in bytes of the opaque <c>cublasLtMatmulHeuristicResult_t</c> struct.
    /// Callers must pass a buffer of at least this size to <see cref="MatmulAlgoGetHeuristic"/>.
    /// </summary>
    public const int HeuristicResultSize = 8 * 8 + 4 * 8; // algo(8*8) + workspaceSize(8) + state(4) + wavesCount(4) + reserved[4]

    /// <summary>
    /// Queries cuBLASLt for algorithm suggestions and writes them into
    /// <paramref name="resultsBuffer"/>.
    /// </summary>
    /// <returns>Number of algorithms written (0 if none found).</returns>
    public static int MatmulAlgoGetHeuristic(
        nint ltHandle, nint opDesc,
        nint aDesc, nint bDesc, nint cDesc, nint dDesc,
        nint preference, int maxAlgos, byte[] resultsBuffer)
    {
        int count;
        fixed (byte* pBuf = resultsBuffer)
            Check(_algoGetHeur.Value(ltHandle, opDesc, aDesc, bDesc, cDesc, dDesc,
                preference, maxAlgos, pBuf, &count), "cublasLtMatmulAlgoGetHeuristic");
        return count;
    }

    /// <summary>
    /// Executes D = alpha * op(A) * op(B) + beta * C on the GPU.
    /// All pointers must point to device memory except <paramref name="algo"/>
    /// which is the opaque algorithm buffer returned by <see cref="MatmulAlgoGetHeuristic"/>.
    /// </summary>
    public static void Matmul(
        nint ltHandle, nint opDesc,
        void* alpha, void* a, nint aDesc,
        void* b, nint bDesc,
        void* beta,  void* c, nint cDesc,
        void* d, nint dDesc,
        void* algo, void* workspace, nuint workspaceSize, nint stream) =>
        Check(_matmul.Value(ltHandle, opDesc, alpha, a, aDesc, b, bDesc, beta, c, cDesc,
            d, dDesc, algo, workspace, workspaceSize, stream), "cublasLtMatmul");

    // ── Public API (classic cuBLAS) ───────────────────────────────────────────

    /// <summary>Creates a classic cuBLAS handle. Caller must call <see cref="Destroy"/> when done.</summary>
    public static nint Create()
    {
        nint h;
        Check(_create.Value(&h), "cublasCreate_v2");
        return h;
    }

    /// <summary>Destroys a classic cuBLAS handle.</summary>
    public static void Destroy(nint handle) =>
        Check(_destroy.Value(handle), "cublasDestroy_v2");

    /// <summary>Associates a CUDA stream with the classic cuBLAS handle.</summary>
    public static void SetStream(nint handle, nint stream) =>
        Check(_setStream.Value(handle, stream), "cublasSetStream_v2");

    /// <summary>
    /// Single-precision GEMM: C = alpha * op(A) * op(B) + beta * C.
    /// All matrices must reside in device memory (column-major layout).
    /// </summary>
    /// <param name="handle">Classic cuBLAS handle.</param>
    /// <param name="transa">Transpose operation for A.</param>
    /// <param name="transb">Transpose operation for B.</param>
    /// <param name="m">Rows of op(A) and C.</param>
    /// <param name="n">Columns of op(B) and C.</param>
    /// <param name="k">Inner dimension.</param>
    /// <param name="alpha">Scalar multiplier for A*B.</param>
    /// <param name="a">Device pointer to A.</param>
    /// <param name="lda">Leading dimension of A.</param>
    /// <param name="b">Device pointer to B.</param>
    /// <param name="ldb">Leading dimension of B.</param>
    /// <param name="beta">Scalar multiplier for existing C.</param>
    /// <param name="c">Device pointer to C (in/out).</param>
    /// <param name="ldc">Leading dimension of C.</param>
    public static void Sgemm(
        nint handle, CuBlasOperation transa, CuBlasOperation transb,
        int m, int n, int k,
        float* alpha, float* a, int lda,
        float* b, int ldb,
        float* beta, float* c, int ldc) =>
        Check(_sgemm.Value(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc), "cublasSgemm_v2");

    /// <summary>
    /// Double-precision GEMM: C = alpha * op(A) * op(B) + beta * C.
    /// All matrices must reside in device memory (column-major layout).
    /// </summary>
    /// <inheritdoc cref="Sgemm"/>
    public static void Dgemm(
        nint handle, CuBlasOperation transa, CuBlasOperation transb,
        int m, int n, int k,
        double* alpha, double* a, int lda,
        double* b, int ldb,
        double* beta, double* c, int ldc) =>
        Check(_dgemm.Value(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc), "cublasDgemm_v2");

    /// <summary>
    /// Half-precision GEMM: C = alpha * op(A) * op(B) + beta * C.
    /// All matrices must reside in device memory (column-major layout).
    /// </summary>
    /// <inheritdoc cref="Sgemm"/>
    public static void Hgemm(
        nint handle, CuBlasOperation transa, CuBlasOperation transb,
        int m, int n, int k,
        Half* alpha, Half* a, int lda,
        Half* b, int ldb,
        Half* beta, Half* c, int ldc) =>
        Check(_hgemm.Value(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc), "cublasHgemm");
}

