// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using System.Runtime.InteropServices;

namespace NVMathNet.Interop;

/// <summary>cuTENSOR status codes.</summary>
public enum CuTensorStatus : int
{
    /// <summary>Operation completed successfully.</summary>
    Success                  = 0,
    /// <summary>cuTENSOR library not initialised.</summary>
    NotInitialized           = 1,
    /// <summary>GPU memory allocation failed.</summary>
    AllocFailed              = 3,
    /// <summary>A parameter has an invalid value.</summary>
    InvalidValue             = 7,
    /// <summary>GPU architecture does not support the operation.</summary>
    ArchMismatch             = 8,
    /// <summary>GPU memory mapping failed.</summary>
    MappingError             = 11,
    /// <summary>Execution on the GPU failed.</summary>
    ExecutionFailed          = 13,
    /// <summary>Internal driver error.</summary>
    InternalError            = 14,
    /// <summary>Operation is not supported with the given parameters.</summary>
    NotSupported             = 15,
    /// <summary>cuTENSOR license check failed.</summary>
    LicenseError             = 16,
    /// <summary>An underlying CUDA error occurred.</summary>
    CudaError                = 17,
    /// <summary>Workspace buffer is too small.</summary>
    InsufficientWorkspace    = 18,
    /// <summary>CUDA driver version is below the minimum required.</summary>
    InsufficientDriver       = 19,
    /// <summary>File I/O error (e.g. kernel cache).</summary>
    IoError                  = 20,
}

/// <summary>cuTENSOR element-wise unary operator applied during tensor operations.</summary>
public enum CuTensorOperator : int
{
    /// <summary>No-op — pass the element through unchanged.</summary>
    Identity     = 1,
    /// <summary>Element-wise square root.</summary>
    Sqrt         = 2,
    /// <summary>Element-wise ReLU activation.</summary>
    Relu         = 8,
    /// <summary>Complex conjugate.</summary>
    Conj         = 9,
    /// <summary>Reciprocal square root.</summary>
    RcpSqrt      = 3,
    /// <summary>Element-wise sine.</summary>
    Sin          = 4,
    /// <summary>Element-wise reciprocal.</summary>
    Rcp          = 5,
    /// <summary>Sigmoid activation.</summary>
    Sigmoid      = 6,
    /// <summary>Hyperbolic tangent activation.</summary>
    Tanh         = 7,
    /// <summary>Element-wise natural exponent.</summary>
    Exp          = 22,
    /// <summary>Element-wise natural logarithm.</summary>
    Log          = 23,
    /// <summary>Element-wise absolute value.</summary>
    Abs          = 24,
    /// <summary>Element-wise negation.</summary>
    Neg          = 25,
    /// <summary>Binary addition (for fused operations).</summary>
    Add          = 10,
    /// <summary>Binary multiplication (for fused operations).</summary>
    Mul          = 16,
    /// <summary>Binary maximum.</summary>
    Max          = 26,
    /// <summary>Binary minimum.</summary>
    Min          = 27,
}

/// <summary>cuTENSOR contraction algorithms.</summary>
public enum CuTensorAlgo : int
{
    /// <summary>Patient default — tries more algorithms than <see cref="Default"/>.</summary>
    DefaultPatient  = -6,
    /// <summary>Benchmark all algorithms and pick the fastest.</summary>
    Optimal         = -5,
    /// <summary>Automatic algorithm selection (recommended).</summary>
    Default         = -1,
    /// <summary>Transpose-Transpose-GEMM-Transpose algorithm.</summary>
    Ttgt            = 1,
    /// <summary>Transpose-GEMM-Transpose-Transpose algorithm.</summary>
    Tgett           = 3,
    /// <summary>Pure GEMM-based algorithm.</summary>
    Gett            = 4,
    /// <summary>Alternate Transpose-GEMM-Transpose-Transpose variant.</summary>
    Tgett_2         = 5,
}

/// <summary>cuTENSOR workspace size preference hint.</summary>
public enum CuTensorWorksizePreference : int
{
    /// <summary>Use only the minimum required workspace.</summary>
    Min         = 0,
    /// <summary>Use the recommended workspace size for best performance (default).</summary>
    Recommended = 1,
    /// <summary>Allocate the maximum workspace for potentially highest performance.</summary>
    Max         = 2,
}

/// <summary>
/// Raw P/Invoke bindings for cuTENSOR v2.
/// </summary>
public static unsafe class CuTensorNative
{
    private const string LibWindows = "cutensor.dll";
    private const string LibLinux   = "libcutensor.so.2";

    private static readonly string LibName =
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? LibWindows : LibLinux;

    // ── Delegate types ─────────────────────────────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorCreateDelegate(nint* handle);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorDestroyDelegate(nint handle);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorCreateTensorDescriptorDelegate(
        nint handle, nint* desc, uint numModes,
        long* extent, long* stride,
        CudaDataType dataType, CuTensorOperator unaryOp);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorDestroyTensorDescriptorDelegate(nint desc);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorCreateContractionDescriptorDelegate(
        nint handle, nint* desc,
        nint descA, int* modeA, uint alignA,
        nint descB, int* modeB, uint alignB,
        nint descC, int* modeC, uint alignC,
        nint descD, int* modeD, uint alignD,
        CuBlasComputeType typeCompute);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorDestroyOperationDescriptorDelegate(nint desc);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorCreateContractionFindDelegate(
        nint handle, nint* find, CuTensorAlgo algo);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorDestroyContractionFindDelegate(nint find);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorContractionGetWorkspaceSizeDelegate(
        nint handle, nint desc, nint find,
        CuTensorWorksizePreference pref, nuint* workspaceSize);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorCreateContractionPlanDelegate(
        nint handle, nint* plan, nint desc, nint find, nuint workspaceSize);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorDestroyContractionPlanDelegate(nint plan);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorContractDelegate(
        nint handle, nint plan,
        void* alpha, void* A, void* B,
        void* beta, void* C, void* D,
        void* workspace, nuint workspaceSize, nint stream);

    // ── Lazy-loaded delegates ─────────────────────────────────────────────────

    private static T Load<T>(string name) where T : Delegate =>
        Marshal.GetDelegateForFunctionPointer<T>(NativeLibraryLoader.GetExport(LibName, name));

    private static readonly Lazy<cutensorCreateDelegate>                        _create         = new(() => Load<cutensorCreateDelegate>("cutensorCreate"));
    private static readonly Lazy<cutensorDestroyDelegate>                       _destroy        = new(() => Load<cutensorDestroyDelegate>("cutensorDestroy"));
    private static readonly Lazy<cutensorCreateTensorDescriptorDelegate>        _createTensorDesc = new(() => Load<cutensorCreateTensorDescriptorDelegate>("cutensorCreateTensorDescriptor"));
    private static readonly Lazy<cutensorDestroyTensorDescriptorDelegate>       _destroyTensorDesc = new(() => Load<cutensorDestroyTensorDescriptorDelegate>("cutensorDestroyTensorDescriptor"));
    private static readonly Lazy<cutensorCreateContractionDescriptorDelegate>   _createContrDesc = new(() => Load<cutensorCreateContractionDescriptorDelegate>("cutensorCreateContractionDescriptor"));
    private static readonly Lazy<cutensorDestroyOperationDescriptorDelegate>    _destroyOpDesc  = new(() => Load<cutensorDestroyOperationDescriptorDelegate>("cutensorDestroyOperationDescriptor"));
    private static readonly Lazy<cutensorCreateContractionFindDelegate>         _createFind     = new(() => Load<cutensorCreateContractionFindDelegate>("cutensorCreateContractionFind"));
    private static readonly Lazy<cutensorDestroyContractionFindDelegate>        _destroyFind    = new(() => Load<cutensorDestroyContractionFindDelegate>("cutensorDestroyContractionFind"));
    private static readonly Lazy<cutensorContractionGetWorkspaceSizeDelegate>   _getWorkspace   = new(() => Load<cutensorContractionGetWorkspaceSizeDelegate>("cutensorContractionGetWorkspaceSize"));
    private static readonly Lazy<cutensorCreateContractionPlanDelegate>         _createPlan     = new(() => Load<cutensorCreateContractionPlanDelegate>("cutensorCreateContractionPlan"));
    private static readonly Lazy<cutensorDestroyContractionPlanDelegate>        _destroyPlan    = new(() => Load<cutensorDestroyContractionPlanDelegate>("cutensorDestroyContractionPlan"));
    private static readonly Lazy<cutensorContractDelegate>                      _contract       = new(() => Load<cutensorContractDelegate>("cutensorContract"));

    // ── Public helpers ─────────────────────────────────────────────────────────

    /// <summary>
    /// Throws <see cref="CuTensorException"/> if <paramref name="status"/> is not
    /// <see cref="CuTensorStatus.Success"/>. Optionally prefixes the message with
    /// <paramref name="context"/>.
    /// </summary>
    public static void Check(CuTensorStatus status, string? context = null)
    {
        if (status == CuTensorStatus.Success) return;
        string msg = $"cuTENSOR error {status} ({(int)status})";
        throw new CuTensorException((int)status, context is null ? msg : $"{context}: {msg}");
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    /// <summary>Creates a cuTENSOR library handle. Caller must call <see cref="Destroy"/> when done.</summary>
    public static nint Create()
    {
        nint h;
        Check(_create.Value(&h), "cutensorCreate");
        return h;
    }

    /// <summary>Destroys a cuTENSOR library handle.</summary>
    public static void Destroy(nint handle) =>
        Check(_destroy.Value(handle), "cutensorDestroy");

    /// <summary>
    /// Creates a tensor descriptor describing an N-dimensional tensor.
    /// </summary>
    /// <param name="handle">cuTENSOR handle.</param>
    /// <param name="numModes">Number of dimensions.</param>
    /// <param name="extent">Size of each dimension.</param>
    /// <param name="stride">Stride for each dimension in elements; <c>null</c> = packed row-major.</param>
    /// <param name="dataType">Element data type.</param>
    /// <param name="unaryOp">Optional pointwise unary operator applied to each element during contraction.</param>
    public static nint CreateTensorDescriptor(
        nint handle, uint numModes, long[] extent, long[]? stride, CudaDataType dataType,
        CuTensorOperator unaryOp = CuTensorOperator.Identity)
    {
        nint desc;
        fixed (long* pe = extent, ps = stride)
            Check(_createTensorDesc.Value(handle, &desc, numModes, pe, ps, dataType, unaryOp), "cutensorCreateTensorDescriptor");
        return desc;
    }

    /// <summary>Destroys a tensor descriptor.</summary>
    public static void DestroyTensorDescriptor(nint desc) =>
        Check(_destroyTensorDesc.Value(desc), "cutensorDestroyTensorDescriptor");

    /// <summary>
    /// Creates a contraction operation descriptor describing
    /// D = alpha * op(A) * op(B) + beta * op(C).
    /// </summary>
    public static nint CreateContractionDescriptor(
        nint handle,
        nint descA, int[] modeA, uint alignA,
        nint descB, int[] modeB, uint alignB,
        nint descC, int[] modeC, uint alignC,
        nint descD, int[] modeD, uint alignD,
        CuBlasComputeType typeCompute)
    {
        nint desc;
        fixed (int* pmA = modeA, pmB = modeB, pmC = modeC, pmD = modeD)
            Check(_createContrDesc.Value(handle, &desc,
                descA, pmA, alignA, descB, pmB, alignB,
                descC, pmC, alignC, descD, pmD, alignD, typeCompute), "cutensorCreateContractionDescriptor");
        return desc;
    }

    /// <summary>Destroys a contraction operation descriptor.</summary>
    public static void DestroyOperationDescriptor(nint desc) =>
        Check(_destroyOpDesc.Value(desc), "cutensorDestroyOperationDescriptor");

    /// <summary>
    /// Creates a contraction algorithm finder descriptor.
    /// </summary>
    /// <param name="handle">cuTENSOR handle.</param>
    /// <param name="algo">Algorithm hint; use <see cref="CuTensorAlgo.Default"/> for autoselection.</param>
    public static nint CreateContractionFind(nint handle, CuTensorAlgo algo = CuTensorAlgo.Default)
    {
        nint find;
        Check(_createFind.Value(handle, &find, algo), "cutensorCreateContractionFind");
        return find;
    }

    /// <summary>Destroys a contraction finder descriptor.</summary>
    public static void DestroyContractionFind(nint find) =>
        Check(_destroyFind.Value(find), "cutensorDestroyContractionFind");

    /// <summary>
    /// Returns the required workspace size in bytes for a contraction
    /// with the given preference.
    /// </summary>
    public static nuint GetWorkspaceSize(
        nint handle, nint desc, nint find,
        CuTensorWorksizePreference pref = CuTensorWorksizePreference.Recommended)
    {
        nuint sz;
        Check(_getWorkspace.Value(handle, desc, find, pref, &sz), "cutensorContractionGetWorkspaceSize");
        return sz;
    }

    /// <summary>Creates an execution plan for the contraction, picking an algorithm.</summary>
    public static nint CreateContractionPlan(nint handle, nint desc, nint find, nuint workspaceSize)
    {
        nint plan;
        Check(_createPlan.Value(handle, &plan, desc, find, workspaceSize), "cutensorCreateContractionPlan");
        return plan;
    }

    /// <summary>Destroys a contraction execution plan.</summary>
    public static void DestroyContractionPlan(nint plan) =>
        Check(_destroyPlan.Value(plan), "cutensorDestroyContractionPlan");

    /// <summary>
    /// Executes D = alpha * A * B + beta * C using the precomputed plan.
    /// All data pointers must point to device memory.
    /// </summary>
    public static void Contract(
        nint handle, nint plan,
        void* alpha, void* a, void* b,
        void* beta, void* c, void* d,
        void* workspace, nuint workspaceSize, nint stream) =>
        Check(_contract.Value(handle, plan, alpha, a, b, beta, c, d, workspace, workspaceSize, stream), "cutensorContract");

    // ── Delegate types (element-wise trinary) ─────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorCreateElementwiseTrinaryDescriptorDelegate(
        nint handle, nint* desc,
        nint descA, int* modeA, CuTensorOperator opA,
        nint descB, int* modeB, CuTensorOperator opB,
        nint descC, int* modeC, CuTensorOperator opC,
        nint descD, int* modeD,
        CuTensorOperator opAB,
        CuTensorOperator opABC,
        CudaDataType typeScalar);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorDestroyElementwiseTrinaryDescriptorDelegate(nint desc);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorElementwiseTrinaryExecuteDelegate(
        nint handle, nint desc,
        void* alpha, void* a,
        void* beta,  void* b,
        void* gamma, void* c,
        void* d,
        nint stream);

    // ── Lazy-loaded delegates (element-wise trinary) ──────────────────────────

    private static readonly Lazy<cutensorCreateElementwiseTrinaryDescriptorDelegate>  _createEwTriDesc  = new(() => Load<cutensorCreateElementwiseTrinaryDescriptorDelegate>("cutensorCreateElementwiseTrinaryDescriptor"));
    private static readonly Lazy<cutensorDestroyElementwiseTrinaryDescriptorDelegate> _destroyEwTriDesc = new(() => Load<cutensorDestroyElementwiseTrinaryDescriptorDelegate>("cutensorDestroyOperationDescriptor"));
    private static readonly Lazy<cutensorElementwiseTrinaryExecuteDelegate>           _ewTriExec        = new(() => Load<cutensorElementwiseTrinaryExecuteDelegate>("cutensorElementwiseTrinaryExecute"));

    // ── Public API (element-wise trinary) ─────────────────────────────────────

    /// <summary>
    /// Creates an element-wise trinary operation descriptor computing:
    /// D[modeD] = opABC(alpha * opA(A[modeA]), opAB(beta * opB(B[modeB]), gamma * opC(C[modeC]))).
    /// </summary>
    /// <param name="handle">cuTENSOR handle.</param>
    /// <param name="descA">Tensor descriptor for A.</param>
    /// <param name="modeA">Mode labels for A.</param>
    /// <param name="opA">Unary operator applied element-wise to A.</param>
    /// <param name="descB">Tensor descriptor for B.</param>
    /// <param name="modeB">Mode labels for B.</param>
    /// <param name="opB">Unary operator applied element-wise to B.</param>
    /// <param name="descC">Tensor descriptor for C.</param>
    /// <param name="modeC">Mode labels for C.</param>
    /// <param name="opC">Unary operator applied element-wise to C.</param>
    /// <param name="descD">Tensor descriptor for D (output).</param>
    /// <param name="modeD">Mode labels for D.</param>
    /// <param name="opAB">Binary operator combining the alpha*opA(A) and beta*opB(B) terms.</param>
    /// <param name="opABC">Binary operator combining the prior result with gamma*opC(C).</param>
    /// <param name="scalarType">Data type of alpha, beta, gamma scalars.</param>
    /// <returns>Opaque operation descriptor handle.</returns>
    public static nint CreateElementwiseTrinaryDescriptor(
        nint handle,
        nint descA, int[] modeA, CuTensorOperator opA,
        nint descB, int[] modeB, CuTensorOperator opB,
        nint descC, int[] modeC, CuTensorOperator opC,
        nint descD, int[] modeD,
        CuTensorOperator opAB  = CuTensorOperator.Add,
        CuTensorOperator opABC = CuTensorOperator.Add,
        CudaDataType scalarType = CudaDataType.R_32F)
    {
        nint desc;
        fixed (int* pmA = modeA, pmB = modeB, pmC = modeC, pmD = modeD)
            Check(_createEwTriDesc.Value(handle, &desc,
                descA, pmA, opA, descB, pmB, opB, descC, pmC, opC, descD, pmD,
                opAB, opABC, scalarType), "cutensorCreateElementwiseTrinaryDescriptor");
        return desc;
    }

    /// <summary>Destroys an element-wise trinary operation descriptor.</summary>
    public static void DestroyElementwiseTrinaryDescriptor(nint desc) =>
        Check(_destroyEwTriDesc.Value(desc), "cutensorDestroyOperationDescriptor(ew-trinary)");

    /// <summary>
    /// Executes D = opABC(alpha * opA(A), opAB(beta * opB(B), gamma * opC(C))) element-wise.
    /// All data pointers must point to device memory.
    /// </summary>
    /// <param name="handle">cuTENSOR handle.</param>
    /// <param name="desc">Operation descriptor created with <see cref="CreateElementwiseTrinaryDescriptor"/>.</param>
    /// <param name="alpha">Pointer to scalar alpha (host or device memory).</param>
    /// <param name="a">Device pointer to tensor A.</param>
    /// <param name="beta">Pointer to scalar beta (host or device memory).</param>
    /// <param name="b">Device pointer to tensor B.</param>
    /// <param name="gamma">Pointer to scalar gamma (host or device memory).</param>
    /// <param name="c">Device pointer to tensor C.</param>
    /// <param name="d">Device pointer to output tensor D.</param>
    /// <param name="stream">CUDA stream handle.</param>
    public static void ElementwiseTrinaryExecute(
        nint handle, nint desc,
        void* alpha, void* a,
        void* beta,  void* b,
        void* gamma, void* c,
        void* d, nint stream) =>
        Check(_ewTriExec.Value(handle, desc, alpha, a, beta, b, gamma, c, d, stream),
            "cutensorElementwiseTrinaryExecute");
}
