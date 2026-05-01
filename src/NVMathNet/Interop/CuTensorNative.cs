// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

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
    /// <summary>GETT algorithm (only applicable to contractions).</summary>
    Gett            = -4,
    /// <summary>Transpose + GETT (only applicable to contractions).</summary>
    Tgett           = -3,
    /// <summary>Transpose-Transpose-GEMM-Transpose algorithm (only applicable to contractions).</summary>
    Ttgt            = -2,
    /// <summary>Automatic algorithm selection (recommended).</summary>
    Default         = -1,
}

/// <summary>cuTENSOR workspace size preference hint.</summary>
public enum CuTensorWorksizePreference : int
{
    /// <summary>Use only the minimum required workspace; at least one algorithm will be available.</summary>
    Min         = 1,
    /// <summary>Aims to attain high performance while also reducing the workspace requirement (default).</summary>
    Default     = 2,
    /// <summary>Allocate the maximum workspace; all algorithms will be available.</summary>
    Max         = 3,
}

/// <summary>cuTENSOR JIT compilation mode.</summary>
public enum CuTensorJitMode : int
{
    /// <summary>No JIT compilation.</summary>
    None = 0,
    /// <summary>Default JIT mode.</summary>
    Default = 1,
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
        CudaDataType dataType, uint alignmentRequirement);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorDestroyTensorDescriptorDelegate(nint desc);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorCreateContractionDelegate(
        nint handle, nint* desc,
        nint descA, int* modeA, CuTensorOperator opA,
        nint descB, int* modeB, CuTensorOperator opB,
        nint descC, int* modeC, CuTensorOperator opC,
        nint descD, int* modeD,
        nint descCompute);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorDestroyOperationDescriptorDelegate(nint desc);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorCreatePlanPreferenceDelegate(
        nint handle, nint* pref, CuTensorAlgo algo, CuTensorJitMode jitMode);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorDestroyPlanPreferenceDelegate(nint pref);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorEstimateWorkspaceSizeDelegate(
        nint handle, nint desc, nint planPref,
        CuTensorWorksizePreference pref, ulong* workspaceSizeEstimate);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorCreatePlanDelegate(
        nint handle, nint* plan, nint desc, nint pref, ulong workspaceSizeLimit);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorDestroyPlanDelegate(nint plan);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorContractDelegate(
        nint handle, nint plan,
        void* alpha, void* A, void* B,
        void* beta, void* C, void* D,
        void* workspace, ulong workspaceSize, nint stream);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorCreateElementwiseTrinaryDelegate(
        nint handle, nint* desc,
        nint descA, int* modeA, CuTensorOperator opA,
        nint descB, int* modeB, CuTensorOperator opB,
        nint descC, int* modeC, CuTensorOperator opC,
        nint descD, int* modeD,
        CuTensorOperator opAB,
        CuTensorOperator opABC,
        nint descCompute);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuTensorStatus cutensorElementwiseTrinaryExecuteDelegate(
        nint handle, nint plan,
        void* alpha, void* a,
        void* beta,  void* b,
        void* gamma, void* c,
        void* d,
        nint stream);

    // ── Lazy-loaded delegates ─────────────────────────────────────────────────

    private static T Load<T>(string name) where T : Delegate =>
        Marshal.GetDelegateForFunctionPointer<T>(NativeLibraryLoader.GetExport(LibName, name));

    private static readonly Lazy<cutensorCreateDelegate>                     _create         = new(() => Load<cutensorCreateDelegate>("cutensorCreate"));
    private static readonly Lazy<cutensorDestroyDelegate>                    _destroy        = new(() => Load<cutensorDestroyDelegate>("cutensorDestroy"));
    private static readonly Lazy<cutensorCreateTensorDescriptorDelegate>     _createTensorDesc = new(() => Load<cutensorCreateTensorDescriptorDelegate>("cutensorCreateTensorDescriptor"));
    private static readonly Lazy<cutensorDestroyTensorDescriptorDelegate>    _destroyTensorDesc = new(() => Load<cutensorDestroyTensorDescriptorDelegate>("cutensorDestroyTensorDescriptor"));
    private static readonly Lazy<cutensorCreateContractionDelegate>          _createContr    = new(() => Load<cutensorCreateContractionDelegate>("cutensorCreateContraction"));
    private static readonly Lazy<cutensorDestroyOperationDescriptorDelegate> _destroyOpDesc  = new(() => Load<cutensorDestroyOperationDescriptorDelegate>("cutensorDestroyOperationDescriptor"));
    private static readonly Lazy<cutensorCreatePlanPreferenceDelegate>       _createPlanPref = new(() => Load<cutensorCreatePlanPreferenceDelegate>("cutensorCreatePlanPreference"));
    private static readonly Lazy<cutensorDestroyPlanPreferenceDelegate>      _destroyPlanPref = new(() => Load<cutensorDestroyPlanPreferenceDelegate>("cutensorDestroyPlanPreference"));
    private static readonly Lazy<cutensorEstimateWorkspaceSizeDelegate>      _estimateWs     = new(() => Load<cutensorEstimateWorkspaceSizeDelegate>("cutensorEstimateWorkspaceSize"));
    private static readonly Lazy<cutensorCreatePlanDelegate>                 _createPlan     = new(() => Load<cutensorCreatePlanDelegate>("cutensorCreatePlan"));
    private static readonly Lazy<cutensorDestroyPlanDelegate>                _destroyPlan    = new(() => Load<cutensorDestroyPlanDelegate>("cutensorDestroyPlan"));
    private static readonly Lazy<cutensorContractDelegate>                   _contract       = new(() => Load<cutensorContractDelegate>("cutensorContract"));
    private static readonly Lazy<cutensorCreateElementwiseTrinaryDelegate>   _createEwTri    = new(() => Load<cutensorCreateElementwiseTrinaryDelegate>("cutensorCreateElementwiseTrinary"));
    private static readonly Lazy<cutensorElementwiseTrinaryExecuteDelegate>  _ewTriExec      = new(() => Load<cutensorElementwiseTrinaryExecuteDelegate>("cutensorElementwiseTrinaryExecute"));

    // ── Compute descriptor global variables ───────────────────────────────────

    private static nint LoadComputeDesc(string symbolName)
    {
        nint addr = NativeLibraryLoader.GetExport(LibName, symbolName);
        return *(nint*)addr;
    }

    private static readonly Lazy<nint> _computeDesc16F = new(() => LoadComputeDesc("CUTENSOR_COMPUTE_DESC_16F"));
    private static readonly Lazy<nint> _computeDesc32F = new(() => LoadComputeDesc("CUTENSOR_COMPUTE_DESC_32F"));
    private static readonly Lazy<nint> _computeDesc64F = new(() => LoadComputeDesc("CUTENSOR_COMPUTE_DESC_64F"));

    /// <summary>Gets the cuTENSOR compute descriptor for 16-bit float computation.</summary>
    public static nint ComputeDesc16F => _computeDesc16F.Value;

    /// <summary>Gets the cuTENSOR compute descriptor for 32-bit float computation.</summary>
    public static nint ComputeDesc32F => _computeDesc32F.Value;

    /// <summary>Gets the cuTENSOR compute descriptor for 64-bit double computation.</summary>
    public static nint ComputeDesc64F => _computeDesc64F.Value;

    // ── Public helpers ─────────────────────────────────────────────────────────

    /// <summary>
    /// Throws <see cref="CuTensorException"/> if <paramref name="status"/> is not
    /// <see cref="CuTensorStatus.Success"/>. Optionally prefixes the message with
    /// <paramref name="context"/>.
    /// </summary>
    public static void Check(CuTensorStatus status, string? context = null)
    {
        if (status == CuTensorStatus.Success)
        {
            return;
        }

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
    /// <param name="alignmentRequirement">Alignment in bytes of the data pointer (default: 128).</param>
    public static nint CreateTensorDescriptor(
        nint handle, uint numModes, long[] extent, long[]? stride, CudaDataType dataType,
        uint alignmentRequirement = 128)
    {
        nint desc;
        fixed (long* pe = extent, ps = stride)
        {
            Check(_createTensorDesc.Value(handle, &desc, numModes, pe, ps, dataType, alignmentRequirement), "cutensorCreateTensorDescriptor");
        }

        return desc;
    }

    /// <summary>Destroys a tensor descriptor.</summary>
    public static void DestroyTensorDescriptor(nint desc) =>
        Check(_destroyTensorDesc.Value(desc), "cutensorDestroyTensorDescriptor");

    /// <summary>
    /// Creates a contraction operation descriptor (cuTENSOR v2 API):
    /// D = alpha * opA(A) * opB(B) + beta * opC(C).
    /// </summary>
    public static nint CreateContraction(
        nint handle,
        nint descA, int[] modeA, CuTensorOperator opA,
        nint descB, int[] modeB, CuTensorOperator opB,
        nint descC, int[] modeC, CuTensorOperator opC,
        nint descD, int[] modeD,
        nint descCompute)
    {
        nint desc;
        fixed (int* pmA = modeA, pmB = modeB, pmC = modeC, pmD = modeD)
        {
            Check(_createContr.Value(handle, &desc,
                descA, pmA, opA, descB, pmB, opB,
                descC, pmC, opC, descD, pmD, descCompute), "cutensorCreateContraction");
        }

        return desc;
    }

    /// <summary>Destroys an operation descriptor.</summary>
    public static void DestroyOperationDescriptor(nint desc) =>
        Check(_destroyOpDesc.Value(desc), "cutensorDestroyOperationDescriptor");

    /// <summary>
    /// Creates a plan preference descriptor that influences algorithm selection.
    /// </summary>
    /// <param name="handle">cuTENSOR handle.</param>
    /// <param name="algo">Algorithm hint; use <see cref="CuTensorAlgo.Default"/> for autoselection.</param>
    /// <param name="jitMode">JIT compilation mode; use <see cref="CuTensorJitMode.None"/> by default.</param>
    public static nint CreatePlanPreference(nint handle, CuTensorAlgo algo = CuTensorAlgo.Default,
        CuTensorJitMode jitMode = CuTensorJitMode.None)
    {
        nint pref;
        Check(_createPlanPref.Value(handle, &pref, algo, jitMode), "cutensorCreatePlanPreference");
        return pref;
    }

    /// <summary>Destroys a plan preference descriptor.</summary>
    public static void DestroyPlanPreference(nint pref) =>
        Check(_destroyPlanPref.Value(pref), "cutensorDestroyPlanPreference");

    /// <summary>
    /// Estimates the required workspace size in bytes for the given operation.
    /// </summary>
    public static ulong EstimateWorkspaceSize(
        nint handle, nint desc, nint planPref,
        CuTensorWorksizePreference pref = CuTensorWorksizePreference.Default)
    {
        ulong sz;
        Check(_estimateWs.Value(handle, desc, planPref, pref, &sz), "cutensorEstimateWorkspaceSize");
        return sz;
    }

    /// <summary>Creates an execution plan for an operation.</summary>
    public static nint CreatePlan(nint handle, nint desc, nint pref, ulong workspaceSizeLimit)
    {
        nint plan;
        Check(_createPlan.Value(handle, &plan, desc, pref, workspaceSizeLimit), "cutensorCreatePlan");
        return plan;
    }

    /// <summary>Destroys an execution plan.</summary>
    public static void DestroyPlan(nint plan) =>
        Check(_destroyPlan.Value(plan), "cutensorDestroyPlan");

    /// <summary>
    /// Executes D = alpha * A * B + beta * C using the precomputed plan.
    /// All data pointers must point to device memory.
    /// </summary>
    public static void Contract(
        nint handle, nint plan,
        void* alpha, void* a, void* b,
        void* beta, void* c, void* d,
        void* workspace, ulong workspaceSize, nint stream) =>
        Check(_contract.Value(handle, plan, alpha, a, b, beta, c, d, workspace, workspaceSize, stream), "cutensorContract");

    // ── Public API (element-wise trinary) ─────────────────────────────────────

    /// <summary>
    /// Creates an element-wise trinary operation descriptor computing:
    /// D[modeD] = opABC(alpha * opA(A[modeA]), opAB(beta * opB(B[modeB]), gamma * opC(C[modeC]))).
    /// </summary>
    public static nint CreateElementwiseTrinary(
        nint handle,
        nint descA, int[] modeA, CuTensorOperator opA,
        nint descB, int[] modeB, CuTensorOperator opB,
        nint descC, int[] modeC, CuTensorOperator opC,
        nint descD, int[] modeD,
        CuTensorOperator opAB  = CuTensorOperator.Add,
        CuTensorOperator opABC = CuTensorOperator.Add,
        nint descCompute = default)
    {
        if (descCompute == default)
        {
            descCompute = ComputeDesc32F;
        }

        nint desc;
        fixed (int* pmA = modeA, pmB = modeB, pmC = modeC, pmD = modeD)
        {
            Check(_createEwTri.Value(handle, &desc,
                descA, pmA, opA, descB, pmB, opB, descC, pmC, opC, descD, pmD,
                opAB, opABC, descCompute), "cutensorCreateElementwiseTrinary");
        }

        return desc;
    }

    /// <summary>
    /// Executes D = opABC(alpha * opA(A), opAB(beta * opB(B), gamma * opC(C))) element-wise
    /// using a precomputed plan.
    /// </summary>
    public static void ElementwiseTrinaryExecute(
        nint handle, nint plan,
        void* alpha, void* a,
        void* beta,  void* b,
        void* gamma, void* c,
        void* d, nint stream) =>
        Check(_ewTriExec.Value(handle, plan, alpha, a, beta, b, gamma, c, d, stream),
            "cutensorElementwiseTrinaryExecute");
}
