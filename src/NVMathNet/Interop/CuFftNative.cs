// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.InteropServices;

namespace NVMathNet.Interop;

/// <summary>
/// cuFFT status codes.
/// </summary>
public enum CuFftResult : int
{
    /// <summary>Operation completed successfully.</summary>
    Success        = 0x0,
    /// <summary>The plan handle is invalid.</summary>
    InvalidPlan    = 0x1,
    /// <summary>cuFFT failed to allocate GPU memory.</summary>
    AllocFailed    = 0x2,
    /// <summary>Unsupported data type.</summary>
    InvalidType    = 0x3,
    /// <summary>A parameter has an invalid value.</summary>
    InvalidValue   = 0x4,
    /// <summary>An internal driver error occurred.</summary>
    InternalError  = 0x5,
    /// <summary>Plan execution failed.</summary>
    ExecFailed     = 0x6,
    /// <summary>Library initialisation failed.</summary>
    SetupFailed    = 0x7,
    /// <summary>The transform size is not supported.</summary>
    InvalidSize    = 0x8,
    /// <summary>Input data is not aligned to the required boundary.</summary>
    UnalignedData  = 0x9,
    /// <summary>One or more required plan parameters are missing.</summary>
    IncompleteParameterList = 0xA,
    /// <summary>Plan was created for a different GPU device.</summary>
    InvalidDevice  = 0xB,
    /// <summary>Error parsing the plan configuration.</summary>
    ParseError     = 0xC,
    /// <summary>No workspace was provided when one is required.</summary>
    NoWorkspace    = 0xD,
    /// <summary>Requested feature is not implemented in this version.</summary>
    NotImplemented = 0xE,
    /// <summary>cuFFT license check failed (deprecated).</summary>
    LicenseError   = 0x0F,
    /// <summary>The operation is not supported on this GPU.</summary>
    NotSupported   = 0x10,
    /// <summary>A required dependency (e.g. cuBLAS) could not be loaded.</summary>
    MissingDependency = 0x11,
    /// <summary>NVRTC JIT compilation failed.</summary>
    NvrtcFailure   = 0x12,
}

/// <summary>
/// cuFFT transform types.
/// </summary>
public enum CuFftType : int
{
    /// <summary>Real single-precision to complex (forward half-spectrum).</summary>
    R2C = 0x2a,
    /// <summary>Complex to real single-precision (inverse half-spectrum).</summary>
    C2R = 0x2c,
    /// <summary>Complex single-precision to complex (full spectrum).</summary>
    C2C = 0x29,
    /// <summary>Real double-precision to complex (forward half-spectrum).</summary>
    D2Z = 0x6a,
    /// <summary>Complex to real double-precision (inverse half-spectrum).</summary>
    Z2D = 0x6c,
    /// <summary>Complex double-precision to complex (full spectrum).</summary>
    Z2Z = 0x69,
}

/// <summary>
/// cuFFT transform direction.
/// </summary>
public enum CuFftDirection : int
{
    /// <summary>Forward FFT (time → frequency domain). Corresponds to the <c>CUFFT_FORWARD</c> constant.</summary>
    Forward = -1,
    /// <summary>Inverse FFT (frequency → time domain). Corresponds to the <c>CUFFT_INVERSE</c> constant.</summary>
    Inverse = 1,
}

/// <summary>
/// Raw P/Invoke bindings for the cuFFT library.
/// </summary>
public static unsafe class CuFftNative
{
    private const string LibWindows = "cufft64_12.dll";
    private const string LibLinux   = "libcufft.so.12";

    private static readonly string LibName =
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? LibWindows : LibLinux;

    // ── Delegate types ────────────────────────────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuFftResult cufftCreateDelegate(nint* plan);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuFftResult cufftDestroyDelegate(nint plan);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuFftResult cufftSetStreamDelegate(nint plan, nint stream);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuFftResult cufftSetAutoAllocationDelegate(nint plan, int autoAllocate);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuFftResult cufftSetWorkAreaDelegate(nint plan, void* workArea);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuFftResult cufftGetSizeDelegate(nint plan, nuint* workSize);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuFftResult cufftMakePlanManyDelegate(
        nint plan, int rank, long* n,
        long* inembed, long istride, long idist,
        long* onembed, long ostride, long odist,
        CuFftType type, long batch, nuint* workSize);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuFftResult cufftXtMakePlanManyDelegate(
        nint plan, int rank, long* n,
        long* inembed, long istride, long idist, int inputType,
        long* onembed, long ostride, long odist, int outputType,
        long batch, nuint* workSize, int executionType);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuFftResult cufftXtExecDelegate(
        nint plan, void* input, void* output, int direction);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuFftResult cufftExecC2CDelegate(nint plan, void* idata, void* odata, int direction);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuFftResult cufftExecZ2ZDelegate(nint plan, void* idata, void* odata, int direction);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuFftResult cufftExecR2CDelegate(nint plan, void* idata, void* odata);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuFftResult cufftExecC2RDelegate(nint plan, void* idata, void* odata);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuFftResult cufftExecD2ZDelegate(nint plan, void* idata, void* odata);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuFftResult cufftExecZ2DDelegate(nint plan, void* idata, void* odata);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuFftResult cufftEstimateManyDelegate(
        int rank, int* n, int* inembed, int istride, int idist,
        int* onembed, int ostride, int odist,
        CuFftType type, int batch, nuint* workSize);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuFftResult cufftGetVersionDelegate(int* version);

    // ── Lazy-loaded delegates ─────────────────────────────────────────────────

    private static T Load<T>(string name) where T : Delegate =>
        Marshal.GetDelegateForFunctionPointer<T>(NativeLibraryLoader.GetExport(LibName, name));

    private static readonly Lazy<cufftCreateDelegate>            _create          = new(() => Load<cufftCreateDelegate>("cufftCreate"));
    private static readonly Lazy<cufftDestroyDelegate>           _destroy         = new(() => Load<cufftDestroyDelegate>("cufftDestroy"));
    private static readonly Lazy<cufftSetStreamDelegate>         _setStream       = new(() => Load<cufftSetStreamDelegate>("cufftSetStream"));
    private static readonly Lazy<cufftSetAutoAllocationDelegate> _setAutoAlloc    = new(() => Load<cufftSetAutoAllocationDelegate>("cufftSetAutoAllocation"));
    private static readonly Lazy<cufftSetWorkAreaDelegate>       _setWorkArea     = new(() => Load<cufftSetWorkAreaDelegate>("cufftSetWorkArea"));
    private static readonly Lazy<cufftGetSizeDelegate>           _getSize         = new(() => Load<cufftGetSizeDelegate>("cufftGetSize"));
    private static readonly Lazy<cufftMakePlanManyDelegate>      _makePlanMany    = new(() => Load<cufftMakePlanManyDelegate>("cufftMakePlanMany64"));
    private static readonly Lazy<cufftXtMakePlanManyDelegate>    _xtMakePlanMany  = new(() => Load<cufftXtMakePlanManyDelegate>("cufftXtMakePlanMany"));
    private static readonly Lazy<cufftXtExecDelegate>            _xtExec          = new(() => Load<cufftXtExecDelegate>("cufftXtExec"));
    private static readonly Lazy<cufftExecC2CDelegate>           _execC2C         = new(() => Load<cufftExecC2CDelegate>("cufftExecC2C"));
    private static readonly Lazy<cufftExecZ2ZDelegate>           _execZ2Z         = new(() => Load<cufftExecZ2ZDelegate>("cufftExecZ2Z"));
    private static readonly Lazy<cufftExecR2CDelegate>           _execR2C         = new(() => Load<cufftExecR2CDelegate>("cufftExecR2C"));
    private static readonly Lazy<cufftExecC2RDelegate>           _execC2R         = new(() => Load<cufftExecC2RDelegate>("cufftExecC2R"));
    private static readonly Lazy<cufftExecD2ZDelegate>           _execD2Z         = new(() => Load<cufftExecD2ZDelegate>("cufftExecD2Z"));
    private static readonly Lazy<cufftExecZ2DDelegate>           _execZ2D         = new(() => Load<cufftExecZ2DDelegate>("cufftExecZ2D"));
    private static readonly Lazy<cufftEstimateManyDelegate>      _estimateMany    = new(() => Load<cufftEstimateManyDelegate>("cufftEstimateMany"));
    private static readonly Lazy<cufftGetVersionDelegate>        _getVersion      = new(() => Load<cufftGetVersionDelegate>("cufftGetVersion"));

    // ── Public helpers ─────────────────────────────────────────────────────────

    /// <summary>
    /// Throws <see cref="CuFftException"/> if <paramref name="result"/> is not
    /// <see cref="CuFftResult.Success"/>. Optionally prefixes the message with
    /// <paramref name="context"/>.
    /// </summary>
    public static void Check(CuFftResult result, string? context = null)
    {
        if (result == CuFftResult.Success)
        {
            return;
        }

        string msg = $"cuFFT error {result} ({(int)result:X})";
        throw new CuFftException((int)result, context is null ? msg : $"{context}: {msg}");
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    /// <summary>Creates a cuFFT plan handle. Caller must call <see cref="Destroy"/> when done.</summary>
    public static nint Create()
    {
        nint plan;
        Check(_create.Value(&plan), "cufftCreate");
        return plan;
    }

    /// <summary>Destroys a cuFFT plan handle.</summary>
    public static void Destroy(nint plan) =>
        Check(_destroy.Value(plan), "cufftDestroy");

    /// <summary>Associates a CUDA stream with the plan for asynchronous execution.</summary>
    public static void SetStream(nint plan, nint stream) =>
        Check(_setStream.Value(plan, stream), "cufftSetStream");

    /// <summary>
    /// Controls whether cuFFT automatically allocates its own workspace.
    /// Set to <c>false</c> to supply a custom workspace via <see cref="SetWorkArea"/>.
    /// </summary>
    public static void SetAutoAllocation(nint plan, bool autoAllocate) =>
        Check(_setAutoAlloc.Value(plan, autoAllocate ? 1 : 0), "cufftSetAutoAllocation");

    /// <summary>Provides a caller-owned device workspace buffer to the plan.</summary>
    public static void SetWorkArea(nint plan, void* workArea) =>
        Check(_setWorkArea.Value(plan, workArea), "cufftSetWorkArea");

    /// <summary>Returns the size of the workspace (in bytes) required by <paramref name="plan"/>.</summary>
    public static nuint GetSize(nint plan)
    {
        nuint size;
        Check(_getSize.Value(plan, &size), "cufftGetSize");
        return size;
    }

    /// <summary>
    /// Builds a batched N-D FFT plan using the 64-bit <c>cufftMakePlanMany64</c> API.
    /// Returns the required workspace size in bytes.
    /// </summary>
    /// <param name="plan">Plan handle.</param>
    /// <param name="rank">Number of FFT dimensions.</param>
    /// <param name="n">Size of each FFT dimension.</param>
    /// <param name="inembed">Storage dimensions of the input data; <c>null</c> = contiguous.</param>
    /// <param name="istride">Distance between successive input elements in the innermost dimension.</param>
    /// <param name="idist">Stride between the first element of consecutive batches in the input.</param>
    /// <param name="onembed">Storage dimensions of the output data; <c>null</c> = contiguous.</param>
    /// <param name="ostride">Distance between successive output elements in the innermost dimension.</param>
    /// <param name="odist">Stride between the first element of consecutive batches in the output.</param>
    /// <param name="type">Transform type (e.g. <see cref="CuFftType.C2C"/>).</param>
    /// <param name="batch">Number of independent transforms in the batch.</param>
    public static nuint MakePlanMany(
        nint plan, int rank, long[] n,
        long[]? inembed, long istride, long idist,
        long[]? onembed, long ostride, long odist,
        CuFftType type, long batch)
    {
        nuint workSize;
        fixed (long* pn = n, pi = inembed, po = onembed)
        {
            Check(_makePlanMany.Value(plan, rank, pn, pi, istride, idist, po, ostride, odist, type, batch, &workSize),
                "cufftMakePlanMany64");
        }

        return workSize;
    }

    /// <summary>
    /// Builds a batched N-D FFT plan using the cuFFTXt <c>cufftXtMakePlanMany</c> API
    /// for mixed-precision and custom data types.
    /// Returns the required workspace size in bytes.
    /// </summary>
    public static nuint XtMakePlanMany(
        nint plan, int rank, long[] n,
        long[]? inembed, long istride, long idist, int inputType,
        long[]? onembed, long ostride, long odist, int outputType,
        long batch, int executionType)
    {
        nuint workSize;
        fixed (long* pn = n, pi = inembed, po = onembed)
        {
            Check(_xtMakePlanMany.Value(plan, rank, pn, pi, istride, idist, inputType,
                po, ostride, odist, outputType, batch, &workSize, executionType),
                "cufftXtMakePlanMany");
        }

        return workSize;
    }

    /// <summary>Executes a cuFFTXt transform (mixed-precision / custom data types).</summary>
    public static void XtExec(nint plan, void* input, void* output, CuFftDirection direction) =>
        Check(_xtExec.Value(plan, input, output, (int)direction), "cufftXtExec");

    /// <summary>Executes a complex-to-complex single-precision FFT (C2C).</summary>
    public static void ExecC2C(nint plan, void* idata, void* odata, CuFftDirection direction) =>
        Check(_execC2C.Value(plan, idata, odata, (int)direction), "cufftExecC2C");

    /// <summary>Executes a complex-to-complex double-precision FFT (Z2Z).</summary>
    public static void ExecZ2Z(nint plan, void* idata, void* odata, CuFftDirection direction) =>
        Check(_execZ2Z.Value(plan, idata, odata, (int)direction), "cufftExecZ2Z");

    /// <summary>Executes a real-to-complex single-precision forward FFT (R2C).</summary>
    public static void ExecR2C(nint plan, void* idata, void* odata) =>
        Check(_execR2C.Value(plan, idata, odata), "cufftExecR2C");

    /// <summary>Executes a complex-to-real single-precision inverse FFT (C2R).</summary>
    public static void ExecC2R(nint plan, void* idata, void* odata) =>
        Check(_execC2R.Value(plan, idata, odata), "cufftExecC2R");

    /// <summary>Executes a real-to-complex double-precision forward FFT (D2Z).</summary>
    public static void ExecD2Z(nint plan, void* idata, void* odata) =>
        Check(_execD2Z.Value(plan, idata, odata), "cufftExecD2Z");

    /// <summary>Executes a complex-to-real double-precision inverse FFT (Z2D).</summary>
    public static void ExecZ2D(nint plan, void* idata, void* odata) =>
        Check(_execZ2D.Value(plan, idata, odata), "cufftExecZ2D");

    /// <summary>Returns the cuFFT library version number (e.g. 11000 = v11.0).</summary>
    public static int GetVersion()
    {
        int v;
        Check(_getVersion.Value(&v), "cufftGetVersion");
        return v;
    }
}
