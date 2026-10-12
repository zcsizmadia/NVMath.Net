// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.InteropServices;

namespace NVMathNet.Interop;

/// <summary>cuDSS status codes (<c>cudssStatus_t</c>).</summary>
public enum CuDssStatus : int
{
    /// <summary>The operation completed successfully.</summary>
    Success = 0,
    /// <summary>The library was not initialised.</summary>
    NotInitialized = 1,
    /// <summary>A memory allocation failed.</summary>
    AllocFailed = 2,
    /// <summary>An argument has an invalid value.</summary>
    InvalidValue = 3,
    /// <summary>The operation is not supported.</summary>
    NotSupported = 4,
    /// <summary>Execution on the GPU failed.</summary>
    ExecutionFailed = 5,
    /// <summary>An internal error occurred.</summary>
    InternalError = 6,
    /// <summary>Iterative refinement failed.</summary>
    IrFailed = 7,
}

/// <summary>Value types used by cuDSS (<c>cudssDataType_t</c>; same numbering as <c>cudaDataType</c>).</summary>
public enum CuDssDataType : int
{
    /// <summary>32-bit float.</summary>
    R32F = 0,
    /// <summary>64-bit float.</summary>
    R64F = 1,
    /// <summary>Complex 32-bit float.</summary>
    C32F = 4,
    /// <summary>Complex 64-bit float.</summary>
    C64F = 5,
    /// <summary>32-bit integer (index/offset type).</summary>
    R32I = 10,
    /// <summary>64-bit integer (index/offset type).</summary>
    R64I = 24,
}

/// <summary>Structure of the system matrix (<c>cudssMatrixType_t</c>).</summary>
public enum CuDssMatrixType : int
{
    /// <summary>General (non-symmetric) matrix; LU factorisation.</summary>
    General = 0,
    /// <summary>Symmetric matrix; LDL^T factorisation.</summary>
    Symmetric = 1,
    /// <summary>Hermitian matrix; LDL^H factorisation.</summary>
    Hermitian = 2,
    /// <summary>Symmetric positive definite; Cholesky factorisation.</summary>
    Spd = 3,
    /// <summary>Hermitian positive definite; Cholesky factorisation.</summary>
    Hpd = 4,
}

/// <summary>Which part of the matrix is stored (<c>cudssMatrixViewType_t</c>).</summary>
public enum CuDssMatrixView : int
{
    /// <summary>The full matrix.</summary>
    Full = 0,
    /// <summary>Only the lower triangle (symmetric/Hermitian/SPD matrices).</summary>
    Lower = 1,
    /// <summary>Only the upper triangle (symmetric/Hermitian/SPD matrices).</summary>
    Upper = 2,
}

/// <summary>Fill-reducing reordering algorithms (<c>cudssReorderingAlg_t</c>).</summary>
public enum CuDssReorderingAlg : int
{
    /// <summary>Default algorithm.</summary>
    Default = 0,
    /// <summary>Block triangular form with COLAMD.</summary>
    BtfColamd = 1,
    /// <summary>COLAMD.</summary>
    Colamd = 2,
    /// <summary>Approximate minimum degree.</summary>
    Amd = 3,
    /// <summary>Nested dissection.</summary>
    NestedDissection = 4,
}

/// <summary>Solver phases (<c>cudssPhase_t</c>); phases can be combined with bitwise or.</summary>
[Flags]
public enum CuDssPhase : int
{
    /// <summary>Reordering.</summary>
    Reordering = 1 << 0,
    /// <summary>Symbolic factorisation.</summary>
    SymbolicFactorization = 1 << 1,
    /// <summary>Reordering and symbolic factorisation.</summary>
    Analysis = Reordering | SymbolicFactorization,
    /// <summary>Numerical factorisation.</summary>
    Factorization = 1 << 2,
    /// <summary>Numerical refactorisation with unchanged sparsity pattern.</summary>
    Refactorization = 1 << 3,
    /// <summary>Forward/backward substitution including permutations and refinement.</summary>
    Solve = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9),
}

/// <summary>Solver configuration parameters (<c>cudssConfigParam_t</c>).</summary>
public enum CuDssConfigParam : int
{
    /// <summary>Reordering algorithm (<see cref="CuDssReorderingAlg"/>).</summary>
    ReorderingAlg = 0,
    /// <summary>Maximum number of iterative-refinement steps (int).</summary>
    IrNSteps = 5,
    /// <summary>Iterative-refinement tolerance (double).</summary>
    IrTol = 6,
}

/// <summary>Solver data parameters (<c>cudssDataParam_t</c>).</summary>
public enum CuDssDataParam : int
{
    /// <summary>Status of the last factorisation (int, output).</summary>
    Info = 0,
    /// <summary>Number of non-zeros in the factors (int64, output).</summary>
    LuNnz = 1,
    /// <summary>Number of pivots perturbed during factorisation (int, output).</summary>
    NPivots = 2,
}

/// <summary>Exception thrown for cuDSS errors.</summary>
/// <param name="code">The raw cuDSS status code.</param>
/// <param name="message">The error message.</param>
public sealed class CuDssException(int code, string message) : Exception(message)
{
    /// <summary>The raw cuDSS status code.</summary>
    public int ErrorCode { get; } = code;
}

/// <summary>
/// Raw bindings for cuDSS (sparse direct solver). Signatures verified against <c>cudss.h</c> 0.8.
/// </summary>
public static unsafe class CuDssNative
{
    private const string LibWindows = "cudss64_0.dll";
    private const string LibLinux = "libcudss.so.0";

    private static readonly string LibName =
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? LibWindows : LibLinux;

    private static nint Fn(string name) => NativeLibraryLoader.GetExport(LibName, name);

    /// <summary>Throws <see cref="CuDssException"/> if <paramref name="status"/> is not success.</summary>
    public static void Check(CuDssStatus status, string? context = null)
    {
        if (status == CuDssStatus.Success)
        {
            return;
        }

        string msg = $"cuDSS error {status} ({(int)status})";
        throw new CuDssException((int)status, context is null ? msg : $"{context}: {msg}");
    }

    // ── Handle / stream ───────────────────────────────────────────────────────

    /// <summary>Creates a cuDSS library handle.</summary>
    public static nint Create()
    {
        nint h;
        Check(((delegate* unmanaged[Cdecl]<nint*, CuDssStatus>)Fn("cudssCreate"))(&h), "cudssCreate");
        return h;
    }

    /// <summary>Destroys a cuDSS library handle.</summary>
    public static void Destroy(nint handle) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuDssStatus>)Fn("cudssDestroy"))(handle), "cudssDestroy");

    /// <summary>Associates a CUDA stream with the handle.</summary>
    public static void SetStream(nint handle, nint stream) =>
        Check(((delegate* unmanaged[Cdecl]<nint, nint, CuDssStatus>)Fn("cudssSetStream"))(handle, stream), "cudssSetStream");

    // ── Config / data ─────────────────────────────────────────────────────────

    /// <summary>Creates a solver configuration object.</summary>
    public static nint ConfigCreate()
    {
        nint c;
        Check(((delegate* unmanaged[Cdecl]<nint*, CuDssStatus>)Fn("cudssConfigCreate"))(&c), "cudssConfigCreate");
        return c;
    }

    /// <summary>Destroys a solver configuration object.</summary>
    public static void ConfigDestroy(nint config) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuDssStatus>)Fn("cudssConfigDestroy"))(config), "cudssConfigDestroy");

    /// <summary>Sets a configuration parameter.</summary>
    public static void ConfigSet(nint config, CuDssConfigParam param, void* value, nuint size) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuDssConfigParam, void*, nuint, CuDssStatus>)Fn("cudssConfigSet"))(
            config, param, value, size), "cudssConfigSet");

    /// <summary>Creates a solver data object (holds the factors).</summary>
    public static nint DataCreate(nint handle)
    {
        nint d;
        Check(((delegate* unmanaged[Cdecl]<nint, nint*, CuDssStatus>)Fn("cudssDataCreate"))(handle, &d), "cudssDataCreate");
        return d;
    }

    /// <summary>Destroys a solver data object.</summary>
    public static void DataDestroy(nint handle, nint data) =>
        Check(((delegate* unmanaged[Cdecl]<nint, nint, CuDssStatus>)Fn("cudssDataDestroy"))(handle, data), "cudssDataDestroy");

    /// <summary>Reads a solver data parameter.</summary>
    public static void DataGet(nint handle, nint data, CuDssDataParam param, void* value, nuint size)
    {
        nuint written;
        Check(((delegate* unmanaged[Cdecl]<nint, nint, CuDssDataParam, void*, nuint, nuint*, CuDssStatus>)Fn("cudssDataGet"))(
            handle, data, param, value, size, &written), "cudssDataGet");
    }

    // ── Matrices ──────────────────────────────────────────────────────────────

    /// <summary>Creates a dense matrix descriptor (column-major).</summary>
    public static nint MatrixCreateDn(long rows, long cols, long ld, void* values, CuDssDataType valueType)
    {
        nint m;
        Check(((delegate* unmanaged[Cdecl]<nint*, long, long, long, void*, CuDssDataType, int, CuDssStatus>)Fn("cudssMatrixCreateDn"))(
            &m, rows, cols, ld, values, valueType, 0 /* CUDSS_LAYOUT_COL_MAJOR */), "cudssMatrixCreateDn");
        return m;
    }

    /// <summary>Creates a CSR sparse matrix descriptor with 32-bit indices and zero-based indexing.</summary>
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns.</param>
    /// <param name="nnz">Number of stored entries.</param>
    /// <param name="rowOffsets">Device pointer to <c>rows+1</c> row offsets.</param>
    /// <param name="colIndices">Device pointer to <c>nnz</c> column indices (sorted within each row).</param>
    /// <param name="values">Device pointer to <c>nnz</c> values.</param>
    /// <param name="valueType">Value type.</param>
    /// <param name="mtype">Matrix structure.</param>
    /// <param name="mview">Stored triangle.</param>
    public static nint MatrixCreateCsr(
        long rows, long cols, long nnz, void* rowOffsets, void* colIndices, void* values,
        CuDssDataType valueType, CuDssMatrixType mtype, CuDssMatrixView mview)
    {
        nint m;
        Check(((delegate* unmanaged[Cdecl]<nint*, long, long, long, void*, void*, void*, void*, CuDssDataType, CuDssDataType, CuDssDataType, CuDssMatrixType, CuDssMatrixView, int, CuDssStatus>)Fn("cudssMatrixCreateCsr"))(
            &m, rows, cols, nnz, rowOffsets, null /* rowEnd: rowOffsets[i+1] */, colIndices, values,
            CuDssDataType.R32I, CuDssDataType.R32I, valueType, mtype, mview, 0 /* CUDSS_BASE_ZERO */), "cudssMatrixCreateCsr");
        return m;
    }

    /// <summary>Rebinds the values array of a matrix (same sparsity pattern).</summary>
    public static void MatrixSetValues(nint matrix, void* values) =>
        Check(((delegate* unmanaged[Cdecl]<nint, void*, CuDssStatus>)Fn("cudssMatrixSetValues"))(matrix, values), "cudssMatrixSetValues");

    /// <summary>Destroys a matrix descriptor.</summary>
    public static void MatrixDestroy(nint matrix) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuDssStatus>)Fn("cudssMatrixDestroy"))(matrix), "cudssMatrixDestroy");

    // ── Execute ───────────────────────────────────────────────────────────────

    /// <summary>Runs one or more solver phases.</summary>
    public static void Execute(nint handle, CuDssPhase phase, nint config, nint data, nint matrix, nint solution, nint rhs) =>
        Check(((delegate* unmanaged[Cdecl]<nint, int, nint, nint, nint, nint, nint, CuDssStatus>)Fn("cudssExecute"))(
            handle, (int)phase, config, data, matrix, solution, rhs), "cudssExecute");
}
