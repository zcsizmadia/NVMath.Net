// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.InteropServices;

namespace NVMathNet.Interop;

/// <summary>cuSOLVER status codes.</summary>
public enum CuSolverStatus : int
{
    /// <summary>Operation completed successfully.</summary>
    Success = 0,
    /// <summary>Library not initialised.</summary>
    NotInitialized = 1,
    /// <summary>GPU memory allocation failed.</summary>
    AllocFailed = 2,
    /// <summary>A parameter has an invalid value.</summary>
    InvalidValue = 3,
    /// <summary>GPU architecture does not support the operation.</summary>
    ArchMismatch = 4,
    /// <summary>GPU memory mapping failed.</summary>
    MappingError = 5,
    /// <summary>Execution on the GPU failed.</summary>
    ExecutionFailed = 6,
    /// <summary>Internal error.</summary>
    InternalError = 7,
    /// <summary>Matrix type is not supported.</summary>
    MatrixTypeNotSupported = 8,
    /// <summary>Operation is not supported.</summary>
    NotSupported = 9,
    /// <summary>Zero pivot encountered during factorisation.</summary>
    ZeroPivot = 10,
    /// <summary>License check failed.</summary>
    InvalidLicense = 11,
    /// <summary>Invalid workspace size.</summary>
    InvalidWorkspace = 31,
}

/// <summary>cuSOLVER eigenvalue computation mode.</summary>
public enum CuSolverEigMode : int
{
    /// <summary>Compute eigenvalues only.</summary>
    NoVector = 0,
    /// <summary>Compute eigenvalues and eigenvectors.</summary>
    Vector = 1,
}

/// <summary>cuBLAS fill mode for symmetric/triangular matrices.</summary>
public enum CuBlasFillMode : int
{
    /// <summary>Lower triangle is stored.</summary>
    Lower = 0,
    /// <summary>Upper triangle is stored.</summary>
    Upper = 1,
    /// <summary>Full matrix is stored.</summary>
    Full = 2,
}

/// <summary>
/// Raw P/Invoke bindings for cuSOLVER dense (cusolverDn).
/// </summary>
public static unsafe class CuSolverNative
{
    private const string LibWindows = "cusolver64_12.dll";
    private const string LibLinux   = "libcusolver.so.12";

    private static readonly string LibName =
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? LibWindows : LibLinux;

    // ── Delegate types ───────────────────────────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnCreateDelegate(nint* handle);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnDestroyDelegate(nint handle);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnSetStreamDelegate(nint handle, nint stream);

    // ── LU factorisation (getrf) ─────────────────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnSgetrf_bufferSizeDelegate(
        nint handle, int m, int n, float* A, int lda, int* lwork);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnDgetrf_bufferSizeDelegate(
        nint handle, int m, int n, double* A, int lda, int* lwork);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnSgetrfDelegate(
        nint handle, int m, int n, float* A, int lda, float* workspace, int* devIpiv, int* devInfo);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnDgetrfDelegate(
        nint handle, int m, int n, double* A, int lda, double* workspace, int* devIpiv, int* devInfo);

    // ── LU solve (getrs) ─────────────────────────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnSgetrsDelegate(
        nint handle, int trans, int n, int nrhs, float* A, int lda, int* devIpiv, float* B, int ldb, int* devInfo);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnDgetrsDelegate(
        nint handle, int trans, int n, int nrhs, double* A, int lda, int* devIpiv, double* B, int ldb, int* devInfo);

    // ── Cholesky factorisation (potrf) ───────────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnSpotrf_bufferSizeDelegate(
        nint handle, CuBlasFillMode uplo, int n, float* A, int lda, int* lwork);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnDpotrf_bufferSizeDelegate(
        nint handle, CuBlasFillMode uplo, int n, double* A, int lda, int* lwork);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnSpotrfDelegate(
        nint handle, CuBlasFillMode uplo, int n, float* A, int lda, float* workspace, int lwork, int* devInfo);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnDpotrfDelegate(
        nint handle, CuBlasFillMode uplo, int n, double* A, int lda, double* workspace, int lwork, int* devInfo);

    // ── SVD (gesvd) ──────────────────────────────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnSgesvd_bufferSizeDelegate(
        nint handle, int m, int n, int* lwork);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnDgesvd_bufferSizeDelegate(
        nint handle, int m, int n, int* lwork);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnSgesvdDelegate(
        nint handle, byte jobu, byte jobvt, int m, int n,
        float* A, int lda, float* S, float* U, int ldu, float* VT, int ldvt,
        float* work, int lwork, float* rwork, int* info);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnDgesvdDelegate(
        nint handle, byte jobu, byte jobvt, int m, int n,
        double* A, int lda, double* S, double* U, int ldu, double* VT, int ldvt,
        double* work, int lwork, double* rwork, int* info);

    // ── Symmetric eigenvalue (syevd) ─────────────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnSsyevd_bufferSizeDelegate(
        nint handle, CuSolverEigMode jobz, CuBlasFillMode uplo, int n, float* A, int lda, float* W, int* lwork);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnDsyevd_bufferSizeDelegate(
        nint handle, CuSolverEigMode jobz, CuBlasFillMode uplo, int n, double* A, int lda, double* W, int* lwork);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnSsyevdDelegate(
        nint handle, CuSolverEigMode jobz, CuBlasFillMode uplo, int n,
        float* A, int lda, float* W, float* work, int lwork, int* info);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnDsyevdDelegate(
        nint handle, CuSolverEigMode jobz, CuBlasFillMode uplo, int n,
        double* A, int lda, double* W, double* work, int lwork, int* info);

    // ── QR factorisation (geqrf) ─────────────────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnSgeqrf_bufferSizeDelegate(
        nint handle, int m, int n, float* A, int lda, int* lwork);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnDgeqrf_bufferSizeDelegate(
        nint handle, int m, int n, double* A, int lda, int* lwork);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnSgeqrfDelegate(
        nint handle, int m, int n, float* A, int lda, float* TAU, float* workspace, int lwork, int* devInfo);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuSolverStatus cusolverDnDgeqrfDelegate(
        nint handle, int m, int n, double* A, int lda, double* TAU, double* workspace, int lwork, int* devInfo);

    // ── Lazy-loaded delegates ────────────────────────────────────────────────

    private static T Load<T>(string name) where T : Delegate =>
        Marshal.GetDelegateForFunctionPointer<T>(NativeLibraryLoader.GetExport(LibName, name));

    private static readonly Lazy<cusolverDnCreateDelegate>    _create    = new(() => Load<cusolverDnCreateDelegate>("cusolverDnCreate"));
    private static readonly Lazy<cusolverDnDestroyDelegate>   _destroy   = new(() => Load<cusolverDnDestroyDelegate>("cusolverDnDestroy"));
    private static readonly Lazy<cusolverDnSetStreamDelegate> _setStream = new(() => Load<cusolverDnSetStreamDelegate>("cusolverDnSetStream"));

    // LU
    private static readonly Lazy<cusolverDnSgetrf_bufferSizeDelegate> _sgetrfBufSize = new(() => Load<cusolverDnSgetrf_bufferSizeDelegate>("cusolverDnSgetrf_bufferSize"));
    private static readonly Lazy<cusolverDnDgetrf_bufferSizeDelegate> _dgetrfBufSize = new(() => Load<cusolverDnDgetrf_bufferSizeDelegate>("cusolverDnDgetrf_bufferSize"));
    private static readonly Lazy<cusolverDnSgetrfDelegate>            _sgetrf        = new(() => Load<cusolverDnSgetrfDelegate>("cusolverDnSgetrf"));
    private static readonly Lazy<cusolverDnDgetrfDelegate>            _dgetrf        = new(() => Load<cusolverDnDgetrfDelegate>("cusolverDnDgetrf"));
    private static readonly Lazy<cusolverDnSgetrsDelegate>            _sgetrs        = new(() => Load<cusolverDnSgetrsDelegate>("cusolverDnSgetrs"));
    private static readonly Lazy<cusolverDnDgetrsDelegate>            _dgetrs        = new(() => Load<cusolverDnDgetrsDelegate>("cusolverDnDgetrs"));

    // Cholesky
    private static readonly Lazy<cusolverDnSpotrf_bufferSizeDelegate> _spotrfBufSize = new(() => Load<cusolverDnSpotrf_bufferSizeDelegate>("cusolverDnSpotrf_bufferSize"));
    private static readonly Lazy<cusolverDnDpotrf_bufferSizeDelegate> _dpotrfBufSize = new(() => Load<cusolverDnDpotrf_bufferSizeDelegate>("cusolverDnDpotrf_bufferSize"));
    private static readonly Lazy<cusolverDnSpotrfDelegate>            _spotrf        = new(() => Load<cusolverDnSpotrfDelegate>("cusolverDnSpotrf"));
    private static readonly Lazy<cusolverDnDpotrfDelegate>            _dpotrf        = new(() => Load<cusolverDnDpotrfDelegate>("cusolverDnDpotrf"));

    // SVD
    private static readonly Lazy<cusolverDnSgesvd_bufferSizeDelegate> _sgesvdBufSize = new(() => Load<cusolverDnSgesvd_bufferSizeDelegate>("cusolverDnSgesvd_bufferSize"));
    private static readonly Lazy<cusolverDnDgesvd_bufferSizeDelegate> _dgesvdBufSize = new(() => Load<cusolverDnDgesvd_bufferSizeDelegate>("cusolverDnDgesvd_bufferSize"));
    private static readonly Lazy<cusolverDnSgesvdDelegate>            _sgesvd        = new(() => Load<cusolverDnSgesvdDelegate>("cusolverDnSgesvd"));
    private static readonly Lazy<cusolverDnDgesvdDelegate>            _dgesvd        = new(() => Load<cusolverDnDgesvdDelegate>("cusolverDnDgesvd"));

    // Eigenvalue
    private static readonly Lazy<cusolverDnSsyevd_bufferSizeDelegate> _ssyevdBufSize = new(() => Load<cusolverDnSsyevd_bufferSizeDelegate>("cusolverDnSsyevd_bufferSize"));
    private static readonly Lazy<cusolverDnDsyevd_bufferSizeDelegate> _dsyevdBufSize = new(() => Load<cusolverDnDsyevd_bufferSizeDelegate>("cusolverDnDsyevd_bufferSize"));
    private static readonly Lazy<cusolverDnSsyevdDelegate>            _ssyevd        = new(() => Load<cusolverDnSsyevdDelegate>("cusolverDnSsyevd"));
    private static readonly Lazy<cusolverDnDsyevdDelegate>            _dsyevd        = new(() => Load<cusolverDnDsyevdDelegate>("cusolverDnDsyevd"));

    // QR
    private static readonly Lazy<cusolverDnSgeqrf_bufferSizeDelegate> _sgeqrfBufSize = new(() => Load<cusolverDnSgeqrf_bufferSizeDelegate>("cusolverDnSgeqrf_bufferSize"));
    private static readonly Lazy<cusolverDnDgeqrf_bufferSizeDelegate> _dgeqrfBufSize = new(() => Load<cusolverDnDgeqrf_bufferSizeDelegate>("cusolverDnDgeqrf_bufferSize"));
    private static readonly Lazy<cusolverDnSgeqrfDelegate>            _sgeqrf        = new(() => Load<cusolverDnSgeqrfDelegate>("cusolverDnSgeqrf"));
    private static readonly Lazy<cusolverDnDgeqrfDelegate>            _dgeqrf        = new(() => Load<cusolverDnDgeqrfDelegate>("cusolverDnDgeqrf"));

    // ── Public helpers ───────────────────────────────────────────────────────

    /// <summary>
    /// Throws <see cref="CuSolverException"/> if <paramref name="status"/> is not
    /// <see cref="CuSolverStatus.Success"/>.
    /// </summary>
    public static void Check(CuSolverStatus status, string? context = null)
    {
        if (status == CuSolverStatus.Success)
        {
            return;
        }

        string msg = $"cuSOLVER error {status} ({(int)status})";
        throw new CuSolverException((int)status, context is null ? msg : $"{context}: {msg}");
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// <summary>Creates a cuSOLVER dense handle.</summary>
    public static nint Create()
    {
        nint h;
        Check(_create.Value(&h), "cusolverDnCreate");
        return h;
    }

    /// <summary>Destroys a cuSOLVER dense handle.</summary>
    public static void Destroy(nint handle) =>
        Check(_destroy.Value(handle), "cusolverDnDestroy");

    /// <summary>Associates a CUDA stream with the handle.</summary>
    public static void SetStream(nint handle, nint stream) =>
        Check(_setStream.Value(handle, stream), "cusolverDnSetStream");

    // ── LU Factorisation ─────────────────────────────────────────────────────

    /// <summary>Queries workspace size for single-precision LU factorisation.</summary>
    public static int SgetrfBufferSize(nint handle, int m, int n, float* A, int lda)
    {
        int lwork;
        Check(_sgetrfBufSize.Value(handle, m, n, A, lda, &lwork), "cusolverDnSgetrf_bufferSize");
        return lwork;
    }

    /// <summary>Queries workspace size for double-precision LU factorisation.</summary>
    public static int DgetrfBufferSize(nint handle, int m, int n, double* A, int lda)
    {
        int lwork;
        Check(_dgetrfBufSize.Value(handle, m, n, A, lda, &lwork), "cusolverDnDgetrf_bufferSize");
        return lwork;
    }

    /// <summary>Single-precision LU factorisation: PA = LU.</summary>
    public static void Sgetrf(nint handle, int m, int n, float* A, int lda, float* workspace, int* devIpiv, int* devInfo) =>
        Check(_sgetrf.Value(handle, m, n, A, lda, workspace, devIpiv, devInfo), "cusolverDnSgetrf");

    /// <summary>Double-precision LU factorisation: PA = LU.</summary>
    public static void Dgetrf(nint handle, int m, int n, double* A, int lda, double* workspace, int* devIpiv, int* devInfo) =>
        Check(_dgetrf.Value(handle, m, n, A, lda, workspace, devIpiv, devInfo), "cusolverDnDgetrf");

    /// <summary>Single-precision LU solve: solves A*X = B given factorised A and pivots.</summary>
    public static void Sgetrs(nint handle, int trans, int n, int nrhs, float* A, int lda, int* devIpiv, float* B, int ldb, int* devInfo) =>
        Check(_sgetrs.Value(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo), "cusolverDnSgetrs");

    /// <summary>Double-precision LU solve: solves A*X = B given factorised A and pivots.</summary>
    public static void Dgetrs(nint handle, int trans, int n, int nrhs, double* A, int lda, int* devIpiv, double* B, int ldb, int* devInfo) =>
        Check(_dgetrs.Value(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo), "cusolverDnDgetrs");

    // ── Cholesky Factorisation ───────────────────────────────────────────────

    /// <summary>Queries workspace size for single-precision Cholesky factorisation.</summary>
    public static int SpotrfBufferSize(nint handle, CuBlasFillMode uplo, int n, float* A, int lda)
    {
        int lwork;
        Check(_spotrfBufSize.Value(handle, uplo, n, A, lda, &lwork), "cusolverDnSpotrf_bufferSize");
        return lwork;
    }

    /// <summary>Queries workspace size for double-precision Cholesky factorisation.</summary>
    public static int DpotrfBufferSize(nint handle, CuBlasFillMode uplo, int n, double* A, int lda)
    {
        int lwork;
        Check(_dpotrfBufSize.Value(handle, uplo, n, A, lda, &lwork), "cusolverDnDpotrf_bufferSize");
        return lwork;
    }

    /// <summary>Single-precision Cholesky factorisation: A = L*L^T or U^T*U.</summary>
    public static void Spotrf(nint handle, CuBlasFillMode uplo, int n, float* A, int lda, float* workspace, int lwork, int* devInfo) =>
        Check(_spotrf.Value(handle, uplo, n, A, lda, workspace, lwork, devInfo), "cusolverDnSpotrf");

    /// <summary>Double-precision Cholesky factorisation: A = L*L^T or U^T*U.</summary>
    public static void Dpotrf(nint handle, CuBlasFillMode uplo, int n, double* A, int lda, double* workspace, int lwork, int* devInfo) =>
        Check(_dpotrf.Value(handle, uplo, n, A, lda, workspace, lwork, devInfo), "cusolverDnDpotrf");

    // ── SVD ──────────────────────────────────────────────────────────────────

    /// <summary>Queries workspace size for single-precision SVD.</summary>
    public static int SgesvdBufferSize(nint handle, int m, int n)
    {
        int lwork;
        Check(_sgesvdBufSize.Value(handle, m, n, &lwork), "cusolverDnSgesvd_bufferSize");
        return lwork;
    }

    /// <summary>Queries workspace size for double-precision SVD.</summary>
    public static int DgesvdBufferSize(nint handle, int m, int n)
    {
        int lwork;
        Check(_dgesvdBufSize.Value(handle, m, n, &lwork), "cusolverDnDgesvd_bufferSize");
        return lwork;
    }

    /// <summary>Single-precision SVD: A = U*S*VT.</summary>
    public static void Sgesvd(
        nint handle, byte jobu, byte jobvt, int m, int n,
        float* A, int lda, float* S, float* U, int ldu, float* VT, int ldvt,
        float* work, int lwork, float* rwork, int* info) =>
        Check(_sgesvd.Value(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info), "cusolverDnSgesvd");

    /// <summary>Double-precision SVD: A = U*S*VT.</summary>
    public static void Dgesvd(
        nint handle, byte jobu, byte jobvt, int m, int n,
        double* A, int lda, double* S, double* U, int ldu, double* VT, int ldvt,
        double* work, int lwork, double* rwork, int* info) =>
        Check(_dgesvd.Value(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info), "cusolverDnDgesvd");

    // ── Symmetric Eigenvalue ─────────────────────────────────────────────────

    /// <summary>Queries workspace size for single-precision symmetric eigenvalue decomposition.</summary>
    public static int SsyevdBufferSize(nint handle, CuSolverEigMode jobz, CuBlasFillMode uplo, int n, float* A, int lda, float* W)
    {
        int lwork;
        Check(_ssyevdBufSize.Value(handle, jobz, uplo, n, A, lda, W, &lwork), "cusolverDnSsyevd_bufferSize");
        return lwork;
    }

    /// <summary>Queries workspace size for double-precision symmetric eigenvalue decomposition.</summary>
    public static int DsyevdBufferSize(nint handle, CuSolverEigMode jobz, CuBlasFillMode uplo, int n, double* A, int lda, double* W)
    {
        int lwork;
        Check(_dsyevdBufSize.Value(handle, jobz, uplo, n, A, lda, W, &lwork), "cusolverDnDsyevd_bufferSize");
        return lwork;
    }

    /// <summary>Single-precision symmetric eigenvalue decomposition.</summary>
    public static void Ssyevd(nint handle, CuSolverEigMode jobz, CuBlasFillMode uplo, int n,
        float* A, int lda, float* W, float* work, int lwork, int* info) =>
        Check(_ssyevd.Value(handle, jobz, uplo, n, A, lda, W, work, lwork, info), "cusolverDnSsyevd");

    /// <summary>Double-precision symmetric eigenvalue decomposition.</summary>
    public static void Dsyevd(nint handle, CuSolverEigMode jobz, CuBlasFillMode uplo, int n,
        double* A, int lda, double* W, double* work, int lwork, int* info) =>
        Check(_dsyevd.Value(handle, jobz, uplo, n, A, lda, W, work, lwork, info), "cusolverDnDsyevd");

    // ── QR Factorisation ─────────────────────────────────────────────────────

    /// <summary>Queries workspace size for single-precision QR factorisation.</summary>
    public static int SgeqrfBufferSize(nint handle, int m, int n, float* A, int lda)
    {
        int lwork;
        Check(_sgeqrfBufSize.Value(handle, m, n, A, lda, &lwork), "cusolverDnSgeqrf_bufferSize");
        return lwork;
    }

    /// <summary>Queries workspace size for double-precision QR factorisation.</summary>
    public static int DgeqrfBufferSize(nint handle, int m, int n, double* A, int lda)
    {
        int lwork;
        Check(_dgeqrfBufSize.Value(handle, m, n, A, lda, &lwork), "cusolverDnDgeqrf_bufferSize");
        return lwork;
    }

    /// <summary>Single-precision QR factorisation: A = Q*R.</summary>
    public static void Sgeqrf(nint handle, int m, int n, float* A, int lda, float* TAU, float* workspace, int lwork, int* devInfo) =>
        Check(_sgeqrf.Value(handle, m, n, A, lda, TAU, workspace, lwork, devInfo), "cusolverDnSgeqrf");

    /// <summary>Double-precision QR factorisation: A = Q*R.</summary>
    public static void Dgeqrf(nint handle, int m, int n, double* A, int lda, double* TAU, double* workspace, int lwork, int* devInfo) =>
        Check(_dgeqrf.Value(handle, m, n, A, lda, TAU, workspace, lwork, devInfo), "cusolverDnDgeqrf");
}
