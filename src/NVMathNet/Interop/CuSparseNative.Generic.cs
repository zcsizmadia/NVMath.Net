// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

namespace NVMathNet.Interop;

/// <summary>Algorithm selector for <c>cusparseSpSV_*</c>.</summary>
public enum CuSparseSpSVAlg : int
{
    /// <summary>Default algorithm.</summary>
    Default = 0,
}

/// <summary>Algorithm selector for <c>cusparseSpSM_*</c>.</summary>
public enum CuSparseSpSMAlg : int
{
    /// <summary>Default algorithm.</summary>
    Default = 0,
}

/// <summary>Algorithm selector for <c>cusparseSDDMM*</c>.</summary>
public enum CuSparseSddmmAlg : int
{
    /// <summary>Default algorithm.</summary>
    Default = 0,
}

/// <summary>Algorithm selector for <c>cusparseSpGEMM_*</c>.</summary>
public enum CuSparseSpGemmAlg : int
{
    /// <summary>Default algorithm.</summary>
    Default = 0,
}

/// <summary>Which triangle of a sparse matrix is referenced by a triangular solve.</summary>
public enum CuSparseFillMode : int
{
    /// <summary>Lower triangle.</summary>
    Lower = 0,
    /// <summary>Upper triangle.</summary>
    Upper = 1,
}

/// <summary>Whether the diagonal of a triangular sparse matrix is implicitly all ones.</summary>
public enum CuSparseDiagType : int
{
    /// <summary>Diagonal values are read from the matrix.</summary>
    NonUnit = 0,
    /// <summary>Diagonal is assumed to be all ones and is not read.</summary>
    Unit = 1,
}

/// <summary>Sparse matrix attributes for <c>cusparseSpMatSetAttribute</c>.</summary>
public enum CuSparseSpMatAttribute : int
{
    /// <summary>Fill mode (<see cref="CuSparseFillMode"/>).</summary>
    FillMode = 0,
    /// <summary>Diagonal type (<see cref="CuSparseDiagType"/>).</summary>
    DiagType = 1,
}

// Generic-API bindings (SpSV, SpSM, SDDMM, SpGEMM, CSR transpose).
// Signatures verified against cusparse.h (CUDA 13.x).
public static unsafe partial class CuSparseNative
{
    private static nint Fn(string name) => NativeLibraryLoader.GetExport(LibName, name);

    // ── Matrix attributes / sizes ─────────────────────────────────────────────

    /// <summary>Sets a sparse matrix attribute (fill mode or diagonal type).</summary>
    public static void SpMatSetAttribute(nint spMat, CuSparseSpMatAttribute attribute, int value) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuSparseSpMatAttribute, void*, nuint, CuSparseStatus>)Fn("cusparseSpMatSetAttribute"))(
            spMat, attribute, &value, sizeof(int)), "cusparseSpMatSetAttribute");

    /// <summary>Gets rows, columns and non-zero count of a sparse matrix descriptor.</summary>
    public static void SpMatGetSize(nint spMat, out long rows, out long cols, out long nnz)
    {
        long r, c, n;
        Check(((delegate* unmanaged[Cdecl]<nint, long*, long*, long*, CuSparseStatus>)Fn("cusparseSpMatGetSize"))(
            spMat, &r, &c, &n), "cusparseSpMatGetSize");
        rows = r;
        cols = c;
        nnz = n;
    }

    /// <summary>Rebinds the CSR arrays of a sparse matrix descriptor.</summary>
    public static void CsrSetPointers(nint spMat, void* rowOffsets, void* colInd, void* values) =>
        Check(((delegate* unmanaged[Cdecl]<nint, void*, void*, void*, CuSparseStatus>)Fn("cusparseCsrSetPointers"))(
            spMat, rowOffsets, colInd, values), "cusparseCsrSetPointers");

    // ── SpSV ──────────────────────────────────────────────────────────────────

    /// <summary>Creates an SpSV (sparse triangular solve, vector RHS) descriptor.</summary>
    public static nint SpSVCreateDescr()
    {
        nint d;
        Check(((delegate* unmanaged[Cdecl]<nint*, CuSparseStatus>)Fn("cusparseSpSV_createDescr"))(&d), "cusparseSpSV_createDescr");
        return d;
    }

    /// <summary>Destroys an SpSV descriptor.</summary>
    public static void SpSVDestroyDescr(nint descr) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuSparseStatus>)Fn("cusparseSpSV_destroyDescr"))(descr), "cusparseSpSV_destroyDescr");

    /// <summary>Returns the workspace size for SpSV analysis.</summary>
    public static nuint SpSVBufferSize(
        nint handle, CuSparseOperation opA, void* alpha, nint matA, nint vecX, nint vecY,
        CudaDataType computeType, CuSparseSpSVAlg alg, nint descr)
    {
        nuint sz;
        Check(((delegate* unmanaged[Cdecl]<nint, CuSparseOperation, void*, nint, nint, nint, CudaDataType, CuSparseSpSVAlg, nint, nuint*, CuSparseStatus>)Fn("cusparseSpSV_bufferSize"))(
            handle, opA, alpha, matA, vecX, vecY, computeType, alg, descr, &sz), "cusparseSpSV_bufferSize");
        return sz;
    }

    /// <summary>Runs the SpSV analysis phase.</summary>
    public static void SpSVAnalysis(
        nint handle, CuSparseOperation opA, void* alpha, nint matA, nint vecX, nint vecY,
        CudaDataType computeType, CuSparseSpSVAlg alg, nint descr, void* buffer) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuSparseOperation, void*, nint, nint, nint, CudaDataType, CuSparseSpSVAlg, nint, void*, CuSparseStatus>)Fn("cusparseSpSV_analysis"))(
            handle, opA, alpha, matA, vecX, vecY, computeType, alg, descr, buffer), "cusparseSpSV_analysis");

    /// <summary>Solves op(A) y = alpha x.</summary>
    public static void SpSVSolve(
        nint handle, CuSparseOperation opA, void* alpha, nint matA, nint vecX, nint vecY,
        CudaDataType computeType, CuSparseSpSVAlg alg, nint descr) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuSparseOperation, void*, nint, nint, nint, CudaDataType, CuSparseSpSVAlg, nint, CuSparseStatus>)Fn("cusparseSpSV_solve"))(
            handle, opA, alpha, matA, vecX, vecY, computeType, alg, descr), "cusparseSpSV_solve");

    // ── SpSM ──────────────────────────────────────────────────────────────────

    /// <summary>Creates an SpSM (sparse triangular solve, matrix RHS) descriptor.</summary>
    public static nint SpSMCreateDescr()
    {
        nint d;
        Check(((delegate* unmanaged[Cdecl]<nint*, CuSparseStatus>)Fn("cusparseSpSM_createDescr"))(&d), "cusparseSpSM_createDescr");
        return d;
    }

    /// <summary>Destroys an SpSM descriptor.</summary>
    public static void SpSMDestroyDescr(nint descr) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuSparseStatus>)Fn("cusparseSpSM_destroyDescr"))(descr), "cusparseSpSM_destroyDescr");

    /// <summary>Returns the workspace size for SpSM analysis.</summary>
    public static nuint SpSMBufferSize(
        nint handle, CuSparseOperation opA, CuSparseOperation opB, void* alpha, nint matA, nint matB, nint matC,
        CudaDataType computeType, CuSparseSpSMAlg alg, nint descr)
    {
        nuint sz;
        Check(((delegate* unmanaged[Cdecl]<nint, CuSparseOperation, CuSparseOperation, void*, nint, nint, nint, CudaDataType, CuSparseSpSMAlg, nint, nuint*, CuSparseStatus>)Fn("cusparseSpSM_bufferSize"))(
            handle, opA, opB, alpha, matA, matB, matC, computeType, alg, descr, &sz), "cusparseSpSM_bufferSize");
        return sz;
    }

    /// <summary>Runs the SpSM analysis phase.</summary>
    public static void SpSMAnalysis(
        nint handle, CuSparseOperation opA, CuSparseOperation opB, void* alpha, nint matA, nint matB, nint matC,
        CudaDataType computeType, CuSparseSpSMAlg alg, nint descr, void* buffer) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuSparseOperation, CuSparseOperation, void*, nint, nint, nint, CudaDataType, CuSparseSpSMAlg, nint, void*, CuSparseStatus>)Fn("cusparseSpSM_analysis"))(
            handle, opA, opB, alpha, matA, matB, matC, computeType, alg, descr, buffer), "cusparseSpSM_analysis");

    /// <summary>Solves op(A) C = alpha op(B).</summary>
    public static void SpSMSolve(
        nint handle, CuSparseOperation opA, CuSparseOperation opB, void* alpha, nint matA, nint matB, nint matC,
        CudaDataType computeType, CuSparseSpSMAlg alg, nint descr) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuSparseOperation, CuSparseOperation, void*, nint, nint, nint, CudaDataType, CuSparseSpSMAlg, nint, CuSparseStatus>)Fn("cusparseSpSM_solve"))(
            handle, opA, opB, alpha, matA, matB, matC, computeType, alg, descr), "cusparseSpSM_solve");

    // ── SDDMM ─────────────────────────────────────────────────────────────────

    /// <summary>Returns the workspace size for SDDMM.</summary>
    public static nuint SddmmBufferSize(
        nint handle, CuSparseOperation opA, CuSparseOperation opB, void* alpha, nint matA, nint matB, void* beta, nint matC,
        CudaDataType computeType, CuSparseSddmmAlg alg)
    {
        nuint sz;
        Check(((delegate* unmanaged[Cdecl]<nint, CuSparseOperation, CuSparseOperation, void*, nint, nint, void*, nint, CudaDataType, CuSparseSddmmAlg, nuint*, CuSparseStatus>)Fn("cusparseSDDMM_bufferSize"))(
            handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, &sz), "cusparseSDDMM_bufferSize");
        return sz;
    }

    /// <summary>Optional SDDMM preprocessing.</summary>
    public static void SddmmPreprocess(
        nint handle, CuSparseOperation opA, CuSparseOperation opB, void* alpha, nint matA, nint matB, void* beta, nint matC,
        CudaDataType computeType, CuSparseSddmmAlg alg, void* buffer) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuSparseOperation, CuSparseOperation, void*, nint, nint, void*, nint, CudaDataType, CuSparseSddmmAlg, void*, CuSparseStatus>)Fn("cusparseSDDMM_preprocess"))(
            handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, buffer), "cusparseSDDMM_preprocess");

    /// <summary>Sampled dense-dense matrix product: C = alpha (op(A) op(B)) ∘ spy(C) + beta C.</summary>
    public static void Sddmm(
        nint handle, CuSparseOperation opA, CuSparseOperation opB, void* alpha, nint matA, nint matB, void* beta, nint matC,
        CudaDataType computeType, CuSparseSddmmAlg alg, void* buffer) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuSparseOperation, CuSparseOperation, void*, nint, nint, void*, nint, CudaDataType, CuSparseSddmmAlg, void*, CuSparseStatus>)Fn("cusparseSDDMM"))(
            handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, buffer), "cusparseSDDMM");

    // ── SpGEMM ────────────────────────────────────────────────────────────────

    /// <summary>Creates an SpGEMM descriptor.</summary>
    public static nint SpGemmCreateDescr()
    {
        nint d;
        Check(((delegate* unmanaged[Cdecl]<nint*, CuSparseStatus>)Fn("cusparseSpGEMM_createDescr"))(&d), "cusparseSpGEMM_createDescr");
        return d;
    }

    /// <summary>Destroys an SpGEMM descriptor.</summary>
    public static void SpGemmDestroyDescr(nint descr) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuSparseStatus>)Fn("cusparseSpGEMM_destroyDescr"))(descr), "cusparseSpGEMM_destroyDescr");

    /// <summary>
    /// SpGEMM work-estimation phase. Pass <paramref name="buffer"/> = null to query the size,
    /// then call again with an allocated buffer.
    /// </summary>
    public static void SpGemmWorkEstimation(
        nint handle, CuSparseOperation opA, CuSparseOperation opB, void* alpha, nint matA, nint matB, void* beta, nint matC,
        CudaDataType computeType, CuSparseSpGemmAlg alg, nint descr, ref nuint bufferSize, void* buffer)
    {
        nuint sz = bufferSize;
        Check(((delegate* unmanaged[Cdecl]<nint, CuSparseOperation, CuSparseOperation, void*, nint, nint, void*, nint, CudaDataType, CuSparseSpGemmAlg, nint, nuint*, void*, CuSparseStatus>)Fn("cusparseSpGEMM_workEstimation"))(
            handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, descr, &sz, buffer), "cusparseSpGEMM_workEstimation");
        bufferSize = sz;
    }

    /// <summary>
    /// SpGEMM compute phase. Pass <paramref name="buffer"/> = null to query the size,
    /// then call again with an allocated buffer.
    /// </summary>
    public static void SpGemmCompute(
        nint handle, CuSparseOperation opA, CuSparseOperation opB, void* alpha, nint matA, nint matB, void* beta, nint matC,
        CudaDataType computeType, CuSparseSpGemmAlg alg, nint descr, ref nuint bufferSize, void* buffer)
    {
        nuint sz = bufferSize;
        Check(((delegate* unmanaged[Cdecl]<nint, CuSparseOperation, CuSparseOperation, void*, nint, nint, void*, nint, CudaDataType, CuSparseSpGemmAlg, nint, nuint*, void*, CuSparseStatus>)Fn("cusparseSpGEMM_compute"))(
            handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, descr, &sz, buffer), "cusparseSpGEMM_compute");
        bufferSize = sz;
    }

    /// <summary>SpGEMM copy phase: writes the result into the arrays bound to matC.</summary>
    public static void SpGemmCopy(
        nint handle, CuSparseOperation opA, CuSparseOperation opB, void* alpha, nint matA, nint matB, void* beta, nint matC,
        CudaDataType computeType, CuSparseSpGemmAlg alg, nint descr) =>
        Check(((delegate* unmanaged[Cdecl]<nint, CuSparseOperation, CuSparseOperation, void*, nint, nint, void*, nint, CudaDataType, CuSparseSpGemmAlg, nint, CuSparseStatus>)Fn("cusparseSpGEMM_copy"))(
            handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, descr), "cusparseSpGEMM_copy");

    // ── CSR → CSC (transpose) ─────────────────────────────────────────────────

    /// <summary>Returns the workspace size for <see cref="Csr2CscEx2"/>.</summary>
    public static nuint Csr2CscEx2BufferSize(
        nint handle, int m, int n, int nnz, void* csrVal, int* csrRowPtr, int* csrColInd,
        void* cscVal, int* cscColPtr, int* cscRowInd, CudaDataType valType)
    {
        nuint sz;
        Check(((delegate* unmanaged[Cdecl]<nint, int, int, int, void*, int*, int*, void*, int*, int*, CudaDataType, int, CuSparseIndexBase, int, nuint*, CuSparseStatus>)Fn("cusparseCsr2cscEx2_bufferSize"))(
            handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType,
            1 /* CUSPARSE_ACTION_NUMERIC */, CuSparseIndexBase.Zero, 1 /* ALG1 */, &sz), "cusparseCsr2cscEx2_bufferSize");
        return sz;
    }

    /// <summary>Converts a CSR matrix to CSC (equivalently, the CSR of its transpose).</summary>
    public static void Csr2CscEx2(
        nint handle, int m, int n, int nnz, void* csrVal, int* csrRowPtr, int* csrColInd,
        void* cscVal, int* cscColPtr, int* cscRowInd, CudaDataType valType, void* buffer) =>
        Check(((delegate* unmanaged[Cdecl]<nint, int, int, int, void*, int*, int*, void*, int*, int*, CudaDataType, int, CuSparseIndexBase, int, void*, CuSparseStatus>)Fn("cusparseCsr2cscEx2"))(
            handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType,
            1 /* CUSPARSE_ACTION_NUMERIC */, CuSparseIndexBase.Zero, 1 /* ALG1 */, buffer), "cusparseCsr2cscEx2");
}
