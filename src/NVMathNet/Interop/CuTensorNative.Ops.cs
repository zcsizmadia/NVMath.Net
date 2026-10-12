// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

namespace NVMathNet.Interop;

// Permutation, element-wise binary and reduction operations.
// Signatures verified against cutensor.h (cuTENSOR 2.8).
public static unsafe partial class CuTensorNative
{
    private static nint OpsFn(string name) => NativeLibraryLoader.GetExport(LibName, name);

    // ── Permutation ───────────────────────────────────────────────────────────

    /// <summary>
    /// Creates a permutation descriptor computing B[modeB] = alpha * opA(A[modeA]).
    /// </summary>
    public static nint CreatePermutation(
        nint handle,
        nint descA, int[] modeA, CuTensorOperator opA,
        nint descB, int[] modeB,
        nint descCompute)
    {
        nint desc;
        fixed (int* pmA = modeA, pmB = modeB)
        {
            Check(((delegate* unmanaged[Cdecl]<nint, nint*, nint, int*, CuTensorOperator, nint, int*, nint, CuTensorStatus>)OpsFn("cutensorCreatePermutation"))(
                handle, &desc, descA, pmA, opA, descB, pmB, descCompute), "cutensorCreatePermutation");
        }

        return desc;
    }

    /// <summary>Executes a permutation plan.</summary>
    public static void Permute(nint handle, nint plan, void* alpha, void* a, void* b, nint stream) =>
        Check(((delegate* unmanaged[Cdecl]<nint, nint, void*, void*, void*, nint, CuTensorStatus>)OpsFn("cutensorPermute"))(
            handle, plan, alpha, a, b, stream), "cutensorPermute");

    // ── Element-wise binary ───────────────────────────────────────────────────

    /// <summary>
    /// Creates an element-wise binary descriptor computing
    /// D[modeD] = opAC(alpha * opA(A[modeA]), gamma * opC(C[modeC])).
    /// </summary>
    public static nint CreateElementwiseBinary(
        nint handle,
        nint descA, int[] modeA, CuTensorOperator opA,
        nint descC, int[] modeC, CuTensorOperator opC,
        nint descD, int[] modeD,
        CuTensorOperator opAC,
        nint descCompute)
    {
        nint desc;
        fixed (int* pmA = modeA, pmC = modeC, pmD = modeD)
        {
            Check(((delegate* unmanaged[Cdecl]<nint, nint*, nint, int*, CuTensorOperator, nint, int*, CuTensorOperator, nint, int*, CuTensorOperator, nint, CuTensorStatus>)OpsFn("cutensorCreateElementwiseBinary"))(
                handle, &desc, descA, pmA, opA, descC, pmC, opC, descD, pmD, opAC, descCompute), "cutensorCreateElementwiseBinary");
        }

        return desc;
    }

    /// <summary>Executes an element-wise binary plan.</summary>
    public static void ElementwiseBinaryExecute(
        nint handle, nint plan, void* alpha, void* a, void* gamma, void* c, void* d, nint stream) =>
        Check(((delegate* unmanaged[Cdecl]<nint, nint, void*, void*, void*, void*, void*, nint, CuTensorStatus>)OpsFn("cutensorElementwiseBinaryExecute"))(
            handle, plan, alpha, a, gamma, c, d, stream), "cutensorElementwiseBinaryExecute");

    // ── Reduction ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Creates a reduction descriptor computing
    /// D[modeD] = alpha * opReduce(opA(A[modeA])) + beta * opC(C[modeC]), reducing the modes of A that
    /// do not appear in D.
    /// </summary>
    public static nint CreateReduction(
        nint handle,
        nint descA, int[] modeA, CuTensorOperator opA,
        nint descC, int[] modeC, CuTensorOperator opC,
        nint descD, int[] modeD,
        CuTensorOperator opReduce,
        nint descCompute)
    {
        nint desc;
        fixed (int* pmA = modeA, pmC = modeC, pmD = modeD)
        {
            Check(((delegate* unmanaged[Cdecl]<nint, nint*, nint, int*, CuTensorOperator, nint, int*, CuTensorOperator, nint, int*, CuTensorOperator, nint, CuTensorStatus>)OpsFn("cutensorCreateReduction"))(
                handle, &desc, descA, pmA, opA, descC, pmC, opC, descD, pmD, opReduce, descCompute), "cutensorCreateReduction");
        }

        return desc;
    }

    /// <summary>Executes a reduction plan.</summary>
    public static void Reduce(
        nint handle, nint plan, void* alpha, void* a, void* beta, void* c, void* d,
        void* workspace, ulong workspaceSize, nint stream) =>
        Check(((delegate* unmanaged[Cdecl]<nint, nint, void*, void*, void*, void*, void*, void*, ulong, nint, CuTensorStatus>)OpsFn("cutensorReduce"))(
            handle, plan, alpha, a, beta, c, d, workspace, workspaceSize, stream), "cutensorReduce");
}
