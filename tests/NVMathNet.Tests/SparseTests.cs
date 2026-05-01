// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Sparse;

namespace NVMathNet.Tests;

/// <summary>
/// Tests for the Sparse module (SpMV, SpMM). Requires a CUDA-capable GPU.
/// </summary>
public sealed class SparseTests
{
    // Build a simple diagonal CSR matrix of size N x N with value 'diagVal'
    private static CsrMatrix<float> MakeDiagCsr(int n, float diagVal)
    {
        float[] values = [.. Enumerable.Repeat(diagVal, n)];
        int[]   colIdx = [.. Enumerable.Range(0, n)];
        int[]   rowPtr = [.. Enumerable.Range(0, n + 1)];

        var dValues  = new DeviceBuffer<float>(n);
        var dColIdx  = new DeviceBuffer<int>(n);
        var dRowPtr  = new DeviceBuffer<int>(n + 1);
        dValues.CopyFrom(values.AsSpan());
        dColIdx.CopyFrom(colIdx.AsSpan());
        dRowPtr.CopyFrom(rowPtr.AsSpan());

        return new CsrMatrix<float>(n, n, dRowPtr, dColIdx, dValues);
    }

    [Test]
    public async Task SpMV_DiagonalMatrix_ScalesVector()
    {
        const int N = 128;
        const float DiagVal = 3.0f;

        using var csr = MakeDiagCsr(N, DiagVal);

        float[] hX = new float[N];
        for (int i = 0; i < N; i++)
        {
            hX[i] = i + 1.0f;
        }

        using var dX = new DeviceBuffer<float>(N);
        using var dY = new DeviceBuffer<float>(N);
        dX.CopyFrom(hX.AsSpan());
        dY.Clear();

        await SparseLinearAlgebra.SpMVAsync(csr, dX, dY, alpha: 1.0f, beta: 0.0f);

        float[] result = dY.ToArray();
        for (int i = 0; i < N; i++)
        {
            float expected = DiagVal * hX[i];
            float err = Math.Abs(result[i] - expected);
            await Assert.That(err).IsLessThan(1e-4f);
        }
    }

    [Test]
    public async Task SpMM_DiagonalMatrix_ScalesMatrix()
    {
        const int N = 32;
        const int K = 4;
        const float DiagVal = 2.5f;

        using var csr = MakeDiagCsr(N, DiagVal);

        // Dense matrix B: N rows x K cols, B[i,j] = i+j+1 (col-major for cuSPARSE)
        float[] hB = new float[N * K];
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < K; j++)
            {
                hB[j * N + i] = i + j + 1; // col-major
            }
        }

        using var dB = new DeviceBuffer<float>(N * K);
        using var dC = new DeviceBuffer<float>(N * K);
        dB.CopyFrom(hB.AsSpan());
        dC.Clear();

        await SparseLinearAlgebra.SpMMAsync(csr, dB, dC, K, alpha: 1.0f, beta: 0.0f);

        float[] result = dC.ToArray();
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < K; j++)
            {
                float expected = DiagVal * hB[j * N + i];
                float err = Math.Abs(result[j * N + i] - expected);
                await Assert.That(err).IsLessThan(1e-4f);
            }
        }
    }

    [Test]
    public async Task SpMV_Tridiagonal_CorrectResult()
    {
        // 5×5 tridiagonal: main diag=2, off-diags=-1
        //   [2 -1  0  0  0]   x = [1,2,3,4,5]
        //   [-1 2 -1  0  0]   y[0] = 2*1 - 1*2 = 0
        //   [0 -1  2 -1  0]   y[1] = -1*1 + 2*2 - 1*3 = 0
        //   [0  0 -1  2 -1]   y[2] = -1*2 + 2*3 - 1*4 = 0
        //   [0  0  0 -1  2]   y[3] = -1*3 + 2*4 - 1*5 = 0
        //                     y[4] = -1*4 + 2*5 = 6
        const int N = 5;
        int[] rowPtr = [0, 2, 5, 8, 11, 13];
        int[] colIdx = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
        float[] vals = [2, -1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 2];
        float[] hX = [1, 2, 3, 4, 5];
        float[] expectedY = [0, 0, 0, 0, 6];

        using var dRowPtr = new DeviceBuffer<int>(rowPtr.Length);
        using var dColIdx = new DeviceBuffer<int>(colIdx.Length);
        using var dVals = new DeviceBuffer<float>(vals.Length);
        dRowPtr.CopyFrom(rowPtr.AsSpan());
        dColIdx.CopyFrom(colIdx.AsSpan());
        dVals.CopyFrom(vals.AsSpan());

        using var csr = new CsrMatrix<float>(N, N, dRowPtr, dColIdx, dVals);
        using var dX = new DeviceBuffer<float>(N);
        using var dY = new DeviceBuffer<float>(N);
        dX.CopyFrom(hX.AsSpan());

        await SparseLinearAlgebra.SpMVAsync(csr, dX, dY, alpha: 1.0f, beta: 0.0f);

        float[] result = dY.ToArray();
        for (int i = 0; i < N; i++)
        {
            float err = Math.Abs(result[i] - expectedY[i]);
            await Assert.That(err).IsLessThan(1e-4f);
        }
    }

    [Test]
    public async Task SpMV_AlphaBeta_Accumulates()
    {
        // y = 2*A*x + 3*y_initial
        // A = 3×3 identity, x = [10,20,30], y_init = [1,2,3]
        // Expected: y = 2*[10,20,30] + 3*[1,2,3] = [23, 46, 69]
        const int N = 3;
        using var csr = MakeDiagCsr(N, 1.0f);

        float[] hX = [10, 20, 30];
        float[] hYInit = [1, 2, 3];
        float[] expected = [23, 46, 69];

        using var dX = new DeviceBuffer<float>(N);
        using var dY = new DeviceBuffer<float>(N);
        dX.CopyFrom(hX.AsSpan());
        dY.CopyFrom(hYInit.AsSpan());

        await SparseLinearAlgebra.SpMVAsync(csr, dX, dY, alpha: 2.0f, beta: 3.0f);

        float[] result = dY.ToArray();
        for (int i = 0; i < N; i++)
        {
            float err = Math.Abs(result[i] - expected[i]);
            await Assert.That(err).IsLessThan(1e-4f);
        }
    }

    [Test]
    public async Task SpMV_Transpose_CorrectResult()
    {
        // A = 2×3 matrix: [[1, 2, 3], [4, 5, 6]]
        // A^T = 3×2: [[1,4],[2,5],[3,6]]
        // x = [1, 1] (length 2)
        // y = A^T * x = [1+4, 2+5, 3+6] = [5, 7, 9]
        const int M = 2, K = 3;
        int[] rowPtr = [0, 3, 6];
        int[] colIdx = [0, 1, 2, 0, 1, 2];
        float[] vals = [1, 2, 3, 4, 5, 6];
        float[] hX = [1, 1];
        float[] expectedY = [5, 7, 9];

        using var dRowPtr = new DeviceBuffer<int>(rowPtr.Length);
        using var dColIdx = new DeviceBuffer<int>(colIdx.Length);
        using var dVals = new DeviceBuffer<float>(vals.Length);
        dRowPtr.CopyFrom(rowPtr.AsSpan());
        dColIdx.CopyFrom(colIdx.AsSpan());
        dVals.CopyFrom(vals.AsSpan());

        using var csr = new CsrMatrix<float>(M, K, dRowPtr, dColIdx, dVals);
        using var dX = new DeviceBuffer<float>(M); // x has length = rows of A (since we transpose)
        using var dY = new DeviceBuffer<float>(K); // y has length = cols of A
        dX.CopyFrom(hX.AsSpan());
        dY.Clear();

        await SparseLinearAlgebra.SpMVAsync(csr, dX, dY, alpha: 1.0f, beta: 0.0f, transpose: true);

        float[] result = dY.ToArray();
        for (int i = 0; i < K; i++)
        {
            float err = Math.Abs(result[i] - expectedY[i]);
            await Assert.That(err).IsLessThan(1e-4f);
        }
    }
}
