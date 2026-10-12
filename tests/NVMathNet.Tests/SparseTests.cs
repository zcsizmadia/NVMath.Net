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

    // ── Generic API: SpSV / SpSM / SDDMM / SpGEMM / Transpose ────────────────

    // A = [[2,0,0],[1,3,0],[4,5,6]] (lower triangular)
    private static CsrMatrix<float> MakeLower3()
    {
        return MakeCsr(3, 3, [0, 1, 3, 6], [0, 0, 1, 0, 1, 2], [2, 1, 3, 4, 5, 6]);
    }

    private static CsrMatrix<float> MakeCsr(int rows, int cols, int[] rowPtr, int[] colIdx, float[] vals)
    {
        var dRow = new DeviceBuffer<int>(rowPtr.Length);
        var dCol = new DeviceBuffer<int>(colIdx.Length);
        var dVal = new DeviceBuffer<float>(vals.Length);
        dRow.CopyFrom(rowPtr.AsSpan());
        dCol.CopyFrom(colIdx.AsSpan());
        dVal.CopyFrom(vals.AsSpan());
        return new CsrMatrix<float>(rows, cols, dRow, dCol, dVal);
    }

    private static float[,] ToDense(CsrMatrix<float> m)
    {
        int[] rp = m.RowOffsets.ToArray();
        int[] ci = m.ColIndices.ToArray();
        float[] v = m.Values.ToArray();
        var d = new float[m.Rows, m.Cols];
        for (int r = 0; r < m.Rows; r++)
        {
            for (int i = rp[r]; i < rp[r + 1]; i++)
            {
                d[r, ci[i]] = v[i];
            }
        }

        return d;
    }

    [Test]
    public async Task SpSV_LowerTriangular_Solves()
    {
        using var a = MakeLower3();
        using var x = new DeviceBuffer<float>(3);
        using var y = new DeviceBuffer<float>(3);
        x.CopyFrom(new float[] { 2, 7, 32 }.AsSpan());
        y.Clear();

        await SparseLinearAlgebra.SpSVAsync(a, x, y, SparseTriangle.Lower);

        float[] r = y.ToArray();
        float[] expected = [1, 2, 3];
        for (int i = 0; i < 3; i++)
        {
            await Assert.That(Math.Abs(r[i] - expected[i])).IsLessThan(1e-4f);
        }
    }

    [Test]
    public async Task SpSV_Transpose_Solves()
    {
        // A^T y = x with A^T = [[2,1,4],[0,3,5],[0,0,6]] and y = [1,2,3] => x = [16,21,18]
        using var a = MakeLower3();
        using var x = new DeviceBuffer<float>(3);
        using var y = new DeviceBuffer<float>(3);
        x.CopyFrom(new float[] { 16, 21, 18 }.AsSpan());
        y.Clear();

        await SparseLinearAlgebra.SpSVAsync(a, x, y, SparseTriangle.Lower, transpose: true);

        float[] r = y.ToArray();
        float[] expected = [1, 2, 3];
        for (int i = 0; i < 3; i++)
        {
            await Assert.That(Math.Abs(r[i] - expected[i])).IsLessThan(1e-4f);
        }
    }

    [Test]
    public async Task SpSM_LowerTriangular_SolvesTwoColumns()
    {
        using var a = MakeLower3();
        using var b = new DeviceBuffer<float>(6);
        using var c = new DeviceBuffer<float>(6);
        b.CopyFrom(new float[] { 2, 7, 32, 4, 14, 64 }.AsSpan()); // col-major 3x2
        c.Clear();

        await SparseLinearAlgebra.SpSMAsync(a, b, c, n: 2, SparseTriangle.Lower);

        float[] r = c.ToArray();
        float[] expected = [1, 2, 3, 2, 4, 6];
        for (int i = 0; i < 6; i++)
        {
            await Assert.That(Math.Abs(r[i] - expected[i])).IsLessThan(1e-4f);
        }
    }

    [Test]
    public async Task Sddmm_DiagonalPattern_ComputesSampledProduct()
    {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]] (col-major); A*B = [[19,22],[43,50]]
        using var dA = new DeviceBuffer<float>(4);
        using var dB = new DeviceBuffer<float>(4);
        dA.CopyFrom(new float[] { 1, 3, 2, 4 }.AsSpan());
        dB.CopyFrom(new float[] { 5, 7, 6, 8 }.AsSpan());
        using var c = MakeCsr(2, 2, [0, 1, 2], [0, 1], [0, 0]);

        await SparseLinearAlgebra.SddmmAsync(dA, dB, c, k: 2);

        float[] vals = c.Values.ToArray();
        await Assert.That(Math.Abs(vals[0] - 19f)).IsLessThan(1e-4f);
        await Assert.That(Math.Abs(vals[1] - 50f)).IsLessThan(1e-4f);
    }

    [Test]
    public async Task SpGemm_LowerTimesLower_MatchesDense()
    {
        using var a = MakeLower3();
        using var b = MakeLower3();

        using var c = await SparseLinearAlgebra.SpGemmAsync(a, b);

        float[,] d = ToDense(c);
        float[,] expected = { { 4, 0, 0 }, { 5, 9, 0 }, { 37, 45, 36 } };
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                await Assert.That(Math.Abs(d[i, j] - expected[i, j])).IsLessThan(1e-3f);
            }
        }
    }

    [Test]
    public async Task CsrMatrix_Transpose_MatchesDense()
    {
        using var a = MakeCsr(2, 3, [0, 2, 3], [0, 2, 1], [1, 2, 3]); // [[1,0,2],[0,3,0]]
        using var t = a.Transpose();

        await Assert.That(t.Rows).IsEqualTo(3);
        await Assert.That(t.Cols).IsEqualTo(2);
        float[,] d = ToDense(t);
        float[,] expected = { { 1, 0 }, { 0, 3 }, { 2, 0 } };
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                await Assert.That(Math.Abs(d[i, j] - expected[i, j])).IsLessThan(1e-5f);
            }
        }
    }

    [Test]
    public async Task SpMV_Double_UsesDoubleScalars()
    {
        // y = 2 * A x + 3 * y with A = diag(5): y = 10 x + 3 y
        const int N = 64;
        double[] values = [.. Enumerable.Repeat(5.0, N)];
        int[] colIdx = [.. Enumerable.Range(0, N)];
        int[] rowPtr = [.. Enumerable.Range(0, N + 1)];
        var dVal = new DeviceBuffer<double>(N);
        var dCol = new DeviceBuffer<int>(N);
        var dRow = new DeviceBuffer<int>(N + 1);
        dVal.CopyFrom(values.AsSpan());
        dCol.CopyFrom(colIdx.AsSpan());
        dRow.CopyFrom(rowPtr.AsSpan());
        using var csr = new CsrMatrix<double>(N, N, dRow, dCol, dVal);

        double[] hX = [.. Enumerable.Range(1, N).Select(i => (double)i)];
        double[] hY = [.. Enumerable.Repeat(1.0, N)];
        using var dX = new DeviceBuffer<double>(N);
        using var dY = new DeviceBuffer<double>(N);
        dX.CopyFrom(hX.AsSpan());
        dY.CopyFrom(hY.AsSpan());

        await SparseLinearAlgebra.SpMVAsync(csr, dX, dY, alpha: 2.0f, beta: 3.0f);

        double[] r = dY.ToArray();
        for (int i = 0; i < N; i++)
        {
            await Assert.That(Math.Abs(r[i] - (10.0 * hX[i] + 3.0))).IsLessThan(1e-9);
        }
    }

    // ── COO / BSR SpMV and SpMM ──────────────────────────────────────────────

    // A = [[1,0,2],[0,3,0]] in COO
    private static CooMatrix<float> MakeCoo()
    {
        var dRow = new DeviceBuffer<int>(3);
        var dCol = new DeviceBuffer<int>(3);
        var dVal = new DeviceBuffer<float>(3);
        dRow.CopyFrom(new[] { 0, 0, 1 }.AsSpan());
        dCol.CopyFrom(new[] { 0, 2, 1 }.AsSpan());
        dVal.CopyFrom(new float[] { 1, 2, 3 }.AsSpan());
        return new CooMatrix<float>(2, 3, dRow, dCol, dVal);
    }

    [Test]
    public async Task SpMV_Coo_MatchesDense()
    {
        using var a = MakeCoo();
        using var x = new DeviceBuffer<float>(3);
        using var y = new DeviceBuffer<float>(2);
        x.CopyFrom(new float[] { 1, 2, 3 }.AsSpan());
        y.Clear();

        await SparseLinearAlgebra.SpMVAsync(a, x, y);

        float[] r = y.ToArray();
        await Assert.That(Math.Abs(r[0] - 7f)).IsLessThan(1e-4f);
        await Assert.That(Math.Abs(r[1] - 6f)).IsLessThan(1e-4f);
    }

    [Test]
    public async Task SpMV_Coo_Transpose_MatchesDense()
    {
        using var a = MakeCoo();
        using var x = new DeviceBuffer<float>(2);
        using var y = new DeviceBuffer<float>(3);
        x.CopyFrom(new float[] { 1, 1 }.AsSpan());
        y.Clear();

        await SparseLinearAlgebra.SpMVAsync(a, x, y, transpose: true);

        float[] r = y.ToArray(); // A^T [1,1] = [1,3,2]
        float[] expected = [1, 3, 2];
        for (int i = 0; i < 3; i++)
        {
            await Assert.That(Math.Abs(r[i] - expected[i])).IsLessThan(1e-4f);
        }
    }

    [Test]
    public async Task SpMM_Coo_MatchesDense()
    {
        using var a = MakeCoo();
        using var b = new DeviceBuffer<float>(6);
        using var c = new DeviceBuffer<float>(4);
        b.CopyFrom(new float[] { 1, 2, 3, 0, 1, 0 }.AsSpan()); // col-major 3x2: cols [1,2,3] and [0,1,0]
        c.Clear();

        await SparseLinearAlgebra.SpMMAsync(a, b, c, n: 2);

        float[] r = c.ToArray(); // col-major 2x2: [7,6] and [0,3]
        float[] expected = [7, 6, 0, 3];
        for (int i = 0; i < 4; i++)
        {
            await Assert.That(Math.Abs(r[i] - expected[i])).IsLessThan(1e-4f);
        }
    }

    [Test]
    public async Task SpMV_Bsr_MatchesDense()
    {
        // 4x4 block-diagonal with 2x2 row-major blocks [[1,2],[3,4]] and [[5,6],[7,8]].
        var dRow = new DeviceBuffer<int>(3);
        var dCol = new DeviceBuffer<int>(2);
        var dVal = new DeviceBuffer<float>(8);
        dRow.CopyFrom(new[] { 0, 1, 2 }.AsSpan());
        dCol.CopyFrom(new[] { 0, 1 }.AsSpan());
        dVal.CopyFrom(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }.AsSpan());
        using var a = new BsrMatrix<float>(2, 2, 2, 2, dRow, dCol, dVal);
        using var x = new DeviceBuffer<float>(4);
        using var y = new DeviceBuffer<float>(4);
        x.CopyFrom(new float[] { 1, 1, 1, 1 }.AsSpan());
        y.Clear();

        await SparseLinearAlgebra.SpMVAsync(a, x, y);

        float[] r = y.ToArray();
        float[] expected = [3, 7, 11, 15];
        for (int i = 0; i < 4; i++)
        {
            await Assert.That(Math.Abs(r[i] - expected[i])).IsLessThan(1e-4f);
        }
    }
}
